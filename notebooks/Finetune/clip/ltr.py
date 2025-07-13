import argparse
import logging
import random

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


class LTRDataset(Dataset):
    def __init__(self, datasplit, config, logger):
        super().__init__()
        logger.info(f"loading {datasplit} dataset")
        self.num_items_per_query = config["items_per_query"]
        self.shuffle = datasplit == "training"
        self.datasplit = datasplit
        # load queries
        df = pd.read_csv(f"{args.datadir}/{datasplit}.similarpairs.csv")
        df = df.query(f"cliptype == 'medium{args.medium}'").copy()
        df["target_tuple"] = list(zip(df["target_matchedid"], df["score"]))
        grouped = df.groupby(["cliptype", "source_matchedid", "source_popularity"])[
            "target_tuple"
        ].apply(list)
        queries = []
        medium_map = {"medium0": 0, "medium1": 1}
        for (cliptype, sourceid, popularity), targets in grouped.items():
            medium_idx = medium_map[cliptype]
            r = {
                "medium": medium_idx,
                "sourceid": sourceid,
                "popularity": np.sqrt(popularity),
                "targets": sorted(
                    [(targetid, score) for (targetid, score) in targets],
                    key=lambda x: x[1],
                    reverse=True,
                ),
                "maxid": config["vocab_sizes"][medium_idx],
            }
            queries.append(r)
        self.queries = queries
        if datasplit == "training":
            # keep implicit negatives disjoint between training and test
            other_split = "test"
            other_df = pd.read_csv(f"{args.datadir}/{other_split}.similarpairs.csv")
            other_df = other_df.query(f"cliptype == 'medium{args.medium}'").copy()
            other_queries = {}
            for (s, t) in zip(other_df["source_matchedid"], other_df["target_matchedid"]):
                if s not in other_queries:
                    other_queries[s] = set()
                other_queries[s].add(t)
            self.exclude_pairs = other_queries
        else:
            self.exclude_pairs = {}
        # load hard negatives
        m = config["vocab_sizes"][args.medium]
        n = config["embed_dim"]
        output_path = f"{args.datadir}/output.embeddings.{args.medium}.h5"
        M = np.zeros((m, n), dtype=np.float32)
        with h5py.File(output_path) as hf:
            for i in range(m):
                M[i, :] = hf[f"{args.medium}.{i}"][:]
            temperature = float(hf["temperature"][:][0])
        M = torch.tensor(M).to(args.device)
        self.sample_probs = np.zeros((m, m), np.float32)
        idxs = [x["sourceid"] for x in queries]
        chunk_size = 4096
        for batch in [
            idxs[i : i + chunk_size] for i in range(0, len(idxs), chunk_size)
        ]:
            with torch.no_grad():
                with torch.amp.autocast(f"cuda:{args.device}", dtype=torch.bfloat16):
                    self.sample_probs[batch, :] = (
                        torch.exp((M[batch, :] @ M.T) * temperature).cpu().numpy()
                    )
        del M
        for i in range(m):
            self.sample_probs[i, i] = 0
        row_sums = self.sample_probs.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        self.sample_probs /= row_sums
        self.choices = list(range(m))

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, idx):
        query_data = self.queries[idx]
        source_id = query_data["sourceid"]
        max_id = query_data["maxid"]
        medium_idx = query_data["medium"]
        popularity = query_data["popularity"]
        if self.shuffle:
            max_num_positives = int(round(self.num_items_per_query) * 0.9)
        else:
            max_num_positives = self.num_items_per_query
        positive_items = query_data["targets"][:max_num_positives]
        positive_y = [item[0] for item in positive_items]
        positive_r = [item[1] for item in positive_items]
        num_positives = len(positive_y)
        num_negatives_to_sample = self.num_items_per_query - num_positives
        negative_y = []
        if num_negatives_to_sample > 0:
            assert self.datasplit != "test"
            p = self.sample_probs[source_id, :]
            existing_ids = set(positive_y) | self.exclude_pairs.get(source_id, set())
            existing_ids.add(source_id)
            while len(negative_y) < num_negatives_to_sample:
                sample_size = (num_negatives_to_sample - len(negative_y)) * 2
                candidates = list(np.random.choice(self.choices, size=sample_size, p=p))
                for cid in candidates:
                    if len(negative_y) == num_negatives_to_sample:
                        break
                    if cid not in existing_ids:
                        negative_y.append(cid)
                        existing_ids.add(cid)
        y = np.array(positive_y + negative_y, dtype=np.int64)
        r = np.array(positive_r + [0.0] * num_negatives_to_sample, dtype=np.float64)
        x = np.full(self.num_items_per_query, source_id, dtype=np.int64)
        w = np.array([popularity], dtype=np.float64)
        return {
            "medium": np.array([medium_idx]),
            "sourceid": x,
            "targetid": y,
            "relevance": r,
            "weight": w,
        }


def worker_init_fn(worker_id):
    np.random.seed(torch.utils.data.get_worker_info().seed % 2**32)


def to_device(data):
    d = {}
    for k in data:
        d[k] = data[k].to(args.device)
    d["weight"] = d["weight"].squeeze(dim=-1)
    return d


class LTRModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.retrieval_embeddings = nn.ModuleDict(
            {str(m): nn.Embedding(config["vocab_sizes"][m], 2048) for m in [0, 1]}
        )
        self.ranking_embeddings = nn.ModuleDict(
            {str(m): nn.Embedding(config["vocab_sizes"][m], 2048) for m in [0, 1]}
        )
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        if args.features == "all":
            self.encoder = nn.Linear(2048 * 2, config["embed_dim"], bias=False)
        elif args.features == "retrieval":
            self.encoder = nn.Linear(2048, config["embed_dim"], bias=False)
        elif args.features == "id":
            self.matchedid_embedding = nn.Embedding(
                config["vocab_sizes"][args.medium], config["embed_dim"]
            )
            x = self.matchedid_embedding
            weight_tied_linear = nn.Linear(*reversed(x.weight.shape), bias=False)
            x.weight = weight_tied_linear.weight
        else:
            assert False

    def get_temperature(self):
        return float(self.logit_scale)

    def load_pretrained_embeddings(self, filepath):
        for medium in [0, 1]:
            print(
                f"Loading pretrained embeddings from '{filepath}' for medium {medium}"
            )
            retrieval_emb_layer = self.retrieval_embeddings[str(medium)]
            ranking_emb_layer = self.ranking_embeddings[str(medium)]
            vocab_size = self.config["vocab_sizes"][medium]
            with h5py.File(filepath, "r") as hf:
                retrieval_weights = torch.stack(
                    [
                        torch.tensor(hf[f"masked.{medium}.{i}"][()])
                        for i in range(vocab_size)
                    ]
                )
                ranking_weights = torch.stack(
                    [
                        torch.tensor(hf[f"causal.{medium}.{i}"][()])
                        for i in range(vocab_size)
                    ]
                )
                retrieval_emb_layer.weight.data = retrieval_weights
                ranking_emb_layer.weight.data = ranking_weights
            self.retrieval_embeddings[str(medium)].weight.requires_grad = False
            self.ranking_embeddings[str(medium)].weight.requires_grad = False

    def embed(self, ids, medium_ids):
        if args.features == "all":
            f_ret = torch.where(
                medium_ids.unsqueeze(-1) == 0,
                self.retrieval_embeddings["0"](ids * (medium_ids == 0)),
                self.retrieval_embeddings["1"](ids * (medium_ids == 1)),
            )
            f_rnk = torch.where(
                medium_ids.unsqueeze(-1) == 0,
                self.ranking_embeddings["0"](ids * (medium_ids == 0)),
                self.ranking_embeddings["1"](ids * (medium_ids == 1)),
            )
            encoded_features = self.encoder(torch.cat([f_ret, f_rnk], dim=-1))
            return F.normalize(encoded_features, dim=-1)
        elif args.features == "retrieval":
            f_ret = torch.where(
                medium_ids.unsqueeze(-1) == 0,
                self.retrieval_embeddings["0"](ids * (medium_ids == 0)),
                self.retrieval_embeddings["1"](ids * (medium_ids == 1)),
            )
            encoded_features = self.encoder(f_ret)
            return F.normalize(encoded_features, dim=-1)
        elif args.features == "id":
            return F.normalize(self.matchedid_embedding(ids), dim=-1)
        else:
            assert False

    def pairwise_diff(self, x):
        return x.reshape(x.shape[0], x.shape[1], 1) - x.reshape(
            x.shape[0], 1, x.shape[1]
        )

    def lambdarank_loss(self, x, y, w):
        order = torch.argsort(torch.argsort(x, descending=True)) + 1
        deltag = self.pairwise_diff(y)
        deltad = self.pairwise_diff(1 / torch.log2(1 + order))
        deltandcg = deltad * deltag
        loss = -F.logsigmoid(self.pairwise_diff(x)) * (self.pairwise_diff(y) > 0)
        wloss = loss * deltandcg.abs()
        return (wloss.sum(dim=[1, 2]) * w).sum() / w.sum()

    def ndcg(self, batch):
        source_features = self.embed(batch["sourceid"], batch["medium"])
        target_features = self.embed(batch["targetid"], batch["medium"])
        pred_scores = (source_features * target_features).sum(
            dim=-1
        ) * self.logit_scale.exp()
        true_relevance = batch["relevance"]
        ranking_indices = torch.argsort(pred_scores, dim=-1, descending=True)
        ranked_relevance = torch.gather(true_relevance, -1, ranking_indices)
        gains = ranked_relevance
        list_size = pred_scores.shape[-1]
        discounts = torch.log2(
            torch.arange(2.0, list_size + 2.0, device=pred_scores.device)
        )
        dcg = (gains / discounts).sum(dim=-1)
        ideal_relevance, _ = torch.sort(true_relevance, dim=-1, descending=True)
        ideal_gains = ideal_relevance
        idcg = (ideal_gains / discounts).sum(dim=-1)
        ndcg = dcg / idcg
        w = batch["weight"]
        return (ndcg * w).sum() / w.sum()

    def forward(self, batch):
        source_features = self.embed(batch["sourceid"], batch["medium"])
        target_features = self.embed(batch["targetid"], batch["medium"])
        x = (source_features * target_features).sum(dim=-1) * self.logit_scale.exp()
        y = batch["relevance"]
        w = batch["weight"]
        return self.lambdarank_loss(x, y, w)


def evaluate_metrics(model, dataloader):
    losses = 0.0
    weights = 0.0
    progress = tqdm(desc="Test batches", mininterval=1)
    model.eval()
    for data in dataloader:
        with torch.no_grad():
            with torch.amp.autocast(f"cuda:{args.device}", dtype=torch.bfloat16):
                d = to_device(data)
                loss = model.ndcg(d)
            w = float(d["weight"].sum())
            losses += float(loss) * w
            weights += w
        progress.update()
    progress.close()
    model.train()
    return 1.0 - (losses / weights)


def train_epoch(
    model,
    dataloader,
    optimizer,
    scheduler,
    scaler,
):
    training_losses = 0.0
    training_weights = 0.0
    progress = tqdm(desc="Training batches", mininterval=1)
    for step, data in enumerate(dataloader):
        with torch.amp.autocast(f"cuda:{args.device}", dtype=torch.bfloat16):
            optimizer.zero_grad(set_to_none=True)
            d = to_device(data)
            loss = model(d)
            w = float(d["weight"].sum())
            training_losses += float(loss.detach()) * w
            training_weights += w
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        progress.update()
    progress.close()
    return training_losses / training_weights


def create_optimizer(model, config):
    param_dict = {pn: p for pn, p in model.named_parameters()}
    param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    return optim.AdamW(
        [
            {"params": decay_params, "weight_decay": 0.1},
            {"params": nodecay_params, "weight_decay": 0.0},
        ],
        lr=config["learning_rate"],
    )


class ConstantScheduler(object):
    def __init__(self):
        self.steps = 0

    def __call__(self, epoch):
        self.steps += 1
        return 1


def create_learning_rate_schedule(optimizer, tokens_per_batch, epochs):
    return optim.lr_scheduler.LambdaLR(optimizer, ConstantScheduler())


class EarlyStopper:
    def __init__(self, patience, rtol):
        self.patience = patience
        self.rtol = rtol
        self.counter = 0
        self.stop_score = float("inf")
        self.early_stop = False
        self.saved_score = float("inf")
        self.save_model = False

    def __call__(self, score):
        if score < self.stop_score * (1 - self.rtol):
            self.counter = 0
            self.stop_score = score
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        if score < self.saved_score:
            self.saved_score = score
            self.save_model = True
        else:
            self.save_model = False


def get_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    formatter = logging.Formatter(
        "%(name)s:%(levelname)s:%(asctime)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    logger.addHandler(stream)
    return logger


def checkpoint_model(model, epoch, training_loss, test_loss, oos_loss, save):
    if save:
        checkpoint = {
            "model": model._orig_mod.state_dict(),
            "epoch": epoch,
            "training_loss": training_loss,
            "test_loss": test_loss,
            "oos_loss": oos_loss,
        }
        torch.save(
            checkpoint,
            f"{args.datadir}/clip.model.{args.medium}.pt",
        )
    csv_fn = f"{args.datadir}/clip.model.{args.medium}.csv"
    create_csv = epoch < 0
    if create_csv:
        with open(csv_fn, "w") as f:
            f.write(",".join(["epoch", "training_loss", "test_loss", "oos_loss", "saved"]) + "\n")
    with open(csv_fn, "a") as f:
        vals = [
            epoch,
            training_loss,
            test_loss,
            oos_loss,
            1 if save else 0,
        ]
        f.write(",".join([str(x) for x in vals]) + "\n")


def training_config():
    vocab_sizes = {
        x: int(pd.read_csv(f"{args.datadir}/../{y}.csv").matchedid.max()) + 1
        for (x, y) in {0: "manga", 1: "anime"}.items()
    }
    config = {
        "vocab_sizes": vocab_sizes,
        "embed_dim": 128,
        "learning_rate": 3e-4,
        "batch_size": 32,
        "items_per_query": 2048,
    }
    return config


def train():
    torch.cuda.set_device(args.device)
    logger = get_logger("clip")
    logger.setLevel(logging.DEBUG)
    config = training_config()
    num_epochs = 1024
    model = LTRModel(config)
    model.load_pretrained_embeddings(f"{args.datadir}/input.h5")
    logger.info(
        f"Created model with {sum(p.numel() for p in model.parameters())} parameters"
        f" and {sum(p.numel() for p in model.parameters() if p.requires_grad)} trainable parameters"
    )
    model = model.to(args.device)
    model = torch.compile(model)
    generate_embeddings(model)
    dataloaders = {
        x: DataLoader(
            LTRDataset(datasplit=x, config=config, logger=logger),
            batch_size=config["batch_size"],
            shuffle=x == "training",
            drop_last=False,
            num_workers=8,
            worker_init_fn=worker_init_fn,
        )
        for x in ["training", "validation", "test"]
    }
    optimizer = create_optimizer(model, config)
    scheduler = create_learning_rate_schedule(
        optimizer,
        len(dataloaders["training"]),
        num_epochs,
    )
    scaler = torch.amp.GradScaler(args.device)
    stopper = EarlyStopper(patience=5, rtol=0.001)
    get_loss = lambda x, y: evaluate_metrics(x, dataloaders[y])
    initial_loss = get_loss(model, "training")
    logger.info(f"Initial Loss: {initial_loss}")
    stopper(initial_loss)
    checkpoint_model(model, -1, np.nan, initial_loss, np.nan, True)
    for epoch in range(num_epochs):
        training_loss = train_epoch(
            model, dataloaders["training"], optimizer, scheduler, scaler
        )
        logger.info(f"Epoch: {epoch}, Training Loss: {training_loss}")
        generate_embeddings(model)
        for x in ["training", "validation"]:
            dataloaders[x] = DataLoader(
                LTRDataset(datasplit=x, config=config, logger=logger),
                batch_size=config["batch_size"],
                shuffle=x == "training",
                drop_last=False,
                num_workers=8,
                worker_init_fn=worker_init_fn,
            )
        test_loss = get_loss(model, "validation")
        logger.info(f"Epoch: {epoch}, Test Loss: {test_loss}")
        oos_loss = get_loss(model, "test")
        logger.info(f"Epoch: {epoch}, OOS Loss: {oos_loss}")
        stopper(test_loss)
        checkpoint_model(model, epoch, training_loss, test_loss, oos_loss, stopper.save_model)
        if stopper.early_stop:
            break
    logger.info(f"Best loss: {stopper.saved_score}")


def load_model():
    config = training_config()
    checkpoint_path = f"{args.datadir}/clip.model.{args.medium}.pt"
    model = LTRModel(config)
    checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))
    model.load_state_dict(checkpoint["model"])
    model.to(args.device)
    model = torch.compile(model)
    model.eval()
    return model


def generate_embeddings(model):
    torch.cuda.set_device(args.device)
    output_path = f"{args.datadir}/output.embeddings.{args.medium}.h5"
    config = training_config()
    device = args.device
    temperature = model.get_temperature()
    num_items = config["vocab_sizes"][args.medium]
    batch_size = config["batch_size"]
    final_embeddings = np.zeros((num_items, config["embed_dim"]), dtype=np.float32)
    all_item_ids = torch.arange(num_items, device=device).unsqueeze(-1)
    all_medium_ids = torch.full((num_items, 1), args.medium, device=device)
    for i in range(0, num_items, batch_size):
        with torch.no_grad():
            batch_ids = all_item_ids[i : i + batch_size, :]
            medium_ids = all_medium_ids[i : i + batch_size, :]
            embs = model.embed(batch_ids, medium_ids)
            final_embeddings[i : i + batch_size, :] = embs.squeeze(dim=1).cpu().numpy()
    with h5py.File(output_path, "w") as hf:
        for i in range(num_items):
            key = f"{args.medium}.{i}"
            hf.create_dataset(key, data=final_embeddings[i])
        hf.create_dataset("temperature", data=np.array([temperature]))


parser = argparse.ArgumentParser()
parser.add_argument("--datadir", type=str)
parser.add_argument("--device", type=int)
parser.add_argument("--medium", type=int)
parser.add_argument("--features", type=str)
args = parser.parse_args()

if __name__ == "__main__":
    train()
    generate_embeddings(load_model())