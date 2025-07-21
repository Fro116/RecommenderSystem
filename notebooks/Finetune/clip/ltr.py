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
        self.datasplit = datasplit
        self.config = config
        self.num_items_per_query = self.config["items_per_query"]
        if self.datasplit == "training":
            self.shuffle = True
            self.max_num_positives = int(round(self.num_items_per_query) * 0.9)
        else:
            self.shuffle = False
            self.max_num_positives = self.num_items_per_query
        self.load_queries()
        self.load_hard_negatives()

    def load_queries(self):
        df = pd.read_csv(f"{args.datadir}/{self.datasplit}.similarpairs.csv")
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
                "maxid": self.config["vocab_sizes"][medium_idx],
            }
            queries.append(r)
        self.queries = queries
        other_split = {"training": "test", "test": "training"}[self.datasplit]
        other_df = pd.read_csv(f"{args.datadir}/{other_split}.similarpairs.csv")
        other_df = other_df.query(f"cliptype == 'medium{args.medium}'").copy()
        other_queries = {}
        for s, t in zip(other_df["source_matchedid"], other_df["target_matchedid"]):
            if s not in other_queries:
                other_queries[s] = set()
            other_queries[s].add(t)
        self.exclude_pairs = other_queries

    def load_hard_negatives(self):
        config = self.config
        m = config["vocab_sizes"][args.medium]
        n = config["embed_dim"]
        output_path = f"{args.datadir}/output.embeddings.{args.medium}.h5"
        M = np.zeros((m, n), dtype=np.float32)
        with h5py.File(output_path) as hf:
            for i in range(m):
                M[i, :] = hf[f"{args.medium}.{i}"][:]
            temperature_exp = float(np.exp(hf["temperature"][:][0]))
        M = torch.tensor(M).to(args.device)
        self.sample_probs = np.zeros((m, m), np.float32)
        idxs = [x["sourceid"] for x in self.queries]
        chunk_size = 4096
        for batch in [
            idxs[i : i + chunk_size] for i in range(0, len(idxs), chunk_size)
        ]:
            with torch.no_grad():
                with torch.amp.autocast(f"cuda:{args.device}", dtype=torch.bfloat16):
                    self.sample_probs[batch, :] = (
                        torch.exp((M[batch, :] @ M.T) * temperature_exp).cpu().numpy()
                    )
        del M
        for i in range(m):
            self.sample_probs[i, i] = 0
        for k, vs in self.exclude_pairs.items():
            for v in vs:
                self.sample_probs[k, v] = 0
        for x in self.queries:
            k = x["sourceid"]
            for v, _ in x["targets"][: self.max_num_positives]:
                self.sample_probs[k, v] = 0
        if self.datasplit == "training":
            self.sample_probs *= training_implicit_negatives
        elif self.datasplit == "test":
            self.sample_probs *= ~training_implicit_negatives
        else:
            assert False
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
        popularity = query_data["popularity"]
        positive_items = query_data["targets"][: self.max_num_positives]
        positive_y = [item[0] for item in positive_items]
        positive_r = [item[1] for item in positive_items]
        num_positives = len(positive_y)
        num_negatives_to_sample = self.num_items_per_query - num_positives
        if num_negatives_to_sample > 0:
            p = self.sample_probs[source_id, :]
            if self.shuffle:
                negative_y = list(
                    np.random.choice(
                        self.choices, size=num_negatives_to_sample, p=p, replace=False
                    )
                )
            else:
                negative_y = list(np.argpartition(p, -num_negatives_to_sample)[-num_negatives_to_sample:])
        else:
            negative_y = []
        y = np.array(positive_y + negative_y, dtype=np.int64)
        r = np.array(positive_r + [0.0] * num_negatives_to_sample, dtype=np.float64)
        x = np.full(self.num_items_per_query, source_id, dtype=np.int64)
        w = np.array([popularity], dtype=np.float64)
        return {
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
        self.retrieval_embeddings = nn.Embedding(
            config["vocab_sizes"][args.medium], 2048
        )
        self.ranking_embeddings = nn.Embedding(config["vocab_sizes"][args.medium], 2048)
        self.item_text_embeddings = nn.Embedding(
            config["vocab_sizes"][args.medium], 3072
        )
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        if args.features == "all":
            self.encoder = nn.Linear(2048 * 2 + 3072, config["embed_dim"], bias=False)
        elif args.features == "rettext":
            self.encoder = nn.Linear(2048 + 3072, config["embed_dim"], bias=False)
        elif args.features == "ranking":
            self.encoder = nn.Linear(2048, config["embed_dim"], bias=False)
        elif args.features == "retrieval":
            self.encoder = nn.Linear(2048, config["embed_dim"], bias=False)
        elif args.features == "text":
            self.encoder = nn.Linear(3072, config["embed_dim"], bias=False)
        elif args.features == "id":
            self.matchedid_embedding = nn.Embedding(
                config["vocab_sizes"][args.medium], config["embed_dim"]
            )
            x = self.matchedid_embedding
            weight_tied_linear = nn.Linear(*reversed(x.weight.shape), bias=False)
            x.weight = weight_tied_linear.weight
            return
        else:
            assert False
        self.encoder = nn.Sequential(nn.Dropout(0.5), self.encoder)

    def get_temperature(self):
        return float(self.logit_scale)

    def load_pretrained_embeddings(self, filepath):
        vocab_size = self.config["vocab_sizes"][args.medium]
        with h5py.File(filepath, "r") as hf:
            self.retrieval_embeddings.weight.data.copy_(
                torch.stack(
                    [
                        torch.tensor(hf[f"masked.{args.medium}.{i}"][()])
                        for i in range(vocab_size)
                    ]
                )
            )
            self.ranking_embeddings.weight.data.copy_(
                torch.stack(
                    [
                        torch.tensor(hf[f"causal.{args.medium}.{i}"][()])
                        for i in range(vocab_size)
                    ]
                )
            )
            self.item_text_embeddings.weight.data.copy_(
                torch.stack(
                    [
                        torch.tensor(hf[f"item_text.{args.medium}.{i}"][()])
                        for i in range(vocab_size)
                    ]
                )
            )
        self.retrieval_embeddings.weight.requires_grad = False
        self.ranking_embeddings.weight.requires_grad = False
        self.item_text_embeddings.weight.requires_grad = False

    def embed(self, ids):
        if args.features == "all":
            f_ret = self.retrieval_embeddings(ids)
            f_rnk = self.ranking_embeddings(ids)
            f_text = self.item_text_embeddings(ids)
            encoded_features = self.encoder(torch.cat([f_ret, f_rnk, f_text], dim=-1))
            return F.normalize(encoded_features, dim=-1)
        elif args.features == "rettext":
            f_ret = self.retrieval_embeddings(ids)
            f_text = self.item_text_embeddings(ids)
            encoded_features = self.encoder(torch.cat([f_ret, f_text], dim=-1))
            return F.normalize(encoded_features, dim=-1)
        elif args.features == "ranking":
            f_rnk = self.ranking_embeddings(ids)
            encoded_features = self.encoder(f_rnk)
            return F.normalize(encoded_features, dim=-1)
        elif args.features == "retrieval":
            f_ret = self.retrieval_embeddings(ids)
            encoded_features = self.encoder(f_ret)
            return F.normalize(encoded_features, dim=-1)
        elif args.features == "text":
            f_text = self.item_text_embeddings(ids)
            encoded_features = self.encoder(f_text)
            return F.normalize(encoded_features, dim=-1)
        elif args.features == "id":
            return F.normalize(self.matchedid_embedding(ids), dim=-1)
        else:
            assert False

    def pairwise_diff(self, x):
        return x.reshape(x.shape[0], x.shape[1], 1) - x.reshape(
            x.shape[0], 1, x.shape[1]
        )

    def process_batch(self, batch):
        source_features = self.embed(batch["sourceid"])
        target_features = self.embed(batch["targetid"])
        x = (source_features * target_features).sum(dim=-1) * self.logit_scale.exp()
        y = batch["relevance"]
        w = batch["weight"]
        return x, y, w

    def lambdarank_loss(self, x, y, w):
        order = torch.argsort(torch.argsort(x, descending=True)) + 1
        deltag = self.pairwise_diff(y)
        deltad = self.pairwise_diff(1 / torch.log2(1 + order))
        deltandcg = deltad * deltag
        loss = -F.logsigmoid(self.pairwise_diff(x)) * (self.pairwise_diff(y) > 0)
        wloss = loss * deltandcg.abs()
        return (wloss.sum(dim=[1, 2]) * w).sum() / w.sum()

    def ndcg(self, batch):
        x, y, w = self.process_batch(batch)
        pred_scores = x
        true_relevance = y
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
        return (ndcg * w).sum() / w.sum()

    def forward(self, batch):
        x, y, w = self.process_batch(batch)
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
        # stops if loss doesn't decrease by rtol in patience epochs
        self.patience = patience
        self.rtol = rtol
        self.counter = 0
        self.stop_score = float("inf")
        self.stop = False
        self.saved_score = float("inf")
        self.save_model = False

    def __call__(self, score):
        assert not self.stop
        if score < self.stop_score * (1 - self.rtol):
            self.counter = 0
            self.stop_score = score
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True
        if score < self.saved_score:
            self.saved_score = score
            self.save_model = True
        else:
            self.save_model = False


class LateStopper:
    def __init__(self, patience, rtol):
        # stops if loss has increased by rtol in the last patience epochs
        self.patience = patience
        self.rtol = rtol
        self.counter = 0
        self.score = float("inf")
        self.stop = False
        self.save_model = False

    def __call__(self, score):
        assert not self.stop
        if score < self.score * (1 + self.rtol):
            self.counter = 0
            self.save_model = True
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True
            self.save_model = False
        if score < self.score:
            self.score = score


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


def checkpoint_model(model, epoch, training_loss, test_loss, save):
    if save:
        checkpoint = {
            "model": model._orig_mod.state_dict(),
            "epoch": epoch,
            "training_loss": training_loss,
            "test_loss": test_loss,
        }
        torch.save(
            checkpoint,
            f"{args.datadir}/clip.model.{args.medium}.pt",
        )
    csv_fn = f"{args.datadir}/clip.model.{args.medium}.csv"
    create_csv = epoch < 0
    if create_csv:
        with open(csv_fn, "w") as f:
            f.write(",".join(["epoch", "training_loss", "test_loss", "saved"]) + "\n")
    with open(csv_fn, "a") as f:
        vals = [
            epoch,
            training_loss,
            test_loss,
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
        "embed_dim": 1024,
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
    global training_implicit_negatives
    rng = np.random.default_rng(seed=42)
    training_implicit_negatives = (
        rng.random(
            (config["vocab_sizes"][args.medium], config["vocab_sizes"][args.medium]),
            dtype=np.float32,
        )
        < 0.9
    )
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
        for x in ["training", "test"]
    }
    optimizer = create_optimizer(model, config)
    scheduler = create_learning_rate_schedule(
        optimizer,
        len(dataloaders["training"]),
        num_epochs,
    )
    scaler = torch.amp.GradScaler(args.device)
    early_stopper = EarlyStopper(patience=5, rtol=0.001)
    late_stopper = LateStopper(patience=32, rtol=0.01)
    get_loss = lambda x, y: evaluate_metrics(x, dataloaders[y])
    training_loss = get_loss(model, "training")
    logger.info(f"Epoch: -1, Training Loss: {training_loss}")
    test_loss = get_loss(model, "test")
    logger.info(f"Epoch: -1, Test Loss: {test_loss}")
    early_stopper(training_loss)
    late_stopper(test_loss)
    checkpoint_model(model, -1, training_loss, test_loss, True)
    best_losses = (training_loss, test_loss)
    for epoch in range(num_epochs):
        train_epoch(model, dataloaders["training"], optimizer, scheduler, scaler)
        generate_embeddings(model)
        for x in ["training", "test"]:
            dataloaders[x] = DataLoader(
                LTRDataset(datasplit=x, config=config, logger=logger),
                batch_size=config["batch_size"],
                shuffle=x == "training",
                drop_last=False,
                num_workers=8,
                worker_init_fn=worker_init_fn,
            )
        training_loss = get_loss(model, "training")
        logger.info(f"Epoch: {epoch}, Training Loss: {training_loss}")
        test_loss = get_loss(model, "test")
        logger.info(f"Epoch: {epoch}, Test Loss: {test_loss}")
        early_stopper(training_loss)
        late_stopper(test_loss)
        save_model = early_stopper.save_model and late_stopper.save_model
        if save_model:
            best_losses = (training_loss, test_loss)
        checkpoint_model(model, epoch, training_loss, test_loss, save_model)
        if early_stopper.stop or late_stopper.stop:
            break
    checkpoint_model(model, epoch, training_loss, test_loss, True)
    logger.info(f"Best losses: {best_losses}")


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
    for i in range(0, num_items, batch_size):
        with torch.no_grad():
            batch_ids = all_item_ids[i : i + batch_size, :]
            embs = model.embed(batch_ids)
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
training_implicit_negatives = None

if __name__ == "__main__":
    print(
        f"[ITEM SIMILARITY] training medium {args.medium} with features {args.features}"
    )
    train()
    generate_embeddings(load_model())