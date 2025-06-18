import argparse
import logging

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, IterableDataset
from tqdm import tqdm


class ClipModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.matchedid_embedding = nn.Embedding(config["vocab_size"], 2048)
        self.retrieval_embedding = nn.Embedding(config["vocab_size"], 2048)
        self.ranking_embedding = nn.Embedding(config["vocab_size"], 2048)
        self.source_encoder = nn.Linear(2048, config["embed_dim"], bias=False)
        self.target_encoder = self.source_encoder
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def load_pretrained_embeddings(self, filepath, medium):
        print(f"Loading pretrained embeddings from '{filepath}' and medium {medium}")
        x = self.matchedid_embedding
        weight_tied_linear = nn.Linear(*reversed(x.weight.shape), bias=False)
        x.weight = weight_tied_linear.weight
        with h5py.File(filepath, "r") as hf:
            for i in range(self.config["vocab_size"]):
                self.retrieval_embedding.weight.data[i] = torch.tensor(
                    hf[f"masked.{medium}.{i}"][()], dtype=torch.float
                )
                self.ranking_embedding.weight.data[i] = torch.tensor(
                    hf[f"causal.{medium}.{i}"][()], dtype=torch.float
                )
        self.matchedid_embedding.weight.requires_grad = True
        self.retrieval_embedding.weight.requires_grad = False
        self.ranking_embedding.weight.requires_grad = False

    def embed(self, x):
        return self.retrieval_embedding(x)

    def forward(self, batch):
        S_f = self.embed(batch["source"])
        T_f = self.embed(batch["target"])
        S_e = F.normalize(self.source_encoder(S_f), dim=1)
        T_e = F.normalize(self.target_encoder(T_f), dim=1)
        logits = S_e @ T_e.T * self.logit_scale.exp()
        weights = batch["weight"]
        n = logits.shape[0]
        labels = torch.arange(n, device=logits.device)
        loss_s = (
            F.cross_entropy(logits, labels, reduction="none") * weights
        ).sum() / weights.sum()
        loss_t = (
            F.cross_entropy(logits.T, labels, reduction="none") * weights
        ).sum() / weights.sum()
        return (loss_s + loss_t) / 2


class ClipDataset(IterableDataset):
    def __init__(self, csv_filepath, cliptype, batch_size, shuffle):
        super().__init__()
        self.batch_size = batch_size
        self.shuffle = shuffle
        df = pd.read_csv(csv_filepath)
        df_filtered = df[(df["cliptype"] == cliptype) & (df["count"] > 0)].copy()
        remainder = len(df_filtered) % self.batch_size
        if remainder != 0:
            num_to_pad = self.batch_size - remainder
            if self.shuffle:
                padding_df = df_filtered.sample(n=num_to_pad, replace=True)
            else:
                padding_df = df_filtered.sample(
                    n=num_to_pad, replace=True, random_state=42
                )
            df_filtered = pd.concat([df_filtered, padding_df], ignore_index=True)
        self.sources = torch.tensor(
            df_filtered["source_matchedid"].values, dtype=torch.long
        )
        self.targets = torch.tensor(
            df_filtered["target_matchedid"].values, dtype=torch.long
        )
        weights = 1 + np.log(np.abs(df_filtered["count"].values))
        self.weights = torch.tensor(weights, dtype=torch.float32)
        self.num_samples = len(self.sources)

    def __iter__(self):
        if self.shuffle:
            shuffled_indices = torch.randperm(self.num_samples)
        else:
            shuffled_indices = torch.arange(self.num_samples)
        for i in range(0, self.num_samples, self.batch_size):
            batch_indices = shuffled_indices[i : i + self.batch_size]
            batch_dict = {
                "source": self.sources[batch_indices],
                "target": self.targets[batch_indices],
                "weight": self.weights[batch_indices],
            }
            yield batch_dict


def worker_init_fn(worker_id):
    np.random.seed(torch.utils.data.get_worker_info().seed % 2**32)


def collate(data):
    assert len(data) == 1
    return data[0]


def to_device(data):
    d = {}
    for k in data:
        d[k] = data[k].to(args.device)
    return d


def evaluate_metrics(model, dataloader):
    losses = 0.0
    weights = 0.0
    progress = tqdm(desc="Test batches", mininterval=1)
    model.eval()
    for data in dataloader:
        with torch.no_grad():
            with torch.amp.autocast(f"cuda:{args.device}", dtype=torch.bfloat16):
                d = to_device(data)
                loss = model(d)
            w = float(d["weight"].sum())
            losses += float(loss) * w
            weights += w
        progress.update()
    progress.close()
    model.train()
    return losses / weights


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
            f.write(
                ",".join(["epoch", "training_loss", "test_loss", "overfit", "saved"])
                + "\n"
            )
    with open(csv_fn, "a") as f:
        vals = [
            epoch,
            training_loss,
            test_loss,
            training_loss / test_loss,
            1 if save else 0,
        ]
        f.write(",".join([str(x) for x in vals]) + "\n")


def training_config():
    m = {0: "manga", 1: "anime"}[args.medium]
    num_items = int(pd.read_csv(f"{args.datadir}/../{m}.csv").matchedid.max()) + 1
    config = {
        "vocab_size": num_items,
        "embed_dim": 512,
        "training_batch_size": 1024,
        "test_batch_size": 1024,
        "learning_rate": 3e-4,
    }
    return config


def train():
    torch.cuda.set_device(args.device)
    logger = get_logger("clip")
    logger.setLevel(logging.DEBUG)
    config = training_config()
    num_epochs = 1024
    dataloaders = {
        x: DataLoader(
            ClipDataset(
                csv_filepath=f"{args.datadir}/{x}.csv",
                cliptype=f"medium{args.medium}",
                batch_size=config[f"{x}_batch_size"],
                shuffle=x == "training",
            ),
            batch_size=1,
            drop_last=False,
            num_workers=1,
            persistent_workers=True,
            collate_fn=collate,
            worker_init_fn=worker_init_fn,
        )
        for x in ["training", "test"]
    }
    model = ClipModel(config)
    model.load_pretrained_embeddings(f"{args.datadir}/input.h5", args.medium)
    logger.info(
        f"Created model with {sum(p.numel() for p in model.parameters())} parameters"
        f" and {sum(p.numel() for p in model.parameters() if p.requires_grad)} trainable parameters"
    )
    model = model.to(args.device)
    model = torch.compile(model)
    optimizer = create_optimizer(model, config)
    scheduler = create_learning_rate_schedule(
        optimizer,
        int(
            np.ceil(
                dataloaders["training"].dataset.num_samples
                / config["training_batch_size"]
            )
        ),
        num_epochs,
    )
    scaler = torch.amp.GradScaler(args.device)
    stopper = EarlyStopper(patience=5, rtol=0.001)
    get_loss = lambda x: evaluate_metrics(x, dataloaders["test"])
    initial_loss = get_loss(model)
    logger.info(f"Initial Loss: {initial_loss}")
    stopper(initial_loss)
    checkpoint_model(model, -1, np.nan, initial_loss, True)
    for epoch in range(num_epochs):
        training_loss = train_epoch(
            model, dataloaders["training"], optimizer, scheduler, scaler
        )
        logger.info(f"Epoch: {epoch}, Training Loss: {training_loss}")
        test_loss = get_loss(model)
        logger.info(f"Epoch: {epoch}, Test Loss: {test_loss}")
        stopper(test_loss)
        checkpoint_model(model, epoch, training_loss, test_loss, stopper.save_model)
        if stopper.early_stop:
            break
    logger.info(f"Best loss: {stopper.saved_score}")


def generate_embeddings():
    torch.cuda.set_device(args.device)
    config = training_config()
    device = args.device
    checkpoint_path = f"{args.datadir}/clip.model.{args.medium}.pt"
    output_path = f"{args.datadir}/output.embeddings.{args.medium}.h5"
    model = ClipModel(config)
    checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model = torch.compile(model)
    model.eval()
    num_items = config["vocab_size"]
    batch_size = config["test_batch_size"]
    final_embeddings = np.zeros((num_items, config["embed_dim"]), dtype=np.float32)
    all_item_ids = torch.arange(num_items, device=device)
    for i in tqdm(range(0, num_items, batch_size), desc="Saving Embeddings"):
        with torch.no_grad():
            batch_ids = all_item_ids[i : i + batch_size]
            initial_embs = model.embed(batch_ids)
            projected_embs = model.source_encoder(initial_embs)
            normalized_embs = F.normalize(projected_embs, dim=1)
            final_embeddings[i : i + batch_size] = normalized_embs.cpu().numpy()
    with h5py.File(output_path, "w") as hf:
        for i in range(num_items):
            key = f"{args.medium}.{i}"
            hf.create_dataset(key, data=final_embeddings[i])


parser = argparse.ArgumentParser()
parser.add_argument("--datadir", type=str)
parser.add_argument("--device", type=int)
parser.add_argument("--medium", type=int)
args = parser.parse_args()

if __name__ == "__main__":
    train()
    generate_embeddings()