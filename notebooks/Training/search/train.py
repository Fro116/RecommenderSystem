import argparse
import glob
import logging

import h5py
import hdf5plugin
import numpy as np
import pandas as pd
import scipy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import (
    BatchSampler,
    DataLoader,
    Dataset,
    RandomSampler,
    SequentialSampler,
)
from tqdm import tqdm


class SearchDataset(Dataset):
    def __init__(self, datasplit):
        super().__init__()
        assert datasplit in ["training", "test"]
        self.datasplit = datasplit
        fn = f"{args.datadir}/{datasplit}.{args.medium}.h5"
        with h5py.File(fn) as f:
            d = {}
            for k in f:
                d[k] = f[k][:]
        self.query_embeddings = d["query_embeddings"]
        n_items, n_queries = d["targets_size"]
        self.targets = scipy.sparse.csc_matrix(
            (d["targets_nzval"], d["targets_rowval"] - 1, d["targets_colptr"] - 1),
            shape=(n_items, n_queries),
        ).T.tocsr()
        self.weights = d["weights"]

    def __len__(self):
        return self.query_embeddings.shape[0]

    def __getitem__(self, idx):
        # uses a batchsampler so idx is a list
        m = self.targets[idx]
        targets = torch.from_numpy(m.toarray())
        return {
            "queries": torch.from_numpy(self.query_embeddings[idx, :]),
            "targets": targets,
            "weights": torch.from_numpy(self.weights[idx, :]),
        }

    def entropy(self):
        # crossentropy = entropy + KL(p || q) so this is a lower bound on the loss
        p = self.targets.astype(np.float64)
        p.eliminate_zeros()
        p.data = -p.data * np.log(p.data)
        row_entropy = np.asarray(p.sum(axis=1)).ravel()
        return float(np.average(row_entropy, weights=self.weights.ravel()))


def get_num_items(medium):
    m = {0: "manga", 1: "anime"}[medium]
    col = "matchedid"
    df = pd.read_csv(f"{args.datadir}/../{m}.csv", low_memory=False)
    return int(df[col].max()) + 1


def get_training_config():
    config = {
        "vocab_sizes": {x: get_num_items(x) for x in [0, 1]},
        "learning_rate": 3e-4,
        "batch_size": 1024,
        "content_embedding_size": 3072,
        "transformer_embedding_size": 2048,
        "weight_decay": 0,
        "dropout": 0.1,
    }
    return config


class SearchModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        size_c = config["content_embedding_size"]
        size_t = config["transformer_embedding_size"]
        self.content_embeddings = nn.Embedding(
            config["vocab_sizes"][args.medium], size_c
        )
        self.transformer_embeddings = nn.Embedding(
            config["vocab_sizes"][args.medium], size_t
        )
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.content_adapter = (
            nn.Linear(size_c, size_c, bias=False)
            if args.features in ["both", "content"]
            else None
        )
        self.transformer_adapter = (
            nn.Linear(size_t, size_c, bias=False)
            if args.features in ["both", "transformer"]
            else None
        )
        # self.alpha = nn.Parameter(torch.zeros(()))
        self.dropout = nn.Dropout(config["dropout"])

    def get_temperature(self):
        return float(self.logit_scale.detach())

    def load_pretrained_embeddings(self, filepath):
        with h5py.File(filepath, "r") as hf:
            self.transformer_embeddings.weight.data.copy_(
               F.normalize(torch.tensor(hf[f"transformer.{args.medium}"][:]), dim=-1)
            )
            self.content_embeddings.weight.data.copy_(
               F.normalize(torch.tensor(hf[f"text.{args.medium}"][:]), dim=-1)
            )
        self.transformer_embeddings.weight.requires_grad = False
        self.content_embeddings.weight.requires_grad = False

    def embed(self):
        if args.features == "notrain":
            return F.normalize(self.content_embeddings.weight, dim=-1)
        content = 0
        transformer = 0
        if self.content_adapter is not None:
            content = self.content_adapter(self.dropout(self.content_embeddings.weight))
        if self.transformer_adapter is not None:
            transformer = self.transformer_adapter(self.dropout(self.transformer_embeddings.weight))
        return F.normalize(content + transformer, dim=-1)

    def forward(self, data):
        queries, targets, weights = data["queries"], data["targets"], data["weights"]
        x = F.normalize(queries, dim=-1)
        w = weights.ravel()
        logits = x.matmul(self.embed().t()) * self.logit_scale.exp()
        ce_loss = F.cross_entropy(logits, targets, reduction="none")
        return (ce_loss * w).sum() / w.sum()


def to_device(data):
    return {k: data[k].to(args.device) for k in data}


def evaluate_metrics(model, dataloader):
    losses = 0.0
    weights = 0.0
    progress = tqdm(desc="Eval batches", mininterval=1)
    model.eval()
    for data in dataloader:
        with torch.no_grad():
            with torch.amp.autocast(f"cuda:{args.device}", dtype=torch.bfloat16):
                d = to_device(data)
                loss = model(d)
            w = float(d["weights"].sum())
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
):
    training_losses = 0.0
    training_weights = 0.0
    progress = tqdm(desc="Training batches", mininterval=1)
    optimizer.zero_grad(set_to_none=True)
    for step, data in enumerate(dataloader):
        with torch.amp.autocast(f"cuda:{args.device}", dtype=torch.bfloat16):
            d = to_device(data)
            loss = model(d)
            w = float(d["weights"].sum())
            training_losses += float(loss.detach()) * w
            training_weights += w
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
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
            {"params": decay_params, "weight_decay": config["weight_decay"]},
            {"params": nodecay_params, "weight_decay": 0.0},
        ],
        lr=config["learning_rate"],
        # betas=(0.9, 0.95),
        fused=True,
    )


class ConstantScheduler(object):
    def __init__(self):
        self.steps = 0

    def __call__(self, epoch):
        self.steps += 1
        return 1


def create_learning_rate_schedule(optimizer):
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
            f"{args.datadir}/search.model.{args.medium}.pt",
        )
    csv_fn = f"{args.datadir}/search.model.{args.medium}.csv"
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


def train():
    torch.cuda.set_device(args.device)
    logger = get_logger("search")
    config = get_training_config()
    num_epochs = 1024
    dataloaders = {}
    for x in ["training", "test"]:
        ds = SearchDataset(x)
        sampler = RandomSampler(ds) if x == "training" else SequentialSampler(ds)
        dataloaders[x] = DataLoader(
            ds,
            sampler=BatchSampler(
                sampler, batch_size=config["batch_size"], drop_last=False
            ),
            batch_size=None,
            num_workers=0,
        )
    model = SearchModel(config)
    model.load_pretrained_embeddings(f"{args.datadir}/../embeddings.h5")
    logger.info(
        f"Created model with {sum(p.numel() for p in model.parameters())} parameters"
        f" and {sum(p.numel() for p in model.parameters() if p.requires_grad)} trainable parameters"
    )
    model = model.to(args.device)
    model = torch.compile(model)
    optimizer = create_optimizer(model, config)
    scheduler = create_learning_rate_schedule(optimizer)
    stopper = EarlyStopper(patience=5, rtol=0.001)
    get_loss = lambda x: evaluate_metrics(x, dataloaders["test"])
    initial_loss = get_loss(model)
    entropy = dataloaders["test"].dataset.entropy()
    logger.info(f"Initial Loss: {initial_loss}, Entropy: {entropy}")
    stopper(initial_loss)
    checkpoint_model(model, -1, initial_loss, initial_loss, True)
    best_losses = (initial_loss, initial_loss)
    for epoch in range(num_epochs):
        training_loss = train_epoch(
            model,
            dataloaders["training"],
            optimizer,
            scheduler,
        )
        logger.info(f"Epoch: {epoch}, Training Loss: {training_loss}")
        test_loss = get_loss(model)
        logger.info(f"Epoch: {epoch}, Test Loss: {test_loss}")
        stopper(test_loss)
        if stopper.save_model:
            best_losses = (training_loss, test_loss)
        checkpoint_model(model, epoch, training_loss, test_loss, stopper.save_model)
        if stopper.early_stop:
            break
    logger.info(f"Best losses: {best_losses}")


def generate_embeddings():
    torch.cuda.set_device(args.device)
    config = get_training_config()
    checkpoint_path = f"{args.datadir}/search.model.{args.medium}.pt"
    model = SearchModel(config)
    checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))
    model.load_state_dict(checkpoint["model"])
    model.to(args.device)
    model = torch.compile(model)
    model.eval()
    output_path = f"{args.datadir}/output.embeddings.{args.medium}.h5"
    temperature = model.get_temperature()
    with torch.no_grad():
        W = model.embed().to("cpu").numpy()
    with h5py.File(output_path, "w") as hf:
        key = f"search.{args.medium}"
        hf.create_dataset(key, data=W)
        hf.create_dataset("temperature", data=np.array([temperature]))


parser = argparse.ArgumentParser()
parser.add_argument("--datadir", type=str)
parser.add_argument("--device", type=int)
parser.add_argument("--medium", type=int)
parser.add_argument("--features", type=str)
args = parser.parse_args()
if __name__ == "__main__":
    train()
    generate_embeddings()