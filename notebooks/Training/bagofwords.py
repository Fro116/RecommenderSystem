import argparse
import glob
import logging
import os

import filelock
import h5py
import hdf5plugin
import numpy as np
import pandas as pd
import scipy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, IterableDataset
from tqdm import tqdm

with open("bagofwords.model.py") as f:
    exec(f.read())


class BagOfWordsDataset(IterableDataset):
    def __init__(self, datadir, medium, metric, split, batch_size):
        self.datadir = f"{datadir}/bagofwords/{split}"
        self.medium = medium
        self.metric = metric
        self.epoch = 0
        self.epochs = len(glob.glob(f"{self.datadir}/*"))
        self.batch_size = batch_size

    def load_sparse_matrix(self, f, name):
        i = f[f"{name}_i"][:] - 1
        j = f[f"{name}_j"][:] - 1
        v = f[f"{name}_v"][:]
        m, n = f[f"{name}_size"][:]
        return scipy.sparse.coo_matrix((v, (j, i)), shape=(n, m)).tocsr()

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            worker_id = 0
            num_workers = 1
        else:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
        fns = sorted(glob.glob(f"{self.datadir}/{self.epoch+1}/*.h5"))
        fns = [x for i, x in enumerate(fns) if i % num_workers == worker_id]
        for fn in fns:
            with h5py.File(fn, "r") as f:
                X = self.load_sparse_matrix(f, "X")
                Y = self.load_sparse_matrix(f, f"Y_{self.medium}_{self.metric}")
                W = self.load_sparse_matrix(f, f"W_{self.medium}_{self.metric}")
            idxs = list(range(X.shape[0]))
            np.random.shuffle(idxs)
            while len(idxs) % self.batch_size != 0:
                idxs.append(np.random.choice(idxs))
            idxs = [
                idxs[i : i + self.batch_size]
                for i in range(0, len(idxs), self.batch_size)
            ]
            for idx in idxs:
                yield X[idx, :], Y[idx, :], W[idx, :]
        self.epoch = (self.epoch + 1) % self.epochs


def collate(data):
    assert len(data) == 1
    return [
        torch.sparse_csr_tensor(x.indptr, x.indices, x.data, x.shape) for x in data[0]
    ]


def to_device(data):
    return [x.to(device).to_dense() for x in data]


def minimize_quadratic(x, y):
    assert len(x) == 3 and len(y) == 3
    A = np.array([[x[0] ** 2, x[0], 1], [x[1] ** 2, x[1], 1], [x[2] ** 2, x[2], 1]])
    B = np.array(y)
    a, b, c = np.linalg.solve(A, B)
    x_extremum = -b / (2 * a)
    y_extremum = a * x_extremum**2 + b * x_extremum + c
    return float(y_extremum)


def evaluate_metrics(model, dataloader, metric):
    losses = [0, 0, 0] if metric == "rating" else 0
    weights = 0
    progress = tqdm(desc="Test batches", mininterval=1)
    model.eval()
    for data in dataloader:
        with torch.no_grad():
            with torch.amp.autocast(device, dtype=torch.bfloat16):
                loss, w = model(*to_device(data), mode="eval")
            w = float(w)
            if w == 0:
                continue
            if isinstance(losses, list):
                for i in range(len(loss)):
                    losses[i] += float(loss[i])
            else:
                losses += float(loss)
            weights += w
        progress.update()
    progress.close()
    model.train()
    if metric == "rating":
        losses = minimize_quadratic([1, 0, -1], losses)
    return losses / weights


class EarlyStopper:
    def __init__(self, patience, rtol):
        self.patience = patience
        self.counter = 0
        self.best_score = float("inf")
        self.early_stop = False
        self.save_model = False
        self.rtol = rtol

    def __call__(self, score):
        if score > self.best_score * (1 - self.rtol):
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.counter = 0
        self.save_model = False
        if score < self.best_score:
            self.best_score = score
            self.save_model = True


def create_optimizer(model):
    learning_rate = 4e-5
    weight_decay = 1e-2
    decay_parameters = []
    no_decay_parameters = []
    for name, param in model.named_parameters():
        if name.startswith("embed") or "norm" in name or "bias" in name:
            no_decay_parameters.append(param)
        else:
            decay_parameters.append(param)
    return optim.AdamW(
        [
            {"params": decay_parameters, "weight_decay": weight_decay},
            {"params": no_decay_parameters, "weight_decay": 0.0},
        ],
        lr=learning_rate,
        betas=(0.9, 0.999),
    )


def get_logger(name):
    logger = logging.getLogger(name)
    formatter = logging.Formatter(
        "%(name)s:%(levelname)s:%(asctime)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    logger.addHandler(stream)
    logger.setLevel(logging.DEBUG)
    return logger


def train_epoch(model, dataloader, optimizer, scaler):
    losses = 0
    weights = 0
    progress = tqdm(desc="Training batches", mininterval=1)
    for data in dataloader:
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device, dtype=torch.bfloat16):
            loss, w = model(*to_device(data), mode="train")
        losses += float(loss)
        weights += float(w)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        progress.update()
    progress.close()
    return losses / weights


def save_model(model, loss, epoch, datadir, medium, metric):
    stem = f"{datadir}/bagofwords.{medium}.{metric}"
    torch.save(model._orig_mod.state_dict(), f"{stem}.pt")
    with open(f"{stem}.csv", "w") as f:
        f.write("loss,epoch\n")
        f.write(f"{loss},{epoch}\n")


def upload(model, datadir, medium, metric, logger):
    logger.info("uploading model")
    templatefn = f"{datadir}/../../environment/database/upload.txt"
    with open(templatefn) as f:
        template = f.read()
    for suffix in ["pt", "csv"]:
        cmd = template.replace(
            "{INPUT}", f"{datadir}/bagofwords.{medium}.{metric}.{suffix}"
        ).replace(
            "{OUTPUT}",
            f"bagofwords.{medium}.{metric}.{suffix}",
        )
        os.system(cmd)


def download(datadir, logger):
    logger.info("waiting for download lock")
    lock = filelock.FileLock(f"{datadir}/bagofwords.lock")
    with lock:
        if os.path.exists(f"{datadir}/bagofwords"):
            return
        logger.info("downloading data")
        templatefn = f"{datadir}/../../environment/database/download.txt"
        with open(templatefn) as f:
            template = f.read()
        for data in ["manga.csv", "anime.csv", "bagofwords"]:
            os.system(f"{template}/{data} {datadir}/{data}")


def train(datadir, medium, metric):
    logger = get_logger("bagofwords")
    download(datadir, logger)
    logger.setLevel(logging.DEBUG)
    logger.info(f"training {medium} {metric}")
    dataloaders = {
        x: DataLoader(
            BagOfWordsDataset(datadir, medium, metric, x, 2048),
            batch_size=1,
            drop_last=False,
            num_workers=16,
            persistent_workers=True,
            collate_fn=collate,
        )
        for x in ["training", "test"]
    }
    model = BagOfWordsModel(datadir, medium, metric)
    model = model.to(device)
    model = torch.compile(model)
    optimizer = create_optimizer(model)
    scaler = torch.amp.GradScaler(device)
    stopper = EarlyStopper(patience=3, rtol=1e-3)
    get_loss = lambda x: evaluate_metrics(x, dataloaders["test"], metric)
    save = lambda m, l, e: save_model(m, l, e, datadir, medium, metric)
    initial_loss = get_loss(model)
    logger.info(f"Initial Loss: {initial_loss}")
    stopper(initial_loss)
    save(model, initial_loss, 0)
    for epoch in range(128):
        training_loss = train_epoch(model, dataloaders["training"], optimizer, scaler)
        logger.info(f"Epoch: {epoch}, Training Loss: {training_loss}")
        test_loss = get_loss(model)
        logger.info(f"Epoch: {epoch}, Test Loss: {test_loss}")
        stopper(test_loss)
        if stopper.save_model:
            save(model, test_loss, epoch)
        if stopper.early_stop:
            break
    upload(model, datadir, medium, metric, logger)


parser = argparse.ArgumentParser()
parser.add_argument("--datadir", type=str)
parser.add_argument("--medium", type=int)
parser.add_argument("--metric", type=str)
parser.add_argument("--device", type=str)
args = parser.parse_args()
device = args.device

if __name__ == "__main__":
    train(args.datadir, args.medium, args.metric)
