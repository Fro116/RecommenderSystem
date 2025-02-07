import argparse
import glob
import logging
import os

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


class BagOfWordsDataset(IterableDataset):
    def __init__(self, datadir, split, batch_size):
        self.datadir = f"{datadir}/bagofwords/{split}"
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
        mediums = [0, 1]
        metrics = ["rating", "watch", "plantowatch", "drop"]
        for fn in fns:
            with h5py.File(fn, "r") as f:
                X = self.load_sparse_matrix(f, "X")
                Y = [
                    self.load_sparse_matrix(f, f"Y_{medium}_{metric}")
                    for medium in mediums
                    for metric in metrics
                ]
                W = [
                    self.load_sparse_matrix(f, f"W_{medium}_{metric}")
                    for medium in mediums
                    for metric in metrics
                ]
            idxs = list(range(X.shape[0]))
            np.random.shuffle(idxs)
            while len(idxs) % self.batch_size != 0:
                idxs.append(np.random.choice(idxs))
            idxs = [
                idxs[i : i + self.batch_size]
                for i in range(0, len(idxs), self.batch_size)
            ]
            for idx in idxs:
                yield X[idx, :], [y[idx, :] for y in Y], [w[idx, :] for w in W]
        self.epoch = (self.epoch + 1) % self.epochs


def collate(data):
    assert len(data) == 1
    return data[0]


def to_device(x):
    if isinstance(x, list):
        return [to_device(y) for y in x]
    return (
        torch.sparse_csr_tensor(x.indptr, x.indices, x.data, x.shape)
        .to(device)
        .to_dense()
    )


class BagOfWordsModel(nn.Module):
    def __init__(self, datadir):
        super(BagOfWordsModel, self).__init__()
        num_items = {
            x: pd.read_csv(f"{datadir}/{y}.csv").matchedid.max() + 1
            for (x, y) in {0: "manga", 1: "anime"}.items()
        }
        self.model = nn.Sequential(
            nn.Linear(sum(num_items.values()) * 2, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
        )
        mediums = [0, 1]
        metrics = ["rating", "watch", "plantowatch", "drop"]
        self.classifiers = nn.ModuleList(
            [nn.Linear(256, num_items[medium]) for medium in mediums for _ in metrics]
        )
        self.logsoftmax = nn.LogSoftmax(dim=-1)
        self.softmax = nn.Softmax(dim=-1)
        self.sigmoid = nn.Sigmoid()
        lossfn_map = {
            "rating": self.mse,
            "watch": self.crossentropy,
            "plantowatch": self.crossentropy,
            "drop": self.binarycrossentropy,
        }
        self.lossfns = [lossfn_map[metric] for _ in mediums for metric in metrics]
        evaluate_map = {
            "rating": self.moments,
            "watch": self.crossentropy,
            "plantowatch": self.crossentropy,
            "drop": self.binarycrossentropy,
        }
        self.evaluatefns = [evaluate_map[metric] for _ in mediums for metric in metrics]

    def mse(self, x, y, w):
        return (torch.square(x - y) * w).sum()

    def crossentropy(self, x, y, w):
        x = self.logsoftmax(x)
        return (-x * y * w).sum()

    def binarycrossentropy(self, x, y, w):
        return nn.functional.binary_cross_entropy_with_logits(
            input=x,
            target=y,
            weight=w,
            reduction="sum",
        )

    def moments(self, x, y, w):
        return (
            self.mse(1 * x, y, w),
            self.mse(0 * x, y, w),
            self.mse(-1 * x, y, w),
        )

    def forward(self, inputs, labels, weights, mode):
        e = self.model(inputs)
        if mode == "training":
            return [
                fn(c(e), l, w)
                for (c, fn, l, w) in zip(
                    self.classifiers, self.lossfns, labels, weights
                )
            ], [w.sum() for w in weights]
        elif mode == "evaluation":
            return [
                fn(c(e), l, w)
                for (c, fn, l, w) in zip(
                    self.classifiers, self.evaluatefns, labels, weights
                )
            ], [w.sum() for w in weights]
        else:
            assert False


def minimize_quadratic(x, y):
    assert len(x) == 3 and len(y) == 3
    A = np.array([[x[0] ** 2, x[0], 1], [x[1] ** 2, x[1], 1], [x[2] ** 2, x[2], 1]])
    B = np.array(y)
    a, b, c = np.linalg.solve(A, B)
    x_extremum = -b / (2 * a)
    y_extremum = a * x_extremum**2 + b * x_extremum + c
    return float(y_extremum)


def evaluate_metrics(model, dataloader, taskweights):
    mediums = [0, 1]
    metrics = ["rating", "watch", "plantowatch", "drop"]
    init = lambda metric: [0, 0, 0] if metric == "rating" else 0
    losses = [init(metric) for _ in mediums for metric in metrics]
    weights = [0 for _ in range(len(mediums) * len(metrics))]
    progress = tqdm(desc="Test batches", mininterval=1)
    model.eval()
    for data in dataloader:
        with torch.no_grad():
            with torch.amp.autocast(device, dtype=torch.bfloat16):
                loss, w = model(*to_device(data), mode="evaluation")
            for i in range(len(loss)):
                w[i] = float(w[i])
                if w[i] != 0:
                    if isinstance(losses[i], list):
                        for j in range(len(losses[i])):
                            losses[i][j] += float(loss[i][j])
                    else:
                        losses[i] += float(loss[i])
                    weights[i] += w[i]
        progress.update()
    progress.close()
    model.train()
    for i in range(len(losses)):
        if isinstance(losses[i], list):
            losses[i] = minimize_quadratic([1, 0, -1], losses[i])
        if weights[i] != 0:
            losses[i] /= weights[i]
        else:
            losses[i] = 0
    return sum(l * w for (l, w) in zip(losses, taskweights)), losses


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


def train_epoch(model, dataloader, optimizer, scaler, taskweights):
    training_losses = [0.0 for _ in taskweights]
    weights = [0 for _ in taskweights]
    progress = tqdm(desc="Training batches", mininterval=1)
    for data in dataloader:
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device, dtype=torch.bfloat16):
            losses, wsums = model(
                *to_device(data),
                mode="training",
            )
        for i in range(len(taskweights)):
            training_losses[i] += float(losses[i])
            weights[i] += float(wsums[i])
        invweights = [sum(weights) / x if x != 0 else 0 for x in weights]
        loss = sum(l * w * z for (l, w, z) in zip(losses, taskweights, invweights))
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        progress.update()
    progress.close()
    tasklosses = [l / w if w != 0 else 0 for (l, w) in zip(training_losses, weights)]
    invweights = [sum(weights) / x if x != 0 else 0 for x in weights]
    return (
        sum(l * w * z for (l, w, z) in zip(tasklosses, taskweights, invweights)),
        tasklosses,
    )


def save_model(model, datadir, taskweights):
    fn = os.path.join(
        datadir,
        "_".join(["bagofwords"] + [str(x) for x in taskweights] + ["model.pt"]),
    )
    temp = fn + "~"
    torch.save(model._orig_mod.state_dict(), temp)
    os.rename(temp, fn)


def load_data(datadir, logger):
    if os.path.exists(f"{datadir}/bagofwords"):
        return
    logger.info("downloading training data")
    templatefn = f"{datadir}/../../environment/database/trainingdata.txt"
    with open(templatefn) as f:
        template = f.read()
    cmd = f"{template}/bagofwords {datadir}/bagofwords"
    os.system(cmd)


def train(datadir, taskweights):
    logger = get_logger("bagofwords")
    load_data(datadir, logger)
    logger.setLevel(logging.DEBUG)
    logger.info(f"training {taskweights}")
    dataloaders = {
        x: DataLoader(
            BagOfWordsDataset(datadir, x, 2048),
            batch_size=1,
            drop_last=False,
            num_workers=8,
            persistent_workers=True,
            pin_memory=True,
            collate_fn=collate,
        )
        for x in ["training", "test"]
    }
    model = BagOfWordsModel(datadir)
    model = model.to(device)
    model = torch.compile(model)
    optimizer = create_optimizer(model)
    scaler = torch.amp.GradScaler(device)
    stopper = EarlyStopper(patience=3, rtol=1e-4)
    starting_epoch = 0
    get_loss = lambda x: evaluate_metrics(x, dataloaders["test"], taskweights)
    save = lambda x: save_model(x, datadir, taskweights)
    save(model)
    initial_loss = get_loss(model)
    logger.info(f"Initial Loss: {initial_loss}")
    stopper(initial_loss[0])
    save(model)
    for epoch in range(starting_epoch, 100):
        training_loss = train_epoch(
            model, dataloaders["training"], optimizer, scaler, taskweights
        )
        logger.info(f"Epoch: {epoch}, Training Loss: {training_loss}")
        test_loss = get_loss(model)
        logger.info(f"Epoch: {epoch}, Test Loss: {test_loss}")
        stopper(test_loss[0])
        if stopper.save_model:
            save(model)
        if stopper.early_stop:
            break


device = "cuda"
parser = argparse.ArgumentParser()
parser.add_argument("--datadir", type=str)
parser.add_argument("--taskweights", type=int, nargs='+')
args = parser.parse_args()

if __name__ == "__main__":
    train(args.datadir, args.taskweights)