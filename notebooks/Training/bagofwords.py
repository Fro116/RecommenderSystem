import argparse
import glob
import logging
import os
import time
import warnings

warnings.filterwarnings("ignore", ".*Sparse CSR tensor support is in beta state.*")

import filelock
import h5py
import hdf5plugin
import msgpack
import numpy as np
import pandas as pd
import scipy
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, IterableDataset
from tqdm import tqdm

with open("bagofwords.model.py") as f:
    exec(f.read())


class BagOfWordsDataset(IterableDataset):
    def __init__(
        self,
        datadir,
        rank,
        world_size,
        mask_rate,
        weight_by_user,
        shuffle,
        batch_size,
    ):
        self.datadir = datadir
        self.mask_rate = mask_rate
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.weight_by_user = weight_by_user
        shards = sorted(glob.glob(f"{self.datadir}/*"))
        assert len(shards) % world_size == 0
        self.fns = []
        for i, x in enumerate(shards):
            if i % world_size == rank:
                self.fns.extend(glob.glob(f"{x}/*.h5"))

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
        fns = [x for i, x in enumerate(self.fns) if i % num_workers == worker_id]
        for fn in fns:
            with h5py.File(fn, "r") as f:
                d = {}
                for m in [0, 1]:
                    for t in ["rating", "watch"]:
                        d[f"X_{m}_{t}"] = self.load_sparse_matrix(f, f"X_{m}_{t}")
                d[f"Y_{medium}_{metric}"] = self.load_sparse_matrix(
                    f, f"Y_{medium}_{metric}"
                )
                d[f"W_{medium}_{metric}"] = self.load_sparse_matrix(
                    f, f"W_{medium}_{metric}"
                )
            N = next(iter(d.values())).shape[0]
            idxs = list(range(N))
            if self.shuffle:
                np.random.shuffle(idxs)
                while len(idxs) % self.batch_size != 0:
                    idxs.append(np.random.choice(idxs))
            idxs = [
                idxs[i : i + self.batch_size]
                for i in range(0, len(idxs), self.batch_size)
            ]
            for idx in idxs:
                ret = {
                    "mask_rate": self.mask_rate,
                    "weight_by_user": self.weight_by_user,
                }
                for k, v in d.items():
                    ret[k] = v[idx, :]
                yield ret


def collate(data):
    assert len(data) == 1
    ret = {}
    for k, v in data[0].items():
        if k in ["mask_rate", "weight_by_user"]:
            ret[k] = v
        else:
            ret[k] = torch.sparse_csr_tensor(v.indptr, v.indices, v.data, v.shape)
    return ret


def to_device(data, baselines, rank):
    Y = data[f"Y_{medium}_{metric}"].to(rank).to_dense()
    W = data[f"W_{medium}_{metric}"].to(rank).to_dense()
    if data["mask_rate"] is not None:
        mask_rate = data["mask_rate"]
        masks = [
            torch.rand(data[f"X_{m}_rating"].shape, device=rank) < mask_rate for m in [0, 1]
        ]
        Y[~masks[medium]] = 0
        W[~masks[medium]] = 0
    if data["weight_by_user"]:
        W = W / W.sum(dim=1).reshape(-1, 1).clip(1)
    d = {}
    for m in [0, 1]:
        d[f"X_{m}_rating"] = data[f"X_{m}_rating"].to(rank).to_dense()
        d[f"X_{m}_watch"] = data[f"X_{m}_watch"].to(rank).to_dense()
        if data["mask_rate"] is not None:
            d[f"X_{m}_rating"][masks[m]] = 0
            d[f"X_{m}_watch"][masks[m]] = 0
        r = d[f"X_{m}_rating"]
        _, λ_u, _, λ_wu, λ_wa = baselines[m]["params"]
        user_count = (r != 0).sum(dim=1)
        user_weights = user_count**λ_wu
        user_weights[user_count == 0] = 0
        item_count = (r != 0) * baselines[m]["item_counts"]
        item_weights = item_count**λ_wa
        item_weights[item_count == 0] = 0
        weights = user_weights.reshape(-1, 1) * item_weights
        user_baseline = ((r - baselines[m]["a"]) * weights).sum(dim=1) / (
            weights.sum(dim=1) + np.exp(λ_u)
        )
        pred = (
            user_baseline.reshape(-1, 1) + baselines[m]["a"].reshape(1, -1)
        ) * baselines[m]["weight"]
        d[f"X_{m}_rating"] = (d[f"X_{m}_rating"] != 0) * (d[f"X_{m}_rating"] - pred)
        if medium == m and metric == "rating":
            Y = (Y != 0) * (Y - pred)
    X = torch.cat(
        (d[f"X_0_rating"], d[f"X_0_watch"], d[f"X_1_rating"], d[f"X_1_watch"]), dim=1
    )
    return X, Y, W


def minimize_quadratic(x, y):
    assert len(x) == 3 and len(y) == 3
    A = np.array([[x[0] ** 2, x[0], 1], [x[1] ** 2, x[1], 1], [x[2] ** 2, x[2], 1]])
    B = np.array(y)
    a, b, c = np.linalg.solve(A, B)
    x_extremum = -b / (2 * a)
    y_extremum = a * x_extremum**2 + b * x_extremum + c
    return float(y_extremum)


def reduce_mean(rank, x, w):
    x = torch.tensor(x).to(rank)
    w = torch.tensor(w).to(rank)
    dist.all_reduce(x, op=dist.ReduceOp.SUM)
    dist.all_reduce(w, op=dist.ReduceOp.SUM)
    return [float(a) / float(b) if float(b) != 0 else 0 for (a, b) in zip(x, w)]


def evaluate_metrics(rank, model, baselines, dataloader, metric):
    losses = [0, 0, 0] if metric == "rating" else 0
    weights = 0
    progress = tqdm(desc="Test batches", mininterval=1, disable=rank != 0)
    model.eval()
    for data in dataloader:
        with torch.no_grad():
            with torch.amp.autocast(f"cuda:{rank}", dtype=torch.bfloat16):
                loss, w = model(*to_device(data, baselines, rank), mode="eval")
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
    return reduce_mean(rank, [losses], [weights])[0]


def train_epoch(rank, model, baselines, dataloader, optimizer, scaler):
    losses = 0
    weights = 0
    progress = tqdm(desc="Training batches", mininterval=1, disable=rank != 0)
    for data in dataloader:
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(f"cuda:{rank}", dtype=torch.bfloat16):
            loss, w = model(*to_device(data, baselines, rank), mode="train")
        losses += float(loss)
        weights += float(w)
        tloss = loss / w if w != 0 else loss
        scaler.scale(tloss).backward()
        scaler.step(optimizer)
        scaler.update()
        progress.update()
    progress.close()
    return reduce_mean(rank, [losses], [weights])[0]


def create_optimizer(model):
    learning_rate = 3e-4 if finetune is None else 3e-5
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


def get_logger(rank, name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    if rank != 0:
        return logger
    formatter = logging.Formatter(
        "%(name)s:%(levelname)s:%(asctime)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    logger.addHandler(stream)
    return logger


def checkpoint_model(rank, model, save, loss, epoch):
    if rank != 0:
        return
    stem = f"{datadir}/bagofwords.{medium}.{metric}"
    if finetune is not None:
        stem = f"{stem}.finetune"
    if save:
        torch.save(model.module._orig_mod.state_dict(), f"{stem}.pt")
    if epoch < 0:
        with open(f"{stem}.csv", "w") as f:
            f.write("epoch,loss\n")
    with open(f"{stem}.csv", "a") as f:
        f.write(f"{epoch},{loss}\n")


def upload(rank, logger):
    if rank != 0:
        return
    logger.info("uploading model")
    templatefn = f"{datadir}/../../environment/database/upload.txt"
    with open(templatefn) as f:
        template = f.read()
    for suffix in ["pt", "csv"]:
        if finetune is not None:
            suffix = f"finetune.{suffix}"
        cmd = template.replace(
            "{INPUT}", f"{datadir}/bagofwords.{medium}.{metric}.{suffix}"
        ).replace(
            "{OUTPUT}",
            f"bagofwords.{medium}.{metric}.{suffix}",
        )
        os.system(cmd)


def ddp_setup():
    init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


def train():
    ddp_setup()
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    baselines = get_baselines(rank)
    logger = get_logger(rank, "bagofwords")
    logger.setLevel(logging.DEBUG)
    logger.info(f"training {medium} {metric}")
    batch_size = 32768 if finetune is None else 2048
    assert batch_size % world_size == 0
    batch_size = batch_size // world_size
    if finetune is None:
        dataloader_args = [("training", 0.25, False, True, 24), ("test", 0.1, True, True, 2)]
    else:
        dataloader_args = [("training", None, True, True, 4), ("test", None, True, False, 1)]
    dataloaders = {
        x: DataLoader(
            BagOfWordsDataset(
                f"{datadir}/bagofwords/{x}", rank, world_size, r, u, s, batch_size
            ),
            batch_size=1,
            drop_last=False,
            num_workers=w,
            persistent_workers=True,
            collate_fn=collate,
        )
        for (x, r, u, s, w) in dataloader_args
    }
    model = BagOfWordsModel(datadir, medium, metric)
    if finetune is not None:
        model.load_state_dict(torch.load(finetune, weights_only=True))
    model = model.to(rank)
    model = torch.compile(model)
    model = DDP(model, device_ids=[rank], output_device=rank)
    optimizer = create_optimizer(model)
    scaler = torch.amp.GradScaler(rank)
    stopper = EarlyStopper(patience=5, rtol=1e-3)
    get_loss = lambda x: evaluate_metrics(
        rank, x, baselines, dataloaders["test"], metric
    )
    checkpoint = lambda m, s, l, e: checkpoint_model(rank, m, s, l, e)
    initial_loss = get_loss(model)
    logger.info(f"Initial Loss: {initial_loss}")
    stopper(initial_loss)
    checkpoint(model, True, initial_loss, -1)
    for epoch in range(64):
        training_loss = train_epoch(
            rank, model, baselines, dataloaders["training"], optimizer, scaler
        )
        logger.info(f"Epoch: {epoch}, Training Loss: {training_loss}")
        test_loss = get_loss(model)
        logger.info(f"Epoch: {epoch}, Test Loss: {test_loss}")
        stopper(test_loss)
        checkpoint(model, stopper.save_model, test_loss, epoch)
        if stopper.early_stop:
            break
    destroy_process_group()
    upload(rank, logger)


def download():
    lock = filelock.FileLock(f"{datadir}/bagofwords.lock")
    with lock:
        if os.path.exists(f"{datadir}/bagofwords"):
            return
        templatefn = f"{datadir}/../../environment/database/download.txt"
        with open(templatefn) as f:
            template = f.read()
        files = (
            ["bagofwords"] +
            [f"{m}.csv" for m in ["manga", "anime"]] +
            [f"baseline.{m}.msgpack" for m in [0, 1]] +
            [f"bagofwords.{m}.{metric}.pt" for m in [0, 1] for metric in ["rating", "watch", "plantowatch", "drop"]]
        )
        for data in files:
            os.system(f"{template}/{data} {datadir}/{data}")


def get_baselines(rank):
    baselines = {}
    for m in [0, 1]:
        with open(f"{datadir}/baseline.{m}.msgpack", "rb") as f:
            baseline = msgpack.unpackb(f.read(), strict_map_key=False)
            d = {}
            d["params"] = baseline["params"]["λ"]
            d["weight"] = baseline["weight"][0]
            d["a"] = torch.tensor(baseline["params"]["a"]).to(rank)
            item_counts = baseline["params"]["item_counts"]
            item_counts = [
                item_counts.get(x, 1) for x in range(len(baseline["params"]["a"]))
            ]
            d["item_counts"] = torch.tensor(item_counts).to(rank)
            baselines[m] = d
    return baselines


parser = argparse.ArgumentParser()
parser.add_argument("--datadir", type=str)
parser.add_argument("--medium", type=int)
parser.add_argument("--metric", type=str)
parser.add_argument("--finetune", type=str, default=None)
args = parser.parse_args()
datadir = args.datadir
medium = args.medium
metric = args.metric
finetune = args.finetune

if __name__ == "__main__":
    download()
    train()
