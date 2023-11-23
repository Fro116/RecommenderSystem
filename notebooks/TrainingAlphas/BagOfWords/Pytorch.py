#!/usr/bin/env python
# coding: utf-8

# prevent multithreading deadlocks
import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import argparse
import glob
import json
import logging
import math
import random
import shutil
import time

import h5py
import numpy as np
import scipy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm


# Logging
def get_logger(outdir):
    logger = logging.getLogger(f"pytorch")
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        "%(name)s:%(levelname)s:%(asctime)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    version = 0
    filename = os.path.join(outdir, f"pytorch.log")
    while os.path.exists(filename):
        version += 1
        filename = os.path.join(outdir, f"pytorch.log.{version}")

    streams = [logging.FileHandler(filename, "a"), logging.StreamHandler()]
    for stream in streams:
        stream.setFormatter(formatter)
        logger.addHandler(stream)
    return logger


# Shared
exec(open("./BagOfWords.py").read())


# Data
class BagOfWordsDataset(Dataset):
    def __init__(self, file, basename, shuffle):
        self.filename = file
        f = h5py.File(os.path.join(self.filename, basename), "r")
        self.length = np.array(f[f"epoch_size"]).item()
        self.shuffle = shuffle
        f.close()

    def load(self, basename):
        f = h5py.File(os.path.join(self.filename, basename), "r")
        self.inputs = self.process_sparse_matrix(f, "inputs")
        self.labels = self.process_sparse_matrix(f, "labels")
        self.weights = self.process_sparse_matrix(f, "weights")
        self.users = f["users"][:]
        valid_users = set(f["valid_users"][:])
        valid_indices = []
        for i in range(len(self.users)):
            if self.users[i] in valid_users:
                valid_indices.append(i)
        self.valid_indices = valid_indices
        self.index = len(self.valid_indices)
        f.close()

    def process_sparse_matrix(self, f, name):
        i = f[f"{name}_i"][:] - 1
        j = f[f"{name}_j"][:] - 1
        v = f[f"{name}_v"][:]
        m, n = f[f"{name}_size"][:]
        return scipy.sparse.coo_matrix((v, (j, i)), shape=(n, m)).tocsr()

    def __len__(self):
        return self.length

    def __getitem__(self, _):
        if self.index >= len(self.valid_indices):
            self.index = 0
            if self.shuffle:
                random.shuffle(self.valid_indices)
        i = self.valid_indices[self.index]
        self.index += 1
        X = self.inputs[i, :]
        Y = self.labels[i, :]
        W = self.weights[i, :]
        user = self.users[i]
        return X, Y, W, user


def to_sparse_tensor(csr):
    return torch.sparse_csr_tensor(csr.indptr, csr.indices, csr.data, csr.shape)


def get_device():
    return "cuda"


def to_device(data, device):
    return [to_sparse_tensor(x).to(device).to_dense() for x in data[:-1]]


def sparse_collate(X):
    return [scipy.sparse.vstack([x[i] for x in X]) for i in range(len(X[0]))]


def dataloader_init_fn(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    worker_info.dataset.load(f"data.{worker_info.id+1}.h5")


def get_dataloader(
    outdir,
    split,
    batch_size,
    num_workers,
    shuffle,
):
    dataset = BagOfWordsDataset(os.path.join(outdir, split), "data.1.h5", shuffle)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        drop_last=False,
        persistent_workers=True,
        pin_memory=True,
        collate_fn=sparse_collate,
        worker_init_fn=dataloader_init_fn,
    )
    return dataloader


# Training
def create_optimizer(model, config):
    decay_parameters = []
    no_decay_parameters = []
    for name, param in model.named_parameters():
        if name.startswith("embed") or "norm" in name or "bias" in name:
            no_decay_parameters.append(param)
        else:
            decay_parameters.append(param)
    return optim.AdamW(
        [
            {"params": decay_parameters, "weight_decay": config["weight_decay"]},
            {"params": no_decay_parameters, "weight_decay": 0.0},
        ],
        lr=config["peak_learning_rate"],
        betas=(0.9, 0.999),
    )


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


def train_epoch(outdir, model, dataloader, config, optimizer, scaler, device):
    training_loss = 0.0
    training_weights = 0
    progress = tqdm(
        desc=f"Batches",
        total=len(dataloader),
        mininterval=1,
    )
    for data in dataloader:
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            dev_data = to_device(data, device)
            loss = model(
                *dev_data,
                mask=config["mask"],
                evaluate=False,
                predict=False,
            )
            training_weights += 1
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        training_loss += float(loss)
        progress.update()
    progress.close()
    return training_loss / training_weights


def minimize_quadratic(x, y):
    c = y[1]
    a = ((y[0] - y[1]) + (y[2] - y[1])) / 2
    b = ((y[0] - y[1]) - (y[2] - y[1])) / 2
    miny = c - (b * b) / (4 * a)
    minx = -b / (2 * a)
    return (miny, minx)


def evaluate_metrics(outdir, model, dataloader, config, device):
    losses = [0.0, 0.0, 0.0] if config["metric"] == "rating" else 0.0
    weights = 0.0
    progress = tqdm(
        desc=f"Batches",
        total=len(dataloader),
        mininterval=1,
    )
    model.eval()
    for data in dataloader:
        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                dev_data = to_device(data, device)
                loss = model(*dev_data, mask=False, evaluate=True, predict=False)
                if config["metric"] == "rating":
                    for i in range(len(losses)):
                        losses[i] += float(loss[i])
                else:
                    losses += float(loss)
                weights += float(dev_data[-1].sum())
        progress.update()
    progress.close()
    model.train()
    if config["metric"] == "rating":
        miny, minx = minimize_quadratic([1, 0, -1], losses)
        return (miny / weights, minx)
    else:
        return (losses / weights,)


def save_model(model, outdir):
    checkpoint = os.path.join(outdir, "model.pt")
    torch.save(model.state_dict(), checkpoint + "~")
    os.rename(checkpoint + "~", checkpoint)


def create_model(config, outdir, device):
    model = BagOfWordsModel(config)
    model.load_state_dict(load_model(outdir))
    model = model.to(device)
    model = torch.compile(model)
    return model


def train(config, outdir, logger, training, test):
    logger.info(f"Training on {training} data. Early stopping on {test} data")
    dataloaders = {
        x: get_dataloader(
            outdir,
            y,
            config["batch_size"],
            num_workers=config["num_data_shards"],
            shuffle=x == "training",
        )
        for (x, y) in zip(["training", "validation"], [training, test])
    }

    logger.info(f"Initializing model")
    device = get_device()
    model = create_model(config, outdir, device)

    starting_epoch = 0
    optimizer = create_optimizer(model, config)
    scaler = torch.cuda.amp.GradScaler()
    stopper = EarlyStopper(patience=5, rtol=0)
    initial_loss = evaluate_metrics(
        outdir, model, dataloaders["validation"], config, device
    )
    logger.info(f"Initial Loss: {initial_loss}")
    stopper(initial_loss[0])

    for epoch in range(starting_epoch, 100):  # TODO  decay on plateau
        training_loss = train_epoch(
            outdir,
            model,
            dataloaders["training"],
            config,
            optimizer,
            scaler,
            device,
        )
        logger.info(f"Epoch: {epoch}, Training Loss: {training_loss}")
        validation_loss = evaluate_metrics(
            outdir, model, dataloaders["validation"], config, device
        )
        logger.info(f"Epoch: {epoch}, Validation Loss: {validation_loss}")
        stopper(validation_loss[0])
        if stopper.save_model:
            save_model(model, outdir)
        if stopper.early_stop:
            break


def record_predictions(model, outdir, dataloader):
    user_batches = []
    embed_batches = []
    progress = tqdm(
        desc=f"Batches",
        total=len(dataloader),
        mininterval=1,
    )
    model.eval()
    device = get_device()
    for data in dataloader:
        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                x = (
                    model(
                        *to_device(data, device),
                        mask=False,
                        evaluate=False,
                        predict=True,
                    )
                    .to("cpu")
                    .to(torch.float32)
                    .numpy()
                )
                users = data[-1].todense()
                user_batches.append(users)
                embed_batches.append(x)
                progress.update(1)
    progress.close()
    model.train()

    f = h5py.File(os.path.join(outdir, "predictions.h5"), "w")
    f.create_dataset("users", data=np.vstack(user_batches))
    f.create_dataset("predictions", data=np.vstack(embed_batches))
    f.close()


def run_process(name, mode):
    outdir = get_data_path(os.path.join("alphas", name))
    logger = get_logger(outdir)
    config_file = os.path.join(outdir, "config.json")
    config = create_training_config(config_file, mode)
    torch.set_float32_matmul_precision("high")
    if mode == "pretrain":
        save_model(BagOfWordsModel(config), outdir)
        train(config, outdir, logger, "training", "validation")
    elif mode == "finetune":
        train(config, outdir, logger, "validation", "test")
    elif mode == "inference":
        model = create_model(config, outdir, get_device())
        dataloader = get_dataloader(
            outdir,
            "inference",
            config["batch_size"],
            num_workers=1,
            shuffle=False,
        )
        record_predictions(model, outdir, dataloader)
    else:
        assert False


# Main
parser = argparse.ArgumentParser(description="Pytorch")
parser.add_argument("--outdir", type=str, help="name of the data directory")
parser.add_argument(
    "--mode",
    type=str,
    choices=["pretrain", "finetune", "inference"],
    help="training strategy",
)
args = parser.parse_args()
if __name__ == "__main__":
    run_process(args.outdir, args.mode)