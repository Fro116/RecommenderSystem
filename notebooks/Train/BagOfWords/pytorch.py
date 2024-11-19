#!/usr/bin/env python
# coding: utf-8

import argparse
import itertools
import json
import logging
import os
import warnings

import h5py
import numpy as np
import scipy
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

exec(open("./bagofwords.py").read())


def get_data_path(file):
    path = os.getcwd()
    while os.path.basename(path) != "notebooks":
        path = os.path.dirname(path)
    path = os.path.dirname(path)
    return os.path.join(path, "data", file)


def get_logger(outdir):
    logger = logging.getLogger("pytorch")
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        "%(name)s:%(levelname)s:%(asctime)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    version = 0
    filename = os.path.join(outdir, "pytorch.log")
    while os.path.exists(filename):
        version += 1
        filename = os.path.join(outdir, f"pytorch.log.{version}")

    streams = [logging.FileHandler(filename, "a"), logging.StreamHandler()]
    for stream in streams:
        stream.setFormatter(formatter)
        logger.addHandler(stream)
    return logger


class BagOfWordsDataset(Dataset):
    def __init__(self, fn):
        def load_sparse_matrix(f, name):
            i = f[f"{name}_i"][:] - 1
            j = f[f"{name}_j"][:] - 1
            v = f[f"{name}_v"][:]
            m, n = f[f"{name}_size"][:]
            return scipy.sparse.coo_matrix((v, (j, i)), shape=(n, m)).tocsr()
            
        with h5py.File(fn, "r") as f:
            self.users = f["users"][:]
            self.inputs = load_sparse_matrix(f, "inputs")
            self.labels = load_sparse_matrix(f, "labels")
            self.weights = load_sparse_matrix(f, "weights")

    def __len__(self):
        return len(self.users)

    def __getitem__(self, i):
        X = self.inputs[i, :]
        Y = self.labels[i, :]
        W = self.weights[i, :]
        return X, Y, W, self.users[i]


def to_sparse_tensor(csr):
    return torch.sparse_csr_tensor(csr.indptr, csr.indices, csr.data, csr.shape)


def get_device():
    return "cuda"


def to_device(data, device):
    return [to_sparse_tensor(x).to(device).to_dense() for x in data[:-1]]


def collate(data):
    X = scipy.sparse.vstack([x[0] for x in data])
    Y = scipy.sparse.vstack([x[1] for x in data])
    W = scipy.sparse.vstack([x[2] for x in data])
    users = [x[3] for x in data]
    return [X, Y, W, users]


def get_dataloader(file, batch_size, shuffle):
    dataset = BagOfWordsDataset(file)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=4,
        shuffle=shuffle,
        drop_last=False,
        persistent_workers=True,
        pin_memory=True,
        collate_fn=collate,
    )
    return dataloader


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
        lr=config["learning_rate"],
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


def train_epoch(model, dataloader, mask, optimizer, scaler):
    training_loss = 0.0
    weights = 0
    device = get_device()
    progress = tqdm(desc="Training batches", total=len(dataloader), mininterval=1)
    for data in dataloader:
        optimizer.zero_grad(set_to_none=True)
        dev_data = to_device(data, device)
        with torch.amp.autocast(device, dtype=torch.bfloat16):
            loss, wsum = model(
                *dev_data,
                mask=mask,
                mode="training",
            )
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        training_loss += float(loss)
        weights += float(wsum)
        progress.update()
    progress.close()
    return training_loss / weights


def minimize_quadratic(x, y):
    assert len(x) == 3 and len(y) == 3
    A = np.array([[x[0] ** 2, x[0], 1], [x[1] ** 2, x[1], 1], [x[2] ** 2, x[2], 1]])
    B = np.array(y)
    a, b, c = np.linalg.solve(A, B)
    x_extremum = -b / (2 * a)
    y_extremum = a * x_extremum**2 + b * x_extremum + c
    return x_extremum, y_extremum


def evaluate_metrics(model, dataloader, mask):
    losses = [0.0, 0.0, 0.0] if model.metric == "rating" else 0.0
    weights = 0.0
    device = get_device()
    progress = tqdm(desc="Test batches", total=len(dataloader), mininterval=1)
    model.eval()
    for data in dataloader:
        with torch.no_grad():
            dev_data = to_device(data, device)
            with torch.amp.autocast(device, dtype=torch.bfloat16):
                loss, wsum = model(*dev_data, mask=mask, mode="evaluation")
            if model.metric == "rating":
                for i in range(len(losses)):
                    losses[i] += float(loss[i])
            else:
                losses += float(loss)
            weights += float(wsum)
        progress.update()
    progress.close()
    model.train()
    if model.metric == "rating":
        _, miny = minimize_quadratic([1, 0, -1], losses)
        return miny / weights
    else:
        return losses / weights


def save_model(model, outdir):
    fn = os.path.join(outdir, "model.pt")
    temp = fn + "~"
    torch.save(model._orig_mod.state_dict(), temp)
    os.rename(temp, fn)
    return fn


def create_model(config, init):
    model = BagOfWordsModel(
        config["input_sizes"],
        config["output_index"],
        config["metric"],
    )
    if init is not None:
        model.load_state_dict(torch.load(init, weights_only=True))
    model = model.to(get_device())
    model = torch.compile(model)
    return model


def train(config, init, outdir):
    logger = get_logger(outdir)
    logger.info(f"Training model {config}")
    model = create_model(config, init)
    dataloaders = {
        x: get_dataloader(
            f"{outdir}/{x}.h5",
            config["batch_size"],
            shuffle=x == "train",
        )
        for x in ["train", "test"]
    }
    starting_epoch = 0
    optimizer = create_optimizer(model, config)
    scaler = torch.amp.GradScaler(get_device())
    stopper = EarlyStopper(patience=5, rtol=1e-4)
    get_loss = lambda x: evaluate_metrics(x, dataloaders["test"], config["mask_rate"])
    initial_loss = get_loss(model)
    logger.info(f"Initial Loss: {initial_loss}")
    stopper(initial_loss)
    save_model(model, outdir)

    for epoch in range(starting_epoch, 100):
        training_loss = train_epoch(
            model,
            dataloaders["train"],
            config["mask_rate"],
            optimizer,
            scaler,
        )
        logger.info(f"Epoch: {epoch}, Training Loss: {training_loss}")
        test_loss = get_loss(model)
        logger.info(f"Epoch: {epoch}, Test Loss: {test_loss}")
        stopper(test_loss)
        if stopper.save_model:
            save_model(model, outdir)
        if stopper.early_stop:
            break


def predict(config, init, file):
    model = create_model(config, init)
    model.eval()
    dataloader = get_dataloader(
        file,
        config["batch_size"],
        shuffle=False,
    )

    user_batches = []
    output_batches = []
    device = get_device()
    found_users = set()
    while len(found_users) < len(dataloader.dataset):
        # so we don't miss any users from having num_dataloader_workers > 1
        progress = tqdm(desc=f"Batches", total=len(dataloader), mininterval=1)
        for data in dataloader:
            with torch.no_grad():
                dev_data = to_device(data, device)
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    x = (
                        model(
                            *dev_data,
                            mask=False,
                            mode="inference",
                        )
                        .to("cpu")
                        .to(torch.float32)
                        .numpy()
                    )
                users = data[-1]
                found_users |= set(users)
                user_batches.append(users)
                output_batches.append(x)
                progress.update(1)
        progress.close()

    with h5py.File(f"{file}.out", "w") as f:
        f.create_dataset("users", data=list(itertools.chain(*user_batches)))
        f.create_dataset("predictions", data=np.vstack(output_batches))


def create_training_config(config_file, init):
    config = json.load(open(config_file, "r"))
    return {
        # model
        "input_sizes": config["input_sizes"],
        "output_index": config["output_index"] - 1,
        "metric": config["metric"],
        # training
        "learning_rate": 4e-5 if init is None else 4e-6,
        "weight_decay": 1e-2,
        "mask_rate": 0.25,
        "batch_size": 2048,
    }


def run_process(outdir, init, predict_file):
    torch.set_float32_matmul_precision("high")
    warnings.filterwarnings("ignore", ".*Sparse CSR tensor support is in beta state.*")
    config_file = os.path.join(outdir, "config.json")
    config = create_training_config(config_file, init)
    if predict_file is not None:
        predict(config, init, predict_file)
    else:
        train(config, init, outdir)


# Main
parser = argparse.ArgumentParser(description="Train BagOfWords models")
parser.add_argument("--outdir", type=str, help="name of the data directory")
parser.add_argument("--init", type=str, help="path of pretrained model")
parser.add_argument("--predict", type=str, help="file to perform inference on")
args = parser.parse_args()
if __name__ == "__main__":
    run_process(args.outdir, args.init, args.predict)