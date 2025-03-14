import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import argparse
import logging

import h5py
import hdf5plugin
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

exec(open("./Ranking.py").read())


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


class RankingDataset(Dataset):
    def __init__(self, file):
        f = h5py.File(file, "r")
        self.F = np.array(f["features"])
        self.P = np.array(f["prios"])
        f.close()

    def __len__(self):
        return self.F.shape[0]

    def __getitem__(self, i):
        return self.F[i, :, :], self.P[i, :, :]


def get_dataloader(file, batch_size, num_workers):
    dataloader = DataLoader(
        RankingDataset(file),
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        drop_last=False,
        persistent_workers=True,
        pin_memory=True,
    )
    return dataloader


def to_device(data, device):
    return [x.to(device) for x in data]


def evaluate_metrics(model, dataloader, device):
    losses = 0.0
    weights = 0
    model.eval()
    progress = tqdm(desc="Test batches", total=len(dataloader), mininterval=1)
    for data in dataloader:
        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                losses += model(*to_device(data, device))
                weights += 1
        progress.update()
    progress.close()
    model.train()
    return losses / weights


def train_epoch(model, dataloader, optimizer, scaler, device):
    losses = 0.0
    weights = 0
    progress = tqdm(desc="Training batches", total=len(dataloader), mininterval=1)
    for data in dataloader:
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            loss = model(*to_device(data, device))
            weights += 1
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        losses += float(loss)
        progress.update()
    progress.close()
    return losses / weights


def create_optimizer(model, learning_rate, weight_decay):
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


def save_model(model, outdir):
    checkpoint = os.path.join(outdir, "model.pt")
    torch.save(model.state_dict(), checkpoint + "~")
    os.rename(checkpoint + "~", checkpoint)


def train(name, interaction_weights):
    outdir = get_data_path(os.path.join("alphas", name))
    logger = get_logger(outdir)
    batch_size = 16
    device = "cuda"
    learning_rate = 1e-3
    weight_decay = 1e-2

    m = RankingModel(interaction_weights).to(device)
    m = torch.compile(m)
    optimizer = create_optimizer(m, learning_rate, weight_decay)
    scaler = torch.cuda.amp.GradScaler()

    train_dl = get_dataloader(
        os.path.join(outdir, "data", "training.h5"), batch_size, 1
    )
    test_dl = get_dataloader(os.path.join(outdir, "data", "test.h5"), batch_size, 1)

    stopper = EarlyStopper(patience=5, rtol=0)
    initial_loss = evaluate_metrics(m, test_dl, device)
    logger.info(f"Initial Loss: {initial_loss}")
    stopper(initial_loss)
    save_model(m, outdir)

    for epoch in range(64):
        training_loss = train_epoch(m, train_dl, optimizer, scaler, device)
        logger.info(f"Epoch: {epoch}, Training Loss: {training_loss}")
        validation_loss = evaluate_metrics(m, test_dl, device)
        logger.info(f"Epoch: {epoch}, Validation Loss: {validation_loss}")
        stopper(validation_loss)
        if stopper.save_model:
            save_model(m, outdir)
        if stopper.early_stop:
            break


# Main
parser = argparse.ArgumentParser(description="Pytorch")
parser.add_argument("--outdir", type=str, help="name of the data directory")
parser.add_argument(
    "--interaction_weights", nargs=3, help="list of interaction weights", type=float
)
args = parser.parse_args()
if __name__ == "__main__":
    train(args.outdir, args.interaction_weights)