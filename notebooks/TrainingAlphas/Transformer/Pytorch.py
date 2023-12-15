#!/usr/bin/env python
# coding: utf-8

# prevent multithreading deadlocks
import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import argparse
import glob
import logging
import math
import shutil
import time

import h5py
import numpy as np
import scipy
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from tqdm.auto import tqdm

exec(open("./Transformer.py").read())

# datasets


class PretrainDataset(Dataset):
    def __init__(self, file):
        self.filename = file
        f = h5py.File(file, "r")

        self.length = f["itemid"].shape[0]
        self.embeddings = [
            f["itemid"][:],
            f["rating"][:].reshape(*f["rating"].shape, 1).astype(np.float32),
            f["updated_at"][:].reshape(*f["updated_at"].shape, 1).astype(np.float32),
            f["status"][:],
            f["position"][:],
        ]
        self.mask = f["userid"][:]

        def process_position(x):
            x = x[:].astype(np.int64)
            return x.reshape(*x.shape, 1)

        self.positions = [
            process_position(f[f"positions_{medium}_{metric}"])
            for medium in ALL_MEDIUMS
            for metric in ALL_METRICS
        ]
        self.labels = [
            np.expand_dims(f[f"labels_{medium}_{metric}"][:], axis=-1)
            for medium in ALL_MEDIUMS
            for metric in ALL_METRICS
        ]
        self.weights = [
            np.expand_dims(f[f"weights_{medium}_{metric}"][:], axis=-1)
            for medium in ALL_MEDIUMS
            for metric in ALL_METRICS
        ]
        f.close()

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        # a true value means that the tokens will not attend to each other
        mask = self.mask[i, :]
        mask = mask.reshape(1, mask.size) != mask.reshape(mask.size, 1)
        embeds = [x[i, :] for x in self.embeddings]
        positions = [x[i, :] for x in self.positions]
        labels = [x[i, :] for x in self.labels]
        weights = [x[i, :] for x in self.weights]
        return embeds, mask, positions, labels, weights


class FinetuneDataset(Dataset):
    def __init__(self, file):
        self.filename = file
        f = h5py.File(file, "r")
        self.length = f["itemid"].shape[0]
        self.embeddings = [
            f["itemid"][:],
            f["rating"][:].reshape(*f["rating"].shape, 1).astype(np.float32),
            f["updated_at"][:].reshape(*f["updated_at"].shape, 1).astype(np.float32),
            f["status"][:],
            f["position"][:],
        ]
        self.mask = f["userid"][:]

        def process_position(x):
            return x[:].flatten().astype(np.int64)

        def process_sparse_matrix(f, name):
            i = f[f"{name}_i"][:]
            j = f[f"{name}_j"][:]
            v = f[f"{name}_v"][:]
            m, n = f[f"{name}_size"][:]
            return scipy.sparse.coo_matrix((v, (j, i)), shape=(n, m)).tocsr()

        self.positions = process_position(f["positions"])
        self.labels = [
            process_sparse_matrix(f, f"labels_{medium}_{metric}")
            for medium in ALL_MEDIUMS
            for metric in ALL_METRICS
        ]
        self.weights = [
            process_sparse_matrix(f, f"weights_{medium}_{metric}")
            for medium in ALL_MEDIUMS
            for metric in ALL_METRICS
        ]
        f.close()

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        # a true value means that the tokens will not attend to each other
        mask = self.mask[i, :]
        mask = mask.reshape(1, mask.size) != mask.reshape(mask.size, 1)
        user = self.mask[i, 0]
        embeds = [x[i, :] for x in self.embeddings]
        positions = self.positions[i]
        labels = [x[i, :].toarray().flatten() for x in self.labels]
        weights = [x[i, :].toarray().flatten() for x in self.weights]
        return embeds, mask, positions, labels, weights, user


class StreamingDataset(Dataset):
    def __init__(self, outdir, epoch_size, chunk_size, mode, epochs):
        self.outdir = outdir
        self.epoch_size = epoch_size
        self.chunk_size = chunk_size
        self.num_epochs = epochs
        self.mode = mode
        self.epoch = 0
        self.chunk = -1
        self.dataset = None
        self.last_index = -1

    def __len__(self):
        return self.epoch_size

    def __getitem__(self, i):
        if i < self.last_index:
            self.epoch += 1
            if self.epoch == self.num_epochs:
                self.epoch = 0
            self.chunk = -1
        self.last_index = i

        chunk, index = divmod(i, self.chunk_size)
        if self.chunk != chunk:
            self.chunk = chunk
            file = os.path.join(self.outdir, str(self.epoch), f"{chunk}.h5")
            if not os.path.exists(file):
                time.sleep(1)
            if self.mode == "pretrain":
                self.dataset = PretrainDataset(file)
            elif self.mode == "finetune":
                self.dataset = FinetuneDataset(file)
            else:
                assert False
        return self.dataset[index]


# training


def to_device(data, device):
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device)


class EarlyStopper:
    def __init__(self, patience, rtol):
        self.patience = patience
        self.rtol = rtol
        self.reset()

    def reset(self):
        self.counter = 0
        self.best_score = float("inf")
        self.early_stop = False
        self.save_model = False

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


def create_learning_rate_schedule(optimizer, config, world_size):
    # TODO try cosine annealing
    if config["mode"] == "pretrain":
        steps_per_epoch = math.ceil(
            config["training_epoch_size"] / (config["training_batch_size"] * world_size)
        )
        total_steps = config["num_epochs"] * steps_per_epoch
        warmup_ratio = config["warmup_ratio"]
        warmup_steps = math.ceil(total_steps * warmup_ratio)
        warmup_lambda = (
            lambda x: x / warmup_steps
            if x < warmup_steps
            else max(0, 1 - (x - warmup_steps) / (total_steps - warmup_steps))
        )
        return optim.lr_scheduler.LambdaLR(optimizer, warmup_lambda)
    elif config["mode"] == "finetune":
        return None
    else:
        assert False


def wsum(values, weights):
    return sum(x * y for (x, y) in zip(values, weights))


def make_task_weights(losses):
    metric_weight = {
        "rating": 1,
        "watch": 1,
        "plantowatch": 1,
        "drop": 1,
    }
    task_weights = [metric_weight[y] for _ in ALL_MEDIUMS for y in ALL_METRICS]
    # rescale tasks so they contribute equally to loss
    weights = []
    for i in range(len(losses)):
        if losses[i] > 0:
            weights.append(1 / losses[i])
        else:
            weights.append(0)
    return [x * w for (x, w) in zip(weights, task_weights)]


def reduce_mean(rank, x, w):
    x = torch.tensor(x).to(rank)
    w = torch.tensor(w).to(rank)
    dist.all_reduce(x, op=dist.ReduceOp.SUM)
    dist.all_reduce(w, op=dist.ReduceOp.SUM)
    return [float(v) / float(w[0]) for v in x]


def train_epoch(
    rank,
    model,
    dataloader,
    optimizer,
    scheduler,
    scaler,
    task_weights,
):
    training_losses = [0.0 for _ in range(len(ALL_MEDIUMS) * len(ALL_METRICS))]
    training_steps = 0
    progress = tqdm(
        desc=f"Batches",
        total=len(dataloader),
        mininterval=1,
        disable=rank != 0,
    )
    for data in dataloader:
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            tloss = model(*to_device(data, rank))
            for i in range(len(tloss)):
                training_losses[i] += float(tloss[i])
            loss = sum(tloss[i] * task_weights[i] for i in range(len(tloss)))
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        if scheduler is not None:
            scheduler.step()
        training_steps += 1
        progress.update()
    progress.close()
    return reduce_mean(rank, training_losses, [training_steps])


def evaluate_metrics(rank, model, dataloader):
    losses = [0.0 for _ in range(len(ALL_MEDIUMS) * len(ALL_METRICS))]
    steps = 0
    progress = tqdm(
        desc=f"Batches",
        total=len(dataloader),
        mininterval=1,
        disable=rank != 0,
    )
    model.eval()
    for data in dataloader:
        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                loss = model(*to_device(data, rank))
            for i in range(len(loss)):
                losses[i] += float(loss[i])
        steps += 1
        progress.update()
    progress.close()
    model.train()
    return reduce_mean(rank, losses, [steps])


def record_predictions(rank, model, outdir, dataloader):
    assert rank == 0

    user_batches = []
    embed_batches = []
    progress = tqdm(
        desc=f"Batches",
        total=len(dataloader),
        mininterval=1,
        disable=rank != 0,
    )
    model.eval()
    for data in dataloader:
        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                x = (
                    model(*to_device(data, rank), embed_only=True)
                    .to("cpu")
                    .to(torch.float32)
                    .numpy()
                )
                users = data[-1].numpy()
                user_batches.append(users)
                embed_batches.append(x)
                progress.update()
    progress.close()
    model.train()

    f = h5py.File(os.path.join(outdir, "embeddings.h5"), "w")
    f.create_dataset("users", data=np.hstack(user_batches))
    f.create_dataset("embedding", data=np.vstack(embed_batches))
    detach = lambda x: x.to("cpu").detach().numpy()
    i = 0
    for medium in ALL_MEDIUMS:
        for metric in ALL_METRICS:
            head = model.classifier[i]
            if metric in ["watch", "plantowatch"]:
                head = head[0]
            f.create_dataset(f"{medium}_{metric}_weight", data=detach(head.weight))
            f.create_dataset(f"{medium}_{metric}_bias", data=detach(head.bias))
            i += 1
    f.close()


# checkpoints


def get_logger(outdir, rank):
    # writes to file and to stdout if rank==0
    logger = logging.getLogger(f"pytorch.{rank}")
    if rank != 0:
        return logger
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


def initialize_model(model, name, logger):
    path = get_data_path(os.path.join("alphas", name, "model.pt"))
    logger.info(f"loading model from {path}")
    model.load_state_dict(load_model(path))


def save_model(rank, model, outdir):
    if rank != 0:
        return
    fn = os.path.join(outdir, "model.pt")
    torch.save(model.module.state_dict(), fn)


# multiprocessing


def setup_multiprocessing(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def get_dataloader(
    rank,
    world_size,
    outdir,
    mode,
    epoch_size,
    chunk_size,
    batch_size,
    epochs,
):
    dataset = StreamingDataset(
        outdir,
        epoch_size,
        chunk_size,
        mode,
        epochs,
    )
    sampler = DistributedSampler(
        dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=4,
        sampler=sampler,
        drop_last=False,
        shuffle=False,
        persistent_workers=True,
        pin_memory=True,
    )
    return dataloader


def run_process(rank, world_size, name, model_init):
    setup_multiprocessing(rank, world_size)
    outdir = get_data_path(os.path.join("alphas", name))
    logger = get_logger(outdir, rank)
    config_file = os.path.join(outdir, "config.json")
    training_config = create_training_config(config_file)
    model_config = create_model_config(training_config)
    torch.cuda.set_device(rank)
    torch.set_float32_matmul_precision("high")

    dataloaders = {
        x: get_dataloader(
            rank,
            world_size,
            os.path.join(outdir, x),
            training_config["mode"],
            training_config[f"{x}_epoch_size"],
            training_config[f"chunk_size"],
            training_config[f"{x}_batch_size"],
            training_config["num_epochs"],
        )
        for x in ["training", "validation"]
    }
    model = TransformerModel(model_config).to(rank)
    if model_init is not None:
        initialize_model(model, model_init, logger)
    if training_config["mode"] == "finetune":
        model = torch.compile(model)
    model = DDP(
        model, device_ids=[rank], output_device=rank, find_unused_parameters=True
    )
    optimizer = create_optimizer(model, training_config)
    scheduler = create_learning_rate_schedule(optimizer, training_config, world_size)
    scaler = torch.cuda.amp.GradScaler()
    stopper = (
        EarlyStopper(patience=1, rtol=0.001)
        if training_config["mode"] == "finetune"
        else None
    )

    initial_loss = evaluate_metrics(rank, model, dataloaders["validation"])
    task_weights = make_task_weights(initial_loss)
    logger.info(
        f"Initial Loss: {sum(initial_loss)},"
        f" {wsum(initial_loss, task_weights)}, {initial_loss}"
    )
    if stopper:
        stopper(wsum(initial_loss, task_weights))

    for epoch in range(training_config["num_epochs"]):
        training_loss = train_epoch(
            rank,
            model,
            dataloaders["training"],
            optimizer,
            scheduler,
            scaler,
            task_weights,
        )
        logger.info(
            f"Epoch: {epoch}, Training Loss: {sum(training_loss)},"
            f" {wsum(training_loss, task_weights)} {training_loss}"
        )
        validation_loss = evaluate_metrics(rank, model, dataloaders["validation"])
        logger.info(
            f"Epoch: {epoch}, Validation Loss: {sum(validation_loss)},"
            f" {wsum(validation_loss, task_weights)}, {validation_loss}"
        )
        if stopper:
            stopper(wsum(validation_loss, task_weights))
            if stopper.save_model:
                save_model(rank, model, outdir)
                task_weights = make_task_weights(validation_loss)
                stopper.reset()
                stopper(wsum(validation_loss, task_weights))
            elif stopper.early_stop:
                break
        else:
            save_model(rank, model, outdir)
            task_weights = make_task_weights(validation_loss)

    if training_config["mode"] == "finetune" and rank == 0:
        model = TransformerModel(model_config).to(rank)
        initialize_model(model, name, logger)
        model = torch.compile(model)
        record_predictions(rank, model, outdir, dataloaders["validation"])

    dist.destroy_process_group()


def cleanup_previous_runs(name):
    outdir = get_data_path(os.path.join("alphas", name, "training"))
    for x in os.listdir(outdir):
        if "read" in x:
            os.remove(os.path.join(outdir, x))


parser = argparse.ArgumentParser(description="PytorchPretrain")
parser.add_argument("--outdir", type=str, help="name of the data directory")
parser.add_argument(
    "--initialize", type=str, help="initialize training from a model checkpoint"
)
parser.add_argument("--gpus", type=int, help="number of gpus to use")
args = parser.parse_args()
if __name__ == "__main__":
    gpus = torch.cuda.device_count() if args.gpus is None else args.gpus
    cleanup_previous_runs(args.outdir)
    mp.spawn(
        run_process,
        args=(gpus, args.outdir, args.initialize),
        nprocs=gpus,
    )