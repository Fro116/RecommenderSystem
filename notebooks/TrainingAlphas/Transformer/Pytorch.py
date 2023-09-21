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
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from tqdm.auto import tqdm


# Logging
def get_logger(outdir, rank):
    logger = logging.getLogger(f"pretrain.{rank}")
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        "%(name)s:%(levelname)s:%(asctime)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    version = 0
    filename = os.path.join(outdir, f"pretrain.{rank}.log")
    while os.path.exists(filename):
        version += 1
        filename = os.path.join(outdir, f"pretrain.{rank}.log.{version}")

    streams = [logging.FileHandler(filename, "w")]
    if rank == 0:
        streams.append(logging.StreamHandler())
    for stream in streams:
        stream.setFormatter(formatter)
        logger.addHandler(stream)
    return logger


# Shared
exec(open("./Transformer.py").read())


# Data
class PretrainDataset(Dataset):
    def __init__(self, file):
        self.filename = file
        f = h5py.File(file, "r")
        self.length = f["anime"].shape[0]
        self.embeddings = [
            f["anime"][:] - 1,
            f["manga"][:] - 1,
            f["rating"][:].reshape(*f["rating"].shape, 1).astype(np.float32),
            f["timestamp"][:].reshape(*f["timestamp"].shape, 1).astype(np.float32),
            f["status"][:] - 1,
            f["completion"][:].reshape(*f["completion"].shape, 1).astype(np.float32),
            f["position"][:] - 1,
        ]
        self.mask = f["user"][:]

        def process_position(x):
            x = x[:].astype(np.int64) - 1
            return x.reshape(*x.shape, 1)

        self.positions = [
            process_position(f[f"positions_{medium}_{task}"])
            for medium in ["anime", "manga"]
            for task in ["item", "rating"]
        ]
        self.labels = [
            np.expand_dims(f[f"labels_{medium}_{task}"][:], axis=-1)
            for medium in ["anime", "manga"]
            for task in ["item", "rating"]
        ]
        self.weights = [
            np.expand_dims(f[f"weights_{medium}_{task}"][:], axis=-1)
            for medium in ["anime", "manga"]
            for task in ["item", "rating"]
        ]

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        embeds = [x[i, :] for x in self.embeddings]

        # a true value means that the tokens will not attend to each other
        mask = self.mask[i, :]
        mask = mask.reshape(1, mask.size) != mask.reshape(mask.size, 1)

        positions = [x[i, :] for x in self.positions]
        labels = [x[i, :] for x in self.labels]
        weights = [x[i, :] for x in self.weights]
        return embeds, mask, positions, labels, weights


class FinetuneDataset(Dataset):
    def __init__(self, file):
        self.filename = file
        f = h5py.File(file, "r")
        self.length = f["anime"].shape[0]
        self.embeddings = [
            f["anime"][:] - 1,
            f["manga"][:] - 1,
            f["rating"][:].reshape(*f["rating"].shape, 1).astype(np.float32),
            f["timestamp"][:].reshape(*f["timestamp"].shape, 1).astype(np.float32),
            f["status"][:] - 1,
            f["completion"][:].reshape(*f["completion"].shape, 1).astype(np.float32),
            f["position"][:] - 1,
        ]
        self.mask = f["user"][:]

        def process_position(x):
            return x[:].flatten().astype(np.int64) - 1

        def process_sparse_matrix(f, name):
            i = f[f"{name}_i"][:] - 1
            j = f[f"{name}_j"][:] - 1
            v = f[f"{name}_v"][:]
            m, n = f[f"{name}_size"][:]
            return scipy.sparse.coo_matrix((v, (j, i)), shape=(n, m)).tocsr()

        self.positions = process_position(f["positions"])
        self.labels = [
            process_sparse_matrix(f, name)
            for name in [
                f"labels_{medium}_{task}"
                for medium in ["anime", "manga"]
                for task in ["item", "rating"]
            ]
        ]
        self.weights = [
            process_sparse_matrix(f, name)
            for name in [
                f"weights_{medium}_{task}"
                for medium in ["anime", "manga"]
                for task in ["item", "rating"]
            ]
        ]

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        embeds = [x[i, :] for x in self.embeddings]

        # a true value means that the tokens will not attend to each other
        mask = self.mask[i, :]
        mask = mask.reshape(1, mask.size) != mask.reshape(mask.size, 1)

        user = self.mask[i, 0]

        positions = self.positions[i]
        labels = [x[i, :].toarray().flatten() for x in self.labels]
        weights = [x[i, :].toarray().flatten() for x in self.weights]
        return embeds, mask, positions, labels, weights, user


class StreamingDataset(Dataset):
    def __init__(
        self,
        rank,
        world_size,
        num_workers,
        outdir,
        split,
        batch_size,
        num_shards,
        mode,
        max_size,
    ):
        self.rank = rank
        self.world_size = world_size
        self.outdir = outdir
        self.split = split
        self.batch_size = batch_size
        self.num_shards = num_shards
        self.mode = mode
        chunk_size = batch_size * world_size
        rounding = math.floor if split == "validation" else math.ceil
        self.max_size = chunk_size * rounding(max_size / chunk_size)
        self.num_workers = num_workers
        self.data = [
            {
                "dataset": None,
                "start_index": 0,
                "epoch": 0,
                "shard": 1,
                "prev_item": 0,
            }
            for _ in range(self.num_workers)
        ]

    def advance_stream(self):
        # wait for the data shard to be written
        workerid = torch.utils.data.get_worker_info().id
        basefile = os.path.join(
            self.outdir, "training", f'{self.split}.{self.data[workerid]["shard"]}.h5'
        )
        completion_file = basefile + ".complete"
        num_workers = torch.utils.data.get_worker_info().num_workers
        read_file = (
            basefile + f".read.{self.rank}.{workerid}.{self.world_size}.{num_workers}"
        )
        while os.path.exists(read_file) or not os.path.exists(completion_file):
            time.sleep(1)

        # read the data shard
        data_file = completion_file[: -len(".complete")]
        if self.mode == "pretrain":
            dataset = PretrainDataset(data_file)
        elif self.mode == "finetune":
            dataset = FinetuneDataset(data_file)
        else:
            assert False
        open(read_file, "w").close()

        if self.data[workerid]["dataset"]:
            self.data[workerid]["start_index"] += len(self.data[workerid]["dataset"])
        self.data[workerid]["dataset"] = dataset
        self.data[workerid]["shard"] = (
            1
            if self.data[workerid]["shard"] == self.num_shards
            else self.data[workerid]["shard"] + 1
        )

    def __len__(self):
        return self.max_size

    def __getitem__(self, i):
        workerid = torch.utils.data.get_worker_info().id

        # if we're starting a new epoch
        while (
            i + self.data[workerid]["epoch"] * self.max_size
            < self.data[workerid]["prev_item"]
        ):
            self.data[workerid]["epoch"] += 1
        i += self.data[workerid]["epoch"] * self.max_size
        self.data[workerid]["prev_item"] = i

        # if we're loading a new shard
        while self.data[workerid]["dataset"] is None or i >= self.data[workerid][
            "start_index"
        ] + len(self.data[workerid]["dataset"]):
            self.advance_stream()
        return self.data[workerid]["dataset"][i - self.data[workerid]["start_index"]]


def to_device(data, device):
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device)


def get_temp_path(file):
    return get_data_path(file)


# Training
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
    if config["mode"] == "pretrain":
        steps_per_epoch = math.ceil(
            config["training_epoch_size"] / (config["batch_size"] * world_size)
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


def train_epoch(
    rank, world_size, outdir, model, dataloader, config, optimizer, scheduler, scaler
):
    training_loss = 0.0
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
            loss = sum(model(*to_device(data, rank)))
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        if scheduler is not None:
            scheduler.step()
        training_loss += float(loss)
        training_steps += 1
        progress.update()
    progress.close()

    training_loss = training_loss / training_steps
    tensor_losses = torch.tensor([training_loss]).to(rank)
    dist.all_reduce(tensor_losses, op=dist.ReduceOp.SUM)
    tensor_losses /= world_size
    return [float(x) for x in tensor_losses]


def evaluate_metrics(rank, world_size, outdir, model, dataloader, config):
    losses = [0.0 for _ in range(4)]
    steps = 0
    # since we're not taking gradients, we can use bigger batches
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
                for i in range(len(losses)):
                    losses[i] += float(loss[i])
        steps += 1
        progress.update()
    progress.close()
    model.train()
    for i in range(len(losses)):
        losses[i] /= steps
    tensor_losses = torch.tensor(losses).to(rank)
    dist.all_reduce(tensor_losses, op=dist.ReduceOp.SUM)
    tensor_losses /= world_size
    return [float(x) for x in tensor_losses]


def record_predictions(rank, world_size, model, outdir, dataloader, usertag):
    assert rank == 0

    f = h5py.File(os.path.join(outdir, "training", "users.h5"), "r")
    all_users = f[usertag][:]
    f.close()

    user_batches = []
    embed_batches = []
    seen_users = set()
    progress = tqdm(
        desc=f"Batches",
        total=len(all_users),
        mininterval=1,
        disable=rank != 0,
    )
    model.eval()
    while len(seen_users) < len(all_users):
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
                    new_users = len(seen_users | set(users)) - len(seen_users)
                    seen_users |= set(users)
                    progress.update(new_users)
            if len(seen_users) >= len(all_users):
                break
    progress.close()
    model.train()

    f = h5py.File(os.path.join(outdir, "embeddings.h5"), "w")
    f.create_dataset("users", data=np.hstack(user_batches))
    f.create_dataset("embedding", data=np.vstack(embed_batches))
    detach = lambda x: x.to("cpu").detach().numpy()
    f.create_dataset(
        "anime_item_weight", data=detach(model.classifier[0][0].weight)
    )
    f.create_dataset("anime_item_bias", data=detach(model.classifier[0][0].bias))
    f.create_dataset(
        "anime_rating_weight", data=detach(model.classifier[1].weight)
    )
    f.create_dataset("anime_rating_bias", data=detach(model.classifier[1].bias))
    f.create_dataset(
        "manga_item_weight", data=detach(model.classifier[2][0].weight)
    )
    f.create_dataset("manga_item_bias", data=detach(model.classifier[2][0].bias))
    f.create_dataset(
        "manga_rating_weight", data=detach(model.classifier[3].weight)
    )
    f.create_dataset("manga_rating_bias", data=detach(model.classifier[3].bias))
    f.close()


def load_checkpoints(outdir):
    checkpoint = None
    epoch = -1
    for x in glob.glob(os.path.join(outdir, f"model.*.pt")):
        trial = int(os.path.basename(x).split(".")[1])
        if trial > epoch:
            epoch = trial
            checkpoint = x
    return checkpoint, epoch


def initialize_model(model, model_init, outdir):
    starting_epoch = 0
    if model_init is not None:
        model_fn = get_data_path(os.path.join("alphas", model_init, "model.pt"))
        model.load_state_dict(load_model(model_fn))
    checkpoint, epoch = load_checkpoints(outdir)
    if checkpoint is not None:
        starting_epoch = epoch + 1
        model.load_state_dict(load_model(checkpoint))
    return starting_epoch


def save_model(rank, world_size, model, epoch, outdir):
    # atomically save model
    if rank == 0:
        checkpoint = os.path.join(outdir, f"model.{epoch}.pt")
        torch.save(model.module.state_dict(), checkpoint + "~")
        os.rename(checkpoint + "~", checkpoint)
        previous_checkpoints = glob.glob(os.path.join(outdir, f"model.*.pt"))
        for fn in previous_checkpoints:
            if fn != checkpoint:
                os.remove(fn)


def publish_model(outdir):
    checkpoint, epoch = load_checkpoints(outdir)
    assert checkpoint is not None
    modelfn = os.path.join(outdir, f"model.pt")
    shutil.copyfile(checkpoint, modelfn + "~")
    os.rename(modelfn + "~", modelfn)
    os.remove(checkpoint)


# Distributed Data Parallel
def setup_multiprocessing(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def get_dataloader(
    rank,
    world_size,
    outdir,
    mode,
    split,
    batch_size,
    num_data_shards,
    epoch_size,
    num_workers=0,
):
    dataset = StreamingDataset(
        rank,
        world_size,
        num_workers,
        outdir,
        split,
        batch_size,
        num_data_shards,
        mode,
        epoch_size,
    )
    sampler = DistributedSampler(
        dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        sampler=sampler,
        drop_last=False,
        shuffle=False,
        persistent_workers=True,
        pin_memory=True,
    )
    return dataloader


def run_process(rank, world_size, name, epochs, model_init):
    setup_multiprocessing(rank, world_size)

    outdir = get_temp_path(os.path.join("alphas", name))
    logger = get_logger(outdir, rank)
    config_file = os.path.join(outdir, "config.json")
    training_config = create_training_config(config_file, epochs)
    model_config = create_model_config(training_config)
    torch.cuda.set_device(rank)
    torch.set_float32_matmul_precision("high")

    dataloaders = {
        x: get_dataloader(
            rank,
            world_size,
            outdir,
            training_config["mode"],
            x,
            training_config["batch_size"] * (1 if x == "training" else 2),
            training_config[f"num_{x}_shards"],
            training_config[f"{x}_epoch_size"],
            num_workers=training_config["num_dataloader_workers"],
        )
        for x in ["training", "validation"]
    }

    model = TransformerModel(model_config).to(rank)
    starting_epoch = initialize_model(model, model_init, outdir)
    if world_size == 1:
        # TODO fix recompilation errors by experimenting with dynamic=True and also try max-autotune
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

    if starting_epoch > 0:
        logger.info(f"Resuming training from epoch {starting_epoch}")
    initial_loss = evaluate_metrics(
        rank, world_size, outdir, model, dataloaders["validation"], training_config
    )
    if stopper:
        stopper(sum(initial_loss))
    logger.info(f"Initial Loss: {sum(initial_loss)}, {initial_loss}")

    for epoch in range(starting_epoch, training_config["num_epochs"]):
        training_loss = train_epoch(
            rank,
            world_size,
            outdir,
            model,
            dataloaders["training"],
            training_config,
            optimizer,
            scheduler,
            scaler,
        )
        logger.info(f"Epoch: {epoch}, Training Loss: {training_loss}")
        validation_loss = evaluate_metrics(
            rank, world_size, outdir, model, dataloaders["validation"], training_config
        )
        logger.info(
            f"Epoch: {epoch}, Validation Loss: {sum(validation_loss)}, {validation_loss}"
        )
        if stopper:
            stopper(sum(validation_loss))
            if stopper.save_model:
                save_model(rank, world_size, model, epoch, outdir)
            if stopper.early_stop:
                break
        else:
            save_model(rank, world_size, model, epoch, outdir)
    publish_model(outdir)

    if training_config["mode"] == "finetune" and rank == 0:
        model = TransformerModel(model_config).to(rank)
        epoch = initialize_model(model, model_init, outdir)
        model = torch.compile(model)
        logger.info(f"Initializing from epoch {epoch}")
        record_predictions(
            rank, world_size, model, outdir, dataloaders["validation"], "test"
        )

    dist.destroy_process_group()


def cleanup_previous_runs(name):
    outdir = get_temp_path(os.path.join("alphas", name, "training"))
    for x in os.listdir(outdir):
        if "read" in x:
            os.remove(os.path.join(outdir, x))


# Main
parser = argparse.ArgumentParser(description="PytorchPretrain")
parser.add_argument("--outdir", type=str, help="name of the data directory")
parser.add_argument(
    "--initialize", type=str, help="initialize training from a model checkpoint"
)
parser.add_argument("--gpus", type=int, help="number of gpus to use")
parser.add_argument("--epochs", type=int, help="number of epochs to use")
args = parser.parse_args()
if __name__ == "__main__":
    name = "all/Transformer/v0" if args.outdir is None else args.outdir
    gpus = torch.cuda.device_count() if args.gpus is None else args.gpus
    epochs = 1 if args.epochs is None else args.epochs
    cleanup_previous_runs(name)
    mp.spawn(run_process, args=(gpus, name, epochs, args.initialize), nprocs=gpus)