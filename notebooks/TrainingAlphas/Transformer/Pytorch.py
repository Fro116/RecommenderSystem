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


def get_logger(outdir, rank):
    # writes to file and also writes to stdout if rank==0
    logger = logging.getLogger(f"pytorch.{rank}")
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        "%(name)s:%(levelname)s:%(asctime)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    version = 0
    filename = os.path.join(outdir, f"pytorch.{rank}.log")
    while os.path.exists(filename):
        version += 1
        filename = os.path.join(outdir, f"pytorch.{rank}.log.{version}")

    streams = [logging.FileHandler(filename, "a")]
    if rank == 0:
        streams.append(logging.StreamHandler())
    for stream in streams:
        stream.setFormatter(formatter)
        logger.addHandler(stream)
    return logger


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
    def __init__(
        self,
        outdir,
        epoch_size,
        chunk_size,
        mode,
    ):
        self.outdir = outdir
        self.epoch_size = epoch_size
        self.chunk_size = chunk_size
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
            self.chunk = -1
        self.last_index = i

        chunk, index = divmod(i, self.chunk_size)
        if self.chunk != chunk:
            self.chunk = chunk
            file = os.path.join(self.outdir, str(self.epoch), f"{chunk}.h5")
            if self.mode == "pretrain":
                self.dataset = PretrainDataset(file)
            elif self.mode == "finetune":
                self.dataset = FinetuneDataset(file)
            else:
                assert False
        return self.dataset[index]


def to_device(data, device):
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device)


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


def make_task_weights(losses):
    weights = []
    for i in range(len(losses)):
        if losses[i] > 0:
            weights.append(1 / losses[i])
        else:
            weights.append(0)
    norm = sum(weights) / len(weights)
    if norm == 0:
        return [1 for _ in range(len(losses))]
    else:
        return [x / norm for x in weights]
    return weights


def train_epoch(
    rank,
    world_size,
    outdir,
    model,
    dataloader,
    config,
    optimizer,
    scheduler,
    scaler,
    task_weights,
):
    training_losses = [0.0 for _ in range(8)]
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
            tloss = None
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        if scheduler is not None:
            scheduler.step()
        training_steps += 1
        progress.update()
    progress.close()

    training_losses = [x / training_steps for x in training_losses]
    tensor_losses = torch.tensor(training_losses).to(rank)
    if world_size > 1:
        dist.all_reduce(tensor_losses, op=dist.ReduceOp.SUM)
    tensor_losses /= world_size
    return [float(x) for x in tensor_losses]


def evaluate_metrics(rank, world_size, outdir, model, dataloader, config):
    losses = [0.0 for _ in range(8)]
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
    for i in range(len(losses)):
        losses[i] /= steps
    tensor_losses = torch.tensor(losses).to(rank)
    if world_size > 1:
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


def load_checkpoints(outdir):
    checkpoint = None
    epoch = -1
    for x in glob.glob(os.path.join(outdir, f"model.*.pt")):
        trial = int(os.path.basename(x).split(".")[1])
        if trial > epoch:
            epoch = trial
            checkpoint = x
    return checkpoint, epoch


def initialize_model(model, model_init, outdir, logger):
    if model_init is not None:
        model_paths = [get_data_path(os.path.join("alphas", model_init, "model.pt"))]
    else:
        model_paths = [os.path.join(outdir, "model.pt")]
    for path in model_paths:
        if os.path.exists(path):
            logger.info(f"loading model from {path}")
            model.load_state_dict(load_model(path))
            return


def save_model(rank, world_size, model, epoch, outdir):
    if rank != 0:
        return
    checkpoint = os.path.join(outdir, f"model.{epoch}.pt")
    if world_size > 1:
        model = model.module
    torch.save(model.state_dict(), checkpoint)
    previous_checkpoints = glob.glob(os.path.join(outdir, f"model.*.pt"))
    for fn in previous_checkpoints:
        if fn != checkpoint:
            os.remove(fn)


def publish_model(outdir, rank):
    if rank != 0:
        return
    checkpoint, epoch = load_checkpoints(outdir)
    assert checkpoint is not None
    modelfn = os.path.join(outdir, f"model.pt")
    shutil.copyfile(checkpoint, modelfn)
    os.remove(checkpoint)


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
    epoch_size,
    chunk_size,
    batch_size,
):
    dataset = StreamingDataset(
        os.path.join(outdir, split),
        epoch_size,
        chunk_size,
        mode,
    )
    sampler = DistributedSampler(
        dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=1,
        sampler=sampler,
        drop_last=False,
        shuffle=False,
        persistent_workers=True,
        pin_memory=True,
    )
    return dataloader


def run_process(rank, world_size, name, epochs, model_init):
    if world_size > 1:
        setup_multiprocessing(rank, world_size)

    outdir = get_data_path(os.path.join("alphas", name))
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
            training_config[f"{x}_epoch_size"],
            training_config[f"chunk_size"],
            training_config[f"{x}_batch_size"],
        )
        for x in ["training", "validation"]
    }
    model = TransformerModel(model_config).to(rank)
    initialize_model(model, model_init, outdir, logger)
    if training_config["mode"] == "finetune":
        model = torch.compile(model)
    if world_size > 1:
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

    initial_loss = evaluate_metrics(
        rank, world_size, outdir, model, dataloaders["validation"], training_config
    )
    if stopper:
        stopper(sum(initial_loss))  # TODO should this be weighted?
    logger.info(f"Initial Loss: {sum(initial_loss)}, {initial_loss}")
    task_weights = make_task_weights(initial_loss)

    for epoch in range(training_config["num_epochs"]):
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
            task_weights,
        )
        logger.info(
            f"Epoch: {epoch}, Training Loss: {sum(training_loss)}, {training_loss}"
        )
        validation_loss = evaluate_metrics(
            rank, world_size, outdir, model, dataloaders["validation"], training_config
        )
        logger.info(
            f"Epoch: {epoch}, Validation Loss: {sum(validation_loss)}, {validation_loss}"
        )
        task_weights = make_task_weights(validation_loss)
        if stopper:
            stopper(sum(validation_loss))
            if stopper.save_model:
                save_model(rank, world_size, model, epoch, outdir)
            if stopper.early_stop:
                break
        else:
            save_model(rank, world_size, model, epoch, outdir)
    publish_model(outdir, rank)

    if training_config["mode"] == "finetune" and rank == 0:
        model = TransformerModel(model_config).to(rank)
        initialize_model(model, None, outdir, logger)
        model = torch.compile(model)
        record_predictions(
            rank, world_size, model, outdir, dataloaders["validation"], "test"
        )

    if world_size > 1:
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
parser.add_argument("--epochs", type=int, help="number of epochs to use")
args = parser.parse_args()
if __name__ == "__main__":
    gpus = torch.cuda.device_count() if args.gpus is None else args.gpus
    cleanup_previous_runs(args.outdir)
    if gpus > 1:
        mp.spawn(
            run_process,
            args=(gpus, args.outdir, args.epochs, args.initialize),
            nprocs=gpus,
        )
    else:
        run_process(0, gpus, args.outdir, args.epochs, args.initialize)