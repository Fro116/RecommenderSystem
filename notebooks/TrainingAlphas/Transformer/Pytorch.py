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
import hdf5plugin
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
    def __init__(self, file, vocab_names, vocab_types):
        self.filename = file
        f = h5py.File(file, "r")

        def process(x, dtype):
            if dtype == "float":
                return f[x][:].reshape(*f[x].shape, 1).astype(np.float32)
            elif dtype == "int":
                return f[x][:]
            else:
                assert False

        self.length = f["userid"].shape[0]
        self.embeddings = [
            process(x, y) for (x, y) in zip(vocab_names, vocab_types) if x != "userid"
        ]
        self.mask = f["userid"][:]

        def process_positions(x):
            x = x[:].astype(np.int64)
            return x.reshape(*x.shape, 1)

        self.positions = [
            process_positions(f[f"positions_{medium}_{metric}"])
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
    def __init__(self, file, vocab_names, vocab_types):
        self.filename = file
        f = h5py.File(file, "r")

        def process(x, dtype):
            if dtype == "float":
                return f[x][:].reshape(*f[x].shape, 1).astype(np.float32)
            elif dtype == "int":
                return f[x][:]
            else:
                assert False

        self.length = f["userid"].shape[0]
        self.embeddings = [
            process(x, y) for (x, y) in zip(vocab_names, vocab_types) if x != "userid"
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
        split,
        epoch_sizes,
        chunk_size,
        batch_size,
        mode,
        epochs,
        vocab_names,
        vocab_types,
    ):
        self.outdir = outdir
        self.split = split
        self.epoch_sizes = epoch_sizes
        self.chunk_size = chunk_size
        self.num_epochs = epochs
        self.mode = mode
        self.epoch = 0
        self.partition = 0
        self.chunk = -1
        self.dataset = None
        self.last_index = -1
        self.vocab_names = vocab_names
        self.vocab_types = vocab_types
        # manually handle DistributedSampler's drop_last=False padding        
        self.unique_len = sum(self.epoch_sizes)
        next_multiple = lambda n, m: ((n + m - 1) // m) * m
        self.expanded_len = next_multiple(self.unique_len, batch_size)
        
    def __len__(self):
        return self.expanded_len

    def calc_index(self, i):
        i = i % self.unique_len
        p = 0
        while i >= self.epoch_sizes[p]:
            i -= self.epoch_sizes[p]
            p += 1
        chunk, index = divmod(i, self.chunk_size)
        return p, chunk, index

    def __getitem__(self, i):
        if i < self.last_index:
            self.epoch += 1
            if self.epoch == self.num_epochs:
                self.epoch = 0
            self.chunk = -1
        self.last_index = i

        partition, chunk, index = self.calc_index(i)
        if (self.partition, self.chunk) != (partition, chunk):
            self.partition = partition
            self.chunk = chunk
            file = os.path.join(
                self.outdir,
                str(self.partition),
                self.split,
                str(self.epoch),
                f"{chunk}.h5",
            )
            while not os.path.exists(file):
                time.sleep(1) # wait for the file to be downloaded
            args = (file, self.vocab_names, self.vocab_types)                
            if self.mode == "pretrain":
                self.dataset = PretrainDataset(*args)
            elif self.mode == "finetune":
                self.dataset = FinetuneDataset(*args)
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
        lr=config["learning_rate"],
        betas=config["adam_beta"],
    )


def create_learning_rate_schedule(optimizer, config, world_size):
    if config["mode"] == "pretrain":
        steps_per_epoch = math.ceil(
            sum(config["training_epoch_sizes"])
            / (config["training_batch_size"] * world_size)
        )
        total_steps = config["num_epochs"] * steps_per_epoch
        warmup_steps = math.ceil(total_steps * config["warmup_ratio"])
        warmup_lambda = (
            lambda x: x / warmup_steps
            if x <= warmup_steps
            else 0.1
            + 0.9
            * 0.5
            * (
                1
                + np.cos(
                    np.pi * (x - warmup_steps) / (total_steps - warmup_steps)
                )
            )
        )
        return optim.lr_scheduler.LambdaLR(optimizer, warmup_lambda)
    elif config["mode"] == "finetune":
        return None
    else:
        assert False


def wsum(values, weights):
    return sum(x * y for (x, y) in zip(values, weights))


def make_task_weights():
    medium_weight = {
        "anime": 2,
        "manga": 1,
    }
    metric_weight = {
        "rating": 1,
        "watch": 4,
        "plantowatch": 1 / 4,
        "drop": 1 / 4,
    }
    weights = [
        medium_weight[x] * metric_weight[y] for x in ALL_MEDIUMS for y in ALL_METRICS
    ]
    weights = [x / sum(weights) for x in weights]
    # rescale losses so each task is equally weighted
    scale = [
        1.036553297051344,
        5.180758325289576,
        5.87909245526433,
        0.061069027408138736,
        1.1299115263959014,
        3.515180369672557,
        4.851891694665132,
        0.0673615127350479,
    ]
    return [(w / s) for (w, s) in zip(weights, scale)]


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
    clip_norm,
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
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
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


def compile(model, mode):
    if mode == "pretrain":
        # we mask a variable number of tokens in each batch,
        # so we can't compile the whole model
        model.embed = torch.compile(model.embed)
        model.transformers = torch.compile(model.transformers)
        for i in range(len(model.classifier)):
            model.classifier[i] = torch.compile(model.classifier[i]) 
        return model
    elif mode == "finetune":
       return torch.compile(model)        
    else:
        assert False


def print_model_size(model, logger):
    num_params = []
    for m in [model, model.embed, model.transformers, model.classifier]:
        model_parameters = filter(lambda p: p.requires_grad, m.parameters())
        params = sum(dict((p.data_ptr(), p.numel()) for p in model_parameters).values())
        num_params.append(params)
    logger.info(f"Loading model with {num_params[0]} unique trainable params")
    for n, p in zip(["Embeddings", "Transformers", "Classifiers"], num_params[1:]):
        logger.info(f"{n} have {p} trainable params")
    logger.info(
        f"{sum(num_params[1:]) - num_params[0]} params are shared between "
        "embeddings and classifiers"
    )


# multiprocessing


def setup_multiprocessing(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def get_dataloader(
    rank,
    world_size,
    outdir,
    split,
    mode,
    epoch_sizes,
    chunk_size,
    batch_size,
    epochs,
    vocab_names,
    vocab_types,
):
    dataset = StreamingDataset(
        outdir,
        split,
        epoch_sizes,
        chunk_size,
        batch_size * world_size,
        mode,
        epochs,
        vocab_names,
        vocab_types,
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
    training_config = create_training_config(outdir)
    logger.info(f"Training config: {training_config}")
    model_config = create_model_config(training_config)
    logger.info(f"Model config: {model_config}")
    torch.cuda.set_device(rank)
    torch.set_float32_matmul_precision("high")

    dataloaders = {
        x: get_dataloader(
            rank,
            world_size,
            outdir,
            x,
            training_config["mode"],
            training_config[f"{x}_epoch_sizes"],
            training_config[f"chunk_size"],
            training_config[f"{x}_batch_size"],
            training_config["num_epochs"],
            training_config["vocab_names"],
            training_config["vocab_types"],
        )
        for x in training_config["splits"]
    }
    model = TransformerModel(model_config).to(rank)
    if model_init is not None:
        initialize_model(model, model_init, logger)
    model = compile(model, training_config["mode"])
    print_model_size(model, logger)
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
    
    task_weights = make_task_weights()
    logger.info(f"Using task_weights {task_weights}")
    initial_loss = evaluate_metrics(rank, model, dataloaders["validation"])
    logger.info(f"Initial Loss: {wsum(initial_loss, task_weights)}, {initial_loss}")    
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
            training_config["clip_norm"],
        )
        logger.info(
            f"Epoch: {epoch}, Training Loss:"
            f" {wsum(training_loss, task_weights)} {training_loss}"
        )
        validation_loss = evaluate_metrics(rank, model, dataloaders["validation"])
        logger.info(
            f"Epoch: {epoch}, Validation Loss:"
            f" {wsum(validation_loss, task_weights)}, {validation_loss}"
        )
        if stopper:
            stopper(wsum(validation_loss, task_weights))
            if stopper.save_model:
                save_model(rank, model, outdir)
            elif stopper.early_stop:
                break
        else:
            save_model(rank, model, outdir)
        if rank == 0:
            # signals to an external memory pager that we can delete this
            # epoch and download the next one. This is useful for training
            # on datasets that are larger than the disk size
            open(os.path.join(outdir, f"epoch.{epoch}.complete"), "w").close()
   
    if training_config["mode"] == "finetune" and rank == 0:
        model = TransformerModel(model_config).to(rank)
        initialize_model(model, name, logger)
        model = compile(model, training_config["mode"])
        record_predictions(rank, model, outdir, dataloaders["test"])

    dist.destroy_process_group()


parser = argparse.ArgumentParser(description="PytorchPretrain")
parser.add_argument("--outdir", type=str, help="name of the data directory")
parser.add_argument(
    "--initialize", type=str, help="initialize training from a model checkpoint"
)
parser.add_argument("--gpus", type=int, help="number of gpus to use")
args = parser.parse_args()
if __name__ == "__main__":
    gpus = torch.cuda.device_count() if args.gpus is None else args.gpus
    mp.spawn(
        run_process,
        args=(gpus, args.outdir, args.initialize),
        nprocs=gpus,
    )