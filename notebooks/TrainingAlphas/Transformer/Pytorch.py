#!/usr/bin/env python
# coding: utf-8

import argparse
import contextlib
import glob
import json
import logging
import math
import os
import pprint
import random
import subprocess
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


# Models
class DiscreteEmbed(nn.Module):
    def __init__(self, vocab_size, embedding_size):
        super(DiscreteEmbed, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size)

    def forward(self, x):
        return self.embedding(x)


class ContinuousEmbed(nn.Module):
    def __init__(self, vocab_size, embed_size):
        super(ContinuousEmbed, self).__init__()
        hidden_size = math.ceil(embed_size / 64)
        self.embedding = nn.Sequential(
            nn.Linear(1, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, embed_size),
        )
        self.scale = 1 / vocab_size

    def forward(self, x):
        return self.embedding(x * self.scale)


class CompositeEmbedding(nn.Module):
    def __init__(self, embeddings, postprocessor):
        super(CompositeEmbedding, self).__init__()
        self.embeddings = nn.ModuleList(embeddings)
        self.postprocessor = postprocessor

    def forward(self, inputs):
        embedding = sum(embed(x) for (embed, x) in zip(self.embeddings, inputs))
        return self.postprocessor(embedding)


class Bert(nn.Module):
    def __init__(
        self,
        num_layers,
        embed_size,
        num_attention_heads,
        intermediate_size,
        activation="gelu",
        dropout=0.1,
    ):
        super(Bert, self).__init__()
        self.num_heads = num_attention_heads
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embed_size,
                nhead=num_attention_heads,
                dim_feedforward=intermediate_size,
                dropout=0.1,
                activation=activation,
                norm_first=True,
                batch_first=True,
            ),
            num_layers=num_layers,
        )

    def forward(self, x, mask):
        # see https://stackoverflow.com/questions/68205894/how-to-prepare-data-for-tpytorchs-3d-attn-mask-argument-in-multiheadattention
        # for why torch.repeat_interleave is necessary
        mask = torch.repeat_interleave(mask, self.num_heads, dim=0)
        return self.encoder(x, mask=mask)


class TransformerModel(nn.Module):
    def __init__(self, config):
        super(TransformerModel, self).__init__()

        # create embeddings
        embeddings = []
        for size, dtype in zip(config["vocab_sizes"], config["vocab_types"]):
            if dtype is None:
                continue
            elif dtype == int:
                embeddings.append(DiscreteEmbed(size, config["embed_size"]))
            elif dtype == float:
                embeddings.append(ContinuousEmbed(size, config["embed_size"]))
            else:
                assert False
        postprocessor = nn.Sequential(
            nn.LayerNorm(config["embed_size"]), nn.Dropout(config["dropout"])
        )
        self.embed = CompositeEmbedding(embeddings, postprocessor)

        # create transformers
        self.transformers = Bert(
            num_layers=config["num_layers"],
            embed_size=config["embed_size"],
            num_attention_heads=config["num_attention_heads"],
            intermediate_size=config["intermediate_size"],
            activation=config["activation"],
            dropout=config["dropout"],
        )

        # create classifier
        self.classifier = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(
                        config["embed_size"],
                        config["vocab_sizes"][config["vocab_names"].index("anime")],
                    ),
                    nn.LogSoftmax(dim=-1),
                ),
                nn.Linear(
                    config["embed_size"],
                    config["vocab_sizes"][config["vocab_names"].index("anime")],
                ),
                nn.Sequential(
                    nn.Linear(
                        config["embed_size"],
                        config["vocab_sizes"][config["vocab_names"].index("manga")],
                    ),
                    nn.LogSoftmax(dim=-1),
                ),
                nn.Linear(
                    config["embed_size"],
                    config["vocab_sizes"][config["vocab_names"].index("manga")],
                ),
            ]
        )

        # create loss functions
        self.lossfns = [
            self.crossentropy_lossfn,
            self.rating_lossfn,
            self.crossentropy_lossfn,
            self.rating_lossfn,
        ]
        if config["mode"] == "pretrain":
            self.forward = self.pretrain_forward
        elif config["mode"] == "finetune":
            self.forward = self.finetune_forward
        else:
            assert False

    def crossentropy_lossfn(self, x, y, w):
        return (-x * y * w).sum() / w.sum()

    def rating_lossfn(self, x, y, w):
        return (torch.square(x - y) * w).sum() / w.sum()

    def pretrain_lossfn(self, embed, lossfn, classifier, positions, labels, weights):
        weight_sum = weights.sum()
        if not torch.is_nonzero(weight_sum):
            return weight_sum
        bp = torch.nonzero(weights, as_tuple=True)
        embed = embed[bp[0], bp[1], :]
        labels = labels[bp[0], bp[1]]
        positions = positions[bp[0], bp[1]]
        weights = weights[bp[0], bp[1]]
        preds = classifier(embed).gather(dim=-1, index=positions)
        return lossfn(preds, labels, weights)

    def pretrain_forward(self, inputs, mask, positions, labels, weights):
        e = self.embed(inputs)
        e = self.transformers(e, mask)
        losses = tuple(
            self.pretrain_lossfn(e, *args)
            for args in zip(self.lossfns, self.classifier, positions, labels, weights)
        )
        return losses

    def finetune_lossfn(self, embed, lossfn, classifier, labels, weights):
        weight_sum = weights.sum()
        if not torch.is_nonzero(weight_sum):
            return weight_sum
        preds = classifier(embed)
        return lossfn(preds, labels, weights)

    def finetune_forward(self, inputs, mask, positions, labels, weights):
        e = self.embed(inputs)
        e = self.transformers(e, mask)
        e = e[range(len(positions)), positions, :]
        losses = tuple(
            self.finetune_lossfn(e, *args)
            for args in zip(self.lossfns, self.classifier, labels, weights)
        )
        return losses


# Configs
def get_batch_size():
    gpu_mem = int(
        round(
            torch.cuda.get_device_properties(torch.device("cuda")).total_memory
            / 2**30
        )
    )
    mult = max(round(gpu_mem / 20), 1)
    return 64 * mult


def create_training_config(config_file, epochs):
    config = json.load(open(config_file, "r"))
    config = {
        # tokenization
        "vocab_sizes": config["vocab_sizes"],
        "vocab_types": [int, int, float, float, int, float, None, int],
        "vocab_names": [
            "anime",
            "manga",
            "rating",
            "timestamp",
            "status",
            "completion",
            "user",
            "position",
        ],
        # model
        "num_layers": 8,
        "hidden_size": 768,
        "max_sequence_length": config["max_sequence_length"],
        # training
        "peak_learning_rate": 3e-4 if config["mode"] == "pretrain" else 1e-5,
        "weight_decay": 1e-2,
        "num_epochs": epochs,
        "training_epoch_size": int(config["training_epoch_size"]),
        "validation_epoch_size": int(config["validation_epoch_size"]),
        "batch_size": get_batch_size(),
        "warmup_ratio": 0.06,
        "mode": config["mode"],
        # data
        "num_training_shards": config["num_training_shards"],
        "num_validation_shards": config["num_validation_shards"],        
        "num_dataloader_workers": config["num_dataloader_workers"],
    }
    assert len(config["vocab_sizes"]) == len(config["vocab_types"])
    assert len(config["vocab_sizes"]) == len(config["vocab_names"])
    return config


def create_model_config(training_config):
    return {
        "dropout": 0.1,
        "activation": "gelu",
        "num_layers": training_config["num_layers"],
        "embed_size": training_config["hidden_size"],
        "max_sequence_length": training_config["max_sequence_length"],
        "vocab_sizes": training_config["vocab_sizes"],
        "vocab_types": training_config["vocab_types"],
        "vocab_names": training_config["vocab_names"],
        "num_attention_heads": int(training_config["hidden_size"] / 64),
        "intermediate_size": training_config["hidden_size"] * 4,
        "mode": training_config["mode"],
    }


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

        positions = self.positions[i]
        labels = [x[i, :].toarray().flatten() for x in self.labels]
        weights = [x[i, :].toarray().flatten() for x in self.weights]
        return embeds, mask, positions, labels, weights


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
        while i + self.data[workerid]["epoch"] * self.max_size < self.data[workerid]["prev_item"]:
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


def get_data_path(file):
    path = os.getcwd()
    while os.path.basename(path) != "notebooks":
        path = os.path.dirname(path)
    path = os.path.dirname(path)
    return os.path.join(path, "data", file)


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
        # TODO optimize params
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=0.1, patience=1, threshold=0.001
        )
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
        if config["mode"] != "finetune":
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
    for data in dataloader:
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                loss = model(*to_device(data, rank))
                for i in range(len(losses)):
                    losses[i] += float(loss[i])
        steps += 1
        progress.update()
    progress.close()
    for i in range(len(losses)):
        losses[i] /= steps
    tensor_losses = torch.tensor(losses).to(rank)
    dist.all_reduce(tensor_losses, op=dist.ReduceOp.SUM)
    tensor_losses /= world_size
    return [float(x) for x in tensor_losses]


def save_model(rank, world_size, model, outdir):
    if rank == 0:
        torch.save(model.module.state_dict(), os.path.join(outdir, "model.pt"))


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
    pin_memory=False,
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
        pin_memory=pin_memory,
        num_workers=num_workers,
        drop_last=False,
        shuffle=False,
        sampler=sampler,
    )
    return dataloader


def run_process(rank, world_size, name, epochs, model_init):
    setup_multiprocessing(rank, world_size)

    outdir = get_temp_path(os.path.join("alphas", name))
    logger = get_logger(outdir, rank)
    config_file = os.path.join(outdir, "config.json")
    training_config = create_training_config(config_file, epochs)
    model_config = create_model_config(training_config)
    torch.set_float32_matmul_precision("high")

    model = TransformerModel(model_config).to(rank)
    if model_init is not None:
        model_outdir = get_data_path(os.path.join("alphas", model_init))
        model.load_state_dict(torch.load(os.path.join(model_outdir, "model.pt")))
    model = DDP(
        model, device_ids=[rank], output_device=rank, find_unused_parameters=True
    )
    optimizer = create_optimizer(model, training_config)
    scheduler = create_learning_rate_schedule(optimizer, training_config, world_size)
    scaler = torch.cuda.amp.GradScaler()
    stopper = (
        EarlyStopper(patience=10, rtol=0.001)
        if training_config["mode"] == "finetune"
        else None
    )

    for epoch in range(training_config["num_epochs"]):
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
                pin_memory=True,
                num_workers=training_config["num_dataloader_workers"],
            )
            for x in ["training", "validation"]
        }
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
        if training_config["mode"] == "finetune":
            scheduler.step(sum(validation_loss))
        logger.info(
            f"Epoch: {epoch}, Validation Loss: {sum(validation_loss)}, {validation_loss}"
        )
        if stopper:
            stopper(sum(validation_loss))
            if stopper.save_model:
                save_model(rank, world_size, model, outdir)
            if stopper.early_stop:
                break
    if not stopper:
        save_model(rank, world_size, model, outdir)
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
    os.environ['OPENBLAS_NUM_THREADS'] = '1'        
    name = "all/Transformer/v0" if args.outdir is None else args.outdir
    gpus = torch.cuda.device_count() if args.gpus is None else args.gpus
    epochs = 1 if args.epochs is None else args.epochs
    cleanup_previous_runs(name)
    mp.spawn(run_process, args=(gpus, name, epochs, args.initialize), nprocs=gpus)