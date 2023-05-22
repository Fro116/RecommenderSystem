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

    def crossentropy_lossfn(self, x, y, w):
        return (-x * y * w).sum() / w.sum()

    def rating_lossfn(self, x, y, w):
        return (torch.square(x - y) * w).sum() / w.sum()

    def lossfn(self, embed, lossfn, classifier, positions, labels, weights):
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

    def forward(self, inputs, mask, positions, labels, weights):
        e = self.embed(inputs)
        e = self.transformers(e, mask)
        losses = tuple(
            self.lossfn(e, *args)
            for args in zip(self.lossfns, self.classifier, positions, labels, weights)
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
    if gpu_mem == 40:
        return 256
    elif gpu_mem == 24:
        return 128
    else:
        return 128


def create_training_config(config_file):
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
        "num_layers": 4,
        "hidden_size": 512,
        "max_sequence_length": config["max_sequence_length"],
        # training
        "peak_learning_rate": 3e-4,
        "weight_decay": 1e-2,
        "num_epochs": 1,
        "tokens_per_epoch": config["tokens_per_epoch"],
        "num_validation_sentences": config["num_validation_sentences"],
        "batch_size": get_batch_size(),
        "warmup_ratio": 0.06,
        # data
        "num_data_workers": config["num_workers"],
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
            f["rating"][:].reshape(*f["rating"].shape, 1),
            f["timestamp"][:].reshape(*f["timestamp"].shape, 1),
            f["status"][:] - 1,
            f["completion"][:].reshape(*f["completion"].shape, 1),
            f["position"][:] - 1,
        ]
        self.mask = f["user"][:]

        def process_position(x):
            x = x[:].astype(np.int64) - 1
            return x.reshape(*x.shape, 1)

        self.positions = [
            process_position(f["positions_anime_item"]),
            process_position(f["positions_anime_rating"]),
            process_position(f["positions_manga_item"]),
            process_position(f["positions_manga_rating"]),
        ]
        self.labels = [
            np.expand_dims(f["labels_anime_item"][:], axis=-1),
            np.expand_dims(f["labels_anime_rating"][:], axis=-1),
            np.expand_dims(f["labels_manga_item"][:], axis=-1),
            np.expand_dims(f["labels_manga_rating"][:], axis=-1),
        ]
        self.weights = [
            np.expand_dims(f["weights_anime_item"][:], axis=-1),
            np.expand_dims(f["weights_anime_rating"][:], axis=-1),
            np.expand_dims(f["weights_manga_item"][:], axis=-1),
            np.expand_dims(f["weights_manga_rating"][:], axis=-1),
        ]

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        embeds = [
            self.embeddings[0][i, :],
            self.embeddings[1][i, :],
            self.embeddings[2][i, :, :],
            self.embeddings[3][i, :, :],
            self.embeddings[4][i, :],
            self.embeddings[5][i, :, :],
            self.embeddings[6][i, :],
        ]

        # a true value means that the tokens will not attend to each other
        mask = self.mask[i, :]
        mask = mask.reshape(1, mask.size) != mask.reshape(mask.size, 1)

        positions = [
            self.positions[0][:][i, :],
            self.positions[1][:][i, :],
            self.positions[2][:][i, :],
            self.positions[3][:][i, :],
        ]
        labels = [
            self.labels[0][:][i, :],
            self.labels[1][:][i, :],
            self.labels[2][:][i, :],
            self.labels[3][:][i, :],
        ]
        weights = [
            self.weights[0][:][i, :],
            self.weights[1][:][i, :],
            self.weights[2][:][i, :],
            self.weights[3][:][i, :],
        ]
        return embeds, mask, positions, labels, weights


def to_device(data, device):
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device)


def get_dataset(rank, outdir, split, batch_size, worker, num_workers):
    # wait for the data shard to be written
    completion_file = os.path.join(outdir, "training", f"{split}.{worker}.h5.complete")
    while not os.path.exists(completion_file):
        time.sleep(1)

    # read the data shard
    data_file = completion_file[: -len(".complete")]
    dataset = PretrainDataset(data_file)

    # remove old data shards
    if rank == 0:
        last_worker = worker - 1
        if last_worker == 0:
            last_worker = num_workers
        completion_file = os.path.join(outdir, "training", f"{split}.{last_worker}.h5.complete")
        data_file = completion_file[: -len(".complete")]
        with contextlib.suppress(FileNotFoundError):
            os.remove(completion_file)
            os.remove(data_file)

    return dataset


def get_data_path(file):
    path = os.getcwd()
    while os.path.basename(path) != "notebooks":
        path = os.path.dirname(path)
    path = os.path.dirname(path)
    return os.path.join(path, "data", file)


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


def create_learning_rate_schedule(optimizer, config):
    steps_per_epoch = int(
        math.ceil(
            config["tokens_per_epoch"]
            / (config["batch_size"] * config["max_sequence_length"])
        )
    )
    total_steps = config["num_epochs"] * steps_per_epoch
    warmup_ratio = config["warmup_ratio"]
    warmup_steps = int(math.ceil(total_steps * warmup_ratio))
    warmup_lambda = (
        lambda x: x / warmup_steps
        if x < warmup_steps
        else max(0, 1 - (x - warmup_steps) / (total_steps - warmup_steps))
    )
    return optim.lr_scheduler.LambdaLR(optimizer, warmup_lambda)


def train_epoch(rank, world_size, outdir, model, config, optimizer, scheduler, scaler):
    training_loss = 0.0
    training_steps = 0
    tokens_remaining = config["tokens_per_epoch"]
    tokens_per_batch = config["max_sequence_length"] * config["batch_size"]
    progress = tqdm(desc=f"Number of Tokens", total=tokens_remaining, mininterval=1, disable = rank != 0)
    data_worker = 1
    while tokens_remaining > 0:
        dataloader = get_dataloader(
            rank,
            world_size,
            outdir,
            "training",
            config["batch_size"],
            data_worker,
            config["num_data_workers"],
            pin_memory = True,
            num_workers = 4,            
        )
        data_worker = (
            1 if data_worker == config["num_data_workers"] else data_worker + 1
        )
        dataloader.sampler.set_epoch(0)
        for data in dataloader:
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                loss = sum(model(*to_device(data, rank)))
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            training_loss += float(loss)
            training_steps += 1
            update_size = tokens_per_batch * world_size
            progress.update(update_size)
            tokens_remaining -= update_size
            if tokens_remaining <= 0:
                break
    progress.close()
    return training_loss / training_steps


def evaluate_metrics(rank, world_size, outdir, model, config):
    losses = [0.0 for _ in range(4)]
    steps = 0
    sentences_remaining = config["num_validation_sentences"]
    # since we're not taking gradients, we can use bigger batches
    batch_size = 2 * config["batch_size"]
    progress = tqdm(
        desc=f"Number of Sentences", total=sentences_remaining, mininterval=1, disable = rank != 0
    )
    data_worker = 1
    while sentences_remaining > 0:
        dataloader = get_dataloader(
            rank,
            world_size,
            outdir,
            "validation",
            config["batch_size"],
            data_worker,
            config["num_data_workers"],
            pin_memory = False,
            num_workers = 0,            
        )
        data_worker = (
            1 if data_worker == config["num_data_workers"] else data_worker + 1
        )
        for data in dataloader:
            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    loss = model(*to_device(data, rank))
                    for i in range(len(losses)):
                        losses[i] += float(loss[i])
            steps += 1
            update_size = batch_size * world_size
            progress.update(update_size)
            sentences_remaining -= update_size
            if sentences_remaining <= 0:
                break
    progress.close()
    for i in range(len(losses)):
        losses[i] /= steps
    return losses


# Distributed Data Parallel
def setup_multiprocessing(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def get_dataloader(
    rank,
    world_size,
    outdir,
    split,
    batch_size,
    data_worker,
    num_data_workers,
    pin_memory=False,
    num_workers=0,
):
    dataset = get_dataset(rank, outdir, split, batch_size, data_worker, num_data_workers)
    # TODO see if rank and world_size are auto inferred
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


def run_process(rank, world_size, name, model_checkpoint):
    setup_multiprocessing(rank, world_size)

    outdir = get_data_path(os.path.join("alphas", name))
    logger = get_logger(outdir, rank)
    config_file = os.path.join(outdir, "training", "config.json")
    training_config = create_training_config(config_file)
    model_config = create_model_config(training_config)
    torch.set_float32_matmul_precision("high")

    model = TransformerModel(model_config).to(rank)
    if model_checkpoint is not None:
        model.load_state_dict(torch.load(os.path.join(outdir, "model.pt")))
    model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=True)
    optimizer = create_optimizer(model, training_config)
    scheduler = create_learning_rate_schedule(optimizer, training_config)
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(training_config["num_epochs"]):
        training_loss = train_epoch(
            rank,
            world_size,
            outdir,
            model,
            training_config,
            optimizer,
            scheduler,
            scaler,
        )
        logger.info(f"Epoch: {epoch}, Training Loss: {training_loss}")
        validation_loss = evaluate_metrics(
            rank, world_size, outdir, model, training_config
        )
        logger.info(
            f"Epoch: {epoch}, Validation Loss: {sum(validation_loss)}, {validation_loss}"
        )

    torch.save(model.module.state_dict(), os.path.join(outdir, 'model.pt'))
    dist.destroy_process_group()


# Main
parser = argparse.ArgumentParser(description='PytorchPretrain')
parser.add_argument('--outdir', type=str, help='name of the data directory')
parser.add_argument('--model_checkpoint', type=str, help='name of the model checkpoint directory')
args = parser.parse_args()
name = "all/Transformer/v0" if args.outdir is None else args.outdir
model_checkpoint = args.model_checkpoint
if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    mp.spawn(run_process, args=(world_size, name, model_checkpoint), nprocs=world_size)