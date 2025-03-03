import argparse
import glob
import logging
import os
import time
import warnings

warnings.filterwarnings("ignore", ".*Sparse CSR tensor support is in beta state.*")

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

cls_val = -1
mask_val = -2
reserved_vals = 2
max_seq_len = 1024
datadir = "../../data/training"
ALL_MEDIUMS = [0, 1]
ALL_METRICS = ["rating", "watch", "plantowatch", "drop"]


class DiscreteEmbed(nn.Module):
    def __init__(self, vocab_size, embed_size):
        super(DiscreteEmbed, self).__init__()
        self.reserved_vals = 2
        self.embedding = nn.Sequential(
            nn.Embedding(vocab_size + self.reserved_vals, embed_size),
            nn.LayerNorm(embed_size),
        )

    def forward(self, x):
        return self.embedding(x + self.reserved_vals)


class ContinuousEmbed(nn.Module):
    def __init__(self, embed_size, dropout):
        super(ContinuousEmbed, self).__init__()
        hidden_size = int(embed_size / 4)
        self.embedding_with_weightdecay = nn.Sequential(
            nn.Linear(1, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, embed_size),
            nn.LayerNorm(embed_size),
        )

    def forward(self, x):
        return self.embedding_with_weightdecay(x.reshape(*x.shape, 1))


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
        activation,
        dropout,
    ):
        super(Bert, self).__init__()
        self.num_heads = num_attention_heads
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embed_size,
                nhead=num_attention_heads,
                dim_feedforward=intermediate_size,
                dropout=dropout,
                activation=activation,
                norm_first=True,
                batch_first=True,
            ),
            num_layers=num_layers,
        )

    def forward(self, x, mask):
        mask = torch.repeat_interleave(mask, self.num_heads, dim=0)
        return self.encoder(x, mask=mask)


class TransformerModel(nn.Module):
    def __init__(self, config):
        super(TransformerModel, self).__init__()
        self.config = config

        # create embeddings
        embeddings = []
        for size in config["vocab_sizes"]:
            if size is not None:
                embeddings.append(DiscreteEmbed(size, config["embed_size"]))
            else:
                embeddings.append(
                    ContinuousEmbed(config["embed_size"], config["dropout"])
                )
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

        # create classifiers
        metric_models = {
            m: nn.Linear(
                config["embed_size"],
                config["embed_size"],
            )
            for m in ALL_METRICS
        }
        medium_models = {}
        for i, m in enumerate(ALL_MEDIUMS):
            linear = nn.Linear(
                *reversed(self.embed.embeddings[i].embedding[0].weight.shape)
            )
            self.embed.embeddings[i].embedding[0].weight = linear.weight  # weight tying
            medium_models[m] = linear

        def create_head(medium, metric):
            base = [
                metric_models[metric],
                medium_models[medium],
            ]
            if metric in ["watch", "plantowatch"]:
                base.append(nn.LogSoftmax(dim=-1))
            return nn.Sequential(*base)

        self.classifier = nn.ModuleList(
            [
                create_head(medium, metric)
                for medium in ALL_MEDIUMS
                for metric in ALL_METRICS
            ]
        )

        # create loss functions
        lossfn_map = {
            "rating": self.mse,
            "watch": self.crossentropy,
            "plantowatch": self.crossentropy,
            "drop": self.binarycrossentropy,
        }
        self.lossfns = [
            lossfn_map[metric] for _ in ALL_MEDIUMS for metric in ALL_METRICS
        ]
        evaluate_map = {
            "rating": self.moments,
            "watch": self.crossentropy,
            "plantowatch": self.crossentropy,
            "drop": self.binarycrossentropy,
        }
        self.evaluatefns = [
            evaluate_map[metric] for _ in ALL_MEDIUMS for metric in ALL_METRICS
        ]
        self.names = [f"{m}.{metric}" for m in ALL_MEDIUMS for metric in ALL_METRICS]

    def mse(self, x, y, w):
        return (torch.square(x - y) * w).sum() / w.sum()

    def moments(self, x, y, w):
        return [
            (torch.square(x - y) * w).sum() / w.sum(),
            (torch.square(0 * x - y) * w).sum() / w.sum(),
            (torch.square(-1 * x - y) * w).sum() / w.sum(),
        ]

    def crossentropy(self, x, y, w):
        return (-x * y * w).sum() / w.sum()

    def binarycrossentropy(self, x, y, w):
        return (
            torch.nn.functional.binary_cross_entropy_with_logits(
                input=x,
                target=y,
                weight=w,
                reduction="sum",
            )
            / w.sum()
        )

    def forward(self, d, evaluate):
        inputs = [d[k] for k in self.config["vocab_names"]]
        mask = d["attn_mask"]
        e = self.embed(inputs)
        e = self.transformers(e, mask)
        lossfns = self.evaluatefns if evaluate else self.lossfns
        losses = []
        for i in range(len(lossfns)):
            k = self.names[i]
            weights = d[f"{k}.weight"]
            if not torch.is_nonzero(weights.sum()):
                losses.append(
                    torch.tensor(
                        [0.0], device=e.get_device(), requires_grad=e.requires_grad
                    )
                )
                continue
            labels = d[f"{k}.label"]
            positions = d[f"{k}.position"]
            positions = positions.reshape(*positions.shape, 1)
            classifier = self.classifier[i]
            bp = torch.nonzero(weights, as_tuple=True)
            embed = e[bp[0], bp[1], :]
            labels = labels[bp[0], bp[1]]
            positions = positions[bp[0], bp[1]]
            weights = weights[bp[0], bp[1]]
            preds = classifier(embed).gather(dim=-1, index=positions).reshape(-1)
            losses.append(lossfns[i](preds, labels, weights))
        return losses


class TransformerDataset(IterableDataset):
    def __init__(
        self,
        datadir,
        rank,
        world_size,
        mask_rate,
        batch_size,
    ):
        self.datadir = datadir
        self.mask_rate = mask_rate
        self.batch_size = batch_size * max_seq_len
        shards = sorted(glob.glob(f"{self.datadir}/*/"))
        assert len(shards) % world_size == 0
        self.fns = []
        for i, x in enumerate(shards):
            if i % world_size == rank:
                self.fns.extend(glob.glob(f"{x}/*.h5"))

    def get_index_permutation(self, arr):
        change_indices = np.where(arr[:-1] != arr[1:])[0] + 1
        blocks = np.split(arr, change_indices)
        block_indices_list = []
        current_start_index = 0
        for block in blocks:
            block_indices_list.append(
                np.arange(current_start_index, current_start_index + len(block))
            )
            current_start_index += len(block)
        num_blocks = len(blocks)
        block_permutation = np.random.permutation(num_blocks)
        index_permutation = []
        for block_index in block_permutation:
            original_block_indices = block_indices_list[block_index]
            index_permutation.extend(original_block_indices)
        return np.array(index_permutation, dtype=int)

    def block_shuffle(self, d):
        p = self.get_index_permutation(d["userid"])
        for k in d:
            d[k] = d[k][p]

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            worker_id = 0
            num_workers = 1
        else:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
        fns = [x for i, x in enumerate(self.fns) if i % num_workers == worker_id]
        np.random.shuffle(fns)
        for fn in fns:
            with h5py.File(fn, "r") as f:
                d = {}
                with h5py.File(fn) as f:
                    for k in f:
                        d[k] = f[k][:]
                self.block_shuffle(d)
            assert len(d["userid"]) % self.batch_size == 0
            idxs = list(range(len(d["userid"])))
            idxs = [
                idxs[i : i + self.batch_size]
                for i in range(0, len(idxs), self.batch_size)
            ]
            for idx in idxs:
                ret = {
                    "mask_rate": self.mask_rate,
                }
                for k, v in d.items():
                    ret[k] = v[idx]
                yield ret


def collate(data):
    assert len(data) == 1
    ret = {}
    for k, v in data[0].items():
        if k in ["mask_rate"]:
            ret[k] = v
        else:
            ret[k] = torch.tensor(v)
    return ret


def to_device(data, rank):
    d = {}
    for k in data:
        if k == "mask_rate":
            mask_rate = data[k]
        else:
            d[k] = data[k].to(rank)
    mask = (torch.rand(data["userid"].shape, device=rank) < mask_rate) & (
        d["position"] != cls_val
    )
    for m in [0, 1]:
        for metric in ["rating", "watch", "plantowatch", "drop"]:
            d[f"{m}.{metric}.position"] = torch.clone(d[f"{m}_matchedid"]).to(
                torch.int64
            )
            d[f"{m}.{metric}.position"][~mask] = 0
            d[f"{m}.{metric}.label"][~mask] = 0
            d[f"{m}.{metric}.weight"][~mask] = 0
    fields_to_mask = [
        "status",
        "rating",
        "progress",
        "0_matchedid",
        "0_distinctid",
        "1_matchedid",
        "1_distinctid",
        "source",
    ]
    for k in fields_to_mask:
        d[k][mask] = mask_val
    for k in d:
        d[k] = d[k].reshape(-1, max_seq_len)
    userid = d["userid"]
    m, n = userid.shape
    attn_mask = userid.reshape(m, 1, n) != userid.reshape(m, n, 1)
    d["attn_mask"] = attn_mask
    del d["userid"]
    return d


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


def evaluate_metrics(rank, model, dataloader):
    init = lambda metric: [0, 0, 0] if metric == "rating" else 0
    losses = [init(metric) for m in ALL_MEDIUMS for metric in ALL_METRICS]
    weights = [0 for _ in range(len(ALL_MEDIUMS) * len(ALL_METRICS))]
    progress = tqdm(desc="Test batches", mininterval=1, disable=rank != 0)
    names = [f"{m}.{metric}" for m in ALL_MEDIUMS for metric in ALL_METRICS]
    model.eval()
    for data in dataloader:
        with torch.no_grad():
            with torch.amp.autocast(f"cuda:{rank}", dtype=torch.bfloat16):
                d = to_device(data, rank)
                loss = model(d, True)
            for i in range(len(losses)):
                w = float(d[f"{names[i]}.weight"].sum())
                if w == 0:
                    continue
                if isinstance(losses[i], list):
                    for j in range(len(losses[i])):
                        losses[i][j] += float(loss[i][j]) * w
                else:
                    losses[i] += float(loss[i]) * w
                weights[i] += w
        progress.update()
    progress.close()
    model.train()
    for i in range(len(losses)):
        if isinstance(losses[i], list):
            losses[i] = minimize_quadratic([1, 0, -1], losses[i])
    return reduce_mean(rank, losses, weights)


def train_epoch(
    rank,
    model,
    dataloader,
    optimizer,
    scheduler,
    scaler,
    task_weights,
):
    training_losses = [0.0 for _ in range(len(task_weights))]
    training_weights = [0.0 for _ in range(len(task_weights))]
    progress = tqdm(desc="Training batches", mininterval=1, disable=rank != 0)
    names = [f"{m}.{metric}" for m in ALL_MEDIUMS for metric in ALL_METRICS]
    for data in dataloader:
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(f"cuda:{rank}", dtype=torch.bfloat16):
            d = to_device(data, rank)
            tloss = model(d, False)
            for i in range(len(tloss)):
                w = float(d[f"{names[i]}.weight"].sum())
                training_losses[i] += float(tloss[i]) * w
                training_weights[i] += w
            loss = sum(tloss[i] * task_weights[i] for i in range(len(tloss)))
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        progress.update()
    progress.close()
    return reduce_mean(rank, training_losses, training_weights)


def create_optimizer(model):
    should_decay = {False: [], True: []}
    for name, param in model.named_parameters():
        # TODO can we simplify this?
        if (
            "norm" in name
            or "bias" in name
            or (name.startswith("module.embed") and "weightdecay" not in name)
        ):
            decay = False
        else:
            decay = True
        should_decay[decay].append(param)
    return optim.AdamW(
        [
            {"params": should_decay[True], "weight_decay": 0.1},
            {"params": should_decay[False], "weight_decay": 0.0},
        ],
        lr=2e-4,
        betas=(0.9, 0.95),
    )


def create_learning_rate_schedule(optimizer, tokens_per_batch, epochs):
    with open(f"{datadir}/transformer/training/num_tokens.txt") as f:
        tokens_per_epoch = int(f.read())
    steps_per_epoch = int(tokens_per_epoch / tokens_per_batch)
    total_steps = epochs * steps_per_epoch
    warmup_steps = int(round(total_steps * 0.01))
    warmup_lambda = lambda x: (
        x / warmup_steps
        if x <= warmup_steps
        else 0.1
        + 0.9
        * 0.5
        * (1 + np.cos(np.pi * (x - warmup_steps) / (total_steps - warmup_steps)))
    )
    return optim.lr_scheduler.LambdaLR(optimizer, warmup_lambda)


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


def wsum(values, weights):
    return sum(x * y for (x, y) in zip(values, weights))


def make_task_weights():
    # TODO can we set this programatically?
    medium_weight = {
        0: 1,
        1: 2,
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
    stem = f"{datadir}/transformer"
    if save:
        torch.save(model.module._orig_mod.state_dict(), f"{stem}.pt")
    names = [f"{m}.{metric}" for m in ALL_MEDIUMS for metric in ALL_METRICS]
    if epoch < 0:
        with open(f"{stem}.csv", "w") as f:
            f.write(",".join(["epoch"] + names) + "\n")
    with open(f"{stem}.csv", "a") as f:
        vals = [epoch] + loss
        f.write(",".join([str(x) for x in vals]) + "\n")


def upload(rank, logger):
    if rank != 0:
        return
    logger.info("uploading model")
    template = "tag=`rclone lsd r2:rsys/database/training/ | sort | tail -n 1 | awk '{print $NF}'`; rclone --retries=10 copyto {INPUT} r2:rsys/database/training/$tag/{OUTPUT}"
    for suffix in ["pt", "csv"]:
        cmd = template.replace("{INPUT}", f"{datadir}/transformer.{suffix}").replace(
            "{OUTPUT}", f"transformer.{suffix}"
        )
        os.system(cmd)


def ddp_setup():
    init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


def train():
    ddp_setup()
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    logger = get_logger(rank, "transformer")
    logger.setLevel(logging.DEBUG)
    logger.info(f"training")
    batch_size = 2048
    num_epochs = 64
    assert batch_size % world_size == 0
    dataloaders = {
        x: DataLoader(
            TransformerDataset(
                f"{datadir}/transformer/{x}",
                rank,
                world_size,
                0.15,
                batch_size // world_size,
            ),
            batch_size=1,
            drop_last=False,
            num_workers=w,
            persistent_workers=True,
            collate_fn=collate,
        )
        for (x, w) in [("training", 8), ("test", 2)]
    }
    num_items = {
        x: pd.read_csv(f"{datadir}/{y}.csv").matchedid.max() + 1
        for (x, y) in {0: "manga", 1: "anime"}.items()
    }
    config = {
        "dropout": 0.1,
        "activation": "gelu",
        "num_layers": 4,
        "embed_size": 768,
        "intermediate_size": 768 * 4,
        "max_sequence_length": max_seq_len,
        "vocab_names": [
            "0_matchedid",
            "1_matchedid",
            "position",
            "rating",
            "status",
            "updated_at",
        ],
        "vocab_sizes": [
            num_items[0],
            num_items[1],
            max_seq_len - reserved_vals,
            None,
            8,
            None,
        ],
        "num_attention_heads": 12,
    }
    model = TransformerModel(config)
    model = model.to(rank)
    model = torch.compile(model)
    model = DDP(
        model, device_ids=[rank], output_device=rank, find_unused_parameters=True
    )
    optimizer = create_optimizer(model)
    scheduler = create_learning_rate_schedule(
        optimizer, batch_size * max_seq_len, num_epochs
    )
    scaler = torch.amp.GradScaler(rank)
    stopper = EarlyStopper(patience=float("inf"), rtol=0)
    task_weights = make_task_weights()
    get_loss = lambda x: evaluate_metrics(rank, x, dataloaders["test"])
    checkpoint = lambda m, s, l, e: checkpoint_model(rank, m, s, l, e)
    initial_loss = get_loss(model)
    logger.info(f"Initial Loss: {wsum(initial_loss, task_weights)}, {initial_loss}")
    stopper(wsum(initial_loss, task_weights))
    checkpoint(model, True, initial_loss, -1)
    for epoch in range(num_epochs):
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
            f"Epoch: {epoch}, Training Loss:"
            f" {wsum(training_loss, task_weights)} {training_loss}"
        )
        test_loss = get_loss(model)
        logger.info(
            f"Epoch: {epoch}, Test Loss:"
            f" {wsum(test_loss, task_weights)} {test_loss}"
        )
        stopper(wsum(test_loss, task_weights))
        checkpoint(model, stopper.save_model, test_loss, epoch)
        if stopper.early_stop:
            break
    destroy_process_group()
    upload(rank, logger)


def download():
    template = "tag=`rclone lsd r2:rsys/database/training/ | sort | tail -n 1 | awk '{print $NF}'`; rclone --retries=10 copyto r2:rsys/database/training/$tag"
    files = ["transformer"] + [f"{m}.csv" for m in ["manga", "anime"]]
    for data in files:
        os.system(f"{template}/{data} {datadir}/{data}")


parser = argparse.ArgumentParser()
parser.add_argument("--download", action=argparse.BooleanOptionalAction)
args = parser.parse_args()

if __name__ == "__main__":
    if args.download:
        download()
    else:
        train()
