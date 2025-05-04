import argparse
import datetime
import glob
import json
import logging
import os
import time
import warnings

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
import torchtune.models.llama3
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.attention.flex_attention import and_masks, create_block_mask
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, IterableDataset
from torchtune.modules.peft import get_adapter_params, set_trainable_params
from tqdm import tqdm

warnings.filterwarnings("ignore", ".*Initializing zero-element tensors is a no-op.*")
warnings.filterwarnings("ignore", ".*Sparse CSR tensor support is in beta state.*")

with open("transformer.model.py") as f:
    exec(f.read())


class PretrainDataset(IterableDataset):
    def __init__(
        self,
        datadir,
        rank,
        world_size,
        batch_size,
        mask_rate,
        causal,
    ):
        self.datadir = datadir
        self.mask_rate = mask_rate
        self.causal = causal
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
                    "causal": self.causal,
                }
                for k, v in d.items():
                    ret[k] = v[idx]
                yield ret


class FinetuneDataset(IterableDataset):
    def __init__(
        self,
        datadir,
        rank,
        world_size,
        batch_size,
        shuffle,
        causal,
    ):
        self.datadir = datadir
        self.batch_size = batch_size
        shards = sorted(glob.glob(f"{self.datadir}/*/"))
        assert len(shards) % world_size == 0
        self.fns = []
        for i, x in enumerate(shards):
            if i % world_size == rank:
                self.fns.extend(glob.glob(f"{x}/*.h5"))
        self.sparse_fields = [
            f"{m}.{metric}.{x}"
            for x in ["label", "weight"]
            for m in ALL_MEDIUMS
            for metric in ALL_METRICS
        ]
        self.shuffle = shuffle
        self.causal = causal

    def load_sparse_matrix(self, f, name):
        i = f[f"{name}_i"][:] - 1
        j = f[f"{name}_j"][:] - 1
        v = f[f"{name}_v"][:]
        m, n = f[f"{name}_size"][:]
        return scipy.sparse.coo_matrix((v, (j, i)), shape=(n, m)).tocsr()

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            worker_id = 0
            num_workers = 1
        else:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
        fns = [x for i, x in enumerate(self.fns) if i % num_workers == worker_id]
        if self.shuffle:
            np.random.shuffle(fns)
        for fn in fns:
            with h5py.File(fn, "r") as f:
                d = {}
                with h5py.File(fn) as f:
                    for k in self.sparse_fields:
                        d[k] = self.load_sparse_matrix(f, k)
                    for k in f:
                        if any(k.startswith(x) for x in self.sparse_fields):
                            continue
                        d[k] = f[k][:]
            N = d["userid"].shape[0]
            idxs = list(range(N))
            if self.shuffle:
                np.random.shuffle(idxs)
                while len(idxs) % self.batch_size != 0:
                    idxs.append(np.random.choice(idxs))
            idxs = [
                idxs[i : i + self.batch_size]
                for i in range(0, len(idxs), self.batch_size)
            ]
            for idx in idxs:
                ret = {k: v[idx, :] for k, v in d.items()}
                ret["causal"] = self.causal
                yield ret


def collate(data):
    assert len(data) == 1
    ret = {}
    for k, v in data[0].items():
        if k in ["mask_rate", "causal"]:
            ret[k] = v
        elif scipy.sparse.issparse(v):
            ret[k] = torch.sparse_csr_tensor(v.indptr, v.indices, v.data, v.shape)
        else:
            ret[k] = torch.tensor(v)
    return ret


def scatter_add(values, group_ids):
    unique_vals, inverse_indices = torch.unique(group_ids, return_inverse=True)
    s = torch.zeros(unique_vals.size(0), dtype=values.dtype, device=group_ids.device)
    s.scatter_add_(0, inverse_indices, values)
    return s[inverse_indices]


def shift_left(x):
    y = torch.zeros_like(x)
    y[..., :-1] = x[..., 1:]
    return y


def to_device(data, rank, baselines):
    d = {}
    for k in data:
        if k == "mask_rate":
            mask_rate = data[k]
        elif k == "causal":
            causal = data[k]
        else:
            d[k] = data[k].to(rank).to_dense()
    if finetune is not None:
        if causal:
            d["mask_index"] = shift_left(d["mask_index"])
        return d
    for m in ALL_MEDIUMS:
        for metric in ALL_METRICS:
            d[f"{m}.{metric}.position"] = d[f"{m}_matchedid"].clip(0).to(torch.int64)
    if causal:
        userid_mask = d["userid"] == shift_left(d["userid"])
        for m in ALL_MEDIUMS:
            for metric in ALL_METRICS:
                for k in ["position", "label", "weight"]:
                    d[f"{m}.{metric}.{k}"] = (
                        shift_left(d[f"{m}.{metric}.{k}"]) * userid_mask
                    )
    else:
        # mask
        # TODO try masking strategies
        mask = (torch.rand(data["userid"].shape, device=rank) < mask_rate) & (
            d["time"] != cls_val
        )
        for m in ALL_MEDIUMS:
            for metric in ALL_METRICS:
                d[f"{m}.{metric}.position"][~mask] = 0
                d[f"{m}.{metric}.label"][~mask] = 0
                d[f"{m}.{metric}.weight"][~mask] = 0
        for k in d:
            if k.endswith(".position") or k.endswith(".label") or k.endswith(".weight"):
                continue
            if k in ["userid"]:
                continue
            d[k][mask] = mask_val
        # residualize
        for metric in ["rating"]:
            rmask = d[metric] > 0
            for m in ALL_MEDIUMS:
                base = baselines[metric][m]
                _, λ_u, _, λ_wu, λ_wa = base["params"]
                idx = d[f"{m}_matchedid"].clip(0).to(torch.int64)
                a = torch.gather(base["a"], 0, idx)
                acount = torch.gather(base["item_counts"], 0, idx)
                rating_mask = (d[f"{m}_matchedid"] >= 0) * rmask
                ucount = scatter_add(rating_mask.to(torch.float32), d["userid"])
                w = (ucount.clip(1) ** λ_wu * acount**λ_wa) * rating_mask
                denom = scatter_add(w, d["userid"]) + np.exp(λ_u)
                numer = scatter_add((d[metric] - a) * w, d["userid"])
                bias = numer / denom
                beta = base["weight"]
                if metric in ALL_METRICS:
                    label_idx = d[f"{m}.{metric}.position"].clip(0).to(torch.int64)
                    label_rating_pred = bias + torch.gather(base["a"], 0, label_idx)
                    label_rating_mask = d[f"{m}.{metric}.weight"] > 0
                    d[f"{m}.{metric}.label"] -= beta * label_rating_pred * label_rating_mask
                d[metric] = d[metric] - beta * (bias + a) * rating_mask
            d[metric] *= rmask
    for k in d:
        d[k] = d[k].reshape(-1, max_seq_len)
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


def evaluate_metrics(rank, model, dataloader, baselines):
    init = lambda metric: [0, 0, 0] if metric in ["rating"] else 0
    losses = [init(metric) for m in ALL_MEDIUMS for metric in ALL_METRICS]
    weights = [0 for _ in range(len(ALL_MEDIUMS) * len(ALL_METRICS))]
    progress = tqdm(desc="Test batches", mininterval=1, disable=rank != 0)
    names = [f"{m}.{metric}" for m in ALL_MEDIUMS for metric in ALL_METRICS]
    model.eval()
    for data in dataloader:
        with torch.no_grad():
            with torch.amp.autocast(f"cuda:{rank}", dtype=torch.bfloat16):
                d = to_device(data, rank, baselines)
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
    baselines,
):
    training_losses = [0.0 for _ in range(len(task_weights))]
    training_weights = [0.0 for _ in range(len(task_weights))]
    progress = tqdm(desc="Training batches", mininterval=1, disable=rank != 0)
    names = [f"{m}.{metric}" for m in ALL_MEDIUMS for metric in ALL_METRICS]
    for data in dataloader:
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(f"cuda:{rank}", dtype=torch.bfloat16):
            d = to_device(data, rank, baselines)
            tloss = model(d, False)
            for i in range(len(tloss)):
                w = float(d[f"{names[i]}.weight"].sum())
                training_losses[i] += float(tloss[i].detach()) * w
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


def create_optimizer(model, config):
    param_dict = {pn: p for pn, p in model.named_parameters()}
    param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    return optim.AdamW(
        [
            {"params": decay_params, "weight_decay": 0.1},
            {"params": nodecay_params, "weight_decay": 0.0},
        ],
        lr=config["learning_rate"],
        betas=(0.9, 0.95),
    )


def create_learning_rate_schedule(optimizer, tokens_per_batch, epochs):
    # TODO tune schedule
    return optim.lr_scheduler.LambdaLR(optimizer, lambda x: 1)


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
    # rescale losses so each task is equally weighted
    scale = {
        0: {
            "watch": 5.402838564127421,
            "rating": 1.2371406205358377,
        },
        1: {
            "watch": 3.157225522146773,
            "rating": 1.0656737615337866,
        },
    }
    scale = [scale[x][y] for x in ALL_MEDIUMS for y in ALL_METRICS]
    # task balancing
    if finetune is None:
        medium_weight = {0: 0.25, 1: 1}
    else:
        medium_weight = {finetune_medium: 1, 1 - finetune_medium: 0}
    metric_weight = {
        "watch": 1,
        "rating": 0.25,
    }
    weights = [
        medium_weight[x] * metric_weight[y] for x in ALL_MEDIUMS for y in ALL_METRICS
    ]
    weights = [x / sum(weights) for x in weights]
    return [(w / s) for (w, s) in zip(weights, scale)]


def get_logger(rank, name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    if rank != 0:
        return logger
    formatter = logging.Formatter(
        "%(name)s:%(levelname)s:%(asctime)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    logger.addHandler(stream)
    return logger


def checkpoint_model(rank, model, config, save, loss, epoch, task_weights):
    if rank != 0:
        return
    stem = f"{datadir}/transformer"
    if finetune is not None:
        stem = f"{stem}.{finetune_medium}.finetune"
    if save:
        torch.save(model.module._orig_mod.state_dict(), f"{stem}.pt")
    names = [f"{m}.{metric}" for m in ALL_MEDIUMS for metric in ALL_METRICS]
    if epoch < 0:
        with open(f"{stem}.csv", "w") as f:
            f.write(",".join(["epoch", "loss"] + names) + "\n")
    with open(f"{stem}.csv", "a") as f:
        vals = [epoch, wsum(loss, task_weights)] + loss
        f.write(",".join([str(x) for x in vals]) + "\n")
    with open(f"{stem}.config", "w") as f:
        json.dump(config, f, indent=4)


def upload(rank, logger):
    if rank != 0 or finetune is not None:
        return
    logger.info("uploading model")
    with open(os.path.join(datadir, "latest"), "r") as f:
        latest = f.read()
    for suffix in ["pt", "csv", "config"]:
        cmd =  f"rclone --retries=10 copyto {datadir}/transformer.{suffix} r2:rsys/database/training/{latest}/transformer.{suffix}"
        os.system(cmd)


def ddp_setup():
    init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


def training_config():
    if finetune is not None:
        with open(f"{datadir}/transformer.config") as f:
            config = json.load(f)
            config["forward"] = "finetune"
            config["lora"] = True
            config["learning_rate"] = 1e-4
            return config
    num_items = {
        x: int(pd.read_csv(f"{datadir}/{y}.csv").matchedid.max()) + 1
        for (x, y) in {0: "manga", 1: "anime"}.items()
    }
    num_distinct_items = {
        x: int(pd.read_csv(f"{datadir}/{y}.csv").distinctid.max()) + 1
        for (x, y) in {0: "manga", 1: "anime"}.items()
    }
    min_ts = datetime.datetime.strptime("20000101", "%Y%m%d").timestamp()
    with open(os.path.join(datadir, "latest"), "r") as f:
        max_ts = datetime.datetime.strptime(f.read().strip(), "%Y%m%d").timestamp()
    return {
        "num_layers": 8,
        "num_heads": 32,
        "num_kv_heads": 16,
        "embed_dim": 2048,
        "intermediate_dim": None, # TODO 8192
        "distinctid_dim": 128,
        "max_sequence_length": max_seq_len,
        "vocab_sizes": {
            "0_matchedid": num_items[0],
            "1_matchedid": num_items[1],
            "0_distinctid": num_distinct_items[0],
            "1_distinctid": num_distinct_items[1],
            "status": 9,
        },
        "reserved_vals": 2,
        "causal": False,
        "min_ts": min_ts,
        "max_ts": max_ts,
        "learning_rate": 2e-4,
        "forward": "pretrain",
        "lora": False,
    }


def train():
    ddp_setup()
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    logger = get_logger(rank, "transformer")
    logger.setLevel(logging.DEBUG)
    batch_size = 512 if finetune is None else 32
    num_epochs = 1024
    assert batch_size % world_size == 0
    baselines = get_baselines(rank)
    config = training_config()

    def TransformerDataset(x):
        if finetune is None:
            return PretrainDataset(
                f"{datadir}/transformer/{x}",
                rank,
                world_size,
                batch_size // world_size,
                mask_rate=0.15,
                causal=config["causal"],
            )
        else:
            return FinetuneDataset(
                f"{datadir}/transformer/{x}",
                rank,
                world_size,
                batch_size // world_size,
                shuffle=x == "training",
                causal=config["causal"],
            )

    dataloaders = {
        x: DataLoader(
            TransformerDataset(x),
            batch_size=1,
            drop_last=False,
            num_workers=w,
            persistent_workers=True,
            collate_fn=collate,
        )
        for (x, w) in [("training", 8), ("test", 2)]
    }
    model = TransformerModel(config)
    logger.info(
        f"Created model with {sum(p.numel() for p in model.parameters())} parameters"
        f" and {sum(p.numel() for p in model.parameters() if p.requires_grad)} trainable parameters"
    )
    if finetune is not None:
        model.load_state_dict(torch.load(finetune, weights_only=True), strict=False)
    model = model.to(rank)
    model = torch.compile(model)
    model = DDP(
        model, device_ids=[rank], output_device=rank, find_unused_parameters=True
    )
    optimizer = create_optimizer(model, config)
    scheduler = create_learning_rate_schedule(
        optimizer, batch_size * max_seq_len, num_epochs
    )
    scaler = torch.amp.GradScaler(rank)
    stopper = (
        EarlyStopper(patience=float("inf"), rtol=0)
        if finetune is None
        else EarlyStopper(patience=1, rtol=0.001)
    )
    task_weights = make_task_weights()
    get_loss = lambda x: evaluate_metrics(rank, x, dataloaders["test"], baselines)
    checkpoint = lambda m, s, l, e: checkpoint_model(rank, m, config, s, l, e, task_weights)
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
            baselines,
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
    files = (
        ["transformer", "latest"]
        + [f"{m}.csv" for m in ["manga", "anime"]]
        + [f"baseline.{metric}.{m}.msgpack" for metric in ["rating"] for m in [0, 1]]
    )
    for data in files:
        os.system(f"{template}/{data} {datadir}/{data}")


def get_baselines(rank):
    baselines = {}
    for metric in ["rating"]:
        baselines[metric] = {}
        for m in [0, 1]:
            with open(f"{datadir}/baseline.{metric}.{m}.msgpack", "rb") as f:
                baseline = msgpack.unpackb(f.read(), strict_map_key=False)
                d = {}
                d["params"] = baseline["params"]["λ"]
                d["weight"] = baseline["weight"][0]
                d["a"] = torch.tensor(baseline["params"]["a"]).to(rank)
                item_counts = baseline["params"]["item_counts"]
                item_counts = [
                    item_counts.get(x, 1) for x in range(len(baseline["params"]["a"]))
                ]
                d["item_counts"] = torch.tensor(item_counts).to(rank)
                baselines[metric][m] = d
    return baselines


parser = argparse.ArgumentParser()
parser.add_argument("--datadir", type=str)
parser.add_argument("--finetune", type=str, default=None)
parser.add_argument("--finetune_medium", type=int, default=None)
parser.add_argument("--download", action=argparse.BooleanOptionalAction)
args = parser.parse_args()
datadir = args.datadir
finetune = args.finetune
finetune_medium = args.finetune_medium

if __name__ == "__main__":
    if args.download:
        download()
    else:
        train()
