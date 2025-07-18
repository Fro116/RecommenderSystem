import argparse
import datetime
import glob
import logging
import os
import subprocess
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
        local_rank,
        local_world_size,
        tokens_per_batch,
    ):
        self.batch_size = tokens_per_batch
        shards = sorted(glob.glob(f"{datadir}/*/"))
        assert len(shards) % local_world_size == 0
        self.fns = []
        for i, x in enumerate(shards):
            if i % local_world_size == local_rank:
                self.fns.extend(glob.glob(f"{x}/*.h5"))

    def get_index_permutation(self, arr):
        change_indices = np.where(arr[:-1] != arr[1:])[0] + 1
        blocks = np.split(arr, change_indices)
        block_indices_list = []
        current_start_index = 0
        for block in blocks:
            p = np.arange(current_start_index, current_start_index + len(block))
            block_indices_list.append(p)
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
            with h5py.File(fn) as f:
                d = {}
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
                yield {k: v[idx] for k, v in d.items()}


class FinetuneDataset(IterableDataset):
    def __init__(
        self,
        datadir,
        local_rank,
        local_world_size,
        batch_size,
        shuffle,
    ):
        self.batch_size = batch_size
        self.shuffle = shuffle
        shards = sorted(glob.glob(f"{datadir}/*/"))
        assert len(shards) % local_world_size == 0
        self.fns = []
        for i, x in enumerate(shards):
            if i % local_world_size == local_rank:
                self.fns.extend(glob.glob(f"{x}/*.h5"))

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
                    for k in f:
                        d[k] = f[k][:]
            N = d["userid"].shape[0]
            idxs = []
            for i in range(N):
                if any(d[f"{args.finetune_medium}.{metric}.weight"][i, :].sum() > 0 for metric in ["watch", "rating"]):
                    idxs.append(i)
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
                yield ret


def worker_init_fn(worker_id):
    np.random.seed(torch.utils.data.get_worker_info().seed % 2**32)


def collate(data):
    assert len(data) == 1
    ret = {}
    for k, v in data[0].items():
        if scipy.sparse.issparse(v):
            ret[k] = torch.sparse_csr_tensor(v.indptr, v.indices, v.data, v.shape)
        else:
            ret[k] = torch.tensor(v)
    return ret


def to_device(data, local_rank):
    d = {}
    for k in data:
        d[k] = data[k].to(local_rank).to_dense()
    for m in ALL_MEDIUMS:
        for metric in ALL_METRICS:
            d[f"{m}.{metric}.position"] = d[f"{m}_matchedid"].to(torch.int64)
    return d


def minimize_quadratic(x, y):
    assert len(x) == 3 and len(y) == 3
    A = np.array([[x[0] ** 2, x[0], 1], [x[1] ** 2, x[1], 1], [x[2] ** 2, x[2], 1]])
    B = np.array(y)
    a, b, c = np.linalg.solve(A, B)
    x_extremum = -b / (2 * a)
    y_extremum = a * x_extremum**2 + b * x_extremum + c
    return float(y_extremum)


def reduce_mean(local_rank, x, w):
    x = torch.tensor(x).to(local_rank)
    w = torch.tensor(w).to(local_rank)
    dist.all_reduce(x, op=dist.ReduceOp.SUM)
    dist.all_reduce(w, op=dist.ReduceOp.SUM)
    return [float(a) / float(b) if float(b) != 0 else 0 for (a, b) in zip(x, w)]


def evaluate_metrics(local_rank, model, dataloader):
    init = lambda metric: [0, 0, 0] if metric in ["rating"] else 0
    losses = [init(metric) for m in ALL_MEDIUMS for metric in ALL_METRICS]
    weights = [0 for _ in range(len(ALL_MEDIUMS) * len(ALL_METRICS))]
    progress = tqdm(desc="Test batches", mininterval=1, disable=local_rank != 0)
    names = [f"{m}.{metric}" for m in ALL_MEDIUMS for metric in ALL_METRICS]
    model.eval()
    for data in dataloader:
        with torch.no_grad():
            with torch.amp.autocast(f"cuda:{local_rank}", dtype=torch.bfloat16):
                d = to_device(data, local_rank)
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
    return reduce_mean(local_rank, losses, weights)


def train_epoch(
    local_rank,
    model,
    dataloader,
    optimizer,
    scheduler,
    scaler,
    task_weights,
    grad_accum_steps,
):
    training_losses = [0.0 for _ in range(len(task_weights))]
    training_weights = [0.0 for _ in range(len(task_weights))]
    progress = tqdm(desc="Training batches", mininterval=1, disable=local_rank != 0)
    names = [f"{m}.{metric}" for m in ALL_MEDIUMS for metric in ALL_METRICS]
    optimizer.zero_grad(set_to_none=True)
    for step, data in enumerate(dataloader):
        with torch.amp.autocast(f"cuda:{local_rank}", dtype=torch.bfloat16):
            d = to_device(data, local_rank)
            tloss = model(d, False)
            for i in range(len(tloss)):
                w = float(d[f"{names[i]}.weight"].sum())
                training_losses[i] += float(tloss[i].detach()) * w
                training_weights[i] += w
            loss = sum(tloss[i] * task_weights[i] for i in range(len(tloss))) / grad_accum_steps
        scaler.scale(loss).backward()
        if (step + 1) % grad_accum_steps != 0:
            continue
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()
        progress.update()
    progress.close()
    return reduce_mean(local_rank, training_losses, training_weights)


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


class ConstantScheduler(object):
    def __init__(self):
        self.steps = 0

    def __call__(self, epoch):
        self.steps += 1
        return 1


def create_learning_rate_schedule(optimizer, tokens_per_batch, epochs):
    return optim.lr_scheduler.LambdaLR(optimizer, ConstantScheduler())


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
    if args.modeltype == "causal":
        scale = {
            0: {
                "watch": 5.564432094339503,
                "rating": 1.2555994665648476,
            },
            1: {
                "watch": 3.7259071988829593,
                "rating": 1.1956148524173345,
            },
        }
    elif args.modeltype == "masked":
        scale = {
            0: {
                "watch": 4.618602403897067,
                "rating": 1.1958987168236102,
            },
            1: {
                "watch": 2.5443243303769867,
                "rating": 1.0527565486045412,
            },
        }
    else:
        assert False
    scale = [scale[x][y] for x in ALL_MEDIUMS for y in ALL_METRICS]
    # task balancing
    if args.modeltype == "causal":
        metric_weight = {"watch": 0.25, "rating": 1}
    elif args.modeltype == "masked":
        metric_weight = {"watch": 1, "rating": 0.25}
    else:
        assert False
    if args.finetune is None:
        medium_weight = {0: 0.25, 1: 1}
    else:
        medium_weight = {args.finetune_medium: 1, 1 - args.finetune_medium: 0}
    weights = [
        medium_weight[x] * metric_weight[y] for x in ALL_MEDIUMS for y in ALL_METRICS
    ]
    weights = [x / sum(weights) for x in weights]
    return [(w / s) for (w, s) in zip(weights, scale)]


def get_logger(local_rank, name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    if local_rank != 0:
        return logger
    formatter = logging.Formatter(
        "%(name)s:%(levelname)s:%(asctime)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    logger.addHandler(stream)
    return logger


def checkpoint_model(
    local_rank,
    global_rank,
    model,
    optimizer,
    scheduler,
    scaler,
    config,
    epoch,
    loss,
    task_weights,
    save,
    logger,
    debug_mode,
):
    if local_rank != 0:
        return
    if args.finetune is None:
        with open(os.path.join(args.datadir, "list_tag"), "r") as f:
            list_tag = f.read()
    # save model
    if save:
        logger.info("checkpointing model")
        if args.finetune is None:
            checkpoint = {
                "model": model.module._orig_mod.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "config": config,
                "epoch": epoch,
                "loss": loss,
            }
            torch.save(checkpoint, f"{args.datadir}/transformer.{args.modeltype}.pt")
            if global_rank == 0 and not debug_mode:
                cmd = f"rclone --retries=10 copyto {args.datadir}/transformer.{args.modeltype}.pt r2:rsys/database/training/{list_tag}/transformer.{args.modeltype}.pt"
                os.system(f"{cmd} &")
        else:
            checkpoint = {
                "model": model.module._orig_mod.state_dict(),
                "config": config,
                "epoch": epoch,
                "loss": loss,
            }
            torch.save(
                checkpoint,
                f"{args.datadir}/transformer.{args.modeltype}.{args.finetune_medium}.finetune.pt",
            )
    # save metrics
    names = [f"{m}.{metric}" for m in ALL_MEDIUMS for metric in ALL_METRICS]
    if args.finetune is None:
        csv_fn = f"{args.datadir}/transformer.{args.modeltype}.csv"
        create_csv = not os.path.exists(csv_fn)
    else:
        csv_fn = f"{args.datadir}/transformer.{args.modeltype}.{args.finetune_medium}.finetune.csv"
        create_csv = epoch < 0
    if create_csv:
        with open(csv_fn, "w") as f:
            f.write(",".join(["epoch", "loss"] + names) + "\n")
    with open(csv_fn, "a") as f:
        vals = [epoch, wsum(loss, task_weights)] + loss
        f.write(",".join([str(x) for x in vals]) + "\n")
    if global_rank == 0 and not debug_mode and args.finetune is None:
        cmd = f"rclone --retries=10 copyto {args.datadir}/transformer.{args.modeltype}.csv r2:rsys/database/training/{list_tag}/transformer.{args.modeltype}.csv"
        os.system(f"{cmd} &")


def upload(global_rank, logger, debug_mode):
    if global_rank != 0 or args.finetune is not None or debug_mode:
        return
    logger.info("uploading model")
    with open(os.path.join(args.datadir, "list_tag"), "r") as f:
        list_tag = f.read()
    with open(f"{args.datadir}/transformer.{args.modeltype}.finished", "w") as f:
        pass
    for suffix in ["pt", "csv", "finished"]:
        cmd = f"rclone --retries=10 copyto {args.datadir}/transformer.{args.modeltype}.{suffix} r2:rsys/database/training/{list_tag}/transformer.{args.modeltype}.{suffix}"
        os.system(cmd)


def training_config():
    if args.finetune is not None:
        checkpoint = torch.load(
            f"{args.datadir}/transformer.{args.modeltype}.pt", weights_only=False, map_location="cpu"
        )
        config = checkpoint["config"]
        config["learning_rate"] = 2e-4
        config["finetune"] = True
        return config
    num_items = {
        x: int(pd.read_csv(f"{args.datadir}/{y}.csv").matchedid.max()) + 1
        for (x, y) in {0: "manga", 1: "anime"}.items()
    }
    num_distinct_items = {
        x: int(pd.read_csv(f"{args.datadir}/{y}.csv").distinctid.max()) + 1
        for (x, y) in {0: "manga", 1: "anime"}.items()
    }
    min_ts = datetime.datetime.strptime("20000101", "%Y%m%d").timestamp()
    with open(os.path.join(args.datadir, "list_tag"), "r") as f:
        max_ts = datetime.datetime.strptime(f.read().strip(), "%Y%m%d").timestamp()
    config = {
        "num_layers": 8,
        "num_heads": 32,
        "num_kv_heads": 16,
        "embed_dim": 2048,
        "intermediate_dim": None,  # TODO 8192
        "distinctid_dim": 128,
        "max_sequence_length": 1024,
        "vocab_sizes": {
            "0_matchedid": num_items[0],
            "1_matchedid": num_items[1],
            "0_distinctid": num_distinct_items[0],
            "1_distinctid": num_distinct_items[1],
            "status": 9,
        },
        "min_ts": min_ts,
        "max_ts": max_ts,
        "rating_mean": 7.6287384,
        "rating_std": 1.778219,
        "forward": "train",
        "finetune": False,
        "learning_rate": 2e-4,
        "causal": args.modeltype == "causal",
        "mask_rate": 0.15,
    }
    return config


def train():
    init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    local_world_size = int(os.environ["LOCAL_WORLD_SIZE"])
    global_rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    logger = get_logger(local_rank, "transformer")
    logger.setLevel(logging.DEBUG)
    config = training_config()
    debug_mode = False
    if config["finetune"]:
        num_epochs = 8
        local_batch_size = 32 if config["causal"] else 64
        grad_accum_steps = 1
        if world_size == 1:
            # emulate training on a 8 gpu setup with gradient accumulation
            single_gpu_batch_size = 8 if config["causal"] else 8
            assert local_batch_size % single_gpu_batch_size == 0
            grad_accum_steps = (8 * local_batch_size) // single_gpu_batch_size
            local_batch_size = single_gpu_batch_size
    else:
        num_epochs = 8 if config["causal"] else 64
        local_batch_size = 16 if config["causal"] else 64
        grad_accum_steps = 1
        if world_size == 1:
            logger.error("LOCAL DEBUG MODE ENABLED")
            num_epochs = 1
            local_batch_size = 4
            debug_mode = True

    def TransformerDataset(x):
        if config["finetune"]:
            return FinetuneDataset(
                f"{args.datadir}/transformer/{x}",
                local_rank,
                local_world_size,
                local_batch_size,
                shuffle=x == "training",
            )
        else:
            return PretrainDataset(
                f"{args.datadir}/transformer/{x}",
                local_rank,
                local_world_size,
                tokens_per_batch=local_batch_size * config["max_sequence_length"],
            )

    dataloaders = {
        x: DataLoader(
            TransformerDataset(x),
            batch_size=1,
            drop_last=False,
            num_workers=w,
            persistent_workers=True,
            collate_fn=collate,
            worker_init_fn=worker_init_fn,
        )
        for (x, w) in [("training", 8), ("test", 2)]
    }
    model = TransformerModel(config)
    logger.info(
        f"Created model with {sum(p.numel() for p in model.parameters())} parameters"
        f" and {sum(p.numel() for p in model.parameters() if p.requires_grad)} trainable parameters"
    )
    checkpoint = None
    if config["finetune"]:
        model.load_state_dict(
            torch.load(
                args.finetune, weights_only=False, map_location=f"cuda:{local_rank}"
            )["model"],
            strict=False,
        )
    else:
        checkpoint_fn = f"{args.datadir}/transformer.{args.modeltype}.pt"
        if os.path.exists(checkpoint_fn):
            checkpoint = torch.load(
                checkpoint_fn, weights_only=False, map_location=f"cuda:{local_rank}"
            )
            logger.info(f"loading model from epoch {checkpoint['epoch']}")
            model.load_state_dict(checkpoint["model"])
            del checkpoint["model"]
    model = model.to(local_rank)
    model = torch.compile(model)
    model = DDP(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=True,
    )
    optimizer = create_optimizer(model, config)
    scheduler = create_learning_rate_schedule(
        optimizer,
        local_batch_size * world_size * config["max_sequence_length"],
        num_epochs,
    )
    scaler = torch.amp.GradScaler(local_rank)
    if checkpoint is not None:
        logger.info(f"loading optimizer state from epoch {checkpoint['epoch']}")
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        del checkpoint["optimizer"]
        del checkpoint["scaler"]
        del checkpoint["scheduler"]

    stopper = (
        EarlyStopper(patience=1, rtol=0.001)
        if config["finetune"]
        else EarlyStopper(patience=float("inf"), rtol=0)
    )
    task_weights = make_task_weights()
    get_loss = lambda x: evaluate_metrics(local_rank, x, dataloaders["test"])
    initial_loss = get_loss(model)
    logger.info(f"Initial Loss: {wsum(initial_loss, task_weights)}, {initial_loss}")
    stopper(wsum(initial_loss, task_weights))
    starting_epoch = 0 if checkpoint is None else checkpoint["epoch"] + 1
    checkpoint_model(
        local_rank,
        global_rank,
        model,
        optimizer,
        scheduler,
        scaler,
        config,
        starting_epoch - 1,
        initial_loss,
        task_weights,
        config["finetune"],
        logger,
        debug_mode,
    )
    for epoch in range(starting_epoch, num_epochs):
        training_loss = train_epoch(
            local_rank,
            model,
            dataloaders["training"],
            optimizer,
            scheduler,
            scaler,
            task_weights,
            grad_accum_steps,
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
        checkpoint_model(
            local_rank,
            global_rank,
            model,
            optimizer,
            scheduler,
            scaler,
            config,
            epoch,
            test_loss,
            task_weights,
            stopper.save_model,
            logger,
            debug_mode,
        )
        if stopper.early_stop:
            break
    try:
        destroy_process_group()
    except Exception as e:
        logger.info(f"Destroying process group failed with {e}")
    upload(global_rank, logger, debug_mode)


def download(node, num_nodes):
    template = "tag=`rclone lsd r2:rsys/database/training/ | sort | tail -n 1 | awk '{print $NF}'`; rclone --retries=10 copyto r2:rsys/database/training/$tag"
    files = (
        [
            "list_tag",
            f"transformer.{args.modeltype}.pt",
            f"transformer.{args.modeltype}.csv",
        ]
        + [f"transformer/{x}/num_tokens.txt" for x in ["training", "test"]]
        + [f"{m}.csv" for m in ["manga", "anime"]]
    )
    for data in files:
        os.system(f"{template}/{data} {args.datadir}/{data}")
    with open(f"{args.datadir}/list_tag") as f:
        list_tag = f.read()
    for x in ["training", "test"]:
        cmd = f"rclone lsd r2:rsys/database/training/{list_tag}/transformer/{x} | wc -l"
        res = subprocess.run(cmd, capture_output=True, text=True, shell=True)
        parts = int(res.stdout)
        assert parts % num_nodes == 0
        for i in range(parts):
            if i % num_nodes == node:
                os.system(
                    f"rclone --retries=10 copyto r2:rsys/database/training/{list_tag}/transformer/{x}/{i+1} {args.datadir}/transformer/{x}/{i+1}"
                )


parser = argparse.ArgumentParser()
parser.add_argument("--datadir", type=str)
parser.add_argument("--modeltype", type=str)
parser.add_argument("--finetune", type=str, default=None)
parser.add_argument("--finetune_medium", type=int, default=None)
parser.add_argument("--download", metavar="N", type=int, nargs="+", default=None)
args = parser.parse_args()

if __name__ == "__main__":
    if args.download is not None:
        node, num_nodes = args.download
        download(node, num_nodes)
    else:
        train()