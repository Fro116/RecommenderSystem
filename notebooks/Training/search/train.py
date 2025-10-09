import argparse
import glob
import logging

import h5py
import hdf5plugin
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, IterableDataset
from tqdm import tqdm


class SearchDataset(IterableDataset):
    def __init__(self, datadir, datasplit, batch_size, shuffle):
        self.fns = sorted(glob.glob(f"{datadir}/{datasplit}.*.h5"))
        self.batch_size = batch_size
        self.shuffle = shuffle

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
            with h5py.File(fn) as f:
                d = {}
                for k in f:
                    d[k] = f[k][:]
            mask = d["mediums"] == args.medium  # TODO cross medium
            for k in d:
                d[k] = d[k][mask]
            d["weight"] = np.sqrt(d["counts"])
            del d["counts"]
            N = d["matchedids"].shape[0]
            idxs = list(range(len(d["matchedids"])))
            if self.shuffle:
                np.random.shuffle(idxs)
                while len(idxs) % self.batch_size != 0:
                    idxs.append(np.random.choice(idxs))
            idxs = [
                idxs[i : i + self.batch_size]
                for i in range(0, len(idxs), self.batch_size)
            ]
            for idx in idxs:
                yield {k: v[idx, ...] for k, v in d.items()}


def worker_init_fn(worker_id):
    np.random.seed(torch.utils.data.get_worker_info().seed % 2**32)


def collate(data):
    assert len(data) == 1
    ret = {}
    for k, v in data[0].items():
        ret[k] = torch.tensor(v)
    return ret


def to_device(data):
    d = {}
    for k in data:
        d[k] = data[k].to(args.device)
    return d


class SearchModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        n_items = sum(config["vocab_sizes"].values())
        self.retrieval_embeddings = nn.Embedding(n_items, 2048)
        # self.ranking_embeddings = nn.Embedding(n_items, 2048)
        # self.analysis_embeddings = nn.Embedding(n_items, 3072)
        # self.metadata_embeddings = nn.Embedding(n_items, 3072)
        self.logit_scale = nn.Parameter(torch.ones([])) # TODO # nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.encoder = nn.Linear(2048, 3072, bias=False)

    def get_temperature(self):
        return float(self.logit_scale.detach())

    def load_pretrained_embeddings(self, filepath):
        with h5py.File(filepath, "r") as f:
            self.retrieval_embeddings.weight.data.copy_(
                torch.tensor(np.vstack([f[f"masked.{m}"][:] for m in [0, 1]]))
            )
            # self.ranking_embeddings.weight.data.copy_(
            #     torch.tensor(np.vstack([f[f"causal.{m}"][:] for m in [0, 1]]))
            # )
            # self.analysis_embeddings.weight.data.copy_(
            #     torch.tensor(np.vstack([f[f"analysis.{m}"][:] for m in [0, 1]]))
            # )
            # self.metadata_embeddings.weight.data.copy_(
            #     torch.tensor(np.vstack([f[f"metadata.{m}"][:] for m in [0, 1]]))
            # )
        self.retrieval_embeddings.weight.requires_grad = False
        # self.ranking_embeddings.weight.requires_grad = False
        # self.analysis_embeddings.weight.requires_grad = False
        # self.metadata_embeddings.weight.requires_grad = False

    def embed(self):
        return self.encoder(self.retrieval_embeddings.weight)

    def forward(self, batch):
        x = batch["queries"]
        y = batch["matchedids"] + torch.where(
            batch["mediums"] == 1, self.config["vocab_sizes"][0], 0
        )
        w = batch["weight"]
        w = w / w.sum()
        W = self.embed()
        logits = x.matmul(W.t()) * self.logit_scale.exp()
        K = self.config["vocab_sizes"][0]
        logsoft = torch.hstack(
            [
                torch.nn.functional.log_softmax(logits[..., :K], dim=1),
                torch.nn.functional.log_softmax(logits[..., K:], dim=1),
            ]
        )
        loss = (-logsoft[torch.arange(y.size(0)), y] * w).sum() / w.sum()
        return loss


def evaluate_metrics(model, dataloader):
    losses = 0.0
    weights = 0.0
    progress = tqdm(desc="Eval batches", mininterval=1)
    model.eval()
    for data in dataloader:
        with torch.no_grad():
            with torch.amp.autocast(f"cuda:{args.device}", dtype=torch.bfloat16):
                d = to_device(data)
                loss = model(d)
            w = float(d["weight"].sum())
            losses += float(loss) * w
            weights += w
        progress.update()
    progress.close()
    model.train()
    return losses / weights


def train_epoch(
    model,
    dataloader,
    optimizer,
    scheduler,
    scaler,
):
    training_losses = 0.0
    training_weights = 0.0
    progress = tqdm(desc="Training batches", mininterval=1)
    for step, data in enumerate(dataloader):
        with torch.amp.autocast(f"cuda:{args.device}", dtype=torch.bfloat16):
            optimizer.zero_grad(set_to_none=True)
            d = to_device(data)
            loss = model(d)
            w = float(d["weight"].sum())
            training_losses += float(loss.detach()) * w
            training_weights += w
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        progress.update()
    progress.close()
    return training_losses / training_weights


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
    )


class ConstantScheduler(object):
    def __init__(self):
        self.steps = 0

    def __call__(self, epoch):
        self.steps += 1
        return 1


def create_learning_rate_schedule(optimizer):
    return optim.lr_scheduler.LambdaLR(optimizer, ConstantScheduler())


class EarlyStopper:
    def __init__(self, patience, rtol):
        # stops if loss doesn't decrease by rtol in patience epochs
        self.patience = patience
        self.rtol = rtol
        self.counter = 0
        self.stop_score = float("inf")
        self.stop = False
        self.saved_score = float("inf")
        self.save_model = False

    def __call__(self, score):
        assert not self.stop
        if score < self.stop_score * (1 - self.rtol):
            self.counter = 0
            self.stop_score = score
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True
        if score < self.saved_score:
            self.saved_score = score
            self.save_model = True
        else:
            self.save_model = False


def get_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    formatter = logging.Formatter(
        "%(name)s:%(levelname)s:%(asctime)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    logger.addHandler(stream)
    return logger


def checkpoint_model(model, epoch, training_loss, test_loss, save):
    if save:
        checkpoint = {
            "model": model._orig_mod.state_dict(),
            "epoch": epoch,
            "training_loss": training_loss,
            "test_loss": test_loss,
        }
        torch.save(
            checkpoint,
            f"{args.datadir}/search.model.{args.medium}.pt",
        )
    csv_fn = f"{args.datadir}/search.model.{args.medium}.csv"
    create_csv = epoch < 0
    if create_csv:
        with open(csv_fn, "w") as f:
            f.write(",".join(["epoch", "training_loss", "test_loss", "saved"]) + "\n")
    with open(csv_fn, "a") as f:
        vals = [
            epoch,
            training_loss,
            test_loss,
            1 if save else 0,
        ]
        f.write(",".join([str(x) for x in vals]) + "\n")


def get_num_items(medium):
    m = {0: "manga", 1: "anime"}[medium]
    col = "matchedid"
    df = pd.read_csv(f"{args.datadir}/../{m}.csv", low_memory=False)
    return int(df[col].max()) + 1


def training_config():
    config = {
        "vocab_sizes": {x: get_num_items(x) for x in [0, 1]},
        "learning_rate": 3e-4,
        "batch_size": 1024,
    }
    return config


def train():
    torch.cuda.set_device(args.device)
    logger = get_logger("search")
    logger.setLevel(logging.DEBUG)
    config = training_config()
    rng = np.random.default_rng(seed=42)
    num_epochs = 1024
    model = SearchModel(config)
    model.load_pretrained_embeddings(f"{args.datadir}/features.h5")
    logger.info(
        f"Created model with {sum(p.numel() for p in model.parameters())} parameters"
        f" and {sum(p.numel() for p in model.parameters() if p.requires_grad)} trainable parameters"
    )
    model = model.to(args.device)
    model = torch.compile(model)
    dataloaders = {
        x: DataLoader(
            SearchDataset(args.datadir, x, config["batch_size"], x == "training"),
            batch_size=1,
            shuffle=False,
            drop_last=False,
            num_workers=2,
            persistent_workers=True,
            collate_fn=collate,
            worker_init_fn=worker_init_fn,
        )
        for x in ["training", "test"]
    }
    optimizer = create_optimizer(model, config)
    scheduler = create_learning_rate_schedule(optimizer)
    scaler = torch.amp.GradScaler(args.device)
    early_stopper = EarlyStopper(patience=5, rtol=0.001)
    get_loss = lambda x, y: evaluate_metrics(x, dataloaders[y])
    training_loss = float("inf")
    test_loss = get_loss(model, "test")
    logger.info(f"Epoch: -1, Test Loss: {test_loss}")
    early_stopper(test_loss)
    checkpoint_model(model, -1, training_loss, test_loss, True)
    best_losses = (training_loss, test_loss)
    for epoch in range(num_epochs):
        training_loss = train_epoch(model, dataloaders["training"], optimizer, scheduler, scaler)
        logger.info(f"Epoch: {epoch}, Training Loss: {training_loss}")
        test_loss = get_loss(model, "test")
        logger.info(f"Epoch: {epoch}, Test Loss: {test_loss}")
        early_stopper(test_loss)
        if early_stopper.save_model:
            best_losses = (training_loss, test_loss)
        checkpoint_model(
            model, epoch, training_loss, test_loss, early_stopper.save_model
        )
        if early_stopper.stop:
            break
    logger.info(f"Best losses: {best_losses}")


def generate_embeddings():
    torch.cuda.set_device(args.device)
    config = training_config()
    checkpoint_path = f"{args.datadir}/search.model.{args.medium}.pt"
    model = SearchModel(config)
    checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))
    model.load_state_dict(checkpoint["model"])
    model.to(args.device)
    model = torch.compile(model)
    model.eval()
    output_path = f"{args.datadir}/output.embeddings.{args.medium}.h5"
    device = args.device
    temperature = model.get_temperature()
    K = model.config["vocab_sizes"][0]
    with torch.no_grad():
        W = model.embed().to("cpu").numpy()
    if args.medium == 0:
        W = W[:K, ...]
    elif args.medium == 1:
        W = W[K:, ...]
    else:
        assert False
    with h5py.File(output_path, "w") as hf:
        key = f"search.{args.medium}"
        hf.create_dataset(key, data=W)
        hf.create_dataset("temperature", data=np.array([temperature]))

parser = argparse.ArgumentParser()
parser.add_argument("--datadir", type=str)
parser.add_argument("--device", type=int)
parser.add_argument("--medium", type=int)
args = parser.parse_args()

if __name__ == "__main__":
    print(f"[SEARCH] training medium {args.medium}")
    train()
    generate_embeddings()
