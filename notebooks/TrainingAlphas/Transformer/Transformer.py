import json
import math
import os

import numpy as np
import torch
import torch.nn as nn

ALL_METRICS = ["rating", "watch", "plantowatch", "drop"]
ALL_MEDIUMS = ["manga", "anime"]


# Models
class DiscreteEmbed(nn.Module):
    def __init__(self, vocab_size, embed_size):
        super(DiscreteEmbed, self).__init__()
        self.embedding = nn.Sequential(
            nn.Embedding(vocab_size, embed_size),
            nn.LayerNorm(embed_size),
        )

    def forward(self, x):
        return self.embedding(x)


class ContinuousEmbed(nn.Module):
    def __init__(self, vocab_size, embed_size, dropout):
        super(ContinuousEmbed, self).__init__()
        hidden_size = math.ceil(embed_size / 4)
        self.embedding_with_weightdecay = nn.Sequential(
            nn.Linear(1, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, embed_size),
            nn.LayerNorm(embed_size),
        )
        self.scale = 1 / vocab_size

    def forward(self, x):
        return self.embedding_with_weightdecay(x * self.scale * 2 - 1)


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

        # create embeddings
        embeddings = []
        for size, dtype in zip(config["vocab_sizes"], config["vocab_types"]):
            if dtype == "int":
                embeddings.append(DiscreteEmbed(size, config["embed_size"]))
            elif dtype == "float":
                embeddings.append(
                    ContinuousEmbed(size, config["embed_size"], config["dropout"])
                )
            elif dtype == "none":
                continue
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
                config["embed_size"],
                config["vocab_sizes"][i],
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
        if config["mode"] == "pretrain":
            self.forward = self.pretrain_forward
        elif config["mode"] == "finetune":
            self.forward = self.finetune_forward
        else:
            assert False

    def mse(self, x, y, w):
        return (torch.square(x - y) * w).sum() / w.sum()

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

    def pretrain_lossfn(self, embed, lossfn, classifier, positions, labels, weights):
        if not torch.is_nonzero(weights.sum()):
            return torch.tensor(
                [0.0], device=embed.get_device(), requires_grad=embed.requires_grad
            )
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
        if not torch.is_nonzero(weights.sum()):
            return torch.tensor(
                [0.0], device=embed.get_device(), requires_grad=embed.requires_grad
            )
        preds = classifier(embed)
        return lossfn(preds, labels, weights)

    def finetune_forward(
        self, inputs, mask, positions, labels, weights, users, embed_only=False
    ):
        e = self.embed(inputs)
        e = self.transformers(e, mask)
        e = e[range(len(positions)), positions, :]
        if embed_only:
            return e
        losses = tuple(
            self.finetune_lossfn(e, *args)
            for args in zip(self.lossfns, self.classifier, labels, weights)
        )
        return losses


# Configs
def get_batch_size(split, mode):
    if mode == "pretrain":
        return 256
    elif mode == "finetune":
        return 16
    else:
        assert False


def create_training_config(outdir):
    config = {}

    # setup data config
    partition = 0
    while os.path.exists(f"{outdir}/{partition}/config.json"):
        c = json.load(open(f"{outdir}/{partition}/config.json", "r"))
        partition += 1
        data_config = {
            "vocab_names": c["vocab_names"],
            "vocab_sizes": c["vocab_sizes"],
            "vocab_types": c["vocab_types"],
            "media_sizes": c["media_sizes"],
            "max_sequence_length": c["max_sequence_length"],
            "mode": c["mode"],
            "chunk_size": c["batch_size"],
            "splits": [
                x for x in ["training", "validation", "test"] if f"{x}_epoch_size" in c
            ],
        }
        for k in data_config:
            if k not in config:
                config[k] = data_config[k]
            else:
                assert config[k] == data_config[k], k
        for x in config["splits"]:
            k1 = f"{x}_epoch_size"
            k2 = k1 + "s"
            if k2 not in config:
                config[k2] = []
            config[k2].append(c[k1])

    # setup model config
    config["num_layers"] = 4
    config["embed_size"] = 768
    config["num_attention_heads"] = 12
    config["learning_rate"] = 2e-4 if config["mode"] == "pretrain" else 1e-6
    config["adam_beta"] = (0.9, 0.95)
    config["weight_decay"] = 0.1
    config["num_epochs"] = 128 if config["mode"] == "pretrain" else 16
    config["warmup_ratio"] = 0.01
    config["dropout"] = 0.1
    config["clip_norm"] = 1.0
    for x in config["splits"]:
        config[f"{x}_batch_size"] = get_batch_size(x, config["mode"])
    return config


def create_model_config(config):
    return {
        "dropout": config["dropout"],
        "activation": "gelu",
        "num_layers": config["num_layers"],
        "embed_size": config["embed_size"],
        "max_sequence_length": config["max_sequence_length"],
        "vocab_sizes": config["vocab_sizes"],
        "vocab_types": config["vocab_types"],
        "media_sizes": config["media_sizes"],
        "num_attention_heads": config["num_attention_heads"],
        "intermediate_size": config["embed_size"] * 4,
        "mode": config["mode"],
    }


# I/O
def get_data_path(file):
    path = os.getcwd()
    while os.path.basename(path) != "notebooks":
        path = os.path.dirname(path)
    path = os.path.dirname(path)
    return os.path.join(path, "data", file)


def load_model(fn, map_location=None):
    state_dict = torch.load(fn, map_location=map_location)
    compile_prefix = "_orig_mod."
    for k, v in list(state_dict.items()):
        if k.startswith(compile_prefix):
            state_dict[k[len(compile_prefix) :]] = state_dict.pop(k)
    return state_dict