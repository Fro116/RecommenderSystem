import os
import json
import math

import numpy as np
import torch
import torch.nn as nn

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
        if not torch.is_nonzero(weights.sum()):
            return torch.tensor([0.0], device = embed.get_device(), requires_grad=True)
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
            return torch.tensor([0.0], device = embed.get_device(), requires_grad=True)
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
def get_batch_size():
    gpu_mem = int(
        round(
            torch.cuda.get_device_properties(torch.device("cuda")).total_memory
            / 2**30
        )
    )
    gpu_mult = max(round(gpu_mem / 20), 1)
    batch_size = 16
    return batch_size * gpu_mult


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
        "peak_learning_rate": 1e-4 if config["mode"] == "pretrain" else 1e-6,
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


# I/O
def get_data_path(file):
    path = os.getcwd()
    while os.path.basename(path) != "notebooks":
        path = os.path.dirname(path)
    path = os.path.dirname(path)
    return os.path.join(path, "data", file)


def load_model(fn, map_location=None):
    state_dict = torch.load(fn, map_location=map_location)
    compile_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(compile_prefix):
            state_dict[k[len(compile_prefix):]] = state_dict.pop(k)        
    return state_dict