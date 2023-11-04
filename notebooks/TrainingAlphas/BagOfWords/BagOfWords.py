import json
import math
import os

import numpy as np
import torch
import torch.nn as nn


class BagOfWordsModel(nn.Module):
    def __init__(self, config):
        super(BagOfWordsModel, self).__init__()

        self.mask_rate = config["mask_rate"]
        self.input_sizes = config["input_sizes"]
        self.output_size_index = config["output_size_index"] - 1
        self.output_size = self.input_sizes[self.output_size_index]
        self.model = nn.Sequential(
            nn.Linear(sum(self.input_sizes) * 2, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, self.output_size),
        )
        self.content = config["content"]
        if config["content"] == "explicit":
            self.lossfn = self.rating_lossfn
        elif config["content"] == "implicit":
            self.logsoftmax = nn.LogSoftmax(dim=-1)
            self.softmax = nn.Softmax(dim=-1)
            self.lossfn = self.crossentropy_lossfn
        else:
            assert False

    def rating_lossfn(self, x, y, w):
        return (torch.square(x - y) * w).sum() / w.sum()

    def crossentropy_lossfn(self, x, y, w):
        x = self.logsoftmax(x)
        return (-x * y * w).sum() / w.sum()

    def forward(self, inputs, labels, weights, mask, evaluate, predict):
        if predict:
            x = self.model(inputs)
            if self.content == "implicit":
                x = self.softmax(x)
            return x
        if evaluate:
            if self.content == "explicit":
                # return the full path of loss values
                x = self.model(inputs)
                return (
                    self.lossfn(1 * x, labels, weights),
                    self.lossfn(0 * x, labels, weights),
                    self.lossfn(-1 * x, labels, weights),
                )
            elif self.content == "implicit":
                return self.lossfn(self.model(inputs), labels, weights)
        if mask:
            masks = [
                torch.rand(inputs.shape[0], x, device=inputs.device) > self.mask_rate
                for x in self.input_sizes
            ]
            input_mask = torch.cat([masks[0], masks[1], masks[0], masks[1]], 1)
            output_mask = ~masks[self.output_size_index]
            inputs = inputs * input_mask
            weights = weights * output_mask
        return self.lossfn(self.model(inputs), labels, weights)


def create_training_config(config_file, mode):
    config = json.load(open(config_file, "r"))
    config = {
        # model
        "input_sizes": config["input_sizes"],
        "output_size_index": config["output_size_index"],
        "content": config["content"],
        # training
        "peak_learning_rate": 4e-5 if mode == "pretrain" else 4e-6,  # TODO scan
        "weight_decay": 1e-2,
        "mask_rate": config["mask_rate"],
        "batch_size": 2048,
        "mask": mode == "pretrain",
        # data
        "num_data_shards": config["num_shards"],
    }
    return config


# I/O
def get_data_path(file):
    path = os.getcwd()
    while os.path.basename(path) != "notebooks":
        path = os.path.dirname(path)
    path = os.path.dirname(path)
    return os.path.join(path, "data", file)


def load_model(outdir, map_location=None):
    fn = os.path.join(outdir, "model.pt")
    state_dict = torch.load(fn, map_location=map_location)
    compile_prefix = "_orig_mod."
    for k, v in list(state_dict.items()):
        if k.startswith(compile_prefix):
            state_dict[k[len(compile_prefix) :]] = state_dict.pop(k)
    return state_dict