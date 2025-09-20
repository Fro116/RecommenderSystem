import json
import h5py
import pandas as pd
import torch
import torch.nn as nn
import torchtune.models.llama3
import numpy as np

datadir = "../../../data/finetune"

with open("../Training/transformer.model.py") as f:
    exec(f.read())

def register_transformer(modeltype):
    device = "cpu"
    checkpoint = torch.load(
        f"{datadir}/transformer.{modeltype}.pt",
        weights_only=False,
        map_location=device,
    )
    config = checkpoint["config"]
    config["forward"] = "inference"
    model = TransformerModel(config)
    model.load_state_dict(checkpoint["model"])
    model = model.to(device)
    model.eval()
    n_0 = model.config["vocab_sizes"]["0_matchedid"]
    n_1 = model.config["vocab_sizes"]["1_matchedid"]
    embs = model.item_embedding(torch.arange(0, n_0 + n_1 - 1))
    return {
        f"transformer.{modeltype}.0.watch.weight": embs[:n_0, :].detach().numpy(),
        f"transformer.{modeltype}.1.watch.weight": embs[n_0:(n_0 + n_1), :].detach().numpy(),
        f"transformer.{modeltype}.0.rating_mean": config['rating_mean'],
        f"transformer.{modeltype}.1.rating_mean": config['rating_mean'],
    }

datadir = "../../data/finetune"


def register():
    ret = {}
    for modeltype in ["causal", "masked"]:
        ret.update(register_transformer(modeltype))
    with h5py.File(f"{datadir}/model.registry.h5", "w") as hf:
        for k, v in ret.items():
            hf.create_dataset(k, data=v)


register()
