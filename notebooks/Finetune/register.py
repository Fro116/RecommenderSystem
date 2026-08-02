import json
import h5py
import os
import pandas as pd
import torch
import torch.nn as nn
import numpy as np

datadir = "../../data/finetune"

with open("../Training/transformer.model.py") as f:
    exec(f.read())

def register_transformer():
    device = "cpu"
    checkpoint = torch.load(
        f"{datadir}/transformer.masked.pt",
        weights_only=False,
        map_location=device,
    )
    config = checkpoint["config"]
    config["forward"] = "inference"
    model = RecommenderModel(config)
    model.load_state_dict(checkpoint["model"])
    model = model.to(device)
    model.eval()
    n_0 = model.config["vocab_sizes"]["0_matchedid"]
    n_1 = model.config["vocab_sizes"]["1_matchedid"]
    embs = model.item_embedding(torch.arange(0, n_0 + n_1))
    ret =  {
        "0.watch.weight": embs[:n_0, :].detach().numpy(),
        "1.watch.weight": embs[n_0:(n_0 + n_1), :].detach().numpy(),
        "0.rating_mean": config['rating_mean'],
        "1.rating_mean": config['rating_mean'],
    }
    with h5py.File(f"{datadir}/model.registry.h5", "w") as hf:
        for k, v in ret.items():
            hf.create_dataset(k, data=v)

def dedup_finetune_models():
    files = [
        f"{datadir}/transformer.masked.{m}.{t}.finetune"
        for m in [0, 1]
        for t in ["watch", "rating"]
    ]
    base_fn = f"{datadir}/transformer.masked.finetune.base.pt"
    base = None
    for fn in files:
        ckpt = torch.load(f"{fn}.pt", weights_only=False, map_location="cpu")
        trunk = {k: v for k, v in ckpt["model"].items() if "lora_" not in k}
        lora = {k: v for k, v in ckpt["model"].items() if "lora_" in k}
        assert lora, f"{fn}.pt: no lora keys found"
        assert trunk, f"{fn}.pt: no trunk keys found; is this a full checkpoint?"
        if base is None:
            base = trunk
            torch.save({"model": base}, base_fn)
        else:
            assert set(trunk.keys()) == set(base.keys()), f"{fn}.pt: key mismatch"
            for k, v in trunk.items():
                assert torch.equal(v, base[k]), f"{fn}.pt: {k} differs from base"
        ckpt["model"] = lora
        torch.save(ckpt, f"{fn}.lora.pt")


dedup_finetune_models()
register_transformer()
