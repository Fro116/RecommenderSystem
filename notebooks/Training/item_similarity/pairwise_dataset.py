import h5py
import ijson
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

datadir = "../../../data/training"

with open("../transformer.model.py") as f:
    exec(f.read())


def get_transformer_embeddings():
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
    return {
        "transformer.0": embs[:n_0, :].detach().numpy(),
        "transformer.1": embs[n_0 : n_0 + n_1, :].detach().numpy(),
    }


def get_num_items(medium):
    m = {0: "manga", 1: "anime"}[medium]
    df = pd.read_csv(f"{datadir}/{m}.csv", usecols=["matchedid"])
    return int(df["matchedid"].max()) + 1


def get_content_embeddings():
    ret = {}
    for medium in [0, 1]:
        text_embs = np.zeros((get_num_items(medium), 3072), dtype=np.float32)
        image_embs = np.zeros((get_num_items(medium), 3072), dtype=np.float32)
        m = {0: "manga", 1: "anime"}[medium]
        with open(f"{datadir}/{m}.json", "rb") as f:
            for x in ijson.items(f, "item"):
                text_embs[x["matchedid"], :] = x["text_embedding"]["embedding"]
                image_embs[x["matchedid"], :] = x["image_embedding"]
        ret[f"text.{medium}"] = text_embs
        ret[f"image.{medium}"] = image_embs
    return ret


def save_embeddings():
    ds = [
        get_transformer_embeddings(),
        get_content_embeddings(),
    ]
    ret = {}
    for d in ds:
        for k in d:
            ret[k] = d[k]
    with h5py.File(f"{datadir}/item_similarity/features.h5", "w") as hf:
        for k, v in ret.items():
            hf.create_dataset(k, data=v)


save_embeddings()