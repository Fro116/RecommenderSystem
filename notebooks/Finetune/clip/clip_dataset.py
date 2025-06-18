import h5py
import pandas as pd
import torch

datadir = "../../../data/finetune"


def get_transformer_embeddings(modeltype):
    fn = f"{datadir}/transformer.{modeltype}.pt"
    d = torch.load(fn, weights_only=True, map_location="cpu")
    ret = {}
    for m in [0, 1]:
        emb = d["model"][f"item_embedding.matchedid_embeddings.{m}.embedding.weight"]
        for i in range(emb.shape[0]):
            k = f"{modeltype}.{m}.{i}"
            ret[k] = list(emb[i, :].numpy())
    return ret


def save_embeddings():
    ds = [
        get_transformer_embeddings("masked"),
        get_transformer_embeddings("causal"),
    ]
    ret = {}
    for d in ds:
        for k in d:
            ret[k] = d[k]
    with h5py.File(f"{datadir}/clip/input.h5", "w") as hf:
        for k, v in ret.items():
            hf.create_dataset(k, data=v)


save_embeddings()
