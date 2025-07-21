import json

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


def get_item_text_embeddings():
    ret = {}
    for medium in [0, 1]:
        m = {0: "manga", 1: "anime"}[medium]
        df = pd.read_csv(f"{datadir}/{m}.csv")
        with open(f"{datadir}/item_text_embeddings.{medium}.json") as file:
            embs = json.load(file)
        embs = {(x["source"], x["itemid"]): x["embedding"] for x in embs}
        matchedid_embs = {}
        for _, x in df.iterrows():
            k = (x.source, x.itemid)
            if k in embs:
                v = embs[k]
                if len(v) == 2:  # TODO remove on the version
                    v = v[0]
                matchedid_embs[x.matchedid] = v
        emb_dim = len(next(iter(matchedid_embs.values())))
        for i in range(max(df.matchedid)+1):
            k = f"item_text.{medium}.{i}"
            ret[k] = matchedid_embs.get(i, [0] * emb_dim)
    return ret


def save_embeddings():
    ds = [
        get_transformer_embeddings("masked"),
        get_transformer_embeddings("causal"),
        get_item_text_embeddings(),
    ]
    ret = {}
    for d in ds:
        for k in d:
            ret[k] = d[k]
    with h5py.File(f"{datadir}/clip/input.h5", "w") as hf:
        for k, v in ret.items():
            hf.create_dataset(k, data=v)


save_embeddings()