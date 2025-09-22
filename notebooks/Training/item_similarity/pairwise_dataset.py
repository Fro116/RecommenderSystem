import json
import h5py
import pandas as pd
import torch
import torch.nn as nn
import torchtune.models.llama3
import numpy as np

datadir = "../../../data/training"

with open("../transformer.model.py") as f:
    exec(f.read())

def get_transformer_embeddings(modeltype):
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
    embs = model.item_embedding(torch.arange(0, n_0 + n_1))
    return {
        f"{modeltype}.0": embs[:n_0, :].detach().numpy(),
        f"{modeltype}.1": embs[n_0:n_0+n_1, :].detach().numpy(),
    }

def get_num_items(medium):
    m = {0: "manga", 1: "anime"}[medium]
    col = "matchedid"
    df = pd.read_csv(f"{datadir}/{m}.csv", low_memory=False)
    return int(df[col].max()) + 1
        
def get_text_embeddings():
    ret = {}
    for medium in [0, 1]:
        metadata_embs = np.zeros((get_num_items(medium), 3072))
        analysis_embs = np.zeros((get_num_items(medium), 3072))
        m = {0: "manga", 1: "anime"}[medium]
        with open(f"{datadir}/{m}.json") as f:
            embs = json.load(f)
        for x in embs:
            metadata_embs[x['matchedid'], :] = x['embedding']['metadata']
            analysis_embs[x['matchedid'], :] = x['embedding']['analysis']
        ret[f"metadata.{medium}"] = metadata_embs
        ret[f"analysis.{medium}"] = analysis_embs
    return ret

def save_embeddings():
    ds = [
        get_transformer_embeddings("masked"),
        get_transformer_embeddings("causal"),
        get_text_embeddings(),
    ]
    ret = {}
    for d in ds:
        for k in d:
            ret[k] = d[k]
    with h5py.File(f"{datadir}/item_similarity/features.h5", "w") as hf:
        for k, v in ret.items():
            hf.create_dataset(k, data=v)

save_embeddings()