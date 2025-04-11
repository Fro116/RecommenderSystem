import h5py
import msgpack
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchtune.models.llama3
import warnings

warnings.filterwarnings("ignore", ".*Initializing zero-element tensors is a no-op.*")


with open("../Training/bagofwords.model.py") as f:
    exec(f.read())

with open("../Training/transformer.model.py") as f:
    exec(f.read())


def register_bagofwords(medium, metric):
    m = BagOfWordsModel(datadir, medium, metric)
    fn = f"{datadir}/bagofwords.{medium}.{metric}.finetune.pt"
    m.load_state_dict(torch.load(fn, weights_only=True, map_location="cpu"))
    return {
        "name": f"bagofwords.{medium}.{metric}",
        "bias": m.classifier.bias.detach().numpy(),
        "weight": m.classifier.weight.detach().numpy(),
    }


def register_baseline(medium, metric):
    with open(f"{datadir}/baseline.rating.{medium}.msgpack", "rb") as f:
        baseline = msgpack.unpackb(f.read(), strict_map_key=False)
        baseline["bias"] = np.array(baseline["bias"])
        baseline["weight"] = np.array([baseline["weight"]])
        if metric not in ["rating"]:
            baseline["bias"] *= 0
            baseline["weight"] *= 0            
        return {
            "name": f"baseline.{medium}.{metric}",
            "bias": baseline["bias"],
            "weight": baseline["weight"],
        }


def register_transformer(medium, metric):
    # TODO serialize config
    num_items = {
        x: pd.read_csv(f"{datadir}/{y}.csv").matchedid.max() + 1
        for (x, y) in {0: "manga", 1: "anime"}.items()
    }
    config = {
        "num_layers": 4,
        "num_heads": 12,
        "num_kv_heads": 12,
        "embed_size": 768,
        "intermediate_dim": None,
        "max_sequence_length": max_seq_len,
        "vocab_names": [
            "0_matchedid",
            "1_matchedid",
            "rating",
            "status",
            "updated_at",
            "delta_time",
        ],
        "vocab_sizes": [
            num_items[0],
            num_items[1],
            None,
            8,
            None,
            None,
        ],
        "forward": "finetune",
    }
    m = TransformerModel(config)
    fn = f"{datadir}/transformer.{medium}.finetune.pt"
    m.load_state_dict(torch.load(fn, weights_only=True, map_location="cpu"))
    classifier = m.classifier[m.names.index(f"{medium}.{metric}")]
    M1 = classifier[0]
    M2 = classifier[1]
    M = M2.weight @ M1.weight
    b = M2.weight @ M1.bias + M2.bias
    M = M[: num_items[medium], :]
    b = b[: num_items[medium]]
    return {
        "name": f"transformer.{medium}.{metric}",
        "bias": b.detach().numpy(),
        "weight": M.detach().numpy(),
    }


def register():
    mediums = [0, 1]
    metrics = ["watch", "rating", "status"]
    models = []
    for medium in mediums:
        for metric in metrics:
            models.append(register_transformer(medium, metric))
            models.append(register_baseline(medium, metric))
        for metric in ["rating"]:
            models.append(register_bagofwords(medium, metric))

    with h5py.File(f"{datadir}/model.registry.h5", "w") as hf:
        for x in models:
            hf.create_dataset(f'{x["name"]}.bias', data=x["bias"])
            hf.create_dataset(f'{x["name"]}.weight', data=x["weight"])


datadir = "../../data/finetune"
register()
