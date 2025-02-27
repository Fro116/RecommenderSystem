import msgpack
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

with open("../Training/bagofwords.model.py") as f:
    exec(f.read())


def register_bagofwords(medium, metric):
    m = BagOfWordsModel(datadir, medium, metric)
    fn = f"{datadir}/bagofwords.{medium}.{metric}.finetune.pt"
    m.load_state_dict(torch.load(fn, weights_only=True, map_location="cpu"))
    return {
        "name": f"bagofwords.{medium}.{metric}",
        "bias": m.classifier.bias.detach().numpy().tolist(),
        "weight": m.classifier.weight.detach().numpy().tolist(),
    }


def register_baseline(medium, metric):
    with open(f"{datadir}/baseline.{medium}.msgpack", "rb") as f:
        baseline = msgpack.unpackb(f.read(), strict_map_key=False)
        if metric in ["watch", "plantowatch", "drop"]:
            baseline["bias"] = (np.array(baseline["bias"]) * 0).tolist()
            baseline["weight"] = (np.array(baseline["weight"]) * 0).tolist()
        return {
            "name": f"baseline.{medium}.{metric}",
            "bias": baseline["bias"],
            "weight": [baseline["weight"]],
        }


def register():
    mediums = [0, 1]
    metrics = ["rating", "watch", "plantowatch", "drop"]
    models = []
    for medium in mediums:
        for metric in metrics:
            models.append(register_bagofwords(medium, metric))
            models.append(register_baseline(medium, metric))
    with open(f"{datadir}/model.registry.msgpack", "wb") as f:
        msgpack.pack(models, f, use_single_float=True)


datadir = "../../data/finetune"
register()
