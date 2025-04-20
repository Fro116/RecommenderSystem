import h5py
import msgpack
import numpy as np
import torch

datadir = "../../data/finetune"


def register_baseline(medium):
    metric = "rating"
    with open(f"{datadir}/baseline.{metric}.{medium}.msgpack", "rb") as f:
        baseline = msgpack.unpackb(f.read(), strict_map_key=False)
        return {
            f"baseline.{medium}.{metric}.weight": np.array(baseline["weight"]),
            f"baseline.{medium}.{metric}.bias": np.array(baseline["bias"]),
        }


def register_bagofwords(medium):
    metric = "rating"
    fn = f"{datadir}/bagofwords.{medium}.{metric}.finetune.pt"
    d = torch.load(fn, weights_only=True, map_location="cpu")
    return {
        f"bagofwords.{medium}.{metric}.weight": d["classifier.weight"].numpy(),
        f"bagofwords.{medium}.{metric}.bias": d["classifier.bias"].numpy(),
    }


def register_transformer(medium):
    fn = f"{datadir}/transformer.{medium}.finetune.pt"
    d = torch.load(fn, weights_only=True, map_location="cpu")
    m = medium
    watch = 2 * m
    rating = 2 * m + 1
    return {
        f"transformer.{m}.embedding": d[f"classifier.{watch}.0.weight"].numpy(),
        f"transformer.{m}.watch.bias": d[f"classifier.{watch}.0.bias"].numpy(),
        f"transformer.{m}.rating.weight.1": d[f"classifier.{rating}.0.weight"].numpy(),
        f"transformer.{m}.rating.bias.1": d[f"classifier.{rating}.0.bias"].numpy(),
        f"transformer.{m}.rating.weight.2": d[f"classifier.{rating}.2.weight"].numpy(),
        f"transformer.{m}.rating.bias.2": d[f"classifier.{rating}.2.bias"].numpy(),
    }

def register():
    ret = {}
    for m in [0, 1]:
        ret.update(register_baseline(m))
        ret.update(register_bagofwords(m))
        ret.update(register_transformer(m))
    with h5py.File(f"{datadir}/model.registry.h5", "w") as hf:
        for k, v in ret.items():
            hf.create_dataset(k, data=v)


datadir = "../../data/finetune"
register()
