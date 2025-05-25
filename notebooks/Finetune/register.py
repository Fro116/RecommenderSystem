import h5py
import torch

datadir = "../../data/finetune"

def register_transformer(modeltype, medium):
    fn = f"{datadir}/transformer.{modeltype}.{medium}.finetune.pt"
    d = torch.load(fn, weights_only=True, map_location="cpu")
    ret = {
        f"transformer.{modeltype}.{medium}.watch.weight": d["model"][f"watch_heads.{medium}.0.weight"].numpy(),
        f"transformer.{modeltype}.{medium}.watch.bias": d["model"][f"watch_heads.{medium}.0.bias"].numpy(),
        f"transformer.{modeltype}.{medium}.rating_mean": d['config']['rating_mean'],
    }
    if modeltype == "masked":
        ret |= {
            f"transformer.{modeltype}.{medium}.rating.weight.1": d["model"][f"rating_head.0.weight"].numpy(),
            f"transformer.{modeltype}.{medium}.rating.bias.1": d["model"][f"rating_head.0.bias"].numpy(),
            f"transformer.{modeltype}.{medium}.rating.weight.2": d["model"][f"rating_head.2.weight"].numpy(),
            f"transformer.{modeltype}.{medium}.rating.bias.2": d["model"][f"rating_head.2.bias"].numpy(),
        }
    return ret


def register():
    ret = {}
    for modeltype in ["causal", "masked"]:
        for m in [0, 1]:
            ret.update(register_transformer(modeltype, m))
    with h5py.File(f"{datadir}/model.registry.h5", "w") as hf:
        for k, v in ret.items():
            hf.create_dataset(k, data=v)


register()
