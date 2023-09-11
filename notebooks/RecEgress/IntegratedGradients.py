exec(open("../TrainingAlphas/Transformer/Transformer.py").read())

import argparse
import json

import h5py
from captum.attr import IntegratedGradients
from torch.utils.data import DataLoader, Dataset


class InferenceDataset(Dataset):
    def __init__(self, file):
        self.filename = file
        f = h5py.File(file, "r")
        self.length = f["anime"].shape[0]
        self.embeddings = [
            f["anime"][:] - 1,
            f["manga"][:] - 1,
            f["rating"][:].reshape(*f["rating"].shape, 1).astype(np.float32),
            f["timestamp"][:].reshape(*f["timestamp"].shape, 1).astype(np.float32),
            f["status"][:] - 1,
            f["completion"][:].reshape(*f["completion"].shape, 1).astype(np.float32),
            f["position"][:] - 1,
        ]
        self.mask = f["user"][:]

        def process_position(x):
            return x[:].flatten().astype(np.int64) - 1

        self.positions = process_position(f["positions"])

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        embeds = tuple(x[i, :] for x in self.embeddings)
        mask = self.mask[i, :]
        mask = mask.reshape(1, mask.size) != mask.reshape(mask.size, 1)
        positions = self.positions[i]
        return embeds, mask, positions


def to_device(data, device):
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device)


def get_model(medium, task):
    device = torch.device("cuda")
    source_dir = get_data_path(
        os.path.join("alphas", medium, task, "Transformer", "v1")
    )
    model_file = os.path.join(source_dir, "model.pt")
    config_file = os.path.join(source_dir, "config.json")
    training_config = create_training_config(config_file, 1)
    model_config = create_model_config(training_config)
    model = TransformerModel(model_config)
    model.load_state_dict(load_model(model_file, map_location="cpu"))
    model = model.to(device)
    model.eval()
    return model


def get_data(username, medium, task):
    device = torch.device("cuda")
    outdir = get_data_path(
        f"recommendations/{username}/alphas/{medium}/{task}/Transformer/v1"
    )
    dataloader = DataLoader(
        InferenceDataset(os.path.join(outdir, "inference.h5")),
        batch_size=1,
        shuffle=False,
    )
    it = iter(dataloader)
    data = tuple(next(it))
    return to_device(data, device)


def get_baseline(medium, task, inputs, positions):
    # load configs
    source_dir = get_data_path(
        os.path.join("alphas", medium, task, "Transformer", "v1")
    )
    config_file = os.path.join(source_dir, "config.json")
    config = json.load(open(config_file, "r"))
    training_config = create_training_config(config_file, 1)

    # get mask tokens
    assert len(training_config["vocab_types"]) == 8
    assert training_config["vocab_types"][6] == None
    mask_tokens = config["mask_tokens"][:6] + config["mask_tokens"][7:]
    empty_tokens = config["empty_tokens"][:6] + config["empty_tokens"][7:]
    vocab_types = (
        training_config["vocab_types"][:6] + training_config["vocab_types"][7:]
    )
    for i in range(len(vocab_types)):
        mask_tokens[i] -= vocab_types[i] == int
        empty_tokens[i] -= vocab_types[i] == int

    # mask out item inputs
    end_of_seq_pos = positions.cpu().numpy()[0]
    baseline_inputs = [x.clone() for x in inputs]
    for i in [0, 1]:
        for j in range(1, end_of_seq_pos):
            if inputs[i][0, j] != empty_tokens[i]:
                baseline_inputs[i][0, j] = mask_tokens[i]
    return baseline_inputs


def utility_model(embedding, model=None, positions=None, mask=None, coefs=None, medium=None):
    hidden = model.transformers(embedding, mask)
    output = hidden[range(len(positions)), positions, :]
    if medium == "anime":
        idxp = 0
        idxr = 1
    elif medium == "manga":
        idxp = 2
        idxr = 3
    else:
        assert False
    p = model.classifier[idxp](output)
    r = model.classifier[idxr](output)
    # transform into MLE utility
    p = torch.exp(p)
    r = torch.relu((r + 10) / 10)
    u = (
        np.exp(coefs[0]) * p
        + np.exp(coefs[1]) * r
        + np.exp(coefs[2]) * torch.log(p)
        + np.exp(coefs[3]) * torch.log(r)
    )
    return u


def cpu(x):
    return x.detach().cpu().numpy().squeeze().tolist()


def compute_attributions(username, medium, task, coefs, items):
    model = get_model(medium, task)
    inputs, mask, positions = get_data(username, medium, task)
    baseline_inputs = get_baseline(medium, task, inputs, positions)

    embedding = model.embed(inputs)
    baseline_embedding = model.embed(baseline_inputs)

    ig = IntegratedGradients(utility_model)
    attributions = {}
    for item in items:
        attrs = ig.attribute(
            embedding,
            baseline_embedding,
            target=item,
            internal_batch_size=16,
            additional_forward_args=(model, positions, mask, coefs, medium),
            n_steps=20,
        )
        attributions[item] = cpu(attrs.sum(dim=2))
    return attributions, cpu(inputs[0]), cpu(inputs[1])


def save_attributions(username, medium, task, coefs, items):
    cache_fn = (
        f"../../data/recommendations/{username}/"
        f"explanations/{medium}/integrated_gradients.json"
    )
    cache = {"attributions": {}}
    if os.path.exists(cache_fn):
        try:
            with open(cache_fn, "r") as f:
                cache = json.load(f)
        except:
            # cache file was corrupted
            pass

    new_items = set(items) - {int(x) for x in cache["attributions"].keys()}
    if new_items:
        new_attrs, anime, manga = compute_attributions(
            username, medium, task, coefs, new_items
        )
        cache["anime"] = anime
        cache["manga"] = manga
        cache["attributions"] |= new_attrs
        os.makedirs(os.path.dirname(cache_fn), exist_ok=True)
        with open(cache_fn, "w") as f:
            json.dump(cache, f)


parser = argparse.ArgumentParser(description="IntegratedGradients")
parser.add_argument("--username", type=str, help="username")
parser.add_argument("--medium", type=str, help="medium")
parser.add_argument("--task", type=str, help="task")
parser.add_argument(
    "--coefs", nargs="+", type=float, help="coefficients for the utility model"
)
parser.add_argument(
    "--items", nargs="+", type=int, help="list of items to generate explanations for"
)

args = parser.parse_args()
if __name__ == "__main__":
    save_attributions(args.username, args.medium, args.task, args.coefs, args.items)