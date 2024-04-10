import warnings

import msgpack
from flask import Flask, request, Response
from torch.utils.data import DataLoader, Dataset

exec(open("TrainingAlphas/Transformer/transformer.py").read())
app = Flask(__name__)


def pack(data):
    return Response(
        msgpack.packb(data, use_single_float=True), mimetype="application/msgpack"
    )


def unpack(response):
    return msgpack.unpackb(response.data)


class InferenceDataset(Dataset):
    def __init__(self, data, vocab_names, vocab_types):
        def process(x, dtype):
            if dtype == "float":
                return np.array(data[x]).reshape(2, -1, 1).astype(np.float32)
            elif dtype == "int":
                return np.array(data[x]).reshape(2, -1).astype(np.int32)
            else:
                assert False

        self.embeddings = [
            process(x, y) for (x, y) in zip(vocab_names, vocab_types) if x != "userid"
        ]
        self.mask = process("userid", "int")
        self.length = self.mask.shape[0]
        self.positions = np.array(data["positions"]).astype(np.int64)

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        # a true value means that the tokens will not attend to each other
        mask = self.mask[i, :]
        mask = mask.reshape(1, mask.size) != mask.reshape(mask.size, 1)
        user = self.mask[i, 0]
        embeds = [x[i, :] for x in self.embeddings]
        positions = self.positions[i]
        return embeds, mask, positions, np.array([]), np.array([]), user


def load_transformer_model(medium):
    device = torch.device("cpu")
    source_dir = get_data_path(os.path.join("alphas", medium, "Transformer", "v1"))
    model_file = os.path.join(source_dir, "model.pt")
    if not os.path.exists(model_file):
        return None    
    training_config = create_training_config(
        get_data_path(os.path.join("alphas", "all", "Transformer", "v1"))
    )
    training_config["mode"] = "finetune"
    warnings.filterwarnings("ignore")
    model_config = create_model_config(training_config)
    model = TransformerModel(model_config)
    model.load_state_dict(load_model(model_file, map_location="cpu"))
    model = model.to(device)
    return training_config, model.eval()


def detach(x):
    x = x.to("cpu").to(torch.float32).numpy().astype(float)
    return [list(x[0, :]), list(x[1, :])]


def compute_embeddings(data, medium):
    config, model = MODELS[medium]
    dataloader = DataLoader(
        InferenceDataset(
            data,
            config["vocab_names"],
            config["vocab_types"],
        ),
        batch_size=2,
        shuffle=False,
    )
    with torch.no_grad():
        data = next(iter(dataloader))
        output = model(*data, inference=True)        
        embeddings = [detach(y) for y in output]            
    names = [f"{medium}_{metric}" for medium in ALL_MEDIUMS for metric in ALL_METRICS]
    d = {x: y for (x, y) in zip(names, embeddings)}
    return {x: y for (x, y) in d.items() if medium in x}


def precompile():
    for k, v in MODELS.items():
        if v is None:
            continue
        config, model = v
        data = {}
        size = 4
        assert size < int(config["max_sequence_length"])
        for name, dtype in zip(config["vocab_names"], config["vocab_types"]):
            if dtype == "int":
                data[name] = np.zeros((2, size), dtype=np.int32)
            elif dtype == "float":
                data[name] = np.zeros((2, size), dtype=np.float32)
            elif dtype == "none":
                assert name == "userid"
                data[name] = np.zeros((2, size), dtype=np.int32)
            else:
                assert False
        data["positions"] = np.ones((2, 1), dtype=np.int32)
        compute_embeddings(data, k)


MODELS = {x: load_transformer_model(x) for x in ["manga", "anime"]}
precompile()


@app.route("/query", methods=["POST"])
def query():
    data = unpack(request)
    medium = request.args.get("medium", type=str)
    return pack(compute_embeddings(data, medium))


@app.route("/wake")
def wake():
    return pack({"success": True})


if __name__ == "__main__":
    app.run()