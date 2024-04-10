import io
import warnings

import msgpack
import scipy
import zstandard as zstd
from flask import Flask, Response, request
from torch.utils.data import DataLoader, Dataset

exec(open("TrainingAlphas/BagOfWords/bagofwords.py").read())
app = Flask(__name__)


def pack(data):
    return Response(
        zstd.ZstdCompressor().compress(msgpack.packb(data, use_single_float=True)),
        mimetype="application/msgpack",
        headers={"Content-Encoding": "zstd", "Content-Type": "application/msgpack"},
    )


def unpack(response):
    reader = zstd.ZstdDecompressor().stream_reader(io.BytesIO(response.data))
    return msgpack.unpackb(reader.read())


class InferenceDataset(Dataset):
    def __init__(self, data):
        self.inputs = self.process_sparse_matrix(data, "inputs")

    def process_sparse_matrix(self, data, name):
        i = np.array(data[f"{name}_i"]) - 1
        j = np.array(data[f"{name}_j"]) - 1
        v = np.array(data[f"{name}_v"])
        m, n = data[f"{name}_size"]
        return scipy.sparse.coo_matrix((v, (j, i)), shape=(n, m)).tocsr()

    def __len__(self):
        return 1

    def __getitem__(self, i):
        return self.inputs[i, :].astype(np.float32).todense()


def sparse_collate(X):
    return [scipy.sparse.vstack([x[i] for x in X]) for i in range(len(X[0]))]


def load_bagofwords_model(medium, metric):
    device = torch.device("cpu")
    base_dir = os.path.join("alphas", medium, "BagOfWords", "v1", metric)
    source_dir = get_data_path(base_dir)
    model_file = os.path.join(source_dir, "model.pt")
    if not os.path.exists(model_file):
        return None
    config_file = os.path.join(source_dir, "config.json")
    config = create_training_config(config_file, "inference")
    model = BagOfWordsModel(config)
    model.load_state_dict(load_model(source_dir, map_location="cpu"))
    model = model.to(device)
    return model.eval()


def detach(x):
    return list(x.to("cpu").to(torch.float32).numpy().flatten().astype(float))


def compute_embeddings(data, medium, metric):
    dataloader = DataLoader(
        InferenceDataset(data),
        batch_size=1,
        shuffle=False,
    )
    embeddings = {}
    with torch.no_grad():
        name = f"{medium}_{metric}"
        embeddings[name] = detach(
            MODELS[(medium, metric)](
                *next(iter(dataloader)),
                None,
                None,
                mask=False,
                evaluate=False,
                inference=True,
            )
        )
    return embeddings


def precompile():
    for k, v in MODELS.items():
        if v is None:
            continue
        medium, metric = k
        data = {
            "inputs_i": [],
            "inputs_j": [],
            "inputs_v": [],
            "inputs_size": [v.model[0].weight.shape[1], 1],
        }        
        compute_embeddings(data, medium, metric)


ALL_METRICS = ["rating", "watch", "plantowatch", "drop"]
ALL_MEDIUMS = ["manga", "anime"]
MODELS = {
    (x, y): load_bagofwords_model(x, y) for x in ALL_MEDIUMS for y in ALL_METRICS
}
precompile()


@app.route("/query", methods=["POST"])
def query():
    data = unpack(request)
    medium = request.args.get("medium", type=str)
    metric = request.args.get("metric", type=str)    
    return pack(compute_embeddings(data, medium, metric))


@app.route("/wake")
def wake():
    return pack({"success": True})


if __name__ == "__main__":  
    app.run()