import warnings

import scipy
from flask import Flask, jsonify, request
from torch.utils.data import DataLoader, Dataset

exec(open("TrainingAlphas/BagOfWords/bagofwords.py").read())
app = Flask(__name__)


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
    config_file = os.path.join(source_dir, "config.json")
    config = create_training_config(config_file, "inference")
    model = BagOfWordsModel(config)
    model.load_state_dict(load_model(source_dir, map_location="cpu"))
    model = model.to(device)
    return model.eval()


def detach(x):
    return list(x.to("cpu").to(torch.float32).numpy().flatten().astype(float))


def compute_embeddings(data, medium):
    dataloader = DataLoader(
        InferenceDataset(data),
        batch_size=1,
        shuffle=False,
    )
    embeddings = {}
    for metric in ALL_METRICS:
        with torch.no_grad():
            name = f"{medium}_{metric}"
            embeddings[name] = detach(
                MODELS[name](
                    *next(iter(dataloader)),
                    None,
                    None,
                    mask=False,
                    evaluate=False,
                    inference=True,
                )
            )
    return embeddings


ALL_METRICS = ["rating", "watch", "plantowatch", "drop"]
ALL_MEDIUMS = ["manga", "anime"]
MODELS = {
    f"{x}_{y}": load_bagofwords_model(x, y) for x in ALL_MEDIUMS for y in ALL_METRICS
}


@app.route("/query", methods=["POST"])
def query():
    data = request.get_json()
    medium = request.args.get("medium", type=str)
    return jsonify(compute_embeddings(data, medium))


@app.route("/wake")
def wake():
    return jsonify({"success": True})


if __name__ == "__main__":
    app.run()