from flask import Flask, jsonify, request

app = Flask(__name__)

exec(open("../TrainingAlphas/BagOfWords/bagofwords.py").read())

import warnings

import h5py
import hdf5plugin
import scipy
from torch.utils.data import DataLoader, Dataset


class InferenceDataset(Dataset):
    def __init__(self, file):
        self.filename = file
        f = h5py.File(file, "r")
        self.length = np.array(f[f"epoch_size"]).item()
        self.inputs = self.process_sparse_matrix(f, "inputs")
        self.users = f["users"][:]
        f.close()

    def process_sparse_matrix(self, f, name):
        i = f[f"{name}_i"][:] - 1
        j = f[f"{name}_j"][:] - 1
        v = f[f"{name}_v"][:]
        m, n = f[f"{name}_size"][:]
        return scipy.sparse.coo_matrix((v, (j, i)), shape=(n, m)).tocsr()

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        X = self.inputs[i, :]
        user = self.users[i]
        return X, user


def to_sparse_tensor(csr):
    return torch.sparse_csr_tensor(csr.indptr, csr.indices, csr.data, csr.shape)


def get_device():
    return "cpu"


def to_device(data, device):
    return [to_sparse_tensor(x).to(device).to_dense() for x in data[:-1]]


def sparse_collate(X):
    return [scipy.sparse.vstack([x[i] for x in X]) for i in range(len(X[0]))]


def record_predictions(model, outdir, dataloader):
    user_batches = []
    embed_batches = []
    model.eval()
    device = get_device()
    for data in dataloader:
        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                x = (
                    model(
                        *to_device(data, device),
                        None,
                        None,
                        mask=False,
                        evaluate=False,
                        inference=True,
                    )
                    .to("cpu")
                    .to(torch.float32)
                    .numpy()
                )
                users = data[-1].todense()
                user_batches.append(users)
                embed_batches.append(x)

    f = h5py.File(os.path.join(outdir, "predictions.h5"), "w")
    f.create_dataset("users", data=np.vstack(user_batches))
    f.create_dataset("predictions", data=np.vstack(embed_batches))
    f.close()


def load_bagofwords_model(medium, metric):
    base_dir = os.path.join("alphas", medium, "BagOfWords", "v1", metric)
    source_dir = get_data_path(base_dir)
    model_file = os.path.join(source_dir, "model.pt")
    config_file = os.path.join(source_dir, "config.json")
    config = create_training_config(config_file, "inference")
    device = get_device()
    model = BagOfWordsModel(config)
    model.load_state_dict(load_model(source_dir, map_location="cpu"))
    model = model.to(device)    
    return model


def save_embeddings(username, source, medium, metric, version):
    warnings.filterwarnings("ignore")
    datadir = get_data_path(
        os.path.join("recommendations", source, username, "alphas", "BagOfWords", "v1")
    )
    dataloader = DataLoader(
        InferenceDataset(os.path.join(datadir, "inference.h5")),
        batch_size=16,
        shuffle=False,
        collate_fn=sparse_collate,
    )
    base_dir = os.path.join("alphas", medium, "BagOfWords", "v1", metric)
    outdir = get_data_path(os.path.join("recommendations", source, username, base_dir))
    os.makedirs(outdir, exist_ok=True)
    model = MODELS[(medium, metric)]
    record_predictions(model, outdir, dataloader)


MODELS = {
    (medium, metric): load_bagofwords_model(medium, metric)
    for medium in ["manga", "anime"]
    for metric in ["rating", "watch", "plantowatch", "drop"]
}


@app.route("/query", methods=["GET"])
def query():
    source = request.args.get("source", type=str)
    username = request.args.get("username", type=str)
    medium = request.args.get("medium", type=str)
    metric = request.args.get("metric", type=str)
    version = "v1"
    save_embeddings(username, source, medium, metric, version)
    return jsonify({"result": "Success!"})


if __name__ == "__main__":
    app.run()