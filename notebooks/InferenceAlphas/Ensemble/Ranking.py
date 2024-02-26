exec(open("../../TrainingAlphas/Ensemble/Ranking.py").read())

import argparse

import h5py
import hdf5plugin
from torch.utils.data import DataLoader, Dataset


class RankingDataset(Dataset):
    def __init__(self, file):
        f = h5py.File(file, "r")
        self.F = np.array(f["features"])
        f.close()

    def __len__(self):
        return self.F.shape[0]

    def __getitem__(self, i):
        return self.F[i, :], np.array([])


def get_device():
    return "cpu"


def to_device(data, device):
    return [x.to(device) for x in data]


def create_model(outdir, device):
    model = RankingModel(None)
    model.load_state_dict(load_model(outdir, map_location="cpu"))
    model = model.to(device)
    return model


def record_predictions(model, outdir, dataloader):
    embed_batches = []
    model.eval()
    device = get_device()
    for data in dataloader:
        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                x = (
                    model(*to_device(data, device), inference=True)
                    .to("cpu")
                    .to(torch.float32)
                    .numpy()
                )
                embed_batches.append(x)
    f = h5py.File(os.path.join(outdir, "predictions.h5"), "w")
    f.create_dataset("predictions", data=np.vstack(embed_batches))
    f.close()


def save_embeddings(outdir, medium):
    base_dir = os.path.join("alphas", medium, "Ranking")
    source_dir = get_data_path(base_dir)
    model_file = os.path.join(source_dir, "model.pt")
    model = create_model(source_dir, get_device())
    dataloader = DataLoader(
        RankingDataset(os.path.join(outdir, "inference.h5")),
        batch_size=1024,
        shuffle=False,
    )
    record_predictions(model, outdir, dataloader)

parser = argparse.ArgumentParser(description="Ranking")
parser.add_argument("--outdir", type=str, help="outdir")
parser.add_argument("--medium", type=str, help="medium")
args = parser.parse_args()    

if __name__ == "__main__":
    save_embeddings(args.outdir, args.medium)