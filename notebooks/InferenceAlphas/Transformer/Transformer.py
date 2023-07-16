exec(open("../../TrainingAlphas/Transformer/Transformer.py").read())

import argparse
import h5py
import warnings
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
        embeds = [x[i, :] for x in self.embeddings]
        mask = self.mask[i, :]
        mask = mask.reshape(1, mask.size) != mask.reshape(mask.size, 1)
        positions = self.positions[i]
        return embeds, mask, positions
    
    
def to_device(data, device):
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device)
    
    
def save_embeddings(username, medium, task):
    device = torch.device("cuda")
    source_dir = get_data_path(os.path.join("alphas", medium, task, "Transformer", "v1"))
    model_file = os.path.join(source_dir, "model.pt")    
    config_file = os.path.join(source_dir, "config.json")
    training_config = create_training_config(config_file, 1)
    model_config = create_model_config(training_config)
    model = TransformerModel(model_config)
    model.load_state_dict(load_model(model_file, map_location="cpu"))
    model = model.to(device)
    model.eval()
    warnings.filterwarnings("ignore")
    
    outdir = get_data_path(
        f"recommendations/{username}/alphas/{medium}/{task}/Transformer/v1"
    )    
    dataloader = DataLoader(
        InferenceDataset(os.path.join(outdir, "inference.h5")),
        batch_size=16,
        shuffle=False,
    )    
    embeddings = []
    with torch.no_grad():
        for data in dataloader:
            embeddings.append(
                model(*to_device(data, device), None, None, None, embed_only=True)
                .to("cpu").detach().to(torch.float32).numpy()
            )    
    embeddings = np.vstack(embeddings)            
        
    f = h5py.File(os.path.join(outdir, "embeddings.h5"), "w")
    f.create_dataset("embedding", data=embeddings)
    detach = lambda x: x.to("cpu").detach().numpy()
    f.create_dataset(
        "anime_item_weight", data=detach(model.classifier[0][0].weight)
    )
    f.create_dataset("anime_item_bias", data=detach(model.classifier[0][0].bias))
    f.create_dataset(
        "anime_rating_weight", data=detach(model.classifier[1].weight)
    )
    f.create_dataset("anime_rating_bias", data=detach(model.classifier[1].bias))
    f.create_dataset(
        "manga_item_weight", data=detach(model.classifier[2][0].weight)
    )
    f.create_dataset("manga_item_bias", data=detach(model.classifier[2][0].bias))
    f.create_dataset(
        "manga_rating_weight", data=detach(model.classifier[3].weight)
    )
    f.create_dataset("manga_rating_bias", data=detach(model.classifier[3].bias))
    f.close()                
    
    
parser = argparse.ArgumentParser(description="Transformer")
parser.add_argument("--username", type=str, help="username")
parser.add_argument("--medium", type=str, help="medium")
parser.add_argument("--task", type=str, help="task")
args = parser.parse_args()    

if __name__ == "__main__":
    save_embeddings(args.username, args.medium, args.task)