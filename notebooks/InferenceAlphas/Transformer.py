exec(open("../TrainingAlphas/Transformer/Transformer.py").read())

import argparse
import h5py
import hdf5plugin
import warnings
from torch.utils.data import DataLoader, Dataset

class InferenceDataset(Dataset):
    def __init__(self, file, vocab_names, vocab_types):
        self.filename = file
        f = h5py.File(file, "r")

        def process(x, dtype):
            if dtype == "float":
                return f[x][:].reshape(*f[x].shape, 1).astype(np.float32)
            elif dtype == "int":
                return f[x][:]
            else:
                assert False

        self.length = f["userid"].shape[0]
        self.embeddings = [
            process(x, y) for (x, y) in zip(vocab_names, vocab_types) if x != "userid"
        ]
        self.mask = f["userid"][:]

        def process_position(x):
            return x[:].flatten().astype(np.int64)

        self.positions = process_position(f["positions"])
        f.close()

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
    
    
def to_device(data, device):
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]     
    return data.to(device)

    
def save_embeddings(source, username, medium):
    device = torch.device("cpu")
    source_dir = get_data_path(os.path.join("alphas", medium, "Transformer", "v1"))
    model_file = os.path.join(source_dir, "model.pt")    
    training_config = create_training_config(get_data_path(os.path.join("alphas", "all", "Transformer", "v1")))
    training_config["mode"] = "finetune"
    warnings.filterwarnings("ignore")
    model_config = create_model_config(training_config)
    model = TransformerModel(model_config)
    model.load_state_dict(load_model(model_file, map_location="cpu"))
    model = model.to(device)
    model.eval()

    dataloader = DataLoader(
        InferenceDataset(
            get_data_path(
                f"recommendations/{source}/{username}/alphas/Transformer/v1/inference.h5"
            ),
            training_config["vocab_names"],
            training_config["vocab_types"],            
        ),
        batch_size=16,
        shuffle=False,
    ) 

    user_batches = []
    embed_batches = []
    model.eval()
    for data in dataloader:
        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                x = [
                    y
                    .to("cpu")
                    .to(torch.float32)
                    .numpy()
                    for y in model(*to_device(data, device), inference=True)
                ]
                users = data[-1].numpy()
                user_batches.append(users)
                embed_batches.append(x)
    outdir = get_data_path(f"recommendations/{source}/{username}/alphas/{medium}/Transformer/v1")
    os.makedirs(outdir, exist_ok=True)
    f = h5py.File(os.path.join(outdir, "embeddings.h5"), "w")
    f.create_dataset("users", data=np.hstack(user_batches))
    i = 0
    for medium in ALL_MEDIUMS:
        for metric in ALL_METRICS:
            f.create_dataset(f"{medium}_{metric}", data=np.vstack([x[i] for x in embed_batches]))
            i += 1
    f.close()
  
    
parser = argparse.ArgumentParser(description="Transformer")
parser.add_argument("--source", type=str, help="source")
parser.add_argument("--username", type=str, help="username")
parser.add_argument("--medium", type=str, help="medium")
args = parser.parse_args()    

if __name__ == "__main__":
    save_embeddings(args.source, args.username, args.medium)