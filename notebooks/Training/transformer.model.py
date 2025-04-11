cls_val = -1
mask_val = -2
reserved_vals = 2
max_seq_len = 1024
ALL_MEDIUMS = [0, 1]
ALL_METRICS = ["watch", "rating", "status"]


class DiscreteEmbed(nn.Module):
    def __init__(self, vocab_size, embed_size):
        super(DiscreteEmbed, self).__init__()
        self.reserved_vals = 2
        self.embedding = nn.Sequential(
            nn.Embedding(vocab_size + self.reserved_vals, embed_size),
            nn.RMSNorm(embed_size),
        )

    def forward(self, x):
        return self.embedding(x + self.reserved_vals)


class ContinuousEmbed(nn.Module):
    def __init__(self, embed_size):
        super(ContinuousEmbed, self).__init__()
        self.embedding = nn.Sequential(
            nn.Linear(1, embed_size),
            nn.RMSNorm(embed_size),
        )

    def forward(self, x):
        return self.embedding(x.reshape(*x.shape, 1))


class CompositeEmbedding(nn.Module):
    def __init__(self, embeddings):
        super(CompositeEmbedding, self).__init__()
        self.embeddings = nn.ModuleList(embeddings)

    def forward(self, inputs):
        return sum(embed(x) for (embed, x) in zip(self.embeddings, inputs))


class Llama3(nn.Module):
    def __init__(self, config):
        super(Llama3, self).__init__()
        llama3 = torchtune.models.llama3.llama3(
            vocab_size=0,
            num_layers=config["num_layers"],
            num_heads=config["num_heads"],
            num_kv_heads=config["num_kv_heads"],
            embed_dim=config["embed_size"],
            intermediate_dim=config["intermediate_dim"],
            max_seq_len=config["max_sequence_length"],
        )
        self.layers = llama3.layers
        self.norm = llama3.norm

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask=mask)
        x = self.norm(x)
        return x


class TransformerModel(nn.Module):
    def __init__(self, config):
        super(TransformerModel, self).__init__()
        self.config = config

        # create embeddings
        embeddings = []
        for size in config["vocab_sizes"]:
            if size is not None:
                embeddings.append(DiscreteEmbed(size, config["embed_size"]))
            else:
                embeddings.append(ContinuousEmbed(config["embed_size"]))
        self.embed = CompositeEmbedding(embeddings)

        # create transformers
        self.transformers = Llama3(config)

        # create classifiers
        metric_models = {
            m: nn.Linear(
                config["embed_size"],
                config["embed_size"],
            )
            for m in ALL_METRICS
        }
        medium_models = {}
        for i, m in enumerate(ALL_MEDIUMS):
            x = self.embed.embeddings[i].embedding[0] # weight tying
            linear = nn.Linear(*reversed(x.weight.shape))
            x.weight = linear.weight
            medium_models[m] = linear

        def create_head(medium, metric):
            base = [
                metric_models[metric],
                medium_models[medium],
            ]
            if metric in ["watch", "plantowatch"]:
                base.append(nn.LogSoftmax(dim=-1))
            return nn.Sequential(*base)

        self.classifier = nn.ModuleList(
            [
                create_head(medium, metric)
                for medium in ALL_MEDIUMS
                for metric in ALL_METRICS
            ]
        )

        # create loss functions
        lossfn_map = {
            "watch": self.crossentropy,
            "rating": self.mse,
            "status": self.mse,
        }
        self.lossfns = [
            lossfn_map[metric] for _ in ALL_MEDIUMS for metric in ALL_METRICS
        ]
        evaluate_map = {
            "watch": self.crossentropy,
            "rating": self.moments,
            "status": self.moments,
        }
        self.evaluatefns = [
            evaluate_map[metric] for _ in ALL_MEDIUMS for metric in ALL_METRICS
        ]
        self.names = [f"{m}.{metric}" for m in ALL_MEDIUMS for metric in ALL_METRICS]
        if config["forward"] == "pretrain":
            self.forward = self.pretrain_forward
        elif config["forward"] == "finetune":
            self.forward = self.finetune_forward
        elif config["forward"] == "embed":
            self.forward = self.embed_forward
        else:
            assert False

    def mse(self, x, y, w):
        return (torch.square(x - y) * w).sum() / w.sum()

    def moments(self, x, y, w):
        return [
            (torch.square(x - y) * w).sum() / w.sum(),
            (torch.square(0 * x - y) * w).sum() / w.sum(),
            (torch.square(-1 * x - y) * w).sum() / w.sum(),
        ]

    def crossentropy(self, x, y, w):
        return (-x * y * w).sum() / w.sum()

    def to_embedding(self, d):
        inputs = [d[k] for k in self.config["vocab_names"]]
        userid = d["userid"]
        m, n = userid.shape
        def maskfn(b, h, q_idx, kv_idx):
            return userid[b][q_idx] == userid[b][kv_idx]
        block_mask = create_block_mask(maskfn, B=m, H=None, Q_LEN=n, KV_LEN=n)
        e = self.embed(inputs)
        e = self.transformers(e, block_mask)
        return e

    def pretrain_forward(self, d, evaluate):
        e = self.to_embedding(d)
        lossfns = self.evaluatefns if evaluate else self.lossfns
        losses = []
        for i in range(len(lossfns)):
            k = self.names[i]
            weights = d[f"{k}.weight"]
            if not torch.is_nonzero(weights.sum()):
                losses.append(
                    torch.tensor(
                        [0.0], device=e.get_device(), requires_grad=e.requires_grad
                    )
                )
                continue
            labels = d[f"{k}.label"]
            positions = d[f"{k}.position"]
            positions = positions.reshape(*positions.shape, 1)
            classifier = self.classifier[i]
            bp = torch.nonzero(weights, as_tuple=True)
            embed = e[bp[0], bp[1], :]
            labels = labels[bp[0], bp[1]]
            positions = positions[bp[0], bp[1]]
            weights = weights[bp[0], bp[1]]
            preds = classifier(embed).gather(dim=-1, index=positions).reshape(-1)
            losses.append(lossfns[i](preds, labels, weights))
        return losses

    def finetune_forward(self, d, evaluate):
        e = self.to_embedding(d)
        bp = torch.nonzero(d["mask_index"], as_tuple=True)
        embed = e[bp[0], bp[1], :]
        lossfns = self.evaluatefns if evaluate else self.lossfns
        losses = []
        for i in range(len(lossfns)):
            k = self.names[i]
            weights = d[f"{k}.weight"]
            if not torch.is_nonzero(weights.sum()):
                losses.append(
                    torch.tensor(
                        [0.0], device=e.get_device(), requires_grad=e.requires_grad
                    )
                )
                continue
            labels = d[f"{k}.label"]
            classifier = self.classifier[i]
            preds = classifier(embed)
            losses.append(lossfns[i](preds, labels, weights))
        return losses

    def embed_forward(self, d):
        e = self.to_embedding(d)
        bp = torch.nonzero(d["mask_index"], as_tuple=True)
        return e[bp[0], bp[1], :]
