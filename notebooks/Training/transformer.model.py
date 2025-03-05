cls_val = -1
mask_val = -2
reserved_vals = 2
max_seq_len = 1024
ALL_MEDIUMS = [0, 1] # TODO rename
ALL_METRICS = ["rating", "watch", "plantowatch", "drop"]


class DiscreteEmbed(nn.Module):
    def __init__(self, vocab_size, embed_size):
        super(DiscreteEmbed, self).__init__()
        self.reserved_vals = 2
        self.embedding = nn.Sequential(
            nn.Embedding(vocab_size + self.reserved_vals, embed_size),
            nn.LayerNorm(embed_size),
        )

    def forward(self, x):
        return self.embedding(x + self.reserved_vals)


class ContinuousEmbed(nn.Module):
    def __init__(self, embed_size, dropout):
        super(ContinuousEmbed, self).__init__()
        hidden_size = int(embed_size / 4)
        self.embedding_with_weightdecay = nn.Sequential(
            nn.Linear(1, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, embed_size),
            nn.LayerNorm(embed_size),
        )

    def forward(self, x):
        return self.embedding_with_weightdecay(x.reshape(*x.shape, 1))


class CompositeEmbedding(nn.Module):
    def __init__(self, embeddings, postprocessor):
        super(CompositeEmbedding, self).__init__()
        self.embeddings = nn.ModuleList(embeddings)
        self.postprocessor = postprocessor

    def forward(self, inputs):
        embedding = sum(embed(x) for (embed, x) in zip(self.embeddings, inputs))
        return self.postprocessor(embedding)


class Bert(nn.Module):
    def __init__(
        self,
        num_layers,
        embed_size,
        num_attention_heads,
        intermediate_size,
        activation,
        dropout,
    ):
        super(Bert, self).__init__()
        self.num_heads = num_attention_heads
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embed_size,
                nhead=num_attention_heads,
                dim_feedforward=intermediate_size,
                dropout=dropout,
                activation=activation,
                norm_first=True,
                batch_first=True,
            ),
            num_layers=num_layers,
        )

    def forward(self, x, mask):
        mask = torch.repeat_interleave(mask, self.num_heads, dim=0)
        return self.encoder(x, mask=mask)


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
                embeddings.append(
                    ContinuousEmbed(config["embed_size"], config["dropout"])
                )
        postprocessor = nn.Sequential(
            nn.LayerNorm(config["embed_size"]), nn.Dropout(config["dropout"])
        )
        self.embed = CompositeEmbedding(embeddings, postprocessor)

        # create transformers
        self.transformers = Bert(
            num_layers=config["num_layers"],
            embed_size=config["embed_size"],
            num_attention_heads=config["num_attention_heads"],
            intermediate_size=config["intermediate_size"],
            activation=config["activation"],
            dropout=config["dropout"],
        )

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
            "rating": self.mse,
            "watch": self.crossentropy,
            "plantowatch": self.crossentropy,
            "drop": self.binarycrossentropy,
        }
        self.lossfns = [
            lossfn_map[metric] for _ in ALL_MEDIUMS for metric in ALL_METRICS
        ]
        evaluate_map = {
            "rating": self.moments,
            "watch": self.crossentropy,
            "plantowatch": self.crossentropy,
            "drop": self.binarycrossentropy,
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

    def binarycrossentropy(self, x, y, w):
        return (
            torch.nn.functional.binary_cross_entropy_with_logits(
                input=x,
                target=y,
                weight=w,
                reduction="sum",
            )
            / w.sum()
        )

    def pretrain_forward(self, d, evaluate):
        inputs = [d[k] for k in self.config["vocab_names"]]
        mask = d["attn_mask"]
        e = self.embed(inputs)
        e = self.transformers(e, mask)
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
        inputs = [d[k] for k in self.config["vocab_names"]]
        mask = d["attn_mask"]
        e = self.embed(inputs)
        e = self.transformers(e, mask)
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
        inputs = [d[k] for k in self.config["vocab_names"]]
        mask = d["attn_mask"]
        e = self.embed(inputs)
        e = self.transformers(e, mask)
        bp = torch.nonzero(d["mask_index"], as_tuple=True)
        return e[bp[0], bp[1], :]
