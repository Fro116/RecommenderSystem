cls_val = -1
mask_val = -2
reserved_vals = 2
max_seq_len = 1024
ALL_MEDIUMS = [0, 1]
ALL_METRICS = ["watch", "rating"]


class InputEmbedding(nn.Module):
    def __init__(self, config):
        super(InputEmbedding, self).__init__()
        self.config = config
        self.item_embeddings = nn.ModuleList(
            [
                nn.Embedding(
                    config["vocab_sizes"][f"{m}_matchedid"] + config["reserved_vals"],
                    config["embed_dim"],
                )
                for m in ALL_MEDIUMS
            ]
        )
        N = sum(
            [
                config["embed_dim"],
                config["embed_dim"],
                1,
                (config["vocab_sizes"]["status"] + config["reserved_vals"]),
                1,
                1,
            ]
        )
        self.linear = nn.Linear(N, config["embed_dim"])

    def forward(self, d):
        config = self.config
        item_embs = [
            self.item_embeddings[m](d[f"{m}_matchedid"] + config["reserved_vals"])
            for m in ALL_MEDIUMS
        ]
        rating_emb = d["rating"].reshape(*d["rating"].shape, 1)
        status_emb = torch.nn.functional.one_hot(
            (d["status"] + config["reserved_vals"]).to(torch.int64),
            config["vocab_sizes"]["status"] + config["reserved_vals"],
        )
        time_emb = d["time"].reshape(*d["time"].shape, 1)
        delta_time_emb = d["delta_time"].reshape(*d["time"].shape, 1)
        emb = torch.cat(
            (
                *item_embs,
                rating_emb,
                status_emb,
                time_emb,
                delta_time_emb,
            ),
            dim=-1,
        )
        return self.linear(emb)


class Llama3(nn.Module):
    def __init__(self, config):
        super(Llama3, self).__init__()
        llama_config = {
            "vocab_size": 0,
            "num_layers": config["num_layers"],
            "num_heads": config["num_heads"],
            "num_kv_heads": config["num_kv_heads"],
            "embed_dim": config["embed_dim"],
            "intermediate_dim": config["intermediate_dim"],
            "max_seq_len": config["max_sequence_length"],
        }
        if config["lora"]:
            llama3 = torchtune.models.llama3.lora_llama3(
                lora_attn_modules=["q_proj", "v_proj"],
                lora_rank=8,
                lora_alpha=16,
                **llama_config,
            )
            set_trainable_params(llama3, get_adapter_params(llama3))
        else:
            llama3 = torchtune.models.llama3.llama3(**llama_config)
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
        self.embed = InputEmbedding(config)
        self.transformers = Llama3(config)
        def create_head(medium, metric):
            if metric == "watch":
                x = self.embed.item_embeddings[medium]
                weight_tied_linear = nn.Linear(*reversed(x.weight.shape))
                x.weight = weight_tied_linear.weight
                return nn.Sequential(weight_tied_linear, nn.LogSoftmax(dim=-1))
            elif metric == "rating":
                return nn.Sequential(
                    nn.Linear(2 * config["embed_dim"], config["embed_dim"]),
                    nn.GELU(),
                    nn.Linear(config["embed_dim"], 1),
                )
            else:
                assert False
            return nn.Sequential(*base)
        self.classifier = nn.ModuleList(
            [
                create_head(medium, metric)
                for medium in ALL_MEDIUMS
                for metric in ALL_METRICS
            ]
        )
        self.empty_loss = nn.Parameter(torch.tensor(0.))

        if config["lora"]:
            for name, param in self.embed.named_parameters():
                param.requires_grad = False
            for name, param in self.classifier.named_parameters():
                param.requires_grad = False

        # create loss functions
        lossfn_map = {
            "watch": self.crossentropy,
            "rating": self.mse,
        }
        self.lossfns = [
            lossfn_map[metric] for _ in ALL_MEDIUMS for metric in ALL_METRICS
        ]
        evaluate_map = {
            "watch": self.crossentropy,
            "rating": self.moments,
        }
        self.evaluatefns = [
            evaluate_map[metric] for _ in ALL_MEDIUMS for metric in ALL_METRICS
        ]
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
        min_ts = self.config["min_ts"]
        max_ts = self.config["max_ts"]
        ts = d["time"].clip(min_ts, max_ts)
        d["time"] = ((ts - min_ts) / (max_ts - min_ts)).to(torch.float32)
        dts = d["delta_time"].clip(0, max_ts - min_ts)
        d["delta_time"] = (dts / (max_ts - min_ts)).to(torch.float32)
        userid = d["userid"]
        m, n = userid.shape

        def document_mask(b, h, q_idx, kv_idx):
            return userid[b][q_idx] == userid[b][kv_idx]

        def causal_mask(b, h, q_idx, kv_idx):
            return q_idx >= kv_idx

        maskfn = document_mask
        if self.config["causal"]:
            maskfn = and_masks(maskfn, causal_mask)
        block_mask = create_block_mask(maskfn, B=m, H=None, Q_LEN=n, KV_LEN=n)
        e = self.embed(d)
        e = self.transformers(e, block_mask)
        return e

    def pretrain_forward(self, d, evaluate):
        e = self.to_embedding(d)
        lossfns = self.evaluatefns if evaluate else self.lossfns
        losses = []
        i = -1
        for medium in ALL_MEDIUMS:
            for metric in ALL_METRICS:
                i += 1
                weights = d[f"{medium}.{metric}.weight"]
                if not torch.is_nonzero(weights.sum()):
                    losses.append(torch.square(self.empty_loss).sum())
                    continue
                labels = d[f"{medium}.{metric}.label"]
                positions = d[f"{medium}.{metric}.position"]
                positions = positions.reshape(*positions.shape, 1)
                bp = torch.nonzero(weights, as_tuple=True)
                embed = e[bp[0], bp[1], :]
                labels = labels[bp[0], bp[1]]
                positions = positions[bp[0], bp[1]]
                weights = weights[bp[0], bp[1]]
                classifier = self.classifier[i]
                if metric == "watch":
                    preds = classifier(embed).gather(dim=-1, index=positions).reshape(-1)
                elif metric == "rating":
                    X = torch.cat(
                        (
                            embed,
                            self.embed.item_embeddings[medium](positions.squeeze(dim=1)),
                        ),
                        dim=-1,
                    )
                    preds = classifier(X).reshape(-1)
                else:
                    assert False
                losses.append(lossfns[i](preds, labels, weights))
        return losses

    def finetune_forward(self, d, evaluate):
        e = self.to_embedding(d)
        bp = torch.nonzero(d["mask_index"], as_tuple=True)
        embed = e[bp[0], bp[1], :]
        lossfns = self.evaluatefns if evaluate else self.lossfns
        losses = []
        i = -1
        for medium in ALL_MEDIUMS:
            for metric in ALL_METRICS:
                i += 1
                weights = d[f"{medium}.{metric}.weight"]
                if not torch.is_nonzero(weights.sum()):
                    losses.append(torch.square(self.empty_loss).sum())
                    continue
                classifier = self.classifier[i]
                if metric == "watch":
                    preds = classifier(embed)
                    labels = d[f"{medium}.{metric}.label"]
                elif metric == "rating":
                    item_bp = torch.nonzero(weights, as_tuple=True)
                    X = torch.cat(
                        (
                            embed[item_bp[0], :],
                            self.embed.item_embeddings[medium](item_bp[1]),
                        ),
                        dim=-1,
                    )
                    preds = classifier(X).reshape(-1)
                    labels = d[f"{medium}.{metric}.label"][item_bp[0], item_bp[1]]
                    weights = weights[item_bp[0], item_bp[1]]
                else:
                    assert False
                losses.append(lossfns[i](preds, labels, weights))
        return losses

    def embed_forward(self, d):
        e = self.to_embedding(d)
        bp = torch.nonzero(d["mask_index"], as_tuple=True)
        return e[bp[0], bp[1], :]
