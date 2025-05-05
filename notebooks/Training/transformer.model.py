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
        self.distinctid_embeddings = nn.ModuleList(
            [
                nn.Embedding(
                    config["vocab_sizes"][f"{m}_distinctid"] + config["reserved_vals"],
                    config["distinctid_dim"],
                )
                for m in ALL_MEDIUMS
            ]
        )
        self.periodic_time_cos = nn.Parameter(torch.zeros(4))
        self.periodic_time_sin = nn.Parameter(torch.zeros(4))
        N = sum(
            [
                config["embed_dim"],
                config["distinctid_dim"],
                1, # rating
                (config["vocab_sizes"]["status"] + config["reserved_vals"]),
                1, # time
                1, # delta time
                1, # progress
                4, # periodic time cos
                4, # periodic time sin
            ]
        )
        self.linear = nn.Linear(N, config["embed_dim"])

    def forward(self, d):
        config = self.config
        item_emb_0 = self.item_embeddings[0](
            d["0_matchedid"] + config["reserved_vals"]
        )
        item_emb_1 = self.item_embeddings[1](
            d["1_matchedid"] + config["reserved_vals"]
        )
        item_emb = torch.where(
            (d["0_matchedid"] >= 0).unsqueeze(-1), item_emb_0, item_emb_1
        )
        distinctid_emb_0 = self.distinctid_embeddings[0](
            d["0_matchedid"] + config["reserved_vals"]
        )
        distinctid_emb_1 = self.distinctid_embeddings[1](
            d["1_matchedid"] + config["reserved_vals"]
        )
        distinctid_emb = torch.where(
            (d["0_matchedid"] >= 0).unsqueeze(-1), distinctid_emb_0, distinctid_emb_1
        )
        rating_emb = d["rating"].reshape(*d["rating"].shape, 1)
        status_emb = torch.nn.functional.one_hot(
            (d["status"] + config["reserved_vals"]).to(torch.int64),
            config["vocab_sizes"]["status"] + config["reserved_vals"],
        )
        time_emb = d["time"].reshape(*d["time"].shape, 1)
        delta_time_emb = d["delta_time"].reshape(*d["time"].shape, 1)
        progress_emb = d["progress"].reshape(*d["progress"].shape, 1)
        periodic_time_cos_emb = torch.cos(d["periodic_time"] + self.periodic_time_cos)
        periodic_time_sin_emb = torch.sin(d["periodic_time"] + self.periodic_time_sin)
        emb = torch.cat(
            (
                item_emb,
                distinctid_emb,
                rating_emb,
                status_emb,
                time_emb,
                delta_time_emb,
                progress_emb,
                periodic_time_cos_emb,
                periodic_time_sin_emb,
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
        # create model
        self.config = config
        self.embed = InputEmbedding(config)
        self.transformers = Llama3(config)

        def create_lm_head(medium):
            x = self.embed.item_embeddings[medium]
            weight_tied_linear = nn.Linear(*reversed(x.weight.shape))
            x.weight = weight_tied_linear.weight
            return nn.Sequential(weight_tied_linear, nn.LogSoftmax(dim=-1))

        self.watch_heads = nn.ModuleList(
            [create_lm_head(medium) for medium in ALL_MEDIUMS]
        )
        self.rating_head = nn.Sequential(
            nn.Linear(2 * config["embed_dim"], config["embed_dim"]),
            nn.GELU(),
            nn.Linear(config["embed_dim"], 1),
        )
        self.empty_loss = nn.Parameter(torch.tensor(0.0))
        if config["lora"]:
            for layer in [self.embed, self.watch_heads, self.rating_head]:
                for _, param in layer.named_parameters():
                    param.requires_grad = False
        # create loss functions
        self.lossfn_map = {
            "watch": self.crossentropy,
            "rating": self.mse,
        }
        self.evaluate_map = {
            "watch": self.crossentropy,
            "rating": self.moments,
        }
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
        # linear time embedding
        min_ts = self.config["min_ts"]
        max_ts = self.config["max_ts"]
        ts = d["time"].clip(min_ts, max_ts)
        d["time"] = ((ts - min_ts) / (max_ts - min_ts)).to(torch.float32)
        dts = d["delta_time"].clip(0, max_ts - min_ts)
        d["delta_time"] = (dts / (max_ts - min_ts)).to(torch.float32)
        # periodic time embedding
        periodic_ts = ts.reshape(*ts.shape, 1)
        secs_in_day = 86400
        secs_in_week = secs_in_day * 7
        secs_in_year = secs_in_day * 365.25
        secs_in_season = secs_in_year / 4
        periods = [secs_in_day, secs_in_week, secs_in_season, secs_in_year]
        periodic_ts = torch.cat([2 * np.pi * periodic_ts / p for p in periods], dim=-1)
        d["periodic_time"] = periodic_ts.to(torch.float32)
        # other features
        d["progress"] = d["progress"].clip(0, 1)
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
        losses = []
        for medium in ALL_MEDIUMS:
            for metric in ALL_METRICS:
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
                if metric == "watch":
                    preds = (
                        self.watch_heads[medium](embed)
                        .gather(dim=-1, index=positions)
                        .reshape(-1)
                    )
                elif metric == "rating":
                    X = torch.cat(
                        (
                            embed,
                            self.embed.item_embeddings[medium](
                                positions.squeeze(dim=1)
                            ),
                        ),
                        dim=-1,
                    )
                    preds = self.rating_head(X).reshape(-1)
                else:
                    assert False
                lossfn = (
                    self.evaluate_map[metric] if evaluate else self.lossfn_map[metric]
                )
                losses.append(lossfn(preds, labels, weights))
        return losses

    def finetune_forward(self, d, evaluate):
        e = self.to_embedding(d)
        bp = torch.nonzero(d["mask_index"], as_tuple=True)
        embed = e[bp[0], bp[1], :]
        losses = []
        for medium in ALL_MEDIUMS:
            for metric in ALL_METRICS:
                weights = d[f"{medium}.{metric}.weight"]
                if not torch.is_nonzero(weights.sum()):
                    losses.append(torch.square(self.empty_loss).sum())
                    continue
                if metric == "watch":
                    preds = self.watch_heads[medium](embed)
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
                    preds = self.rating_head(X).reshape(-1)
                    labels = d[f"{medium}.{metric}.label"][item_bp[0], item_bp[1]]
                    weights = weights[item_bp[0], item_bp[1]]
                else:
                    assert False
                lossfn = (
                    self.evaluate_map[metric] if evaluate else self.lossfn_map[metric]
                )
                losses.append(lossfn(preds, labels, weights))
        return losses

    def embed_forward(self, d):
        e = self.to_embedding(d)
        bp = torch.nonzero(d["mask_index"], as_tuple=True)
        return e[bp[0], bp[1], :]
