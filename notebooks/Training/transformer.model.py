ALL_MEDIUMS = [0, 1]
ALL_METRICS = ["watch", "rating"]


class PromptEmbedding(nn.Module):
    def __init__(self, config):
        super(PromptEmbedding, self).__init__()
        self.config = config
        self.input_pos_embedding = nn.Embedding(
            config["max_sequence_length"],
            config["embed_dim"],
        )
        self.periodic_time_cos = nn.Parameter(torch.zeros(4))
        self.periodic_time_sin = nn.Parameter(torch.zeros(4))
        N = sum(
            [
                config["embed_dim"],  # input_pos
                1,  # time
                4,  # periodic time cos
                4,  # periodic time sin
            ]
        )
        self.linear = nn.Linear(N, config["embed_dim"])

    def forward(self, d):
        # linear time embedding
        min_ts = self.config["min_ts"]
        max_ts = self.config["max_ts"]
        ts = d["time"].clip(min_ts)
        d["time"] = ((ts - min_ts) / (max_ts - min_ts)).to(torch.float32)
        # periodic time embedding
        periodic_ts = ts.reshape(*ts.shape, 1)
        secs_in_day = 86400
        secs_in_week = secs_in_day * 7
        secs_in_year = secs_in_day * 365.25
        secs_in_season = secs_in_year / 4
        periods = [secs_in_day, secs_in_week, secs_in_season, secs_in_year]
        periodic_ts = torch.cat([2 * np.pi * periodic_ts / p for p in periods], dim=-1)
        periodic_ts = periodic_ts.to(torch.float32)
        # embed
        input_pos_emb = self.input_pos_embedding(
            d["input_pos"] % self.config["max_sequence_length"]
        )
        time_emb = d["time"].reshape(*d["time"].shape, 1)
        periodic_time_cos_emb = torch.cos(periodic_ts + self.periodic_time_cos)
        periodic_time_sin_emb = torch.sin(periodic_ts + self.periodic_time_sin)
        emb = torch.cat(
            (
                input_pos_emb,
                time_emb,
                periodic_time_cos_emb,
                periodic_time_sin_emb,
            ),
            dim=-1,
        )
        return self.linear(emb)


class ItemEmbedding(nn.Module):
    def __init__(self, config):
        super(ItemEmbedding, self).__init__()
        self.item_embeddings = nn.ModuleList(
            [
                nn.Embedding(
                    config["vocab_sizes"][f"{m}_matchedid"],
                    config["embed_dim"],
                )
                for m in ALL_MEDIUMS
            ]
        )
        self.distinctid_embeddings = nn.ModuleList(
            [
                nn.Embedding(
                    config["vocab_sizes"][f"{m}_distinctid"],
                    config["distinctid_dim"],
                )
                for m in ALL_MEDIUMS
            ]
        )
        N = sum(
            [
                config["embed_dim"],  # matchedid
                config["distinctid_dim"],  # distictid
            ]
        )
        self.linear = nn.Linear(N, config["embed_dim"])

    def forward(self, d):
        # if medium=0, then 1_matchedid=-1 and 1_distinctid=-1 and vice versa
        item_emb_0 = self.item_embeddings[0](d["0_matchedid"].clip(0))
        item_emb_1 = self.item_embeddings[1](d["1_matchedid"].clip(0))
        item_emb = torch.where(
            (d["0_matchedid"] >= 0).unsqueeze(-1), item_emb_0, item_emb_1
        )
        distinctid_emb_0 = self.distinctid_embeddings[0](d["0_matchedid"].clip(0))
        distinctid_emb_1 = self.distinctid_embeddings[1](d["1_matchedid"].clip(0))
        distinctid_emb = torch.where(
            (d["0_matchedid"] >= 0).unsqueeze(-1), distinctid_emb_0, distinctid_emb_1
        )
        emb = torch.cat(
            (
                item_emb,
                distinctid_emb,
            ),
            dim=-1,
        )
        return self.linear(emb)


class ActionEmbedding(nn.Module):
    def __init__(self, config):
        super(ActionEmbedding, self).__init__()
        self.config = config
        N = sum(
            [
                1,  # has_rating
                1,  # rating
                config["vocab_sizes"]["status"],  # status
                1,  # progress
            ]
        )
        self.linear = nn.Linear(N, config["embed_dim"])

    def forward(self, d):
        has_rating_emb = (d["rating"] != 0).int().reshape(*d["rating"].shape, 1)
        rating_emb = d["rating"].reshape(*d["rating"].shape, 1)
        rating_emb = has_rating_emb * ((rating_emb - self.config["rating_mean"]) / self.config["rating_std"])
        status_emb = torch.nn.functional.one_hot(
            (d["status"]).to(torch.int64),
            self.config["vocab_sizes"]["status"],
        )
        progress_emb = d["progress"].reshape(*d["progress"].shape, 1)
        emb = torch.cat(
            (
                has_rating_emb,
                rating_emb,
                status_emb,
                progress_emb,
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
            # we split each token into a prompt, item, and action token
            "max_seq_len": config["max_sequence_length"] * 3,
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

    def forward(self, x, mask, input_pos):
        for layer in self.layers:
            x = layer(x, mask=mask, input_pos=input_pos)
        x = self.norm(x)
        return x


class TransformerModel(nn.Module):
    def __init__(self, config):
        super(TransformerModel, self).__init__()
        # create model
        self.config = config
        self.prompt_embedding = PromptEmbedding(config)
        self.item_embedding = ItemEmbedding(config)
        self.action_embedding = ActionEmbedding(config)
        self.transformers = Llama3(config)

        def create_lm_head(medium):
            x = self.item_embedding.item_embeddings[medium]
            weight_tied_linear = nn.Linear(*reversed(x.weight.shape))
            x.weight = weight_tied_linear.weight
            return nn.Sequential(weight_tied_linear, nn.LogSoftmax(dim=-1))

        self.watch_heads = nn.ModuleList(
            [create_lm_head(medium) for medium in ALL_MEDIUMS]
        )
        # TODO simplify
        self.rating_head = nn.Sequential(
            nn.Linear(config["embed_dim"], config["embed_dim"]),
            nn.GELU(),
            nn.Linear(config["embed_dim"], 1),
        )
        self.empty_loss = nn.Parameter(torch.tensor(0.0))
        if config["lora"]:
            for layer in [
                self.prompt_embedding,
                self.item_embedding,
                self.action_embedding,
                self.watch_heads,
                self.rating_head,
            ]:
                for _, param in layer.named_parameters():
                    param.requires_grad = False
        if config["forward"] == "pretrain":
            self.forward = self.pretrain_forward
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

    def interleave(self, x, y, z):
        # interleave the tokens so that it goes x -> y -> z -> x, etc
        dims = x.shape
        if len(dims) == 2:
            ret = torch.stack([x, y, z], dim=2)
            ret = ret.reshape(dims[0], dims[1] * 3)
            return ret
        elif len(dims) == 3:
            ret = torch.stack((x, y, z), dim=2)
            ret = ret.reshape(dims[0], dims[1] * 3, dims[2])
            return ret
        else:
            assert False

    def to_embedding(self, d):
        e_p = self.prompt_embedding(d["prompt_tokens"])
        e_i = self.item_embedding(d["item_tokens"])
        e_a = self.action_embedding(d["action_tokens"])
        e = self.interleave(e_p, e_i, e_a)
        userid = self.interleave(
            *[d[k]["userid"] for k in ["prompt_tokens", "item_tokens", "action_tokens"]]
        )
        # attention_masks
        m, n = userid.shape

        def document_mask(b, h, q_idx, kv_idx):
            return userid[b][q_idx] == userid[b][kv_idx]

        def causal_mask(b, h, q_idx, kv_idx):
            return q_idx >= kv_idx

        maskfn = and_masks(document_mask, causal_mask)
        block_mask = create_block_mask(maskfn, B=m, H=None, Q_LEN=n, KV_LEN=n)
        e = self.transformers(e, block_mask, None)
        return e

    def split_tokens(self, d):
        prompt_tokens = {}
        item_tokens = {}
        action_tokens = {}
        for k, v in d.items():
            if k in ["userid"]:
                prompt_tokens[k] = v
                item_tokens[k] = v
                action_tokens[k] = v
            elif (
                k
                in [
                    "input_pos",
                    "time",
                ]
                or k.startswith("0.watch")
                or k.startswith("1.watch")
            ):
                prompt_tokens[k] = v
            elif (
                k
                in [
                    "0_matchedid",
                    "0_distinctid",
                    "1_matchedid",
                    "1_distinctid",
                ]
                or k.startswith("0.rating")
                or k.startswith("1.rating")
                or k.startswith("0.status")
                or k.startswith("1.status")
            ):
                item_tokens[k] = v
            elif k in ["status", "rating", "progress"]:
                action_tokens[k] = v
            elif k in ["istest"]:
                pass # todo remove
            else:
                assert False, k
        return {
            "prompt_tokens": prompt_tokens,
            "item_tokens": item_tokens,
            "action_tokens": action_tokens,
        }

    def pretrain_forward(self, d, evaluate):
        if not self.config["finetune"]:
            for k in d:
                d[k] = d[k].reshape(-1, self.config["max_sequence_length"])
        d = self.split_tokens(d)
        e = self.to_embedding(d)
        losses = []
        for medium in ALL_MEDIUMS:
            for metric in ALL_METRICS:
                if metric == "watch":
                    w = d["prompt_tokens"][f"{medium}.{metric}.weight"]
                    weights = self.interleave(w, w * 0, w * 0)
                    l = d["prompt_tokens"][f"{medium}.{metric}.label"]
                    labels = self.interleave(l, l * 0, l * 0)
                    p = d["prompt_tokens"][f"{medium}.{metric}.position"]
                    positions = self.interleave(p, p * 0, p * 0)
                elif metric == "rating":
                    w = d["item_tokens"][f"{medium}.{metric}.weight"]
                    weights = self.interleave(w * 0, w, w * 0)
                    l = d["item_tokens"][f"{medium}.{metric}.label"]
                    labels = self.interleave(l * 0, l, l * 0)
                    p = d["item_tokens"][f"{medium}.{metric}.position"]
                    positions = self.interleave(p * 0, p, p * 0)
                else:
                    assert False
                if not torch.is_nonzero(weights.sum()):
                    losses.append(torch.square(self.empty_loss).sum())
                    continue
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
                    losses.append(self.crossentropy(preds, labels, weights))
                elif metric == "rating":
                    preds = self.rating_head(embed).reshape(-1)
                    labels = labels - self.config["rating_mean"]
                    if evaluate:
                        losses.append(self.moments(preds, labels, weights))
                    else:
                        losses.append(self.mse(preds, labels, weights))
                else:
                    assert False
        return losses

    def embed_forward(self, d):
        e = self.to_embedding(d)
        bp = torch.nonzero(d["mask_index"], as_tuple=True)
        return e[bp[0], bp[1], :]
