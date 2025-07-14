ALL_MEDIUMS = [0, 1]
ALL_METRICS = ["watch", "rating"]


class MaskedEmbedding(nn.Module):
    """Supports using -1 as a masked value"""

    def __init__(self, vocab_size, embed_dim):
        super(MaskedEmbedding, self).__init__()
        self.vocab_size = vocab_size + 1
        self.embedding = nn.Embedding(self.vocab_size, embed_dim)

    def forward(self, x):
        return self.embedding(torch.where(x == -1, self.vocab_size-1, x))


class ActionEmbedding(nn.Module):
    def __init__(self, config):
        super(ActionEmbedding, self).__init__()
        self.config = config
        self.periodic_time_cos = nn.Parameter(torch.zeros(2))
        self.periodic_time_sin = nn.Parameter(torch.zeros(2))
        self.status_embedding = MaskedEmbedding(config["vocab_sizes"]["status"], 16)
        N = sum(
            [
                1,  # time
                2,  # periodic time cos
                2,  # periodic time sin
                1,  # has_rating
                1,  # rating
                16, # status
                1,  # progress
            ]
        )
        self.linear = nn.Linear(N, config["embed_dim"])

    def forward(self, d):
        # linear time embedding
        min_ts = self.config["min_ts"]
        max_ts = self.config["max_ts"]
        ts = d["time"].clip(min_ts)
        # periodic time embedding
        periodic_ts = ts.reshape(*ts.shape, 1)
        secs_in_day = 86400
        secs_in_week = secs_in_day * 7
        periods = [secs_in_day, secs_in_week]
        periodic_ts = torch.cat([2 * np.pi * periodic_ts / p for p in periods], dim=-1)
        periodic_ts = periodic_ts.to(torch.float32)
        # embed
        time_emb = ((ts - min_ts) / (max_ts - min_ts)).to(torch.float32).reshape(*ts.shape, 1)
        periodic_time_cos_emb = torch.cos(periodic_ts + self.periodic_time_cos)
        periodic_time_sin_emb = torch.sin(periodic_ts + self.periodic_time_sin)
        # actions
        has_rating_emb = (d["rating"] != 0).int().reshape(*d["rating"].shape, 1)
        rating_emb = d["rating"].reshape(*d["rating"].shape, 1)
        rating_emb = has_rating_emb * (
            (rating_emb - self.config["rating_mean"]) / self.config["rating_std"]
        )
        status_emb = self.status_embedding(d["status"])
        progress_emb = d["progress"].reshape(*d["progress"].shape, 1)
        emb = torch.cat(
            (
                time_emb,
                periodic_time_cos_emb,
                periodic_time_sin_emb,
                has_rating_emb,
                rating_emb,
                status_emb,
                progress_emb,
            ),
            dim=-1,
        )
        return self.linear(emb)


class ItemEmbedding(nn.Module):
    def __init__(self, config):
        super(ItemEmbedding, self).__init__()
        self.matchedid_embeddings = nn.ModuleList(
            [
                MaskedEmbedding(
                    config["vocab_sizes"][f"{m}_matchedid"],
                    config["embed_dim"],
                )
                for m in ALL_MEDIUMS
            ]
        )
        self.distinctid_embeddings = nn.ModuleList(
            [
                MaskedEmbedding(
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
        matchedid_emb_0 = self.matchedid_embeddings[0](d["0_matchedid"])
        matchedid_emb_1 = self.matchedid_embeddings[1](d["1_matchedid"])
        matchedid_emb = torch.where(
            (d["0_matchedid"] >= 0).unsqueeze(-1), matchedid_emb_0, matchedid_emb_1
        )
        distinctid_emb_0 = self.distinctid_embeddings[0](d["0_distinctid"])
        distinctid_emb_1 = self.distinctid_embeddings[1](d["1_distinctid"])
        distinctid_emb = torch.where(
            (d["0_distinctid"] >= 0).unsqueeze(-1), distinctid_emb_0, distinctid_emb_1
        )
        emb = torch.cat(
            (
                matchedid_emb,
                distinctid_emb,
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
        }
        if config["causal"]:
            # we split each token into an item and action token
            llama_config["max_seq_len"] = 2 * config["max_sequence_length"]
        else:
            llama_config["max_seq_len"] = config["max_sequence_length"]
        if config["finetune"]:
            llama3 = torchtune.models.llama3.lora_llama3(
                lora_attn_modules=["q_proj", "v_proj"],
                lora_rank=8,
                lora_alpha=16,
                lora_dropout=0.1,
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
        self.config = config
        self.action_embedding = ActionEmbedding(config)
        self.item_embedding = ItemEmbedding(config)
        self.transformers = Llama3(config)

        def create_lm_head(medium):
            x = self.item_embedding.matchedid_embeddings[medium].embedding
            weight_tied_linear = nn.Linear(*reversed(x.weight.shape))
            x.weight = weight_tied_linear.weight
            return nn.Sequential(weight_tied_linear, nn.LogSoftmax(dim=-1))

        self.watch_heads = nn.ModuleList(
            [create_lm_head(medium) for medium in ALL_MEDIUMS]
        )
        if config["causal"]:
            self.rating_head = nn.Sequential(
                nn.Linear(config["embed_dim"], config["embed_dim"]),
                nn.GELU(),
                nn.Linear(config["embed_dim"], 1),
            )
        else:
            self.rating_head = nn.Sequential(
                nn.Linear(2 * config["embed_dim"], config["embed_dim"]),
                nn.GELU(),
                nn.Linear(config["embed_dim"], 1),
            )
        self.empty_loss = nn.Parameter(torch.tensor(0.0))
        if config["finetune"]:
            for layer in [
                self.action_embedding,
                self.item_embedding,
                self.watch_heads,
                self.rating_head,
            ]:
                for _, param in layer.named_parameters():
                    param.requires_grad = False
        if config["forward"] == "train":
            self.forward = self.train_forward
        elif config["forward"] == "inference":
            self.forward = self.inference_forward
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

    def interleave(self, x, y):
        # interleave the tokens so that it goes x -> y -> x -> y, etc
        dims = x.shape
        if len(dims) == 2:
            ret = torch.stack([x, y], dim=2)
            ret = ret.reshape(dims[0], dims[1] * 2)
            return ret
        elif len(dims) == 3:
            ret = torch.stack((x, y), dim=2)
            ret = ret.reshape(dims[0], dims[1] * 2, dims[2])
            return ret
        else:
            assert False

    def shift_right(self, x):
        y = torch.zeros_like(x)
        y[..., 1:] = x[..., :-1]
        return y

    def split_tokens(self, d):
        item_tokens = {}
        action_tokens = {}
        userid_mask = d["userid"] == self.shift_right(d["userid"])
        for k, v in d.items():
            if k in ["userid", "rope_input_pos", "token_mask_ids"]:
                item_tokens[k] = v
                action_tokens[k] = v
            elif (
                k in ["time"]
                or k.startswith("0.watch")
                or k.startswith("1.watch")
            ):
                action_tokens[k] = v
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
                action_tokens[k] = self.shift_right(v) * userid_mask
            else:
                assert False, k
        return {
            "item_tokens": item_tokens,
            "action_tokens": action_tokens,
        }

    def mask_tokens(self, d):
        if self.config["finetune"]:
            mask = torch.zeros_like(d["userid"])
            for k in d:
                if k.endswith(".weight"):
                    mask += (d[k] > 0)
            mask = mask > 0
        else:
            mask = (
                torch.rand(d["userid"].shape, device=d["userid"].device)
                < self.config["mask_rate"]
            )
        for k in d:
            if k.endswith(".position") or k.endswith(".label") or k.endswith(".weight"):
                d[k][~mask] = 0
            elif k in ["userid", "time"]:
                pass  # don't mask
            elif k in [
                "0_matchedid",
                "0_distinctid",
                "1_matchedid",
                "1_distinctid",
                "status",
            ]:
                d[k][mask] = -1
            elif k in ["rating", "progress"]:
                d[k][mask] = 0
            else:
                assert False
        return d

    def to_embedding(self, d):
        if self.config["causal"]:
            e_a = self.action_embedding(d["action_tokens"])
            e_i = self.item_embedding(d["item_tokens"])
            e = self.interleave(e_a, e_i)
            userid = self.interleave(
                *[d[k]["userid"] for k in ["action_tokens", "item_tokens"]]
            )
            if "rope_input_pos" in d["action_tokens"]:
                rope_input_pos = self.interleave(
                    2*d["action_tokens"]["rope_input_pos"], 2*d["item_tokens"]["rope_input_pos"]+1
                )
                token_mask_ids = self.interleave(
                    d["action_tokens"]["token_mask_ids"], d["item_tokens"]["token_mask_ids"]
                )
                # TODO make a binary mask of duplicated item ids, shuffle them to the end, and cut
            else:
                rope_input_pos = None
                token_mask_ids = torch.zeros_like(userid)
            # attention_masks
            m, n = userid.shape

            def document_mask(b, h, q_idx, kv_idx):
                return userid[b][q_idx] == userid[b][kv_idx]

            def causal_mask(b, h, q_idx, kv_idx):
                return q_idx >= kv_idx

            def token_mask(b, h, q_idx, kv_idx):
                return (token_mask_ids[b][kv_idx] == 0) | (token_mask_ids[b][q_idx] == token_mask_ids[b][kv_idx])

            maskfn = and_masks(document_mask, causal_mask, token_mask)
            block_mask = create_block_mask(maskfn, B=m, H=None, Q_LEN=n, KV_LEN=n)
            e = self.transformers(e, block_mask, rope_input_pos)
            return e
        else:
            userid = d["userid"]
            m, n = userid.shape

            def document_mask(b, h, q_idx, kv_idx):
                return userid[b][q_idx] == userid[b][kv_idx]

            block_mask = create_block_mask(
                document_mask, B=m, H=None, Q_LEN=n, KV_LEN=n
            )
            e = self.action_embedding(d) + self.item_embedding(d)
            e = self.transformers(e, block_mask, None)
            return e

    def train_forward(self, d, evaluate):
        if not self.config["finetune"]:
            for k in d:
                d[k] = d[k].reshape(-1, self.config["max_sequence_length"])
        if self.config["causal"]:
            d = self.split_tokens(d)
        else:
            d = self.mask_tokens(d)
        e = self.to_embedding(d)
        losses = []
        for medium in ALL_MEDIUMS:
            for metric in ALL_METRICS:
                if self.config["causal"]:
                    if metric == "watch":
                        w = d["action_tokens"][f"{medium}.{metric}.weight"]
                        weights = self.interleave(w, w * 0)
                        l = d["action_tokens"][f"{medium}.{metric}.label"]
                        labels = self.interleave(l, l * 0)
                        p = d["action_tokens"][f"{medium}.{metric}.position"]
                        positions = self.interleave(p, p * 0)
                    elif metric == "rating":
                        w = d["item_tokens"][f"{medium}.{metric}.weight"]
                        weights = self.interleave(w * 0, w)
                        l = d["item_tokens"][f"{medium}.{metric}.label"]
                        labels = self.interleave(l * 0, l)
                        p = d["item_tokens"][f"{medium}.{metric}.position"]
                        positions = self.interleave(p * 0, p)
                    else:
                        assert False
                else:
                    weights = d[f"{medium}.{metric}.weight"]
                    labels = d[f"{medium}.{metric}.label"]
                    positions = d[f"{medium}.{metric}.position"]
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
                    if self.config["causal"]:
                        preds = self.rating_head(embed).reshape(-1)
                    else:
                        X = torch.cat(
                            (
                                embed,
                                self.item_embedding.matchedid_embeddings[medium](
                                    positions.squeeze(dim=1)
                                ),
                            ),
                            dim=-1,
                        )
                        preds = self.rating_head(X).reshape(-1)
                    labels = labels - self.config["rating_mean"]
                    if evaluate:
                        losses.append(self.moments(preds, labels, weights))
                    else:
                        losses.append(self.mse(preds, labels, weights))
                else:
                    assert False
        return losses

    def inference_forward(self, d, task):
        if self.config["causal"]:
            d = self.split_tokens(d)
        e = self.to_embedding(d)
        if task == "retrieval":
            return e
        elif task == "ranking":
            if self.config["causal"]:
                return e, self.rating_head(e)
            else:
                return e
        else:
            assert False
