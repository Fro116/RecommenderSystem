ALL_MEDIUMS = [0, 1]
ALL_METRICS = ["watch", "rating"]


def init_weights(module, std=0.006):
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=std)
        if hasattr(module, "bias") and module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=std)
        module.weight.data[-1].fill_(0)


class MaskedEmbedding(nn.Module):
    """Supports using -1 as a masked value"""

    def __init__(self, vocab_size, embed_dim):
        super(MaskedEmbedding, self).__init__()
        self.vocab_size = vocab_size + 1
        self.embedding = nn.Embedding(self.vocab_size, embed_dim)

    def forward(self, x):
        return self.embedding(torch.where(x == -1, self.vocab_size - 1, x))


class ActionEmbedding(nn.Module):
    def __init__(self, config):
        super(ActionEmbedding, self).__init__()
        self.config = config
        self.periodic_time_cos = nn.Parameter(torch.zeros(2))
        self.periodic_time_sin = nn.Parameter(torch.zeros(2))
        self.status_embedding = MaskedEmbedding(config["vocab_sizes"]["status"], 16)
        self.gender_embedding = MaskedEmbedding(config["vocab_sizes"]["gender"], 4)
        self.source_embedding = MaskedEmbedding(config["vocab_sizes"]["source"], 4)
        N = sum(
            [
                1,  # time
                2,  # periodic time cos
                2,  # periodic time sin
                4,  # gender
                4,  # source
                1,  # has_rating
                1,  # rating
                16,  # status
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
        # time
        time_emb = (
            ((ts - min_ts) / (max_ts - min_ts)).to(torch.float32).reshape(*ts.shape, 1)
        )
        periodic_time_cos_emb = torch.cos(periodic_ts + self.periodic_time_cos)
        periodic_time_sin_emb = torch.sin(periodic_ts + self.periodic_time_sin)
        # user features
        gender_emb = self.gender_embedding(d["gender"])
        source_emb = self.source_embedding(d["source"])
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
                gender_emb,
                source_emb,
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
        super().__init__()
        self.config = config
        vocab_size = sum(config["vocab_sizes"][f"{m}_matchedid"] for m in ALL_MEDIUMS)
        embed_dim = config["embed_dim"]
        metadata_dim = config["metadata_emb_size"]
        self.vocab_size = vocab_size
        self.matchedid_embedding = MaskedEmbedding(vocab_size, embed_dim)
        self.metadata_embedding = MaskedEmbedding(vocab_size, metadata_dim)
        for p in self.metadata_embedding.parameters():
            p.requires_grad = False
        self.projection_layer = nn.Linear(metadata_dim, embed_dim)

    def forward(self, x):
        matched = self.matchedid_embedding(x)
        metadata = self.projection_layer(self.metadata_embedding(x))
        return matched + metadata


class DualItemEmbedding(nn.Module):
    def __init__(self, item_embedding: ItemEmbedding):
        super().__init__()
        self.item_embedding = item_embedding

    def forward(self, inputs) -> torch.Tensor:
        x, medium = inputs
        F = nn.functional
        K = self.item_embedding.config["vocab_sizes"]["0_matchedid"]
        if medium == 0:
            start, end = 0, K
        elif medium == 1:
            start, end = K, -1
        else:
            assert False
        m1_sliced = self.item_embedding.matchedid_embedding.embedding.weight[start:end]
        m2_sliced = self.item_embedding.metadata_embedding.embedding.weight[start:end]
        W = self.item_embedding.projection_layer.weight
        b = self.item_embedding.projection_layer.bias
        items = m1_sliced + F.linear(m2_sliced, W, bias=b)
        return F.linear(x, items)


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
            # we split each token into an item and action token
            "max_seq_len": 2 * config["max_sequence_length"],
        }
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

        self.watch_head = DualItemEmbedding(self.item_embedding)
        self.rating_head = nn.Sequential(
            nn.Linear(config["embed_dim"], config["embed_dim"]),
            nn.GELU(),
            nn.Linear(config["embed_dim"], 1),
        )
        self.apply(init_weights)
        if config["finetune"]:
            for layer in [
                self.action_embedding,
                self.item_embedding,
                self.watch_head,
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

    def load_pretrained_embeddings(self, datadir):
        with h5py.File(f"{datadir}/media_embeddings.h5") as f:
            d = {}
            for k in f:
                d[k] = f[k][:]
        W = d["metadata"]
        weights = self.item_embedding.metadata_embedding.embedding.weight
        weights.zero_()
        assert W.shape[0] + 1 == weights.shape[0]
        assert W.shape[1] <= weights.shape[1] < W.shape[1] + 16
        W_t = torch.as_tensor(W, dtype=weights.dtype, device=weights.device)
        weights[: W.shape[0], : W.shape[1]].copy_(W_t)

    def mse(self, x, y, w):
        return (torch.square(x - y) * w).sum() / w.sum().clamp(min=1e-8)

    def moments(self, x, y, w):
        w_sum = w.sum().clamp(min=1e-8)
        return [
            (torch.square(x - y) * w).sum() / w_sum,
            (torch.square(0 * x - y) * w).sum() / w_sum,
            (torch.square(-1 * x - y) * w).sum() / w_sum,
        ]

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

    def mask_tokens(self, d):
        if self.config["finetune"]:
            watch_mask = torch.zeros_like(d["userid"])
            rating_mask = torch.zeros_like(d["userid"])
            for k in d:
                if self.config["finetune_metric"] == "watch":
                    if "watch" not in k:
                        continue
                    if k.endswith(".weight"):
                        watch_mask += d[k] > 0
                elif self.config["finetune_metric"] == "rating":
                    if "rating" not in k:
                        continue
                    if k.endswith(".weight"):
                        rating_mask += d[k] > 0
                else:
                    assert False
            watch_mask = watch_mask > 0
            rating_mask = rating_mask > 0
        else:
            randval = torch.rand(d["userid"].shape, device=d["userid"].device)
            mask_rate = self.config["mask_rate"]
            watch_mask = randval < mask_rate
            rating_mask = (randval >= mask_rate) & (randval < 2 * mask_rate)
        d["token_mask_ids"] *= rating_mask
        for k in d:
            if k.endswith(".position") or k.endswith(".label") or k.endswith(".weight"):
                if "watch" in k:
                    d[k][~watch_mask] = 0
                elif "rating" in k:
                    d[k][~rating_mask] = 0
                elif "status" in k:
                    pass
                else:
                    assert False
            elif k in ["userid", "token_mask_ids", "time", "gender", "source"]:
                pass  # don't mask
            elif k in ["matchedid"]:
                d[k][watch_mask] = -1
            elif k in ["status"]:
                d[k][watch_mask | rating_mask] = -1
            elif k in ["rating", "progress"]:
                d[k][watch_mask | rating_mask] = 0
            else:
                assert False
        return d

    def to_embedding(self, d):
        e_a = self.action_embedding(d)
        e_i = self.item_embedding(d["matchedid"])
        e = self.interleave(e_i, e_a)
        userid = self.interleave(d["userid"], d["userid"])
        token_mask_ids = self.interleave(d["token_mask_ids"], d["token_mask_ids"])
        if "rope_input_pos" in d:
            rope_input_pos = self.interleave(
                2 * d["rope_input_pos"],
                2 * d["rope_input_pos"] + 1,
            )
        else:
            rope_input_pos = None
        m, n = userid.shape

        def document_mask(b, h, q_idx, kv_idx):
            return userid[b][q_idx] == userid[b][kv_idx]

        def token_mask(b, h, q_idx, kv_idx):
            return (token_mask_ids[b][kv_idx] == 0) | (
                token_mask_ids[b][q_idx] == token_mask_ids[b][kv_idx]
            )

        maskfn = and_masks(document_mask, token_mask)
        block_mask = create_block_mask(maskfn, B=m, H=None, Q_LEN=n, KV_LEN=n)
        return self.transformers(e, block_mask, rope_input_pos)

    def train_forward(self, d, evaluate):
        if not self.config["finetune"]:
            for k in d:
                d[k] = d[k].reshape(-1, self.config["max_sequence_length"])
        d = self.mask_tokens(d)
        embeds = self.to_embedding(d)
        e0 = embeds[:, 0::2]
        e1 = embeds[:, 1::2]
        topk = self.config["mask_topk"] * self.config["local_batch_size"]
        losses = []
        for medium in ALL_MEDIUMS:
            for metric in ALL_METRICS:
                w = d[f"{medium}.{metric}.weight"]
                l = d[f"{medium}.{metric}.label"]
                p = d[f"{medium}.{metric}.position"]
                e = e0 if metric == "watch" else e1
                w = d[f"{medium}.{metric}.weight"]
                _, bp = torch.topk(w.reshape(-1), k=topk)
                embed = e.reshape(-1, e.shape[-1])[bp, :]
                labels = l.reshape(-1)[bp]
                positions = p.reshape(-1)[bp]
                weights = w.reshape(-1)[bp]
                if metric == "watch":
                    w_sum = weights.sum().clamp(min=1e-8)
                    logits = self.watch_head((embed, medium))
                    target = positions.to(torch.long)
                    ce_loss = nn.functional.cross_entropy(
                        logits, target, reduction="none"
                    )
                    losses.append((ce_loss * labels * weights).sum() / w_sum)
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

    def inference_forward(self, d, task):
        e = self.to_embedding(d)
        if task == "retrieval":
            return e
        elif task == "ranking":
            return self.rating_head(e)
        else:
            assert False