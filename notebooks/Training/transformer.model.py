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


def precompute_freqs_cis(dim: int, end: int, theta: float = 500000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    freqs = torch.cat((freqs, freqs), dim=-1)
    freqs_cos = torch.cos(freqs)
    freqs_sin = torch.sin(freqs)
    return freqs_cos, freqs_sin


def apply_rotary_emb(x: torch.Tensor, freqs_cos: torch.Tensor, freqs_sin: torch.Tensor):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    x_rot = torch.cat((-x2, x1), dim=-1)
    freqs_cos = freqs_cos.unsqueeze(-2).to(x.dtype)
    freqs_sin = freqs_sin.unsqueeze(-2).to(x.dtype)
    return (x * freqs_cos) + (x_rot * freqs_sin)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class SwiGLU(nn.Module):
    def __init__(self, dim: int, intermediate_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, intermediate_dim, bias=False)
        self.w2 = nn.Linear(intermediate_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, intermediate_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class FlashAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.finetune = config["finetune"]
        self.num_heads = config["num_heads"]
        self.num_kv_heads = config["num_kv_heads"]
        self.head_dim = config["embed_dim"] // self.num_heads
        self.wq = nn.Linear(
            config["embed_dim"], self.num_heads * self.head_dim, bias=False
        )
        self.wk = nn.Linear(
            config["embed_dim"], self.num_kv_heads * self.head_dim, bias=False
        )
        self.wv = nn.Linear(
            config["embed_dim"], self.num_kv_heads * self.head_dim, bias=False
        )
        self.wo = nn.Linear(
            self.num_heads * self.head_dim, config["embed_dim"], bias=False
        )
        if self.finetune:
            self.lora_rank = 8
            self.lora_scaling = 16 / 8
            self.lora_dropout = nn.Dropout(0.1)
            self.wq_lora_A = nn.Linear(config["embed_dim"], self.lora_rank, bias=False)
            self.wq_lora_B = nn.Linear(
                self.lora_rank, self.num_heads * self.head_dim, bias=False
            )
            self.wv_lora_A = nn.Linear(config["embed_dim"], self.lora_rank, bias=False)
            self.wv_lora_B = nn.Linear(
                self.lora_rank, self.num_kv_heads * self.head_dim, bias=False
            )
            nn.init.normal_(self.wq_lora_A.weight, std=0.006)
            nn.init.zeros_(self.wq_lora_B.weight)
            nn.init.normal_(self.wv_lora_A.weight, std=0.006)
            nn.init.zeros_(self.wv_lora_B.weight)

    def forward(
        self, x, freqs_cos, freqs_sin, cu_seqlens=None, max_seqlen=None, block_mask=None
    ):
        if self.finetune:
            B, S, _ = x.shape
            q = (
                self.wq(x)
                + self.wq_lora_B(self.wq_lora_A(self.lora_dropout(x)))
                * self.lora_scaling
            )
            k = self.wk(x)
            v = (
                self.wv(x)
                + self.wv_lora_B(self.wv_lora_A(self.lora_dropout(x)))
                * self.lora_scaling
            )
            q = q.view(B, S, self.num_heads, self.head_dim)
            k = k.view(B, S, self.num_kv_heads, self.head_dim)
            v = v.view(B, S, self.num_kv_heads, self.head_dim)
            q = apply_rotary_emb(q, freqs_cos, freqs_sin)
            k = apply_rotary_emb(k, freqs_cos, freqs_sin)
            q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
            if self.num_kv_heads != self.num_heads:
                num_groups = self.num_heads // self.num_kv_heads
                k = k.repeat_interleave(num_groups, dim=1)
                v = v.repeat_interleave(num_groups, dim=1)
            attn_out = flex_attention(q, k, v, block_mask=block_mask)
            return self.wo(attn_out.transpose(1, 2).contiguous().view(B, S, -1))
        else:
            total_tokens = x.shape[0]
            q = self.wq(x).view(total_tokens, self.num_heads, self.head_dim)
            k = self.wk(x).view(total_tokens, self.num_kv_heads, self.head_dim)
            v = self.wv(x).view(total_tokens, self.num_kv_heads, self.head_dim)
            q = apply_rotary_emb(q, freqs_cos, freqs_sin)
            k = apply_rotary_emb(k, freqs_cos, freqs_sin)
            attn_out = flash_attn_varlen_func(
                q,
                k,
                v,
                cu_seqlens_q=cu_seqlens,
                cu_seqlens_k=cu_seqlens,
                max_seqlen_q=max_seqlen,
                max_seqlen_k=max_seqlen,
                causal=False,
            )
            return self.wo(attn_out.view(total_tokens, -1))


class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = FlashAttention(config)
        self.feed_forward = SwiGLU(config["embed_dim"], config["intermediate_dim"])
        self.attention_norm = RMSNorm(config["embed_dim"])
        self.ffn_norm = RMSNorm(config["embed_dim"])

    def forward(
        self, x, freqs_cos, freqs_sin, cu_seqlens=None, max_seqlen=None, block_mask=None
    ):
        h = x + self.attention(
            self.attention_norm(x),
            freqs_cos,
            freqs_sin,
            cu_seqlens,
            max_seqlen,
            block_mask,
        )
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


class Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        config = {
            "num_layers": config["num_layers"],
            "num_heads": config["num_heads"],
            "num_kv_heads": config["num_kv_heads"],
            "embed_dim": config["embed_dim"],
            "intermediate_dim": config["intermediate_dim"],
            # we split each token into an item and action token
            "max_seq_len": 2 * config["max_sequence_length"],
            "finetune": config["finetune"],
        }
        self.config = config
        self.layers = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config["num_layers"])]
        )
        self.norm = RMSNorm(config["embed_dim"])
        head_dim = config["embed_dim"] // config["num_heads"]
        freqs_cos, freqs_sin = precompute_freqs_cis(head_dim, config["max_seq_len"])
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    def forward(
        self, input_embeddings, position_ids, document_ids=None, token_mask_ids=None
    ):
        if self.config["finetune"]:
            m, n = document_ids.shape

            def document_mask(b, h, q_idx, kv_idx):
                return document_ids[b][q_idx] == document_ids[b][kv_idx]

            def token_mask(b, h, q_idx, kv_idx):
                return (token_mask_ids[b][kv_idx] == 0) | (
                    token_mask_ids[b][q_idx] == token_mask_ids[b][kv_idx]
                )

            def causal_mask(b, h, q_idx, kv_idx):
                return q_idx >= kv_idx

            maskfn = and_masks(document_mask, token_mask, causal_mask)  # TODO cleanup
            block_mask = create_block_mask(maskfn, B=m, H=None, Q_LEN=n, KV_LEN=n)
            freqs_cos, freqs_sin = (
                self.freqs_cos[position_ids],
                self.freqs_sin[position_ids],
            )
            h = input_embeddings
            for layer in self.layers:
                h = layer(h, freqs_cos, freqs_sin, block_mask=block_mask)
            return self.norm(h)
        else:
            B, S, H = input_embeddings.shape
            flat_embeds = input_embeddings.view(B * S, H)
            flat_pos_ids = position_ids.view(B * S)
            max_doc_id = document_ids.max() + 1
            batch_offsets = (
                torch.arange(B, device=input_embeddings.device).unsqueeze(1)
                * max_doc_id
            )
            global_doc_ids = (document_ids + batch_offsets).view(-1)
            changes = global_doc_ids[1:] != global_doc_ids[:-1]
            boundaries = torch.nonzero(changes).squeeze(-1) + 1
            cu_seqlens = torch.cat(
                [
                    torch.tensor(
                        [0], dtype=torch.int32, device=input_embeddings.device
                    ),
                    boundaries.to(torch.int32),
                    torch.tensor(
                        [B * S], dtype=torch.int32, device=input_embeddings.device
                    ),
                ]
            )
            max_seqlen_in_batch = self.config["max_seq_len"]
            freqs_cos = self.freqs_cos[flat_pos_ids]
            freqs_sin = self.freqs_sin[flat_pos_ids]
            h = flat_embeds
            for layer in self.layers:
                h = layer(h, freqs_cos, freqs_sin, cu_seqlens, max_seqlen_in_batch)
            h = self.norm(h)
            return h.view(B, S, H)


class RecommenderModel(nn.Module):
    def __init__(self, config):
        super(RecommenderModel, self).__init__()
        self.config = config
        self.action_embedding = ActionEmbedding(config)
        self.item_embedding = ItemEmbedding(config)
        self.transformers = Transformer(config)

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
            for name, param in self.transformers.named_parameters():
                param.requires_grad = "lora_" in name
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
        assert W.shape[1] <= weights.shape[1] < W.shape[1] + 128
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

        assert "rope_input_pos" not in d
        b, s = d["userid"].shape
        d["rope_input_pos"] = torch.arange(s, device=d["userid"].device).unsqueeze(0).repeat(b, 1)
        return d

    def to_embedding(self, d):
        e_a = self.action_embedding(d)
        e_i = self.item_embedding(d["matchedid"])
        e = self.interleave(e_i, e_a)
        userid = self.interleave(d["userid"], d["userid"])
        rope_input_pos = self.interleave(
            2 * d["rope_input_pos"],
            2 * d["rope_input_pos"] + 1,
        )
        if self.config["finetune"]:
            token_mask_ids = self.interleave(d["token_mask_ids"], d["token_mask_ids"])
        else:
            token_mask_ids = None
        return self.transformers(
            e,
            position_ids=rope_input_pos,
            document_ids=userid,
            token_mask_ids=token_mask_ids,
        )

    def train_forward(self, d, evaluate):
        if not self.config["finetune"]:
            for k in d:
                d[k] = d[k].reshape(-1, self.config["max_sequence_length"])
        d = self.mask_tokens(d)
        embeds = self.to_embedding(d)
        e0 = embeds[:, 0::2]
        e1 = embeds[:, 1::2]
        if self.config["finetune"]:
            topk = self.config["local_batch_size"]
        else:
            topk = self.config["mask_topk"] * self.config["local_batch_size"]
        losses = []
        for medium in ALL_MEDIUMS:
            for metric in ALL_METRICS:
                w = d[f"{medium}.{metric}.weight"]
                l = d[f"{medium}.{metric}.label"]
                p = d[f"{medium}.{metric}.position"]
                e = e0 if metric == "watch" else e1
                w = d[f"{medium}.{metric}.weight"]
                _, bp = torch.topk(w.reshape(-1), k=topk, sorted=False)
                embed = e.reshape(-1, e.shape[-1])[bp, :]
                labels = l.reshape(-1)[bp]
                positions = p.reshape(-1)[bp]
                weights = w.reshape(-1)[bp]
                if metric == "watch":
                    w_sum = weights.sum().clamp(min=1e-8)
                    logits = self.watch_head((embed, medium))
                    target = positions.to(torch.long)
                    ce_loss = F.cross_entropy(logits, target, reduction="none")
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
