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
        time_emb = (
            ((ts - min_ts) / (max_ts - min_ts)).to(torch.float32).reshape(*ts.shape, 1)
        )
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
    def __init__(self, config, medium):
        super().__init__()
        self.medium = medium
        vocab_size = config["vocab_sizes"][f"{medium}_matchedid"]
        embed_dim = config["embed_dim"]
        text_dim = config["text_emb_size"]
        self.matchedid_embedding = MaskedEmbedding(vocab_size, embed_dim)
        self.text_embedding = MaskedEmbedding(vocab_size, text_dim)
        for p in self.text_embedding.parameters():
            p.requires_grad = False
        self.projection_layer = nn.Linear(text_dim, embed_dim)

    def forward(self, x):
        matched = self.matchedid_embedding(x)
        text = self.text_embedding(x)
        proj = self.projection_layer(text)
        return matched + proj


class DualItemEmbedding(nn.Module):
    def __init__(self, item_embedding: ItemEmbedding):
        super().__init__()
        self.item_embedding = item_embedding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        F = nn.functional
        m1 = self.item_embedding.matchedid_embedding.embedding.weight
        m2 = self.item_embedding.text_embedding.embedding.weight
        W = self.item_embedding.projection_layer.weight
        b = self.item_embedding.projection_layer.bias
        logits1 = F.linear(x, m1)
        logits2 = F.linear(x.matmul(W), m2)
        bias = x.matmul(b).unsqueeze(-1)
        return logits1 + logits2 + bias


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
        aux_losses = torch.tensor(0.0, device=x.device, dtype=x.dtype)
        return x, aux_losses

class MixtureOfExperts(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int,
        num_experts: int = 8,
        top_k: int = 2,
        loss_coef: float = 1e-2,
        capacity_factor: float = 1.25,
    ):
        super().__init__()
        assert top_k <= num_experts
        assert hidden_dim % top_k == 0
        hidden_dim = hidden_dim // top_k
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.loss_coef = loss_coef
        self.capacity_factor = capacity_factor

        self.gate = nn.Linear(embed_dim, num_experts, bias=False)
        E, H, Hh = num_experts, embed_dim, hidden_dim
        self.W_up   = nn.Parameter(torch.empty(E, H,  Hh))
        self.W_gate = nn.Parameter(torch.empty(E, H,  Hh))
        self.W_down = nn.Parameter(torch.empty(E, Hh, H))

    def forward(self, x: torch.Tensor):
        B, S, H = x.shape
        N = B * S
        E = self.num_experts
        K = self.top_k
        Hh = self.hidden_dim
        device = x.device
        dtype = x.dtype

        x_flat = x.reshape(N, H)
        router_logits = self.gate(x_flat)
        routing_probs = F.softmax(router_logits, dim=-1, dtype=torch.float32)
        topk_vals, topk_idx = torch.topk(routing_probs, K, dim=-1)
        topk_vals = (topk_vals / topk_vals.sum(dim=-1, keepdim=True)).to(dtype)
        mean_router_probs = routing_probs.mean(dim=0)
        counts = torch.bincount(topk_idx.reshape(-1), minlength=E)
        fraction_tokens_per_expert = counts.to(routing_probs.dtype) / float(N)
        aux_loss = self.loss_coef * E * (fraction_tokens_per_expert * mean_router_probs).sum()
        token_idx = torch.arange(N, device=device).unsqueeze(1).expand(N, K).reshape(-1)
        expert_idx = topk_idx.reshape(-1)
        weights    = topk_vals.reshape(-1)

        order = torch.argsort(expert_idx)
        expert_idx = expert_idx[order]
        token_idx  = token_idx[order]
        weights    = weights[order]
        per_e_counts = torch.bincount(expert_idx, minlength=E)
        offsets = torch.zeros(E + 1, device=device, dtype=torch.long)
        offsets[1:] = torch.cumsum(per_e_counts, dim=0)
        total_assign = int(expert_idx.numel())
        C = max(1, math.ceil(self.capacity_factor * total_assign / E))
        starts = offsets[expert_idx]
        positions = torch.arange(total_assign, device=device, dtype=torch.long)
        ranks = positions - starts
        keep = ranks < C
        expert_idx = expert_idx[keep]
        token_idx  = token_idx[keep]
        weights    = weights[keep]
        ranks      = ranks[keep]

        EC = E * C
        x_pad = x_flat.new_zeros((E, C, H))
        flat_pos = expert_idx * C + ranks
        x_pad.view(EC, H).index_copy_(0, flat_pos, x_flat.index_select(0, token_idx))
        up   = torch.bmm(x_pad, self.W_up)
        gate = torch.bmm(x_pad, self.W_gate)
        act  = F.silu(gate) * up
        y_pad = torch.bmm(act, self.W_down)
        y_flat = y_pad.view(EC, H).index_select(0, flat_pos)
        y_flat = y_flat * weights.unsqueeze(-1)
        out_flat = torch.zeros_like(x_flat)
        out_flat.scatter_add_(0, token_idx.unsqueeze(-1).expand_as(y_flat), y_flat)
        out = out_flat.view(B, S, H)
        return out, aux_loss

class TransformerLayer(nn.Module):
    def __init__(self, attn, mlp, sa_norm, mlp_norm):
        super().__init__()
        self.attn = attn
        self.mlp = mlp
        self.sa_norm = sa_norm or nn.Identity()
        self.mlp_norm = mlp_norm or nn.Identity()

    def forward(self, x, mask=None, input_pos=None):
        h = self.sa_norm(x)
        attn_out = self.attn(h, h, mask=mask, input_pos=input_pos)
        h = attn_out + x
        mlp_out = self.mlp(self.mlp_norm(h))
        if isinstance(mlp_out, tuple):
            mlp_out, aux_loss = mlp_out
        else:
            aux_loss = 0
        out = h + mlp_out
        return out, aux_loss


class TransformerStack(nn.Module):
    def __init__(self, config):
        super(TransformerStack, self).__init__()
        embed_dim = config["embed_dim"]
        num_kv_heads = config["num_kv_heads"]
        num_heads = config["num_heads"]
        max_seq_len = config["max_sequence_length"]
        if config["causal"]:
            # we split each token into an item and action token
            max_seq_len *= 2
        hidden_dim = config["intermediate_dim"]
        assert num_heads % num_kv_heads == 0
        assert embed_dim % num_heads == 0
        head_dim = embed_dim // num_heads
        rope = torchtune.modules.RotaryPositionalEmbeddings(
            dim=head_dim,
            max_seq_len=max_seq_len,
            base=500_000,
        )
        layers = nn.ModuleList()
        for layer_idx in range(config["num_layers"]):
            if config["finetune"]:
                q_proj = torchtune.modules.peft.LoRALinear(
                    embed_dim,
                    num_heads * head_dim,
                    rank=8,
                    alpha=16,
                    dropout=0.1
                )
                k_proj = nn.Linear(embed_dim, num_kv_heads * head_dim, bias=False)
                v_proj = torchtune.modules.peft.LoRALinear(
                    embed_dim,
                    num_kv_heads * head_dim,
                    rank=8,
                    alpha=16,
                    dropout=0.1
                )
            else:
                q_proj = nn.Linear(embed_dim, num_heads * head_dim, bias=False)
                k_proj = nn.Linear(embed_dim, num_kv_heads * head_dim, bias=False)
                v_proj = nn.Linear(embed_dim, num_kv_heads * head_dim, bias=False)
            self_attn = torchtune.modules.MultiHeadAttention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                q_proj=q_proj,
                k_proj=k_proj,
                v_proj=v_proj,
                output_proj=nn.Linear(embed_dim, embed_dim, bias=False),
                pos_embeddings=rope,
                max_seq_len=max_seq_len,
                attn_dropout=0,
            )
            if layer_idx < 1:
                mlp = torchtune.modules.FeedForward(
                    gate_proj=nn.Linear(embed_dim, hidden_dim, bias=False),
                    down_proj=nn.Linear(hidden_dim, embed_dim, bias=False),
                    up_proj=nn.Linear(embed_dim, hidden_dim, bias=False),
                )
            else:
                mlp = MixtureOfExperts(
                    embed_dim=embed_dim,
                    hidden_dim=hidden_dim
                )
            layer = TransformerLayer(
                attn=self_attn,
                mlp=mlp,
                sa_norm=torchtune.modules.RMSNorm(dim=embed_dim, eps=1e-5),
                mlp_norm=torchtune.modules.RMSNorm(dim=embed_dim, eps=1e-5),
            )
            layers.append(layer)
        self.rope = rope
        self.layers = layers
        self.norm = torchtune.modules.RMSNorm(dim=embed_dim, eps=1e-5)
        if config["finetune"]:
            set_trainable_params(self.layers, get_adapter_params(self.layers))

    def forward(self, x, mask, input_pos):
        # TODO should we norm x to make resid stream cleaner
        aux_losses = torch.tensor(0.0, device=x.device, dtype=x.dtype)
        for layer in self.layers:
            x, aux_loss = layer(x, mask=mask, input_pos=input_pos)
            aux_losses += aux_loss
        x = self.norm(x)
        return x, aux_losses


class TransformerModel(nn.Module):
    def __init__(self, config):
        super(TransformerModel, self).__init__()
        self.config = config
        self.action_embedding = ActionEmbedding(config)
        self.item_embeddings = nn.ModuleList(
            [ItemEmbedding(config, m) for m in ALL_MEDIUMS]
        )
        if args.moe:
            self.transformers = TransformerStack(config)
        else:
            self.transformers = Llama3(config)

        def create_lm_head(medium):
            return nn.Sequential(
                DualItemEmbedding(self.item_embeddings[medium]), nn.LogSoftmax(dim=-1)
            )

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
        self.apply(init_weights)
        if config["use_pretrained_embeddings"]:
            self.load_pretrained_embeddings()
        self.empty_loss = nn.Parameter(torch.tensor(0.0))
        if config["finetune"]:
            for layer in [
                self.action_embedding,
                self.item_embeddings[0],
                self.item_embeddings[1],
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

    def load_pretrained_embeddings(self):
        for medium in [0, 1]:
            weights = self.item_embeddings[medium].text_embedding.embedding.weight
            weights.zero_()
            m = {0: "manga", 1: "anime"}[medium]
            with open(f'{args.datadir}/{m}.json', 'r') as f:
                data = json.load(f)
                for x in data:
                    weights[x["matchedid"], :] = torch.tensor(x["embedding"])

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
            elif k in ["time"] or k.startswith("0.watch") or k.startswith("1.watch"):
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
                    mask += d[k] > 0
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
            e_i = torch.where(
                (d["item_tokens"]["0_matchedid"] >= 0).unsqueeze(-1),
                self.item_embeddings[0](d["item_tokens"]["0_matchedid"]),
                self.item_embeddings[1](d["item_tokens"]["1_matchedid"]),
            )
            e = self.interleave(e_a, e_i)
            userid = self.interleave(
                *[d[k]["userid"] for k in ["action_tokens", "item_tokens"]]
            )
            if "rope_input_pos" in d["action_tokens"]:
                rope_input_pos = self.interleave(
                    2 * d["action_tokens"]["rope_input_pos"],
                    2 * d["item_tokens"]["rope_input_pos"] + 1,
                )
                token_mask_ids = self.interleave(
                    d["action_tokens"]["token_mask_ids"],
                    d["item_tokens"]["token_mask_ids"],
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
                return (token_mask_ids[b][kv_idx] == 0) | (
                    token_mask_ids[b][q_idx] == token_mask_ids[b][kv_idx]
                )

            maskfn = and_masks(document_mask, causal_mask, token_mask)
            block_mask = create_block_mask(maskfn, B=m, H=None, Q_LEN=n, KV_LEN=n)
            return self.transformers(e, block_mask, rope_input_pos)
        else:
            userid = d["userid"]
            m, n = userid.shape

            def document_mask(b, h, q_idx, kv_idx):
                return userid[b][q_idx] == userid[b][kv_idx]

            block_mask = create_block_mask(
                document_mask, B=m, H=None, Q_LEN=n, KV_LEN=n
            )
            e_a = self.action_embedding(d)
            e_i = torch.where(
                (d["0_matchedid"] >= 0).unsqueeze(-1),
                self.item_embeddings[0](d["0_matchedid"]),
                self.item_embeddings[1](d["1_matchedid"]),
            )
            e = e_a + e_i
            return self.transformers(e, block_mask, None)

    def train_forward(self, d, evaluate):
        if not self.config["finetune"]:
            for k in d:
                d[k] = d[k].reshape(-1, self.config["max_sequence_length"])
        if self.config["causal"]:
            d = self.split_tokens(d)
        else:
            d = self.mask_tokens(d)
        e, aux_loss = self.to_embedding(d)
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
                                self.item_embeddings[medium](positions.squeeze(dim=1)),
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
        return losses, aux_loss

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