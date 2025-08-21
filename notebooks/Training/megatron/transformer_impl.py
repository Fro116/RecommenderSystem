# transformer_impl.py
from __future__ import annotations

import contextlib
import warnings
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

warnings.filterwarnings("ignore", ".*Apex is not installed*")

from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_with_transformer_engine_spec,
)
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.enums import AttnBackend, AttnMaskType
from megatron.core.transformer.spec_utils import build_module
from megatron.core.transformer.transformer_config import TransformerConfig
from transformer_engine.common.recipe import DelayedScaling, MXFP8BlockScaling
from transformer_engine.pytorch import RMSNorm


@dataclass
class TransformerStackConfig:
    # Model size (~0.5B active params)
    num_layers: int = 24
    hidden_size: int = 1536
    ffn_hidden_size: int = 4096
    num_attention_heads: int = 16
    num_query_groups: int = 8
    # Dropouts
    dropout: float = 0.0
    attn_dropout: float = 0.0
    # MoE
    use_moe: bool = False
    num_moe_experts: int = 8
    moe_router_topk: int = 2
    moe_grouped_gemm: bool = True
    moe_aux_loss_coeff: float = 1e-2
    # FP8
    use_fp8: bool = True
    fp8_wgrad: bool = True
    # Attention
    causal: Optional[bool] = None
    attn_backend: AttnBackend = AttnBackend.auto
    rotary_base: float = 10000.0


def _cuda_cc(device_index: Optional[int] = None) -> Tuple[int, int]:
    if not torch.cuda.is_available():
        return (0, 0)
    dev = torch.cuda.current_device() if device_index is None else device_index
    return torch.cuda.get_device_capability(dev)


def _supports_mxfp8() -> bool:
    major, _ = _cuda_cc()
    return major >= 10  # Blackwell or newer


def _make_layer_config(cfg: TransformerStackConfig) -> TransformerConfig:
    return TransformerConfig(
        # Disable model sharding across GPUs
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        context_parallel_size=1,
        # Transformer sizes
        num_layers=cfg.num_layers,
        hidden_size=cfg.hidden_size,
        num_attention_heads=cfg.num_attention_heads,
        num_query_groups=cfg.num_query_groups,
        ffn_hidden_size=cfg.ffn_hidden_size,
        hidden_dropout=cfg.dropout,
        attention_dropout=cfg.attn_dropout,
        attention_backend=cfg.attn_backend,
        # Precision
        bf16=True,
        params_dtype=torch.bfloat16,
        normalization="RMSNorm",
        fp8="hybrid" if cfg.use_fp8 else None,
        fp8_wgrad=cfg.fp8_wgrad,
        fp8_dot_product_attention=True,
        fp8_multi_head_attention=True,
        attention_softmax_in_fp32=True,
        # MLP activation: SwiGLU (gated MLP + SiLU)
        gated_linear_unit=True,
        activation_func=F.silu,
        # MoE (always on)
        num_moe_experts=cfg.num_moe_experts,
        moe_router_topk=cfg.moe_router_topk,
        moe_aux_loss_coeff=cfg.moe_aux_loss_coeff,
        moe_grouped_gemm=cfg.moe_grouped_gemm,
        moe_ffn_hidden_size=cfg.ffn_hidden_size,
        add_bias_linear=False,
        add_qkv_bias=False,
    )


def _rle_lengths_1d(ids_1d: torch.Tensor) -> torch.Tensor:
    """Run-length encode contiguous equal IDs -> lengths per segment"""
    changes = torch.ones_like(ids_1d, dtype=torch.bool)
    changes[1:] = ids_1d[1:] != ids_1d[:-1]
    starts = torch.nonzero(changes, as_tuple=False).flatten()
    last = torch.tensor([ids_1d.numel()], device=ids_1d.device, dtype=starts.dtype)
    ends = torch.cat([starts[1:], last])
    lens = (ends - starts).to(torch.int32)
    return lens


def build_packed_seq_params_from_ids_concat(ids: torch.Tensor) -> PackedSeqParams:
    assert ids.dim() == 2, "ids must be [B, S]"
    B, S = ids.shape
    max_id = int(ids.max().item()) if ids.numel() > 0 else 0
    stride = max_id + 1
    offsets = torch.arange(B, device=ids.device, dtype=ids.dtype).view(B, 1) * stride
    flat_ids = (ids + offsets).reshape(-1)  # [B*S]
    lens = _rle_lengths_1d(flat_ids)
    cu = torch.zeros(lens.numel() + 1, dtype=torch.int32, device=ids.device)
    cu[1:] = torch.cumsum(lens, dim=0)
    max_len = int(lens.max().item()) if lens.numel() > 0 else 0
    return PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=cu,
        cu_seqlens_kv=cu,
        cu_seqlens_q_padded=cu,
        cu_seqlens_kv_padded=cu,
        max_seqlen_q=max_len,
        max_seqlen_kv=max_len,
    )


def build_rope_from_positions(
    positions: torch.Tensor,
    head_dim: int,
    base: float = 10000.0,
    out_dtype: torch.dtype = torch.bfloat16,
) -> Tuple[torch.Tensor, torch.Tensor]:
    B, S = positions.shape
    device = positions.device
    dtype = torch.float32
    half = head_dim // 2
    inv_freq = 1.0 / (
        base ** (torch.arange(0, half, device=device, dtype=dtype) / half)
    )
    theta = positions.to(dtype).unsqueeze(-1) * inv_freq.unsqueeze(0).unsqueeze(
        0
    )  # [B,S,half]
    cos = torch.cos(theta)
    sin = torch.sin(theta)
    cos_full = torch.zeros(B, S, head_dim, device=device, dtype=dtype)
    sin_full = torch.zeros(B, S, head_dim, device=device, dtype=dtype)
    cos_full[..., :half] = cos
    cos_full[..., half:] = cos
    sin_full[..., :half] = sin
    sin_full[..., half:] = sin
    cos_bshd = cos_full.permute(1, 0, 2).unsqueeze(2).to(out_dtype)  # [S,B,1,D]
    sin_bshd = sin_full.permute(1, 0, 2).unsqueeze(2).to(out_dtype)
    return cos_bshd, sin_bshd


class TransformerStack(nn.Module):
    def __init__(self, cfg: TransformerStackConfig):
        super().__init__()
        self.cfg = cfg
        assert cfg.causal is not None
        if cfg.use_moe:
            layer_spec = get_gpt_layer_with_transformer_engine_spec(
                cfg.num_moe_experts, cfg.moe_grouped_gemm
            )
        else:
            layer_spec = get_gpt_layer_with_transformer_engine_spec()
        layer_cfg = _make_layer_config(cfg)
        layers: List[nn.Module] = []
        for i in range(cfg.num_layers):
            lyr = build_module(layer_spec, config=layer_cfg, layer_number=i + 1)
            layers.append(lyr)
        self.layers = nn.ModuleList(layers)
        self.final_norm = RMSNorm(cfg.hidden_size)
        for lyr in self.layers:
            self._set_layer_causality(lyr, cfg.causal)
        if _supports_mxfp8():
            self._fp8_recipe = MXFP8BlockScaling()
        else:
            self._fp8_recipe = DelayedScaling()

    def _set_layer_causality(self, layer: nn.Module, causal: bool) -> None:
        desired = AttnMaskType.causal if causal else AttnMaskType.padding
        sa = getattr(layer, "self_attention", None)
        if sa is not None:
            if hasattr(sa, "attn_mask_type"):
                sa.attn_mask_type = desired
            ca = getattr(sa, "core_attention", None)
            if ca is not None and hasattr(ca, "attn_mask_type"):
                ca.attn_mask_type = desired
            win = (-1, 0) if causal else (-1, -1)
            if hasattr(sa, "window_size"):
                sa.window_size = win
            if ca is not None and hasattr(ca, "window_size"):
                ca.window_size = win

    def forward(
        self,
        embeddings: torch.Tensor,
        ids: torch.Tensor,
        input_pos: torch.Tensor,
    ) -> torch.Tensor:
        assert embeddings.dim() == 3, "embeddings must be [B,S,H]"
        B, S, H = embeddings.shape
        assert ids.shape == (B, S)
        assert input_pos.shape == (B, S)
        assert (
            H % self.cfg.num_attention_heads == 0
        ), "hidden_size must be divisible by num_attention_heads"

        x = embeddings.reshape(B * S, 1, H).contiguous()
        psp = build_packed_seq_params_from_ids_concat(ids)
        # RoPE cos/sin for flattened positions
        pos_flat = input_pos.reshape(1, B * S)
        head_dim = H // self.cfg.num_attention_heads
        assert head_dim % 2 == 0, "per-head dimension must be even for RoPE"
        cos, sin = build_rope_from_positions(
            positions=pos_flat,
            head_dim=head_dim,
            base=self.cfg.rotary_base,
            out_dtype=embeddings.dtype,
        )
        for layer in self.layers:
            out = layer(
                x,
                attention_mask=None,
                rotary_pos_emb=(cos, sin),
                packed_seq_params=psp,
            )
            x = out[0] if isinstance(out, tuple) else out
        x = self.final_norm(x)
        x = x.reshape(B, S, H).contiguous()
        return x