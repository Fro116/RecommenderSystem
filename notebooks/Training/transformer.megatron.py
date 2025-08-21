import math
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from megatron.core import parallel_state
from megatron.core.extensions.transformer_engine_spec_provider import TESpecProvider
from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
from megatron.core.models.backends import BackendSpecProvider
from megatron.core.transformer.attention import SelfAttention, SelfAttentionSubmodules
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.mlp import MLP, MLPSubmodules
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import (
    TransformerLayer,
    TransformerLayerSubmodules,
)
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.attention.flex_attention import and_masks, create_block_mask
from transformer_engine.pytorch import RMSNorm

def get_mlp_module_spec_for_backend(
    backend: BackendSpecProvider,
    use_te_op_fuser: Optional[bool] = False,
) -> ModuleSpec:
    """Helper function to get module spec for MLP/MoE"""

    linear_fc2 = backend.row_parallel_linear()
    # Dense MLP w/ or w/o TE modules.
    if use_te_op_fuser:
        return ModuleSpec(module=TEFusedMLP)
    elif backend.fuse_layernorm_and_linear():
        linear_fc1 = backend.column_parallel_layer_norm_linear()
        assert linear_fc1 is not None
    else:
        linear_fc1 = backend.column_parallel_linear()
    return ModuleSpec(
        module=MLP,
        submodules=MLPSubmodules(linear_fc1=linear_fc1, linear_fc2=linear_fc2),
    )



def get_transformer_layer_spec(
    causal: bool,
    qk_layernorm: Optional[bool] = False,
    qk_l2_norm: Optional[bool] = False,
    use_te_op_fuser: Optional[bool] = False,
) -> ModuleSpec:
    """Use this spec to use lower-level Transformer Engine modules (required for fp8 training).


    Args:
        qk_layernorm (bool, optional): To use layernorm for queries/keys. Defaults to False.
        qk_l2_norm (bool, optional): To use l2 norm for queries/keys. Defaults to False.
        use_te_op_fuser (bool, optional): Use Transformer Engine's operation-based API, which may
                                          enable certain operation fusions. Defaults to False.

    Returns:
        ModuleSpec: Module specification with TE modules

    """

    backend = TESpecProvider()
    mlp = get_mlp_module_spec_for_backend(
        backend=backend,
        use_te_op_fuser=use_te_op_fuser,
    )
    qk_norm = backend.layer_norm(for_qk=True)
    return ModuleSpec(
        module=TransformerLayer,
        submodules=TransformerLayerSubmodules(
            self_attention=ModuleSpec(
                module=SelfAttention,
                params={"attn_mask_type": AttnMaskType.causal if causal else AttnMaskType.no_mask},
                submodules=SelfAttentionSubmodules(
                    linear_qkv=backend.column_parallel_layer_norm_linear(),
                    core_attention=backend.core_attention(),
                    linear_proj=backend.row_parallel_linear(),
                    q_layernorm=(
                        L2Norm
                        if qk_l2_norm
                        else (qk_norm if qk_layernorm else IdentityOp)
                    ),
                    k_layernorm=(
                        L2Norm
                        if qk_l2_norm
                        else (qk_norm if qk_layernorm else IdentityOp)
                    ),
                ),
            ),
            self_attn_bda=get_bias_dropout_add,
            pre_mlp_layernorm=IdentityOp,
            mlp=mlp,
            mlp_bda=get_bias_dropout_add,
            sharded_state_dict_keys_map={
                "mlp.0.weight": "mlp.linear_fc1.layer_norm_weight",
                "mlp.0.bias": "mlp.linear_fc1.layer_norm_bias",
                "mlp.1.basic_ops.0.weight": "mlp.linear_fc1.weight",
                "mlp.1.basic_ops.1.bias": "mlp.linear_fc1.bias",
                "mlp.3.basic_ops.0.weight": "mlp.linear_fc2.weight",
                "mlp.3.basic_ops.1.bias": "mlp.linear_fc2.bias",
            },
        ),
    )

class RotaryEmbedding(nn.Module):
    def __init__(
        self,
        hidden_dim,
        num_heads,
        rotary_base: int = 10000,
    ) -> None:
        super().__init__()
        assert hidden_dim % num_heads == 0
        dim = hidden_dim // num_heads
        inv_freq = 1.0 / (
            rotary_base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)
        )
        self.register_buffer("inv_freq", inv_freq)

    def get_freqs(self, input_pos):
        freqs = torch.einsum(
            "ij,k->ijk", input_pos, self.inv_freq
        )  # [batch, seqlen, dim]
        return freqs

    def forward(self, input_pos):
        freqs = self.get_freqs(input_pos)
        b, s, d = freqs.shape
        emb = torch.stack((freqs.view(b, -1, 1), freqs.view(b, -1, 1)), dim=-1).view(
            b, s, -1
        )  # [batch, seqlen, hidden]
        emb = emb.permute(1, 0, 2).unsqueeze(2)  # [seqlen, batch, heads, hidden]
        return emb


class TransformerStack(nn.Module):
    def __init__(self, config):
        super().__init__()
        layer_config = TransformerConfig(
            # Transformer sizes
            num_layers=config["num_layers"],
            hidden_size=config["embed_dim"],
            num_attention_heads=config["num_heads"],
            num_query_groups=config["num_kv_heads"],
            ffn_hidden_size=config["intermediate_dim"],
            # hidden_dropout=cfg.dropout,
            # attention_dropout=cfg.attn_dropout,
            normalization="RMSNorm",
            # MLP activation: SwiGLU (gated MLP + SiLU)
            gated_linear_unit=True,
            activation_func=F.silu,
            # add_bias_linear=False,
            # add_qkv_bias=False,
        )
        layer_spec = get_transformer_layer_spec(config["causal"])
        layers: List[nn.Module] = []
        for i in range(layer_config.num_layers):
            lyr = build_module(layer_spec, config=layer_config, layer_number=i + 1)
            layers.append(lyr)
        self.layers = nn.ModuleList(layers)
        self.final_norm = RMSNorm(layer_config.hidden_size)
        self.rope = RotaryEmbedding(
            layer_config.hidden_size, layer_config.num_attention_heads
        )

    def mask_to_bias(self, mask):
        # true means tokens can attend. mask has shape [batch, seqlen, seqlen]
        dtype = torch.bfloat16
        neginf = -(2**50) if dtype == torch.bfloat16 else -(2**15)
        bias = torch.zeros(mask.shape, dtype=dtype, device=mask.device)
        bias.masked_fill_(mask == 0, neginf)
        bias.requires_grad = False
        return bias.unsqueeze(1)

    def forward(self, x, mask, input_pos):
        rope = self.rope(input_pos)
        bias = self.mask_to_bias(mask)
        x = x.permute((1, 0, 2))  # [seq, batch, hidden]
        for layer in self.layers:
            x, _ = layer(x, rotary_pos_emb=rope, attention_bias=bias)
        x = self.final_norm(x)
        x = x.permute((1, 0, 2))  # [batch, seq, hidden]
        return x