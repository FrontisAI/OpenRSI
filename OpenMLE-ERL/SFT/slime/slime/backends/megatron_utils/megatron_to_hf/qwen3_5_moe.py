"""Map Qwen3.5 MoE parameters from Megatron torch_dist to Hugging Face.

This module handles the Qwen3.5-35B-A3B text backbone. The model combines
linear and full attention. Hugging Face stores MoE expert weights as grouped
tensors whose first dimension indexes experts, so the older Qwen2/Qwen3
mapping that emits one key per expert cannot be reused.
"""

from __future__ import annotations

import re

import torch


def _convert_full_attention_qkv(args, layer_idx: str, param: torch.Tensor):
    """Split fused QKV weights using the gated Qwen3.5 Q-projection layout."""

    head_dim = args.kv_channels if args.kv_channels is not None else args.hidden_size // args.num_attention_heads
    value_num_per_group = args.num_attention_heads // args.num_query_groups
    param = param.view(args.num_query_groups, -1, head_dim, args.hidden_size)
    q_param, k_param, v_param = torch.split(
        param,
        split_size_or_sections=[2 * value_num_per_group, 1, 1],
        dim=1,
    )
    q_param = (
        q_param.reshape(args.num_query_groups, 2, value_num_per_group, head_dim, args.hidden_size)
        .transpose(1, 2)
        .reshape(-1, args.hidden_size)
    )
    prefix = f"model.language_model.layers.{layer_idx}.self_attn"
    return [
        (f"{prefix}.q_proj.weight", q_param),
        (f"{prefix}.k_proj.weight", k_param.reshape(-1, args.hidden_size)),
        (f"{prefix}.v_proj.weight", v_param.reshape(-1, args.hidden_size)),
    ]


def convert_qwen3_5_moe_to_hf(args, name: str, param: torch.Tensor):
    """Map one Megatron parameter to one or more Qwen3.5 HF parameters."""

    if name == "module.module.embedding.word_embeddings.weight":
        return [("model.language_model.embed_tokens.weight", param)]
    if name == "module.module.output_layer.weight":
        return [("lm_head.weight", param)]
    if name == "module.module.decoder.final_layernorm.weight":
        return [("model.language_model.norm.weight", param)]

    match = re.match(r"module\.module\.decoder\.layers\.(\d+)\.(.+)", name)
    if not match:
        raise ValueError(f"Unknown Qwen3.5 parameter name: {name}")

    layer_idx, rest = match.groups()
    layer_prefix = f"model.language_model.layers.{layer_idx}"

    # Linear-attention weight names match HF; only the layer prefix changes.
    linear_attention_prefix = "self_attention.linear_attn."
    if rest.startswith(linear_attention_prefix):
        suffix = rest.removeprefix(linear_attention_prefix)
        return [(f"{layer_prefix}.linear_attn.{suffix}", param)]

    # Megatron fuses full-attention QKV weights, including the gated Q part.
    if rest == "self_attention.linear_qkv.weight":
        return _convert_full_attention_qkv(args, layer_idx, param)

    direct_names = {
        "self_attention.linear_proj.weight": f"{layer_prefix}.self_attn.o_proj.weight",
        "self_attention.q_layernorm.weight": f"{layer_prefix}.self_attn.q_norm.weight",
        "self_attention.k_layernorm.weight": f"{layer_prefix}.self_attn.k_norm.weight",
        "self_attention.linear_qkv.layer_norm_weight": f"{layer_prefix}.input_layernorm.weight",
        "self_attention.input_layernorm.weight": f"{layer_prefix}.input_layernorm.weight",
        "pre_mlp_layernorm.weight": f"{layer_prefix}.post_attention_layernorm.weight",
        "mlp.router.weight": f"{layer_prefix}.mlp.gate.weight",
        "mlp.experts.experts.linear_fc1.weight": f"{layer_prefix}.mlp.experts.gate_up_proj",
        "mlp.experts.experts.linear_fc2.weight": f"{layer_prefix}.mlp.experts.down_proj",
        "mlp.shared_experts.linear_fc2.weight": f"{layer_prefix}.mlp.shared_expert.down_proj.weight",
        "mlp.shared_experts.gate_weight": f"{layer_prefix}.mlp.shared_expert_gate.weight",
    }
    if rest in direct_names:
        return [(direct_names[rest], param)]

    # HF stores the shared expert's fused gate/up layer as two parameters.
    if rest == "mlp.shared_experts.linear_fc1.weight":
        gate_weight, up_weight = param.chunk(2, dim=0)
        return [
            (f"{layer_prefix}.mlp.shared_expert.gate_proj.weight", gate_weight),
            (f"{layer_prefix}.mlp.shared_expert.up_proj.weight", up_weight),
        ]

    raise ValueError(f"Unknown Qwen3.5 parameter name: {name}")
