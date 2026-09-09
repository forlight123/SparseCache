"""Reference attention whose per-row arithmetic matches qlen=1 decoding.

The projections and MLP remain block-parallel.  Only the attention reductions
are split into independent query rows, each over the causally visible KV
prefix.  This is an unfused correctness prototype for a future CUDA kernel.
"""

from __future__ import annotations

from contextlib import contextmanager

import torch
from torch.nn import functional as F
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.qwen3.modeling_qwen3 import repeat_kv


def shape_invariant_attention_forward(
    module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    dropout: float = 0.0,
    **_kwargs,
):
    """Evaluate every query with the same matmul shapes as sequential decode."""

    keys = repeat_kv(key, module.num_key_value_groups)
    values = repeat_kv(value, module.num_key_value_groups)
    query_length = query.shape[-2]
    prefix_length = keys.shape[-2] - query_length
    if prefix_length < 0:
        raise ValueError("KV sequence cannot be shorter than the query block")
    outputs = []
    weights = []
    for row in range(query_length):
        visible = prefix_length + row + 1
        score = torch.matmul(
            query[:, :, row:row + 1], keys[:, :, :visible].transpose(2, 3)
        ) * scaling
        if attention_mask is not None:
            score = score + attention_mask[:, :, row:row + 1, :visible]
        probability = F.softmax(score, dim=-1, dtype=torch.float32).to(query.dtype)
        probability = F.dropout(probability, p=dropout, training=module.training)
        outputs.append(torch.matmul(probability, values[:, :, :visible]))
        weights.append(probability)
    output = torch.cat(outputs, dim=2).transpose(1, 2).contiguous()
    # Consumers in this project do not request attention weights. Padding them
    # would add work and can obscure the verifier's measured critical path.
    return output, None


def register_shape_invariant_attention() -> str:
    name = "sparsecache_shape_invariant"
    if name not in ALL_ATTENTION_FUNCTIONS:
        ALL_ATTENTION_FUNCTIONS.register(name, shape_invariant_attention_forward)
    return name


@contextmanager
def row_invariant_verifier_ops(target):
    """Keep the remaining shape-sensitive ops equal to qlen=1 decode.

    This Python prototype intentionally emits one call per query row.  A fused
    kernel should map rows to independent program instances while preserving
    each row's reduction order.
    """

    originals = []
    modules = [target.model.norm]
    for layer in target.model.layers:
        modules.extend([
            layer.input_layernorm,
            layer.self_attn.q_norm,
            layer.self_attn.k_norm,
            layer.post_attention_layernorm,
            layer.mlp.down_proj,
        ])
    for module in modules:
        original = module.forward
        originals.append((module, original))

        def forward(hidden, original=original):
            if hidden.ndim != 3:
                # q_norm/k_norm see [B,Q,H,D], with the same query dimension.
                if hidden.ndim != 4:
                    raise ValueError("row-invariant op expects [B,Q,...]")
            return torch.cat([
                # Calling the original bound method is essential: it preserves
                # the endpoint implementation as well as the qlen=1 shape.
                original(hidden[:, row:row + 1])
                for row in range(hidden.shape[1])
            ], dim=1)

        module.forward = forward
    try:
        yield
    finally:
        for module, original in originals:
            module.forward = original
