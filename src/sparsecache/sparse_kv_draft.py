# SPDX-License-Identifier: Apache-2.0
"""A lightweight drafter that cross-attends exact, sparse target-model KV."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import random
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .rope import apply_rope


@dataclass(frozen=True)
class SparseKVDraftConfig:
    """Architecture fields that are independent of one target checkpoint."""

    target_hidden_size: int = 4096
    draft_hidden_size: int = 512
    head_dim: int = 128
    num_draft_heads: int = 4
    num_memory_heads: int = 8
    num_memory_layers: int = 3
    num_seed_layers: int = 3
    num_blocks: int = 1
    mlp_ratio: int = 4
    rms_norm_eps: float = 1e-5

    def __post_init__(self) -> None:
        if self.draft_hidden_size != self.num_draft_heads * self.head_dim:
            raise ValueError(
                "draft_hidden_size must equal num_draft_heads * head_dim"
            )
        for name in (
            "target_hidden_size",
            "num_memory_heads",
            "num_memory_layers",
            "num_seed_layers",
            "num_blocks",
            "mlp_ratio",
        ):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SparseKVDraftConfig":
        return cls(**payload)


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        variance = hidden.float().pow(2).mean(dim=-1, keepdim=True)
        normalized = hidden * torch.rsqrt(variance + self.eps).to(hidden.dtype)
        return normalized * self.weight.to(hidden.dtype)


class CausalDraftAttention(nn.Module):
    def __init__(self, config: SparseKVDraftConfig) -> None:
        super().__init__()
        self.num_heads = config.num_draft_heads
        self.head_dim = config.head_dim
        hidden_size = config.draft_hidden_size
        self.qkv = nn.Linear(hidden_size, 3 * hidden_size, bias=False)
        self.output = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(
        self,
        hidden: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        batch, tokens, _ = hidden.shape
        query, key, value = self.qkv(hidden).chunk(3, dim=-1)

        def heads(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.view(batch, tokens, self.num_heads, self.head_dim).transpose(
                1, 2
            )

        query = apply_rope(heads(query), cos, sin, sequence_dim=2)
        key = apply_rope(heads(key), cos, sin, sequence_dim=2)
        value = heads(value)
        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=0.0,
            is_causal=True,
        )
        return self.output(attended.transpose(1, 2).reshape(batch, tokens, -1))


class SparseTargetKVAttention(nn.Module):
    """Issue learned queries into a few exact target KV layers."""

    def __init__(self, config: SparseKVDraftConfig) -> None:
        super().__init__()
        hidden_size = config.draft_hidden_size
        memory_width = config.num_memory_heads * config.head_dim
        self.num_heads = config.num_memory_heads
        self.head_dim = config.head_dim
        self.num_memory_layers = config.num_memory_layers
        self.query = nn.ModuleList(
            nn.Linear(hidden_size, memory_width, bias=False)
            for _ in range(config.num_memory_layers)
        )
        self.output = nn.ModuleList(
            nn.Linear(memory_width, hidden_size, bias=False)
            for _ in range(config.num_memory_layers)
        )
        self.layer_gates = nn.Parameter(torch.zeros(config.num_memory_layers))

    def forward(
        self,
        hidden: torch.Tensor,
        memory_keys: torch.Tensor,
        memory_values: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        layer_logits_bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if memory_keys.shape != memory_values.shape:
            raise ValueError("memory key/value shapes differ")
        expected = (self.num_memory_layers, self.num_heads)
        if memory_keys.ndim != 4 or memory_keys.shape[:2] != expected:
            raise ValueError(
                "memory must have shape [memory_layers, heads, tokens, head_dim]"
            )
        if memory_keys.shape[-1] != self.head_dim or memory_keys.shape[-2] == 0:
            raise ValueError("memory has an invalid token or head dimension")

        batch, tokens, _ = hidden.shape
        layer_logits = self.layer_gates
        if layer_logits_bias is not None:
            if layer_logits_bias.shape != layer_logits.shape:
                raise ValueError("memory layer-logit bias has an invalid shape")
            layer_logits = layer_logits + layer_logits_bias.to(layer_logits.dtype)
        weights = layer_logits.softmax(dim=0)
        fused = torch.zeros_like(hidden)
        for layer_index in range(self.num_memory_layers):
            query = self.query[layer_index](hidden).view(
                batch, tokens, self.num_heads, self.head_dim
            )
            query = query.transpose(1, 2)
            query = apply_rope(query, cos, sin, sequence_dim=2)
            key = memory_keys[layer_index].unsqueeze(0).to(query.dtype)
            value = memory_values[layer_index].unsqueeze(0).to(query.dtype)
            attended = F.scaled_dot_product_attention(
                query,
                key,
                value,
                dropout_p=0.0,
                is_causal=False,
            )
            attended = attended.transpose(1, 2).reshape(batch, tokens, -1)
            fused = fused + weights[layer_index].to(hidden.dtype) * self.output[
                layer_index
            ](attended)
        return fused


class SparseKVDraftBlock(nn.Module):
    def __init__(self, config: SparseKVDraftConfig) -> None:
        super().__init__()
        hidden_size = config.draft_hidden_size
        eps = config.rms_norm_eps
        self.self_norm = RMSNorm(hidden_size, eps)
        self.self_attention = CausalDraftAttention(config)
        self.memory_norm = RMSNorm(hidden_size, eps)
        self.memory_attention = SparseTargetKVAttention(config)
        self.mlp_norm = RMSNorm(hidden_size, eps)
        intermediate = hidden_size * config.mlp_ratio
        self.gate = nn.Linear(hidden_size, intermediate, bias=False)
        self.up = nn.Linear(hidden_size, intermediate, bias=False)
        self.down = nn.Linear(intermediate, hidden_size, bias=False)

    def forward(
        self,
        hidden: torch.Tensor,
        memory_keys: torch.Tensor,
        memory_values: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        hidden = hidden + self.self_attention(self.self_norm(hidden), cos, sin)
        hidden = hidden + self.memory_attention(
            self.memory_norm(hidden), memory_keys, memory_values, cos, sin
        )
        mlp_input = self.mlp_norm(hidden)
        hidden = hidden + self.down(F.silu(self.gate(mlp_input)) * self.up(mlp_input))
        return hidden


class SparseKVDrafter(nn.Module):
    """Predict target tokens from one seed feature and sparse prompt target KV.

    Target input-embedding and LM-head weights are bound at runtime. They are
    frozen, shared with the verifier, and deliberately excluded from this
    module's checkpoint.
    """

    def __init__(self, config: SparseKVDraftConfig) -> None:
        super().__init__()
        self.config = config
        target_hidden = config.target_hidden_size
        draft_hidden = config.draft_hidden_size
        self.embedding_down = nn.Linear(target_hidden, draft_hidden, bias=False)
        self.seed_projection = nn.Linear(
            target_hidden * config.num_seed_layers, draft_hidden, bias=False
        )
        self.stage_projection = nn.Sequential(
            nn.Linear(2, draft_hidden, bias=False),
            nn.SiLU(),
            nn.Linear(draft_hidden, draft_hidden, bias=False),
        )
        self.blocks = nn.ModuleList(
            SparseKVDraftBlock(config) for _ in range(config.num_blocks)
        )
        self.final_norm = RMSNorm(draft_hidden, config.rms_norm_eps)
        self.output_up = nn.Linear(draft_hidden, target_hidden, bias=False)
        object.__setattr__(self, "_target_embedding_weight", None)
        object.__setattr__(self, "_target_lm_head_weight", None)

    def bind_target_weights(
        self,
        embedding_weight: torch.Tensor,
        lm_head_weight: torch.Tensor,
    ) -> None:
        if embedding_weight.ndim != 2 or lm_head_weight.ndim != 2:
            raise ValueError("target embedding and LM-head weights must be matrices")
        expected = self.config.target_hidden_size
        if embedding_weight.shape[1] != expected or lm_head_weight.shape[1] != expected:
            raise ValueError("target shared-weight hidden dimension is incompatible")
        if embedding_weight.shape[0] != lm_head_weight.shape[0]:
            raise ValueError("target embedding and LM-head vocabularies differ")
        object.__setattr__(self, "_target_embedding_weight", embedding_weight.detach())
        object.__setattr__(self, "_target_lm_head_weight", lm_head_weight.detach())

    def _bound_weights(self) -> tuple[torch.Tensor, torch.Tensor]:
        embedding = self._target_embedding_weight
        lm_head = self._target_lm_head_weight
        if embedding is None or lm_head is None:
            raise RuntimeError("bind_target_weights must be called before forward")
        return embedding, lm_head

    def forward(
        self,
        input_ids: torch.Tensor,
        seed_hidden: torch.Tensor,
        memory_keys: torch.Tensor,
        memory_values: torch.Tensor,
        query_cos: torch.Tensor,
        query_sin: torch.Tensor,
        *,
        visible_fraction: float,
        prompt_tokens: int,
        return_features: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise ValueError("the selection model currently requires batch size one")
        if seed_hidden.shape != (
            1,
            self.config.num_seed_layers,
            self.config.target_hidden_size,
        ):
            raise ValueError("seed hidden-state shape is incompatible")
        tokens = input_ids.shape[1]
        if query_cos.shape[-2:] != (tokens, self.config.head_dim):
            raise ValueError("query RoPE shape does not match draft tokens")
        if query_sin.shape != query_cos.shape:
            raise ValueError("query RoPE cosine/sine shapes differ")
        if not 0.0 < visible_fraction <= 1.0 or prompt_tokens <= 0:
            raise ValueError("visibility and prompt length must be positive")

        embedding, lm_head = self._bound_weights()
        embedded = F.embedding(input_ids, embedding)
        hidden = self.embedding_down(embedded)
        seed = self.seed_projection(seed_hidden.reshape(1, -1)).unsqueeze(1)
        hidden[:, :1] = hidden[:, :1] + seed
        stage = torch.tensor(
            [[visible_fraction, math.log2(prompt_tokens + 1) / 17.0]],
            dtype=hidden.dtype,
            device=hidden.device,
        )
        hidden = hidden + self.stage_projection(stage).unsqueeze(1)
        for block in self.blocks:
            hidden = block(
                hidden,
                memory_keys,
                memory_values,
                query_cos,
                query_sin,
            )
        target_space = self.output_up(self.final_norm(hidden))
        logits = F.linear(target_space, lm_head)
        if return_features:
            return logits, target_space
        return logits


def select_visible_token_indices(
    prompt_tokens: int,
    *,
    page_size: int,
    visible_fraction: float,
    seed: int,
    protected_prefix_pages: int = 1,
    protected_suffix_pages: int = 1,
    mode: str = "random",
    page_scores: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return sorted exact-token indices for one sparse page view."""
    if prompt_tokens <= 0 or page_size <= 0:
        raise ValueError("prompt length and page size must be positive")
    if not 0.0 < visible_fraction <= 1.0:
        raise ValueError("visible_fraction must be in (0, 1]")
    num_pages = math.ceil(prompt_tokens / page_size)
    budget = min(num_pages, max(1, math.ceil(num_pages * visible_fraction)))
    protected = list(range(min(protected_prefix_pages, num_pages)))
    suffix_start = max(0, num_pages - protected_suffix_pages)
    protected.extend(range(suffix_start, num_pages))
    selected = set(protected)
    budget = min(num_pages, max(budget, len(selected)))
    candidates = [page for page in range(num_pages) if page not in selected]
    need = budget - len(selected)
    if mode == "random":
        random.Random(seed).shuffle(candidates)
        selected.update(candidates[:need])
    elif mode == "priority":
        if page_scores is None or page_scores.shape != (num_pages,):
            raise ValueError("priority mode requires one score per prompt page")
        ranked = sorted(
            candidates,
            key=lambda page: (-float(page_scores[page].item()), page),
        )
        selected.update(ranked[:need])
    elif mode == "strided":
        if need:
            step = len(candidates) / need
            selected.update(
                candidates[
                    min(int((index + 0.5) * step), len(candidates) - 1)
                ]
                for index in range(need)
            )
    else:
        raise ValueError(f"unsupported page-selection mode: {mode}")
    token_indices = []
    for page in sorted(selected):
        start = page * page_size
        token_indices.extend(range(start, min(prompt_tokens, start + page_size)))
    return torch.tensor(token_indices, dtype=torch.long)


def accepted_prefix_length(proposal: list[int], target: list[int]) -> int:
    """Count matching greedy tokens before the first rejection."""
    for index, (draft_token, target_token) in enumerate(zip(proposal, target)):
        if draft_token != target_token:
            return index
    return min(len(proposal), len(target))
