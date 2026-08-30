# SPDX-License-Identifier: Apache-2.0
"""EAGLE-3 language prior conditioned on exact sparse verifier KV.

By default the pretrained EAGLE trunk remains frozen while a small trainable
adapter issues queries directly into selected K/V tensors from several verifier
layers.  The training driver can optionally unfreeze the EAGLE projection or
recurrent layer for controlled hybrid ablations.  This is deliberately
different from running an unmodified EAGLE model with an incomplete prompt:
sparse verifier KV is an explicit model input and receives gradients.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .rope import apply_rope
from .sparse_kv_draft import RMSNorm, SparseKVDraftConfig, SparseTargetKVAttention


@dataclass(frozen=True)
class SparseKVEagleConfig:
    target_hidden_size: int = 4096
    hidden_size: int = 4096
    intermediate_size: int = 14336
    head_dim: int = 128
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    num_memory_heads: int = 8
    num_memory_layers: int = 3
    num_seed_layers: int = 3
    target_vocab_size: int = 128256
    draft_vocab_size: int = 32000
    rms_norm_eps: float = 1e-5
    fusion_mode: str = "scalar"

    def __post_init__(self) -> None:
        if self.hidden_size != self.num_attention_heads * self.head_dim:
            raise ValueError("hidden size must match the attention head geometry")
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError("query heads must be divisible by key/value heads")
        if self.num_memory_heads * self.head_dim <= 0:
            raise ValueError("memory attention geometry must be positive")
        if min(
            self.target_hidden_size,
            self.intermediate_size,
            self.num_memory_layers,
            self.num_seed_layers,
            self.target_vocab_size,
            self.draft_vocab_size,
        ) <= 0:
            raise ValueError("all architecture sizes must be positive")
        if self.fusion_mode not in {"scalar", "gated_delta"}:
            raise ValueError("fusion_mode must be scalar or gated_delta")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SparseKVEagleConfig":
        return cls(**payload)


class EagleSelfAttention(nn.Module):
    """The first EAGLE-3 layer consumes token embedding + recurrent feature."""

    def __init__(self, config: SparseKVEagleConfig) -> None:
        super().__init__()
        input_size = 2 * config.hidden_size
        query_width = config.num_attention_heads * config.head_dim
        kv_width = config.num_key_value_heads * config.head_dim
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.q_proj = nn.Linear(input_size, query_width, bias=False)
        self.k_proj = nn.Linear(input_size, kv_width, bias=False)
        self.v_proj = nn.Linear(input_size, kv_width, bias=False)
        self.o_proj = nn.Linear(query_width, config.hidden_size, bias=False)

    def forward(
        self,
        hidden: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        past_key: torch.Tensor | None,
        past_value: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, tokens, _ = hidden.shape
        if tokens != 1 or cos.shape[-2:] != (1, self.head_dim):
            raise ValueError("recurrent EAGLE attention expects exactly one position")
        query = self.q_proj(hidden).view(
            batch, tokens, self.num_heads, self.head_dim
        ).transpose(1, 2)
        key = self.k_proj(hidden).view(
            batch, tokens, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        value = self.v_proj(hidden).view(
            batch, tokens, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        query = apply_rope(query, cos, sin, sequence_dim=2)
        key = apply_rope(key, cos, sin, sequence_dim=2)
        if past_key is not None:
            if past_value is None:
                raise ValueError("EAGLE key/value cache is incomplete")
            key = torch.cat((past_key, key), dim=2)
            value = torch.cat((past_value, value), dim=2)
        groups = self.num_heads // self.num_key_value_heads
        attended = F.scaled_dot_product_attention(
            query,
            key.repeat_interleave(groups, dim=1),
            value.repeat_interleave(groups, dim=1),
            dropout_p=0.0,
            is_causal=False,
        )
        output = attended.transpose(1, 2).reshape(batch, tokens, -1)
        return self.o_proj(output), key, value


class EagleMLP(nn.Module):
    def __init__(self, config: SparseKVEagleConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.up_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.down_proj = nn.Linear(
            config.intermediate_size, config.hidden_size, bias=False
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(hidden)) * self.up_proj(hidden))


class EagleDraftLayer(nn.Module):
    def __init__(self, config: SparseKVEagleConfig) -> None:
        super().__init__()
        self.self_attn = EagleSelfAttention(config)
        self.mlp = EagleMLP(config)
        self.hidden_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, config.rms_norm_eps
        )

    def forward(
        self,
        input_embedding: torch.Tensor,
        hidden: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        past_key: torch.Tensor | None,
        past_value: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        residual = hidden
        attention_input = torch.cat(
            (self.input_layernorm(input_embedding), self.hidden_norm(hidden)), dim=-1
        )
        attended, key, value = self.self_attn(
            attention_input, cos, sin, past_key, past_value
        )
        hidden = residual + attended
        hidden = hidden + self.mlp(self.post_attention_layernorm(hidden))
        return hidden, key, value


class SparseKVEagleDrafter(nn.Module):
    """Frozen EAGLE-3 proposal trunk plus a trainable sparse-target-KV adapter."""

    def __init__(self, config: SparseKVEagleConfig) -> None:
        super().__init__()
        self.config = config
        self.midlayer = EagleDraftLayer(config)
        self.fc = nn.Linear(
            config.target_hidden_size * config.num_seed_layers,
            config.hidden_size,
            bias=False,
        )
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.lm_head = nn.Linear(
            config.hidden_size, config.draft_vocab_size, bias=False
        )
        memory_config = SparseKVDraftConfig(
            target_hidden_size=config.target_hidden_size,
            draft_hidden_size=config.hidden_size,
            head_dim=config.head_dim,
            num_draft_heads=config.num_attention_heads,
            num_memory_heads=config.num_memory_heads,
            num_memory_layers=config.num_memory_layers,
            num_seed_layers=config.num_seed_layers,
            num_blocks=1,
            mlp_ratio=1,
            rms_norm_eps=config.rms_norm_eps,
        )
        self.memory_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.memory_attention = SparseTargetKVAttention(memory_config)
        self.memory_gate = nn.Linear(3 * config.hidden_size, config.hidden_size)
        self.memory_delta = nn.Linear(config.hidden_size, config.hidden_size)
        nn.init.zeros_(self.memory_gate.weight)
        nn.init.constant_(self.memory_gate.bias, -1.0)
        nn.init.eye_(self.memory_delta.weight)
        nn.init.zeros_(self.memory_delta.bias)
        if config.fusion_mode == "scalar":
            for parameter in (
                *self.memory_gate.parameters(),
                *self.memory_delta.parameters(),
            ):
                parameter.requires_grad_(False)
        self.adapter_scale = nn.Parameter(torch.tensor(-4.0))
        self.stage_layer_bias = nn.Linear(2, config.num_memory_layers, bias=True)
        nn.init.zeros_(self.stage_layer_bias.weight)
        nn.init.zeros_(self.stage_layer_bias.bias)
        self.register_buffer(
            "d2t", torch.zeros(config.draft_vocab_size, dtype=torch.long)
        )
        self.register_buffer(
            "t2d", torch.zeros(config.target_vocab_size, dtype=torch.bool)
        )
        self.register_buffer(
            "draft_to_target",
            torch.arange(config.draft_vocab_size, dtype=torch.long),
        )
        self.register_buffer(
            "target_to_draft",
            torch.full((config.target_vocab_size,), -1, dtype=torch.long),
        )
        object.__setattr__(self, "_target_embedding_weight", None)

    def bind_target_embedding(self, embedding_weight: torch.Tensor) -> None:
        expected = (self.config.target_vocab_size, self.config.target_hidden_size)
        if embedding_weight.shape != expected:
            raise ValueError("target embedding shape is incompatible with EAGLE")
        object.__setattr__(self, "_target_embedding_weight", embedding_weight.detach())

    def load_eagle_checkpoint(self, checkpoint_path: str | Path) -> None:
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        missing, unexpected = self.load_state_dict(state, strict=False)
        allowed_missing = {
            "adapter_scale",
            "draft_to_target",
            "target_to_draft",
            "memory_norm.weight",
            "memory_attention.layer_gates",
            "stage_layer_bias.weight",
            "stage_layer_bias.bias",
        }
        allowed_missing.update(
            f"memory_attention.{kind}.{index}.weight"
            for kind in ("query", "output")
            for index in range(self.config.num_memory_layers)
        )
        allowed_missing.update(
            {
                "memory_gate.weight",
                "memory_gate.bias",
                "memory_delta.weight",
                "memory_delta.bias",
            }
        )
        if set(missing) != allowed_missing or unexpected:
            raise ValueError(
                "incompatible EAGLE checkpoint: "
                f"missing={missing}, unexpected={unexpected}"
            )
        draft_to_target = torch.arange(
            self.config.draft_vocab_size, device=self.d2t.device
        ) + self.d2t
        if draft_to_target.unique().numel() != draft_to_target.numel():
            raise ValueError("EAGLE compressed-vocabulary mapping is not one-to-one")
        target_to_draft = torch.full_like(self.target_to_draft, -1)
        target_to_draft[draft_to_target] = torch.arange(
            self.config.draft_vocab_size, device=self.d2t.device
        )
        self.draft_to_target.copy_(draft_to_target)
        self.target_to_draft.copy_(target_to_draft)

    def freeze_eagle(self) -> None:
        for module in (self.midlayer, self.fc, self.norm, self.lm_head):
            for parameter in module.parameters():
                parameter.requires_grad_(False)

    def adapter_parameters(self) -> list[nn.Parameter]:
        return [parameter for parameter in self.parameters() if parameter.requires_grad]

    def adapter_state_dict(self) -> dict[str, torch.Tensor]:
        trainable = {
            name
            for name, parameter in self.named_parameters()
            if parameter.requires_grad
        }
        return {
            name: tensor.detach().cpu()
            for name, tensor in self.state_dict().items()
            if name in trainable
        }

    def target_ids(self, draft_ids: torch.Tensor) -> torch.Tensor:
        return self.draft_to_target[draft_ids]

    def draft_ids(self, target_ids: torch.Tensor) -> torch.Tensor:
        return self.target_to_draft[target_ids]

    def _memory_delta(
        self,
        hidden: torch.Tensor,
        memory_keys: torch.Tensor,
        memory_values: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        stage: torch.Tensor,
    ) -> torch.Tensor:
        stage_gates = self.stage_layer_bias(stage.float()).squeeze(0)
        delta = self.memory_attention(
            self.memory_norm(hidden),
            memory_keys,
            memory_values,
            cos,
            sin,
            layer_logits_bias=stage_gates,
        )
        scale = torch.sigmoid(self.adapter_scale).to(hidden.dtype)
        if self.config.fusion_mode == "scalar":
            return scale * delta
        correction = delta - hidden
        gate_input = torch.cat((hidden, delta, correction), dim=-1)
        gate = torch.sigmoid(self.memory_gate(gate_input))
        return scale * gate * self.memory_delta(correction)

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
        use_memory: bool = True,
        return_features: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        embedding = self._target_embedding_weight
        if embedding is None:
            raise RuntimeError(
                "bind_target_embedding must be called before forward"
            )
        if input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise ValueError("sparse EAGLE currently requires batch size one")
        if seed_hidden.shape != (
            1,
            self.config.num_seed_layers,
            self.config.target_hidden_size,
        ):
            raise ValueError("seed hidden-state shape is incompatible")
        tokens = input_ids.shape[1]
        if query_cos.shape != (tokens, self.config.head_dim):
            raise ValueError("query RoPE shape does not match input tokens")
        if query_sin.shape != query_cos.shape:
            raise ValueError("query RoPE cosine/sine shapes differ")
        if not 0.0 < visible_fraction <= 1.0 or prompt_tokens <= 0:
            raise ValueError("invalid sparse-KV stage metadata")
        hidden = self.fc(seed_hidden.reshape(1, -1)).unsqueeze(1)
        past_key = None
        past_value = None
        logits = []
        features = []
        stage = torch.tensor(
            [
                [
                    visible_fraction,
                    torch.tensor(prompt_tokens + 1.0).log2().item() / 17.0,
                ]
            ],
            device=input_ids.device,
        )
        for position in range(tokens):
            token_embedding = F.embedding(
                input_ids[:, position : position + 1], embedding
            )
            hidden, past_key, past_value = self.midlayer(
                token_embedding,
                hidden,
                query_cos[position : position + 1],
                query_sin[position : position + 1],
                past_key,
                past_value,
            )
            if use_memory:
                hidden = hidden + self._memory_delta(
                    hidden,
                    memory_keys,
                    memory_values,
                    query_cos[position : position + 1],
                    query_sin[position : position + 1],
                    stage,
                )
            normalized = self.norm(hidden)
            logits.append(self.lm_head(normalized))
            features.append(normalized)
        all_logits = torch.cat(logits, dim=1)
        if return_features:
            return all_logits, torch.cat(features, dim=1)
        return all_logits
