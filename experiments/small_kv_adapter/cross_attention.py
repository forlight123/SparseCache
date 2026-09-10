"""Block-wise cross-attention from a small drafter into sparse Target KV.

Unlike the cache-residual pilot, this adapter gives every speculative query an
explicit read path to the already-arrived Target memory.  The frozen small
model still supplies token embeddings, local self-attention, and its LM head;
only the five memory bridges are trained.
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from experiments.kvshot.model import apply_rope


@dataclass(frozen=True)
class CrossAttentionConfig:
    target_layer_ids: tuple[int, ...] = (1, 9, 17, 25, 33)
    draft_layer_ids: tuple[int, ...] = (1, 7, 14, 21, 27)
    hidden_size: int = 1024
    num_key_value_heads: int = 8
    head_dim: int = 128
    rope_theta: float = 1_000_000.0
    initial_gate: float = 0.05

    def __post_init__(self) -> None:
        if not self.target_layer_ids or len(self.target_layer_ids) != len(
            self.draft_layer_ids
        ):
            raise ValueError("Target and draft layer maps must be nonempty and aligned")
        if len(set(self.target_layer_ids)) != len(self.target_layer_ids):
            raise ValueError("Target layer IDs must be distinct")
        if len(set(self.draft_layer_ids)) != len(self.draft_layer_ids):
            raise ValueError("draft layer IDs must be distinct")
        if min(self.target_layer_ids + self.draft_layer_ids) < 0:
            raise ValueError("layer IDs cannot be negative")
        if self.hidden_size != self.num_key_value_heads * self.head_dim:
            raise ValueError("cross-attention hidden width must equal Hkv * head_dim")
        if not 0.0 <= self.initial_gate <= 1.0:
            raise ValueError("initial gate must lie in [0, 1]")


class TargetKVMemoryBridge(nn.Module):
    """One query block that reads one native Target-KV layer."""

    def __init__(self, config: CrossAttentionConfig) -> None:
        super().__init__()
        self.config = config
        width = config.hidden_size
        self.input_norm = nn.LayerNorm(width)
        self.q_proj = nn.Linear(width, width, bias=False)
        # Shared head-wise maps retain the eight Target GQA head identities.
        self.k_proj = nn.Linear(config.head_dim, config.head_dim, bias=False)
        self.v_proj = nn.Linear(config.head_dim, config.head_dim, bias=False)
        self.o_proj = nn.Linear(width, width, bias=False)
        self.gate = nn.Parameter(torch.tensor(config.initial_gate))

    def project_memory(
        self,
        target_key: torch.Tensor,
        target_value: torch.Tensor,
        positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch = target_key.shape[0]
        position_ids = positions[None].expand(batch, -1)
        key = apply_rope(
            target_key,
            position_ids,
            self.config.rope_theta,
            inverse=True,
        )
        key = F.rms_norm(self.k_proj(key), (self.config.head_dim,))
        key = apply_rope(key, position_ids, self.config.rope_theta)
        value = self.v_proj(target_value)
        return key, value

    def forward(
        self,
        hidden: torch.Tensor,
        memory: tuple[torch.Tensor, torch.Tensor],
        query_positions: torch.Tensor,
    ) -> torch.Tensor:
        batch, query_length, _ = hidden.shape
        query = self.q_proj(self.input_norm(hidden)).view(
            batch,
            query_length,
            self.config.num_key_value_heads,
            self.config.head_dim,
        )
        query = query.transpose(1, 2).contiguous()
        query = F.rms_norm(query, (self.config.head_dim,))
        position_ids = query_positions.reshape(1, -1).expand(batch, -1)
        query = apply_rope(query, position_ids, self.config.rope_theta)
        key, value = memory
        attention = F.scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=0.0,
            is_causal=False,
        )
        attention = attention.transpose(1, 2).reshape(batch, query_length, -1)
        return hidden + torch.tanh(self.gate) * self.o_proj(attention)


class SparseTargetKVCrossAttention(nn.Module):
    """Install temporary memory hooks into selected frozen draft layers."""

    def __init__(self, config: CrossAttentionConfig) -> None:
        super().__init__()
        self.config = config
        self.bridges = nn.ModuleList(
            [TargetKVMemoryBridge(config) for _ in config.target_layer_ids]
        )

    def _validate_target(
        self,
        target_keys: torch.Tensor,
        target_values: torch.Tensor,
        positions: torch.Tensor,
    ) -> None:
        expected = (
            len(self.config.target_layer_ids),
            self.config.num_key_value_heads,
            positions.numel(),
            self.config.head_dim,
        )
        if target_keys.ndim != 5 or target_values.shape != target_keys.shape:
            raise ValueError("Target KV must have matching [B,L,H,T,D] shapes")
        if tuple(target_keys.shape[1:]) != expected:
            raise ValueError(
                f"Target KV shape {tuple(target_keys.shape[1:])} != {expected}"
            )
        if positions.dtype != torch.long or positions.ndim != 1:
            raise ValueError("positions must be one-dimensional torch.long")

    def prepare_memory(
        self,
        target_keys: torch.Tensor,
        target_values: torch.Tensor,
        positions: torch.Tensor,
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
        self._validate_target(target_keys, target_values, positions)
        return tuple(
            bridge.project_memory(
                target_keys[:, index], target_values[:, index], positions
            )
            for index, bridge in enumerate(self.bridges)
        )

    @contextmanager
    def activate(
        self,
        model: nn.Module,
        memory: tuple[tuple[torch.Tensor, torch.Tensor], ...],
        query_positions: torch.Tensor,
    ):
        if len(memory) != len(self.bridges):
            raise ValueError("prepared-memory depth differs from bridge depth")
        if query_positions.dtype != torch.long or query_positions.ndim != 1:
            raise ValueError("query positions must be one-dimensional torch.long")
        decoder_layers = model.model.layers
        if max(self.config.draft_layer_ids) >= len(decoder_layers):
            raise ValueError("adapter draft layer lies outside the small model")
        handles = []
        for slot, draft_layer_id in enumerate(self.config.draft_layer_ids):

            def inject(module, args, kwargs, *, bridge_slot=slot):
                del module
                if args:
                    updated = self.bridges[bridge_slot](
                        args[0], memory[bridge_slot], query_positions
                    )
                    return (updated, *args[1:]), kwargs
                updated_kwargs = dict(kwargs)
                updated_kwargs["hidden_states"] = self.bridges[bridge_slot](
                    kwargs["hidden_states"], memory[bridge_slot], query_positions
                )
                return args, updated_kwargs

            handles.append(
                decoder_layers[draft_layer_id].register_forward_pre_hook(
                    inject, with_kwargs=True
                )
            )
        try:
            yield
        finally:
            for handle in handles:
                handle.remove()

    def save_checkpoint(self, output: str | Path) -> None:
        destination = Path(output)
        destination.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), destination / "cross_attention_adapter.pt")
        config = asdict(self.config)
        config["target_layer_ids"] = list(self.config.target_layer_ids)
        config["draft_layer_ids"] = list(self.config.draft_layer_ids)
        (destination / "cross_attention_config.json").write_text(
            json.dumps(config, indent=2) + "\n", encoding="utf-8"
        )

    @classmethod
    def load_checkpoint(cls, source: str | Path):
        root = Path(source)
        raw = json.loads((root / "cross_attention_config.json").read_text())
        raw["target_layer_ids"] = tuple(raw["target_layer_ids"])
        raw["draft_layer_ids"] = tuple(raw["draft_layer_ids"])
        model = cls(CrossAttentionConfig(**raw))
        model.load_state_dict(
            torch.load(
                root / "cross_attention_adapter.pt",
                map_location="cpu",
                weights_only=True,
            )
        )
        return model
