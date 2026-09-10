"""Inject sparse Target KV as residual memory into a small draft model.

Qwen3-0.6B and Qwen3-8B share an eight-head, 128-dimensional KV geometry.
The adapter exploits that compatibility without pretending their features are
already aligned: Target keys are first returned to their pre-RoPE coordinates,
a small learned K/V residual is produced, and the key residual is rotated back
at the original document position before it is installed in the small model's
cache.  Non-selected cache positions and all small-model weights are untouched.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from experiments.kvshot.model import apply_rope


@dataclass(frozen=True)
class SmallKVAdapterConfig:
    target_layer_ids: tuple[int, ...] = (1, 9, 17, 25, 33)
    draft_layer_ids: tuple[int, ...] = (1, 7, 14, 21, 27)
    num_key_value_heads: int = 8
    head_dim: int = 128
    bottleneck_size: int = 128
    rope_theta: float = 1_000_000.0

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
        if min(self.num_key_value_heads, self.head_dim, self.bottleneck_size) <= 0:
            raise ValueError("adapter dimensions must be positive")


class HeadwiseKVResidual(nn.Module):
    """A shared per-head K/V map with an exact zero-residual initialization."""

    def __init__(self, head_dim: int, bottleneck_size: int) -> None:
        super().__init__()
        width = 2 * head_dim
        self.norm = nn.LayerNorm(width)
        self.down = nn.Linear(width, bottleneck_size, bias=False)
        self.up = nn.Linear(bottleneck_size, width, bias=False)
        nn.init.zeros_(self.up.weight)

    def forward(self, target_key: torch.Tensor, target_value: torch.Tensor):
        features = torch.cat((target_key, target_value), dim=-1)
        residual = self.up(F.silu(self.down(self.norm(features))))
        return residual.chunk(2, dim=-1)


class SparseTargetKVAdapter(nn.Module):
    """Write differentiable sparse residuals into selected draft-cache layers."""

    def __init__(self, config: SmallKVAdapterConfig) -> None:
        super().__init__()
        self.config = config
        self.adapters = nn.ModuleList(
            [
                HeadwiseKVResidual(config.head_dim, config.bottleneck_size)
                for _ in config.target_layer_ids
            ]
        )

    def _validate(
        self,
        cache,
        target_keys: torch.Tensor,
        target_values: torch.Tensor,
        positions: torch.Tensor,
    ) -> None:
        expected_layers = len(self.config.target_layer_ids)
        if target_keys.ndim != 5 or target_values.shape != target_keys.shape:
            raise ValueError("Target KV must have matching [B,L,H,T,D] shapes")
        if target_keys.shape[1] != expected_layers:
            raise ValueError("Target KV layer count differs from adapter configuration")
        if target_keys.shape[2] != self.config.num_key_value_heads:
            raise ValueError("Target KV head count differs from adapter configuration")
        if target_keys.shape[-1] != self.config.head_dim:
            raise ValueError(
                "Target KV head dimension differs from adapter configuration"
            )
        if positions.ndim != 1 or positions.numel() != target_keys.shape[-2]:
            raise ValueError("positions must identify every sparse Target KV token")
        if positions.dtype != torch.long:
            raise ValueError("positions must use torch.long")
        if positions.numel() == 0 or int(positions.min()) < 0:
            raise ValueError("positions must be nonempty and non-negative")
        if max(self.config.draft_layer_ids) >= len(cache.layers):
            raise ValueError("adapter draft layer lies outside the small-model cache")

    def forward(
        self,
        cache,
        target_keys: torch.Tensor,
        target_values: torch.Tensor,
        positions: torch.Tensor,
    ):
        """Modify ``cache`` in place and return it for model-call composition."""

        self._validate(cache, target_keys, target_values, positions)
        batch = target_keys.shape[0]
        rope_positions = positions[None].expand(batch, -1)
        for index, (draft_layer_id, residual) in enumerate(
            zip(self.config.draft_layer_ids, self.adapters, strict=True)
        ):
            layer = cache.layers[draft_layer_id]
            small_keys = layer.keys
            small_values = layer.values
            if small_keys is None or small_values is None:
                raise ValueError("draft cache layer is not initialized")
            if (
                small_keys.shape[:2]
                != (
                    batch,
                    self.config.num_key_value_heads,
                )
                or small_keys.shape[-1] != self.config.head_dim
            ):
                raise ValueError("small-model cache has incompatible KV geometry")
            if int(positions.max()) >= small_keys.shape[-2]:
                raise ValueError("sparse Target position lies outside the draft cache")

            target_key = target_keys[:, index]
            target_value = target_values[:, index]
            target_key_unrotated = apply_rope(
                target_key,
                rope_positions,
                self.config.rope_theta,
                inverse=True,
            )
            delta_key, delta_value = residual(target_key_unrotated, target_value)

            selected_small_key = small_keys.index_select(-2, positions)
            selected_small_value = small_values.index_select(-2, positions)
            rotated_delta_key = apply_rope(
                delta_key,
                rope_positions,
                self.config.rope_theta,
            )
            adapted_key = selected_small_key + rotated_delta_key
            adapted_value = selected_small_value + delta_value
            layer.keys = small_keys.index_copy(-2, positions, adapted_key)
            layer.values = small_values.index_copy(-2, positions, adapted_value)
        return cache

    def save_checkpoint(self, output: str | Path) -> None:
        destination = Path(output)
        destination.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), destination / "small_kv_adapter.pt")
        config = asdict(self.config)
        config["target_layer_ids"] = list(self.config.target_layer_ids)
        config["draft_layer_ids"] = list(self.config.draft_layer_ids)
        (destination / "small_kv_adapter_config.json").write_text(
            json.dumps(config, indent=2) + "\n", encoding="utf-8"
        )

    @classmethod
    def load_checkpoint(cls, source: str | Path) -> SparseTargetKVAdapter:
        root = Path(source)
        raw = json.loads((root / "small_kv_adapter_config.json").read_text())
        raw["target_layer_ids"] = tuple(raw["target_layer_ids"])
        raw["draft_layer_ids"] = tuple(raw["draft_layer_ids"])
        model = cls(SmallKVAdapterConfig(**raw))
        model.load_state_dict(
            torch.load(
                root / "small_kv_adapter.pt", map_location="cpu", weights_only=True
            )
        )
        return model
