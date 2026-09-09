"""KV-only drafter used by the local KVShot reproduction.

The implementation follows the pure-KV path in *When Hidden States Drift*:

* verified-prefix memory is copied from a few target-model KV layers;
* a learned linear projection reduces the concatenated target layers to one
  GQA KV space;
* later speculative positions append draft-generated KV states; and
* draft queries are produced from token embeddings without target hidden
  states.

This is intentionally a small HuggingFace reference implementation.  It is
not yet a fused serving kernel.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class KVShotConfig:
    hidden_size: int = 4096
    intermediate_size: int = 12288
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    head_dim: int = 128
    num_draft_layers: int = 2
    num_target_kv_layers: int = 3
    draft_vocab_size: int = 32000
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1_000_000.0
    rope_fix: bool = True

    @property
    def kv_width(self) -> int:
        return self.num_key_value_heads * self.head_dim


class RMSNorm(nn.Module):
    def __init__(self, width: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        variance = x.float().square().mean(dim=-1, keepdim=True)
        return (x.float() * torch.rsqrt(variance + self.eps)).to(dtype) * self.weight


class SwiGLU(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    left, right = x.chunk(2, dim=-1)
    return torch.cat((-right, left), dim=-1)


def rope_cos_sin(
    position_ids: torch.Tensor,
    head_dim: int,
    theta: float,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    inv_freq = 1.0 / (
        theta
        ** (
            torch.arange(0, head_dim, 2, device=position_ids.device).float()
            / head_dim
        )
    )
    angles = position_ids.float().unsqueeze(-1) * inv_freq
    angles = torch.cat((angles, angles), dim=-1)
    return angles.cos().to(dtype), angles.sin().to(dtype)


def apply_rope(
    states: torch.Tensor,
    position_ids: torch.Tensor,
    theta: float,
    *,
    inverse: bool = False,
) -> torch.Tensor:
    """Apply Qwen/Llama RoPE to ``[batch, heads, seq, head_dim]`` states."""

    cos, sin = rope_cos_sin(position_ids, states.shape[-1], theta, states.dtype)
    cos = cos[:, None, :, :]
    sin = sin[:, None, :, :]
    if inverse:
        sin = -sin
    return states * cos + rotate_half(states) * sin


def repeat_kv(states: torch.Tensor, repetitions: int) -> torch.Tensor:
    if repetitions == 1:
        return states
    batch, kv_heads, seq_len, head_dim = states.shape
    return (
        states[:, :, None, :, :]
        .expand(batch, kv_heads, repetitions, seq_len, head_dim)
        .reshape(batch, kv_heads * repetitions, seq_len, head_dim)
    )


class MultiLayerKVProjector(nn.Module):
    """Project several target-layer KV spaces to one draft-visible KV space."""

    def __init__(self, config: KVShotConfig) -> None:
        super().__init__()
        input_width = config.num_target_kv_layers * config.kv_width
        self.config = config
        self.k_proj = nn.Linear(input_width, config.kv_width, bias=False)
        self.v_proj = nn.Linear(input_width, config.kv_width, bias=False)
        self.reset_as_layer_select(config.num_target_kv_layers // 2)

    @torch.no_grad()
    def reset_as_layer_select(self, layer_index: int) -> None:
        """Start from one real target layer instead of a destructive random mix."""

        self.k_proj.weight.zero_()
        self.v_proj.weight.zero_()
        width = self.config.kv_width
        identity = torch.eye(width, dtype=self.k_proj.weight.dtype)
        start = layer_index * width
        self.k_proj.weight[:, start : start + width].copy_(identity)
        self.v_proj.weight[:, start : start + width].copy_(identity)

    def forward(
        self,
        target_keys: torch.Tensor,
        target_values: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Inputs: [batch, sampled_layers, kv_heads, seq, head_dim].
        if target_keys.ndim != 5 or target_values.shape != target_keys.shape:
            raise ValueError(
                "target KV must have matching [B, Ls, Hkv, T, D] shapes, got "
                f"{tuple(target_keys.shape)} and {tuple(target_values.shape)}"
            )
        if target_keys.shape[1] != self.config.num_target_kv_layers:
            raise ValueError(
                f"expected {self.config.num_target_kv_layers} sampled KV layers, "
                f"got {target_keys.shape[1]}"
            )

        keys = target_keys
        if self.config.rope_fix:
            batch, sampled, heads, seq_len, head_dim = keys.shape
            keys = apply_rope(
                keys.reshape(batch * sampled, heads, seq_len, head_dim),
                position_ids.repeat_interleave(sampled, dim=0),
                self.config.rope_theta,
                inverse=True,
            ).reshape_as(keys)

        # Keep layer-major flattening, matching [KV_l1; KV_l2; ...].
        batch, sampled, heads, seq_len, head_dim = keys.shape
        keys_flat = (
            keys.permute(0, 3, 1, 2, 4)
            .contiguous()
            .reshape(batch, seq_len, sampled * heads * head_dim)
        )
        values_flat = (
            target_values.permute(0, 3, 1, 2, 4)
            .contiguous()
            .reshape(batch, seq_len, sampled * heads * head_dim)
        )
        keys = self.k_proj(keys_flat).view(batch, seq_len, heads, head_dim)
        values = self.v_proj(values_flat).view(batch, seq_len, heads, head_dim)
        keys = keys.transpose(1, 2).contiguous()
        values = values.transpose(1, 2).contiguous()

        if self.config.rope_fix:
            keys = apply_rope(keys, position_ids, self.config.rope_theta)
        return keys, values


class KVShotLayer(nn.Module):
    def __init__(self, config: KVShotConfig) -> None:
        super().__init__()
        self.config = config
        self.prefix_projector = MultiLayerKVProjector(config)
        self.attn_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.draft_k_proj = nn.Linear(config.hidden_size, config.kv_width, bias=False)
        self.draft_v_proj = nn.Linear(config.hidden_size, config.kv_width, bias=False)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.post_attention_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.mlp = SwiGLU(config.hidden_size, config.intermediate_size)

    def project_prefix(
        self,
        target_keys: torch.Tensor,
        target_values: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.prefix_projector(target_keys, target_values, position_ids)

    def forward_step(
        self,
        hidden: torch.Tensor,
        prefix_kv: tuple[torch.Tensor, torch.Tensor],
        draft_kv: tuple[torch.Tensor, torch.Tensor] | None,
        position_ids: torch.Tensor,
        *,
        append_draft_kv: bool,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        residual = hidden
        normed = self.attn_norm(hidden)
        batch, seq_len, _ = normed.shape
        if seq_len != 1:
            raise ValueError("KVShot autoregressive step expects exactly one token")

        q = self.q_proj(normed).view(
            batch, seq_len, self.config.num_attention_heads, self.config.head_dim
        )
        q = q.transpose(1, 2).contiguous()
        q = apply_rope(q, position_ids, self.config.rope_theta)

        next_draft_kv = draft_kv
        if append_draft_kv:
            key = self.draft_k_proj(normed).view(
                batch,
                seq_len,
                self.config.num_key_value_heads,
                self.config.head_dim,
            )
            value = self.draft_v_proj(normed).view_as(key)
            key = key.transpose(1, 2).contiguous()
            value = value.transpose(1, 2).contiguous()
            key = apply_rope(key, position_ids, self.config.rope_theta)
            if draft_kv is not None:
                key = torch.cat((draft_kv[0], key), dim=2)
                value = torch.cat((draft_kv[1], value), dim=2)
            next_draft_kv = (key, value)

        memory_k, memory_v = prefix_kv
        if next_draft_kv is not None:
            memory_k = torch.cat((memory_k, next_draft_kv[0]), dim=2)
            memory_v = torch.cat((memory_v, next_draft_kv[1]), dim=2)

        groups = self.config.num_attention_heads // self.config.num_key_value_heads
        memory_k = repeat_kv(memory_k, groups)
        memory_v = repeat_kv(memory_v, groups)
        attn = F.scaled_dot_product_attention(q, memory_k, memory_v, is_causal=False)
        attn = attn.transpose(1, 2).reshape(batch, seq_len, self.config.hidden_size)
        hidden = residual + self.o_proj(attn)
        hidden = hidden + self.mlp(self.post_attention_norm(hidden))
        return hidden, next_draft_kv


class KVShotDraft(nn.Module):
    """Autoregressive pure-KV drafter with a reduced EAGLE vocabulary."""

    def __init__(self, config: KVShotConfig) -> None:
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList(
            [KVShotLayer(config) for _ in range(config.num_draft_layers)]
        )
        self.final_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.lm_head = nn.Linear(
            config.hidden_size, config.draft_vocab_size, bias=False
        )
        self.register_buffer(
            "draft_to_target", torch.arange(config.draft_vocab_size), persistent=True
        )

    @torch.no_grad()
    def initialize_from_eagle3(self, checkpoint_path: str | Path) -> None:
        """Warm-start common blocks from the local Qwen3-8B EAGLE-3 head."""

        state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if "d2t" in state:
            offsets = state["d2t"].long()
            self.draft_to_target.copy_(
                torch.arange(offsets.numel(), dtype=torch.long) + offsets
            )

        for layer in self.layers:
            # EAGLE concatenates [token_embedding, target_hidden] before QKV.
            layer.q_proj.weight.copy_(
                state["midlayer.self_attn.q_proj.weight"][:, : self.config.hidden_size]
            )
            layer.draft_k_proj.weight.copy_(
                state["midlayer.self_attn.k_proj.weight"][:, : self.config.hidden_size]
            )
            layer.draft_v_proj.weight.copy_(
                state["midlayer.self_attn.v_proj.weight"][:, : self.config.hidden_size]
            )
            layer.o_proj.weight.copy_(state["midlayer.self_attn.o_proj.weight"])
            layer.mlp.gate_proj.weight.copy_(state["midlayer.mlp.gate_proj.weight"])
            layer.mlp.up_proj.weight.copy_(state["midlayer.mlp.up_proj.weight"])
            layer.mlp.down_proj.weight.copy_(state["midlayer.mlp.down_proj.weight"])
            layer.attn_norm.weight.copy_(state["midlayer.input_layernorm.weight"])
            layer.post_attention_norm.weight.copy_(
                state["midlayer.post_attention_layernorm.weight"]
            )
        self.final_norm.weight.copy_(state["norm.weight"])
        self.lm_head.weight.copy_(state["lm_head.weight"])

    @property
    def selected_target_ids(self) -> torch.Tensor:
        return self.draft_to_target

    def prepare_prefix(
        self,
        target_keys: torch.Tensor,
        target_values: torch.Tensor,
        prefix_length: int,
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        keys = target_keys[..., :prefix_length, :]
        values = target_values[..., :prefix_length, :]
        batch = keys.shape[0]
        positions = torch.arange(prefix_length, device=keys.device).expand(batch, -1)
        return [layer.project_prefix(keys, values, positions) for layer in self.layers]

    def step(
        self,
        input_ids: torch.Tensor,
        embed_tokens: nn.Module,
        prefix_memories: Sequence[tuple[torch.Tensor, torch.Tensor]],
        draft_memories: Sequence[tuple[torch.Tensor, torch.Tensor] | None],
        position: int,
        *,
        prefix_has_current_token: bool,
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor] | None]]:
        hidden = embed_tokens(input_ids)
        batch = input_ids.shape[0]
        position_ids = torch.full(
            (batch, 1), position, dtype=torch.long, device=input_ids.device
        )
        next_memories: list[tuple[torch.Tensor, torch.Tensor] | None] = []
        for layer, prefix_kv, draft_kv in zip(
            self.layers, prefix_memories, draft_memories
        ):
            hidden, next_kv = layer.forward_step(
                hidden,
                prefix_kv,
                draft_kv,
                position_ids,
                append_draft_kv=not prefix_has_current_token,
            )
            next_memories.append(next_kv)
        return self.lm_head(self.final_norm(hidden)), next_memories

    def unroll_teacher_forced(
        self,
        input_ids: torch.Tensor,
        embed_tokens: nn.Module,
        target_keys: torch.Tensor,
        target_values: torch.Tensor,
        cut_position: int,
        length: int,
    ) -> torch.Tensor:
        prefix = self.prepare_prefix(target_keys, target_values, cut_position + 1)
        draft_memories: list[tuple[torch.Tensor, torch.Tensor] | None] = [
            None for _ in self.layers
        ]
        logits = []
        for step_index in range(length):
            step_logits, draft_memories = self.step(
                input_ids[:, cut_position + step_index : cut_position + step_index + 1],
                embed_tokens,
                prefix,
                draft_memories,
                cut_position + step_index,
                prefix_has_current_token=step_index == 0,
            )
            logits.append(step_logits)
        return torch.cat(logits, dim=1)

    @torch.no_grad()
    def propose(
        self,
        current_token: torch.Tensor,
        embed_tokens: nn.Module,
        target_keys: torch.Tensor,
        target_values: torch.Tensor,
        prefix_length: int,
        length: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        prefix = self.prepare_prefix(target_keys, target_values, prefix_length)
        memories: list[tuple[torch.Tensor, torch.Tensor] | None] = [
            None for _ in self.layers
        ]
        proposals = []
        draft_logits = []
        token = current_token
        for step_index in range(length):
            logits, memories = self.step(
                token,
                embed_tokens,
                prefix,
                memories,
                prefix_length - 1 + step_index,
                prefix_has_current_token=step_index == 0,
            )
            draft_id = logits[:, -1].argmax(dim=-1)
            token = self.draft_to_target[draft_id, None]
            proposals.append(token)
            draft_logits.append(logits)
        return torch.cat(proposals, dim=1), torch.cat(draft_logits, dim=1)

    def save_checkpoint(self, output_dir: str | Path) -> None:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), output / "kvshot_draft.pt")
        (output / "kvshot_config.json").write_text(
            json.dumps(asdict(self.config), indent=2) + "\n"
        )

    @classmethod
    def load_checkpoint(
        cls, output_dir: str | Path, *, map_location: str | torch.device = "cpu"
    ) -> "KVShotDraft":
        output = Path(output_dir)
        config = KVShotConfig(
            **json.loads((output / "kvshot_config.json").read_text())
        )
        model = cls(config)
        model.load_state_dict(
            torch.load(
                output / "kvshot_draft.pt",
                map_location=map_location,
                weights_only=True,
            )
        )
        return model


def stack_sampled_target_kv(
    past_key_values: object, layer_ids: Sequence[int]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return sampled HF cache layers as ``[B, Ls, Hkv, T, D]`` tensors."""

    layers = getattr(past_key_values, "layers", None)
    keys = []
    values = []
    for layer_id in layer_ids:
        if layers is not None:
            layer = layers[layer_id]
            key = layer.keys
            value = layer.values
        else:
            key, value = past_key_values[layer_id][:2]
        keys.append(key.detach())
        values.append(value.detach())
    return torch.stack(keys, dim=1), torch.stack(values, dim=1)
