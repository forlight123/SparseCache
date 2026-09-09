"""Reference block drafter whose only long-context state is Target KV.

This is the mechanism path for CacheDraft, not a serving implementation.  A
known verifier token and a small block of learned mask embeddings query selected
Target KV layers.  All unknown block positions are processed in parallel, so
the drafter neither prefills the document nor owns document-sized KV state.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file

from experiments.kvshot.model import (
    KVShotConfig,
    KVShotDraft,
    MultiLayerKVProjector,
    RMSNorm,
    SwiGLU,
    apply_rope,
    repeat_kv,
)


def build_packed_attention_mask(
    prefix_lengths: torch.Tensor,
    *,
    context_length: int,
    block_size: int,
    causal_block: bool = False,
) -> torch.Tensor:
    """Return ``[B,1,N*block,S+N*block]`` DFlash-style allow mask."""

    batch, num_blocks = prefix_lengths.shape
    device = prefix_lengths.device
    query_length = num_blocks * block_size
    query_blocks = torch.arange(query_length, device=device) // block_size
    target_indices = torch.arange(context_length, device=device)
    context_mask = target_indices.view(1, 1, -1) < prefix_lengths[:, :, None]
    context_mask = context_mask[:, query_blocks, :]
    block_mask = query_blocks[:, None].eq(query_blocks[None, :])
    if causal_block:
        block_offsets = torch.arange(query_length, device=device) % block_size
        block_mask = block_mask & block_offsets[None, :].le(block_offsets[:, None])
    block_mask = block_mask.view(1, query_length, query_length).expand(
        batch, -1, -1
    )
    return torch.cat((context_mask, block_mask), dim=-1).unsqueeze(1)


@dataclass(frozen=True)
class BlockKVConfig:
    hidden_size: int = 4096
    intermediate_size: int = 12288
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    head_dim: int = 128
    num_draft_layers: int = 5
    num_target_kv_layers: int = 0
    block_size: int = 8
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1_000_000.0
    target_kv_fusion: str = "per_layer"
    shift_labels: bool = False
    causal_block: bool = False
    vocab_size: int = 151936
    correction_hidden_size: int = 0
    correction_bottleneck_size: int = 256
    correction_mode: str = "vocab"
    correction_topk: int = 0
    use_seed_hidden: bool = False
    normalize_projected_keys: bool = False
    tomography_depth_rank: int = 0

    def __post_init__(self) -> None:
        if self.num_target_kv_layers == 0:
            object.__setattr__(self, "num_target_kv_layers", self.num_draft_layers)
        if self.target_kv_fusion not in {
            "per_layer",
            "projected",
            "dflash_hidden",
            "dflash_lowrank_hidden",
            "repair_aware_hidden",
        }:
            raise ValueError(
                "target_kv_fusion must be per_layer, projected, dflash_hidden, "
                "dflash_lowrank_hidden, or repair_aware_hidden"
            )
        if (
            self.target_kv_fusion == "per_layer"
            and self.num_target_kv_layers != self.num_draft_layers
        ):
            raise ValueError("per_layer fusion needs one Target KV per draft layer")
        if self.correction_hidden_size < 0:
            raise ValueError("correction_hidden_size cannot be negative")
        if self.correction_mode not in {"vocab", "rerank"}:
            raise ValueError("correction_mode must be vocab or rerank")
        if self.correction_mode == "rerank" and self.correction_topk <= 0:
            raise ValueError("rerank correction requires a positive top-k")
        if self.correction_topk > self.vocab_size:
            raise ValueError("correction top-k cannot exceed the vocabulary")
        if self.target_kv_fusion == "dflash_lowrank_hidden":
            if self.tomography_depth_rank <= 0:
                raise ValueError(
                    "dflash_lowrank_hidden requires positive tomography_depth_rank"
                )
        elif self.tomography_depth_rank:
            raise ValueError(
                "tomography_depth_rank is only valid for dflash_lowrank_hidden"
            )

    @property
    def kv_width(self) -> int:
        return self.num_key_value_heads * self.head_dim


class TargetKVBlockLayer(nn.Module):
    """One parallel block layer attending to one selected Target KV layer."""

    def __init__(self, config: BlockKVConfig) -> None:
        super().__init__()
        self.config = config
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, config.kv_width, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.kv_width, bias=False)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.q_norm = RMSNorm(config.head_dim, config.rms_norm_eps)
        self.k_norm = RMSNorm(config.head_dim, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, config.rms_norm_eps
        )
        self.mlp = SwiGLU(config.hidden_size, config.intermediate_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        target_key: torch.Tensor,
        target_value: torch.Tensor,
        position_ids: torch.Tensor,
        prefix_lengths: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        return_context_importance: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if target_key.ndim != 4 or target_value.shape != target_key.shape:
            raise ValueError("Target KV must have matching [B,Hkv,T,D] shapes")
        batch, query_length, _ = hidden_states.shape
        if query_length % self.config.block_size:
            raise ValueError(
                "query length must contain an integral number of draft blocks"
            )
        num_blocks = query_length // self.config.block_size
        context_length = target_key.shape[-2]
        if prefix_lengths is None:
            prefix_lengths = torch.full(
                (batch, num_blocks),
                context_length,
                dtype=torch.long,
                device=hidden_states.device,
            )
        if prefix_lengths.shape != (batch, num_blocks):
            raise ValueError("prefix_lengths must have shape [B,num_blocks]")

        residual = hidden_states
        normed = self.input_layernorm(hidden_states)
        query = self.q_proj(normed).view(
            batch,
            query_length,
            self.config.num_attention_heads,
            self.config.head_dim,
        )
        block_key = self.k_proj(normed).view(
            batch,
            query_length,
            self.config.num_key_value_heads,
            self.config.head_dim,
        )
        block_value = self.v_proj(normed).view_as(block_key)

        query = self.q_norm(query).transpose(1, 2).contiguous()
        block_key = self.k_norm(block_key).transpose(1, 2).contiguous()
        block_value = block_value.transpose(1, 2).contiguous()
        query = apply_rope(query, position_ids, self.config.rope_theta)
        block_key = apply_rope(block_key, position_ids, self.config.rope_theta)

        memory_key = torch.cat((target_key, block_key), dim=2)
        memory_value = torch.cat((target_value, block_value), dim=2)
        groups = self.config.num_attention_heads // self.config.num_key_value_heads
        memory_key = repeat_kv(memory_key, groups)
        memory_value = repeat_kv(memory_value, groups)

        # Pack many independently anchored blocks into one short-query
        # attention call. Every block can read only Target positions preceding
        # its own anchor and its own current-token/MASK block. It cannot see
        # future Target KV or another training block.
        if attention_mask is None:
            attention_mask = build_packed_attention_mask(
                prefix_lengths,
                context_length=context_length,
                block_size=self.config.block_size,
            )
        attention = F.scaled_dot_product_attention(
            query,
            memory_key,
            memory_value,
            attn_mask=attention_mask,
            is_causal=False,
        )
        attention = attention.transpose(1, 2).reshape(
            batch, query_length, self.config.hidden_size
        )
        hidden_states = residual + self.o_proj(attention)
        hidden_states = hidden_states + self.mlp(
            self.post_attention_layernorm(hidden_states)
        )
        if return_context_importance:
            # The reference SDPA API does not expose probabilities.  Recompute
            # only qK/softmax for diagnosis; a serving implementation must
            # fuse this reduction into attention or replace it with a learned
            # low-rank utility head.  Positions remain aligned one-to-one with
            # the original Target cache even for learned multi-layer fusion.
            scores = torch.matmul(
                query.float(), memory_key.float().transpose(-1, -2)
            ) * (self.config.head_dim**-0.5)
            scores.masked_fill_(~attention_mask, torch.finfo(scores.dtype).min)
            probabilities = torch.softmax(scores, dim=-1)
            importance = probabilities[..., :context_length].mean(dim=(1, 2))
            importance = importance / importance.sum(dim=-1, keepdim=True).clamp_min(
                1e-12
            )
            return hidden_states, importance
        return hidden_states


class TargetKVHiddenProjector(nn.Module):
    """Map reusable multi-layer Target K/V to shared DFlash context features."""

    def __init__(self, config: BlockKVConfig) -> None:
        super().__init__()
        self.config = config
        input_width = 2 * config.num_target_kv_layers * config.kv_width
        self.proj = nn.Linear(input_width, config.hidden_size, bias=False)
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(
        self,
        target_keys: torch.Tensor,
        target_values: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        if target_keys.ndim != 5 or target_values.shape != target_keys.shape:
            raise ValueError("Target KV must have matching [B,L,Hkv,T,D] shapes")
        batch, sampled, heads, seq_len, head_dim = target_keys.shape
        if sampled != self.config.num_target_kv_layers:
            raise ValueError("Target KV layer count does not match hidden projector")
        unrotated_keys = apply_rope(
            target_keys.reshape(batch * sampled, heads, seq_len, head_dim),
            position_ids.repeat_interleave(sampled, dim=0),
            self.config.rope_theta,
            inverse=True,
        ).reshape_as(target_keys)
        keys_flat = (
            unrotated_keys.permute(0, 3, 1, 2, 4)
            .contiguous()
            .reshape(batch, seq_len, sampled * heads * head_dim)
        )
        values_flat = (
            target_values.permute(0, 3, 1, 2, 4)
            .contiguous()
            .reshape(batch, seq_len, sampled * heads * head_dim)
        )
        return self.norm(self.proj(torch.cat((keys_flat, values_flat), dim=-1)))


class LowRankDepthTargetKVHiddenProjector(nn.Module):
    """Recover DFlash context through a low-rank basis over Target depth.

    A dense concatenation projector has ``O(L * d_kv * d_hidden)`` weights.
    Here each K/V channel learns ``R`` signed mixtures over Target layers,
    followed by one ``R * 2d_kv -> d_hidden`` map.  This preserves information
    from the full cross-depth trajectory while factorizing the projector along
    its layer mode.  With Qwen3-8B, ``R=2`` uses about 16.8M weights instead of
    125.8M for a dense 15-layer concatenation.
    """

    def __init__(self, config: BlockKVConfig) -> None:
        super().__init__()
        self.config = config
        rank = config.tomography_depth_rank
        layer_width = 2 * config.kv_width
        self.depth_mix = nn.Parameter(
            torch.empty(rank, config.num_target_kv_layers, layer_width)
        )
        nn.init.normal_(
            self.depth_mix,
            mean=0.0,
            std=config.num_target_kv_layers**-0.5,
        )
        self.proj = nn.Linear(rank * layer_width, config.hidden_size, bias=False)
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(
        self,
        target_keys: torch.Tensor,
        target_values: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        if target_keys.ndim != 5 or target_values.shape != target_keys.shape:
            raise ValueError("Target KV must have matching [B,L,Hkv,T,D] shapes")
        batch, sampled, heads, seq_len, head_dim = target_keys.shape
        if sampled != self.config.num_target_kv_layers:
            raise ValueError("Target KV layer count does not match hidden projector")
        unrotated_keys = apply_rope(
            target_keys.reshape(batch * sampled, heads, seq_len, head_dim),
            position_ids.repeat_interleave(sampled, dim=0),
            self.config.rope_theta,
            inverse=True,
        ).reshape_as(target_keys)
        keys_by_layer = unrotated_keys.permute(0, 1, 3, 2, 4).reshape(
            batch, sampled, seq_len, self.config.kv_width
        )
        values_by_layer = target_values.permute(0, 1, 3, 2, 4).reshape_as(
            keys_by_layer
        )
        layer_features = torch.cat((keys_by_layer, values_by_layer), dim=-1)
        # Signed, channel-wise unit-norm depth bases can represent both layer
        # averages and finite differences without allowing their scale to
        # absorb the downstream projection's scale.
        depth_mix = self.depth_mix.float()
        depth_mix = depth_mix / depth_mix.square().sum(
            dim=1, keepdim=True
        ).sqrt().clamp_min(1e-6)
        mixed = torch.einsum(
            "bltc,rlc->btrc",
            layer_features,
            depth_mix.to(dtype=layer_features.dtype),
        ).reshape(batch, seq_len, -1)
        return self.norm(self.proj(mixed))


class RepairAwareTargetKVHiddenProjector(nn.Module):
    """Fuse Target layers while preserving token/layer reliability decisions.

    The old DFlash adapter applies one linear map to ``[K_1..K_L,V_1..V_L]``.
    A linear map over a concatenation is exactly a sum of per-layer maps.  This
    module uses that equivalent factorization, then learns a bounded residual
    gate for every token and Target layer.  The gate sees cheap statistics from
    the available K/V plus CacheBlend's recomputation bit; it never needs exact
    K/V, document tokens, or a second long-context state.

    ``initialize_from_concat`` makes all gates exactly one and splits an old
    adapter's weight without approximation.  Consequently a converted
    checkpoint must reproduce the source model before reliability training.
    """

    def __init__(self, config: BlockKVConfig, gate_hidden_size: int = 32) -> None:
        super().__init__()
        self.config = config
        self.layer_projs = nn.ModuleList(
            [
                nn.Linear(2 * config.kv_width, config.hidden_size, bias=False)
                for _ in range(config.num_target_kv_layers)
            ]
        )
        self.feature_proj = nn.Linear(3, gate_hidden_size, bias=False)
        self.layer_embedding = nn.Parameter(
            torch.zeros(config.num_target_kv_layers, gate_hidden_size)
        )
        self.gate_out = nn.Linear(gate_hidden_size, 1, bias=False)
        # Zero output is the identity gate: 1 + tanh(0) == 1.
        nn.init.zeros_(self.gate_out.weight)
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)

    @torch.no_grad()
    def initialize_from_concat(self, source: TargetKVHiddenProjector) -> None:
        """Exactly factor a trained concatenated K/V projection by layer."""

        if source.config.num_target_kv_layers != self.config.num_target_kv_layers:
            raise ValueError("source and destination Target layer counts disagree")
        if source.config.kv_width != self.config.kv_width:
            raise ValueError("source and destination Target KV widths disagree")
        width = self.config.kv_width
        layers = self.config.num_target_kv_layers
        source_weight = source.proj.weight
        for index, destination in enumerate(self.layer_projs):
            destination.weight[:, :width].copy_(
                source_weight[:, index * width : (index + 1) * width]
            )
            value_start = layers * width + index * width
            destination.weight[:, width:].copy_(
                source_weight[:, value_start : value_start + width]
            )
        self.norm.load_state_dict(source.norm.state_dict())
        # Keep the randomly initialized feature path so gate_out receives a
        # gradient on the first step. gate_out itself stays zero, which is
        # sufficient for exact functional equivalence at conversion time.
        nn.init.normal_(self.layer_embedding, mean=0.0, std=0.02)
        self.gate_out.weight.zero_()

    def forward(
        self,
        target_keys: torch.Tensor,
        target_values: torch.Tensor,
        position_ids: torch.Tensor,
        repair_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if target_keys.ndim != 5 or target_values.shape != target_keys.shape:
            raise ValueError("Target KV must have matching [B,L,Hkv,T,D] shapes")
        batch, sampled, heads, seq_len, head_dim = target_keys.shape
        if sampled != self.config.num_target_kv_layers:
            raise ValueError("Target KV layer count does not match repair-aware adapter")
        if repair_mask is None:
            repair_mask = torch.ones(
                (batch, seq_len), dtype=torch.bool, device=target_keys.device
            )
        if repair_mask.shape != (batch, seq_len):
            raise ValueError("repair_mask must have shape [B,T]")

        unrotated_keys = apply_rope(
            target_keys.reshape(batch * sampled, heads, seq_len, head_dim),
            position_ids.repeat_interleave(sampled, dim=0),
            self.config.rope_theta,
            inverse=True,
        ).reshape_as(target_keys)
        keys_by_layer = unrotated_keys.permute(0, 1, 3, 2, 4).reshape(
            batch, sampled, seq_len, self.config.kv_width
        )
        values_by_layer = target_values.permute(0, 1, 3, 2, 4).reshape_as(
            keys_by_layer
        )

        # RMS features are computed in fp32 for stable signals under BF16.
        # log1p keeps rare large norms from dominating the small gate network.
        key_rms = keys_by_layer.float().square().mean(dim=-1).sqrt().log1p()
        value_rms = values_by_layer.float().square().mean(dim=-1).sqrt().log1p()
        repaired = repair_mask[:, None, :].expand(-1, sampled, -1).float()
        gate_features = torch.stack((key_rms, value_rms, repaired), dim=-1).to(
            target_keys.dtype
        )
        gate_hidden = self.feature_proj(gate_features)
        gate_hidden = F.silu(gate_hidden + self.layer_embedding[None, :, None, :])
        gates = 1.0 + torch.tanh(self.gate_out(gate_hidden)).squeeze(-1)

        fused = None
        for index, projector in enumerate(self.layer_projs):
            layer_input = torch.cat(
                (keys_by_layer[:, index], values_by_layer[:, index]), dim=-1
            )
            contribution = projector(layer_input) * gates[:, index, :, None]
            fused = contribution if fused is None else fused + contribution
        assert fused is not None
        return self.norm(fused)


class BlockKVDraft(nn.Module):
    """Predict a complete speculative suffix from selected Target KV layers."""

    def __init__(self, config: BlockKVConfig) -> None:
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList(
            [TargetKVBlockLayer(config) for _ in range(config.num_draft_layers)]
        )
        if config.target_kv_fusion == "projected":
            projector_config = KVShotConfig(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                num_attention_heads=config.num_attention_heads,
                num_key_value_heads=config.num_key_value_heads,
                head_dim=config.head_dim,
                num_draft_layers=config.num_draft_layers,
                num_target_kv_layers=config.num_target_kv_layers,
                rms_norm_eps=config.rms_norm_eps,
                rope_theta=config.rope_theta,
                rope_fix=True,
            )
            self.prefix_projectors = nn.ModuleList(
                [
                    MultiLayerKVProjector(projector_config)
                    for _ in range(config.num_draft_layers)
                ]
            )
        else:
            self.prefix_projectors = None
        if config.target_kv_fusion == "dflash_hidden":
            self.target_hidden_projector = TargetKVHiddenProjector(config)
        elif config.target_kv_fusion == "dflash_lowrank_hidden":
            self.target_hidden_projector = LowRankDepthTargetKVHiddenProjector(config)
        elif config.target_kv_fusion == "repair_aware_hidden":
            self.target_hidden_projector = RepairAwareTargetKVHiddenProjector(config)
        else:
            self.target_hidden_projector = None
        if config.correction_hidden_size:
            self.correction_gru = nn.GRU(
                input_size=config.hidden_size,
                hidden_size=config.correction_hidden_size,
                num_layers=1,
                batch_first=True,
                bias=False,
            )
            self.correction_head = nn.Sequential(
                nn.Linear(
                    config.hidden_size + config.correction_hidden_size,
                    config.correction_bottleneck_size,
                    bias=False,
                ),
                nn.SiLU(),
                nn.Linear(config.correction_bottleneck_size, (
                    config.vocab_size
                    if config.correction_mode == "vocab" else config.hidden_size
                ), bias=False),
            )
            # Adding the causal head to an existing direct-KV checkpoint is an
            # exact behavioral warm start; training learns a residual.
            nn.init.zeros_(self.correction_head[-1].weight)
        else:
            self.correction_gru = None
            self.correction_head = None
        self.final_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.mask_embedding = nn.Parameter(torch.empty(config.hidden_size))
        nn.init.normal_(self.mask_embedding, mean=0.0, std=0.02)
        if config.use_seed_hidden:
            self.seed_hidden_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
            self.seed_hidden_proj = nn.Linear(
                config.hidden_size, config.hidden_size, bias=False
            )
            # A false->true warm start initially reproduces the KV-only model;
            # training then learns only the useful residual from the live
            # Target query feature.
            nn.init.zeros_(self.seed_hidden_proj.weight)
        else:
            self.seed_hidden_norm = None
            self.seed_hidden_proj = None

    @torch.no_grad()
    def initialize_from_target(
        self,
        target: nn.Module,
        target_layer_ids: Sequence[int],
        *,
        mask_token_id: int | None = None,
    ) -> None:
        if len(target_layer_ids) != len(self.layers):
            raise ValueError("one Target layer ID is required per draft layer")
        target_layers = target.model.layers
        for draft_layer, layer_id in zip(self.layers, target_layer_ids):
            source = target_layers[layer_id]
            draft_layer.q_proj.weight.copy_(source.self_attn.q_proj.weight)
            draft_layer.k_proj.weight.copy_(source.self_attn.k_proj.weight)
            draft_layer.v_proj.weight.copy_(source.self_attn.v_proj.weight)
            draft_layer.o_proj.weight.copy_(source.self_attn.o_proj.weight)
            # Qwen3 has per-head Q/K RMSNorm while Llama 3 does not. Keeping
            # the draft norms at their identity initialization gives the
            # latter a compatible path without inventing Target-side state.
            if hasattr(source.self_attn, "q_norm"):
                draft_layer.q_norm.weight.copy_(source.self_attn.q_norm.weight)
            if hasattr(source.self_attn, "k_norm"):
                draft_layer.k_norm.weight.copy_(source.self_attn.k_norm.weight)
            draft_layer.input_layernorm.weight.copy_(source.input_layernorm.weight)
            draft_layer.post_attention_layernorm.weight.copy_(
                source.post_attention_layernorm.weight
            )
            draft_layer.mlp.gate_proj.weight.copy_(source.mlp.gate_proj.weight)
            draft_layer.mlp.up_proj.weight.copy_(source.mlp.up_proj.weight)
            draft_layer.mlp.down_proj.weight.copy_(source.mlp.down_proj.weight)
        self.final_norm.weight.copy_(target.model.norm.weight)
        if mask_token_id is not None:
            if not 0 <= mask_token_id < target.model.embed_tokens.num_embeddings:
                raise ValueError("mask token ID lies outside the Target vocabulary")
            self.mask_embedding.copy_(target.model.embed_tokens.weight[mask_token_id])

    @torch.no_grad()
    def initialize_from_kvshot(
        self,
        checkpoint_path: str | Path,
        *,
        mask_embedding: torch.Tensor | None = None,
    ) -> None:
        """Warm-start the KV adapter/backbone from a trained pure-KV drafter."""

        if self.prefix_projectors is None:
            raise ValueError("KVShot warm start requires projected Target-KV fusion")
        source = KVShotDraft.load_checkpoint(checkpoint_path)
        if source.config.num_target_kv_layers != self.config.num_target_kv_layers:
            raise ValueError("KVShot and block drafter Target-KV counts disagree")
        for index, (layer, projector) in enumerate(
            zip(self.layers, self.prefix_projectors)
        ):
            source_layer = source.layers[index % len(source.layers)]
            projector.load_state_dict(source_layer.prefix_projector.state_dict())
            layer.input_layernorm.load_state_dict(source_layer.attn_norm.state_dict())
            layer.q_proj.load_state_dict(source_layer.q_proj.state_dict())
            layer.k_proj.load_state_dict(source_layer.draft_k_proj.state_dict())
            layer.v_proj.load_state_dict(source_layer.draft_v_proj.state_dict())
            layer.o_proj.load_state_dict(source_layer.o_proj.state_dict())
            layer.post_attention_layernorm.load_state_dict(
                source_layer.post_attention_norm.state_dict()
            )
            layer.mlp.load_state_dict(source_layer.mlp.state_dict())
        self.final_norm.load_state_dict(source.final_norm.state_dict())
        if mask_embedding is not None:
            self.mask_embedding.copy_(mask_embedding)

    @torch.no_grad()
    def initialize_from_dflash(
        self,
        checkpoint_path: str | Path,
        *,
        mask_embedding: torch.Tensor,
    ) -> None:
        """Reuse a trained DFlash block generator with a new KV memory path.

        DFlash's original long-context memory is a projection of complete
        Target hidden histories. CacheDraft cannot construct that history for
        reused RAG chunks, so only the already-trained block/noise backbone is
        imported. The per-layer Target-KV projectors remain freshly initialized
        and are the intended first-stage trainable parameters.
        """

        if self.config.target_kv_fusion == "per_layer":
            raise ValueError("DFlash adaptation requires learned Target-KV fusion")
        source_dir = Path(checkpoint_path)
        source_config = json.loads((source_dir / "config.json").read_text())
        expected = {
            "hidden_size": self.config.hidden_size,
            "intermediate_size": self.config.intermediate_size,
            "num_attention_heads": self.config.num_attention_heads,
            "num_key_value_heads": self.config.num_key_value_heads,
            "head_dim": self.config.head_dim,
            "num_hidden_layers": self.config.num_draft_layers,
            "block_size": self.config.block_size,
        }
        for field, value in expected.items():
            if int(source_config[field]) != value:
                raise ValueError(
                    f"DFlash checkpoint {field} mismatch: "
                    f"{source_config[field]!r} != {value!r}"
                )
        # DFlash's source ``target_layer_ids`` describe the hidden histories
        # used to train its block backbone. They need not equal the number of
        # reusable Target KV layers consumed by a replacement memory adapter:
        # the latter may use extra, already-cached layers to reconstruct the
        # same DFlash memory. The memory teacher separately validates the
        # exact source hidden-layer contract when it is enabled.
        if mask_embedding.shape != (self.config.hidden_size,):
            raise ValueError("DFlash mask embedding must have shape [H]")

        state = load_file(str(source_dir / "model.safetensors"), device="cpu")
        for index, layer in enumerate(self.layers):
            prefix = f"layers.{index}."
            mappings = {
                "input_layernorm.weight": layer.input_layernorm.weight,
                "self_attn.q_proj.weight": layer.q_proj.weight,
                "self_attn.k_proj.weight": layer.k_proj.weight,
                "self_attn.v_proj.weight": layer.v_proj.weight,
                "self_attn.o_proj.weight": layer.o_proj.weight,
                "self_attn.q_norm.weight": layer.q_norm.weight,
                "self_attn.k_norm.weight": layer.k_norm.weight,
                "post_attention_layernorm.weight": (
                    layer.post_attention_layernorm.weight
                ),
                "mlp.gate_proj.weight": layer.mlp.gate_proj.weight,
                "mlp.up_proj.weight": layer.mlp.up_proj.weight,
                "mlp.down_proj.weight": layer.mlp.down_proj.weight,
            }
            for source_name, destination in mappings.items():
                source = state[prefix + source_name]
                if source.shape != destination.shape:
                    raise ValueError(
                        f"DFlash tensor shape mismatch for {prefix + source_name}"
                    )
                destination.copy_(source)
        self.final_norm.weight.copy_(state["norm.weight"])
        self.mask_embedding.copy_(mask_embedding)

    @torch.no_grad()
    def initialize_from_blockdraft(self, checkpoint_path: str | Path) -> None:
        """Warm-start the block backbone while adding a new auxiliary head."""

        source = BlockKVDraft.load_checkpoint(checkpoint_path)
        comparable = (
            "hidden_size",
            "intermediate_size",
            "num_attention_heads",
            "num_key_value_heads",
            "head_dim",
            "num_draft_layers",
            "num_target_kv_layers",
            "block_size",
            "target_kv_fusion",
            "shift_labels",
            "causal_block",
            "normalize_projected_keys",
        )
        for field in comparable:
            if getattr(source.config, field) != getattr(self.config, field):
                raise ValueError(f"block warm start changes incompatible field {field}")
        result = self.load_state_dict(source.state_dict(), strict=False)
        allowed_missing = {
            name
            for name, _ in self.named_parameters()
            if name.startswith("correction_gru.")
            or name.startswith("correction_head.")
            or name.startswith("seed_hidden_norm.")
            or name.startswith("seed_hidden_proj.")
        }
        if set(result.missing_keys) != allowed_missing or result.unexpected_keys:
            raise RuntimeError(
                "unexpected block warm-start mismatch: "
                f"missing={result.missing_keys}, unexpected={result.unexpected_keys}"
            )

    @torch.no_grad()
    def initialize_repair_aware_from_blockdraft(
        self, checkpoint_path: str | Path
    ) -> None:
        """Convert a trained concatenated hidden adapter without changing logits."""

        if self.config.target_kv_fusion != "repair_aware_hidden":
            raise ValueError("destination must use repair_aware_hidden fusion")
        source = BlockKVDraft.load_checkpoint(checkpoint_path)
        if source.config.target_kv_fusion != "dflash_hidden":
            raise ValueError("source must use dflash_hidden fusion")
        comparable = (
            "hidden_size",
            "intermediate_size",
            "num_attention_heads",
            "num_key_value_heads",
            "head_dim",
            "num_draft_layers",
            "num_target_kv_layers",
            "block_size",
            "shift_labels",
            "causal_block",
            "vocab_size",
            "correction_hidden_size",
            "correction_bottleneck_size",
            "correction_mode",
            "correction_topk",
            "use_seed_hidden",
            "normalize_projected_keys",
        )
        for field in comparable:
            if getattr(source.config, field) != getattr(self.config, field):
                raise ValueError(f"repair-aware conversion changes {field}")

        source_state = source.state_dict()
        destination_state = self.state_dict()
        adapter_prefix = "target_hidden_projector."
        copied = {}
        for name, destination in destination_state.items():
            if name.startswith(adapter_prefix):
                continue
            if name not in source_state or source_state[name].shape != destination.shape:
                raise RuntimeError(f"cannot copy block parameter {name}")
            copied[name] = source_state[name]
        result = self.load_state_dict(copied, strict=False)
        expected_missing = {
            name for name in destination_state if name.startswith(adapter_prefix)
        }
        if set(result.missing_keys) != expected_missing or result.unexpected_keys:
            raise RuntimeError(
                "unexpected repair-aware conversion mismatch: "
                f"missing={result.missing_keys}, unexpected={result.unexpected_keys}"
            )
        assert isinstance(source.target_hidden_projector, TargetKVHiddenProjector)
        assert isinstance(
            self.target_hidden_projector, RepairAwareTargetKVHiddenProjector
        )
        self.target_hidden_projector.initialize_from_concat(
            source.target_hidden_projector
        )

    def project_target_memories(
        self,
        target_keys: torch.Tensor,
        target_values: torch.Tensor,
        repair_mask: torch.Tensor | None = None,
        *,
        position_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Map Target KV to every draft layer's DFlash memory space.

        The returned tensors have shape ``[B,Ldraft,Hkv,T,D]``.  Projected
        keys are normalized before RoPE, matching the released DFlash
        attention order exactly.
        """

        if self.config.target_kv_fusion == "per_layer":
            raise ValueError("learned Target-KV fusion is not configured")
        if target_keys.ndim != 5 or target_values.shape != target_keys.shape:
            raise ValueError("Target KV must have matching [B,L,Hkv,T,D] shapes")
        if target_keys.shape[1] != self.config.num_target_kv_layers:
            raise ValueError("Target KV layer count does not match draft config")
        batch, _, _, context_length, _ = target_keys.shape
        if position_ids is None:
            position_ids = torch.arange(
                context_length, device=target_keys.device
            ).unsqueeze(0).expand(batch, -1)
        elif position_ids.shape != (batch, context_length):
            raise ValueError("position_ids must have shape [B,T]")
        elif position_ids.device != target_keys.device:
            raise ValueError("position_ids and Target KV must share a device")
        memory_keys = []
        memory_values = []
        if self.config.target_kv_fusion == "projected":
            assert self.prefix_projectors is not None
            for layer, projector in zip(self.layers, self.prefix_projectors):
                memory_key, memory_value = projector(
                    target_keys,
                    target_values,
                    position_ids,
                )
                if self.config.normalize_projected_keys:
                    memory_key = apply_rope(
                        memory_key,
                        position_ids,
                        self.config.rope_theta,
                        inverse=True,
                    )
                    memory_key = layer.k_norm(memory_key)
                    memory_key = apply_rope(
                        memory_key,
                        position_ids,
                        self.config.rope_theta,
                    )
                memory_keys.append(memory_key)
                memory_values.append(memory_value)
        else:
            shared_hidden = self.project_target_hidden(
                target_keys,
                target_values,
                repair_mask=repair_mask,
                position_ids=position_ids,
            )
            for layer in self.layers:
                memory_key = layer.k_proj(shared_hidden).view(
                    batch,
                    context_length,
                    self.config.num_key_value_heads,
                    self.config.head_dim,
                )
                memory_value = layer.v_proj(shared_hidden).view_as(memory_key)
                memory_key = layer.k_norm(memory_key).transpose(1, 2).contiguous()
                memory_value = memory_value.transpose(1, 2).contiguous()
                memory_keys.append(
                    apply_rope(memory_key, position_ids, self.config.rope_theta)
                )
                memory_values.append(memory_value)
        return torch.stack(memory_keys, dim=1), torch.stack(memory_values, dim=1)

    def project_target_hidden(
        self,
        target_keys: torch.Tensor,
        target_values: torch.Tensor,
        repair_mask: torch.Tensor | None = None,
        *,
        position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return shared pseudo-context hidden features from Target KV only."""

        if self.target_hidden_projector is None:
            raise ValueError("dflash_hidden Target-KV fusion is not configured")
        if target_keys.ndim != 5 or target_values.shape != target_keys.shape:
            raise ValueError("Target KV must have matching [B,L,Hkv,T,D] shapes")
        batch, _, _, context_length, _ = target_keys.shape
        if position_ids is None:
            position_ids = torch.arange(
                context_length, device=target_keys.device
            ).unsqueeze(0).expand(batch, -1)
        elif position_ids.shape != (batch, context_length):
            raise ValueError("position_ids must have shape [B,T]")
        elif position_ids.device != target_keys.device:
            raise ValueError("position_ids and Target KV must share a device")
        if isinstance(
            self.target_hidden_projector, RepairAwareTargetKVHiddenProjector
        ):
            return self.target_hidden_projector(
                target_keys,
                target_values,
                position_ids,
                repair_mask=repair_mask,
            )
        return self.target_hidden_projector(target_keys, target_values, position_ids)

    def forward_hidden(
        self,
        current_token: torch.Tensor,
        embed_tokens: nn.Module,
        target_keys: torch.Tensor,
        target_values: torch.Tensor,
        prefix_length: int,
        repair_mask: torch.Tensor | None = None,
        seed_hidden: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if current_token.ndim != 2 or current_token.shape[1] != 1:
            raise ValueError("current_token must have shape [B,1]")
        if target_keys.ndim != 5 or target_values.shape != target_keys.shape:
            raise ValueError("Target KV must have matching [B,L,Hkv,T,D] shapes")
        if target_keys.shape[1] != self.config.num_target_kv_layers:
            raise ValueError("Target KV layer count does not match draft config")
        if prefix_length <= 0 or prefix_length > target_keys.shape[-2]:
            raise ValueError("prefix_length is outside the supplied Target KV")

        anchor_seed_hidden = None
        if seed_hidden is not None:
            expected = (current_token.shape[0], self.config.hidden_size)
            if seed_hidden.shape != expected:
                raise ValueError(
                    "single-anchor seed_hidden must have shape [B,H]"
                )
            anchor_seed_hidden = seed_hidden.unsqueeze(1)

        prefix_lengths = torch.full(
            (current_token.shape[0], 1),
            prefix_length,
            dtype=torch.long,
            device=current_token.device,
        )
        return self.forward_hidden_anchors(
            current_token,
            embed_tokens,
            target_keys,
            target_values,
            prefix_lengths,
            repair_mask=repair_mask,
            seed_hidden=anchor_seed_hidden,
        )[:, 0]

    def forward_hidden_anchors(
        self,
        current_tokens: torch.Tensor,
        embed_tokens: nn.Module,
        target_keys: torch.Tensor,
        target_values: torch.Tensor,
        prefix_lengths: torch.Tensor,
        repair_mask: torch.Tensor | None = None,
        seed_hidden: torch.Tensor | None = None,
        return_context_importance: bool = False,
        context_importance_layer_ids: Sequence[int] | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Run many independent draft anchors over one copy of Target KV.

        ``current_tokens`` and ``prefix_lengths`` are ``[B,N]``. The returned
        states are ``[B,N,block_size,H]``. This is the dense-anchor training
        layout used by DFlash-family training, adapted so the only long-context
        representation is Target KV rather than Target hidden history.
        """

        if current_tokens.ndim != 2:
            raise ValueError("current_tokens must have shape [B,num_blocks]")
        if prefix_lengths.shape != current_tokens.shape:
            raise ValueError("prefix_lengths must match current_tokens")
        if target_keys.ndim != 5 or target_values.shape != target_keys.shape:
            raise ValueError("Target KV must have matching [B,L,Hkv,T,D] shapes")
        if target_keys.shape[0] != current_tokens.shape[0]:
            raise ValueError("Target KV batch does not match current_tokens")
        if target_keys.shape[1] != self.config.num_target_kv_layers:
            raise ValueError("Target KV layer count does not match draft config")
        if repair_mask is not None and repair_mask.shape != (
            target_keys.shape[0],
            target_keys.shape[-2],
        ):
            raise ValueError("repair_mask must have shape [B,T]")

        if self.config.target_kv_fusion == "per_layer":
            memory_keys, memory_values = target_keys, target_values
        else:
            memory_keys, memory_values = self.project_target_memories(
                target_keys,
                target_values,
                repair_mask=repair_mask,
            )
        return self.forward_hidden_anchors_from_memories(
            current_tokens,
            embed_tokens,
            memory_keys,
            memory_values,
            prefix_lengths,
            seed_hidden=seed_hidden,
            return_context_importance=return_context_importance,
            context_importance_layer_ids=context_importance_layer_ids,
        )

    def forward_hidden_anchors_from_memories(
        self,
        current_tokens: torch.Tensor,
        embed_tokens: nn.Module,
        memory_keys: torch.Tensor,
        memory_values: torch.Tensor,
        prefix_lengths: torch.Tensor,
        seed_hidden: torch.Tensor | None = None,
        return_context_importance: bool = False,
        context_importance_layer_ids: Sequence[int] | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Run draft anchors from already projected, cacheable memories.

        ``memory_keys`` and ``memory_values`` have shape
        ``[B,Ldraft,Hkv,T,D]``.  Unlike :meth:`forward_hidden_anchors`, this
        path never reads or projects the long Target KV history.  It is the
        serving contract for constructing the memory once, then appending only
        the newly verified Target tokens after every speculative iteration.
        """

        if current_tokens.ndim != 2:
            raise ValueError("current_tokens must have shape [B,num_blocks]")
        if prefix_lengths.shape != current_tokens.shape:
            raise ValueError("prefix_lengths must match current_tokens")
        if memory_keys.ndim != 5 or memory_values.shape != memory_keys.shape:
            raise ValueError("draft memories must have matching [B,L,Hkv,T,D] shapes")
        if memory_keys.shape[0] != current_tokens.shape[0]:
            raise ValueError("draft memory batch does not match current_tokens")
        if memory_keys.shape[1] != self.config.num_draft_layers:
            raise ValueError("draft memory layer count does not match draft config")
        if memory_keys.shape[2] != self.config.num_key_value_heads:
            raise ValueError("draft memory KV-head count does not match draft config")
        if memory_keys.shape[-1] != self.config.head_dim:
            raise ValueError("draft memory head dimension does not match draft config")
        if bool((prefix_lengths <= 0).any()) or bool(
            (prefix_lengths > memory_keys.shape[-2]).any()
        ):
            raise ValueError("prefix_lengths lie outside the supplied draft memory")

        current = embed_tokens(current_tokens)
        batch, num_blocks, _ = current.shape
        if self.seed_hidden_proj is not None:
            if seed_hidden is None or seed_hidden.shape != current.shape:
                raise ValueError("seed_hidden must have shape [B,num_blocks,H]")
            current = current + self.seed_hidden_proj(
                self.seed_hidden_norm(seed_hidden)
            )
        elif seed_hidden is not None:
            raise ValueError("checkpoint is not configured to consume seed_hidden")
        masks = self.mask_embedding.view(1, 1, 1, -1).expand(
            batch,
            num_blocks,
            self.config.block_size - 1,
            -1,
        )
        hidden_states = torch.cat((current.unsqueeze(2), masks), dim=2).reshape(
            batch,
            num_blocks * self.config.block_size,
            self.config.hidden_size,
        )
        offsets = torch.arange(
            self.config.block_size,
            device=current.device,
        ).view(1, 1, -1)
        position_ids = (prefix_lengths.unsqueeze(-1) + offsets).reshape(
            batch, num_blocks * self.config.block_size
        )
        attention_mask = build_packed_attention_mask(
            prefix_lengths,
            context_length=memory_keys.shape[-2],
            block_size=self.config.block_size,
            causal_block=self.config.causal_block,
        )
        if context_importance_layer_ids is None:
            importance_layer_ids = set(range(len(self.layers)))
        else:
            importance_layer_ids = {int(item) for item in context_importance_layer_ids}
            if not importance_layer_ids or min(importance_layer_ids) < 0 or max(
                importance_layer_ids
            ) >= len(self.layers):
                raise ValueError("context importance layer lies outside draft")
        context_importances = []
        for index, layer in enumerate(self.layers):
            memory_key = memory_keys[:, index]
            memory_value = memory_values[:, index]
            capture_importance = (
                return_context_importance and index in importance_layer_ids
            )
            layer_output = layer(
                hidden_states,
                memory_key,
                memory_value,
                position_ids,
                prefix_lengths,
                attention_mask,
                return_context_importance=capture_importance,
            )
            if capture_importance:
                hidden_states, context_importance = layer_output
                context_importances.append(context_importance)
            else:
                hidden_states = layer_output
        hidden_states = self.final_norm(hidden_states).reshape(
            batch,
            num_blocks,
            self.config.block_size,
            self.config.hidden_size,
        )
        if return_context_importance:
            if not context_importances:
                raise RuntimeError("no draft layer produced context importance")
            importance = torch.stack(context_importances).mean(dim=0)
            importance = importance / importance.sum(
                dim=-1, keepdim=True
            ).clamp_min(1e-12)
            return hidden_states, importance
        return hidden_states

    def forward_hidden_from_memories(
        self,
        current_token: torch.Tensor,
        embed_tokens: nn.Module,
        memory_keys: torch.Tensor,
        memory_values: torch.Tensor,
        prefix_length: int,
        seed_hidden: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Single-anchor convenience wrapper for cached draft memories."""

        if current_token.ndim != 2 or current_token.shape[1] != 1:
            raise ValueError("current_token must have shape [B,1]")
        prefix_lengths = torch.full(
            (current_token.shape[0], 1),
            prefix_length,
            dtype=torch.long,
            device=current_token.device,
        )
        seed = None if seed_hidden is None else seed_hidden.unsqueeze(1)
        return self.forward_hidden_anchors_from_memories(
            current_token,
            embed_tokens,
            memory_keys,
            memory_values,
            prefix_lengths,
            seed_hidden=seed,
        )[:, 0]

    def forward_logits_from_memories(
        self,
        current_token: torch.Tensor,
        embed_tokens: nn.Module,
        lm_head: nn.Module,
        memory_keys: torch.Tensor,
        memory_values: torch.Tensor,
        prefix_length: int,
        seed_hidden: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute block logits without revisiting the Target KV history."""

        hidden = self.forward_hidden_from_memories(
            current_token,
            embed_tokens,
            memory_keys,
            memory_values,
            prefix_length,
            seed_hidden=seed_hidden,
        )
        selected = hidden[:, :-1] if self.config.shift_labels else hidden[:, 1:]
        if self.correction_gru is not None:
            raise ValueError(
                "correction training requires the autoregressive proposal path"
            )
        return lm_head(selected)

    def forward_logits(
        self,
        current_token: torch.Tensor,
        embed_tokens: nn.Module,
        lm_head: nn.Module,
        target_keys: torch.Tensor,
        target_values: torch.Tensor,
        prefix_length: int,
        repair_mask: torch.Tensor | None = None,
        seed_hidden: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden = self.forward_hidden(
            current_token,
            embed_tokens,
            target_keys,
            target_values,
            prefix_length,
            repair_mask=repair_mask,
            seed_hidden=seed_hidden,
        )
        # Position zero is the known verifier token. Unknown block positions
        # predict themselves, following the DFlash block contract.
        selected = hidden[:, :-1] if self.config.shift_labels else hidden[:, 1:]
        if self.correction_gru is not None:
            raise ValueError(
                "correction training requires forward_logits_anchors with previous_tokens"
            )
        return lm_head(selected)

    def forward_logits_anchors(
        self,
        current_tokens: torch.Tensor,
        embed_tokens: nn.Module,
        lm_head: nn.Module,
        target_keys: torch.Tensor,
        target_values: torch.Tensor,
        prefix_lengths: torch.Tensor,
        previous_tokens: torch.Tensor | None = None,
        repair_mask: torch.Tensor | None = None,
        seed_hidden: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden = self.forward_hidden_anchors(
            current_tokens,
            embed_tokens,
            target_keys,
            target_values,
            prefix_lengths,
            repair_mask=repair_mask,
            seed_hidden=seed_hidden,
        )
        selected = hidden[:, :, :-1] if self.config.shift_labels else hidden[:, :, 1:]
        base_logits = lm_head(selected)
        if self.correction_gru is None:
            return base_logits
        if previous_tokens is None or previous_tokens.shape != selected.shape[:3]:
            raise ValueError("previous_tokens must have shape [B,N,draft_length]")
        batch, num_blocks, draft_length = previous_tokens.shape
        previous_embeddings = embed_tokens(previous_tokens).reshape(
            batch * num_blocks, draft_length, self.config.hidden_size
        )
        gru_output, _ = self.correction_gru(previous_embeddings)
        gru_output = gru_output.reshape(
            batch,
            num_blocks,
            draft_length,
            self.config.correction_hidden_size,
        )
        correction_input = torch.cat(
            (selected[:, :, 1:], gru_output[:, :, 1:]), dim=-1
        )
        correction_logits = self.correction_head(correction_input)
        return torch.cat(
            (base_logits[:, :, :1], base_logits[:, :, 1:] + correction_logits),
            dim=2,
        )

    @torch.no_grad()
    def propose(
        self,
        current_token: torch.Tensor,
        embed_tokens: nn.Module,
        lm_head: nn.Module,
        target_keys: torch.Tensor,
        target_values: torch.Tensor,
        prefix_length: int,
        repair_mask: torch.Tensor | None = None,
        seed_hidden: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.correction_gru is None:
            return self.forward_logits(
                current_token,
                embed_tokens,
                lm_head,
                target_keys,
                target_values,
                prefix_length,
                repair_mask=repair_mask,
                seed_hidden=seed_hidden,
            ).argmax(dim=-1)
        hidden = self.forward_hidden(
            current_token,
            embed_tokens,
            target_keys,
            target_values,
            prefix_length,
            repair_mask=repair_mask,
            seed_hidden=seed_hidden,
        )
        hidden = hidden[:, :-1] if self.config.shift_labels else hidden[:, 1:]
        base_logits = lm_head(hidden)
        proposals = [base_logits[:, 0].argmax(dim=-1, keepdim=True)]
        _, gru_state = self.correction_gru(embed_tokens(current_token))
        for position in range(1, hidden.shape[1]):
            gru_output, gru_state = self.correction_gru(
                embed_tokens(proposals[-1]), gru_state
            )
            correction = self.correction_head(
                torch.cat((hidden[:, position : position + 1], gru_output), dim=-1)
            )
            proposals.append(
                (base_logits[:, position : position + 1] + correction)
                .argmax(dim=-1)
            )
        return torch.cat(proposals, dim=1)

    @torch.no_grad()
    def propose_from_memories(
        self,
        current_token: torch.Tensor,
        embed_tokens: nn.Module,
        lm_head: nn.Module,
        memory_keys: torch.Tensor,
        memory_values: torch.Tensor,
        prefix_length: int,
        seed_hidden: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Propose from cacheable DFlash memories without Target-KV projection."""

        if self.correction_gru is None:
            return self.forward_logits_from_memories(
                current_token,
                embed_tokens,
                lm_head,
                memory_keys,
                memory_values,
                prefix_length,
                seed_hidden=seed_hidden,
            ).argmax(dim=-1)
        hidden = self.forward_hidden_from_memories(
            current_token,
            embed_tokens,
            memory_keys,
            memory_values,
            prefix_length,
            seed_hidden=seed_hidden,
        )
        hidden = hidden[:, :-1] if self.config.shift_labels else hidden[:, 1:]
        base_logits = lm_head(hidden)
        proposals = [base_logits[:, 0].argmax(dim=-1, keepdim=True)]
        _, gru_state = self.correction_gru(embed_tokens(current_token))
        for position in range(1, hidden.shape[1]):
            gru_output, gru_state = self.correction_gru(
                embed_tokens(proposals[-1]), gru_state
            )
            correction = self.correction_head(
                torch.cat((hidden[:, position : position + 1], gru_output), dim=-1)
            )
            proposals.append(
                (base_logits[:, position : position + 1] + correction)
                .argmax(dim=-1)
            )
        return torch.cat(proposals, dim=1)

    @torch.no_grad()
    def propose_with_context_importance(
        self,
        current_token: torch.Tensor,
        embed_tokens: nn.Module,
        lm_head: nn.Module,
        target_keys: torch.Tensor,
        target_values: torch.Tensor,
        prefix_length: int,
        repair_mask: torch.Tensor | None = None,
        seed_hidden: torch.Tensor | None = None,
        importance_layer_ids: Sequence[int] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Propose one block and expose its draft-to-context attention.

        The importance vector is produced from the same draft queries and
        projected Target memories used for proposal generation.  The reference
        implementation recomputes qK probabilities because PyTorch SDPA does
        not return them; this overhead must not be described as free until a
        fused reduction or a cheaper learned utility head is implemented.
        """

        if current_token.ndim != 2 or current_token.shape[1] != 1:
            raise ValueError("current_token must have shape [B,1]")
        prefix_lengths = torch.full(
            (current_token.shape[0], 1),
            prefix_length,
            dtype=torch.long,
            device=current_token.device,
        )
        seed = None if seed_hidden is None else seed_hidden.unsqueeze(1)
        output = self.forward_hidden_anchors(
            current_token,
            embed_tokens,
            target_keys,
            target_values,
            prefix_lengths,
            repair_mask=repair_mask,
            seed_hidden=seed,
            return_context_importance=True,
            context_importance_layer_ids=importance_layer_ids,
        )
        hidden, importance = output
        hidden = hidden[:, 0]
        selected = hidden[:, :-1] if self.config.shift_labels else hidden[:, 1:]
        base_logits = lm_head(selected)
        if self.correction_gru is None:
            return base_logits.argmax(dim=-1), importance
        proposals = [base_logits[:, 0].argmax(dim=-1, keepdim=True)]
        _, gru_state = self.correction_gru(embed_tokens(current_token))
        for position in range(1, hidden.shape[1] - 1):
            gru_output, gru_state = self.correction_gru(
                embed_tokens(proposals[-1]), gru_state
            )
            correction = self.correction_head(
                torch.cat((selected[:, position : position + 1], gru_output), dim=-1)
            )
            proposals.append(
                (base_logits[:, position : position + 1] + correction)
                .argmax(dim=-1)
            )
        return torch.cat(proposals, dim=1), importance

    def save_checkpoint(self, output_dir: str | Path) -> None:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), output / "block_kv_draft.pt")
        (output / "block_kv_config.json").write_text(
            json.dumps(asdict(self.config), indent=2) + "\n"
        )

    @classmethod
    def load_checkpoint(
        cls, output_dir: str | Path, *, map_location: str | torch.device = "cpu"
    ) -> "BlockKVDraft":
        output = Path(output_dir)
        payload = json.loads((output / "block_kv_config.json").read_text())
        payload.setdefault("num_target_kv_layers", payload["num_draft_layers"])
        payload.setdefault("use_seed_hidden", False)
        payload.setdefault("normalize_projected_keys", False)
        config = BlockKVConfig(**payload)
        model = cls(config)
        model.load_state_dict(
            torch.load(
                output / "block_kv_draft.pt",
                map_location=map_location,
                weights_only=True,
            )
        )
        return model
