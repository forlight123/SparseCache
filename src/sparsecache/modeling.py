# SPDX-License-Identifier: Apache-2.0
"""Direct KV reuse and CacheBlend-style selective recomputation."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from time import perf_counter
from typing import Any

import torch
import torch.nn.functional as functional
from transformers.cache_utils import DynamicCache

from .rope import apply_rope
from .store import LegacyCache


@dataclass(frozen=True)
class GenerationResult:
    """Generated tokens and timing/algorithm metadata."""

    token_ids: tuple[int, ...]
    prefill_ms: float
    decode_ms: float
    metadata: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable result."""
        return asdict(self)


@torch.inference_mode()
def baseline_generate(
    model,
    *,
    full_ids: tuple[int, ...],
    max_new_tokens: int,
    eos_token_ids: set[int],
) -> GenerationResult:
    """Run ordinary full-prompt prefill followed by greedy decode."""
    device = model.model.embed_tokens.weight.device
    input_ids = torch.tensor([full_ids], dtype=torch.long, device=device)
    _synchronize(device)
    started = perf_counter()
    output = model(input_ids=input_ids, use_cache=True, return_dict=True)
    _synchronize(device)
    prefill_ms = (perf_counter() - started) * 1000
    tokens, decode_ms = _greedy_decode(
        model,
        logits=output.logits[:, -1],
        cache=output.past_key_values,
        position=len(full_ids),
        max_new_tokens=max_new_tokens,
        eos_token_ids=eos_token_ids,
    )
    return GenerationResult(tokens, prefill_ms, decode_ms, {"full_prefill": True})


@torch.inference_mode()
def direct_reuse_generate(
    model,
    *,
    prefix_cache: LegacyCache,
    suffix_ids: tuple[int, ...],
    max_new_tokens: int,
    eos_token_ids: set[int],
) -> GenerationResult:
    """Load the complete reusable prefix KV and prefill only the question suffix."""
    device = model.model.embed_tokens.weight.device
    _synchronize(device)
    transfer_started = perf_counter()
    gpu_cache = tuple(
        (key.to(device), value.to(device)) for key, value in prefix_cache
    )
    _synchronize(device)
    transfer_ms = (perf_counter() - transfer_started) * 1000
    dynamic = DynamicCache.from_legacy_cache(gpu_cache)
    prefix_tokens = gpu_cache[0][0].shape[2]
    suffix = torch.tensor([suffix_ids], dtype=torch.long, device=device)
    positions = torch.arange(
        prefix_tokens,
        prefix_tokens + len(suffix_ids),
        dtype=torch.long,
        device=device,
    )
    attention_mask = torch.ones(
        (1, prefix_tokens + len(suffix_ids)),
        dtype=torch.long,
        device=device,
    )
    _synchronize(device)
    started = perf_counter()
    output = model(
        input_ids=suffix,
        attention_mask=attention_mask,
        position_ids=positions.unsqueeze(0),
        cache_position=positions,
        past_key_values=dynamic,
        use_cache=True,
        return_dict=True,
    )
    _synchronize(device)
    prefill_ms = (perf_counter() - started) * 1000
    tokens, decode_ms = _greedy_decode(
        model,
        logits=output.logits[:, -1],
        cache=output.past_key_values,
        position=prefix_tokens + len(suffix_ids),
        max_new_tokens=max_new_tokens,
        eos_token_ids=eos_token_ids,
    )
    return GenerationResult(
        tokens,
        prefill_ms,
        decode_ms,
        {
            "full_prefill": False,
            "document_prefill_tokens": 0,
            "online_suffix_tokens": len(suffix_ids),
            "prefix_h2d_ms": transfer_ms,
        },
    )


@torch.inference_mode()
def cacheblend_generate(
    model,
    *,
    full_ids: tuple[int, ...],
    prefix_cache: LegacyCache,
    system_tokens: int,
    suffix_tokens: int,
    check_layer: int,
    recompute_ratio: float,
    max_new_tokens: int,
    eos_token_ids: set[int],
) -> GenerationResult:
    """Run CacheBlend-style layerwise selective token recomputation.

    Layers before ``check_layer`` are recomputed for the full prompt. At the
    check layer, document tokens with the largest relocated-key deviation are
    selected. Later layers update only those tokens plus every online suffix
    token; all other cache slots retain their independently produced KV.
    """
    layers = model.model.layers
    if not 0 <= check_layer < len(layers):
        raise ValueError("check_layer lies outside the model")
    if not 0 < recompute_ratio <= 1:
        raise ValueError("recompute_ratio must lie in (0, 1]")

    device = model.model.embed_tokens.weight.device
    full_length = len(full_ids)
    prefix_length = full_length - suffix_tokens
    if prefix_cache[0][0].shape[2] != prefix_length:
        raise ValueError("prefix cache and full prompt lengths disagree")
    _synchronize(device)
    transfer_started = perf_counter()
    old_cache = tuple(
        (key.to(device), value.to(device)) for key, value in prefix_cache
    )
    _synchronize(device)
    transfer_ms = (perf_counter() - transfer_started) * 1000
    input_ids = torch.tensor([full_ids], dtype=torch.long, device=device)
    positions_full = torch.arange(full_length, device=device, dtype=torch.long)
    hidden_states = model.model.embed_tokens(input_ids)
    cos_full, sin_full = model.model.rotary_emb(
        hidden_states,
        positions_full.unsqueeze(0),
    )
    active_positions = positions_full
    final_cache: list[tuple[torch.Tensor, torch.Tensor]] = []
    selected_positions: torch.Tensor | None = None
    selected_document_positions: torch.Tensor | None = None

    _synchronize(device)
    started = perf_counter()
    for layer_index, layer in enumerate(layers):
        residual = hidden_states
        normalized = layer.input_layernorm(hidden_states)
        if layer_index <= check_layer:
            layer_positions = positions_full
            layer_cos, layer_sin = cos_full, sin_full
        else:
            layer_positions = active_positions
            layer_cos = cos_full.index_select(1, active_positions)
            layer_sin = sin_full.index_select(1, active_positions)

        query, key, value = _project_qkv(
            layer,
            normalized,
            layer_cos,
            layer_sin,
        )

        if layer_index < check_layer:
            attention = _attention(
                layer,
                query,
                key,
                value,
                query_positions=positions_full,
                key_positions=positions_full,
                full_causal=True,
            )
            cache_key, cache_value = key, value
        elif layer_index == check_layer:
            document_end = prefix_length
            document_positions = torch.arange(
                system_tokens,
                document_end,
                device=device,
                dtype=torch.long,
            )
            cached_key = old_cache[layer_index][0]
            differences = (
                (key[:, :, system_tokens:document_end].float()
                 - cached_key[:, :, system_tokens:document_end].float())
                .square()
                .sum(dim=(0, 1, 3))
            )
            recompute_tokens = max(
                1,
                math.ceil(document_positions.numel() * recompute_ratio),
            )
            local_indices = torch.topk(
                differences,
                k=min(recompute_tokens, differences.numel()),
                sorted=False,
            ).indices
            selected_document_positions = document_positions.index_select(
                0, local_indices
            )
            suffix_positions = torch.arange(
                prefix_length,
                full_length,
                device=device,
                dtype=torch.long,
            )
            selected_positions = torch.cat(
                (selected_document_positions, suffix_positions)
            ).sort().values
            query = query.index_select(2, selected_positions)
            attention = _attention(
                layer,
                query,
                key,
                value,
                query_positions=selected_positions,
                key_positions=positions_full,
                full_causal=False,
            )
            residual = residual.index_select(1, selected_positions)
            active_positions = selected_positions
            cache_key, cache_value = key, value
        else:
            cache_key = _pad_prefix_cache(
                old_cache[layer_index][0], full_length
            )
            cache_value = _pad_prefix_cache(
                old_cache[layer_index][1], full_length
            )
            cache_key[:, :, active_positions] = key
            cache_value[:, :, active_positions] = value
            attention = _attention(
                layer,
                query,
                cache_key,
                cache_value,
                query_positions=active_positions,
                key_positions=positions_full,
                full_causal=False,
            )

        attention = attention.transpose(1, 2).reshape(
            residual.shape[0], residual.shape[1], -1
        )
        hidden_states = residual + layer.self_attn.o_proj(attention)
        residual = hidden_states
        hidden_states = layer.post_attention_layernorm(hidden_states)
        hidden_states = residual + layer.mlp(hidden_states)
        final_cache.append((cache_key, cache_value))

    assert selected_positions is not None
    assert selected_document_positions is not None
    last_matches = (selected_positions == full_length - 1).nonzero(as_tuple=False)
    if last_matches.numel() != 1:
        raise RuntimeError("the final suffix token was not selected")
    normalized = model.model.norm(hidden_states)
    last_hidden = normalized[:, int(last_matches[0, 0])]
    logits = model.lm_head(last_hidden)
    _synchronize(device)
    prefill_ms = (perf_counter() - started) * 1000

    dynamic = DynamicCache.from_legacy_cache(tuple(final_cache))
    tokens, decode_ms = _greedy_decode(
        model,
        logits=logits,
        cache=dynamic,
        position=full_length,
        max_new_tokens=max_new_tokens,
        eos_token_ids=eos_token_ids,
    )
    selected_count = int(selected_document_positions.numel())
    post_check_layers = len(layers) - check_layer - 1
    layer_token_work = (
        check_layer * full_length
        + selected_positions.numel()
        + post_check_layers * selected_positions.numel()
    )
    return GenerationResult(
        tokens,
        prefill_ms,
        decode_ms,
        {
            "check_layer": check_layer,
            "recompute_ratio": recompute_ratio,
            "document_tokens_selected": selected_count,
            "document_tokens": prefix_length - system_tokens,
            "mandatory_suffix_tokens": suffix_tokens,
            "prefix_h2d_ms": transfer_ms,
            "selected_document_fraction": selected_count
            / max(1, prefix_length - system_tokens),
            "approximate_layer_token_fraction": layer_token_work
            / (len(layers) * full_length),
        },
    )


def _project_qkv(
    layer,
    hidden_states: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    shape = (*hidden_states.shape[:-1], -1, layer.self_attn.head_dim)
    query = layer.self_attn.q_proj(hidden_states).view(shape).transpose(1, 2)
    key = layer.self_attn.k_proj(hidden_states).view(shape).transpose(1, 2)
    value = layer.self_attn.v_proj(hidden_states).view(shape).transpose(1, 2)
    query = apply_rope(query, cos, sin, sequence_dim=2)
    key = apply_rope(key, cos, sin, sequence_dim=2)
    return query, key, value


def _attention(
    layer,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    query_positions: torch.Tensor,
    key_positions: torch.Tensor,
    full_causal: bool,
) -> torch.Tensor:
    mask = None
    if not full_causal:
        mask = (
            key_positions.unsqueeze(0) <= query_positions.unsqueeze(1)
        ).unsqueeze(0).unsqueeze(0)
    return functional.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=mask,
        dropout_p=0.0,
        is_causal=full_causal,
        scale=layer.self_attn.scaling,
        enable_gqa=query.shape[1] != key.shape[1],
    )


def _pad_prefix_cache(cache: torch.Tensor, full_length: int) -> torch.Tensor:
    if cache.shape[2] == full_length:
        return cache.clone()
    padding = torch.zeros(
        (cache.shape[0], cache.shape[1], full_length - cache.shape[2], cache.shape[3]),
        dtype=cache.dtype,
        device=cache.device,
    )
    return torch.cat((cache, padding), dim=2)


@torch.inference_mode()
def _greedy_decode(
    model,
    *,
    logits: torch.Tensor,
    cache,
    position: int,
    max_new_tokens: int,
    eos_token_ids: set[int],
) -> tuple[tuple[int, ...], float]:
    device = logits.device
    _synchronize(device)
    started = perf_counter()
    generated = []
    for step in range(max_new_tokens):
        token = int(logits.argmax(dim=-1).item())
        generated.append(token)
        if token in eos_token_ids:
            break
        if step + 1 == max_new_tokens:
            break
        cache_position = torch.tensor([position], dtype=torch.long, device=device)
        output = model(
            input_ids=torch.tensor([[token]], dtype=torch.long, device=device),
            position_ids=cache_position.unsqueeze(0),
            cache_position=cache_position,
            past_key_values=cache,
            use_cache=True,
            return_dict=True,
        )
        cache = output.past_key_values
        logits = output.logits[:, -1]
        position += 1
    _synchronize(device)
    return tuple(generated), (perf_counter() - started) * 1000


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
