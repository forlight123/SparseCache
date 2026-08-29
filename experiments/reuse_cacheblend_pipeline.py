# SPDX-License-Identifier: Apache-2.0
"""True independent-document KV reuse with CacheBlend and progressive verify.

Document KV is produced once in local coordinates without a query or system
prompt, retained in pinned CPU memory, and incrementally copied to the GPU.
Every arrived cumulative view is repaired with CacheBlend-style selective
recomputation before it is used for drafting or verification.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import gc
import json
import math
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace
from typing import Any, Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

from progressive_kv_feasibility import (
    Document,
    answer_scores,
    clean_answer,
    decode_from_output,
    decode_text,
    eos_token_ids,
    full_prefill_generate,
    load_jsonl,
    normalize_answer,
    parse_fractions,
    ranked_document_indices,
    stage_document_sets,
    stratified_indices,
    synchronized_call,
    to_legacy_cache,
    tokenize_prompt,
)


@dataclass(frozen=True)
class PreparedRequest:
    dataset_index: int
    row: dict[str, Any]
    prefix: tuple[int, ...]
    documents: tuple[Document, ...]
    suffix: tuple[int, ...]
    selected_sets: tuple[set[int], ...]
    prompt: tuple[int, ...]


@dataclass(frozen=True)
class StageTransfer:
    document_indices: tuple[int, ...]
    destination_end: int


class ReusableDocumentKVStager:
    """Copy query-agnostic per-document KV from pinned CPU memory by tranche."""

    def __init__(
        self,
        document_store,
        *,
        prefix_tokens: int,
        documents: Sequence[Document],
        selected_sets: Sequence[set[int]],
        device: torch.device,
    ) -> None:
        self.device = device
        self.documents = documents
        self.document_store = document_store
        self.copy_stream = torch.cuda.Stream(device=device)
        self.starts = [torch.cuda.Event(enable_timing=True) for _ in selected_sets]
        self.ends = [torch.cuda.Event(enable_timing=True) for _ in selected_sets]
        self.launched = [False] * len(selected_sets)

        template = document_store[documents[0].token_ids]
        self.element_size = template[0][0].element_size()
        self.layer_elements_per_token = sum(
            key.shape[1] * key.shape[3] + value.shape[1] * value.shape[3]
            for key, value in template
        )
        total_tokens = prefix_tokens + sum(
            len(document.token_ids) for document in documents
        )
        self.gpu_cache = tuple(
            (
                torch.zeros(
                    (key.shape[0], key.shape[1], total_tokens, key.shape[3]),
                    dtype=key.dtype,
                    device=device,
                ),
                torch.zeros(
                    (value.shape[0], value.shape[1], total_tokens, value.shape[3]),
                    dtype=value.dtype,
                    device=device,
                ),
            )
            for key, value in template
        )

        prior: set[int] = set()
        stages = []
        for selected in selected_sets:
            count = len(selected)
            if selected != set(range(count)):
                raise ValueError("selected document stages must be ranked prefixes")
            new_indices = tuple(index for index in range(count) if index not in prior)
            destination_end = (
                prefix_tokens
                + sum(len(documents[index].token_ids) for index in range(count))
            )
            stages.append(StageTransfer(new_indices, destination_end))
            prior = set(selected)
        if stages[-1].destination_end != total_tokens:
            raise RuntimeError("final reuse stage does not contain every document")
        self.stages = tuple(stages)
        self.stage_caches = tuple(
            tuple(
                (
                    key[:, :, :stage.destination_end, :],
                    value[:, :, :stage.destination_end, :],
                )
                for key, value in self.gpu_cache
            )
            for stage in self.stages
        )

    def _stage_tokens(self, stage_index: int) -> int:
        return sum(
            len(self.documents[index].token_ids)
            for index in self.stages[stage_index].document_indices
        )

    def launch(self, stage_index: int) -> None:
        if self.launched[stage_index]:
            return
        stage = self.stages[stage_index]
        with torch.cuda.stream(self.copy_stream):
            self.starts[stage_index].record(self.copy_stream)
            for document_index in stage.document_indices:
                document = self.documents[document_index]
                source = self.document_store[document.token_ids]
                for (cpu_key, cpu_value), (gpu_key, gpu_value) in zip(
                    source, self.gpu_cache
                ):
                    gpu_key[:, :, document.start:document.end, :].copy_(
                        cpu_key, non_blocking=True
                    )
                    gpu_value[:, :, document.start:document.end, :].copy_(
                        cpu_value, non_blocking=True
                    )
            self.ends[stage_index].record(self.copy_stream)
        self.launched[stage_index] = True

    def wait_on_compute_stream(self, stage_index: int) -> None:
        if not self.launched[stage_index]:
            raise RuntimeError(f"stage {stage_index} was not launched")
        torch.cuda.current_stream(device=self.device).wait_event(
            self.ends[stage_index]
        )

    def launch_and_wait(self, stage_index: int) -> None:
        self.launch(stage_index)
        self.wait_on_compute_stream(stage_index)

    def finish(self) -> dict[str, Any]:
        self.copy_stream.synchronize()
        indices = [index for index, launched in enumerate(self.launched) if launched]
        stage_tokens = [self._stage_tokens(index) for index in indices]
        stage_bytes = [
            tokens * self.layer_elements_per_token * self.element_size
            for tokens in stage_tokens
        ]
        stage_h2d_ms = [
            self.starts[index].elapsed_time(self.ends[index]) for index in indices
        ]
        return {
            "launched_stages": len(indices),
            "stage_tokens": stage_tokens,
            "stage_bytes": stage_bytes,
            "stage_h2d_ms": stage_h2d_ms,
            "total_bytes": sum(stage_bytes),
            "total_h2d_ms": sum(stage_h2d_ms),
            "first_stage_h2d_ms": stage_h2d_ms[0],
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--sample-count", type=int, default=150)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--count", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--draft-tokens", type=int, default=4)
    parser.add_argument("--answer-word-limit", type=int, default=5)
    parser.add_argument(
        "--target-document-tokens",
        type=int,
        help="proportionally repeat documents for a timing-only scale sweep",
    )
    parser.add_argument("--stage-fractions", default="0.1,0.4,0.7,1.0")
    parser.add_argument("--recompute-ratio", type=float, default=0.15)
    parser.add_argument("--check-layer", type=int, default=1)
    parser.add_argument(
        "--deviation-metric", choices=("value", "key"), default="value"
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    args.stage_fractions = parse_fractions(args.stage_fractions)
    if min(
        args.sample_count,
        args.count,
        args.max_new_tokens,
        args.draft_tokens,
        args.answer_word_limit,
    ) <= 0:
        parser.error("sample/count/generation lengths must be positive")
    if not 0 < args.recompute_ratio <= 1:
        parser.error("recompute-ratio must lie in (0, 1]")
    if args.offset < 0 or args.offset + args.count > args.sample_count:
        parser.error("invalid sample shard")
    if args.target_document_tokens is not None and args.target_document_tokens <= 0:
        parser.error("target-document-tokens must be positive")
    return args


def expand_documents(documents, target_tokens, prefix_tokens):
    """Proportionally repeat ranked documents for timing-only scale runs."""
    if target_tokens is None:
        return documents
    original_total = sum(len(document.token_ids) for document in documents)
    if target_tokens < original_total:
        raise ValueError("target-document-tokens cannot truncate the prompt")
    exact = [
        target_tokens * len(document.token_ids) / original_total
        for document in documents
    ]
    lengths = [max(1, math.floor(value)) for value in exact]
    remainder = target_tokens - sum(lengths)
    order = sorted(
        range(len(documents)),
        key=lambda index: (-(exact[index] - math.floor(exact[index])), index),
    )
    for index in order[:remainder]:
        lengths[index] += 1
    cursor = prefix_tokens
    expanded = []
    for document, length in zip(documents, lengths):
        repeats = math.ceil(length / len(document.token_ids))
        token_ids = (document.token_ids * repeats)[:length]
        expanded.append(
            Document(
                token_ids=token_ids,
                text=document.text,
                supporting=document.supporting,
                start=cursor,
                end=cursor + length,
            )
        )
        cursor += length
    return tuple(expanded)


def prepare_request(tokenizer, row, dataset_index, args) -> PreparedRequest:
    prefix, documents, suffix = tokenize_prompt(
        tokenizer, row, args.answer_word_limit
    )
    ranking = ranked_document_indices(
        "query",
        question=row["question"],
        documents=documents,
        random_seed=args.seed + dataset_index,
    )
    cursor = len(prefix)
    ranked = []
    for original_index in ranking:
        document = documents[original_index]
        ranked.append(
            Document(
                token_ids=document.token_ids,
                text=document.text,
                supporting=document.supporting,
                start=cursor,
                end=cursor + len(document.token_ids),
            )
        )
        cursor += len(document.token_ids)
    documents = expand_documents(
        tuple(ranked), args.target_document_tokens, len(prefix)
    )
    ranked_indices = tuple(range(len(documents)))
    selected_sets = tuple(
        stage_document_sets(ranked_indices, documents, args.stage_fractions)
    )
    prompt = tuple(prefix)
    for document in documents:
        prompt += document.token_ids
    prompt += tuple(suffix)
    return PreparedRequest(
        dataset_index=dataset_index,
        row=row,
        prefix=tuple(prefix),
        documents=documents,
        suffix=tuple(suffix),
        selected_sets=selected_sets,
        prompt=prompt,
    )


@torch.inference_mode()
def encode_local_segment(model, token_ids, device):
    ids = torch.tensor([token_ids], dtype=torch.long, device=device)
    positions = torch.arange(len(token_ids), device=device).unsqueeze(0)
    output = model.model(
        input_ids=ids,
        attention_mask=torch.ones_like(ids),
        position_ids=positions,
        use_cache=True,
        return_dict=True,
    )
    cache = to_legacy_cache(output.past_key_values)
    del output
    return cache


def copy_cache_to_pinned(cache, device):
    pinned = []
    for key, value in cache:
        cpu_key = torch.empty(
            key.shape, dtype=key.dtype, device="cpu", pin_memory=True
        )
        cpu_value = torch.empty(
            value.shape, dtype=value.dtype, device="cpu", pin_memory=True
        )
        cpu_key.copy_(key, non_blocking=True)
        cpu_value.copy_(value, non_blocking=True)
        pinned.append((cpu_key, cpu_value))
    torch.cuda.synchronize(device)
    return tuple(pinned)


def precompute_document_store(model, requests, device):
    unique = {}
    for request in requests:
        for document in request.documents:
            unique.setdefault(document.token_ids, None)
    started = perf_counter()
    logical_bytes = 0
    for position, token_ids in enumerate(unique, 1):
        gpu_cache = encode_local_segment(model, token_ids, device)
        pinned = copy_cache_to_pinned(gpu_cache, device)
        unique[token_ids] = pinned
        logical_bytes += sum(
            (key.numel() + value.numel()) * key.element_size()
            for key, value in pinned
        )
        del gpu_cache
        if position % 50 == 0 or position == len(unique):
            print(
                json.dumps(
                    {
                        "event": "producer_progress",
                        "documents": position,
                        "total": len(unique),
                    }
                ),
                flush=True,
            )
    return unique, {
        "unique_documents": len(unique),
        "producer_calls": len(unique),
        "logical_bytes": logical_bytes,
        "wall_ms": (perf_counter() - started) * 1000,
    }


def rotate_half(tensor):
    first, second = tensor.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


def rope_cos_sin(model, positions, template):
    positions = positions.to(device=template.device, dtype=torch.long)
    cos, sin = model.model.rotary_emb(
        torch.empty(
            (1, len(positions), model.config.hidden_size),
            device=template.device,
            dtype=template.dtype,
        ),
        positions.unsqueeze(0),
    )
    return cos.squeeze(0), sin.squeeze(0)


def relocate_document_keys(model, raw_cache, prefix_tokens, documents):
    document_tokens = sum(len(document.token_ids) for document in documents)
    source_positions = torch.cat(
        [
            torch.arange(len(document.token_ids), device=raw_cache[0][0].device)
            for document in documents
        ]
    )
    target_positions = torch.arange(
        prefix_tokens,
        prefix_tokens + document_tokens,
        device=raw_cache[0][0].device,
    )
    if source_positions.numel() != document_tokens:
        raise RuntimeError("document relocation map has the wrong length")
    source_cos, source_sin = rope_cos_sin(
        model, source_positions, raw_cache[0][0]
    )
    target_cos, target_sin = rope_cos_sin(
        model, target_positions, raw_cache[0][0]
    )
    source_cos = source_cos.unsqueeze(0).unsqueeze(0)
    source_sin = source_sin.unsqueeze(0).unsqueeze(0)
    target_cos = target_cos.unsqueeze(0).unsqueeze(0)
    target_sin = target_sin.unsqueeze(0).unsqueeze(0)

    relocated = []
    for key, value in raw_cache:
        local_key = key[:, :, prefix_tokens:, :]
        canonical = (
            local_key.float() * source_cos
            - rotate_half(local_key.float()) * source_sin
        )
        document_key = (
            canonical * target_cos + rotate_half(canonical) * target_sin
        ).to(key.dtype)
        relocated.append(
            (
                torch.cat((key[:, :, :prefix_tokens, :], document_key), dim=2),
                value,
            )
        )
    return tuple(relocated)


def project_qkv(layer, hidden_states, cos, sin):
    shape = (*hidden_states.shape[:-1], -1, layer.self_attn.head_dim)
    query = layer.self_attn.q_proj(hidden_states).view(shape).transpose(1, 2)
    key = layer.self_attn.k_proj(hidden_states).view(shape).transpose(1, 2)
    value = layer.self_attn.v_proj(hidden_states).view(shape).transpose(1, 2)
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    query = query * cos + rotate_half(query) * sin
    key = key * cos + rotate_half(key) * sin
    return query, key, value


def attention(
    layer,
    query,
    key,
    value,
    *,
    query_positions,
    key_positions,
    full_causal,
):
    mask = None
    if not full_causal:
        mask = (
            key_positions.unsqueeze(0) <= query_positions.unsqueeze(1)
        ).unsqueeze(0).unsqueeze(0)
    return torch.nn.functional.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=mask,
        dropout_p=0.0,
        is_causal=full_causal,
        scale=layer.self_attn.scaling,
        enable_gqa=query.shape[1] != key.shape[1],
    )


def pad_cache(cache, full_length):
    if cache.shape[2] == full_length:
        return cache.clone()
    return torch.cat(
        (
            cache,
            torch.zeros(
                (
                    cache.shape[0],
                    cache.shape[1],
                    full_length - cache.shape[2],
                    cache.shape[3],
                ),
                dtype=cache.dtype,
                device=cache.device,
            ),
        ),
        dim=2,
    )


@torch.inference_mode()
def cacheblend_repair(
    model,
    raw_cache,
    *,
    prefix_ids,
    documents,
    suffix_ids,
    final_prompt_tokens,
    check_layer,
    recompute_ratio,
    deviation_metric,
    device,
):
    input_tokens = tuple(prefix_ids)
    for document in documents:
        input_tokens += document.token_ids
    input_tokens += tuple(suffix_ids)
    raw_length = raw_cache[0][0].shape[2]
    expected_raw_length = len(input_tokens) - len(suffix_ids)
    if raw_length != expected_raw_length:
        raise ValueError("raw reuse cache and stage prompt lengths disagree")

    local_positions = torch.arange(len(input_tokens), device=device)
    global_positions = local_positions.clone()
    if suffix_ids:
        global_positions[-len(suffix_ids):] = torch.arange(
            final_prompt_tokens - len(suffix_ids),
            final_prompt_tokens,
            device=device,
        )

    def run_repair():
        old_cache = relocate_document_keys(
            model, raw_cache, len(prefix_ids), documents
        )
        ids = torch.tensor([input_tokens], dtype=torch.long, device=device)
        hidden_states = model.model.embed_tokens(ids)
        cos, sin = model.model.rotary_emb(
            hidden_states, global_positions.unsqueeze(0)
        )
        layers = model.model.layers
        active_local = local_positions
        final_cache = []
        selected_document_positions = None
        mandatory_prefix = torch.arange(len(prefix_ids), device=device)
        mandatory_suffix = torch.arange(raw_length, len(input_tokens), device=device)

        for layer_index, layer in enumerate(layers):
            residual = hidden_states
            normalized = layer.input_layernorm(hidden_states)
            if layer_index <= check_layer:
                layer_local = local_positions
                layer_cos, layer_sin = cos, sin
            else:
                layer_local = active_local
                layer_cos = cos.index_select(1, active_local)
                layer_sin = sin.index_select(1, active_local)
            query, key, value = project_qkv(
                layer, normalized, layer_cos, layer_sin
            )

            if layer_index < check_layer:
                attended = attention(
                    layer,
                    query,
                    key,
                    value,
                    query_positions=global_positions,
                    key_positions=global_positions,
                    full_causal=True,
                )
                cache_key, cache_value = key, value
            elif layer_index == check_layer:
                document_positions = torch.arange(
                    len(prefix_ids), raw_length, device=device
                )
                fresh = key if deviation_metric == "key" else value
                cached = old_cache[layer_index][0 if deviation_metric == "key" else 1]
                differences = (
                    (
                        fresh[:, :, len(prefix_ids):raw_length].float()
                        - cached[:, :, len(prefix_ids):raw_length].float()
                    )
                    .square()
                    .sum(dim=(0, 1, 3))
                )
                recompute_tokens = max(
                    1, math.ceil(document_positions.numel() * recompute_ratio)
                )
                local_indices = torch.topk(
                    differences,
                    k=min(recompute_tokens, differences.numel()),
                    sorted=False,
                ).indices
                selected_document_positions = document_positions.index_select(
                    0, local_indices
                )
                active_local = torch.cat(
                    (
                        mandatory_prefix,
                        selected_document_positions,
                        mandatory_suffix,
                    )
                ).sort().values
                query = query.index_select(2, active_local)
                attended = attention(
                    layer,
                    query,
                    key,
                    value,
                    query_positions=global_positions.index_select(0, active_local),
                    key_positions=global_positions,
                    full_causal=False,
                )
                residual = residual.index_select(1, active_local)
                cache_key, cache_value = key, value
            else:
                cache_key = pad_cache(old_cache[layer_index][0], len(input_tokens))
                cache_value = pad_cache(old_cache[layer_index][1], len(input_tokens))
                cache_key[:, :, active_local] = key
                cache_value[:, :, active_local] = value
                attended = attention(
                    layer,
                    query,
                    cache_key,
                    cache_value,
                    query_positions=global_positions.index_select(0, active_local),
                    key_positions=global_positions,
                    full_causal=False,
                )

            attended = attended.transpose(1, 2).reshape(
                residual.shape[0], residual.shape[1], -1
            )
            hidden_states = residual + layer.self_attn.o_proj(attended)
            residual = hidden_states
            hidden_states = layer.post_attention_layernorm(hidden_states)
            hidden_states = residual + layer.mlp(hidden_states)
            final_cache.append((cache_key, cache_value))

        if selected_document_positions is None:
            raise RuntimeError("CacheBlend check layer was not reached")
        last_local = len(input_tokens) - 1
        match = (active_local == last_local).nonzero(as_tuple=False)
        if match.numel() != 1:
            raise RuntimeError("final online suffix token is not active")
        normalized = model.model.norm(hidden_states)
        last_hidden = normalized[:, int(match[0, 0])]
        logits = model.lm_head(last_hidden)
        selected_sorted = selected_document_positions.sort().values.to(torch.int64)
        selection_checksum = (
            selected_sorted
            * torch.arange(
                1,
                selected_sorted.numel() + 1,
                dtype=torch.int64,
                device=device,
            )
        ).sum()
        top_logits, top_tokens = logits.float().topk(2, dim=-1)
        return (
            tuple(final_cache),
            logits,
            int(selected_document_positions.numel()),
            selection_checksum,
            top_tokens,
            top_logits,
        )

    (
        cache,
        logits,
        selected_count,
        selection_checksum,
        top_tokens,
        top_logits,
    ), repair_ms = synchronized_call(
        device, run_repair, synchronize_device=False
    )
    return {
        "cache": cache,
        "logits": logits,
        "attention_mask": torch.ones(
            (1, len(input_tokens)), dtype=torch.long, device=device
        ),
        "repair_ms": repair_ms,
        "document_tokens": raw_length - len(prefix_ids),
        "selected_document_tokens": selected_count,
        "selected_document_fraction": selected_count
        / max(1, raw_length - len(prefix_ids)),
        "selection_checksum": int(selection_checksum.item()),
        "next_token_top2": top_tokens[0].tolist(),
        "next_token_margin": float((top_logits[0, 0] - top_logits[0, 1]).item()),
    }


def decode_repair_state(
    model,
    repair,
    *,
    max_new_tokens,
    position_base,
    eos_ids,
    device,
):
    output = SimpleNamespace(
        logits=repair["logits"].unsqueeze(1),
        past_key_values=DynamicCache.from_legacy_cache(repair["cache"]),
    )
    generated, cache, decode_ms = decode_from_output(
        model,
        output,
        repair["attention_mask"],
        start_position=position_base,
        max_new_tokens=max_new_tokens,
        eos_ids=eos_ids,
        device=device,
        synchronize_device=False,
    )
    del output, cache
    return generated, decode_ms


@torch.inference_mode()
def continue_repaired_cache(
    model,
    repaired_cache,
    existing_ids,
    *,
    additional_tokens,
    position_base,
    eos_ids,
    device,
):
    ids = torch.tensor([existing_ids], dtype=torch.long, device=device)
    positions = torch.arange(
        position_base, position_base + len(existing_ids), device=device
    ).unsqueeze(0)
    base_length = repaired_cache[0][0].shape[2]
    mask = torch.ones(
        (1, base_length + len(existing_ids)), dtype=torch.long, device=device
    )
    dynamic = DynamicCache.from_legacy_cache(repaired_cache)
    output, replay_ms = synchronized_call(
        device,
        lambda: model(
            input_ids=ids,
            attention_mask=mask,
            position_ids=positions,
            past_key_values=dynamic,
            use_cache=True,
            return_dict=True,
        ),
        synchronize_device=False,
    )
    generated, cache, decode_ms = decode_from_output(
        model,
        output,
        mask,
        start_position=position_base + len(existing_ids),
        max_new_tokens=additional_tokens,
        eos_ids=eos_ids,
        device=device,
        synchronize_device=False,
    )
    del output, cache, dynamic
    return generated, replay_ms, decode_ms


@torch.inference_mode()
def verify_repaired_cache(
    model,
    repair,
    sequence,
    *,
    position_base,
    device,
    return_state,
):
    ids = torch.tensor([sequence], dtype=torch.long, device=device)
    positions = torch.arange(
        position_base, position_base + len(sequence), device=device
    ).unsqueeze(0)
    base_length = repair["cache"][0][0].shape[2]
    mask = torch.ones(
        (1, base_length + len(sequence)), dtype=torch.long, device=device
    )
    dynamic = DynamicCache.from_legacy_cache(repair["cache"])
    output, verify_ms = synchronized_call(
        device,
        lambda: model(
            input_ids=ids,
            attention_mask=mask,
            position_ids=positions,
            past_key_values=dynamic,
            use_cache=True,
            return_dict=True,
        ),
        synchronize_device=False,
    )
    score_parts = [repair["logits"].unsqueeze(1)]
    if len(sequence) > 1:
        score_parts.append(output.logits[:, :len(sequence) - 1])
    predictions = torch.cat(score_parts, dim=1).argmax(dim=-1)[0].tolist()
    result = {
        "predictions": predictions,
        "verify_ms": verify_ms,
    }
    if return_state:
        result.update(output=output, dynamic=dynamic, attention_mask=mask)
    else:
        del output, dynamic
    return result


def stage_documents(documents, selected):
    count = len(selected)
    if selected != set(range(count)):
        raise ValueError("selected reuse view must be a ranked prefix")
    return tuple(documents[:count])


def run_full_cacheblend(
    model,
    document_store,
    request,
    *,
    args,
    eos_ids,
    device,
):
    selected = (set(range(len(request.documents))),)
    stager = ReusableDocumentKVStager(
        document_store,
        prefix_tokens=len(request.prefix),
        documents=request.documents,
        selected_sets=selected,
        device=device,
    )
    started = perf_counter()
    stager.launch_and_wait(0)
    repair = cacheblend_repair(
        model,
        stager.stage_caches[0],
        prefix_ids=request.prefix,
        documents=request.documents,
        suffix_ids=request.suffix,
        final_prompt_tokens=len(request.prompt),
        check_layer=args.check_layer,
        recompute_ratio=args.recompute_ratio,
        deviation_metric=args.deviation_metric,
        device=device,
    )
    first_token_ready_ms = (perf_counter() - started) * 1000
    generated, decode_ms = decode_repair_state(
        model,
        repair,
        max_new_tokens=args.max_new_tokens,
        position_base=len(request.prompt),
        eos_ids=eos_ids,
        device=device,
    )
    torch.cuda.current_stream(device=device).synchronize()
    response_ms = (perf_counter() - started) * 1000
    transfer = stager.finish()
    result = {
        "token_ids": generated,
        "generated_tokens": len(generated),
        "response_ms": response_ms,
        "first_token_ready_ms": first_token_ready_ms,
        "repair_ms": repair["repair_ms"],
        "decode_ms": decode_ms,
        "selected_document_fraction": repair["selected_document_fraction"],
        "selection_checksum": repair["selection_checksum"],
        "next_token_top2": repair["next_token_top2"],
        "next_token_margin": repair["next_token_margin"],
        "transfer": transfer,
    }
    del repair, stager
    torch.cuda.empty_cache()
    return result


def run_progressive_cacheblend(
    model,
    document_store,
    request,
    *,
    args,
    eos_ids,
    device,
    overlap,
):
    stager = ReusableDocumentKVStager(
        document_store,
        prefix_tokens=len(request.prefix),
        documents=request.documents,
        selected_sets=request.selected_sets,
        device=device,
    )
    committed = []
    pending = []
    trace = []
    total_repair_ms = 0.0
    total_verify_ms = 0.0
    total_draft_replay_ms = 0.0
    total_draft_decode_ms = 0.0
    first_draft_ms = None
    first_committed_ms = None
    s1_cache = None
    started = perf_counter()
    if overlap:
        stager.launch(0)

    for stage_index, selected in enumerate(request.selected_sets):
        if overlap:
            stager.wait_on_compute_stream(stage_index)
            if stage_index + 1 < len(request.selected_sets):
                stager.launch(stage_index + 1)
        else:
            stager.launch_and_wait(stage_index)
        documents = stage_documents(request.documents, selected)
        repair = cacheblend_repair(
            model,
            stager.stage_caches[stage_index],
            prefix_ids=request.prefix,
            documents=documents,
            suffix_ids=request.suffix,
            final_prompt_tokens=len(request.prompt),
            check_layer=args.check_layer,
            recompute_ratio=args.recompute_ratio,
            deviation_metric=args.deviation_metric,
            device=device,
        )
        total_repair_ms += repair["repair_ms"]
        stage_row = {
            "stage": stage_index,
            "document_fraction": sum(len(item.token_ids) for item in documents)
            / sum(len(item.token_ids) for item in request.documents),
            "selected_document_fraction": repair["selected_document_fraction"],
            "selection_checksum": repair["selection_checksum"],
            "next_token_top2": repair["next_token_top2"],
            "next_token_margin": repair["next_token_margin"],
            "committed_before": len(committed),
            "pending_before": len(pending),
            "accepted_pending": 0,
            "rejected_pending": 0,
            "drafted": 0,
            "repair_ms": repair["repair_ms"],
        }

        if stage_index == 0:
            s1_cache = repair["cache"]
            additions, decode_ms = decode_repair_state(
                model,
                repair,
                max_new_tokens=min(args.draft_tokens, args.max_new_tokens),
                position_base=len(request.prompt),
                eos_ids=eos_ids,
                device=device,
            )
            total_draft_decode_ms += decode_ms
            pending.extend({"token": token, "passes": 0} for token in additions)
            stage_row["drafted"] = len(additions)
            if additions:
                first_draft_ms = (perf_counter() - started) * 1000
            trace.append(
                {
                    **stage_row,
                    "committed_after": len(committed),
                    "pending_after": len(pending),
                }
            )
            continue

        sequence = committed + [item["token"] for item in pending]
        final_stage = stage_index == len(request.selected_sets) - 1
        verified = verify_repaired_cache(
            model,
            repair,
            sequence,
            position_base=len(request.prompt),
            device=device,
            return_state=final_stage,
        )
        total_verify_ms += verified["verify_ms"]
        predictions = verified["predictions"]
        pending_predictions = predictions[len(committed):]
        accepted = 0
        for item, prediction in zip(pending, pending_predictions):
            if item["token"] != prediction:
                break
            accepted += 1
        stage_row["accepted_pending"] = accepted
        stage_row["rejected_pending"] = len(pending) - accepted
        survivors = pending[:accepted]
        for item in survivors:
            item["passes"] += 1
        rejected = accepted < len(pending)
        if rejected:
            pending = survivors + [
                {"token": pending_predictions[accepted], "passes": 0}
            ]
        else:
            pending = survivors
        while pending and pending[0]["passes"] >= 1:
            committed.append(pending.pop(0)["token"])
        if committed and first_committed_ms is None:
            first_committed_ms = (perf_counter() - started) * 1000

        if committed and committed[-1] in eos_ids:
            if final_stage:
                del verified["output"], verified["dynamic"]
            trace.append(
                {
                    **stage_row,
                    "committed_after": len(committed),
                    "pending_after": len(pending),
                }
            )
            break

        if final_stage:
            committed_before_verify = len(sequence) - len(pending_predictions)
            committed.extend(item["token"] for item in pending)
            pending.clear()
            if committed and first_committed_ms is None:
                first_committed_ms = (perf_counter() - started) * 1000
            if not committed or committed[-1] not in eos_ids:
                if rejected:
                    retained = committed_before_verify + accepted
                    base_length = repair["cache"][0][0].shape[2]
                    verified["dynamic"].crop(base_length + retained)
                    correction = committed[-1]
                    correction_mask = torch.ones(
                        (1, base_length + retained + 1),
                        dtype=torch.long,
                        device=device,
                    )
                    corrected, correction_ms = synchronized_call(
                        device,
                        lambda: model(
                            input_ids=torch.tensor(
                                [[correction]], dtype=torch.long, device=device
                            ),
                            attention_mask=correction_mask,
                            position_ids=torch.tensor(
                                [[len(request.prompt) + retained]], device=device
                            ),
                            past_key_values=verified["dynamic"],
                            use_cache=True,
                            return_dict=True,
                        ),
                        synchronize_device=False,
                    )
                    del verified["output"]
                    verified["output"] = corrected
                    verified["attention_mask"] = correction_mask
                    total_verify_ms += correction_ms
                additions, cache, decode_ms = decode_from_output(
                    model,
                    verified["output"],
                    verified["attention_mask"],
                    start_position=len(request.prompt) + len(committed),
                    max_new_tokens=args.max_new_tokens - len(committed),
                    eos_ids=eos_ids,
                    device=device,
                    synchronize_device=False,
                )
                del cache
                total_draft_decode_ms += decode_ms
                committed.extend(additions)
            del verified["output"], verified["dynamic"]
            trace.append(
                {
                    **stage_row,
                    "committed_after": len(committed),
                    "pending_after": 0,
                }
            )
            break

        sequence = committed + [item["token"] for item in pending]
        if not sequence or sequence[-1] not in eos_ids:
            additions, replay_ms, decode_ms = continue_repaired_cache(
                model,
                s1_cache,
                sequence,
                additional_tokens=min(
                    args.draft_tokens, args.max_new_tokens - len(sequence)
                ),
                position_base=len(request.prompt),
                eos_ids=eos_ids,
                device=device,
            )
            total_draft_replay_ms += replay_ms
            total_draft_decode_ms += decode_ms
            pending.extend({"token": token, "passes": 0} for token in additions)
            stage_row["drafted"] = len(additions)
        trace.append(
            {
                **stage_row,
                "committed_after": len(committed),
                "pending_after": len(pending),
            }
        )
        if stage_index > 0:
            del repair

    torch.cuda.current_stream(device=device).synchronize()
    response_ms = (perf_counter() - started) * 1000
    transfer = stager.finish()
    result = {
        "token_ids": committed,
        "generated_tokens": len(committed),
        "response_ms": response_ms,
        "repair_ms": total_repair_ms,
        "verify_ms": total_verify_ms,
        "draft_replay_ms": total_draft_replay_ms,
        "draft_decode_ms": total_draft_decode_ms,
        "first_draft_batch_ms": first_draft_ms,
        "first_committed_ms": first_committed_ms,
        "trace": trace,
        "terminated_stage": trace[-1]["stage"],
        "transfer": transfer,
    }
    del stager, s1_cache
    torch.cuda.empty_cache()
    return result


def evaluate_request(model, tokenizer, document_store, request, args, device):
    eos_ids = eos_token_ids(model, tokenizer)
    # Warm the standard full-prefill path just as the reuse paths below are
    # warmed.  Shape-specific first-use costs must not be charged to only one
    # side of the latency comparison.
    full_prefill_generate(
        model,
        request.prompt,
        max_new_tokens=args.max_new_tokens,
        eos_ids=eos_ids,
        device=device,
    )
    full_ids, full_prefill_ms, full_decode_ms = full_prefill_generate(
        model,
        request.prompt,
        max_new_tokens=args.max_new_tokens,
        eos_ids=eos_ids,
        device=device,
    )
    full_text = decode_text(tokenizer, full_ids)
    golds = [request.row["answer"], *request.row.get("answer_aliases", [])]
    full_em, full_f1 = answer_scores(full_text, golds)

    # Warm all document-copy and repair shapes outside measured response paths.
    run_full_cacheblend(
        model, document_store, request, args=args, eos_ids=eos_ids, device=device
    )
    cacheblend = run_full_cacheblend(
        model, document_store, request, args=args, eos_ids=eos_ids, device=device
    )
    run_progressive_cacheblend(
        model,
        document_store,
        request,
        args=args,
        eos_ids=eos_ids,
        device=device,
        overlap=True,
    )
    progressive = run_progressive_cacheblend(
        model,
        document_store,
        request,
        args=args,
        eos_ids=eos_ids,
        device=device,
        overlap=True,
    )
    serial = run_progressive_cacheblend(
        model,
        document_store,
        request,
        args=args,
        eos_ids=eos_ids,
        device=device,
        overlap=False,
    )

    cacheblend_ids = cacheblend.pop("token_ids")
    progressive_ids = progressive.pop("token_ids")
    serial_ids = serial.pop("token_ids")
    cacheblend_text = decode_text(tokenizer, cacheblend_ids)
    progressive_text = decode_text(tokenizer, progressive_ids)
    serial_text = decode_text(tokenizer, serial_ids)
    cacheblend_em, cacheblend_f1 = answer_scores(cacheblend_text, golds)
    progressive_em, progressive_f1 = answer_scores(progressive_text, golds)
    return {
        "dataset_index": request.dataset_index,
        "id": request.row.get("id"),
        "question": request.row["question"],
        "gold_answers": golds,
        "prompt_tokens": len(request.prompt),
        "document_tokens": sum(
            len(document.token_ids) for document in request.documents
        ),
        "document_count": len(request.documents),
        "full_prefill": {
            "answer": clean_answer(full_text),
            "em": full_em,
            "f1": full_f1,
            "response_ms": full_prefill_ms + full_decode_ms,
            "prefill_ms": full_prefill_ms,
            "decode_ms": full_decode_ms,
        },
        "cacheblend_15": {
            "answer": clean_answer(cacheblend_text),
            "em": cacheblend_em,
            "f1": cacheblend_f1,
            "token_match_full_prefill": cacheblend_ids == full_ids,
            "text_match_full_prefill": cacheblend_text == full_text,
            "normalized_match_full_prefill": (
                normalize_answer(cacheblend_text) == normalize_answer(full_text)
            ),
            **cacheblend,
        },
        "sparsecache_progressive": {
            "answer": clean_answer(progressive_text),
            "em": progressive_em,
            "f1": progressive_f1,
            "token_match_full_prefill": progressive_ids == full_ids,
            "text_match_full_prefill": progressive_text == full_text,
            "normalized_match_full_prefill": (
                normalize_answer(progressive_text) == normalize_answer(full_text)
            ),
            "token_match_cacheblend": progressive_ids == cacheblend_ids,
            "answer_match_cacheblend": progressive_text == cacheblend_text,
            "serial_pipeline_token_match": progressive_ids == serial_ids,
            "serial_pipeline_answer_match": progressive_text == serial_text,
            "serial_response_ms": serial["response_ms"],
            **progressive,
        },
    }


def main():
    args = parse_args()
    output = Path(args.output)
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"{output} exists; pass --overwrite")
    output.parent.mkdir(parents=True, exist_ok=True)

    rows = load_jsonl(args.dataset)
    selected = stratified_indices(rows, args.sample_count, args.seed)
    shard_indices = selected[args.offset:args.offset + args.count]
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        local_files_only=True,
        torch_dtype=getattr(torch, args.dtype),
        attn_implementation="sdpa",
    ).eval().to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if args.check_layer >= len(model.model.layers):
        raise ValueError("check layer lies outside the model")

    requests = [
        prepare_request(tokenizer, rows[index], index, args)
        for index in shard_indices
    ]
    document_store, producer = precompute_document_store(model, requests, device)
    protocol = {
        "experiment": "independent-document-cacheblend-progressive-reuse",
        "model": str(Path(args.model).resolve()),
        "dataset": str(Path(args.dataset).resolve()),
        "sample_count": args.sample_count,
        "sample_seed": args.seed,
        "shard_offset": args.offset,
        "shard_count": args.count,
        "stage_fractions": args.stage_fractions,
        "draft_tokens": args.draft_tokens,
        "max_new_tokens": args.max_new_tokens,
        "recompute_ratio": args.recompute_ratio,
        "check_layer": args.check_layer,
        "deviation_metric": args.deviation_metric,
        "target_document_tokens": args.target_document_tokens,
        "timing_only": args.target_document_tokens is not None,
        "document_cache": "query-agnostic independent document KV in pinned CPU",
        "system_and_query_online": True,
        "document_order": "query BM25 ranking",
        "h2d": "real nonblocking per-document KV copies on dedicated stream",
        "ssd": False,
        "network_pacing": False,
        "decoding": "greedy",
    }
    with output.open("w", encoding="utf-8") as stream:
        for local_index, request in enumerate(requests, 1):
            result = evaluate_request(
                model, tokenizer, document_store, request, args, device
            )
            result["producer"] = producer
            result["protocol"] = protocol
            stream.write(json.dumps(result, ensure_ascii=False) + "\n")
            stream.flush()
            print(
                json.dumps(
                    {
                        "event": "example_complete",
                        "local": local_index,
                        "count": len(requests),
                        "dataset_index": request.dataset_index,
                    }
                ),
                flush=True,
            )
    print(json.dumps({"event": "complete", "output": str(output)}), flush=True)
    del document_store
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
