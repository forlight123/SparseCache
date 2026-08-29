# SPDX-License-Identifier: Apache-2.0
"""Semantic pre-experiment for progressive document-KV arrival.

This runner deliberately uses Transformers instead of a serving runtime.  It
keeps independently produced document KV at fixed global positions and changes
only the visibility mask as higher-priority document groups "arrive".  The
experiment therefore isolates verifier drift and staged commitment from SSD,
PCIe, and scheduler implementation details.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import gc
import json
import math
from pathlib import Path
import random
import re
import string
from time import perf_counter
from typing import Iterable, Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache


SYSTEM_PROMPT = (
    "Answer the question using the supplied documents. Return only the exact "
    "short answer without explanation."
)
TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


@dataclass(frozen=True)
class Document:
    token_ids: tuple[int, ...]
    text: str
    supporting: bool
    start: int
    end: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--sample-count", type=int, default=150)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--count", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--draft-tokens", type=int, default=4)
    parser.add_argument("--answer-word-limit", type=int, default=5)
    parser.add_argument("--stage-fractions", default="0.2,0.4,0.6,0.8,1.0")
    parser.add_argument(
        "--schedules",
        default="query,oracle,random,support_last",
        help="comma-separated subset of query,oracle,random,support_last",
    )
    parser.add_argument(
        "--commit-windows",
        default="1,2,inf",
        help="comma-separated verification counts; inf is the lossless gate",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.sample_count <= 0 or args.count <= 0 or args.offset < 0:
        parser.error("sample-count/count must be positive and offset non-negative")
    if args.offset + args.count > args.sample_count:
        parser.error("offset + count exceeds sample-count")
    if args.max_new_tokens <= 0 or args.draft_tokens <= 0:
        parser.error("generation lengths must be positive")
    args.stage_fractions = parse_fractions(args.stage_fractions)
    args.schedules = parse_schedules(args.schedules)
    args.commit_windows = parse_windows(args.commit_windows)
    return args


def parse_fractions(value: str) -> tuple[float, ...]:
    fractions = tuple(float(item) for item in value.split(",") if item.strip())
    if not fractions or fractions[-1] != 1.0:
        raise ValueError("stage fractions must end at 1.0")
    if any(not 0 < item <= 1 for item in fractions):
        raise ValueError("stage fractions must lie in (0, 1]")
    if any(left >= right for left, right in zip(fractions, fractions[1:])):
        raise ValueError("stage fractions must be strictly increasing")
    return fractions


def parse_schedules(value: str) -> tuple[str, ...]:
    allowed = {
        "query",
        "oracle",
        "p_attention",
        "random",
        "sequential",
        "support_last",
        "uniform",
    }
    schedules = tuple(item.strip() for item in value.split(",") if item.strip())
    if not schedules or any(item not in allowed for item in schedules):
        raise ValueError(f"schedules must be drawn from {sorted(allowed)}")
    return schedules


def parse_windows(value: str) -> tuple[int | None, ...]:
    parsed = []
    for item in value.split(","):
        item = item.strip().lower()
        if not item:
            continue
        if item in {"inf", "infinity"}:
            parsed.append(None)
        else:
            window = int(item)
            if window <= 0:
                raise ValueError("commit windows must be positive")
            parsed.append(window)
    if not parsed:
        raise ValueError("at least one commit window is required")
    return tuple(parsed)


def load_jsonl(path: str | Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def stratified_indices(rows: Sequence[dict], count: int, seed: int) -> list[int]:
    """Select a deterministic sample preserving the 2/3/4-hop mixture."""
    if count > len(rows):
        raise ValueError("sample-count exceeds dataset size")
    groups: dict[int, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        groups[len(row.get("question_decomposition", []))].append(index)
    exact = {hop: count * len(indices) / len(rows) for hop, indices in groups.items()}
    quotas = {hop: min(len(groups[hop]), math.floor(value)) for hop, value in exact.items()}
    remaining = count - sum(quotas.values())
    for hop in sorted(groups, key=lambda key: exact[key] - quotas[key], reverse=True):
        if remaining == 0:
            break
        if quotas[hop] < len(groups[hop]):
            quotas[hop] += 1
            remaining -= 1
    rng = random.Random(seed)
    selected = []
    for hop, indices in sorted(groups.items()):
        shuffled = list(indices)
        rng.shuffle(shuffled)
        selected.extend(shuffled[:quotas[hop]])
    rng.shuffle(selected)
    if len(selected) != count:
        raise RuntimeError("failed to construct the requested stratified sample")
    return selected


def tokenize_prompt(tokenizer, row: dict, answer_word_limit: int):
    placeholder = "{DOCUMENTS}"
    answer_instruction = (
        "Return only the exact short answer without explanation. "
        f"Answer within {answer_word_limit} words."
    )
    user = (
        f"<DOCUMENTS>\n{placeholder}\n</DOCUMENTS>\n\n"
        f"Question: {row['question'].strip()}\n{answer_instruction}"
    )
    rendered = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )
    pivot = rendered.index(placeholder)
    prefix = tuple(tokenizer.encode(rendered[:pivot], add_special_tokens=False))
    suffix = tuple(
        tokenizer.encode(
            rendered[pivot + len(placeholder):], add_special_tokens=False
        )
    )
    cursor = len(prefix)
    documents = []
    for paragraph in row["paragraphs"]:
        title = paragraph.get("title", "").strip()
        text = paragraph["paragraph_text"].strip()
        rendered_document = f"\n\nDocument:\n{title}\n{text}"
        token_ids = tuple(
            tokenizer.encode(rendered_document, add_special_tokens=False)
        )
        documents.append(
            Document(
                token_ids=token_ids,
                text=f"{title}\n{text}",
                supporting=bool(paragraph["is_supporting"]),
                start=cursor,
                end=cursor + len(token_ids),
            )
        )
        cursor += len(token_ids)
    return prefix, tuple(documents), suffix


def lexical_scores(question: str, documents: Sequence[Document]) -> list[float]:
    """Small per-request BM25 scorer with no answer/support-label leakage."""
    query_terms = TOKEN_PATTERN.findall(question.lower())
    tokenized = [TOKEN_PATTERN.findall(document.text.lower()) for document in documents]
    document_frequency = Counter()
    for words in tokenized:
        document_frequency.update(set(words))
    average_length = sum(map(len, tokenized)) / max(1, len(tokenized))
    scores = []
    for words in tokenized:
        frequencies = Counter(words)
        score = 0.0
        for term in query_terms:
            frequency = frequencies[term]
            if frequency == 0:
                continue
            df = document_frequency[term]
            inverse_document_frequency = math.log(
                1 + (len(documents) - df + 0.5) / (df + 0.5)
            )
            denominator = frequency + 1.5 * (
                1 - 0.75 + 0.75 * len(words) / max(1.0, average_length)
            )
            score += inverse_document_frequency * frequency * 2.5 / denominator
        scores.append(score)
    return scores


def ranked_document_indices(
    schedule: str,
    *,
    question: str,
    documents: Sequence[Document],
    random_seed: int,
    attention_scores: Sequence[float] | None = None,
) -> list[int]:
    scores = lexical_scores(question, documents)
    indices = list(range(len(documents)))
    if schedule == "sequential":
        return indices
    if schedule == "uniform":
        # Van der Corput order gives every prefix broad coverage of the
        # document.  It is deterministic, cumulative, and requires no labels.
        width = max(1, math.ceil(math.log2(max(1, len(indices)))))

        def reverse_bits(index: int) -> int:
            value = 0
            for _ in range(width):
                value = (value << 1) | (index & 1)
                index >>= 1
            return value

        return sorted(indices, key=lambda index: (reverse_bits(index), index))
    if schedule == "query":
        return sorted(indices, key=lambda index: (-scores[index], index))
    if schedule == "p_attention":
        if attention_scores is None or len(attention_scores) != len(documents):
            raise ValueError(
                "p_attention requires one producer-side score per document"
            )
        return sorted(
            indices,
            key=lambda index: (-attention_scores[index], -scores[index], index),
        )
    if schedule == "oracle":
        return sorted(
            indices,
            key=lambda index: (not documents[index].supporting, -scores[index], index),
        )
    if schedule == "support_last":
        return sorted(
            indices,
            key=lambda index: (documents[index].supporting, -scores[index], index),
        )
    if schedule == "random":
        random.Random(random_seed).shuffle(indices)
        return indices
    raise ValueError(f"unknown schedule: {schedule}")


def stage_document_sets(
    ranking: Sequence[int],
    documents: Sequence[Document],
    fractions: Sequence[float],
) -> list[set[int]]:
    total_tokens = sum(len(document.token_ids) for document in documents)
    stages = []
    for fraction in fractions:
        if fraction == 1.0:
            stages.append(set(ranking))
            continue
        target = math.ceil(total_tokens * fraction)
        selected = set()
        selected_tokens = 0
        for index in ranking:
            selected.add(index)
            selected_tokens += len(documents[index].token_ids)
            if selected_tokens >= target:
                break
        stages.append(selected)
    if any(not left.issubset(right) for left, right in zip(stages, stages[1:])):
        raise RuntimeError("stage document sets are not cumulative")
    return stages


def to_legacy_cache(cache) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
    if hasattr(cache, "to_legacy_cache"):
        cache = cache.to_legacy_cache()
    elif hasattr(cache, "layers"):
        return tuple((layer.keys, layer.values) for layer in cache.layers)
    return tuple((layer[0], layer[1]) for layer in cache)


def dynamic_cache_from_legacy(cache) -> DynamicCache:
    """Construct a DynamicCache across old and current Transformers APIs."""
    factory = getattr(DynamicCache, "from_legacy_cache", None)
    if factory is not None:
        return factory(cache)
    return DynamicCache(cache)


def rotate_half(tensor: torch.Tensor) -> torch.Tensor:
    first, second = tensor.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


def rope_cos_sin(model, layer_index: int, positions: torch.Tensor, template):
    rotary = getattr(model.model, "rotary_emb", None)
    if rotary is None:
        rotary = model.model.layers[layer_index].self_attn.rotary_emb
    positions = positions.to(device=template.device, dtype=torch.long)
    if model.config.model_type == "mistral":
        sequence_length = int(positions.max().item()) + 1
        cos, sin = rotary(template.float(), seq_len=sequence_length)
        return cos.index_select(0, positions), sin.index_select(0, positions)
    position_ids = positions.unsqueeze(0)
    cos, sin = rotary(template.float(), position_ids)
    return cos.squeeze(0), sin.squeeze(0)


def relocate_post_rope_key(model, key, layer_index: int, target_start: int):
    if target_start == 0:
        return key
    tokens = key.shape[2]
    source_positions = torch.arange(tokens, device=key.device)
    target_positions = torch.arange(
        target_start, target_start + tokens, device=key.device
    )
    source_cos, source_sin = rope_cos_sin(
        model, layer_index, source_positions, key
    )
    target_cos, target_sin = rope_cos_sin(
        model, layer_index, target_positions, key
    )
    source_cos = source_cos.unsqueeze(0).unsqueeze(0)
    source_sin = source_sin.unsqueeze(0).unsqueeze(0)
    target_cos = target_cos.unsqueeze(0).unsqueeze(0)
    target_sin = target_sin.unsqueeze(0).unsqueeze(0)
    canonical = key.float() * source_cos - rotate_half(key.float()) * source_sin
    relocated = canonical * target_cos + rotate_half(canonical) * target_sin
    return relocated.to(dtype=key.dtype)


@torch.inference_mode()
def encode_independent_cache(model, prefix: Sequence[int], documents: Sequence[Document], device):
    segments = [(tuple(prefix), 0)] + [
        (document.token_ids, document.start) for document in documents
    ]
    layer_parts = None
    for token_ids, start in segments:
        ids = torch.tensor([token_ids], dtype=torch.long, device=device)
        # Produce every document in its own local coordinate system.  Its keys
        # are then explicitly relocated into the immutable target slots.
        positions = torch.arange(len(token_ids), device=device).unsqueeze(0)
        mask = torch.ones_like(ids)
        output = model.model(
            input_ids=ids,
            attention_mask=mask,
            position_ids=positions,
            use_cache=True,
            return_dict=True,
        )
        cache = tuple(
            (
                relocate_post_rope_key(model, key, layer_index, start),
                value,
            )
            for layer_index, (key, value) in enumerate(
                to_legacy_cache(output.past_key_values)
            )
        )
        if layer_parts is None:
            layer_parts = [[key] for key, _ in cache], [[value] for _, value in cache]
        else:
            for layer_index, (key, value) in enumerate(cache):
                layer_parts[0][layer_index].append(key)
                layer_parts[1][layer_index].append(value)
        del output, cache
    assert layer_parts is not None
    return tuple(
        (
            torch.cat(layer_parts[0][layer_index], dim=2),
            torch.cat(layer_parts[1][layer_index], dim=2),
        )
        for layer_index in range(len(layer_parts[0]))
    )


@torch.inference_mode()
def encode_contiguous_cache(model, token_ids: Sequence[int], device):
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


def prefix_visibility_mask(
    prefix_tokens: int,
    documents: Sequence[Document],
    selected: set[int],
    device,
) -> torch.Tensor:
    total = prefix_tokens + sum(len(document.token_ids) for document in documents)
    mask = torch.zeros((1, total), dtype=torch.long, device=device)
    mask[:, :prefix_tokens] = 1
    for index in selected:
        document = documents[index]
        mask[:, document.start:document.end] = 1
    return mask


def synchronized_call(device, function, *, synchronize_device=True):
    if synchronize_device:
        def synchronizer():
            torch.cuda.synchronize(device)
    else:
        synchronizer = torch.cuda.current_stream(device=device).synchronize
    synchronizer()
    started = perf_counter()
    result = function()
    synchronizer()
    return result, (perf_counter() - started) * 1000


@torch.inference_mode()
def full_prefill_generate(
    model,
    prompt_ids: Sequence[int],
    *,
    max_new_tokens: int,
    eos_ids: set[int],
    device,
):
    ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    positions = torch.arange(len(prompt_ids), device=device).unsqueeze(0)
    mask = torch.ones_like(ids)
    output, prefill_ms = synchronized_call(
        device,
        lambda: model(
            input_ids=ids,
            attention_mask=mask,
            position_ids=positions,
            use_cache=True,
            return_dict=True,
        ),
    )
    generated, cache, decode_ms = decode_from_output(
        model,
        output,
        mask,
        start_position=len(prompt_ids),
        max_new_tokens=max_new_tokens,
        eos_ids=eos_ids,
        device=device,
    )
    del cache, output
    return generated, prefill_ms, decode_ms


@torch.inference_mode()
def continue_from_view(
    model,
    base_cache,
    prefix_mask: torch.Tensor,
    suffix_ids: Sequence[int],
    existing_ids: Sequence[int],
    *,
    additional_tokens: int,
    eos_ids: set[int],
    device,
    position_base=None,
    synchronize_device=True,
    return_margins=False,
):
    base_length = prefix_mask.shape[1]
    position_base = base_length if position_base is None else position_base
    input_tokens = tuple(suffix_ids) + tuple(existing_ids)
    ids = torch.tensor([input_tokens], dtype=torch.long, device=device)
    positions = torch.arange(
        position_base, position_base + len(input_tokens), device=device
    ).unsqueeze(0)
    attention_mask = torch.cat(
        (
            prefix_mask,
            torch.ones((1, len(input_tokens)), dtype=torch.long, device=device),
        ),
        dim=1,
    )
    dynamic = dynamic_cache_from_legacy(base_cache)
    output, prefill_ms = synchronized_call(
        device,
        lambda: model(
            input_ids=ids,
            attention_mask=attention_mask,
            position_ids=positions,
            past_key_values=dynamic,
            use_cache=True,
            return_dict=True,
        ),
        synchronize_device=synchronize_device,
    )
    decoded = decode_from_output(
        model,
        output,
        attention_mask,
        start_position=position_base + len(input_tokens),
        max_new_tokens=additional_tokens,
        eos_ids=eos_ids,
        device=device,
        synchronize_device=synchronize_device,
        return_margins=return_margins,
    )
    if return_margins:
        generated, cache, decode_ms, margins = decoded
    else:
        generated, cache, decode_ms = decoded
    del cache, output
    if return_margins:
        return generated, prefill_ms, decode_ms, margins
    return generated, prefill_ms, decode_ms


@torch.inference_mode()
def decode_from_output(
    model,
    output,
    attention_mask: torch.Tensor,
    *,
    start_position: int,
    max_new_tokens: int,
    eos_ids: set[int],
    device,
    synchronize_device=True,
    return_margins=False,
):
    generated = []
    margins = []
    cache = output.past_key_values
    logits = output.logits[:, -1]
    decode_ms = 0.0
    for step in range(max_new_tokens):
        if return_margins:
            top_two = logits.float().topk(k=2, dim=-1).values
            margins.append(float((top_two[0, 0] - top_two[0, 1]).item()))
        token = int(logits.argmax(dim=-1).item())
        generated.append(token)
        if token in eos_ids or step + 1 == max_new_tokens:
            break
        token_tensor = torch.tensor([[token]], dtype=torch.long, device=device)
        position = start_position + step
        attention_mask = torch.cat(
            (
                attention_mask,
                torch.ones((1, 1), dtype=torch.long, device=device),
            ),
            dim=1,
        )
        output, elapsed = synchronized_call(
            device,
            lambda: model(
                input_ids=token_tensor,
                attention_mask=attention_mask,
                position_ids=torch.tensor([[position]], device=device),
                past_key_values=cache,
                use_cache=True,
                return_dict=True,
            ),
            synchronize_device=synchronize_device,
        )
        decode_ms += elapsed
        cache = output.past_key_values
        logits = output.logits[:, -1]
    if return_margins:
        return generated, cache, decode_ms, margins
    return generated, cache, decode_ms


@torch.inference_mode()
def teacher_predictions(
    model,
    base_cache,
    prefix_mask: torch.Tensor,
    suffix_ids: Sequence[int],
    target_ids: Sequence[int],
    *,
    device,
    position_base=None,
    synchronize_device=True,
    return_state=False,
):
    base_length = prefix_mask.shape[1]
    position_base = base_length if position_base is None else position_base
    input_tokens = tuple(suffix_ids) + tuple(target_ids)
    ids = torch.tensor([input_tokens], dtype=torch.long, device=device)
    positions = torch.arange(
        position_base, position_base + len(input_tokens), device=device
    ).unsqueeze(0)
    attention_mask = torch.cat(
        (
            prefix_mask,
            torch.ones((1, len(input_tokens)), dtype=torch.long, device=device),
        ),
        dim=1,
    )
    dynamic = dynamic_cache_from_legacy(base_cache)
    output, elapsed_ms = synchronized_call(
        device,
        lambda: model(
            input_ids=ids,
            attention_mask=attention_mask,
            position_ids=positions,
            past_key_values=dynamic,
            use_cache=True,
            return_dict=True,
        ),
        synchronize_device=synchronize_device,
    )
    start = len(suffix_ids) - 1
    scores = output.logits[0, start:start + len(target_ids)].float()
    predictions = scores.argmax(dim=-1).tolist()
    log_probs = torch.log_softmax(scores, dim=-1)
    targets = torch.tensor(target_ids, dtype=torch.long, device=device).unsqueeze(1)
    target_log_probs = log_probs.gather(1, targets).squeeze(1).tolist()
    top_two = torch.topk(scores, k=2, dim=-1).values
    margins = (top_two[:, 0] - top_two[:, 1]).tolist()
    result = {
        "predictions": predictions,
        "target_log_probs": target_log_probs,
        "top1_margins": margins,
        "mean_target_logprob": mean(target_log_probs),
        "mean_top1_margin": mean(margins),
        "forward_ms": elapsed_ms,
    }
    if return_state:
        # The final verifier has already materialized refreshed KV for every
        # teacher-forced token.  Keeping that state lets the caller continue
        # target decoding without replaying the same suffix/history again.
        result["output"] = output
        result["attention_mask"] = attention_mask
        result["dynamic"] = dynamic
    else:
        del output, dynamic
    del scores, log_probs, top_two
    return result


@torch.inference_mode()
def progressive_chain(
    model,
    base_cache,
    masks: Sequence[torch.Tensor],
    suffix_ids: Sequence[int],
    *,
    commit_window: int | None,
    draft_tokens: int,
    max_new_tokens: int,
    eos_ids: set[int],
    device,
    stage_caches=None,
    position_base=None,
    stage_wait=None,
    stage_start=None,
    synchronize_device=True,
    draft_cache=None,
    draft_mask=None,
    reuse_final_verify=False,
    verify_intermediate=True,
    draft_intermediate=True,
):
    committed: list[int] = []
    pending: list[dict[str, int]] = []
    trace = []
    total_draft_ms = 0.0
    total_draft_prefill_ms = 0.0
    total_draft_decode_ms = 0.0
    total_final_prefill_ms = 0.0
    total_final_correction_ms = 0.0
    total_final_decode_ms = 0.0
    total_verify_ms = 0.0
    ignored_committed_flips = 0
    committed_checks = 0
    window = commit_window if commit_window is not None else len(masks) + 1
    chain_started = perf_counter()
    first_draft_batch_ms = None
    first_committed_ms = None

    for stage_index, mask in enumerate(masks):
        if stage_wait is not None:
            stage_wait(stage_index)
        if stage_start is not None and stage_index + 1 < len(masks):
            stage_start(stage_index + 1)
        stage_cache = (
            base_cache if stage_caches is None else stage_caches[stage_index]
        )
        stage_row = {
            "stage": stage_index,
            "verified": False,
            "committed_before": len(committed),
            "pending_before": len(pending),
            "accepted_pending": 0,
            "rejected_pending": 0,
            "drafted": 0,
        }
        sequence = committed + [item["token"] for item in pending]
        final_stage = stage_index == len(masks) - 1
        verify_state = None
        rejected = False
        accepted = 0
        committed_before_verify = len(committed)
        should_verify = (
            stage_index > 0
            and bool(sequence)
            and (final_stage or verify_intermediate)
        )
        if should_verify:
            stage_row["verified"] = True
            teacher = teacher_predictions(
                model,
                stage_cache,
                mask,
                suffix_ids,
                sequence,
                device=device,
                position_base=position_base,
                synchronize_device=synchronize_device,
                return_state=reuse_final_verify and final_stage,
            )
            if reuse_final_verify and final_stage:
                verify_state = {
                    "output": teacher.pop("output"),
                    "attention_mask": teacher.pop("attention_mask"),
                    "dynamic": teacher.pop("dynamic"),
                }
            verify_ms = teacher["forward_ms"]
            total_verify_ms += verify_ms
            predictions = teacher["predictions"]
            stage_row["verify_mean_top1_margin"] = teacher["mean_top1_margin"]
            stage_row["verify_min_top1_margin"] = min(
                teacher["top1_margins"], default=None
            )
            committed_checks += len(committed)
            ignored_committed_flips += sum(
                prediction != token
                for prediction, token in zip(predictions[:len(committed)], committed)
            )
            pending_predictions = predictions[len(committed):]
            accepted = 0
            for item, prediction in zip(pending, pending_predictions):
                if item["token"] != prediction:
                    break
                accepted += 1
            correction_index = len(committed) + accepted
            stage_row["correction_top1_margin"] = (
                teacher["top1_margins"][correction_index]
                if correction_index < len(teacher["top1_margins"])
                else None
            )
            stage_row["accepted_pending"] = accepted
            stage_row["rejected_pending"] = len(pending) - accepted
            survivors = pending[:accepted]
            for item in survivors:
                item["passes"] += 1
            if accepted < len(pending):
                rejected = True
                replacement = pending_predictions[accepted]
                pending = survivors + [{"token": replacement, "passes": 0}]
            else:
                pending = survivors
            while pending and pending[0]["passes"] >= window:
                committed.append(pending.pop(0)["token"])
            if committed and first_committed_ms is None:
                first_committed_ms = (perf_counter() - chain_started) * 1000

        if committed and committed[-1] in eos_ids:
            if verify_state is not None:
                del verify_state["output"], verify_state["dynamic"]
            stage_row["committed_after"] = len(committed)
            stage_row["pending_after"] = len(pending)
            trace.append(stage_row)
            return chain_payload(
                committed,
                trace,
                total_draft_ms,
                total_verify_ms,
                ignored_committed_flips,
                committed_checks,
                terminated_stage=stage_index,
                draft_prefill_ms=total_draft_prefill_ms,
                draft_decode_ms=total_draft_decode_ms,
                final_prefill_ms=total_final_prefill_ms,
                final_correction_ms=total_final_correction_ms,
                final_decode_ms=total_final_decode_ms,
                first_draft_batch_ms=first_draft_batch_ms,
                first_committed_ms=first_committed_ms,
            )

        if final_stage:
            # Every surviving/corrected pending token now agrees with the fixed
            # final verifier under the immutable committed history.
            committed.extend(item["token"] for item in pending)
            pending.clear()
            if committed and first_committed_ms is None:
                first_committed_ms = (perf_counter() - chain_started) * 1000
            if not committed or committed[-1] not in eos_ids:
                remaining = max_new_tokens - len(committed)
                if verify_state is not None:
                    if rejected:
                        # Standard speculative correction: retain the verifier's
                        # KV through the accepted prefix, crop the rejected tail,
                        # append the verifier-chosen replacement once, and then
                        # continue.  This avoids replaying suffix + committed
                        # history even when the final-stage draft is rejected.
                        retained_tokens = committed_before_verify + accepted
                        retained_cache_length = (
                            mask.shape[1] + len(suffix_ids) + retained_tokens
                        )
                        verify_state["dynamic"].crop(retained_cache_length)
                        correction = committed[-1]
                        correction_mask = torch.cat(
                            (
                                verify_state["attention_mask"][
                                    :, :retained_cache_length
                                ],
                                torch.ones(
                                    (1, 1), dtype=torch.long, device=device
                                ),
                            ),
                            dim=1,
                        )
                        correction_position = (
                            (mask.shape[1] if position_base is None else position_base)
                            + len(suffix_ids)
                            + retained_tokens
                        )
                        corrected_output, correction_ms = synchronized_call(
                            device,
                            lambda: model(
                                input_ids=torch.tensor(
                                    [[correction]], dtype=torch.long, device=device
                                ),
                                attention_mask=correction_mask,
                                position_ids=torch.tensor(
                                    [[correction_position]], device=device
                                ),
                                past_key_values=verify_state["dynamic"],
                                use_cache=True,
                                return_dict=True,
                            ),
                            synchronize_device=synchronize_device,
                        )
                        del verify_state["output"]
                        verify_state["output"] = corrected_output
                        verify_state["attention_mask"] = correction_mask
                        total_draft_ms += correction_ms
                        total_final_correction_ms += correction_ms
                    additions, final_cache, decode_ms = decode_from_output(
                        model,
                        verify_state["output"],
                        verify_state["attention_mask"],
                        start_position=(
                            (mask.shape[1] if position_base is None else position_base)
                            + len(suffix_ids)
                            + len(committed)
                        ),
                        max_new_tokens=remaining,
                        eos_ids=eos_ids,
                        device=device,
                        synchronize_device=synchronize_device,
                    )
                    prefill_ms = 0.0
                    del final_cache
                else:
                    additions, prefill_ms, decode_ms = continue_from_view(
                        model,
                        stage_cache,
                        mask,
                        suffix_ids,
                        committed,
                        additional_tokens=remaining,
                        eos_ids=eos_ids,
                        device=device,
                        position_base=position_base,
                        synchronize_device=synchronize_device,
                    )
                total_draft_ms += prefill_ms + decode_ms
                total_final_prefill_ms += prefill_ms
                total_final_decode_ms += decode_ms
                committed.extend(additions)
            if verify_state is not None:
                del verify_state["output"], verify_state["dynamic"]
            stage_row["committed_after"] = len(committed)
            stage_row["pending_after"] = 0
            trace.append(stage_row)
            return chain_payload(
                committed,
                trace,
                total_draft_ms,
                total_verify_ms,
                ignored_committed_flips,
                committed_checks,
                terminated_stage=stage_index,
                draft_prefill_ms=total_draft_prefill_ms,
                draft_decode_ms=total_draft_decode_ms,
                final_prefill_ms=total_final_prefill_ms,
                final_correction_ms=total_final_correction_ms,
                final_decode_ms=total_final_decode_ms,
                first_draft_batch_ms=first_draft_batch_ms,
                first_committed_ms=first_committed_ms,
                reused_final_verify=verify_state is not None,
            )

        sequence = committed + [item["token"] for item in pending]
        should_draft = stage_index == 0 or draft_intermediate
        if should_draft and (not sequence or sequence[-1] not in eos_ids):
            remaining = max_new_tokens - len(sequence)
            active_draft_cache = stage_cache if draft_cache is None else draft_cache
            active_draft_mask = mask if draft_mask is None else draft_mask
            additions, prefill_ms, decode_ms, draft_margins = continue_from_view(
                model,
                active_draft_cache,
                active_draft_mask,
                suffix_ids,
                sequence,
                additional_tokens=min(draft_tokens, remaining),
                eos_ids=eos_ids,
                device=device,
                position_base=position_base,
                synchronize_device=synchronize_device,
                return_margins=True,
            )
            total_draft_ms += prefill_ms + decode_ms
            total_draft_prefill_ms += prefill_ms
            total_draft_decode_ms += decode_ms
            pending.extend({"token": token, "passes": 0} for token in additions)
            stage_row["drafted"] = len(additions)
            stage_row["draft_token_ids"] = list(additions)
            stage_row["draft_top1_margins"] = draft_margins
            if additions and first_draft_batch_ms is None:
                first_draft_batch_ms = (perf_counter() - chain_started) * 1000
        stage_row["committed_after"] = len(committed)
        stage_row["pending_after"] = len(pending)
        trace.append(stage_row)
    raise RuntimeError("progressive chain failed to terminate")


@torch.inference_mode()
def grafted_draft_batch(
    model,
    prompt_cache,
    prompt_mask,
    generated_tail,
    next_input,
    *,
    draft_tokens,
    eos_ids,
    device,
    position_base,
    synchronize_device=True,
):
    """Draft without replay when the visible prompt cache grows.

    The generated-token tail was produced under earlier prompt views.  It is
    grafted after the newly enlarged exact prompt cache and remains an
    approximate draft-only representation.  The immutable final verifier
    never consumes this state.
    """

    prompt_length = prompt_mask.shape[1]
    tail_length = 0 if generated_tail is None else generated_tail[0][0].shape[2]
    graft_ms = 0.0
    if generated_tail is None:
        base_cache = prompt_cache
    else:
        base_cache, graft_ms = synchronized_call(
            device,
            lambda: tuple(
                (
                    torch.cat((prompt_key, tail_key), dim=2),
                    torch.cat((prompt_value, tail_value), dim=2),
                )
                for (prompt_key, prompt_value), (tail_key, tail_value) in zip(
                    prompt_cache, generated_tail
                )
            ),
            synchronize_device=synchronize_device,
        )

    dynamic = dynamic_cache_from_legacy(base_cache)
    attention_mask = torch.ones(
        (1, prompt_length + tail_length), dtype=torch.long, device=device
    )
    proposals = []
    margins = []
    model_ms = 0.0
    output = None
    for step in range(draft_tokens):
        attention_mask = torch.cat(
            (
                attention_mask,
                torch.ones((1, 1), dtype=torch.long, device=device),
            ),
            dim=1,
        )
        position = position_base + tail_length + step
        output, elapsed_ms = synchronized_call(
            device,
            lambda: model(
                input_ids=torch.tensor(
                    [[next_input]], dtype=torch.long, device=device
                ),
                attention_mask=attention_mask,
                position_ids=torch.tensor([[position]], device=device),
                past_key_values=dynamic,
                use_cache=True,
                return_dict=True,
            ),
            synchronize_device=synchronize_device,
        )
        model_ms += elapsed_ms
        top_two = output.logits[:, -1].float().topk(k=2, dim=-1).values
        margins.append(float((top_two[0, 0] - top_two[0, 1]).item()))
        next_input = int(output.logits[:, -1].argmax(dim=-1).item())
        proposals.append(next_input)
        if next_input in eos_ids:
            break

    if output is None:
        tail = generated_tail
    else:
        legacy = to_legacy_cache(output.past_key_values)
        tail = tuple(
            (
                key[:, :, prompt_length:, :],
                value[:, :, prompt_length:, :],
            )
            for key, value in legacy
        )
        del output
    if generated_tail is not None:
        del base_cache
    return proposals, tail, next_input, model_ms, graft_ms, margins


@torch.inference_mode()
def progressive_graft_chain(
    model,
    masks: Sequence[torch.Tensor],
    suffix_ids: Sequence[int],
    *,
    draft_tokens: int,
    max_new_tokens: int,
    eos_ids: set[int],
    device,
    stage_caches,
    position_base,
    stage_wait=None,
    stage_start=None,
    synchronize_device=True,
    reuse_final_verify=True,
):
    """Multi-stage zero-replay draft with one immutable full-KV verifier."""

    if len(suffix_ids) != 1:
        raise ValueError("progressive graft currently requires exactly one P seed token")
    proposals: list[int] = []
    generated_tail = None
    next_input = int(suffix_ids[0])
    trace = []
    draft_model_ms = 0.0
    draft_graft_ms = 0.0
    verify_ms = 0.0
    correction_ms = 0.0
    final_decode_ms = 0.0
    chain_started = perf_counter()
    first_draft_batch_ms = None

    for stage_index, (stage_cache, mask) in enumerate(zip(stage_caches, masks)):
        if stage_wait is not None:
            stage_wait(stage_index)
        if stage_start is not None and stage_index + 1 < len(masks):
            stage_start(stage_index + 1)
        final_stage = stage_index == len(masks) - 1
        stage_row = {
            "stage": stage_index,
            "verified": final_stage,
            "committed_before": 0,
            "pending_before": len(proposals),
            "accepted_pending": 0,
            "rejected_pending": 0,
            "drafted": 0,
        }
        if not final_stage:
            remaining = max_new_tokens - len(proposals)
            if remaining > 0 and (not proposals or proposals[-1] not in eos_ids):
                (
                    additions,
                    generated_tail,
                    next_input,
                    model_ms,
                    graft_ms,
                    draft_margins,
                ) = (
                    grafted_draft_batch(
                        model,
                        stage_cache,
                        mask,
                        generated_tail,
                        next_input,
                        draft_tokens=min(draft_tokens, remaining),
                        eos_ids=eos_ids,
                        device=device,
                        position_base=position_base,
                        synchronize_device=synchronize_device,
                    )
                )
                proposals.extend(additions)
                draft_model_ms += model_ms
                draft_graft_ms += graft_ms
                stage_row["drafted"] = len(additions)
                stage_row["draft_token_ids"] = list(additions)
                stage_row["draft_model_ms"] = model_ms
                stage_row["draft_graft_ms"] = graft_ms
                stage_row["draft_top1_margins"] = draft_margins
                if additions and first_draft_batch_ms is None:
                    first_draft_batch_ms = (perf_counter() - chain_started) * 1000
            stage_row["committed_after"] = 0
            stage_row["pending_after"] = len(proposals)
            trace.append(stage_row)
            continue

        teacher = teacher_predictions(
            model,
            stage_cache,
            mask,
            suffix_ids,
            proposals,
            device=device,
            position_base=position_base,
            synchronize_device=synchronize_device,
            return_state=reuse_final_verify,
        )
        verify_ms += teacher["forward_ms"]
        predictions = teacher["predictions"]
        accepted = 0
        for proposal, prediction in zip(proposals, predictions):
            if proposal != prediction:
                break
            accepted += 1
        rejected = accepted < len(proposals)
        committed = list(proposals[:accepted])
        if rejected:
            committed.append(predictions[accepted])
        stage_row.update(
            {
                "accepted_pending": accepted,
                "rejected_pending": len(proposals) - accepted,
                "verify_mean_top1_margin": teacher["mean_top1_margin"],
                "verify_min_top1_margin": min(
                    teacher["top1_margins"], default=None
                ),
                "correction_top1_margin": (
                    teacher["top1_margins"][accepted]
                    if rejected else None
                ),
            }
        )
        first_committed_ms = (perf_counter() - chain_started) * 1000
        verify_state = None
        if reuse_final_verify:
            verify_state = {
                "output": teacher.pop("output"),
                "attention_mask": teacher.pop("attention_mask"),
                "dynamic": teacher.pop("dynamic"),
            }
        if generated_tail is not None:
            del generated_tail

        if not committed or committed[-1] not in eos_ids:
            remaining = max_new_tokens - len(committed)
            if verify_state is None:
                additions, prefill_ms, decode_ms = continue_from_view(
                    model,
                    stage_cache,
                    mask,
                    suffix_ids,
                    committed,
                    additional_tokens=remaining,
                    eos_ids=eos_ids,
                    device=device,
                    position_base=position_base,
                    synchronize_device=synchronize_device,
                )
                draft_model_ms += prefill_ms
            else:
                if rejected:
                    retained_cache_length = (
                        mask.shape[1] + len(suffix_ids) + accepted
                    )
                    verify_state["dynamic"].crop(retained_cache_length)
                    correction = committed[-1]
                    correction_mask = torch.cat(
                        (
                            verify_state["attention_mask"][
                                :, :retained_cache_length
                            ],
                            torch.ones((1, 1), dtype=torch.long, device=device),
                        ),
                        dim=1,
                    )
                    correction_position = (
                        position_base + len(suffix_ids) + accepted
                    )
                    corrected_output, correction_ms = synchronized_call(
                        device,
                        lambda: model(
                            input_ids=torch.tensor(
                                [[correction]], dtype=torch.long, device=device
                            ),
                            attention_mask=correction_mask,
                            position_ids=torch.tensor(
                                [[correction_position]], device=device
                            ),
                            past_key_values=verify_state["dynamic"],
                            use_cache=True,
                            return_dict=True,
                        ),
                        synchronize_device=synchronize_device,
                    )
                    del verify_state["output"]
                    verify_state["output"] = corrected_output
                    verify_state["attention_mask"] = correction_mask
                additions, final_cache, decode_ms = decode_from_output(
                    model,
                    verify_state["output"],
                    verify_state["attention_mask"],
                    start_position=(
                        position_base + len(suffix_ids) + len(committed)
                    ),
                    max_new_tokens=remaining,
                    eos_ids=eos_ids,
                    device=device,
                    synchronize_device=synchronize_device,
                )
                del final_cache
            final_decode_ms += decode_ms
            committed.extend(additions)
        if verify_state is not None:
            del verify_state["output"], verify_state["dynamic"]
        stage_row["committed_after"] = len(committed)
        stage_row["pending_after"] = 0
        trace.append(stage_row)
        payload = chain_payload(
            committed,
            trace,
            draft_model_ms + draft_graft_ms + correction_ms + final_decode_ms,
            verify_ms,
            0,
            0,
            terminated_stage=stage_index,
            draft_prefill_ms=draft_graft_ms,
            draft_decode_ms=draft_model_ms,
            final_correction_ms=correction_ms,
            final_decode_ms=final_decode_ms,
            first_draft_batch_ms=first_draft_batch_ms,
            first_committed_ms=first_committed_ms,
            reused_final_verify=verify_state is not None,
        )
        payload["draft_graft_ms"] = draft_graft_ms
        payload["draft_model_ms"] = draft_model_ms
        return payload
    raise RuntimeError("progressive graft chain failed to terminate")


@torch.inference_mode()
def continuous_graft_chain(
    model,
    masks: Sequence[torch.Tensor],
    suffix_ids: Sequence[int],
    *,
    draft_tokens: int,
    max_new_tokens: int,
    eos_ids: set[int],
    device,
    stage_caches,
    position_base,
    stage_wait,
    stage_ready,
    stage_poll=None,
    synchronize_device=True,
    reuse_final_verify=True,
):
    """Arrival-driven sparse draft with one immutable full-KV verifier.

    The wire is already streaming every exact prompt tranche.  Before each
    proposal step, this path polls completion deadlines and expands the prompt
    view to every newly ready non-final tranche.  Generated-token K/V remains
    an approximate draft-only tail.  There is no replay, arrival barrier, or
    intermediate verifier; the full cache is consumed only by the final
    verifier.
    """

    if len(suffix_ids) != 1:
        raise ValueError("continuous graft requires exactly one P seed token")
    if len(masks) < 2:
        raise ValueError("continuous graft requires a sparse and a full stage")
    if stage_wait is None or stage_ready is None:
        raise ValueError("continuous graft requires wait and readiness callbacks")

    chain_started = perf_counter()
    final_stage = len(masks) - 1
    stage_wait(0)
    visible_stage = 0
    current_prompt_length = masks[0].shape[1]
    dynamic = dynamic_cache_from_legacy(stage_caches[0])
    attention_mask = torch.ones(
        (1, current_prompt_length), dtype=torch.long, device=device
    )
    next_input = int(suffix_ids[0])
    proposals: list[int] = []
    draft_visibility_stages: list[int] = []
    visibility_events = [
        {
            "stage": 0,
            "proposal_index": 0,
            "elapsed_ms": (perf_counter() - chain_started) * 1000,
        }
    ]
    trace = []
    stage_row = {
        "stage": 0,
        "verified": False,
        "committed_before": 0,
        "pending_before": 0,
        "accepted_pending": 0,
        "rejected_pending": 0,
        "drafted": 0,
        "draft_token_ids": [],
        "draft_top1_margins": [],
    }
    draft_model_ms = 0.0
    draft_graft_ms = 0.0
    completion_poll_ms = 0.0
    first_draft_batch_ms = None

    while len(proposals) < min(draft_tokens, max_new_tokens):
        if proposals and proposals[-1] in eos_ids:
            break

        poll_started = perf_counter()
        if stage_poll is not None:
            stage_poll()
        if stage_ready(final_stage):
            completion_poll_ms += (perf_counter() - poll_started) * 1000
            break
        newly_visible = visible_stage
        while (
            newly_visible + 1 < final_stage
            and stage_ready(newly_visible + 1)
        ):
            newly_visible += 1
        completion_poll_ms += (perf_counter() - poll_started) * 1000

        if newly_visible != visible_stage:
            stage_row["committed_after"] = 0
            stage_row["pending_after"] = len(proposals)
            trace.append(stage_row)

            legacy = to_legacy_cache(dynamic)
            generated_tail = tuple(
                (
                    key[:, :, current_prompt_length:, :],
                    value[:, :, current_prompt_length:, :],
                )
                for key, value in legacy
            )
            dynamic = None
            for stage_index in range(visible_stage + 1, newly_visible + 1):
                stage_wait(stage_index)
            visible_stage = newly_visible
            current_prompt_length = masks[visible_stage].shape[1]
            base_cache, graft_ms = synchronized_call(
                device,
                lambda: tuple(
                    (
                        torch.cat((prompt_key, tail_key), dim=2),
                        torch.cat((prompt_value, tail_value), dim=2),
                    )
                    for (prompt_key, prompt_value), (tail_key, tail_value) in zip(
                        stage_caches[visible_stage], generated_tail
                    )
                ),
                synchronize_device=synchronize_device,
            )
            draft_graft_ms += graft_ms
            dynamic = dynamic_cache_from_legacy(base_cache)
            attention_mask = torch.ones(
                (1, current_prompt_length + len(proposals)),
                dtype=torch.long,
                device=device,
            )
            del base_cache, generated_tail, legacy
            visibility_events.append(
                {
                    "stage": visible_stage,
                    "proposal_index": len(proposals),
                    "elapsed_ms": (perf_counter() - chain_started) * 1000,
                }
            )
            stage_row = {
                "stage": visible_stage,
                "verified": False,
                "committed_before": 0,
                "pending_before": len(proposals),
                "accepted_pending": 0,
                "rejected_pending": 0,
                "drafted": 0,
                "draft_token_ids": [],
                "draft_top1_margins": [],
            }

        attention_mask = torch.cat(
            (
                attention_mask,
                torch.ones((1, 1), dtype=torch.long, device=device),
            ),
            dim=1,
        )
        position = position_base + len(proposals)
        output, model_ms = synchronized_call(
            device,
            lambda: model(
                input_ids=torch.tensor(
                    [[next_input]], dtype=torch.long, device=device
                ),
                attention_mask=attention_mask,
                position_ids=torch.tensor([[position]], device=device),
                past_key_values=dynamic,
                use_cache=True,
                return_dict=True,
            ),
            synchronize_device=synchronize_device,
        )
        draft_model_ms += model_ms
        dynamic = output.past_key_values
        top_two = output.logits[:, -1].float().topk(k=2, dim=-1).values
        margin = float((top_two[0, 0] - top_two[0, 1]).item())
        next_input = int(output.logits[:, -1].argmax(dim=-1).item())
        proposals.append(next_input)
        draft_visibility_stages.append(visible_stage)
        stage_row["drafted"] += 1
        stage_row["draft_token_ids"].append(next_input)
        stage_row["draft_top1_margins"].append(margin)
        if first_draft_batch_ms is None:
            first_draft_batch_ms = (perf_counter() - chain_started) * 1000
        del output, top_two

    stage_row["committed_after"] = 0
    stage_row["pending_after"] = len(proposals)
    trace.append(stage_row)

    # Install every remaining tranche in stream order before the one immutable
    # verifier.  Some deadlines may already have passed; stage_wait then only
    # enqueues the real H2D and its compute-stream dependency.
    for stage_index in range(visible_stage + 1, len(masks)):
        stage_wait(stage_index)
    full_cache = stage_caches[-1]
    full_mask = masks[-1]
    teacher = teacher_predictions(
        model,
        full_cache,
        full_mask,
        suffix_ids,
        proposals,
        device=device,
        position_base=position_base,
        synchronize_device=synchronize_device,
        return_state=reuse_final_verify,
    )
    verify_ms = teacher["forward_ms"]
    predictions = teacher["predictions"]
    accepted = 0
    for proposal, prediction in zip(proposals, predictions):
        if proposal != prediction:
            break
        accepted += 1
    rejected = accepted < len(proposals)
    committed = list(proposals[:accepted])
    if rejected:
        committed.append(predictions[accepted])
    final_row = {
        "stage": final_stage,
        "verified": True,
        "committed_before": 0,
        "pending_before": len(proposals),
        "accepted_pending": accepted,
        "rejected_pending": len(proposals) - accepted,
        "drafted": 0,
        "verify_mean_top1_margin": teacher["mean_top1_margin"],
        "verify_min_top1_margin": min(teacher["top1_margins"], default=None),
        "correction_top1_margin": (
            teacher["top1_margins"][accepted] if rejected else None
        ),
    }
    first_committed_ms = (perf_counter() - chain_started) * 1000
    verify_state = None
    if reuse_final_verify:
        verify_state = {
            "output": teacher.pop("output"),
            "attention_mask": teacher.pop("attention_mask"),
            "dynamic": teacher.pop("dynamic"),
        }
    reused_final_verify = verify_state is not None
    correction_ms = 0.0
    final_decode_ms = 0.0
    final_prefill_ms = 0.0

    if not committed or committed[-1] not in eos_ids:
        remaining = max_new_tokens - len(committed)
        if verify_state is None:
            additions, final_prefill_ms, final_decode_ms = continue_from_view(
                model,
                full_cache,
                full_mask,
                suffix_ids,
                committed,
                additional_tokens=remaining,
                eos_ids=eos_ids,
                device=device,
                position_base=position_base,
                synchronize_device=synchronize_device,
            )
        else:
            if rejected:
                retained_cache_length = (
                    full_mask.shape[1] + len(suffix_ids) + accepted
                )
                verify_state["dynamic"].crop(retained_cache_length)
                correction = committed[-1]
                correction_mask = torch.cat(
                    (
                        verify_state["attention_mask"][:, :retained_cache_length],
                        torch.ones((1, 1), dtype=torch.long, device=device),
                    ),
                    dim=1,
                )
                correction_position = position_base + len(suffix_ids) + accepted
                corrected_output, correction_ms = synchronized_call(
                    device,
                    lambda: model(
                        input_ids=torch.tensor(
                            [[correction]], dtype=torch.long, device=device
                        ),
                        attention_mask=correction_mask,
                        position_ids=torch.tensor(
                            [[correction_position]], device=device
                        ),
                        past_key_values=verify_state["dynamic"],
                        use_cache=True,
                        return_dict=True,
                    ),
                    synchronize_device=synchronize_device,
                )
                del verify_state["output"]
                verify_state["output"] = corrected_output
                verify_state["attention_mask"] = correction_mask
            additions, final_cache, final_decode_ms = decode_from_output(
                model,
                verify_state["output"],
                verify_state["attention_mask"],
                start_position=(
                    position_base + len(suffix_ids) + len(committed)
                ),
                max_new_tokens=remaining,
                eos_ids=eos_ids,
                device=device,
                synchronize_device=synchronize_device,
            )
            del final_cache
        committed.extend(additions)
    if verify_state is not None:
        del verify_state["output"], verify_state["dynamic"]
    final_row["committed_after"] = len(committed)
    final_row["pending_after"] = 0
    trace.append(final_row)
    payload = chain_payload(
        committed,
        trace,
        (
            draft_model_ms
            + draft_graft_ms
            + correction_ms
            + final_prefill_ms
            + final_decode_ms
        ),
        verify_ms,
        0,
        0,
        terminated_stage=final_stage,
        draft_prefill_ms=draft_graft_ms,
        draft_decode_ms=draft_model_ms,
        final_prefill_ms=final_prefill_ms,
        final_correction_ms=correction_ms,
        final_decode_ms=final_decode_ms,
        first_draft_batch_ms=first_draft_batch_ms,
        first_committed_ms=first_committed_ms,
        reused_final_verify=reused_final_verify,
    )
    payload.update(
        {
            "draft_graft_ms": draft_graft_ms,
            "draft_model_ms": draft_model_ms,
            "completion_poll_ms": completion_poll_ms,
            "draft_visibility_stages": draft_visibility_stages,
            "visibility_events": visibility_events,
        }
    )
    return payload


def chain_payload(
    token_ids,
    trace,
    draft_ms,
    verify_ms,
    ignored_flips,
    committed_checks,
    *,
    terminated_stage,
    draft_prefill_ms=0.0,
    draft_decode_ms=0.0,
    final_prefill_ms=0.0,
    final_correction_ms=0.0,
    final_decode_ms=0.0,
    first_draft_batch_ms=None,
    first_committed_ms=None,
    reused_final_verify=False,
):
    return {
        "token_ids": list(token_ids),
        "trace": trace,
        "draft_and_refresh_ms": draft_ms,
        "draft_prefill_ms": draft_prefill_ms,
        "draft_decode_ms": draft_decode_ms,
        "final_prefill_ms": final_prefill_ms,
        "final_correction_ms": final_correction_ms,
        "final_decode_ms": final_decode_ms,
        "verify_ms": verify_ms,
        "ignored_committed_flips": ignored_flips,
        "committed_token_checks": committed_checks,
        "ignored_committed_flip_rate": ignored_flips / max(1, committed_checks),
        "terminated_stage": terminated_stage,
        "first_draft_batch_ms": first_draft_batch_ms,
        "first_committed_ms": first_committed_ms,
        "reused_final_verify": reused_final_verify,
    }


def eos_token_ids(model, tokenizer) -> set[int]:
    values = {tokenizer.eos_token_id}
    configured = model.generation_config.eos_token_id
    if isinstance(configured, int):
        values.add(configured)
    elif configured is not None:
        values.update(configured)
    return {value for value in values if value is not None}


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def clean_answer(text: str) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[0] if lines else ""


def normalize_answer(text: str) -> str:
    text = text.lower()
    text = "".join(char for char in text if char not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def answer_scores(text: str, golds: Sequence[str]) -> tuple[float, float]:
    prediction = normalize_answer(clean_answer(text))
    em = float(any(prediction == normalize_answer(gold) for gold in golds))
    predicted_tokens = prediction.split()
    f1_scores = []
    for gold in golds:
        expected = normalize_answer(gold).split()
        overlap = sum((Counter(predicted_tokens) & Counter(expected)).values())
        if not overlap:
            f1_scores.append(0.0)
            continue
        precision = overlap / max(1, len(predicted_tokens))
        recall = overlap / max(1, len(expected))
        f1_scores.append(2 * precision * recall / (precision + recall))
    return em, max(f1_scores, default=0.0)


def decode_text(tokenizer, token_ids: Sequence[int]) -> str:
    return tokenizer.decode(token_ids, skip_special_tokens=True).strip()


def level_metrics(
    teacher_rows: Sequence[dict],
    target_ids: Sequence[int],
    stage_sets: Sequence[set[int]],
    documents: Sequence[Document],
    fractions: Sequence[float],
):
    final_predictions = teacher_rows[-1]["predictions"]
    total_document_tokens = sum(len(document.token_ids) for document in documents)
    total_supporting_tokens = sum(
        len(document.token_ids) for document in documents if document.supporting
    )
    rows = []
    for index, (teacher, selected, configured_fraction) in enumerate(
        zip(teacher_rows, stage_sets, fractions)
    ):
        predictions = teacher["predictions"]
        final_agreement = mean(
            left == right for left, right in zip(predictions, final_predictions)
        )
        adjacent_agreement = None
        if index > 0:
            adjacent_agreement = mean(
                left == right
                for left, right in zip(
                    teacher_rows[index - 1]["predictions"], predictions
                )
            )
        strict_survival = mean(
            all(
                later["predictions"][position] == final_predictions[position]
                for later in teacher_rows[index:]
            )
            for position in range(len(target_ids))
        )
        selected_tokens = sum(len(documents[item].token_ids) for item in selected)
        selected_support = sum(
            len(documents[item].token_ids)
            for item in selected
            if documents[item].supporting
        )
        rows.append(
            {
                "stage": index,
                "configured_fraction": configured_fraction,
                "actual_document_fraction": selected_tokens / total_document_tokens,
                "supporting_token_coverage": selected_support / max(1, total_supporting_tokens),
                "adjacent_top1_agreement": adjacent_agreement,
                "final_top1_agreement": final_agreement,
                "strict_final_survival": strict_survival,
                "mean_target_logprob": teacher["mean_target_logprob"],
                "mean_top1_margin": teacher["mean_top1_margin"],
                "teacher_forward_ms": teacher["forward_ms"],
            }
        )
    return rows


def evaluate_example(model, tokenizer, row: dict, dataset_index: int, args, device):
    prefix, documents, suffix = tokenize_prompt(
        tokenizer, row, args.answer_word_limit
    )
    full_prompt = tuple(prefix)
    for document in documents:
        full_prompt += document.token_ids
    full_prompt += tuple(suffix)
    eos_ids = eos_token_ids(model, tokenizer)

    fullprefill_ids, fullprefill_ms, fullprefill_decode_ms = full_prefill_generate(
        model,
        full_prompt,
        max_new_tokens=args.max_new_tokens,
        eos_ids=eos_ids,
        device=device,
    )
    contiguous_prefix = tuple(prefix)
    for document in documents:
        contiguous_prefix += document.token_ids
    contiguous_cache, contiguous_producer_ms = synchronized_call(
        device,
        lambda: encode_contiguous_cache(model, contiguous_prefix, device),
    )
    contiguous_mask = torch.ones(
        (1, len(contiguous_prefix)), dtype=torch.long, device=device
    )
    contiguous_ids, contiguous_ms, contiguous_decode_ms = continue_from_view(
        model,
        contiguous_cache,
        contiguous_mask,
        suffix,
        (),
        additional_tokens=args.max_new_tokens,
        eos_ids=eos_ids,
        device=device,
    )
    contiguous_text = decode_text(tokenizer, contiguous_ids)
    del contiguous_cache, contiguous_mask
    base_cache, producer_ms = synchronized_call(
        device,
        lambda: encode_independent_cache(model, prefix, documents, device),
    )
    all_documents = set(range(len(documents)))
    full_mask = prefix_visibility_mask(len(prefix), documents, all_documents, device)
    fullreuse_ids, fullreuse_ms, fullreuse_decode_ms = continue_from_view(
        model,
        base_cache,
        full_mask,
        suffix,
        (),
        additional_tokens=args.max_new_tokens,
        eos_ids=eos_ids,
        device=device,
    )
    final_teacher = teacher_predictions(
        model,
        base_cache,
        full_mask,
        suffix,
        fullreuse_ids,
        device=device,
    )
    final_teacher_self_agreement = mean(
        predicted == generated
        for predicted, generated in zip(
            final_teacher["predictions"], fullreuse_ids
        )
    )

    golds = [row["answer"], *row.get("answer_aliases", [])]
    fullprefill_text = decode_text(tokenizer, fullprefill_ids)
    fullreuse_text = decode_text(tokenizer, fullreuse_ids)
    fullprefill_em, fullprefill_f1 = answer_scores(fullprefill_text, golds)
    contiguous_em, contiguous_f1 = answer_scores(contiguous_text, golds)
    fullreuse_em, fullreuse_f1 = answer_scores(fullreuse_text, golds)
    schedules = {}
    for schedule in args.schedules:
        ranking = ranked_document_indices(
            schedule,
            question=row["question"],
            documents=documents,
            random_seed=args.seed + dataset_index,
        )
        selected_sets = stage_document_sets(
            ranking, documents, args.stage_fractions
        )
        masks = [
            prefix_visibility_mask(len(prefix), documents, selected, device)
            for selected in selected_sets
        ]
        teacher_rows = []
        for stage_index, mask in enumerate(masks):
            if stage_index == len(masks) - 1:
                teacher_rows.append(final_teacher)
            else:
                teacher_rows.append(
                    teacher_predictions(
                        model,
                        base_cache,
                        mask,
                        suffix,
                        fullreuse_ids,
                        device=device,
                    )
                )
        chains = {}
        for commit_window in args.commit_windows:
            chain = progressive_chain(
                model,
                base_cache,
                masks,
                suffix,
                commit_window=commit_window,
                draft_tokens=args.draft_tokens,
                max_new_tokens=args.max_new_tokens,
                eos_ids=eos_ids,
                device=device,
            )
            chain_ids = chain.pop("token_ids")
            chain_text = decode_text(tokenizer, chain_ids)
            em, f1 = answer_scores(chain_text, golds)
            chain["answer"] = clean_answer(chain_text)
            chain["em"] = em
            chain["f1"] = f1
            chain["token_match_fullreuse"] = chain_ids == fullreuse_ids
            chain["raw_match_fullreuse"] = chain_text == fullreuse_text
            chain["normalized_match_fullreuse"] = (
                normalize_answer(chain_text) == normalize_answer(fullreuse_text)
            )
            name = "inf" if commit_window is None else str(commit_window)
            if commit_window is None:
                chain["lossless_gate_pass"] = chain["token_match_fullreuse"]
            chains[name] = chain
        schedules[schedule] = {
            "ranking": ranking,
            "levels": level_metrics(
                teacher_rows,
                fullreuse_ids,
                selected_sets,
                documents,
                args.stage_fractions,
            ),
            "chains": chains,
        }

    result = {
        "dataset_index": dataset_index,
        "id": row.get("id"),
        "hop_count": len(row.get("question_decomposition", [])),
        "question": row["question"],
        "gold_answers": golds,
        "document_count": len(documents),
        "document_tokens": sum(len(document.token_ids) for document in documents),
        "supporting_document_count": sum(document.supporting for document in documents),
        "supporting_document_tokens": sum(
            len(document.token_ids) for document in documents if document.supporting
        ),
        "logical_bf16_kv_bytes": sum(
            key.numel() * key.element_size() + value.numel() * value.element_size()
            for key, value in base_cache
        ),
        "producer_ms": producer_ms,
        "fullprefill": {
            "answer": clean_answer(fullprefill_text),
            "token_ids": fullprefill_ids,
            "em": fullprefill_em,
            "f1": fullprefill_f1,
            "prefill_ms": fullprefill_ms,
            "decode_ms": fullprefill_decode_ms,
        },
        "fullreuse": {
            "answer": clean_answer(fullreuse_text),
            "token_ids": fullreuse_ids,
            "em": fullreuse_em,
            "f1": fullreuse_f1,
            "prefill_ms": fullreuse_ms,
            "decode_ms": fullreuse_decode_ms,
            "teacher_self_top1_agreement": final_teacher_self_agreement,
            "raw_match_fullprefill": fullreuse_ids == fullprefill_ids,
            "normalized_match_fullprefill": (
                normalize_answer(fullreuse_text) == normalize_answer(fullprefill_text)
            ),
            "raw_match_fullcontext_cache": fullreuse_ids == contiguous_ids,
            "normalized_match_fullcontext_cache": (
                normalize_answer(fullreuse_text) == normalize_answer(contiguous_text)
            ),
        },
        "fullcontext_cache": {
            "answer": clean_answer(contiguous_text),
            "token_ids": contiguous_ids,
            "em": contiguous_em,
            "f1": contiguous_f1,
            "token_match_one_shot_fullprefill": contiguous_ids == fullprefill_ids,
            "normalized_match_one_shot_fullprefill": (
                normalize_answer(contiguous_text) == normalize_answer(fullprefill_text)
            ),
            "producer_ms": contiguous_producer_ms,
            "prefill_ms": contiguous_ms,
            "decode_ms": contiguous_decode_ms,
        },
        "schedules": schedules,
    }
    del base_cache, full_mask
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main() -> None:
    args = parse_args()
    output_path = Path(args.output)
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"{output_path} exists; pass --overwrite to replace it")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = load_jsonl(args.dataset)
    selected = stratified_indices(rows, args.sample_count, args.seed)
    shard_indices = selected[args.offset:args.offset + args.count]

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    dtype = getattr(torch, args.dtype)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        local_files_only=True,
        torch_dtype=dtype,
        attn_implementation="sdpa",
    ).eval().to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    with output_path.open("w", encoding="utf-8") as output:
        for local_index, dataset_index in enumerate(shard_indices):
            result = evaluate_example(
                model, tokenizer, rows[dataset_index], dataset_index, args, device
            )
            result["protocol"] = {
                "model": str(Path(args.model).resolve()),
                "dataset": str(Path(args.dataset).resolve()),
                "sample_count": args.sample_count,
                "sample_seed": args.seed,
                "shard_offset": args.offset,
                "shard_count": args.count,
                "stage_fractions": args.stage_fractions,
                "schedules": args.schedules,
                "commit_windows": [
                    "inf" if item is None else item for item in args.commit_windows
                ],
                "draft_tokens": args.draft_tokens,
                "max_new_tokens": args.max_new_tokens,
                "independent_document_kv": True,
                "static_global_positions": True,
                "representation_refresh": "all generated token identities",
                "decoding": "greedy",
            }
            output.write(json.dumps(result, ensure_ascii=False) + "\n")
            output.flush()
            print(
                json.dumps(
                    {
                        "event": "example_complete",
                        "local": local_index + 1,
                        "count": len(shard_indices),
                        "dataset_index": dataset_index,
                        "hop_count": result["hop_count"],
                        "fullprefill_f1": result["fullprefill"]["f1"],
                        "fullreuse_f1": result["fullreuse"]["f1"],
                    }
                ),
                flush=True,
            )
    print(json.dumps({"event": "complete", "output": str(output_path)}), flush=True)


if __name__ == "__main__":
    main()
