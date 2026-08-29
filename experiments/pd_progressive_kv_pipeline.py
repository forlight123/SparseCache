# SPDX-License-Identifier: Apache-2.0
"""Real pinned-CPU to GPU progressive KV draft/verify pipeline experiment."""

from __future__ import annotations

import argparse
import gc
import json
import math
import random
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter, sleep
from typing import Any

import torch
from longbench_metrics import score_prediction
from longbench_prompts import (
    LONG_BENCH_MAX_NEW_TOKENS,
    LONG_BENCH_PROMPTS,
    LONG_BENCH_V2_MAX_NEW_TOKENS,
    LONG_BENCH_V2_PROMPT,
)
from progressive_kv_feasibility import (
    Document,
    answer_scores,
    clean_answer,
    continue_from_view,
    continuous_graft_chain,
    decode_text,
    encode_contiguous_cache,
    eos_token_ids,
    full_prefill_generate,
    load_jsonl,
    normalize_answer,
    parse_fractions,
    parse_schedules,
    parse_windows,
    progressive_chain,
    progressive_graft_chain,
    ranked_document_indices,
    stage_document_sets,
    stratified_indices,
    synchronized_call,
    to_legacy_cache,
    tokenize_prompt,
)
from transformers import AutoModelForCausalLM, AutoTokenizer


def rotate_half(tensor):
    first, second = tensor.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


@torch.inference_mode()
def encode_cache_with_producer_attention_scores(
    model,
    token_ids,
    documents,
    query_positions,
    device,
):
    """Prefill exact KV and score pages using P-side last-query attention.

    Four evenly spaced transformer layers are sampled.  Only the normalized
    selected question-token states are retained by each hook, so the scorer
    does not keep all hidden states alive.  Scoring runs on P and is outside D
    response latency.
    """

    layers = model.model.layers
    layer_indices = sorted(
        {
            max(0, round((len(layers) - 1) * fraction))
            for fraction in (0.25, 0.5, 0.75, 1.0)
        }
    )
    normalized_last = {}
    handles = []

    def capture(layer_index):
        def hook(_module, _inputs, output):
            normalized_last[layer_index] = output.index_select(
                1, query_positions
            ).detach()

        return hook

    for layer_index in layer_indices:
        handles.append(
            layers[layer_index].input_layernorm.register_forward_hook(
                capture(layer_index)
            )
        )
    ids = torch.tensor([token_ids], dtype=torch.long, device=device)
    positions = torch.arange(len(token_ids), device=device).unsqueeze(0)
    try:
        output, prefill_ms = synchronized_call(
            device,
            lambda: model.model(
                input_ids=ids,
                attention_mask=torch.ones_like(ids),
                position_ids=positions,
                use_cache=True,
                return_dict=True,
            ),
        )
    finally:
        for handle in handles:
            handle.remove()
    cache = to_legacy_cache(output.past_key_values)

    score_started = perf_counter()
    per_layer = []
    scored_positions = positions.index_select(1, query_positions)
    for layer_index in layer_indices:
        attention = layers[layer_index].self_attn
        hidden = normalized_last[layer_index]
        query = (
            attention.q_proj(hidden).view(1, 1, -1, attention.head_dim).transpose(1, 2)
        )
        if hasattr(attention, "q_norm"):
            query = attention.q_norm(query)
        cos, sin = model.model.rotary_emb(hidden, scored_positions)
        query = query * cos.unsqueeze(1) + rotate_half(query) * sin.unsqueeze(1)
        key = cache[layer_index][0]
        groups = query.shape[1] // key.shape[1]
        if groups > 1:
            key = key.repeat_interleave(groups, dim=1)
        logits = (
            torch.einsum("bhqd,bhkd->bhqk", query.float(), key.float())
            * attention.scaling
        )
        per_layer.append(logits.softmax(dim=-1).mean(dim=(1, 2))[0])
    token_scores = torch.stack(per_layer).mean(dim=0)
    document_scores = [
        float(token_scores[document.start : document.end].sum().item())
        for document in documents
    ]
    torch.cuda.current_stream(device=device).synchronize()
    score_ms = (perf_counter() - score_started) * 1000
    del output, per_layer, token_scores
    return (
        cache,
        document_scores,
        {
            "layers": layer_indices,
            "query_tokens": int(query_positions.numel()),
            "prefill_ms": prefill_ms,
            "score_ms": score_ms,
        },
    )


def question_token_positions(tokenizer, prompt, question, device, limit=16):
    """Locate actual question tokens in the rendered prompt."""

    candidates = []
    for text in (question.strip(), " " + question.strip()):
        encoded = tokenizer.encode(text, add_special_tokens=False)
        if encoded and encoded not in candidates:
            candidates.append(encoded)
    start = None
    length = None
    for candidate in candidates:
        for index in range(len(prompt) - len(candidate), -1, -1):
            if list(prompt[index : index + len(candidate)]) == candidate:
                start = index
                length = len(candidate)
                break
        if start is not None:
            break
    if start is None:
        length = min(limit, len(prompt))
        start = len(prompt) - length
    positions = list(range(start, start + length))
    if len(positions) > limit:
        positions = [
            positions[round(index * (len(positions) - 1) / (limit - 1))]
            for index in range(limit)
        ]
    return torch.tensor(positions, dtype=torch.long, device=device)


@dataclass(frozen=True)
class StageTransfer:
    ranges: tuple[tuple[int, int], ...]
    destination_start: int
    destination_end: int


class PinnedKVStager:
    """Append disjoint KV tranches into a compact GPU cache on a copy stream."""

    def __init__(
        self,
        cpu_cache,
        *,
        prefix_tokens: int,
        documents,
        suffix_tokens: int,
        ranking: Sequence[int],
        selected_sets: Sequence[set[int]],
        device: torch.device,
        transport_gbps: float | None = None,
        eager_serial_wire: bool = False,
    ) -> None:
        self.cpu_cache = cpu_cache
        self.device = device
        self.copy_stream = torch.cuda.Stream(device=device)
        self.transport_gbps = transport_gbps
        self.eager_serial_wire = eager_serial_wire
        self.starts = [torch.cuda.Event(enable_timing=True) for _ in selected_sets]
        self.ends = [torch.cuda.Event(enable_timing=True) for _ in selected_sets]
        self.launched = [False] * len(selected_sets)
        self.physical_launched = [False] * len(selected_sets)
        self.wire_started_at: list[float | None] = [None] * len(selected_sets)
        self.wire_ready_at: list[float | None] = [None] * len(selected_sets)
        self.wire_wait_ms = [0.0] * len(selected_sets)
        self.element_size = cpu_cache[0][0].element_size()
        self.layer_elements_per_token = sum(
            key.shape[1] * key.shape[3] + value.shape[1] * value.shape[3]
            for key, value in cpu_cache
        )
        total_tokens = (
            prefix_tokens
            + sum(len(document.token_ids) for document in documents)
            + suffix_tokens
        )
        self.gpu_cache = tuple(
            (
                torch.empty(key.shape, dtype=key.dtype, device=device),
                torch.empty(value.shape, dtype=value.dtype, device=device),
            )
            for key, value in cpu_cache
        )

        document_end = prefix_tokens + sum(
            len(document.token_ids) for document in documents
        )
        stages = []
        prior: set[int] = set()
        destination = 0
        for stage_index, selected in enumerate(selected_sets):
            ranges = []
            if stage_index == 0:
                if prefix_tokens:
                    ranges.append((0, prefix_tokens))
                if suffix_tokens:
                    ranges.append((document_end, total_tokens))
            for index in ranking:
                if index in selected and index not in prior:
                    document = documents[index]
                    ranges.append((document.start, document.end))
            stage_tokens = sum(end - start for start, end in ranges)
            stages.append(
                StageTransfer(
                    ranges=tuple(ranges),
                    destination_start=destination,
                    destination_end=destination + stage_tokens,
                )
            )
            destination += stage_tokens
            prior = set(selected)
        if destination != total_tokens:
            raise RuntimeError("final progressive stage does not contain full KV")
        self.stages = tuple(stages)
        self.stage_cpu_cache = tuple(self._pack_stage(stage) for stage in self.stages)
        self.stage_caches = tuple(
            tuple(
                (
                    key[:, :, : stage.destination_end, :],
                    value[:, :, : stage.destination_end, :],
                )
                for key, value in self.gpu_cache
            )
            for stage in self.stages
        )
        self.masks = tuple(
            torch.ones((1, stage.destination_end), dtype=torch.long, device=device)
            for stage in self.stages
        )

    def _stage_bytes(self, stage_index: int) -> int:
        stage = self.stages[stage_index]
        tokens = stage.destination_end - stage.destination_start
        return tokens * self.layer_elements_per_token * self.element_size

    def _pack_stage(self, stage: StageTransfer):
        stage_tokens = stage.destination_end - stage.destination_start
        packed = []
        for cpu_key, cpu_value in self.cpu_cache:
            key = torch.empty(
                (cpu_key.shape[0], cpu_key.shape[1], stage_tokens, cpu_key.shape[3]),
                dtype=cpu_key.dtype,
                device="cpu",
                pin_memory=True,
            )
            value = torch.empty(
                (
                    cpu_value.shape[0],
                    cpu_value.shape[1],
                    stage_tokens,
                    cpu_value.shape[3],
                ),
                dtype=cpu_value.dtype,
                device="cpu",
                pin_memory=True,
            )
            cursor = 0
            for source_start, source_end in stage.ranges:
                width = source_end - source_start
                end = cursor + width
                key[:, :, cursor:end, :].copy_(
                    cpu_key[:, :, source_start:source_end, :]
                )
                value[:, :, cursor:end, :].copy_(
                    cpu_value[:, :, source_start:source_end, :]
                )
                cursor = end
            packed.append((key, value))
        return tuple(packed)

    def launch(self, stage_index: int) -> None:
        if self.launched[stage_index]:
            return
        if self.transport_gbps is not None and self.eager_serial_wire:
            if stage_index != 0:
                raise RuntimeError(
                    "eager serial wire must be initialized by launching stage 0"
                )
            started = perf_counter()
            ready_at = started
            for index in range(len(self.stages)):
                self.launched[index] = True
                self.wire_started_at[index] = ready_at
                ready_at += self._stage_bytes(index) * 8 / (self.transport_gbps * 1e9)
                self.wire_ready_at[index] = ready_at
            return
        self.launched[stage_index] = True
        if self.transport_gbps is not None:
            started = perf_counter()
            wire_seconds = (
                self._stage_bytes(stage_index) * 8 / (self.transport_gbps * 1e9)
            )
            self.wire_started_at[stage_index] = started
            self.wire_ready_at[stage_index] = started + wire_seconds
            return
        self._launch_copy(stage_index)

    def _launch_copy(self, stage_index: int) -> None:
        if self.physical_launched[stage_index]:
            return
        stage = self.stages[stage_index]
        destination_start = stage.destination_start
        destination_end = stage.destination_end
        with torch.cuda.stream(self.copy_stream):
            self.starts[stage_index].record(self.copy_stream)
            for (cpu_key, cpu_value), (gpu_key, gpu_value) in zip(
                self.stage_cpu_cache[stage_index], self.gpu_cache
            ):
                gpu_key[:, :, destination_start:destination_end, :].copy_(
                    cpu_key, non_blocking=True
                )
                gpu_value[:, :, destination_start:destination_end, :].copy_(
                    cpu_value, non_blocking=True
                )
            self.ends[stage_index].record(self.copy_stream)
        self.physical_launched[stage_index] = True

    def wait_on_compute_stream(self, stage_index: int) -> None:
        if not self.launched[stage_index]:
            raise RuntimeError(f"stage {stage_index} was not launched")
        ready_at = self.wire_ready_at[stage_index]
        if ready_at is not None:
            remaining = ready_at - perf_counter()
            if remaining > 0:
                sleep_started = perf_counter()
                sleep(remaining)
                self.wire_wait_ms[stage_index] += (
                    perf_counter() - sleep_started
                ) * 1000
            self._launch_copy(stage_index)
        torch.cuda.current_stream(device=self.device).wait_event(self.ends[stage_index])

    def launch_and_wait(self, stage_index: int) -> None:
        self.launch(stage_index)
        self.wait_on_compute_stream(stage_index)

    def poll_ready_copies(self) -> None:
        """Launch H2D for every wire-complete tranche without blocking compute."""
        now = perf_counter()
        for stage_index, launched in enumerate(self.launched):
            if not launched or self.physical_launched[stage_index]:
                continue
            ready_at = self.wire_ready_at[stage_index]
            if ready_at is None or now >= ready_at:
                self._launch_copy(stage_index)

    def stage_copy_ready(self, stage_index: int) -> bool:
        """Return whether a tranche is fully installed on the copy stream."""
        return self.physical_launched[stage_index] and self.ends[stage_index].query()

    def finish(self) -> dict[str, Any]:
        self.copy_stream.synchronize()
        finished_at = perf_counter()
        launched_indices = [
            index for index, launched in enumerate(self.physical_launched) if launched
        ]
        stage_h2d_ms = [
            self.starts[index].elapsed_time(self.ends[index])
            for index in launched_indices
        ]
        stage_tokens = [
            self.stages[index].destination_end - self.stages[index].destination_start
            for index in launched_indices
        ]
        stage_bytes = [self._stage_bytes(index) for index in launched_indices]
        requested_indices = [
            index for index, launched in enumerate(self.launched) if launched
        ]
        if self.transport_gbps is None:
            wire_bytes = sum(stage_bytes)
        else:
            bytes_per_second = self.transport_gbps * 1e9 / 8
            wire_bytes = sum(
                min(
                    self._stage_bytes(index),
                    max(0.0, finished_at - self.wire_started_at[index])
                    * bytes_per_second,
                )
                for index in requested_indices
            )
        return {
            "launched_stages": len(launched_indices),
            "requested_stages": len(requested_indices),
            "stage_tokens": stage_tokens,
            "stage_bytes": stage_bytes,
            "stage_h2d_ms": stage_h2d_ms,
            "total_bytes": sum(stage_bytes),
            "total_h2d_ms": sum(stage_h2d_ms),
            "first_stage_h2d_ms": stage_h2d_ms[0],
            "transport_gbps": self.transport_gbps,
            "stage_wire_ms": [
                self._stage_bytes(index) * 8 / (self.transport_gbps * 1e6)
                for index in requested_indices
            ]
            if self.transport_gbps is not None
            else [],
            "stage_wire_wait_ms": [
                self.wire_wait_ms[index] for index in requested_indices
            ],
            "wire_bytes_at_finish": wire_bytes,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument(
        "--dataset-format",
        choices=("musique", "longbench", "longbench_v2", "ruler"),
        default="musique",
    )
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
    parser.add_argument(
        "--auto-draft-tpot-ms",
        type=float,
        help=(
            "profiled sparse-draft milliseconds/token; when set, choose the "
            "per-request draft length from the residual wire window"
        ),
    )
    parser.add_argument("--auto-draft-utilization", type=float, default=0.8)
    parser.add_argument("--seed-tokens", type=int, default=1)
    parser.add_argument("--answer-word-limit", type=int, default=5)
    parser.add_argument("--context-page-tokens", type=int, default=256)
    parser.add_argument(
        "--ruler-placement",
        choices=(
            "original",
            "evidence_first",
            "evidence_uniform",
            "evidence_last",
            "adversarial_split",
        ),
        default="original",
    )
    parser.add_argument(
        "--longbench-prompt-mode",
        choices=("generic", "official", "official_chat"),
        default="official_chat",
        help=(
            "official uses the public dataset template verbatim; official_chat "
            "also applies the model chat template except on code-completion tasks"
        ),
    )
    parser.add_argument(
        "--official-output-length",
        action="store_true",
        help="override max-new-tokens with the public LongBench dataset limit",
    )
    parser.add_argument("--max-context-tokens", type=int)
    parser.add_argument(
        "--max-prompt-tokens",
        type=int,
        help=(
            "hard cap for the complete tokenized prompt, including template "
            "tokens; non-supporting documents are trimmed proportionally while "
            "supporting RULER evidence is preserved"
        ),
    )
    parser.add_argument(
        "--context-truncation",
        choices=("head", "middle"),
        default="middle",
        help="when max-context-tokens is set, retain the head only or both ends",
    )
    parser.add_argument("--target-document-tokens", type=int)
    parser.add_argument("--stage-fractions", default="0.2,0.4,0.6,0.8,1.0")
    parser.add_argument(
        "--draft-cache-mode",
        choices=("progressive", "s1", "grafted", "continuous"),
        default="progressive",
        help=(
            "s1 uses a physically compact first-stage KV for every draft pass; "
            "grafted appends newly arrived prompt KV in fixed batches; "
            "continuous polls arrival before every proposal and expands the "
            "draft prompt view without replay"
        ),
    )
    parser.add_argument(
        "--transport-gbps",
        type=float,
        help="pace serial tranche arrival at this network bandwidth before real H2D",
    )
    parser.add_argument(
        "--eager-serial-wire",
        action="store_true",
        help=(
            "schedule every tranche on one continuous wire stream when stage 0 "
            "launches; compute never leaves an artificial gap between tranches"
        ),
    )
    parser.add_argument("--reuse-final-verify", action="store_true")
    parser.add_argument(
        "--verification-mode",
        choices=("progressive", "progressive_tentative", "final_only"),
        default="progressive",
        help=(
            "progressive verifies and may commit at every arrival stage; "
            "progressive_tentative may repair drafts but commits only after the "
            "full-KV verifier; final_only skips all intermediate verification"
        ),
    )
    parser.add_argument(
        "--draft-stage-policy",
        choices=("first", "each"),
        default="each",
        help=(
            "first drafts only after S1 arrives; each drafts again after every "
            "intermediate arrival stage"
        ),
    )
    parser.add_argument("--schedules", default="query,oracle")
    parser.add_argument("--commit-windows", default="1,2")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--fixed-token-horizon",
        action="store_true",
        help=(
            "ignore EOS in timed D decoding so baseline and method generate "
            "the same token count; truncate at EOS only for quality metrics"
        ),
    )
    parser.add_argument(
        "--paired-fixed-control",
        action="store_true",
        help=(
            "for grafted/continuous runs, alternate an equal-total-proposal "
            "two-stage fixed-S1 control in the same model process"
        ),
    )
    parser.add_argument(
        "--paired-measurement",
        action="store_true",
        help=(
            "use the paper latency protocol: warm both paths, alternate whether "
            "the full-transfer baseline or pipeline is measured first by dataset "
            "index, and omit the extra serial-chain timing"
        ),
    )
    parser.add_argument(
        "--skip-serial",
        action="store_true",
        help="omit the no-overlap chain control from a selection/paired-control run",
    )
    parser.add_argument(
        "--require-exclusive-gpu",
        action="store_true",
        help=(
            "fail before model loading when more than 1 GiB is already used "
            "on the selected visible GPU"
        ),
    )
    args = parser.parse_args()
    args.stage_fractions = parse_fractions(args.stage_fractions)
    args.schedules = parse_schedules(args.schedules)
    args.commit_windows = parse_windows(args.commit_windows)
    if args.verification_mode in {"final_only", "progressive_tentative"}:
        if args.commit_windows != (None,):
            parser.error(f"{args.verification_mode} requires --commit-windows inf")
    elif None in args.commit_windows:
        parser.error("progressive verification requires a finite commit window")
    if args.draft_cache_mode in {"grafted", "continuous"}:
        if args.verification_mode != "final_only":
            parser.error(
                f"{args.draft_cache_mode} draft cache requires "
                "--verification-mode final_only"
            )
        if args.draft_cache_mode == "grafted" and args.draft_stage_policy != "each":
            parser.error("grafted draft cache requires --draft-stage-policy each")
        if args.seed_tokens != 1:
            parser.error(
                f"{args.draft_cache_mode} draft cache currently requires "
                "--seed-tokens 1"
            )
        if args.auto_draft_tpot_ms is not None:
            parser.error(
                f"{args.draft_cache_mode} draft cache does not yet support "
                "auto draft sizing"
            )
    if args.draft_cache_mode == "continuous":
        if args.transport_gbps is None or not args.eager_serial_wire:
            parser.error(
                "continuous draft cache requires --transport-gbps and "
                "--eager-serial-wire"
            )
        if len(args.stage_fractions) < 3:
            parser.error("continuous draft cache requires at least three stages")
    if args.paired_fixed_control and args.draft_cache_mode not in {
        "grafted",
        "continuous",
    }:
        parser.error("paired fixed control is only defined for grafted/continuous runs")
    if args.paired_measurement:
        if len(args.commit_windows) != 1:
            parser.error("paired measurement requires exactly one commit window")
        if args.paired_fixed_control:
            parser.error(
                "paired measurement and paired fixed control cannot be combined"
            )
    if (
        min(
            args.sample_count,
            args.count,
            args.max_new_tokens,
            args.draft_tokens,
            args.seed_tokens,
        )
        <= 0
    ):
        parser.error("sample/count/generation lengths must be positive")
    if args.seed_tokens >= args.max_new_tokens:
        parser.error("seed-tokens must be smaller than max-new-tokens")
    if args.target_document_tokens is not None and args.target_document_tokens <= 0:
        parser.error("target-document-tokens must be positive")
    if args.context_page_tokens <= 0:
        parser.error("context-page-tokens must be positive")
    if args.max_context_tokens is not None and args.max_context_tokens <= 0:
        parser.error("max-context-tokens must be positive")
    if args.max_prompt_tokens is not None and args.max_prompt_tokens <= 0:
        parser.error("max-prompt-tokens must be positive")
    if args.transport_gbps is not None and args.transport_gbps <= 0:
        parser.error("transport-gbps must be positive")
    if args.auto_draft_tpot_ms is not None:
        if args.auto_draft_tpot_ms <= 0:
            parser.error("auto-draft-tpot-ms must be positive")
        if args.transport_gbps is None:
            parser.error("auto draft sizing requires --transport-gbps")
    if not 0 < args.auto_draft_utilization <= 1:
        parser.error("auto-draft-utilization must lie in (0, 1]")
    if args.offset < 0 or args.offset + args.count > args.sample_count:
        parser.error("invalid sample shard")
    return args


def tokenize_longbench_prompt(
    tokenizer,
    row,
    *,
    page_tokens,
    max_context_tokens=None,
    prompt_mode="official_chat",
    context_truncation="middle",
):
    placeholder = "{CONTEXT}"
    if prompt_mode == "generic":
        messages = [
            {
                "role": "system",
                "content": (
                    "Complete the user's task using only the supplied long "
                    "context. Be accurate and do not omit requested details."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"<CONTEXT>\n{placeholder}\n</CONTEXT>\n\n{row['input'].strip()}"
                ),
            },
        ]
        rendered = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    else:
        dataset_name = row["dataset"]
        if dataset_name not in LONG_BENCH_PROMPTS:
            raise ValueError(
                f"no vendored LongBench prompt template for {dataset_name}"
            )
        official = LONG_BENCH_PROMPTS[dataset_name].format(
            context=placeholder,
            input=row["input"],
        )
        if prompt_mode == "official_chat" and dataset_name not in {
            "lcc",
            "repobench-p",
        }:
            rendered = tokenizer.apply_chat_template(
                [{"role": "user", "content": official}],
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            rendered = official
    return tokenize_rendered_context_prompt(
        tokenizer,
        rendered,
        placeholder,
        row["context"],
        page_tokens=page_tokens,
        max_context_tokens=max_context_tokens,
        context_truncation=context_truncation,
    )


def tokenize_longbench_v2_prompt(
    tokenizer,
    row,
    *,
    page_tokens,
    max_context_tokens=None,
    prompt_mode="official_chat",
    context_truncation="middle",
):
    placeholder = "{CONTEXT}"
    official = LONG_BENCH_V2_PROMPT.format(
        context=placeholder,
        question=row["question"],
        choice_A=row["choice_A"],
        choice_B=row["choice_B"],
        choice_C=row["choice_C"],
        choice_D=row["choice_D"],
    )
    if prompt_mode in {"generic", "official_chat"}:
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": official}],
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        rendered = official
    return tokenize_rendered_context_prompt(
        tokenizer,
        rendered,
        placeholder,
        row["context"],
        page_tokens=page_tokens,
        max_context_tokens=max_context_tokens,
        context_truncation=context_truncation,
    )


def tokenize_ruler_prompt(tokenizer, row, *, placement):
    placeholder = "{DOCUMENTS}"
    rendered = (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>"
        "You are a helpful assistant<|eot_id|><|start_header_id|>user"
        "<|end_header_id|>Answer the question based on the given documents. "
        "Only give me the answer and do not output any other words.\n\nThe "
        f"following are given documents.\n\n{placeholder}\n\nAnswer the question "
        "based on the given documents. Only give me the answer and do not "
        f"output any other words.\n\nQuestion: {row['question']}<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|> Answer:"
    )
    pivot = rendered.index(placeholder)
    prefix = tuple(tokenizer.encode(rendered[:pivot], add_special_tokens=False))
    suffix = tuple(
        tokenizer.encode(rendered[pivot + len(placeholder) :], add_special_tokens=False)
    )
    order = row["placements"][placement]
    cursor = len(prefix)
    documents = []
    for displayed_index, source_index in enumerate(order, start=1):
        source = row["documents"][source_index]
        separator = "\n\n" if displayed_index < len(order) else ""
        text = f"Document {displayed_index}:\n{source['text']}{separator}"
        token_ids = tuple(tokenizer.encode(text, add_special_tokens=False))
        documents.append(
            Document(
                token_ids=token_ids,
                text=text,
                supporting=bool(source["supporting"]),
                start=cursor,
                end=cursor + len(token_ids),
            )
        )
        cursor += len(token_ids)
    return prefix, tuple(documents), suffix


def trim_documents_to_prompt_budget(
    tokenizer,
    prefix,
    documents,
    suffix,
    max_prompt_tokens,
):
    """Fit a prompt budget without changing any supporting document tokens.

    The remaining budget is divided proportionally across distractors. Every
    distractor retains at least one token, so document order and stage topology
    remain unchanged across RULER evidence-placement controls.
    """
    if max_prompt_tokens is None:
        return documents
    fixed_tokens = len(prefix) + len(suffix)
    document_budget = max_prompt_tokens - fixed_tokens
    if document_budget <= 0:
        raise ValueError("max-prompt-tokens does not leave room for documents")
    original_total = sum(len(document.token_ids) for document in documents)
    if original_total <= document_budget:
        return documents

    supporting_total = sum(
        len(document.token_ids) for document in documents if document.supporting
    )
    distractor_indices = [
        index for index, document in enumerate(documents) if not document.supporting
    ]
    distractor_budget = document_budget - supporting_total
    if not distractor_indices:
        raise ValueError("supporting documents alone exceed max-prompt-tokens")
    if distractor_budget < len(distractor_indices):
        raise ValueError("max-prompt-tokens cannot retain every distractor document")

    distractor_lengths = {
        index: len(documents[index].token_ids) for index in distractor_indices
    }
    if any(length <= 0 for length in distractor_lengths.values()):
        raise ValueError("cannot trim an empty distractor document")
    removable_total = sum(length - 1 for length in distractor_lengths.values())
    extra_budget = distractor_budget - len(distractor_indices)
    if extra_budget > removable_total:
        raise AssertionError("distractor budget exceeds original prompt")

    lengths = {index: 1 for index in distractor_indices}
    if removable_total:
        exact_extra = {
            index: extra_budget * (distractor_lengths[index] - 1) / removable_total
            for index in distractor_indices
        }
        for index, value in exact_extra.items():
            lengths[index] += math.floor(value)
        remainder = distractor_budget - sum(lengths.values())
        order = sorted(
            distractor_indices,
            key=lambda index: (
                -(exact_extra[index] - math.floor(exact_extra[index])),
                index,
            ),
        )
        for index in order[:remainder]:
            lengths[index] += 1

    cursor = len(prefix)
    trimmed = []
    for index, document in enumerate(documents):
        length = len(document.token_ids) if document.supporting else lengths[index]
        token_ids = document.token_ids[:length]
        text = (
            document.text
            if document.supporting
            else tokenizer.decode(token_ids, skip_special_tokens=True)
        )
        trimmed.append(
            Document(
                token_ids=token_ids,
                text=text,
                supporting=document.supporting,
                start=cursor,
                end=cursor + length,
            )
        )
        cursor += length
    if cursor + len(suffix) != max_prompt_tokens:
        raise AssertionError("prompt-budget allocation is not exact")
    return tuple(trimmed)


def tokenize_rendered_context_prompt(
    tokenizer,
    rendered,
    placeholder,
    context,
    *,
    page_tokens,
    max_context_tokens,
    context_truncation,
):
    pivot = rendered.index(placeholder)
    prefix = tuple(tokenizer.encode(rendered[:pivot], add_special_tokens=False))
    suffix = tuple(
        tokenizer.encode(rendered[pivot + len(placeholder) :], add_special_tokens=False)
    )
    context_ids = tuple(tokenizer.encode(context, add_special_tokens=False))
    if max_context_tokens is not None and len(context_ids) > max_context_tokens:
        if context_truncation == "head":
            context_ids = context_ids[:max_context_tokens]
        else:
            head = math.ceil(max_context_tokens / 2)
            tail = max_context_tokens - head
            context_ids = context_ids[:head] + (context_ids[-tail:] if tail else ())
    cursor = len(prefix)
    documents = []
    for start in range(0, len(context_ids), page_tokens):
        token_ids = context_ids[start : start + page_tokens]
        text = tokenizer.decode(token_ids, skip_special_tokens=True)
        documents.append(
            Document(
                token_ids=token_ids,
                text=text,
                supporting=False,
                start=cursor,
                end=cursor + len(token_ids),
            )
        )
        cursor += len(token_ids)
    return prefix, tuple(documents), suffix


def categorical_stratified_indices(rows, count, seed, fields):
    if count > len(rows):
        raise ValueError("sample-count exceeds dataset size")
    groups = {}
    for index, row in enumerate(rows):
        key = tuple(row.get(field) for field in fields)
        groups.setdefault(key, []).append(index)
    exact = {key: count * len(indices) / len(rows) for key, indices in groups.items()}
    quotas = {key: math.floor(value) for key, value in exact.items()}
    remaining = count - sum(quotas.values())
    for key in sorted(
        groups,
        key=lambda item: (-(exact[item] - quotas[item]), item),
    )[:remaining]:
        quotas[key] += 1
    rng = random.Random(seed)
    selected = []
    for key in sorted(groups):
        candidates = list(groups[key])
        rng.shuffle(candidates)
        selected.extend(candidates[: quotas[key]])
    rng.shuffle(selected)
    if len(selected) != count:
        raise RuntimeError("failed to construct categorical stratified sample")
    return selected


def task_scores(text, golds, dataset_format, dataset_name, all_classes=None):
    if dataset_format in {"longbench", "longbench_v2"}:
        return score_prediction(dataset_name, text, golds, all_classes)
    em, score = answer_scores(text, golds)
    return "qa_f1", em, score


def first_difference(left, right):
    for index, (left_item, right_item) in enumerate(zip(left, right)):
        if left_item != right_item:
            return index
    return None if len(left) == len(right) else min(len(left), len(right))


def truncate_at_first_eos(token_ids, eos_ids):
    for index, token_id in enumerate(token_ids):
        if token_id in eos_ids:
            return list(token_ids[: index + 1])
    return list(token_ids)


def expand_documents(documents, target_tokens, prefix_tokens):
    """Proportionally repeat document tokens for a timing-only scale sweep."""
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
        repeats = (length + len(document.token_ids) - 1) // len(document.token_ids)
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


def offload_to_pinned_cpu(cache, device) -> tuple[tuple[Any, ...], float]:
    torch.cuda.synchronize(device)
    started = perf_counter()
    cpu_cache = []
    for key, value in cache:
        cpu_key = torch.empty(key.shape, dtype=key.dtype, device="cpu", pin_memory=True)
        cpu_value = torch.empty(
            value.shape, dtype=value.dtype, device="cpu", pin_memory=True
        )
        cpu_key.copy_(key, non_blocking=True)
        cpu_value.copy_(value, non_blocking=True)
        cpu_cache.append((cpu_key, cpu_value))
    torch.cuda.synchronize(device)
    return tuple(cpu_cache), (perf_counter() - started) * 1000


def run_full_target(
    model,
    cpu_cache,
    *,
    prefix_tokens,
    documents,
    suffix_tokens,
    ranking,
    seed_ids,
    prompt_tokens,
    max_new_tokens,
    eos_ids,
    device,
    transport_gbps=None,
):
    all_documents = set(range(len(documents)))
    stager = PinnedKVStager(
        cpu_cache,
        prefix_tokens=prefix_tokens,
        documents=documents,
        suffix_tokens=suffix_tokens,
        ranking=ranking,
        selected_sets=(all_documents,),
        device=device,
        transport_gbps=transport_gbps,
    )
    started = perf_counter()
    stager.launch(0)
    stager.wait_on_compute_stream(0)
    tail, seed_forward_ms, decode_ms = continue_from_view(
        model,
        stager.stage_caches[0],
        stager.masks[0],
        seed_ids,
        (),
        additional_tokens=max_new_tokens - len(seed_ids),
        eos_ids=eos_ids,
        device=device,
        position_base=prompt_tokens,
        synchronize_device=False,
    )
    torch.cuda.current_stream(device=device).synchronize()
    response_ms = (perf_counter() - started) * 1000
    transfer = stager.finish()
    del stager
    torch.cuda.empty_cache()
    return list(seed_ids) + list(tail), {
        "response_ms": response_ms,
        "seed_forward_ms": seed_forward_ms,
        "decode_ms": decode_ms,
        "transfer": transfer,
    }


def run_chain(
    model,
    cpu_cache,
    *,
    prefix_tokens,
    documents,
    suffix_tokens,
    ranking,
    selected_sets,
    seed_ids,
    prompt_tokens,
    commit_window,
    draft_tokens,
    max_new_tokens,
    eos_ids,
    device,
    overlap,
    draft_cache_mode="progressive",
    transport_gbps=None,
    reuse_final_verify=False,
    verify_intermediate=True,
    draft_intermediate=True,
    eager_serial_wire=False,
):
    stager = PinnedKVStager(
        cpu_cache,
        prefix_tokens=prefix_tokens,
        documents=documents,
        suffix_tokens=suffix_tokens,
        ranking=ranking,
        selected_sets=selected_sets,
        device=device,
        transport_gbps=transport_gbps,
        eager_serial_wire=eager_serial_wire and overlap,
    )
    started = perf_counter()
    if overlap:
        stager.launch(0)
        stage_wait = stager.wait_on_compute_stream
        stage_start = stager.launch
    else:
        stage_wait = stager.launch_and_wait
        stage_start = None
    if draft_cache_mode == "grafted":
        chain = progressive_graft_chain(
            model,
            stager.masks,
            seed_ids,
            draft_tokens=draft_tokens,
            max_new_tokens=max_new_tokens - len(seed_ids),
            eos_ids=eos_ids,
            device=device,
            stage_caches=stager.stage_caches,
            position_base=prompt_tokens,
            stage_wait=stage_wait,
            stage_start=stage_start,
            synchronize_device=False,
            reuse_final_verify=reuse_final_verify,
        )
    elif draft_cache_mode == "continuous":
        if transport_gbps is None:
            # Shape warmup: force every prompt-view graft once.  The result is
            # discarded by the caller and never enters a response timer.
            chain = progressive_graft_chain(
                model,
                stager.masks,
                seed_ids,
                draft_tokens=max(1, math.ceil(draft_tokens / (len(stager.masks) - 1))),
                max_new_tokens=max_new_tokens - len(seed_ids),
                eos_ids=eos_ids,
                device=device,
                stage_caches=stager.stage_caches,
                position_base=prompt_tokens,
                stage_wait=stage_wait,
                stage_start=stage_start,
                synchronize_device=False,
                reuse_final_verify=reuse_final_verify,
            )
        else:
            chain = continuous_graft_chain(
                model,
                stager.masks,
                seed_ids,
                draft_tokens=draft_tokens,
                max_new_tokens=max_new_tokens - len(seed_ids),
                eos_ids=eos_ids,
                device=device,
                stage_caches=stager.stage_caches,
                position_base=prompt_tokens,
                stage_wait=stage_wait,
                stage_ready=stager.stage_copy_ready,
                stage_poll=stager.poll_ready_copies,
                synchronize_device=False,
                reuse_final_verify=reuse_final_verify,
            )
    else:
        chain = progressive_chain(
            model,
            None,
            stager.masks,
            seed_ids,
            commit_window=commit_window,
            draft_tokens=draft_tokens,
            max_new_tokens=max_new_tokens - len(seed_ids),
            eos_ids=eos_ids,
            device=device,
            stage_caches=stager.stage_caches,
            position_base=prompt_tokens,
            stage_wait=stage_wait,
            stage_start=stage_start,
            synchronize_device=False,
            draft_cache=(stager.stage_caches[0] if draft_cache_mode == "s1" else None),
            draft_mask=(stager.masks[0] if draft_cache_mode == "s1" else None),
            reuse_final_verify=reuse_final_verify,
            verify_intermediate=verify_intermediate,
            draft_intermediate=draft_intermediate,
        )
    torch.cuda.current_stream(device=device).synchronize()
    response_ms = (perf_counter() - started) * 1000
    transfer = stager.finish()
    token_ids = list(seed_ids) + chain.pop("token_ids")
    del stager
    torch.cuda.empty_cache()
    return token_ids, {"response_ms": response_ms, "transfer": transfer, **chain}


def evaluate_example(model, tokenizer, row, dataset_index, args, device):
    if args.dataset_format == "longbench":
        prefix, documents, suffix = tokenize_longbench_prompt(
            tokenizer,
            row,
            page_tokens=args.context_page_tokens,
            max_context_tokens=args.max_context_tokens,
            prompt_mode=args.longbench_prompt_mode,
            context_truncation=args.context_truncation,
        )
        question = row["input"]
        golds = list(row.get("answers", []))
        dataset_name = row.get("dataset", "longbench")
    elif args.dataset_format == "longbench_v2":
        prefix, documents, suffix = tokenize_longbench_v2_prompt(
            tokenizer,
            row,
            page_tokens=args.context_page_tokens,
            max_context_tokens=args.max_context_tokens,
            prompt_mode=args.longbench_prompt_mode,
            context_truncation=args.context_truncation,
        )
        question = row["question"]
        golds = [row["answer"]]
        dataset_name = "longbench_v2"
    elif args.dataset_format == "ruler":
        prefix, documents, suffix = tokenize_ruler_prompt(
            tokenizer, row, placement=args.ruler_placement
        )
        question = row["question"]
        golds = list(row["answers"])
        dataset_name = "ruler_qa2"
    else:
        prefix, documents, suffix = tokenize_prompt(
            tokenizer, row, args.answer_word_limit
        )
        question = row["question"]
        golds = [row["answer"], *row.get("answer_aliases", [])]
        dataset_name = "musique"
    untrimmed_prompt_tokens = (
        len(prefix)
        + sum(len(document.token_ids) for document in documents)
        + len(suffix)
    )
    documents = trim_documents_to_prompt_budget(
        tokenizer,
        prefix,
        documents,
        suffix,
        args.max_prompt_tokens,
    )
    documents = expand_documents(documents, args.target_document_tokens, len(prefix))
    prompt = tuple(prefix)
    for document in documents:
        prompt += document.token_ids
    prompt += tuple(suffix)
    if args.max_prompt_tokens is not None and len(prompt) > args.max_prompt_tokens:
        raise ValueError(
            "target-document-tokens expanded the prompt beyond max-prompt-tokens"
        )
    max_positions = getattr(model.config, "max_position_embeddings", None)
    if max_positions is not None and len(prompt) + args.max_new_tokens > max_positions:
        raise ValueError(
            f"prompt ({len(prompt)}) + output ({args.max_new_tokens}) exceeds "
            f"model context window ({max_positions})"
        )
    quality_eos_ids = eos_token_ids(model, tokenizer)
    timing_eos_ids = set() if args.fixed_token_horizon else quality_eos_ids

    p_tokens, p_prefill_ms, p_decode_ms = full_prefill_generate(
        model,
        prompt,
        max_new_tokens=args.seed_tokens,
        eos_ids=quality_eos_ids,
        device=device,
    )
    producer_attention_scores = None
    producer_attention_profile = None
    if "p_attention" in args.schedules:
        p_query_positions = question_token_positions(
            tokenizer, prompt, question, device
        )
        (
            full_cache,
            producer_attention_scores,
            producer_attention_profile,
        ) = encode_cache_with_producer_attention_scores(
            model, prompt, documents, p_query_positions, device
        )
    else:
        full_cache = encode_contiguous_cache(model, prompt, device)
    cpu_cache, producer_d2h_ms = offload_to_pinned_cpu(full_cache, device)
    logical_bytes = sum(
        key.numel() * key.element_size() + value.numel() * value.element_size()
        for key, value in cpu_cache
    )
    del full_cache
    gc.collect()
    torch.cuda.empty_cache()

    seed_ids = list(p_tokens[: args.seed_tokens])
    schedules = {}
    for schedule in args.schedules:
        ranking = ranked_document_indices(
            schedule,
            question=question,
            documents=documents,
            random_seed=args.seed + dataset_index,
            attention_scores=producer_attention_scores,
        )
        selected_sets = stage_document_sets(ranking, documents, args.stage_fractions)
        selected_first_tokens = (
            len(prefix)
            + len(suffix)
            + sum(len(documents[index].token_ids) for index in selected_sets[0])
        )
        residual_tokens = len(prompt) - selected_first_tokens
        if args.auto_draft_tpot_ms is None:
            request_draft_tokens = args.draft_tokens
            residual_wire_ms = None
        else:
            bytes_per_token = logical_bytes / len(prompt)
            residual_wire_ms = (
                residual_tokens * bytes_per_token * 8 / (args.transport_gbps * 1e6)
            )
            request_draft_tokens = max(
                1,
                min(
                    args.draft_tokens,
                    math.floor(
                        residual_wire_ms
                        * args.auto_draft_utilization
                        / args.auto_draft_tpot_ms
                    ),
                ),
            )

        def arguments_for_window(
            commit_window,
            ranking=ranking,
            selected_sets=selected_sets,
            request_draft_tokens=request_draft_tokens,
        ):
            return {
                "prefix_tokens": len(prefix),
                "documents": documents,
                "suffix_tokens": len(suffix),
                "ranking": ranking,
                "selected_sets": selected_sets,
                "seed_ids": seed_ids,
                "prompt_tokens": len(prompt),
                "commit_window": commit_window,
                "draft_tokens": request_draft_tokens,
                "max_new_tokens": args.max_new_tokens,
                "eos_ids": timing_eos_ids,
                "device": device,
                "draft_cache_mode": args.draft_cache_mode,
                "transport_gbps": args.transport_gbps,
                "reuse_final_verify": args.reuse_final_verify,
                "verify_intermediate": (args.verification_mode != "final_only"),
                "draft_intermediate": args.draft_stage_policy == "each",
                "eager_serial_wire": args.eager_serial_wire,
            }

        # Warm both full-length paths before either paired measurement.  This
        # removes first-use SDPA/copy effects without favoring one timed path.
        run_full_target(
            model,
            cpu_cache,
            prefix_tokens=len(prefix),
            documents=documents,
            suffix_tokens=len(suffix),
            ranking=ranking,
            seed_ids=seed_ids,
            prompt_tokens=len(prompt),
            max_new_tokens=args.max_new_tokens,
            eos_ids=timing_eos_ids,
            device=device,
            transport_gbps=None,
        )
        paired_pipeline = None
        measurement_order = "legacy"
        if args.paired_measurement:
            paired_arguments = arguments_for_window(args.commit_windows[0])
            run_chain(
                model,
                cpu_cache,
                overlap=True,
                **{**paired_arguments, "transport_gbps": None},
            )
            if dataset_index % 2:
                paired_pipeline = run_chain(
                    model,
                    cpu_cache,
                    overlap=True,
                    **paired_arguments,
                )
                measurement_order = "pipeline_then_full_target"
            else:
                measurement_order = "full_target_then_pipeline"
        target_ids, target_timing = run_full_target(
            model,
            cpu_cache,
            prefix_tokens=len(prefix),
            documents=documents,
            suffix_tokens=len(suffix),
            ranking=ranking,
            seed_ids=seed_ids,
            prompt_tokens=len(prompt),
            max_new_tokens=args.max_new_tokens,
            eos_ids=timing_eos_ids,
            device=device,
            transport_gbps=args.transport_gbps,
        )
        target_quality_ids = truncate_at_first_eos(target_ids, quality_eos_ids)
        target_text = decode_text(tokenizer, target_quality_ids)
        metric_name, target_em, target_f1 = task_scores(
            target_text,
            golds,
            args.dataset_format,
            dataset_name,
            row.get("all_classes"),
        )
        chains = {}
        for commit_window in args.commit_windows:
            chain_arguments = arguments_for_window(commit_window)
            # Warm every stage-specific model/copy shape. The warmup is not
            # retained and is outside both response timers.
            if not args.paired_measurement:
                run_chain(
                    model,
                    cpu_cache,
                    overlap=True,
                    **{
                        **chain_arguments,
                        "transport_gbps": None,
                    },
                )
            fixed_arguments = None
            if args.paired_fixed_control:
                fixed_arguments = {
                    **chain_arguments,
                    "selected_sets": (
                        selected_sets[0],
                        set(range(len(documents))),
                    ),
                    "draft_tokens": (
                        request_draft_tokens * (len(selected_sets) - 1)
                        if args.draft_cache_mode == "grafted"
                        else request_draft_tokens
                    ),
                    "draft_cache_mode": (
                        "continuous" if args.draft_cache_mode == "continuous" else "s1"
                    ),
                    "verify_intermediate": False,
                    "draft_intermediate": False,
                }
                run_chain(
                    model,
                    cpu_cache,
                    overlap=True,
                    **{
                        **fixed_arguments,
                        "transport_gbps": None,
                    },
                )
            window_order_key = 0 if commit_window is None else commit_window
            if args.paired_measurement:
                if paired_pipeline is None:
                    pipeline_ids, pipeline = run_chain(
                        model,
                        cpu_cache,
                        overlap=True,
                        **chain_arguments,
                    )
                else:
                    pipeline_ids, pipeline = paired_pipeline
                serial_ids = None
                serial = None
            elif (dataset_index + window_order_key) % 2:
                pipeline_ids, pipeline = run_chain(
                    model,
                    cpu_cache,
                    overlap=True,
                    **chain_arguments,
                )
                if fixed_arguments is not None:
                    fixed_ids, fixed_control = run_chain(
                        model,
                        cpu_cache,
                        overlap=True,
                        **fixed_arguments,
                    )
                if args.skip_serial:
                    serial_ids = None
                    serial = None
                else:
                    serial_ids, serial = run_chain(
                        model,
                        cpu_cache,
                        overlap=False,
                        **chain_arguments,
                    )
            else:
                if args.skip_serial:
                    serial_ids = None
                    serial = None
                else:
                    serial_ids, serial = run_chain(
                        model,
                        cpu_cache,
                        overlap=False,
                        **chain_arguments,
                    )
                if fixed_arguments is not None:
                    fixed_ids, fixed_control = run_chain(
                        model,
                        cpu_cache,
                        overlap=True,
                        **fixed_arguments,
                    )
                pipeline_ids, pipeline = run_chain(
                    model,
                    cpu_cache,
                    overlap=True,
                    **chain_arguments,
                )
            pipeline_quality_ids = truncate_at_first_eos(pipeline_ids, quality_eos_ids)
            pipeline_text = decode_text(tokenizer, pipeline_quality_ids)
            _, em, f1 = task_scores(
                pipeline_text,
                golds,
                args.dataset_format,
                dataset_name,
                row.get("all_classes"),
            )
            fixed_payload = None
            if fixed_arguments is not None:
                fixed_quality_ids = truncate_at_first_eos(fixed_ids, quality_eos_ids)
                fixed_text = decode_text(tokenizer, fixed_quality_ids)
                _, fixed_em, fixed_f1 = task_scores(
                    fixed_text,
                    golds,
                    args.dataset_format,
                    dataset_name,
                    row.get("all_classes"),
                )
                fixed_payload = {
                    "answer": clean_answer(fixed_text),
                    "token_ids": fixed_quality_ids,
                    "timing_token_ids": fixed_ids,
                    "em": fixed_em,
                    "f1": fixed_f1,
                    "task_metric": metric_name,
                    "task_score": fixed_f1,
                    "configured_draft_tokens": fixed_arguments["draft_tokens"],
                    "token_match_target": (fixed_quality_ids == target_quality_ids),
                    "timing_token_match_target": fixed_ids == target_ids,
                    "normalized_match_target": (
                        normalize_answer(fixed_text) == normalize_answer(target_text)
                    ),
                    "first_difference_target": first_difference(
                        fixed_quality_ids, target_quality_ids
                    ),
                    "speedup_vs_full_target": (
                        target_timing["response_ms"] / fixed_control["response_ms"]
                    ),
                    **fixed_control,
                }
            window_name = "inf" if commit_window is None else str(commit_window)
            chains[window_name] = {
                "answer": clean_answer(pipeline_text),
                "token_ids": pipeline_quality_ids,
                "timing_token_ids": pipeline_ids,
                "em": em,
                "f1": f1,
                "task_metric": metric_name,
                "task_score": f1,
                "configured_draft_tokens": request_draft_tokens,
                "estimated_residual_wire_ms": residual_wire_ms,
                "token_match_target": (pipeline_quality_ids == target_quality_ids),
                "timing_token_match_target": pipeline_ids == target_ids,
                "normalized_match_target": (
                    normalize_answer(pipeline_text) == normalize_answer(target_text)
                ),
                "first_difference_target": first_difference(
                    pipeline_quality_ids, target_quality_ids
                ),
                "serial_pipeline_token_match": (
                    None if serial_ids is None else serial_ids == pipeline_ids
                ),
                "serial": serial,
                "pipeline": pipeline,
                "pipeline_speedup_vs_serial": (
                    None
                    if serial is None
                    else serial["response_ms"] / pipeline["response_ms"]
                ),
                "pipeline_speedup_vs_full_target": (
                    target_timing["response_ms"] / pipeline["response_ms"]
                ),
                "paired_fixed_control": fixed_payload,
                "measurement_order": measurement_order,
            }
        schedules[schedule] = {
            "ranking": ranking,
            "stage_document_counts": [len(selected) for selected in selected_sets],
            "full_target": {
                "answer": clean_answer(target_text),
                "token_ids": target_quality_ids,
                "timing_token_ids": target_ids,
                "em": target_em,
                "f1": target_f1,
                "task_metric": metric_name,
                "task_score": target_f1,
                **target_timing,
            },
            "chains": chains,
        }

    del cpu_cache
    gc.collect()
    torch.cuda.empty_cache()
    return {
        "dataset_index": dataset_index,
        "id": row.get("id", row.get("_id")),
        "hop_count": len(row.get("question_decomposition", [])),
        "dataset_name": dataset_name,
        "domain": row.get("domain"),
        "sub_domain": row.get("sub_domain"),
        "difficulty": row.get("difficulty"),
        "source_length": row.get("length", row.get("source_length_tokens")),
        "target_length": row.get("target_length"),
        "ruler_placement": (
            args.ruler_placement if args.dataset_format == "ruler" else None
        ),
        "question": question,
        "gold_answers": golds,
        "all_classes": row.get("all_classes"),
        "prompt_tokens": len(prompt),
        "untrimmed_prompt_tokens": untrimmed_prompt_tokens,
        "prompt_trimmed_tokens": untrimmed_prompt_tokens - len(prompt),
        "anchor_tokens": len(prefix) + len(suffix),
        "document_tokens": sum(len(document.token_ids) for document in documents),
        "document_pages": len(documents),
        "logical_bf16_kv_bytes": logical_bytes,
        "producer": {
            "prefill_ms": p_prefill_ms,
            "decode_ms": p_decode_ms,
            "d2h_to_pinned_cpu_ms": producer_d2h_ms,
            "attention_page_scorer": producer_attention_profile,
        },
        "schedules": schedules,
    }


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"{output} exists; pass --overwrite")
    output.parent.mkdir(parents=True, exist_ok=True)
    if args.dataset_format == "longbench_v2":
        rows = json.loads(Path(args.dataset).read_text(encoding="utf-8"))
        if not isinstance(rows, list):
            raise ValueError("LongBench-v2 dataset must be a JSON list")
    else:
        rows = load_jsonl(args.dataset)
    if args.official_output_length:
        if args.dataset_format == "longbench_v2":
            args.max_new_tokens = LONG_BENCH_V2_MAX_NEW_TOKENS
        elif args.dataset_format == "ruler":
            args.max_new_tokens = 32
        elif args.dataset_format == "longbench":
            dataset_names = {row.get("dataset") for row in rows}
            if len(dataset_names) != 1:
                raise ValueError(
                    "official output length requires one LongBench dataset"
                )
            dataset_name = next(iter(dataset_names))
            if dataset_name not in LONG_BENCH_MAX_NEW_TOKENS:
                raise ValueError(
                    f"no vendored LongBench output limit for {dataset_name}"
                )
            args.max_new_tokens = LONG_BENCH_MAX_NEW_TOKENS[dataset_name]
        else:
            raise ValueError("official output length is defined only for LongBench")
        if args.seed_tokens >= args.max_new_tokens:
            raise ValueError("seed-tokens must be smaller than official output limit")
    if args.dataset_format == "longbench_v2":
        selected = categorical_stratified_indices(
            rows,
            args.sample_count,
            args.seed,
            ("domain", "difficulty", "length"),
        )
    else:
        selected = stratified_indices(rows, args.sample_count, args.seed)
    shard_indices = selected[args.offset : args.offset + args.count]

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    initial_free_bytes, initial_total_bytes = torch.cuda.mem_get_info(device)
    args.initial_gpu_used_bytes = initial_total_bytes - initial_free_bytes
    args.initial_gpu_total_bytes = initial_total_bytes
    if args.require_exclusive_gpu and args.initial_gpu_used_bytes > 1 * 2**30:
        raise RuntimeError(
            "selected GPU is not exclusive: "
            f"{args.initial_gpu_used_bytes / 2**30:.2f} GiB was already used "
            "before model loading"
        )
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = (
        AutoModelForCausalLM.from_pretrained(
            args.model,
            local_files_only=True,
            torch_dtype=getattr(torch, args.dtype),
            attn_implementation="sdpa",
        )
        .eval()
        .to(device)
    )
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    protocol = {
        "experiment": "pd-progressive-real-pinned-cpu-h2d",
        "model": str(Path(args.model).resolve()),
        "dataset": str(Path(args.dataset).resolve()),
        "dataset_format": args.dataset_format,
        "longbench_prompt_mode": (
            args.longbench_prompt_mode
            if args.dataset_format in {"longbench", "longbench_v2"}
            else None
        ),
        "official_output_length": args.official_output_length,
        "max_new_tokens": args.max_new_tokens,
        "task_metrics": (
            (
                "LongBench-v2 direct-answer choice accuracy"
                if args.dataset_format == "longbench_v2"
                else "public LongBench dataset-to-metric mapping with rouge==1.0.1 and fuzzywuzzy==0.18.0"
            )
            if args.dataset_format in {"longbench", "longbench_v2"}
            else "normalized QA EM/F1"
        ),
        "context_page_tokens": (
            args.context_page_tokens
            if args.dataset_format in {"longbench", "longbench_v2"}
            else None
        ),
        "max_context_tokens": args.max_context_tokens,
        "max_prompt_tokens": args.max_prompt_tokens,
        "context_truncation": args.context_truncation,
        "ruler_placement": (
            args.ruler_placement if args.dataset_format == "ruler" else None
        ),
        "sample_count": args.sample_count,
        "sample_seed": args.seed,
        "shard_offset": args.offset,
        "shard_count": args.count,
        "stage_fractions": args.stage_fractions,
        "target_document_tokens": args.target_document_tokens,
        "schedules": args.schedules,
        "commit_windows": [
            "inf" if window is None else window for window in args.commit_windows
        ],
        "draft_tokens": args.draft_tokens,
        "draft_tokens_semantics": (
            "fixed per active non-final stage"
            if args.draft_cache_mode == "grafted"
            else (
                "maximum per request under residual-window auto sizing"
                if args.auto_draft_tpot_ms is not None
                else "fixed per request"
            )
        ),
        "auto_draft_tpot_ms": args.auto_draft_tpot_ms,
        "auto_draft_utilization": args.auto_draft_utilization,
        "seed_tokens_from_prefill_side": args.seed_tokens,
        "cache_source": "pinned CPU memory; no SSD",
        "transfer": "real nonblocking H2D on a dedicated CUDA stream",
        "transport_gbps": args.transport_gbps,
        "transport_pacing": (
            (
                "one continuous serial wire stream with cumulative per-tranche "
                "arrival deadlines followed by real H2D"
                if args.eager_serial_wire
                else "stage-triggered serial wire-arrival deadlines followed by real H2D"
            )
            if args.transport_gbps is not None
            else "native pinned-memory H2D"
        ),
        "eager_serial_wire": args.eager_serial_wire,
        "draft_cache_mode": args.draft_cache_mode,
        "draft_kv_fraction": (
            args.stage_fractions[0] if args.draft_cache_mode == "s1" else None
        ),
        "reuse_final_verify": args.reuse_final_verify,
        "verification_mode": args.verification_mode,
        "draft_stage_policy": args.draft_stage_policy,
        "paired_fixed_control": args.paired_fixed_control,
        "paired_measurement": args.paired_measurement,
        "skip_serial": args.skip_serial,
        "require_exclusive_gpu": args.require_exclusive_gpu,
        "initial_gpu_used_bytes": args.initial_gpu_used_bytes,
        "initial_gpu_total_bytes": args.initial_gpu_total_bytes,
        "latency_measurement_order": (
            "alternate full-target-first and pipeline-first by dataset index; "
            "both paths warmed; extra serial chain omitted"
            if args.paired_measurement
            else "legacy target-first measurement with serial-chain control"
        ),
        "commitment": (
            "no token is exposed before the single immutable full-KV verifier"
            if args.verification_mode in {"final_only", "progressive_tentative"}
            else "finite-window commitment under progressive verifiers"
        ),
        "static_global_rope_positions": True,
        "representation_refresh": (
            "generated identities replayed only against fixed compact S1"
            if args.draft_cache_mode == "s1"
            else (
                "arrival-polled exact prompt KV grafted before a persistent "
                "approximate token tail before individual proposal steps; "
                "no model replay or intermediate verification"
                if args.draft_cache_mode == "continuous"
                else (
                    "new exact prompt KV grafted before persistent approximate "
                    "token tail; no model replay between arrival stages"
                    if args.draft_cache_mode == "grafted"
                    else "all generated token identities against each progressive stage"
                )
            )
        ),
        "gpu_buffer_allocation_in_response_timer": False,
        "producer_d2h_in_decode_response_timer": False,
        "decoding": "greedy",
        "fixed_token_horizon": args.fixed_token_horizon,
        "quality_eos_handling": (
            "truncate timed tokens at first EOS before task scoring"
            if args.fixed_token_horizon
            else "stop timed decoding at EOS"
        ),
    }
    with output.open("w", encoding="utf-8") as stream:
        for local_index, dataset_index in enumerate(shard_indices):
            result = evaluate_example(
                model, tokenizer, rows[dataset_index], dataset_index, args, device
            )
            result["protocol"] = protocol
            stream.write(json.dumps(result, ensure_ascii=False) + "\n")
            stream.flush()
            print(
                json.dumps(
                    {
                        "event": "example_complete",
                        "local": local_index + 1,
                        "count": len(shard_indices),
                        "dataset_index": dataset_index,
                    }
                ),
                flush=True,
            )
    print(json.dumps({"event": "complete", "output": str(output)}))


if __name__ == "__main__":
    main()
