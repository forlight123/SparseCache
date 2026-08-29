# SPDX-License-Identifier: Apache-2.0
"""Probe vLLM self-speculation with a progressively sparse draft KV path.

The target always uses ordinary full FlashAttention. The draft is the same
model with the experimental PROGRESSIVE_KV backend, so the final vLLM
verifier remains immutable. In addition to the in-engine control, an
independent sparse draft process atomically hands its suffix to a full target
through a custom proposer. This probe measures kernel-path acceptance,
handoff correctness, and latency after warmup; it does not yet perform real
asynchronous P/D transport.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from itertools import pairwise
from pathlib import Path
from threading import Event, Thread
from time import perf_counter, perf_counter_ns
from typing import Any

SYSTEM_PREFIX = (
    "You are a careful question-answering assistant. Use the supplied passages "
    "to answer the question.\n\nPassages:\n"
)
QUESTION_PREFIX = "\n\nQuestion: "
ANSWER_SUFFIX = "\nReturn only the answer without explanation.\nAnswer:"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--max-prompt-tokens", type=int, default=8192)
    parser.add_argument("--max-new-tokens", type=int, default=9)
    parser.add_argument("--draft-tokens", type=int, default=8)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--visible-fractions", default="0.02,0.25,0.5,0.75,1.0")
    parser.add_argument(
        "--completion-trace-ms",
        help=(
            "optional cumulative completion times matched to visible-fractions; "
            "drives the progressive backend through connector-style events"
        ),
    )
    parser.add_argument(
        "--page-order",
        default="sequential",
        help="sequential, reverse, uniform, or comma-separated logical block IDs",
    )
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--cuda-visible-devices", default="0")
    parser.add_argument("--require-exclusive-gpu", action="store_true")
    parser.add_argument(
        "--worker-mode",
        choices=(
            "target",
            "spec_full",
            "spec_progressive",
            "draft_only",
            "external_verify",
        ),
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--prompt-file", help=argparse.SUPPRESS)
    parser.add_argument("--result-file", help=argparse.SUPPRESS)
    parser.add_argument("--stats-file", help=argparse.SUPPRESS)
    parser.add_argument("--completion-file", help=argparse.SUPPRESS)
    parser.add_argument("--proposal-file", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if (
        min(
            args.max_prompt_tokens,
            args.max_new_tokens,
            args.draft_tokens,
            args.block_size,
        )
        <= 0
    ):
        parser.error("token lengths and block size must be positive")
    if not 0.0 < args.gpu_memory_utilization <= 1.0:
        parser.error("gpu-memory-utilization must lie in (0, 1]")
    if args.max_new_tokens <= args.draft_tokens:
        parser.error("max-new-tokens must exceed draft-tokens for final verification")
    try:
        args.completion_trace = parse_completion_trace(
            args.visible_fractions, args.completion_trace_ms
        )
    except ValueError as error:
        parser.error(str(error))
    return args


def parse_completion_trace(
    raw_fractions: str, raw_times_ms: str | None
) -> tuple[tuple[float, float], ...]:
    """Validate a cumulative connector-completion trace."""
    if raw_times_ms is None:
        return ()
    try:
        fractions = tuple(float(value) for value in raw_fractions.split(","))
        times_ms = tuple(float(value) for value in raw_times_ms.split(","))
    except ValueError as error:
        raise ValueError("completion trace must contain numeric values") from error
    if len(fractions) != len(times_ms):
        raise ValueError("completion-trace-ms must have one entry per visible fraction")
    if not fractions or any(not 0.0 < value <= 1.0 for value in fractions):
        raise ValueError("visible fractions must lie in (0, 1]")
    if any(left > right for left, right in pairwise(fractions)):
        raise ValueError("visible fractions must be monotonic")
    if times_ms[0] != 0.0 or any(value < 0.0 for value in times_ms):
        raise ValueError("completion trace must start at 0 ms and be non-negative")
    if any(left > right for left, right in pairwise(times_ms)):
        raise ValueError("completion times must be monotonic")
    return tuple(zip(times_ms, fractions, strict=True))


def publish_completion(path: Path, *, epoch: str, fraction: float) -> None:
    """Atomically publish one connector-compatible completion snapshot."""
    payload = {
        "epoch": epoch,
        "completed_fraction": fraction,
        "completed_at_ns": perf_counter_ns(),
    }
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload), encoding="utf-8")
    os.replace(temporary, path)


def publish_external_draft(
    path: Path,
    *,
    epoch: str,
    seed_token_id: int,
    draft_token_ids: list[int],
) -> None:
    """Atomically publish one suffix for the external target proposer."""
    payload = {
        "state": "ready",
        "epoch": epoch,
        "seed_token_id": seed_token_id,
        "draft_token_ids": draft_token_ids,
        "created_at_ns": perf_counter_ns(),
    }
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload), encoding="utf-8")
    os.replace(temporary, path)


def start_completion_publisher(
    path: Path,
    trace: tuple[tuple[float, float], ...],
    first_draft_stats_path: Path,
) -> tuple[Event, Event, Thread]:
    """Start a publisher synchronized to the first real draft proposal."""
    start = Event()
    stop = Event()

    def publish() -> None:
        start.wait()
        # The timed generate includes a long prompt prefill. The first sparse
        # attention record is emitted immediately after the first proposal has
        # latched S1; subsequent transfer deadlines must be relative to that
        # point, not to the beginning of prefill.
        while not (
            first_draft_stats_path.exists()
            and first_draft_stats_path.stat().st_size > 0
        ):
            if stop.wait(0.001):
                return
        started = perf_counter()
        for deadline_ms, fraction in trace[1:]:
            remaining = deadline_ms / 1000 - (perf_counter() - started)
            if stop.wait(max(0.0, remaining)):
                return
            publish_completion(path, epoch="timed", fraction=fraction)

    thread = Thread(target=publish, name="progressive-kv-completion", daemon=True)
    thread.start()
    return start, stop, thread


def load_case(path: str, offset: int) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as source:
        for index, line in enumerate(source):
            if index == offset:
                row = json.loads(line)
                if not all(key in row for key in ("input", "context")):
                    raise ValueError("dataset row must contain input and context")
                return row
    raise ValueError(f"dataset has no row at offset {offset}")


def prepare_prompt(args: argparse.Namespace) -> dict[str, Any]:
    from transformers import AutoTokenizer

    row = load_case(args.dataset, args.offset)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    prefix = tokenizer.encode(SYSTEM_PREFIX, add_special_tokens=True)
    document = tokenizer.encode(str(row["context"]), add_special_tokens=False)
    suffix = tokenizer.encode(
        QUESTION_PREFIX + str(row["input"]) + ANSWER_SUFFIX,
        add_special_tokens=False,
    )
    available = args.max_prompt_tokens - len(prefix) - len(suffix)
    if available < args.block_size:
        raise ValueError("prompt budget leaves no complete document block")
    document = document[:available]
    prompt = prefix + document + suffix
    doc_start = len(prefix)
    doc_end = doc_start + len(document)
    candidate_pages = doc_end // args.block_size - math.ceil(
        doc_start / args.block_size
    )
    if candidate_pages <= 0:
        raise ValueError("document interval contains no complete vLLM block")
    return {
        "prompt_token_ids": prompt,
        "doc_start_token": doc_start,
        "doc_end_token": doc_end,
        "candidate_pages": candidate_pages,
        "question": row["input"],
        "answers": row.get("answers", []),
    }


def metric_snapshot(metrics: list[Any]) -> dict[str, Any]:
    names = {
        "vllm:spec_decode_num_drafts",
        "vllm:spec_decode_num_draft_tokens",
        "vllm:spec_decode_num_accepted_tokens",
        "vllm:spec_decode_num_accepted_tokens_per_pos",
    }
    snapshot: dict[str, Any] = {}
    for metric in metrics:
        if metric.name not in names:
            continue
        value = getattr(metric, "value", None)
        if value is None:
            value = list(getattr(metric, "values", []))
        snapshot[metric.name] = value
    return snapshot


def subtract_metrics(after: dict[str, Any], before: dict[str, Any]) -> dict[str, Any]:
    delta = {}
    for name, value in after.items():
        previous = before.get(name, [] if isinstance(value, list) else 0)
        if isinstance(value, list):
            delta[name] = [
                item - (previous[index] if index < len(previous) else 0)
                for index, item in enumerate(value)
            ]
        else:
            delta[name] = value - previous
    drafts = delta.get("vllm:spec_decode_num_drafts", 0)
    accepted = delta.get("vllm:spec_decode_num_accepted_tokens", 0)
    per_position = delta.get("vllm:spec_decode_num_accepted_tokens_per_pos", [])
    delta["mean_acceptance_length"] = 1 + accepted / drafts if drafts else 1.0
    delta["acceptance_rate_per_position"] = [
        count / drafts if drafts else 0.0 for count in per_position
    ]
    return delta


def summarize_visibility(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    by_step: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        by_step.setdefault(row["decode_step"], []).append(row)
    return {
        "records": len(rows),
        "layers": len({row["layer"] for row in rows}),
        "steps": [
            {
                "decode_step": step,
                "visible_fraction": group[0]["visible_fraction"],
                "visibility_source": group[0]["visibility_source"],
                "completion_epoch": group[0]["completion_epoch"],
                "visible_pages": group[0]["visible_pages"],
                "candidate_pages": group[0]["candidate_pages"],
                "sparse_seq_len": group[0]["sparse_seq_len"],
                "full_seq_len": group[0]["full_seq_len"],
            }
            for step, group in sorted(by_step.items())
        ],
    }


def assert_exclusive_gpu() -> tuple[int, int]:
    import torch

    free_bytes, total_bytes = torch.cuda.mem_get_info(0)
    used_bytes = total_bytes - free_bytes
    if used_bytes > 1 * 2**30:
        raise RuntimeError(
            f"selected GPU is not exclusive: {used_bytes / 2**30:.2f} GiB already used"
        )
    return used_bytes, total_bytes


def run_worker(args: argparse.Namespace) -> None:
    from vllm import LLM, SamplingParams

    if not args.prompt_file or not args.result_file or not args.stats_file:
        raise ValueError("worker mode requires prompt/result/stats files")
    initial_used = 0
    initial_total = 0
    if args.require_exclusive_gpu:
        initial_used, initial_total = assert_exclusive_gpu()
    prompt = json.loads(Path(args.prompt_file).read_text())
    completion_path = Path(args.completion_file) if args.completion_file else None
    if args.completion_trace:
        if args.worker_mode not in {"spec_progressive", "draft_only"} or (
            completion_path is None
        ):
            raise ValueError(
                "completion trace requires a progressive draft mode and file"
            )
        publish_completion(completion_path, epoch="warmup", fraction=1.0)
    speculative_config = None
    if args.worker_mode in {"spec_full", "spec_progressive"}:
        speculative_config = {
            "method": "draft_model",
            "model": args.model,
            "num_speculative_tokens": args.draft_tokens,
            "attention_backend": "PROGRESSIVE_KV",
            "enforce_eager": True,
        }
    elif args.worker_mode == "external_verify":
        if not args.proposal_file:
            raise ValueError("external verify mode requires a proposal file")
        speculative_config = {
            "method": "custom_class",
            "model": (
                "vllm.v1.spec_decode.progressive_external_proposer."
                "ProgressiveExternalDraftProposer"
            ),
            "num_speculative_tokens": args.draft_tokens,
        }
    attention_backend = (
        "PROGRESSIVE_KV" if args.worker_mode == "draft_only" else "FLASH_ATTN"
    )
    engine = LLM(
        model=args.model,
        tokenizer=args.model,
        dtype="bfloat16",
        max_model_len=args.max_prompt_tokens + args.max_new_tokens,
        max_num_seqs=1,
        block_size=args.block_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=True,
        enable_prefix_caching=False,
        disable_log_stats=False,
        attention_config={"backend": attention_backend},
        speculative_config=speculative_config,
    )
    output_tokens = (
        args.draft_tokens if args.worker_mode == "draft_only" else args.max_new_tokens
    )
    sampling = SamplingParams(
        temperature=0.0,
        max_tokens=output_tokens,
        ignore_eos=True,
    )
    request_token_ids = list(prompt["prompt_token_ids"])
    if args.worker_mode == "draft_only":
        if "seed_token_id" not in prompt:
            raise ValueError("draft-only mode requires a producer seed token")
        request_token_ids.append(prompt["seed_token_id"])
    request = {"prompt_token_ids": request_token_ids}
    engine.generate(request, sampling, use_tqdm=False)
    before = metric_snapshot(engine.get_metrics())
    stats_path = Path(args.stats_file)
    stats_path.unlink(missing_ok=True)
    publisher = None
    if args.completion_trace:
        assert completion_path is not None
        publish_completion(
            completion_path,
            epoch="timed",
            fraction=args.completion_trace[0][1],
        )
        publisher = start_completion_publisher(
            completion_path,
            args.completion_trace,
            stats_path,
        )
    started = perf_counter()
    if publisher is not None:
        publisher[0].set()
    output = engine.generate(request, sampling, use_tqdm=False)[0]
    elapsed_ms = (perf_counter() - started) * 1000
    if publisher is not None:
        publisher[1].set()
        publisher[2].join()
    after = metric_snapshot(engine.get_metrics())
    completion = output.outputs[0]
    result = {
        "mode": args.worker_mode,
        "token_ids": list(completion.token_ids),
        "text": completion.text,
        "warmed_generate_ms": elapsed_ms,
        "speculation": subtract_metrics(after, before),
        "initial_gpu_used_bytes": initial_used,
        "initial_gpu_total_bytes": initial_total,
    }
    Path(args.result_file).write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    )


def launch_worker(
    args: argparse.Namespace,
    mode: str,
    prompt_path: Path,
    result_path: Path,
    stats_path: Path,
) -> None:
    environment = os.environ.copy()
    source_root = Path(__file__).resolve().parents[1] / "vllm"
    old_pythonpath = environment.get("PYTHONPATH", "")
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": args.cuda_visible_devices,
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
            "PYTHONPATH": (
                str(source_root)
                if not old_pythonpath
                else str(source_root) + os.pathsep + old_pythonpath
            ),
        }
    )
    prompt = json.loads(prompt_path.read_text())
    if mode in {"spec_full", "spec_progressive", "draft_only"}:
        environment.update(
            {
                "VLLM_PROGRESSIVE_KV_DOC_START_TOKEN": str(prompt["doc_start_token"]),
                "VLLM_PROGRESSIVE_KV_DOC_END_TOKEN": str(prompt["doc_end_token"]),
                "VLLM_PROGRESSIVE_KV_VISIBLE_FRACTIONS": (
                    "1.0" if mode == "spec_full" else args.visible_fractions
                ),
                "VLLM_PROGRESSIVE_KV_PAGE_ORDER": args.page_order,
                "VLLM_PROGRESSIVE_KV_STATS_PATH": str(stats_path),
            }
        )
        if mode in {"spec_progressive", "draft_only"} and args.completion_trace:
            environment["VLLM_PROGRESSIVE_KV_COMPLETION_PATH"] = str(
                completion_path := stats_path.with_name(f"{mode}_completion.json")
            )
    if mode == "external_verify":
        proposal_path = stats_path.with_name("external_proposal.json")
        environment.update(
            {
                "VLLM_PROGRESSIVE_EXTERNAL_DRAFT_PATH": str(proposal_path),
                "VLLM_PROGRESSIVE_EXTERNAL_DRAFT_STATS_PATH": str(stats_path),
            }
        )
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--dataset",
        args.dataset,
        "--model",
        args.model,
        "--output-dir",
        args.output_dir,
        "--offset",
        str(args.offset),
        "--max-prompt-tokens",
        str(args.max_prompt_tokens),
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--draft-tokens",
        str(args.draft_tokens),
        "--block-size",
        str(args.block_size),
        "--visible-fractions",
        args.visible_fractions,
        "--page-order",
        args.page_order,
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--cuda-visible-devices",
        args.cuda_visible_devices,
        "--worker-mode",
        mode,
        "--prompt-file",
        str(prompt_path),
        "--result-file",
        str(result_path),
        "--stats-file",
        str(stats_path),
    ]
    if mode in {"spec_progressive", "draft_only"} and (
        args.completion_trace_ms is not None
    ):
        command.extend(["--completion-trace-ms", args.completion_trace_ms])
    if mode in {"spec_progressive", "draft_only"} and args.completion_trace:
        command.extend(["--completion-file", str(completion_path)])
    if mode == "external_verify":
        command.extend(["--proposal-file", str(proposal_path)])
    if args.require_exclusive_gpu:
        command.append("--require-exclusive-gpu")
    subprocess.run(command, env=environment, check=True)


def token_agreement(reference: list[int], candidate: list[int]) -> float:
    if not reference and not candidate:
        return 1.0
    length = max(len(reference), len(candidate))
    return (
        sum(left == right for left, right in zip(reference, candidate, strict=False))
        / length
    )


def run_parent(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prompt = prepare_prompt(args)
    prompt_path = output_dir / "prompt.json"
    prompt_path.write_text(json.dumps(prompt) + "\n")
    results = {}
    for mode in ("target", "spec_full", "spec_progressive"):
        result_path = output_dir / f"{mode}.json"
        stats_path = output_dir / f"{mode}_stats.jsonl"
        stats_path.unlink(missing_ok=True)
        launch_worker(args, mode, prompt_path, result_path, stats_path)
        result = json.loads(result_path.read_text())
        result["visibility"] = summarize_visibility(stats_path)
        results[mode] = result
    reference = results["target"]["token_ids"]
    if not reference:
        raise RuntimeError("target produced no exact seed token")
    prompt["seed_token_id"] = reference[0]
    prompt_path.write_text(json.dumps(prompt) + "\n")

    draft_result_path = output_dir / "draft_only.json"
    draft_stats_path = output_dir / "draft_only_stats.jsonl"
    draft_stats_path.unlink(missing_ok=True)
    launch_worker(
        args,
        "draft_only",
        prompt_path,
        draft_result_path,
        draft_stats_path,
    )
    draft_result = json.loads(draft_result_path.read_text())
    draft_result["visibility"] = summarize_visibility(draft_stats_path)
    results["draft_only"] = draft_result

    proposal_path = output_dir / "external_proposal.json"
    publish_external_draft(
        proposal_path,
        epoch="mechanism-probe",
        seed_token_id=reference[0],
        draft_token_ids=draft_result["token_ids"],
    )
    verify_result_path = output_dir / "external_verify.json"
    verify_stats_path = output_dir / "external_verify_stats.jsonl"
    verify_stats_path.unlink(missing_ok=True)
    launch_worker(
        args,
        "external_verify",
        prompt_path,
        verify_result_path,
        verify_stats_path,
    )
    verify_result = json.loads(verify_result_path.read_text())
    verify_result["handoff"] = (
        [json.loads(line) for line in verify_stats_path.read_text().splitlines()]
        if verify_stats_path.exists()
        else []
    )
    results["external_verify"] = verify_result
    summary = {
        "protocol": {
            "purpose": "kernel-path and verifier probe; no P/D transfer timing",
            "target_attention": "full FLASH_ATTN",
            "draft_attention": "PROGRESSIVE_KV compact block table",
            "draft_model": "same checkpoint as target; separate vLLM model instance",
            "verification": "standard immutable full-target vLLM verifier",
            "external_handoff": (
                "independent progressive draft -> atomic proposal snapshot -> "
                "nonblocking custom proposer -> full target verification"
            ),
            "warmup": "one identical request before measurement",
            "exclusive_gpu_required": args.require_exclusive_gpu,
            "completion_event_trace_ms": args.completion_trace_ms,
        },
        "model": args.model,
        "dataset": args.dataset,
        "offset": args.offset,
        "prompt_tokens": len(prompt["prompt_token_ids"]),
        "candidate_pages": prompt["candidate_pages"],
        "visible_fractions": args.visible_fractions,
        "page_order": args.page_order,
        "full_spec_token_agreement": token_agreement(
            reference, results["spec_full"]["token_ids"]
        ),
        "progressive_spec_token_agreement": token_agreement(
            reference, results["spec_progressive"]["token_ids"]
        ),
        "external_verify_token_agreement": token_agreement(
            reference, results["external_verify"]["token_ids"]
        ),
        "results": results,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


def main() -> None:
    args = parse_args()
    if args.worker_mode is None:
        run_parent(args)
    else:
        run_worker(args)


if __name__ == "__main__":
    main()
