"""Paired live benchmark for the SparseCache-PD 1P1D proxy."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import statistics
import time
import uuid
from pathlib import Path
from typing import Any

import httpx

VALID_ARMS = ("baseline", "fixed_s1", "continuous")


def _parse_arms(raw: str) -> tuple[str, ...]:
    arms = tuple(item.strip() for item in raw.split(",") if item.strip())
    if len(arms) < 2 or len(arms) != len(set(arms)):
        raise argparse.ArgumentTypeError("arms must contain at least two unique values")
    invalid = [arm for arm in arms if arm not in VALID_ARMS]
    if invalid:
        raise argparse.ArgumentTypeError(f"unknown benchmark arms: {invalid}")
    return arms


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument(
        "--endpoint",
        choices=("/v1/completions", "/v1/chat/completions"),
        default="/v1/completions",
    )
    parser.add_argument("--requests-jsonl", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--num-requests", type=int, default=100)
    parser.add_argument("--request-offset", type=int, default=0)
    parser.add_argument("--warmup-requests", type=int, default=4)
    parser.add_argument("--fixed-output-tokens", type=int)
    parser.add_argument("--priority-jsonl", type=Path)
    parser.add_argument("--arms", type=_parse_arms, default=("baseline", "continuous"))
    return parser.parse_args()


def _read_priority_rows(path: Path) -> dict[int, list[int]]:
    priorities = {}
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            row = json.loads(line)
            index = row.get("request_index")
            priority = row.get("priority_chunks")
            if (
                not isinstance(index, int)
                or isinstance(index, bool)
                or index < 0
                or index in priorities
            ):
                raise ValueError(f"{path}:{line_number} has invalid request_index")
            if (
                not isinstance(priority, list)
                or not priority
                or any(
                    not isinstance(item, int) or isinstance(item, bool) or item < 0
                    for item in priority
                )
                or len(priority) != len(set(priority))
            ):
                raise ValueError(f"{path}:{line_number} has invalid priority_chunks")
            if sorted(priority) != list(range(len(priority))):
                raise ValueError(
                    f"{path}:{line_number} priority is not a full permutation"
                )
            priorities[index] = priority
    if not priorities:
        raise ValueError(f"no priority rows in {path}")
    return priorities


def _read_requests(
    path: Path,
    limit: int,
    offset: int = 0,
    priorities: dict[int, list[int]] | None = None,
) -> list[dict[str, Any]]:
    requests = []
    with path.open(encoding="utf-8") as source:
        for source_index, line in enumerate(source):
            if source_index < offset:
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise TypeError("each JSONL row must be an OpenAI request object")
            if priorities is not None:
                try:
                    payload["sparsecache_priority_chunks"] = priorities[source_index]
                except KeyError as error:
                    raise ValueError(
                        f"priority sidecar has no row {source_index}"
                    ) from error
            payload["stream"] = True
            payload["return_token_ids"] = True
            payload["temperature"] = 0
            requests.append(payload)
            if len(requests) == limit:
                break
    if len(requests) < limit:
        raise ValueError(f"requested {limit} rows but {path} contains {len(requests)}")
    return requests


def _execution_plan(
    requests: list[dict[str, Any]], warmup_requests: int, source_offset: int = 0
) -> list[tuple[int | None, int, dict[str, Any]]]:
    if not requests:
        raise ValueError("execution plan requires at least one request")
    warmups = [
        (
            None,
            source_offset + index % len(requests),
            requests[index % len(requests)],
        )
        for index in range(warmup_requests)
    ]
    measured = [
        (index, source_offset + index, request)
        for index, request in enumerate(requests)
    ]
    return warmups + measured


def _tokens_and_text(payload: dict[str, Any]) -> tuple[list[int], str]:
    choices = payload.get("choices") or []
    if not choices:
        return [], ""
    choice = choices[0]
    token_ids = choice.get("token_ids") or []
    delta = choice.get("delta") or {}
    text = choice.get("text") or delta.get("content") or ""
    return [int(token_id) for token_id in token_ids], str(text)


async def _run_one(
    client: httpx.AsyncClient,
    endpoint: str,
    payload: dict[str, Any],
    arm: str,
    prefill_group: str,
) -> dict[str, Any]:
    request = dict(payload)
    request["sparsecache_prefill_group"] = prefill_group
    if arm == "baseline":
        request["sparsecache_mode"] = "baseline"
    else:
        request["sparsecache_mode"] = "progressive"
        request["sparsecache_visibility_mode"] = arm
    started = time.perf_counter()
    header_at: float | None = None
    first_at: float | None = None
    token_ids: list[int] = []
    token_arrival_ms: list[float] = []
    text_parts: list[str] = []
    response_headers: dict[str, str] = {}
    async with client.stream("POST", endpoint, json=request) as response:
        header_at = time.perf_counter()
        response.raise_for_status()
        response_headers = dict(response.headers)
        async for line in response.aiter_lines():
            if not line.startswith("data: "):
                continue
            body = line[6:]
            if body == "[DONE]":
                continue
            chunk = json.loads(body)
            new_tokens, new_text = _tokens_and_text(chunk)
            arrived_at = time.perf_counter()
            if first_at is None and (new_tokens or new_text):
                first_at = arrived_at
            token_ids.extend(new_tokens)
            token_arrival_ms.extend([(arrived_at - started) * 1000] * len(new_tokens))
            text_parts.append(new_text)
    completed = time.perf_counter()
    if header_at is None or first_at is None:
        raise RuntimeError("stream completed without a visible output token")
    if len(token_arrival_ms) != len(token_ids):
        raise RuntimeError("token arrival telemetry does not cover returned token IDs")
    raw_seed = response_headers.get("x-sparsecache-seed-token", "")
    raw_reused = response_headers.get("x-sparsecache-prefill-reused", "")
    if not raw_seed.isdecimal() or raw_reused not in {"true", "false"}:
        raise RuntimeError("proxy did not return paired-prefill telemetry headers")
    return {
        "arm": arm,
        "runtime_mode": response_headers.get("x-sparsecache-mode", ""),
        "header_ms": (header_at - started) * 1000,
        "ttft_ms": (first_at - started) * 1000,
        "decode_ttft_ms": (first_at - header_at) * 1000,
        "completion_ms": (completed - started) * 1000,
        "decode_completion_ms": (completed - header_at) * 1000,
        "token_ids": token_ids,
        "token_arrival_ms": token_arrival_ms,
        "text": "".join(text_parts),
        "server_timing": response_headers.get("server-timing", ""),
        "proxy_request_id": response_headers.get("x-sparsecache-request-id", ""),
        "prefill_group": response_headers.get("x-sparsecache-prefill-group", ""),
        "prefill_reused": raw_reused == "true",
        "seed_token_id": int(raw_seed),
    }


def _percentile(values: list[float], quantile: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    index = min(len(ordered) - 1, math.ceil(quantile * len(ordered)) - 1)
    return ordered[index]


def _comparison(
    complete_pairs: list[dict[str, dict[str, Any]]], left: str, right: str
) -> dict[str, Any]:
    completion_deltas = [
        pair[left]["decode_completion_ms"] - pair[right]["decode_completion_ms"]
        for pair in complete_pairs
    ]
    ttft_deltas = [
        pair[left]["decode_ttft_ms"] - pair[right]["decode_ttft_ms"]
        for pair in complete_pairs
    ]
    equality = [
        pair[left]["token_ids"] == pair[right]["token_ids"] for pair in complete_pairs
    ]

    def distribution(values: list[float]) -> dict[str, float]:
        return {
            "mean": statistics.fmean(values) if values else math.nan,
            "p05": _percentile(values, 0.05),
            "p50": _percentile(values, 0.50),
            "p95": _percentile(values, 0.95),
        }

    return {
        "left": left,
        "right": right,
        "gain_definition": "left latency minus right latency; positive favors right",
        "token_equality_rate": statistics.fmean(equality) if equality else math.nan,
        "right_faster_completion": sum(delta > 0 for delta in completion_deltas),
        "decode_completion_gain_ms": distribution(completion_deltas),
        "decode_ttft_gain_ms": distribution(ttft_deltas),
    }


def _summary(records: list[dict[str, Any]], arms: tuple[str, ...]) -> dict[str, Any]:
    pairs: dict[int, dict[str, dict[str, Any]]] = {}
    for record in records:
        pairs.setdefault(record["request_index"], {})[record["arm"]] = record
    complete_pairs = [
        pair for pair in pairs.values() if all(arm in pair for arm in arms)
    ]
    comparisons = {}
    if "baseline" in arms:
        for arm in arms:
            if arm != "baseline":
                comparisons[f"baseline_vs_{arm}"] = _comparison(
                    complete_pairs, "baseline", arm
                )
    if "fixed_s1" in arms and "continuous" in arms:
        comparisons["fixed_s1_vs_continuous"] = _comparison(
            complete_pairs, "fixed_s1", "continuous"
        )
    return {
        "arms": arms,
        "pairs": len(complete_pairs),
        "comparisons": comparisons,
    }


async def run(args: argparse.Namespace) -> dict[str, Any]:
    requests = _read_requests(
        args.requests_jsonl,
        args.num_requests,
        offset=args.request_offset,
        priorities=(
            _read_priority_rows(args.priority_jsonl)
            if args.priority_jsonl is not None
            else None
        ),
    )
    if args.fixed_output_tokens is not None:
        if args.fixed_output_tokens <= 0:
            raise ValueError("fixed_output_tokens must be positive")
        for request in requests:
            request["max_tokens"] = args.fixed_output_tokens
            request["ignore_eos"] = True
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    execution = _execution_plan(
        requests, args.warmup_requests, source_offset=args.request_offset
    )
    run_nonce = uuid.uuid4().hex
    async with httpx.AsyncClient(
        base_url=args.url, timeout=None, trust_env=False
    ) as client:
        for execution_index, (request_index, source_index, request) in enumerate(
            execution
        ):
            shift = execution_index % len(args.arms)
            order = args.arms[shift:] + args.arms[:shift]
            prefill_group = f"{run_nonce}:{execution_index}:{source_index}"
            for arm in order:
                result = await _run_one(
                    client,
                    args.endpoint,
                    request,
                    arm,
                    prefill_group,
                )
                result["request_index"] = request_index
                result["source_request_index"] = source_index
                result["warmup"] = request_index is None
                if not result["warmup"]:
                    records.append(result)
                    with args.output_jsonl.open("a", encoding="utf-8") as output:
                        output.write(json.dumps(result, ensure_ascii=False) + "\n")
    summary = _summary(records, args.arms)
    summary["fixed_output_tokens"] = args.fixed_output_tokens
    summary["warmup_requests"] = args.warmup_requests
    summary["request_offset"] = args.request_offset
    summary["priority_jsonl"] = (
        str(args.priority_jsonl) if args.priority_jsonl is not None else None
    )
    summary["source_request_index_definition"] = (
        "zero-based row in requests-jsonl; warmups replay measured rows"
    )
    summary_path = args.output_jsonl.with_suffix(".summary.json")
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    args = parse_args()
    if args.num_requests < 1 or min(args.warmup_requests, args.request_offset) < 0:
        raise ValueError("request counts must be non-negative with num_requests > 0")
    if args.output_jsonl.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_jsonl}")
    print(json.dumps(asyncio.run(run(args)), indent=2))


if __name__ == "__main__":
    main()
