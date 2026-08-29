"""Paired live benchmark for request-scoped progressive KV schedules."""

from __future__ import annotations

import argparse
import asyncio
import json
import uuid
from pathlib import Path
from typing import Any

import httpx

try:
    from experiments.benchmark_progressive_pd_live import (
        _execution_plan,
        _parse_arms,
        _read_priority_rows,
        _read_requests,
        _run_one,
    )
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from benchmark_progressive_pd_live import (
        _execution_plan,
        _parse_arms,
        _read_priority_rows,
        _read_requests,
        _run_one,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--endpoint", default="/v1/completions")
    parser.add_argument("--requests-jsonl", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument(
        "--priority-sidecar",
        action="append",
        required=True,
        metavar="MODE=PATH",
    )
    parser.add_argument("--num-requests", type=int, default=100)
    parser.add_argument("--request-offset", type=int, default=0)
    parser.add_argument("--warmup-requests", type=int, default=4)
    parser.add_argument("--fixed-output-tokens", type=int, default=32)
    parser.add_argument(
        "--arms", type=_parse_arms, default=("fixed_s1", "continuous")
    )
    return parser.parse_args()


def _parse_sidecars(values: list[str]) -> dict[str, Path]:
    sidecars = {}
    for value in values:
        mode, separator, raw_path = value.partition("=")
        mode = mode.strip()
        if not separator or not mode or not raw_path.strip() or mode in sidecars:
            raise ValueError(
                "priority sidecars must be unique non-empty MODE=PATH values"
            )
        sidecars[mode] = Path(raw_path.strip())
    if len(sidecars) < 2:
        raise ValueError("scheduling benchmark requires at least two modes")
    return sidecars


def _conditions(
    modes: tuple[str, ...], arms: tuple[str, ...], shift: int
) -> tuple[tuple[str, str], ...]:
    conditions = tuple((mode, arm) for mode in modes for arm in arms)
    offset = shift % len(conditions)
    return conditions[offset:] + conditions[:offset]


def _prefill_group(
    run_nonce: str, execution_index: int, source_index: int
) -> str:
    """Share one exact producer prefill across every scheduling condition."""
    return f"{run_nonce}:{execution_index}:{source_index}"


async def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.output_jsonl.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_jsonl}")
    if (
        args.num_requests <= 0
        or args.request_offset < 0
        or args.warmup_requests < 0
        or args.fixed_output_tokens <= 0
    ):
        raise ValueError("request counts and output horizon are invalid")
    sidecars = _parse_sidecars(args.priority_sidecar)
    priorities = {
        mode: _read_priority_rows(path) for mode, path in sidecars.items()
    }
    requests = _read_requests(
        args.requests_jsonl,
        args.num_requests,
        offset=args.request_offset,
    )
    for request in requests:
        request["max_tokens"] = args.fixed_output_tokens
        request["ignore_eos"] = True
    execution = _execution_plan(
        requests, args.warmup_requests, source_offset=args.request_offset
    )
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    records = []
    modes = tuple(sidecars)
    run_nonce = uuid.uuid4().hex
    async with httpx.AsyncClient(
        base_url=args.url, timeout=None, trust_env=False
    ) as client:
        for execution_index, (request_index, source_index, request) in enumerate(
            execution
        ):
            for mode, arm in _conditions(modes, args.arms, execution_index):
                try:
                    priority = priorities[mode][source_index]
                except KeyError as error:
                    raise ValueError(
                        f"{mode} sidecar has no source request {source_index}"
                    ) from error
                payload = dict(request)
                payload["sparsecache_priority_chunks"] = priority
                result = await _run_one(
                    client,
                    args.endpoint,
                    payload,
                    arm,
                    _prefill_group(run_nonce, execution_index, source_index),
                )
                result.update(
                    {
                        "schedule_mode": mode,
                        "request_index": request_index,
                        "source_request_index": source_index,
                        "warmup": request_index is None,
                    }
                )
                if request_index is not None:
                    records.append(result)
                    with args.output_jsonl.open("a", encoding="utf-8") as output:
                        output.write(json.dumps(result, ensure_ascii=False) + "\n")
    summary = {
        "schema_version": 1,
        "status": "raw live measurements; aggregation gates not yet applied",
        "modes": list(modes),
        "arms": list(args.arms),
        "requests": args.num_requests,
        "request_offset": args.request_offset,
        "warmup_requests": args.warmup_requests,
        "fixed_output_tokens": args.fixed_output_tokens,
        "measured_records": len(records),
        "expected_records": args.num_requests * len(modes) * len(args.arms),
        "priority_sidecars": {mode: str(path) for mode, path in sidecars.items()},
        "execution_order": "rotated request-by-request over schedule x arm",
        "prefill_pairing": (
            "one exact producer prefill per request, reused by every schedule x arm"
        ),
    }
    args.output_jsonl.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    args = parse_args()
    print(json.dumps(asyncio.run(run(args)), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
