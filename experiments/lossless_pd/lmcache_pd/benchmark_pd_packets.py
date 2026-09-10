"""Replay immutable packet prompts against one live P/D endpoint.

Unlike ``benchmark.py``, this runner does not consume a GPU for a simultaneous
monolithic control.  Exact output token IDs are compared with the immutable
Qwen Target trajectory stored in each packet, making it suitable for a
resource-accounted external-drafter sandwich.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import requests

from experiments.lossless_pd.lmcache_pd.benchmark import (
    bootstrap_mean_ci,
    select_entries,
    stream_completion,
    write_json,
)
from experiments.lossless_pd.reference_packets import load_packet, read_index


def expected_output_ids(packet: dict, max_new_tokens: int) -> list[int]:
    """Return the frozen Target trajectory for an API request of this length."""

    seed = int(packet["seed"].reshape(-1)[0].item())
    suffix = [int(value) for value in packet["reference"].tolist()]
    return [seed, *suffix[: max(0, max_new_tokens - 1)]]


def summarize(rows: list[dict]) -> dict:
    metrics = {
        key: statistics.fmean(row["pd"][key] for row in rows)
        for key in ("headers_ms", "ttft_ms", "total_ms")
    }
    return {
        "requests": len(rows),
        "pd": metrics,
        "pd_bootstrap_95ci_ms": {
            key: bootstrap_mean_ci(row["pd"][key] for row in rows)
            for key in ("ttft_ms", "total_ms")
        },
        "outputs_equal_reference": sum(row["outputs_equal"] for row in rows),
        "mismatch_ordinals": [
            row["ordinal"] for row in rows if not row["outputs_equal"]
        ],
    }


def run(args: argparse.Namespace) -> dict:
    packet_root = Path(args.packets).resolve()
    entries = select_entries(read_index(packet_root)["entries"], args.num_requests)
    session = requests.Session()
    session.trust_env = False
    rows = []
    output_path = Path(args.output)

    def result(status: str) -> dict:
        return {
            "status": status,
            "contract": (
                "single live LMCache P/D endpoint; packet token IDs submitted "
                "directly; greedy output IDs compared with an immutable full-KV "
                "Target trajectory"
            ),
            "packets": str(packet_root),
            "model": args.model,
            "pd_url": args.pd_url,
            "max_input_tokens": args.max_input_tokens,
            "max_new_tokens": args.max_new_tokens,
            "requested_requests": len(entries),
            "rows": rows,
        }

    for ordinal, entry in enumerate(entries):
        packet = load_packet(packet_root, entry)
        prompt_ids = [
            int(value)
            for value in packet["prompt_ids"][0, : args.max_input_tokens].tolist()
        ]
        expected = expected_output_ids(packet, args.max_new_tokens)
        payload = {
            "model": args.model,
            "prompt": prompt_ids,
            "max_tokens": args.max_new_tokens,
            "temperature": 0,
            "stream": True,
        }
        output = stream_completion(session, args.pd_url, payload, timeout=args.timeout)
        row = {
            "ordinal": ordinal,
            "packet_index": entry["index"],
            "record_id": entry["record_id"],
            "packet_prompt_tokens": entry["prompt_tokens"],
            "input_tokens_requested": len(prompt_ids),
            "reference_token_ids": expected,
            "pd": output,
            "outputs_equal": output["token_ids"] == expected,
        }
        rows.append(row)
        # Preserve completed requests if a later streamed response stalls.  A
        # running checkpoint is not a valid final benchmark and is marked as
        # such explicitly.
        write_json(output_path, result("running"))
        print(
            json.dumps(
                {
                    "event": "request",
                    "completed": ordinal + 1,
                    "total": len(entries),
                    "tokens": len(prompt_ids),
                    "pd_ttft_ms": output["ttft_ms"],
                    "pd_total_ms": output["total_ms"],
                    "outputs_equal": row["outputs_equal"],
                }
            ),
            flush=True,
        )

    completed = result("completed")
    completed["summary"] = summarize(rows)
    write_json(output_path, completed)
    print(json.dumps(completed["summary"], indent=2), flush=True)
    return completed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--packets", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="/data/models/qwen/Qwen3-8B")
    parser.add_argument("--pd-url", default="http://127.0.0.1:19100/v1/completions")
    parser.add_argument("--num-requests", type=int, default=64)
    parser.add_argument("--max-input-tokens", type=int, default=8192)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args()
    if min(args.num_requests, args.max_input_tokens, args.max_new_tokens) <= 0:
        parser.error("request counts and token limits must be positive")
    run(args)


if __name__ == "__main__":
    main()
