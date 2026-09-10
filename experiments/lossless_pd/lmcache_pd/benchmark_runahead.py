"""Paired exact P-runahead benchmark against the one-token P/D handoff."""

from __future__ import annotations

import argparse
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


def summarize(rows: list[dict]) -> dict:
    deltas = [row["runahead"]["total_ms"] - row["baseline"]["total_ms"] for row in rows]
    ttft = [row["runahead"]["ttft_ms"] - row["baseline"]["ttft_ms"] for row in rows]
    return {
        "requests": len(rows),
        "outputs_equal": sum(row["outputs_equal"] for row in rows),
        "runahead_minus_baseline_total_ms": {
            "mean": statistics.fmean(deltas),
            "bootstrap_95ci": bootstrap_mean_ci(deltas),
            "runahead_faster": sum(value < 0 for value in deltas),
        },
        "runahead_minus_baseline_ttft_ms": {
            "mean": statistics.fmean(ttft),
            "bootstrap_95ci": bootstrap_mean_ci(ttft),
            "runahead_faster": sum(value < 0 for value in ttft),
        },
    }


def run(args: argparse.Namespace) -> dict:
    packet_root = Path(args.packets).resolve()
    selected = select_entries(
        read_index(packet_root)["entries"], args.num_requests + args.warmup_requests
    )
    warmup_entries = selected[: args.warmup_requests]
    entries = selected[args.warmup_requests :]
    session = requests.Session()
    session.trust_env = False
    for entry in warmup_entries:
        packet = load_packet(packet_root, entry)
        prompt = packet["prompt_ids"][0, : args.max_input_tokens].tolist()
        for budget in (1, args.runahead_tokens):
            stream_completion(
                session,
                args.url,
                {
                    "model": args.model,
                    "prompt": prompt,
                    "max_tokens": args.max_new_tokens,
                    "temperature": 0,
                    "stream": True,
                    "return_token_ids": True,
                    "sparsecache_p_runahead_tokens": budget,
                },
                args.timeout,
            )
    rows = []
    for ordinal, entry in enumerate(entries):
        packet = load_packet(packet_root, entry)
        prompt = packet["prompt_ids"][0, : args.max_input_tokens].tolist()
        order = (
            ("baseline", 1),
            ("runahead", args.runahead_tokens),
        )
        if ordinal % 2:
            order = tuple(reversed(order))
        measurements = {}
        for label, budget in order:
            measurements[label] = stream_completion(
                session,
                args.url,
                {
                    "model": args.model,
                    "prompt": prompt,
                    "max_tokens": args.max_new_tokens,
                    "temperature": 0,
                    "stream": True,
                    "return_token_ids": True,
                    "sparsecache_p_runahead_tokens": budget,
                },
                args.timeout,
            )
            if not measurements[label]["token_ids"]:
                raise RuntimeError(f"{label} response omitted exact output token IDs")
        rows.append(
            {
                "ordinal": ordinal,
                "record_id": entry["record_id"],
                "prompt_tokens": len(prompt),
                "order": [label for label, _ in order],
                **measurements,
                "text_equal": measurements["baseline"]["text"]
                == measurements["runahead"]["text"],
                "token_ids_equal": measurements["baseline"]["token_ids"]
                == measurements["runahead"]["token_ids"],
                "outputs_equal": measurements["baseline"]["token_ids"]
                == measurements["runahead"]["token_ids"],
            }
        )
        print(
            f"{ordinal + 1}/{len(entries)} tokens={len(prompt)} "
            f"equal={rows[-1]['outputs_equal']}",
            flush=True,
        )
    result = {
        "contract": (
            "same live LMCache 1P1D service; exact P Target generates one token "
            "or a bounded runahead block; warmups are discarded; paired request "
            "order alternates; equality compares integer output IDs"
        ),
        "runahead_tokens": args.runahead_tokens,
        "warmup_requests": args.warmup_requests,
        "max_new_tokens": args.max_new_tokens,
        "summary": summarize(rows),
        "rows": rows,
    }
    write_json(Path(args.output), result)
    print(result["summary"], flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--packets", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--url", default="http://127.0.0.1:19100/v1/completions")
    parser.add_argument("--model", default="/data/models/qwen/Qwen3-8B")
    parser.add_argument("--num-requests", type=int, default=8)
    parser.add_argument("--warmup-requests", type=int, default=2)
    parser.add_argument("--max-input-tokens", type=int, default=7800)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--runahead-tokens", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args()
    if min(
        args.num_requests,
        args.max_input_tokens,
        args.max_new_tokens,
        args.runahead_tokens,
    ) <= 0:
        parser.error("counts and token budgets must be positive")
    if args.warmup_requests < 0:
        parser.error("warmup request count cannot be negative")
    run(args)


if __name__ == "__main__":
    main()
