"""Paired latency benchmark for the real LMCache 1P1D baseline.

The benchmark replays immutable SparseCache packet prompts against an LMCache
prefill/decode proxy and a monolithic vLLM endpoint.  It deliberately records
both endpoints for every prompt so that latency deltas are paired by request.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import time
from collections.abc import Iterable
from pathlib import Path

import requests
from transformers import AutoTokenizer

from experiments.lossless_pd.reference_packets import load_packet, read_index


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def bootstrap_mean_ci(
    values: Iterable[float], *, samples: int = 10_000, seed: int = 20260909
) -> list[float]:
    values = list(values)
    if not values:
        raise ValueError("cannot bootstrap an empty sample")
    if len(values) == 1:
        return [values[0], values[0]]
    generator = random.Random(seed)
    means = sorted(
        statistics.fmean(generator.choices(values, k=len(values)))
        for _ in range(samples)
    )
    return [means[int(0.025 * samples)], means[int(0.975 * samples)]]


def summarize(rows: list[dict]) -> dict:
    metrics = {}
    for label in ("pd", "monolithic"):
        metrics[label] = {
            key: statistics.fmean(row[label][key] for row in rows)
            for key in ("headers_ms", "ttft_ms", "total_ms")
        }
    paired = {}
    for key in ("ttft_ms", "total_ms"):
        deltas = [row["pd"][key] - row["monolithic"][key] for row in rows]
        paired[f"pd_minus_monolithic_{key}"] = {
            "mean_ms": statistics.fmean(deltas),
            "bootstrap_95ci_ms": bootstrap_mean_ci(deltas),
            "pd_faster": sum(delta < 0 for delta in deltas),
            "same": sum(delta == 0 for delta in deltas),
            "pd_slower": sum(delta > 0 for delta in deltas),
        }
    return {
        "requests": len(rows),
        **metrics,
        **paired,
        "outputs_equal": sum(row["outputs_equal"] for row in rows),
    }


def stream_completion(
    session: requests.Session,
    endpoint: str,
    payload: dict,
    timeout: float,
) -> dict:
    started = time.perf_counter()
    response = session.post(endpoint, json=payload, stream=True, timeout=timeout)
    headers = time.perf_counter()
    response.raise_for_status()
    first = None
    chunks = 0
    pieces = []
    for line in response.iter_lines(decode_unicode=True):
        if not line or not line.startswith("data: "):
            continue
        body = line[6:]
        if body == "[DONE]":
            continue
        event = json.loads(body)
        if "error" in event:
            raise RuntimeError(f"endpoint stream failed ({endpoint}): {event['error']}")
        if "choices" not in event:
            raise RuntimeError(
                f"endpoint stream returned an unexpected event ({endpoint}): {event}"
            )
        text = event["choices"][0].get("text", "")
        if text and first is None:
            first = time.perf_counter()
        pieces.append(text)
        chunks += 1
    finished = time.perf_counter()
    if first is None:
        raise RuntimeError(f"endpoint returned no generated text: {endpoint}")
    return {
        "headers_ms": (headers - started) * 1000,
        "ttft_ms": (first - started) * 1000,
        "total_ms": (finished - started) * 1000,
        "chunks": chunks,
        "text": "".join(pieces),
    }


def select_entries(entries: list[dict], count: int) -> list[dict]:
    """Select deterministic length-stratified requests from a packet index."""

    if count <= 0:
        raise ValueError("count must be positive")
    ordered = sorted(entries, key=lambda item: (item["prompt_tokens"], item["index"]))
    if count >= len(ordered):
        return ordered
    if count == 1:
        return [ordered[len(ordered) // 2]]
    positions = [
        round(index * (len(ordered) - 1) / (count - 1)) for index in range(count)
    ]
    return [ordered[position] for position in positions]


def run(args: argparse.Namespace) -> dict:
    packet_root = Path(args.packets).resolve()
    index = read_index(packet_root)
    entries = select_entries(index["entries"], args.num_requests)
    tokenizer = (
        None if args.packet_token_ids else AutoTokenizer.from_pretrained(args.tokenizer)
    )
    session = requests.Session()
    session.trust_env = False

    rows = []
    for ordinal, entry in enumerate(entries):
        packet = load_packet(packet_root, entry)
        ids = packet["prompt_ids"][0, : args.max_input_tokens].tolist()
        if args.packet_token_ids:
            prompt = ids
            retokenized = len(ids)
        else:
            assert tokenizer is not None
            prompt = tokenizer.decode(ids, skip_special_tokens=False)
            retokenized = len(tokenizer.encode(prompt, add_special_tokens=False))
        payload = {
            "model": args.model,
            "prompt": prompt,
            "max_tokens": args.max_new_tokens,
            "temperature": 0,
            "stream": True,
        }
        order = ("pd", "monolithic") if ordinal % 2 == 0 else ("monolithic", "pd")
        outputs = {}
        for label in order:
            outputs[label] = stream_completion(
                session,
                args.pd_url if label == "pd" else args.monolithic_url,
                payload,
                args.timeout,
            )
        row = {
            "ordinal": ordinal,
            "packet_index": entry["index"],
            "record_id": entry["record_id"],
            "packet_prompt_tokens": entry["prompt_tokens"],
            "input_tokens_requested": len(ids),
            "input_tokens_retokenized": retokenized,
            "order": list(order),
            "pd": outputs["pd"],
            "monolithic": outputs["monolithic"],
            "outputs_equal": outputs["pd"]["text"] == outputs["monolithic"]["text"],
        }
        rows.append(row)
        print(
            json.dumps(
                {
                    "event": "request",
                    "completed": ordinal + 1,
                    "total": len(entries),
                    "tokens": retokenized,
                    "pd_ttft_ms": row["pd"]["ttft_ms"],
                    "monolithic_ttft_ms": row["monolithic"]["ttft_ms"],
                    "outputs_equal": row["outputs_equal"],
                }
            ),
            flush=True,
        )

    result = {
        "contract": (
            "paired greedy requests; official LMCache full-KV 1P1D versus "
            "monolithic vLLM; alternating endpoint order"
        ),
        "packets": str(packet_root),
        "model": args.model,
        "tokenizer": str(Path(args.tokenizer).resolve()),
        "pd_url": args.pd_url,
        "monolithic_url": args.monolithic_url,
        "max_input_tokens": args.max_input_tokens,
        "max_new_tokens": args.max_new_tokens,
        "packet_token_ids": args.packet_token_ids,
        "summary": summarize(rows),
        "rows": rows,
    }
    write_json(Path(args.output), result)
    print(json.dumps(result["summary"], indent=2), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--packets", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--tokenizer", default="/data/models/qwen/Qwen3-8B")
    parser.add_argument("--model", default="/data/models/qwen/Qwen3-8B")
    parser.add_argument("--pd-url", default="http://127.0.0.1:19100/v1/completions")
    parser.add_argument(
        "--monolithic-url", default="http://127.0.0.1:17350/v1/completions"
    )
    parser.add_argument("--num-requests", type=int, default=16)
    parser.add_argument("--max-input-tokens", type=int, default=7800)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument(
        "--packet-token-ids",
        action="store_true",
        help="submit immutable packet token IDs directly without text round-trip",
    )
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args()
    if min(args.num_requests, args.max_input_tokens, args.max_new_tokens) <= 0:
        parser.error("request counts and token limits must be positive")
    run(args)


if __name__ == "__main__":
    main()
