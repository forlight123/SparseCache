"""Build and validate frozen query-priority sidecars for live LMCache probes.

This is an experimental bridge, not an online scorer.  It converts the
immutable packet's 64-token attention-mass scores into a complete permutation
of 256-token LMCache chunks, keyed by the exact token-ID prompt digest.  The
producer fails closed if a configured sidecar does not contain the request.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from experiments.lossless_pd.reference_packets import load_packet, read_index


def _token_list(token_ids: Any) -> list[int]:
    values = token_ids.tolist() if hasattr(token_ids, "tolist") else list(token_ids)
    if len(values) == 1 and isinstance(values[0], list):
        values = values[0]
    if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
        raise TypeError("prompt token IDs must be a flat integer sequence")
    return values


def token_digest(token_ids: Any) -> str:
    """Match the immutable packet corpus' JSON-token SHA256 convention."""

    return hashlib.sha256(json.dumps(_token_list(token_ids)).encode()).hexdigest()


def chunk_priority_from_page_scores(
    page_scores: Iterable[float],
    *,
    prompt_tokens: int,
    page_tokens: int,
    chunk_tokens: int,
) -> tuple[int, ...]:
    """Aggregate attention mass and return an edge-protected chunk ranking."""

    if min(prompt_tokens, page_tokens, chunk_tokens) <= 0:
        raise ValueError("token counts must be positive")
    if chunk_tokens % page_tokens:
        raise ValueError("LMCache chunks must contain a whole number of score pages")
    scores = [float(value) for value in page_scores]
    needed_pages = math.ceil(prompt_tokens / page_tokens)
    if len(scores) < needed_pages:
        raise ValueError("priority score vector is shorter than the prompt")
    scores = scores[:needed_pages]
    chunk_count = math.ceil(prompt_tokens / chunk_tokens)
    pages_per_chunk = chunk_tokens // page_tokens
    chunk_scores = [
        sum(scores[index * pages_per_chunk : (index + 1) * pages_per_chunk])
        for index in range(chunk_count)
    ]
    protected = tuple(dict.fromkeys((0, chunk_count - 1)))
    rest = [index for index in range(chunk_count) if index not in protected]
    rest.sort(key=lambda index: (chunk_scores[index], -index), reverse=True)
    return (*protected, *rest)


def load_priority_sidecar(path: Path) -> dict[str, tuple[int, ...]]:
    rows: dict[str, tuple[int, ...]] = {}
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            digest = row.get("prompt_digest")
            priority = row.get("priority_chunks")
            if (
                not isinstance(digest, str)
                or len(digest) != 64
                or not isinstance(priority, list)
                or not priority
                or any(
                    isinstance(value, bool) or not isinstance(value, int)
                    for value in priority
                )
                or sorted(priority) != list(range(len(priority)))
            ):
                raise ValueError(f"{path}:{line_number} has an invalid priority row")
            normalized = tuple(priority)
            if digest in rows:
                if rows[digest] != normalized:
                    raise ValueError(
                        f"{path}:{line_number} gives one prompt conflicting priorities"
                    )
                continue
            rows[digest] = normalized
    if not rows:
        raise ValueError("priority sidecar is empty")
    return rows


def build_sidecar(
    packet_root: Path,
    *,
    max_input_tokens: int,
    chunk_tokens: int,
) -> list[dict[str, Any]]:
    index = read_index(packet_root)
    grouped: dict[str, dict[str, Any]] = {}
    for entry in index["entries"]:
        packet = load_packet(packet_root, entry)
        prompt = packet["prompt_ids"][0, :max_input_tokens]
        prompt_tokens = int(prompt.numel())
        page_tokens = int(packet["page_size"])
        digest = token_digest(prompt)
        needed_pages = math.ceil(prompt_tokens / page_tokens)
        scores = [float(value) for value in packet["priority_scores"][:needed_pages]]
        group = grouped.get(digest)
        if group is None:
            grouped[digest] = {
                "record_ids": [str(packet["record_id"])],
                "prompt_tokens": prompt_tokens,
                "page_tokens": page_tokens,
                "score_sums": scores,
                "score_count": 1,
            }
            continue
        if (
            group["prompt_tokens"] != prompt_tokens
            or group["page_tokens"] != page_tokens
            or len(group["score_sums"]) != len(scores)
        ):
            raise ValueError("identical prompt digest has inconsistent score geometry")
        group["record_ids"].append(str(packet["record_id"]))
        group["score_sums"] = [
            left + right
            for left, right in zip(group["score_sums"], scores, strict=True)
        ]
        group["score_count"] += 1

    rows = []
    for digest, group in grouped.items():
        averaged = [value / group["score_count"] for value in group["score_sums"]]
        priority = chunk_priority_from_page_scores(
            averaged,
            prompt_tokens=group["prompt_tokens"],
            page_tokens=group["page_tokens"],
            chunk_tokens=chunk_tokens,
        )
        rows.append(
            {
                "record_ids": group["record_ids"],
                "prompt_digest": digest,
                "prompt_tokens": group["prompt_tokens"],
                "page_tokens": group["page_tokens"],
                "chunk_tokens": chunk_tokens,
                "priority_chunks": list(priority),
                "score_replicates": group["score_count"],
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--packets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-input-tokens", type=int, default=7680)
    parser.add_argument("--chunk-tokens", type=int, default=256)
    args = parser.parse_args()
    if min(args.max_input_tokens, args.chunk_tokens) <= 0:
        parser.error("token limits must be positive")
    rows = build_sidecar(
        args.packets.resolve(),
        max_input_tokens=args.max_input_tokens,
        chunk_tokens=args.chunk_tokens,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as destination:
        for row in rows:
            destination.write(json.dumps(row, separators=(",", ":")) + "\n")
    temporary.replace(args.output)
    print(json.dumps({"output": str(args.output.resolve()), "requests": len(rows)}))


if __name__ == "__main__":
    main()
