# SPDX-License-Identifier: Apache-2.0
"""Build verified request-level chunk schedules for controlled RULER packs."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import random
import re
from typing import Any

try:
    from experiments.build_ruler_pd_pack import _fit_documents, _render_parts
except ModuleNotFoundError:  # Direct execution from experiments/.
    from build_ruler_pd_pack import _fit_documents, _render_parts


MODES = ("sequential", "uniform", "random", "bm25", "oracle")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pack-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--modes", default=",".join(MODES))
    parser.add_argument("--chunk-tokens", type=int, default=256)
    parser.add_argument("--protected-prefix-tokens", type=int, default=256)
    parser.add_argument("--protected-suffix-tokens", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20260827)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows or any(not isinstance(row, dict) for row in rows):
        raise ValueError(f"invalid or empty JSONL: {path}")
    return rows


def _uniform_order(total: int) -> list[int]:
    if total < 3:
        return list(range(total))
    order = [0, total - 1]
    intervals = [(0, total - 1)]
    while intervals:
        next_intervals = []
        for lower, upper in intervals:
            if upper - lower <= 1:
                continue
            midpoint = (lower + upper) // 2
            order.append(midpoint)
            next_intervals.extend(((lower, midpoint), (midpoint, upper)))
        intervals = next_intervals
    return order


def _terms(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def bm25_scores(question: str, documents: list[str]) -> list[float]:
    """Score documents with deterministic per-request BM25."""
    tokenized = [_terms(document) for document in documents]
    query = Counter(_terms(question))
    count = len(tokenized)
    average_length = sum(map(len, tokenized)) / max(count, 1)
    document_frequency = Counter(
        term for document in tokenized for term in set(document)
    )
    scores = []
    for document in tokenized:
        frequencies = Counter(document)
        score = 0.0
        for term, query_frequency in query.items():
            frequency = frequencies[term]
            if not frequency:
                continue
            inverse_frequency = math.log(
                1.0
                + (count - document_frequency[term] + 0.5)
                / (document_frequency[term] + 0.5)
            )
            normalization = frequency + 1.2 * (
                1.0 - 0.75 + 0.75 * len(document) / max(average_length, 1.0)
            )
            score += query_frequency * inverse_frequency * frequency * 2.2 / normalization
        scores.append(score)
    return scores


def _protected_first(
    order: list[int],
    total_chunks: int,
    prefix_chunks: int,
    suffix_chunks: int,
) -> list[int]:
    protected = set(range(min(prefix_chunks, total_chunks)))
    protected.update(range(max(0, total_chunks - suffix_chunks), total_chunks))
    return [item for item in order if item in protected] + [
        item for item in order if item not in protected
    ]


def _chunk_scores(
    spans: list[dict[str, Any]],
    document_scores: list[float],
    total_chunks: int,
    chunk_tokens: int,
) -> list[float]:
    scores = [-math.inf] * total_chunks
    for span, score in zip(spans, document_scores, strict=True):
        first = int(span["start_token"]) // chunk_tokens
        stop = math.ceil(int(span["end_token"]) / chunk_tokens)
        for chunk in range(first, min(stop, total_chunks)):
            scores[chunk] = max(scores[chunk], score)
    return scores


def _find_subsequence(values: list[int], pattern: list[int], start: int) -> int:
    if not pattern:
        raise ValueError("marker token sequence must be non-empty")
    stop = len(values) - len(pattern) + 1
    for index in range(start, stop):
        if values[index : index + len(pattern)] == pattern:
            return index
    raise ValueError(f"prompt does not contain expected marker after token {start}")


def document_spans_from_prompt_markers(
    tokenizer,
    prompt: list[int],
    source: dict[str, Any],
    placement: str,
    protected_suffix_tokens: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Recover exact displayed document starts from immutable prompt tokens."""
    order = source["placements"][placement]
    starts = []
    cursor = 0
    for displayed_index in range(1, len(order) + 1):
        marker = tokenizer.encode(
            f"Document {displayed_index}:\n", add_special_tokens=False
        )
        cursor = _find_subsequence(prompt, marker, cursor)
        starts.append(cursor)
        cursor += len(marker)
    last_end = len(prompt) - protected_suffix_tokens
    if not starts or last_end <= starts[0]:
        raise ValueError("protected suffix leaves no schedulable document tokens")
    spans = []
    documents = []
    for index, source_document_index in enumerate(order):
        document = source["documents"][source_document_index]
        raw_end = starts[index + 1] if index + 1 < len(starts) else len(prompt)
        start = min(starts[index], last_end)
        end = min(raw_end, last_end)
        spans.append(
            {
                "source_document_index": source_document_index,
                "start_token": start,
                "end_token": end,
                "supporting": bool(document["supporting"]),
            }
        )
        documents.append(document)
    return spans, documents


def priority_for_mode(
    mode: str,
    *,
    request_id: str,
    question: str,
    documents: list[dict[str, Any]],
    spans: list[dict[str, Any]],
    total_chunks: int,
    chunk_tokens: int,
    protected_prefix_chunks: int,
    protected_suffix_chunks: int,
    seed: int,
) -> list[int]:
    """Return one complete protected-first chunk permutation."""
    uniform = _uniform_order(total_chunks)
    if mode == "sequential":
        order = list(range(total_chunks))
    elif mode == "uniform":
        order = uniform
    elif mode == "random":
        order = list(range(total_chunks))
        digest = hashlib.sha256(f"{seed}:{request_id}".encode()).digest()
        random.Random(int.from_bytes(digest[:8], "big")).shuffle(order)
    elif mode in {"bm25", "oracle"}:
        if mode == "bm25":
            document_scores = bm25_scores(
                question, [str(document["text"]) for document in documents]
            )
        else:
            document_scores = [
                float(bool(document["supporting"])) for document in documents
            ]
        chunk_scores = _chunk_scores(
            spans, document_scores, total_chunks, chunk_tokens
        )
        uniform_rank = {chunk: rank for rank, chunk in enumerate(uniform)}
        order = sorted(
            range(total_chunks),
            key=lambda chunk: (-chunk_scores[chunk], uniform_rank[chunk]),
        )
    else:
        raise ValueError(f"unsupported priority mode: {mode}")
    result = _protected_first(
        order,
        total_chunks,
        protected_prefix_chunks,
        protected_suffix_chunks,
    )
    if sorted(result) != list(range(total_chunks)):
        raise AssertionError("chunk priority is not a complete permutation")
    return result


def main() -> None:
    args = parse_args()
    modes = tuple(item.strip() for item in args.modes.split(",") if item.strip())
    if not modes or len(modes) != len(set(modes)) or any(
        mode not in MODES for mode in modes
    ):
        raise ValueError(f"modes must be unique values from {MODES}")
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    if min(args.chunk_tokens, args.protected_prefix_tokens) <= 0:
        raise ValueError("chunk and protected-prefix token counts must be positive")
    if args.protected_suffix_tokens <= 0:
        raise ValueError("protected-suffix token count must be positive")

    manifest_path = args.pack_dir / "manifest.json"
    requests_path = args.pack_dir / "requests.jsonl"
    metadata_path = args.pack_dir / "metadata.jsonl"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("dataset") != "ruler_qa2":
        raise ValueError("chunk scheduling currently requires a RULER qa_2 pack")
    prompt_tokens = int(manifest["prompt_tokens"])
    if prompt_tokens % args.chunk_tokens:
        raise ValueError("prompt length must contain complete scheduling chunks")
    source_rows = _read_jsonl(Path(manifest["source"]))
    requests = _read_jsonl(requests_path)
    metadata = _read_jsonl(metadata_path)
    if len(requests) != len(metadata):
        raise ValueError("request and metadata row counts differ")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        manifest["tokenizer"], local_files_only=True
    )
    output_rows = {mode: [] for mode in modes}
    span_rows = []
    reconstruction_modes: Counter[str] = Counter()
    prefix_chunks = math.ceil(args.protected_prefix_tokens / args.chunk_tokens)
    suffix_chunks = math.ceil(args.protected_suffix_tokens / args.chunk_tokens)
    for request, item in zip(requests, metadata, strict=True):
        source_index = int(item["source_index"])
        source = source_rows[source_index]
        prefix, rendered_documents, suffix = _render_parts(
            tokenizer,
            source,
            manifest["placement"],
            disable_thinking=manifest.get("thinking_mode") == "disabled",
        )
        fitted, _, _ = _fit_documents(
            rendered_documents, prompt_tokens - len(prefix) - len(suffix)
        )
        rebuilt = prefix + [token for document in fitted for token in document] + suffix
        if rebuilt == request.get("prompt"):
            reconstruction_mode = "exact_builder_replay"
            observed_prefix_tokens = len(prefix)
            observed_suffix_tokens = len(suffix)
            cursor = len(prefix)
            spans = []
            displayed_documents = []
            for document, token_ids in zip(rendered_documents, fitted, strict=True):
                start = cursor
                cursor += len(token_ids)
                source_document = source["documents"][
                    document["source_document_index"]
                ]
                spans.append(
                    {
                        "source_document_index": document[
                            "source_document_index"
                        ],
                        "start_token": start,
                        "end_token": cursor,
                        "supporting": bool(document["supporting"]),
                    }
                )
                displayed_documents.append(source_document)
        else:
            # Older audited Llama packs predate the current tokenizer chat
            # template (which later gained date headers). Recover coordinates
            # from the immutable prompt itself instead of silently rebuilding
            # a different token sequence.
            reconstruction_mode = "exact_token_marker_scan"
            spans, displayed_documents = document_spans_from_prompt_markers(
                tokenizer,
                request["prompt"],
                source,
                manifest["placement"],
                args.protected_suffix_tokens,
            )
            observed_prefix_tokens = int(spans[0]["start_token"])
            observed_suffix_tokens = len(request["prompt"]) - int(
                spans[-1]["end_token"]
            )
        reconstruction_modes[reconstruction_mode] += 1
        span_rows.append(
            {
                "request_index": item["request_index"],
                "id": item["id"],
                "prefix_tokens": observed_prefix_tokens,
                "suffix_tokens": observed_suffix_tokens,
                "coordinate_reconstruction": reconstruction_mode,
                "document_spans": spans,
            }
        )
        for mode in modes:
            output_rows[mode].append(
                {
                    "request_index": item["request_index"],
                    "id": item["id"],
                    "mode": mode,
                    "priority_chunks": priority_for_mode(
                        mode,
                        request_id=str(item["id"]),
                        question=str(source["question"]),
                        documents=displayed_documents,
                        spans=spans,
                        total_chunks=prompt_tokens // args.chunk_tokens,
                        chunk_tokens=args.chunk_tokens,
                        protected_prefix_chunks=prefix_chunks,
                        protected_suffix_chunks=suffix_chunks,
                        seed=args.seed,
                    ),
                }
            )

    args.output_dir.mkdir(parents=True)
    spans_path = args.output_dir / "document_spans.jsonl"
    with spans_path.open("w", encoding="utf-8") as output:
        for row in span_rows:
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
    priority_hashes = {}
    for mode, rows in output_rows.items():
        path = args.output_dir / f"priority_{mode}.jsonl"
        with path.open("w", encoding="utf-8") as output:
            for row in rows:
                output.write(json.dumps(row, ensure_ascii=False) + "\n")
        priority_hashes[mode] = _sha256(path)
    audit = {
        "schema_version": 1,
        "status": "passed",
        "pack_dir": str(args.pack_dir),
        "requests": len(requests),
        "prompt_tokens": prompt_tokens,
        "chunk_tokens": args.chunk_tokens,
        "protected_prefix_tokens": args.protected_prefix_tokens,
        "protected_suffix_tokens": args.protected_suffix_tokens,
        "modes": list(modes),
        "online_modes": [mode for mode in modes if mode != "oracle"],
        "analysis_upper_bound_modes": [mode for mode in modes if mode == "oracle"],
        "coordinate_reconstruction_modes": dict(reconstruction_modes),
        "requests_sha256": _sha256(requests_path),
        "metadata_sha256": _sha256(metadata_path),
        "document_spans_sha256": _sha256(spans_path),
        "priority_sha256": priority_hashes,
    }
    (args.output_dir / "audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
