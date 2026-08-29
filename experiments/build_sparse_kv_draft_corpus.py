# SPDX-License-Identifier: Apache-2.0
"""Build a deterministic, task-interleaved drafter-training request corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from time import time_ns
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"{path}:{line_number} is not an object")
            rows.append(payload)
    return rows


def length_stratified_indices(
    requests: list[dict[str, Any]],
    count: int,
) -> list[int]:
    if count <= 0 or count > len(requests):
        raise ValueError("sample count is outside the request range")
    ranked = sorted(
        range(len(requests)),
        key=lambda index: (len(requests[index]["prompt"]), index),
    )
    selected = []
    for sample_index in range(count):
        position = min(
            len(ranked) - 1,
            int((sample_index + 0.5) * len(ranked) / count),
        )
        selected.append(ranked[position])
    if len(selected) != len(set(selected)):
        raise RuntimeError("stratified selection produced duplicate indices")
    return selected


def build_corpus(
    pack_root: Path,
    datasets: tuple[str, ...],
    samples_per_dataset: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    if not datasets or len(datasets) != len(set(datasets)):
        raise ValueError("datasets must be non-empty and unique")
    selected_by_dataset = {}
    source_manifests = {}
    for dataset in datasets:
        directory = pack_root / dataset
        requests = load_jsonl(directory / "requests.jsonl")
        metadata = load_jsonl(directory / "metadata.jsonl")
        if len(requests) != len(metadata):
            raise ValueError(f"{dataset} request and metadata counts differ")
        manifest = json.loads(
            (directory / "manifest.json").read_text(encoding="utf-8")
        )
        if manifest.get("dataset") != dataset:
            raise ValueError(f"{dataset} manifest identity differs")
        indices = length_stratified_indices(requests, samples_per_dataset)
        selected_by_dataset[dataset] = [
            (index, requests[index], metadata[index]) for index in indices
        ]
        source_manifests[dataset] = {
            "manifest": str((directory / "manifest.json").resolve()),
            "requests_sha256": sha256_file(directory / "requests.jsonl"),
            "metadata_sha256": sha256_file(directory / "metadata.jsonl"),
            "selected_source_indices": indices,
        }

    corpus_requests = []
    corpus_metadata = []
    for rank in range(samples_per_dataset):
        for dataset in datasets:
            source_index, request, metadata = selected_by_dataset[dataset][rank]
            corpus_index = len(corpus_requests)
            request = dict(request)
            request["source_dataset"] = dataset
            request["source_index"] = source_index
            corpus_requests.append(request)
            corpus_metadata.append(
                {
                    **metadata,
                    "corpus_index": corpus_index,
                    "source_dataset": dataset,
                    "source_index": source_index,
                }
            )
    manifest = {
        "schema_version": 1,
        "format": "sparsecache.sparse-kv-draft-corpus.v1",
        "created_at_ns": time_ns(),
        "pack_root": str(pack_root.resolve()),
        "datasets": list(datasets),
        "samples_per_dataset": samples_per_dataset,
        "requests": len(corpus_requests),
        "interleave": "round_robin_by_length_stratum",
        "source_manifests": source_manifests,
    }
    return corpus_requests, corpus_metadata, manifest


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, separators=(",", ":")) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pack-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--datasets",
        default="qmsum,musique,qasper,multi_news,hotpotqa,narrativeqa",
    )
    parser.add_argument("--samples-per-dataset", type=int, default=100)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    datasets = tuple(item.strip() for item in args.datasets.split(","))
    requests, metadata, manifest = build_corpus(
        Path(args.pack_root),
        datasets,
        args.samples_per_dataset,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    requests_path = output_dir / "requests.jsonl"
    metadata_path = output_dir / "metadata.jsonl"
    write_jsonl(requests_path, requests)
    write_jsonl(metadata_path, metadata)
    manifest["requests_sha256"] = sha256_file(requests_path)
    manifest["metadata_sha256"] = sha256_file(metadata_path)
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "event": "complete",
                "output_dir": str(output_dir),
                "requests": len(requests),
                "datasets": list(datasets),
            }
        )
    )


if __name__ == "__main__":
    main()
