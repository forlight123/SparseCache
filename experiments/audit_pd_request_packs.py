"""Fail-closed audit for materialized SparseCache-PD dataset packs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

DEFAULT_DATASETS = (
    "2wikimqa",
    "hotpotqa",
    "musique",
    "qasper",
    "multifieldqa_en",
    "narrativeqa",
    "qmsum",
    "gov_report",
    "multi_news",
    "passage_retrieval_en",
    "passage_count",
    "lcc",
    "repobench-p",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--datasets", default=",".join(DEFAULT_DATASETS))
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _audit_pack(path: Path, expected_dataset: str) -> dict:
    manifest_path = path / "manifest.json"
    request_path = path / "requests.jsonl"
    metadata_path = path / "metadata.jsonl"
    for required in (manifest_path, request_path, metadata_path):
        if not required.is_file():
            raise FileNotFoundError(required)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["dataset"] != expected_dataset:
        raise ValueError(f"dataset mismatch in {manifest_path}")
    if _sha256(request_path) != manifest["requests_sha256"]:
        raise ValueError(f"request hash mismatch in {path}")
    if _sha256(metadata_path) != manifest["metadata_sha256"]:
        raise ValueError(f"metadata hash mismatch in {path}")
    chunk_tokens = int(manifest["chunk_tokens"])
    rows = 0
    prompt_tokens = 0
    padded_requests = 0
    max_alignment_removed = 0
    with (
        request_path.open(encoding="utf-8") as requests_file,
        metadata_path.open(encoding="utf-8") as metadata_file,
    ):
        while True:
            request_line = requests_file.readline()
            metadata_line = metadata_file.readline()
            if not request_line and not metadata_line:
                break
            if not request_line or not metadata_line:
                raise ValueError(f"request/metadata row count mismatch in {path}")
            request = json.loads(request_line)
            metadata = json.loads(metadata_line)
            if metadata["request_index"] != rows:
                raise ValueError(f"non-contiguous request index in {path}")
            if metadata["dataset"] != expected_dataset:
                raise ValueError(f"row dataset mismatch in {path}")
            tokens = request.get("prompt")
            if not isinstance(tokens, list) or not tokens:
                raise TypeError(f"request prompt is not a non-empty token list in {path}")
            if len(tokens) != metadata["prompt_tokens"]:
                raise ValueError(f"prompt length mismatch in {path}")
            if len(tokens) % chunk_tokens:
                raise ValueError(f"unaligned prompt in {path}")
            if request.get("temperature") != 0:
                raise ValueError(f"non-greedy correctness request in {path}")
            if not metadata.get("answers"):
                raise ValueError(f"missing gold answer in {path}")
            rows += 1
            prompt_tokens += len(tokens)
            alignment_added = int(
                metadata.get("alignment_added_newline_tokens", 0)
            )
            padded_requests += alignment_added > 0
            max_alignment_removed = max(
                max_alignment_removed,
                int(metadata["alignment_removed_context_tokens"]),
            )
    if rows != manifest["requests"]:
        raise ValueError(f"manifest row count mismatch in {path}")
    if max_alignment_removed >= chunk_tokens:
        raise ValueError(f"alignment removed at least one full chunk in {path}")
    return {
        "dataset": expected_dataset,
        "requests": rows,
        "prompt_tokens": prompt_tokens,
        "mean_prompt_tokens": prompt_tokens / rows,
        "max_alignment_removed_context_tokens": max_alignment_removed,
        "newline_padded_requests": padded_requests,
        "request_sha256": manifest["requests_sha256"],
        "metadata_sha256": manifest["metadata_sha256"],
    }


def main() -> None:
    args = parse_args()
    datasets = tuple(item.strip() for item in args.datasets.split(",") if item.strip())
    if not datasets or len(datasets) != len(set(datasets)):
        raise ValueError("datasets must be a non-empty unique list")
    results = [_audit_pack(args.root / name, name) for name in datasets]
    output = {
        "schema_version": 1,
        "status": "passed",
        "root": str(args.root),
        "datasets": results,
        "dataset_count": len(results),
        "total_requests": sum(item["requests"] for item in results),
        "total_prompt_tokens": sum(item["prompt_tokens"] for item in results),
        "total_newline_padded_requests": sum(
            item["newline_padded_requests"] for item in results
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
