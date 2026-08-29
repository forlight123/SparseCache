"""Fail-closed audit for controlled exact-length RULER P/D packs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

DEFAULT_LENGTHS = (16384, 32768, 65536, 130816)
DEFAULT_PLACEMENTS = (
    "original",
    "evidence_first",
    "evidence_uniform",
    "evidence_last",
    "adversarial_split",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--lengths", default=",".join(str(value) for value in DEFAULT_LENGTHS)
    )
    parser.add_argument("--placements", default=",".join(DEFAULT_PLACEMENTS))
    parser.add_argument("--requests-per-cell", type=int, default=200)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _identity(metadata: dict[str, Any]) -> tuple[int, str, str]:
    answers = json.dumps(metadata["answers"], ensure_ascii=False, sort_keys=True)
    return int(metadata["source_index"]), str(metadata["id"]), answers


def audit_pack(
    path: Path,
    *,
    expected_length: int,
    expected_placement: str,
    expected_requests: int,
) -> tuple[dict[str, Any], list[tuple[int, str, str]]]:
    manifest_path = path / "manifest.json"
    request_path = path / "requests.jsonl"
    metadata_path = path / "metadata.jsonl"
    for required in (manifest_path, request_path, metadata_path):
        if not required.is_file():
            raise FileNotFoundError(required)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_manifest = {
        "dataset": "ruler_qa2",
        "placement": expected_placement,
        "requests": expected_requests,
        "prompt_tokens": expected_length,
        "chunk_tokens": 256,
        "output_tokens": 32,
    }
    for key, expected in expected_manifest.items():
        if manifest.get(key) != expected:
            raise ValueError(
                f"manifest {key} mismatch in {path}: "
                f"{manifest.get(key)!r} != {expected!r}"
            )
    request_hash = _sha256(request_path)
    metadata_hash = _sha256(metadata_path)
    if request_hash != manifest.get("requests_sha256"):
        raise ValueError(f"request hash mismatch in {path}")
    if metadata_hash != manifest.get("metadata_sha256"):
        raise ValueError(f"metadata hash mismatch in {path}")

    identities: list[tuple[int, str, str]] = []
    fit_modes: dict[str, int] = {}
    rows = 0
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
            prompt = request.get("prompt")
            if not isinstance(prompt, list) or len(prompt) != expected_length:
                raise ValueError(f"non-exact prompt length at row {rows} in {path}")
            if len(prompt) % 256:
                raise ValueError(f"unaligned prompt at row {rows} in {path}")
            request_protocol = {
                "temperature": 0,
                "max_tokens": 32,
                "ignore_eos": True,
                "stream": True,
                "return_token_ids": True,
            }
            for key, expected in request_protocol.items():
                if request.get(key) != expected:
                    raise ValueError(
                        f"request {key} mismatch at row {rows} in {path}"
                    )
            metadata_protocol = {
                "request_index": rows,
                "dataset": "ruler_qa2",
                "format": "ruler",
                "placement": expected_placement,
                "prompt_tokens": expected_length,
                "supporting_documents": 2,
                "output_tokens": 32,
                "fixed_output_horizon": True,
            }
            for key, expected in metadata_protocol.items():
                if metadata.get(key) != expected:
                    raise ValueError(
                        f"metadata {key} mismatch at row {rows} in {path}"
                    )
            if not metadata.get("answers"):
                raise ValueError(f"missing gold answer at row {rows} in {path}")
            identities.append(_identity(metadata))
            fit_mode = str(metadata["document_fit_mode"])
            fit_modes[fit_mode] = fit_modes.get(fit_mode, 0) + 1
            rows += 1
    if rows != expected_requests:
        raise ValueError(f"row count mismatch in {path}: {rows} != {expected_requests}")
    if fit_modes != manifest.get("document_fit_modes"):
        raise ValueError(f"document fit-mode counts mismatch in {path}")
    return (
        {
            "prompt_tokens": expected_length,
            "placement": expected_placement,
            "requests": rows,
            "total_prompt_tokens": rows * expected_length,
            "document_fit_modes": fit_modes,
            "requests_sha256": request_hash,
            "metadata_sha256": metadata_hash,
        },
        identities,
    )


def audit_matrix(
    root: Path,
    *,
    lengths: tuple[int, ...],
    placements: tuple[str, ...],
    requests_per_cell: int,
) -> dict[str, Any]:
    if not lengths or len(lengths) != len(set(lengths)):
        raise ValueError("lengths must be a non-empty unique list")
    if any(length <= 0 or length % 256 for length in lengths):
        raise ValueError("lengths must be positive multiples of 256")
    if not placements or len(placements) != len(set(placements)):
        raise ValueError("placements must be a non-empty unique list")
    if requests_per_cell <= 0:
        raise ValueError("requests-per-cell must be positive")

    cells = []
    for length in lengths:
        direct_length_root = root / str(length)
        if direct_length_root.is_dir():
            length_root = direct_length_root
        else:
            candidates = []
            for candidate in root.iterdir():
                manifest_path = candidate / placements[0] / "manifest.json"
                if not manifest_path.is_file():
                    continue
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                if manifest.get("prompt_tokens") == length:
                    candidates.append(candidate)
            if len(candidates) != 1:
                raise FileNotFoundError(
                    f"could not uniquely resolve a source directory for "
                    f"prompt length {length}: {candidates}"
                )
            length_root = candidates[0]
        reference_identities = None
        for placement in placements:
            cell, identities = audit_pack(
                length_root / placement,
                expected_length=length,
                expected_placement=placement,
                expected_requests=requests_per_cell,
            )
            cell["source_length_directory"] = length_root.name
            if reference_identities is None:
                reference_identities = identities
            elif identities != reference_identities:
                raise ValueError(
                    f"sample identity/order mismatch for {length}/{placement}"
                )
            cells.append(cell)
    return {
        "schema_version": 1,
        "status": "passed",
        "root": str(root),
        "lengths": list(lengths),
        "placements": list(placements),
        "requests_per_cell": requests_per_cell,
        "cell_count": len(cells),
        "total_requests": sum(cell["requests"] for cell in cells),
        "total_prompt_tokens": sum(cell["total_prompt_tokens"] for cell in cells),
        "cells": cells,
    }


def main() -> None:
    args = parse_args()
    lengths = tuple(int(item.strip()) for item in args.lengths.split(",") if item.strip())
    placements = tuple(
        item.strip() for item in args.placements.split(",") if item.strip()
    )
    result = audit_matrix(
        args.root,
        lengths=lengths,
        placements=placements,
        requests_per_cell=args.requests_per_cell,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
