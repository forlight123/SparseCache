# SPDX-License-Identifier: Apache-2.0
"""Post-hoc task-quality audit for an immutable EAGLE3 timing screen.

The timing benchmark intentionally stores token IDs rather than decoded text.
This helper joins those IDs to the frozen request metadata, decodes every arm
with the target tokenizer, and applies the repository's public LongBench
metric mapping.  It does not rewrite any timing artifact.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random
import statistics
from typing import Any

try:
    from experiments.longbench_metrics import score_prediction
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from longbench_metrics import score_prediction


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--metadata-jsonl", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260827)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows or any(not isinstance(row, dict) for row in rows):
        raise ValueError(f"expected non-empty object JSONL: {path}")
    return rows


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        raise ValueError("percentile requires a non-empty vector")
    ordered = sorted(float(value) for value in values)
    position = fraction * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def bootstrap_mean_ci(
    values: list[float], *, samples: int, seed: int
) -> list[float]:
    if not values:
        raise ValueError("bootstrap requires a non-empty vector")
    if samples <= 0:
        raise ValueError("bootstrap_samples must be positive")
    generator = random.Random(seed)
    estimates = [
        statistics.fmean(generator.choices(values, k=len(values)))
        for _ in range(samples)
    ]
    return [percentile(estimates, 0.025), percentile(estimates, 0.975)]


def _by_offset(rows: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {}
    for row in rows:
        offset = row.get("request_offset")
        if not isinstance(offset, int) or isinstance(offset, bool) or offset in result:
            raise ValueError("result request offsets must be unique integers")
        tokens = row.get("token_ids")
        if not isinstance(tokens, list) or any(type(token) is not int for token in tokens):
            raise ValueError("every result row must contain integer token IDs")
        result[offset] = row
    return result


def _candidate_paths(result_dir: Path) -> list[tuple[int, Path]]:
    candidates = []
    for path in result_dir.glob("eagle3_k*.jsonl"):
        suffix = path.stem.removeprefix("eagle3_k")
        if not suffix.isdigit():
            raise ValueError(f"malformed EAGLE result filename: {path.name}")
        candidates.append((int(suffix), path))
    if not candidates:
        raise ValueError("no EAGLE result files found")
    return sorted(candidates)


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    dense = _by_offset(read_jsonl(args.result_dir / "dense.jsonl"))
    metadata_rows = read_jsonl(args.metadata_jsonl)
    metadata = {
        int(row["request_index"]): row
        for row in metadata_rows
        if isinstance(row.get("request_index"), int)
        and not isinstance(row.get("request_index"), bool)
    }
    if len(metadata) != len(metadata_rows):
        raise ValueError("metadata request indices must be unique integers")
    offsets = sorted(dense)
    if any(offset not in metadata for offset in offsets):
        raise ValueError("metadata does not cover every dense request offset")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    decoded_dense = {
        offset: tokenizer.decode(
            dense[offset]["token_ids"], skip_special_tokens=True
        ).strip()
        for offset in offsets
    }

    def task_score(offset: int, prediction: str) -> tuple[str, float, float]:
        row = metadata[offset]
        answers = row.get("answers")
        if not isinstance(answers, list) or any(
            not isinstance(answer, str) for answer in answers
        ):
            raise ValueError("metadata answers must be a list of strings")
        return score_prediction(
            str(row["dataset"]), prediction, answers, row.get("all_classes")
        )

    dense_scores = [task_score(offset, decoded_dense[offset]) for offset in offsets]
    metric_names = {item[0] for item in dense_scores}
    if len(metric_names) != 1:
        raise ValueError("one quality audit may contain only one task metric")
    metric_name = next(iter(metric_names))
    dense_values = [item[2] for item in dense_scores]

    candidates = []
    decoded_by_horizon: dict[int, dict[int, str]] = {}
    for horizon, path in _candidate_paths(args.result_dir):
        rows = _by_offset(read_jsonl(path))
        if sorted(rows) != offsets:
            raise ValueError(f"k={horizon} offsets do not match the dense arm")
        decoded = {
            offset: tokenizer.decode(
                rows[offset]["token_ids"], skip_special_tokens=True
            ).strip()
            for offset in offsets
        }
        decoded_by_horizon[horizon] = decoded
        values = [task_score(offset, decoded[offset])[2] for offset in offsets]
        deltas = [
            candidate - baseline
            for candidate, baseline in zip(values, dense_values, strict=True)
        ]
        candidates.append(
            {
                "speculative_horizon": horizon,
                "mean_task_score": statistics.fmean(values),
                "mean_paired_delta_vs_dense": statistics.fmean(deltas),
                "paired_delta_bootstrap_95pct_ci": bootstrap_mean_ci(
                    deltas,
                    samples=args.bootstrap_samples,
                    seed=args.seed + horizon,
                ),
                "decoded_text_exact_match_fraction": sum(
                    decoded[offset] == decoded_dense[offset] for offset in offsets
                )
                / len(offsets),
                "task_score_exact_match_fraction": sum(
                    candidate == baseline
                    for candidate, baseline in zip(values, dense_values, strict=True)
                )
                / len(offsets),
                "better_equal_worse_requests": {
                    "better": sum(delta > 0.0 for delta in deltas),
                    "equal": sum(delta == 0.0 for delta in deltas),
                    "worse": sum(delta < 0.0 for delta in deltas),
                },
                "per_request": [
                    {
                        "request_offset": offset,
                        "dense_score": baseline,
                        "candidate_score": candidate,
                        "delta": candidate - baseline,
                    }
                    for offset, baseline, candidate in zip(
                        offsets, dense_values, values, strict=True
                    )
                ],
            }
        )

    horizon_pairs = []
    horizons = sorted(decoded_by_horizon)
    for left_index, left in enumerate(horizons):
        for right in horizons[left_index + 1 :]:
            horizon_pairs.append(
                {
                    "left_horizon": left,
                    "right_horizon": right,
                    "decoded_text_exact_match_fraction": sum(
                        decoded_by_horizon[left][offset]
                        == decoded_by_horizon[right][offset]
                        for offset in offsets
                    )
                    / len(offsets),
                }
            )

    return {
        "schema_version": 1,
        "status": "valid post-hoc task-quality audit; timing artifacts unchanged",
        "scope": (
            "learned-draft screening only; not an end-to-end P/D or Lynx claim"
        ),
        "result_dir": str(args.result_dir),
        "metadata_jsonl": str(args.metadata_jsonl),
        "model": str(args.model),
        "samples": len(offsets),
        "request_offsets": offsets,
        "dataset": str(metadata[offsets[0]]["dataset"]),
        "metric": metric_name,
        "generation_cap_note": (
            "scores describe the benchmark's capped outputs, not an official "
            "full-output LongBench score"
        ),
        "ordinary_target": {
            "mean_task_score": statistics.fmean(dense_values),
        },
        "candidates": candidates,
        "cross_horizon_diagnostic": horizon_pairs,
    }


def main() -> None:
    args = parse_args()
    report = evaluate(args)
    destination = args.output or args.result_dir / "quality_audit.json"
    destination.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
