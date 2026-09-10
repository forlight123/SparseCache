"""Compare wait-full and layer-ready exact-verifier executions.

Both inputs must come from ``experiments.lossless_pd.integrated_probe`` with
identical requests and repetitions.  The comparison is paired at the raw-run
level and keeps the no-draft and sparse-draft effects separate.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
from collections.abc import Callable
from pathlib import Path
from typing import Any


def read(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def paired_bootstrap(
    values: list[float], seed: int = 20260910, repeats: int = 10_000
) -> list[float]:
    if not values:
        raise ValueError("cannot bootstrap an empty comparison")
    rng = random.Random(seed)
    means = sorted(
        statistics.mean(rng.choices(values, k=len(values))) for _ in range(repeats)
    )
    return [means[int(0.025 * repeats)], means[int(0.975 * repeats)]]


def keyed_rows(document: dict[str, Any]) -> dict[tuple[str, int], dict[str, Any]]:
    rows = document.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("results document has no raw rows")
    result = {}
    for row in rows:
        key = (str(row["record_id"]), int(row["repeat"]))
        if key in result:
            raise ValueError(f"duplicate paired row: {key}")
        result[key] = row
    return result


def clustered(
    keys: list[tuple[str, int]], value: Callable[[tuple[str, int]], float]
) -> list[float]:
    grouped: dict[str, list[float]] = {}
    for key in keys:
        grouped.setdefault(key[0], []).append(value(key))
    return [statistics.mean(values) for values in grouped.values()]


def describe(values: list[float]) -> dict[str, Any]:
    return {
        "mean_ms": statistics.mean(values),
        "request_bootstrap_ci95_ms": paired_bootstrap(values),
        "positive_requests": sum(value > 0 for value in values),
        "requests": len(values),
    }


def compare(
    layer_ready_document: dict[str, Any], full_ready_document: dict[str, Any]
) -> dict[str, Any]:
    layer_summary = layer_ready_document["summary"]
    full_summary = full_ready_document["summary"]
    if layer_summary.get("target_start_mode") != "layer_ready":
        raise ValueError("first input is not a layer-ready run")
    if full_summary.get("target_start_mode") != "full_ready":
        raise ValueError("second input is not a wait-full run")

    immutable_fields = (
        "gbps_decimal_bits_per_second",
        "fraction",
        "order",
        "proposal_tokens",
        "verifier_semantics",
    )
    mismatched = [
        field
        for field in immutable_fields
        if layer_summary.get(field) != full_summary.get(field)
    ]
    if mismatched:
        raise ValueError(f"ablation protocol differs: {', '.join(mismatched)}")

    layer_rows = keyed_rows(layer_ready_document)
    full_rows = keyed_rows(full_ready_document)
    if layer_rows.keys() != full_rows.keys():
        raise ValueError("wait-full and layer-ready request/repeat matrices differ")
    keys = sorted(layer_rows)

    for key in keys:
        left = layer_rows[key]
        right = full_rows[key]
        for field in ("context", "actual_fraction", "proposal_tokens"):
            if left.get(field) != right.get(field):
                raise ValueError(f"row {key} differs in {field}")
        for arm in ("baseline", "speculative"):
            if left[arm]["bytes_sent"] != right[arm]["bytes_sent"]:
                raise ValueError(f"row {key} {arm} transfers different bytes")
        if left["progress_tokens"] != right["progress_tokens"]:
            raise ValueError(f"row {key} has different exact progress")

    baseline_hidden = clustered(
        keys,
        lambda key: (
            full_rows[key]["baseline"]["same_progress_ms"]
            - layer_rows[key]["baseline"]["same_progress_ms"]
        ),
    )
    speculative_hidden = clustered(
        keys,
        lambda key: (
            full_rows[key]["speculative"]["same_progress_ms"]
            - layer_rows[key]["speculative"]["same_progress_ms"]
        ),
    )
    interaction = [
        speculative - baseline
        for speculative, baseline in zip(speculative_hidden, baseline_hidden)
    ]
    layer_speedups = clustered(
        keys,
        lambda key: (
            full_rows[key]["speculative"]["same_progress_ms"]
            / layer_rows[key]["speculative"]["same_progress_ms"]
        ),
    )

    mismatch_ids = sorted(
        {
            key[0]
            for key in keys
            if not layer_rows[key]["committed_output_equal"]
            or not full_rows[key]["committed_output_equal"]
        }
    )
    return {
        "contract": (
            "paired 2x2 attribution: wait-full/layer-ready exact Target start x "
            "no-draft/sparse-draft; identical bytes, requests, proposals, and progress"
        ),
        "pairing_scope": (
            "same request/repeat identifiers; the caller must separately establish "
            "whether target-start modes were order-balanced in one process"
        ),
        "requests": len({key[0] for key in keys}),
        "paired_runs": len(keys),
        "protocol": {field: layer_summary.get(field) for field in immutable_fields},
        "layer_ready_hidden_ms_no_draft": describe(baseline_hidden),
        "layer_ready_hidden_ms_sparse_draft": describe(speculative_hidden),
        "difference_in_differences_ms": describe(interaction),
        "mean_sparse_path_speedup_layer_ready_over_wait_full": statistics.mean(
            layer_speedups
        ),
        "strict_output_mismatch_requests": len(mismatch_ids),
        "strict_output_mismatch_record_ids": mismatch_ids,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer-ready", required=True)
    parser.add_argument("--full-ready", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = compare(read(args.layer_ready), read(args.full_ready))
    output = Path(args.output)
    if output.exists():
        raise ValueError(f"refusing to overwrite {output}")
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
