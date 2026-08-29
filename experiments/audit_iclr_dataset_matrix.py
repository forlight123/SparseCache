# SPDX-License-Identifier: Apache-2.0
"""Audit local coverage and protocol metadata for the ICLR dataset matrix."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import statistics

from longbench_metrics import score_prediction
from longbench_prompts import (
    LONG_BENCH_MAX_NEW_TOKENS,
    LONG_BENCH_PROMPTS,
    LONG_BENCH_V2_MAX_NEW_TOKENS,
)


def percentile(values, quantile):
    values = sorted(values)
    return values[round((len(values) - 1) * quantile)]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--matrix", default="configs/iclr_pd_experiment_matrix.json"
    )
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--longbench-v2", default="/data/datasets/LongBench-v2/data.json"
    )
    parser.add_argument("--ruler-root", default="datasets/ruler/normalized")
    return parser.parse_args()


def main():
    args = parse_args()
    matrix = json.loads(Path(args.matrix).read_text(encoding="utf-8"))
    datasets = []
    failures = []
    for specification in matrix["datasets"]:
        path = Path(specification["path"])
        if not path.exists():
            failures.append(f"missing dataset: {path}")
            continue
        with path.open(encoding="utf-8") as stream:
            rows = [json.loads(line) for line in stream if line.strip()]
        name = specification["name"]
        observed_names = {row.get("dataset") for row in rows}
        missing_fields = sorted(
            {
                field
                for row in rows
                for field in ("input", "context", "answers", "dataset")
                if field not in row
            }
        )
        ids = [row.get("_id", row.get("id")) for row in rows]
        template_ok = name in LONG_BENCH_PROMPTS
        output_limit = LONG_BENCH_MAX_NEW_TOKENS.get(name)
        metric_name, _, _ = score_prediction(
            name,
            "",
            list(rows[0].get("answers", [])),
            rows[0].get("all_classes"),
        )
        lengths = [int(row.get("length", 0)) for row in rows]
        row_failures = []
        if len(rows) != specification["available"]:
            row_failures.append(
                f"count {len(rows)} != manifest {specification['available']}"
            )
        if observed_names != {name}:
            row_failures.append(f"dataset field mismatch: {observed_names}")
        if missing_fields:
            row_failures.append(f"missing fields: {missing_fields}")
        if None in ids or len(ids) != len(set(ids)):
            row_failures.append("missing or duplicate request IDs")
        if not template_ok:
            row_failures.append("missing official prompt template")
        if output_limit != specification["official_max_new_tokens"]:
            row_failures.append(
                f"output limit {output_limit} != manifest "
                f"{specification['official_max_new_tokens']}"
            )
        if metric_name != specification["metric"]:
            row_failures.append(
                f"metric {metric_name} != manifest {specification['metric']}"
            )
        failures.extend(f"{name}: {item}" for item in row_failures)
        datasets.append(
            {
                "name": name,
                "path": str(path.resolve()),
                "rows": len(rows),
                "unique_ids": len(set(ids)),
                "length_field": {
                    "mean": statistics.fmean(lengths),
                    "p50": percentile(lengths, 0.50),
                    "p95": percentile(lengths, 0.95),
                    "max": max(lengths),
                },
                "prompt_template": "public LongBench exact string",
                "official_max_new_tokens": output_limit,
                "metric": metric_name,
                "passed": not row_failures,
                "failures": row_failures,
            }
        )

    longbench_v2 = Path(args.longbench_v2)
    longbench_v2_rows = None
    longbench_v2_audit = None
    if longbench_v2.exists():
        payload = json.loads(longbench_v2.read_text(encoding="utf-8"))
        longbench_v2_rows = len(payload)
        required = {
            "_id",
            "domain",
            "sub_domain",
            "difficulty",
            "length",
            "question",
            "choice_A",
            "choice_B",
            "choice_C",
            "choice_D",
            "answer",
            "context",
        }
        missing = sorted(
            {field for row in payload for field in required if field not in row}
        )
        longbench_v2_audit = {
            "unique_ids": len({row.get("_id") for row in payload}),
            "domains": dict(Counter(row["domain"] for row in payload)),
            "difficulty": dict(Counter(row["difficulty"] for row in payload)),
            "length_category": dict(Counter(row["length"] for row in payload)),
            "missing_fields": missing,
            "official_max_new_tokens": LONG_BENCH_V2_MAX_NEW_TOKENS,
            "metric": "choice_accuracy",
            "passed": (
                len(payload) == matrix["longbench_v2"]["available"]
                and not missing
                and len({row.get("_id") for row in payload}) == len(payload)
            ),
        }
        if not longbench_v2_audit["passed"]:
            failures.append("LongBench-v2 local data or schema audit failed")
    else:
        failures.append(f"missing LongBench-v2 dataset: {longbench_v2}")
    ruler_root = Path(args.ruler_root)
    ruler_cells = []
    for nominal, relative_path in matrix["controlled_context"]["paths"].items():
        path = Path(relative_path)
        if not path.is_absolute():
            path = Path(args.matrix).resolve().parents[1] / path
        if not path.exists():
            ruler_cells.append(
                {"nominal_length": nominal, "path": str(path), "ready": False}
            )
            continue
        with path.open(encoding="utf-8") as stream:
            rows = [json.loads(line) for line in stream if line.strip()]
        supporting = {
            sum(bool(document["supporting"]) for document in row["documents"])
            for row in rows
        }
        placements = set(rows[0]["placements"])
        ready = (
            len(rows) == matrix["controlled_context"]["requests_per_cell"]
            and supporting
            == {matrix["controlled_context"]["supporting_documents_per_request"]}
            and set(matrix["controlled_context"]["evidence_placement"])
            <= {
                name.removeprefix("evidence_")
                if name.startswith("evidence_")
                else name
                for name in placements
            }
        )
        ruler_cells.append(
            {
                "nominal_length": nominal,
                "path": str(path.resolve()),
                "rows": len(rows),
                "supporting_document_counts": sorted(supporting),
                "placements": sorted(placements),
                "ready": ready,
            }
        )
    controlled_ready = bool(ruler_cells) and all(
        item["ready"] for item in ruler_cells
    )
    models = []
    for specification in matrix["models"]:
        path = Path(specification["path"])
        config_path = path / "config.json"
        if not config_path.exists():
            failures.append(f"missing model config: {config_path}")
            models.append(
                {"name": specification["name"], "path": str(path), "ready": False}
            )
            continue
        config = json.loads(config_path.read_text(encoding="utf-8"))
        observed_max = config.get("max_position_embeddings")
        ready = observed_max == specification["max_position_embeddings"]
        if not ready:
            failures.append(
                f"{specification['name']} max positions {observed_max} != manifest "
                f"{specification['max_position_embeddings']}"
            )
        models.append(
            {
                "name": specification["name"],
                "path": str(path.resolve()),
                "model_type": config.get("model_type"),
                "hidden_size": config.get("hidden_size"),
                "layers": config.get("num_hidden_layers"),
                "max_position_embeddings": observed_max,
                "ready": ready,
            }
        )
    result = {
        "passed_longbench_matrix": not failures,
        "failures": failures,
        "datasets": datasets,
        "total_rows": sum(item["rows"] for item in datasets),
        "longbench_v2": {
            "path": str(longbench_v2),
            "available": longbench_v2.exists(),
            "rows": longbench_v2_rows,
            "audit": longbench_v2_audit,
        },
        "controlled_ruler": {
            "path": str(ruler_root),
            "ready": controlled_ready,
            "cells": ruler_cells,
            "status": (
                "available"
                if controlled_ready
                else "missing; controlled 16K-128K generation remains required"
            ),
        },
        "models": models,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
