# SPDX-License-Identifier: Apache-2.0
"""Evaluate a trained sparse-KV drafter, including input-reliance ablations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter, time_ns

import torch

from experiments.train_sparse_kv_drafter import (
    evaluate,
    load_target_shared_weights,
    load_teacher_dataset,
    parse_fraction_list,
    parse_int_list,
)
from sparsecache.sparse_kv_draft import SparseKVDraftConfig, SparseKVDrafter
from sparsecache.store import model_fingerprint


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--teacher-manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--eval-indices")
    parser.add_argument("--eval-modulus", type=int)
    parser.add_argument("--eval-remainder", type=int, default=0)
    parser.add_argument("--visibility-fractions", default="0.05,0.10,0.20,1.0")
    parser.add_argument("--page-size", type=int, default=64)
    parser.add_argument(
        "--selection-mode",
        choices=("random", "priority"),
        default="random",
    )
    parser.add_argument("--selection-seed", type=int, default=20960827)
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--attn-implementation", default="sdpa")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    manifest_path = Path(args.teacher_manifest)
    raw_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    available = tuple(
        sorted(int(row["request_index"]) for row in raw_manifest["samples"])
    )
    if args.eval_modulus is not None:
        if args.eval_indices is not None:
            raise ValueError("modulus split cannot be mixed with explicit indices")
        if (
            args.eval_modulus <= 1
            or not 0 <= args.eval_remainder < args.eval_modulus
        ):
            raise ValueError("invalid evaluation modulus split")
        indices = tuple(
            index
            for index in available
            if index % args.eval_modulus == args.eval_remainder
        )
    else:
        if args.eval_indices is None:
            raise ValueError("evaluation indices or a modulus split are required")
        indices = parse_int_list(args.eval_indices, name="evaluation indices")
    fractions = parse_fraction_list(args.visibility_fractions)
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("the reference evaluation requires CUDA")
    torch.cuda.set_device(device)

    manifest, sample_map = load_teacher_dataset(
        manifest_path,
        requested_indices=set(indices),
    )
    missing = set(indices) - set(sample_map)
    if missing:
        raise ValueError(f"teacher samples are missing indices {sorted(missing)}")
    samples = [sample_map[index] for index in indices]
    checkpoint = torch.load(
        args.checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    if checkpoint.get("format") != "sparsecache.sparse-kv-drafter.v1":
        raise ValueError("unsupported drafter checkpoint")
    target_model = str(checkpoint["target_model"])
    fingerprints = {
        checkpoint["target_fingerprint"],
        manifest["target_fingerprint"],
        model_fingerprint(target_model),
    }
    if len(fingerprints) != 1:
        raise ValueError("target fingerprints differ")

    embedding, lm_head, _ = load_target_shared_weights(
        target_model,
        device=device,
        attn_implementation=args.attn_implementation,
    )
    model = SparseKVDrafter(
        SparseKVDraftConfig.from_dict(checkpoint["config"])
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.bind_target_weights(embedding, lm_head)

    started = perf_counter()
    cells = []
    for memory_mode, seed_mode in (
        ("exact", "exact"),
        ("zero", "exact"),
        ("shuffled", "exact"),
        ("exact", "zero"),
        ("zero", "zero"),
    ):
        metrics = evaluate(
            model,
            samples,
            fractions,
            page_size=args.page_size,
            selection_seed=args.selection_seed,
            device=device,
            memory_mode=memory_mode,
            seed_mode=seed_mode,
            selection_mode=args.selection_mode,
        )
        cells.extend(metrics)
        print(
            json.dumps(
                {
                    "event": "ablation",
                    "memory_mode": memory_mode,
                    "seed_mode": seed_mode,
                    "metrics": metrics,
                }
            ),
            flush=True,
        )
    result = {
        "schema_version": 1,
        "format": "sparsecache.sparse-kv-drafter-ablation.v1",
        "created_at_ns": time_ns(),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "teacher_manifest": str(manifest_path.resolve()),
        "eval_indices": list(indices),
        "elapsed_seconds": perf_counter() - started,
        "cells": cells,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "event": "complete",
                "output": str(output),
                "elapsed_seconds": result["elapsed_seconds"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
