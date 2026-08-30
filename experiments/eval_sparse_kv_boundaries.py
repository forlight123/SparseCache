# SPDX-License-Identifier: Apache-2.0
"""Evaluate a drafter at held-out post-verification boundary states."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
from time import perf_counter, time_ns
from typing import Any

import torch

from experiments.train_sparse_kv_drafter import (
    TeacherSample,
    load_target_shared_weights,
    load_teacher_dataset,
    parse_fraction_list,
    select_memory,
    training_window_inputs,
)
from sparsecache.sparse_kv_draft import (
    SparseKVDraftConfig,
    SparseKVDrafter,
    accepted_prefix_length,
)


def select_offsets(horizon: int, maximum: int) -> tuple[int, ...]:
    if horizon <= 1 or maximum <= 0:
        return ()
    candidates = list(range(1, horizon))
    if len(candidates) <= maximum:
        return tuple(candidates)
    return tuple(
        candidates[min(len(candidates) - 1, index * len(candidates) // maximum)]
        for index in range(maximum)
    )


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    tokens = sum(int(row["tokens"]) for row in rows)
    windows = len(rows)
    return {
        "windows": windows,
        "tokens": tokens,
        "teacher_forced_top1": sum(int(row["correct"]) for row in rows) / tokens,
        "mean_accepted_prefix": sum(int(row["accepted"]) for row in rows)
        / windows,
        "full_accept_rate": sum(
            int(row["accepted"]) == int(row["tokens"]) for row in rows
        )
        / windows,
    }


@torch.inference_mode()
def evaluate_boundaries(
    model: Any,
    samples: list[TeacherSample],
    fractions: tuple[float, ...],
    *,
    page_size: int,
    selection_seed: int,
    selection_mode: str,
    memory_mode: str,
    window_tokens: int,
    maximum_windows_per_sample: int,
    device: torch.device,
) -> list[dict[str, Any]]:
    if memory_mode not in {"exact", "zero", "shuffled"}:
        raise ValueError("unsupported boundary memory mode")
    model.eval()
    results = []
    for fraction in fractions:
        rows = []
        for sample_index, sample in enumerate(samples):
            memory_sample = sample
            if memory_mode == "shuffled":
                memory_sample = samples[(sample_index + 1) % len(samples)]
            keys, values, actual_fraction, _ = select_memory(
                memory_sample,
                page_size=page_size,
                requested_fraction=fraction,
                seed=selection_seed + memory_sample.request_index * 1009,
                device=device,
                selection_mode=selection_mode,
            )
            if memory_mode == "zero":
                keys = torch.zeros_like(keys)
                values = torch.zeros_like(values)
            offsets = select_offsets(sample.horizon, maximum_windows_per_sample)
            for offset in offsets:
                (
                    input_ids,
                    seed_hidden,
                    query_cos,
                    query_sin,
                    labels,
                    _,
                    _,
                ) = training_window_inputs(
                    sample,
                    device,
                    offset=offset,
                    window_tokens=window_tokens,
                )
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits = model(
                        input_ids,
                        seed_hidden,
                        keys,
                        values,
                        query_cos,
                        query_sin,
                        visible_fraction=actual_fraction,
                        prompt_tokens=sample.prompt_tokens,
                    )
                predicted = logits[0].argmax(dim=-1)
                if hasattr(model, "target_ids"):
                    predicted = model.target_ids(predicted)
                proposal = []
                prefix = input_ids[:, :1]
                for _ in range(labels.numel()):
                    prefix_tokens = prefix.shape[1]
                    with torch.autocast(
                        device_type="cuda", dtype=torch.bfloat16
                    ):
                        step_logits = model(
                            prefix,
                            seed_hidden,
                            keys,
                            values,
                            query_cos[:prefix_tokens],
                            query_sin[:prefix_tokens],
                            visible_fraction=actual_fraction,
                            prompt_tokens=sample.prompt_tokens,
                        )
                    draft_token = step_logits[0, -1].argmax()
                    if hasattr(model, "target_ids"):
                        draft_token = model.target_ids(draft_token)
                    token = int(draft_token.item())
                    proposal.append(token)
                    prefix = torch.cat(
                        (
                            prefix,
                            torch.tensor(
                                [[token]], dtype=torch.long, device=device
                            ),
                        ),
                        dim=1,
                    )
                rows.append(
                    {
                        "source_group": sample.source_group or "unknown",
                        "tokens": labels.numel(),
                        "correct": int((predicted == labels).sum().item()),
                        "accepted": accepted_prefix_length(
                            proposal, labels.tolist()
                        ),
                    }
                )
        by_group = defaultdict(list)
        for row in rows:
            by_group[str(row["source_group"])].append(row)
        results.append(
            {
                "memory_mode": memory_mode,
                "requested_fraction": fraction,
                **summarize(rows),
                "by_source_group": {
                    group: summarize(group_rows)
                    for group, group_rows in sorted(by_group.items())
                },
            }
        )
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--teacher-manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--eval-modulus", type=int, default=5)
    parser.add_argument("--eval-remainder", type=int, default=0)
    parser.add_argument("--visibility-fractions", default="0.05,0.10,0.20")
    parser.add_argument("--page-size", type=int, default=64)
    parser.add_argument("--selection-mode", default="priority")
    parser.add_argument("--selection-seed", type=int, default=20960827)
    parser.add_argument("--window-tokens", type=int, default=8)
    parser.add_argument("--max-windows-per-sample", type=int, default=8)
    parser.add_argument("--device", default="cuda:2")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.eval_modulus <= 1 or not 0 <= args.eval_remainder < args.eval_modulus:
        raise ValueError("invalid evaluation modulus split")
    fractions = parse_fraction_list(args.visibility_fractions)
    manifest_path = Path(args.teacher_manifest)
    raw_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    indices = {
        int(row["request_index"])
        for row in raw_manifest["samples"]
        if int(row["request_index"]) % args.eval_modulus == args.eval_remainder
    }
    manifest, sample_map = load_teacher_dataset(manifest_path, indices)
    samples = [sample_map[index] for index in sorted(sample_map)]
    if any("continuation_seed_hidden" not in sample.tensors for sample in samples):
        raise ValueError("boundary evaluation requires v5 teacher data")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    if checkpoint.get("format") == "sparsecache.sparse-kv-drafter.v1":
        embedding, lm_head, _ = load_target_shared_weights(
            str(checkpoint["target_model"]),
            device=device,
            attn_implementation="sdpa",
        )
        model = SparseKVDrafter(
            SparseKVDraftConfig.from_dict(checkpoint["config"])
        ).to(device)
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        model.bind_target_weights(embedding, lm_head)
    elif checkpoint.get("format") in {
        "sparsecache.sparse-kv-eagle-adapter.v1",
        "sparsecache.sparse-kv-eagle-adapter.v2",
    }:
        from experiments.train_sparse_kv_eagle import (
            build_model,
            load_target_embedding,
        )

        embedding, target_config = load_target_embedding(
            str(checkpoint["target_model"]), device=device
        )
        model = build_model(
            manifest=manifest,
            eagle_checkpoint=Path(checkpoint["base_eagle_checkpoint"]),
            target_embedding=embedding,
            target_config=target_config,
            device=device,
            fusion_mode=checkpoint.get(
                "fusion_mode",
                checkpoint.get("config", {}).get("fusion_mode", "scalar"),
            ),
        )
        if "fc.weight" in checkpoint["adapter_state_dict"]:
            model.fc.float()
        _, unexpected = model.load_state_dict(
            checkpoint["adapter_state_dict"], strict=False
        )
        if unexpected:
            raise ValueError(f"unexpected EAGLE adapter weights: {unexpected}")
    else:
        raise ValueError("unsupported boundary-evaluation checkpoint")
    started = perf_counter()
    cells = []
    for memory_mode in ("exact", "zero", "shuffled"):
        metrics = evaluate_boundaries(
            model,
            samples,
            fractions,
            page_size=args.page_size,
            selection_seed=args.selection_seed,
            selection_mode=args.selection_mode,
            memory_mode=memory_mode,
            window_tokens=args.window_tokens,
            maximum_windows_per_sample=args.max_windows_per_sample,
            device=device,
        )
        cells.extend(metrics)
        print(
            json.dumps(
                {
                    "event": "boundary_ablation",
                    "memory_mode": memory_mode,
                    "metrics": metrics,
                }
            ),
            flush=True,
        )
    result = {
        "schema_version": 1,
        "format": "sparsecache.sparse-kv-boundary-evaluation.v1",
        "created_at_ns": time_ns(),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "teacher_manifest": str(manifest_path.resolve()),
        "eval_indices": sorted(indices),
        "elapsed_seconds": perf_counter() - started,
        "cells": cells,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"event": "complete", "output": str(output)}), flush=True)


if __name__ == "__main__":
    main()
