"""Evaluate direct-KV block drafters against one immutable P-side packet set."""

from __future__ import annotations

import argparse
import json
import random
import statistics
import time
from pathlib import Path

import torch

from experiments.blockdraft.model import BlockKVDraft
from experiments.kvshot.model import apply_rope
from experiments.lossless_pd.core import (
    ProgressiveBlock,
    accepted_prefix,
    page_order,
    visible_positions,
)
from experiments.lossless_pd.pilot import bootstrap_delta, load_target
from experiments.lossless_pd.reference_packets import (
    digest_file,
    load_packet,
    read_index,
)


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def load_drafter(checkpoint: Path) -> ProgressiveBlock:
    model = ProgressiveBlock(BlockKVDraft.load_checkpoint(checkpoint))
    stage = checkpoint / "stage.pt"
    if stage.exists():
        model.stage.load_state_dict(torch.load(stage, map_location="cpu", weights_only=True))
    return model.to("cuda", dtype=torch.bfloat16).eval()


def select_other_memory(packet: dict, donor: dict, positions: torch.Tensor,
                        rope_theta: float):
    donor_k = donor["keys"].to("cuda", non_blocking=True)
    donor_v = donor["values"].to("cuda", non_blocking=True)
    source = torch.arange(positions.numel(), device="cuda") % donor_k.shape[-2]
    selected_k = donor_k.index_select(-2, source)
    batch, layers, heads, tokens, dim = selected_k.shape
    selected_k = apply_rope(
        selected_k.reshape(batch * layers, heads, tokens, dim),
        source[None].repeat_interleave(layers, 0),
        rope_theta,
        inverse=True,
    )
    selected_k = apply_rope(
        selected_k,
        positions[None].repeat_interleave(layers, 0),
        rope_theta,
    ).reshape(batch, layers, heads, tokens, dim)
    return selected_k, donor_v.index_select(-2, source)


@torch.no_grad()
def evaluate(args) -> None:
    packet_root = Path(args.packets).resolve()
    index = read_index(packet_root)
    target = load_target(args.target)
    model = load_drafter(Path(args.checkpoint))
    if index["layer_ids"] != [1, 9, 17, 25, 33]:
        raise ValueError("packet Target-layer contract differs from this block drafter")
    if args.draft_tokens > model.base.config.block_size - 1:
        raise ValueError("requested horizon exceeds block drafter output positions")
    fractions = [float(value) for value in args.fractions.split(",")]
    cells = [(order, fraction, "exact")
             for order in args.orders.split(",") for fraction in fractions]
    cells += [("priority", .1, "zero"), ("priority", .1, "shuffled")]
    rows, grouped = [], {}
    entries = index["entries"][:args.num_requests or None]
    previous = None
    for request_index, entry in enumerate(entries):
        packet = load_packet(packet_root, entry)
        keys = packet["keys"].to("cuda", non_blocking=True)
        values = packet["values"].to("cuda", non_blocking=True)
        scores = packet["priority_scores"].to("cuda")
        prompt_tokens = int(packet["prompt_tokens"])
        seed = packet["seed"].to("cuda")
        reference = packet["reference"][:args.draft_tokens].tolist()
        stop_ids = set(packet["stop_ids"].tolist())
        for order_mode, fraction, memory_mode in cells:
            if memory_mode == "shuffled" and previous is None:
                continue
            order = page_order(prompt_tokens, index["page_size"], order_mode,
                               args.seed + request_index, scores)
            positions = visible_positions(prompt_tokens, index["page_size"], fraction,
                                          order, torch.device("cuda"))
            selected_k = keys.index_select(-2, positions)
            selected_v = values.index_select(-2, positions)
            if memory_mode == "zero":
                selected_k = torch.zeros_like(selected_k)
                selected_v = torch.zeros_like(selected_v)
            elif memory_mode == "shuffled":
                selected_k, selected_v = select_other_memory(
                    packet, previous, positions, target.config.rope_theta
                )
            torch.cuda.synchronize()
            start = time.perf_counter()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                proposal = model.propose(
                    seed, target.model.embed_tokens, target.lm_head,
                    selected_k, selected_v, positions, prompt_tokens,
                    length=args.draft_tokens,
                )[0].tolist()
            torch.cuda.synchronize()
            elapsed_ms = (time.perf_counter() - start) * 1000
            # Diagnose the ceiling of hidden-space reranking without charging
            # this second forward to the proposal latency. A reranker restricted
            # to the inherited base top-K cannot repair an error whose target is
            # absent from this set, regardless of data or optimizer budget.
            with torch.autocast("cuda", dtype=torch.bfloat16):
                sparse_hidden = model.sparse_hidden(
                    seed, target.model.embed_tokens, selected_k, selected_v,
                    positions, prompt_tokens,
                )[:, :args.draft_tokens]
                base_logits = target.lm_head(sparse_hidden)
                diagnostic_k = min(args.diagnostic_topk, base_logits.shape[-1])
                base_candidates = base_logits.topk(diagnostic_k, dim=-1).indices[0]
                base_proposal = base_logits.argmax(-1)[0].tolist()
            target_tokens = torch.tensor(reference, device=base_candidates.device)
            covered = base_candidates.eq(target_tokens[:, None]).any(-1)
            base_accepted = accepted_prefix(base_proposal, reference, stop_ids)
            first_base_error_covered = (
                None if base_accepted >= len(reference)
                else bool(covered[base_accepted].item())
            )
            row = {
                "record_id": packet["record_id"],
                "input_sha256": packet["input_sha256"],
                "packet_sha256": entry["sha256"],
                "seed": int(seed.item()),
                "reference": reference,
                "proposal": proposal,
                "accepted": accepted_prefix(proposal, reference, stop_ids),
                "base_proposal": base_proposal,
                "base_accepted": base_accepted,
                "diagnostic_topk": diagnostic_k,
                "target_in_base_topk_rate": float(covered.float().mean()),
                "first_base_error_target_in_topk": first_base_error_covered,
                "order": order_mode,
                "fraction": fraction,
                "actual_fraction": positions.numel() / prompt_tokens,
                "memory_mode": memory_mode,
                "draft_ms_unfused_with_projection": elapsed_ms,
            }
            rows.append(row)
            grouped.setdefault(f"{order_mode}/f{fraction:g}/{memory_mode}", []).append(row)
        previous = {"keys": packet["keys"], "values": packet["values"]}
        print(json.dumps({"event": "packet_evaluation", "label": args.label,
                          "completed": request_index + 1, "total": len(entries)}), flush=True)

    summary = {}
    for key, cell in grouped.items():
        accepted = [row["accepted"] for row in cell]
        first_error_coverage = [
            float(row["first_base_error_target_in_topk"])
            for row in cell
            if row["first_base_error_target_in_topk"] is not None
        ]
        summary[key] = {
            "requests": len(cell),
            "mean_accepted": statistics.mean(accepted),
            "accepted_mean_ci95_request_bootstrap": bootstrap_delta(accepted),
            "zero_acceptance_rate": sum(value == 0 for value in accepted) / len(accepted),
            "mean_actual_fraction": statistics.mean(row["actual_fraction"] for row in cell),
            "mean_target_in_base_topk_rate": statistics.mean(
                row["target_in_base_topk_rate"] for row in cell
            ),
            "first_base_error_target_in_topk_rate": (
                statistics.mean(first_error_coverage)
                if first_error_coverage else None
            ),
            "mean_draft_ms_unfused_with_projection": statistics.mean(
                row["draft_ms_unfused_with_projection"] for row in cell
            ),
        }
    result = {
        "label": args.label,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_sha256": digest_file(Path(args.checkpoint) / "block_kv_draft.pt"),
        "packet_root": str(packet_root),
        "packet_index_sha256": digest_file(packet_root / "index.json"),
        "metric_contract": (
            "greedy accepted prefix excluding P seed against one immutable canonical "
            "full-BF16/full-KV Target trajectory"
        ),
        "cells": summary,
        "rows": rows,
    }
    write_json(Path(args.output), result)
    print(json.dumps({"event": "completed", "label": args.label,
                      "output": args.output, "cells": summary}, indent=2))


def paired_ci(values, seed=20260909, repeats=5000):
    rng = random.Random(seed)
    means = sorted(statistics.mean(rng.choices(values, k=len(values)))
                   for _ in range(repeats))
    return [means[int(.025 * repeats)], means[int(.975 * repeats)]]


def row_key(row):
    return row["record_id"], row["order"], row["fraction"], row["memory_mode"]


def compare(args) -> None:
    left = json.loads(Path(args.left).read_text())
    right = json.loads(Path(args.right).read_text())
    if left["packet_index_sha256"] != right["packet_index_sha256"]:
        raise RuntimeError("paired arms did not consume the same immutable packet index")
    right_rows = {row_key(row): row for row in right["rows"]}
    grouped = {}
    for row in left["rows"]:
        partner = right_rows.get(row_key(row))
        if partner is None:
            raise RuntimeError(f"missing paired row {row_key(row)}")
        for field in ("input_sha256", "packet_sha256", "seed", "reference"):
            if row[field] != partner[field]:
                raise RuntimeError(f"paired immutable field differs: {field}")
        key = f"{row['order']}/f{row['fraction']:g}/{row['memory_mode']}"
        grouped.setdefault(key, []).append(partner["accepted"] - row["accepted"])
    result = {
        "left": left["label"],
        "right": right["label"],
        "packet_index_sha256": left["packet_index_sha256"],
        "paired_claim_valid": True,
        "right_minus_left": {
            key: {"pairs": len(values), "mean_accepted_delta": statistics.mean(values),
                  "ci95_request_bootstrap": paired_ci(values)}
            for key, values in grouped.items()
        },
    }
    write_json(Path(args.output), result)
    print(json.dumps(result, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    evaluate_parser = subparsers.add_parser("evaluate")
    evaluate_parser.add_argument("--target", default="/data/models/qwen/Qwen3-8B")
    evaluate_parser.add_argument("--packets", required=True)
    evaluate_parser.add_argument("--checkpoint", required=True)
    evaluate_parser.add_argument("--output", required=True)
    evaluate_parser.add_argument("--label", required=True)
    evaluate_parser.add_argument("--num-requests", type=int, default=0)
    evaluate_parser.add_argument("--draft-tokens", type=int, default=15)
    evaluate_parser.add_argument("--fractions", default="0.05,0.1,0.2,0.5,1")
    evaluate_parser.add_argument("--orders", default="priority,random")
    evaluate_parser.add_argument("--diagnostic-topk", type=int, default=64)
    evaluate_parser.add_argument("--seed", type=int, default=20260909)

    compare_parser = subparsers.add_parser("compare")
    compare_parser.add_argument("--left", required=True)
    compare_parser.add_argument("--right", required=True)
    compare_parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "evaluate":
        if args.num_requests < 0 or min(args.draft_tokens, args.diagnostic_topk) <= 0:
            parser.error("invalid request or draft-token limit")
        evaluate(args)
    else:
        compare(args)


if __name__ == "__main__":
    main()
