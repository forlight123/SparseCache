"""Evaluate a drafter with the exact whole-chunk LMCache Anchor layout."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

from experiments.lossless_pd.core import accepted_prefix
from experiments.lossless_pd.lmcache_pd.anchor_runtime import anchor_indices
from experiments.lossless_pd.packet_eval import load_drafter
from experiments.lossless_pd.pilot import bootstrap_delta, load_target
from experiments.lossless_pd.reference_packets import load_packet, read_index


def runtime_anchor_positions(
    prompt_tokens: int,
    chunk_tokens: int,
    fraction: float,
    mode: str,
    device: torch.device,
) -> torch.Tensor:
    if min(prompt_tokens, chunk_tokens) <= 0:
        raise ValueError("prompt and LMCache chunk lengths must be positive")
    chunks = (prompt_tokens + chunk_tokens - 1) // chunk_tokens
    selected = anchor_indices(chunks, fraction, mode)
    return torch.tensor(
        [
            token
            for chunk in selected
            for token in range(
                chunk * chunk_tokens,
                min(prompt_tokens, (chunk + 1) * chunk_tokens),
            )
        ],
        dtype=torch.long,
        device=device,
    )


@torch.no_grad()
def run(args: argparse.Namespace) -> dict:
    packet_root = Path(args.packets).resolve()
    index = read_index(packet_root)
    target = load_target(args.target)
    drafter = load_drafter(Path(args.checkpoint))
    rows = []
    entries = index["entries"][: args.num_requests or None]
    for ordinal, entry in enumerate(entries):
        packet = load_packet(packet_root, entry)
        prompt_tokens = int(packet["prompt_tokens"])
        positions = runtime_anchor_positions(
            prompt_tokens,
            args.chunk_tokens,
            args.fraction,
            args.mode,
            torch.device("cuda"),
        )
        keys = packet["keys"].to("cuda", non_blocking=True).index_select(-2, positions)
        values = (
            packet["values"].to("cuda", non_blocking=True).index_select(-2, positions)
        )
        seed = packet["seed"].to("cuda")
        reference = packet["reference"][: args.draft_tokens].tolist()
        torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            if (
                drafter.base.correction_gru is not None
                and drafter.base.config.correction_mode == "rerank"
            ):
                prepared = drafter.prepare_rerank_block(
                    seed,
                    target.model.embed_tokens,
                    target.lm_head,
                    keys,
                    values,
                    positions,
                    prompt_tokens,
                    length=args.draft_tokens,
                )
                proposal_tensor = drafter.propose_prepared_rerank(
                    prepared, target.model.embed_tokens, target.lm_head
                )
            else:
                prepared = None
                proposal_tensor = drafter.propose(
                    seed,
                    target.model.embed_tokens,
                    target.lm_head,
                    keys,
                    values,
                    positions,
                    prompt_tokens,
                    length=args.draft_tokens,
                )
        torch.cuda.synchronize()
        draft_ms = (time.perf_counter() - started) * 1000
        proposal = proposal_tensor[0].tolist()
        target_conditioned_proposal = None
        suffix_target_in_base_topk = None
        suffix_target_in_diagnostic_topk = None
        first_conditioned_error_target_in_topk = None
        first_conditioned_error_target_in_diagnostic_topk = None
        repair_ms = None
        if prepared is not None and reference:
            repair_started = time.perf_counter()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                repaired = drafter.propose_prepared_rerank(
                    prepared,
                    target.model.embed_tokens,
                    target.lm_head,
                    first_target_token=torch.tensor(
                        [[reference[0]]], device="cuda", dtype=torch.long
                    ),
                )[0].tolist()
            torch.cuda.synchronize()
            repair_ms = (time.perf_counter() - repair_started) * 1000
            target_conditioned_proposal = [reference[0], *repaired]
            suffix_targets = torch.tensor(
                reference[1:], device=prepared.candidate_ids.device
            )
            suffix_candidates = prepared.candidate_ids[0, 1 : len(reference)]
            suffix_covered = suffix_candidates.eq(suffix_targets[:, None]).any(-1)
            suffix_target_in_base_topk = (
                float(suffix_covered.float().mean()) if suffix_covered.numel() else None
            )
            with torch.autocast("cuda", dtype=torch.bfloat16):
                diagnostic_candidates = (
                    target.lm_head(prepared.selected)
                    .topk(
                        min(args.diagnostic_topk, target.lm_head.weight.shape[0]),
                        dim=-1,
                    )
                    .indices[0]
                )
            diagnostic_suffix = diagnostic_candidates[1 : len(reference)]
            diagnostic_covered = diagnostic_suffix.eq(suffix_targets[:, None]).any(-1)
            suffix_target_in_diagnostic_topk = (
                float(diagnostic_covered.float().mean())
                if diagnostic_covered.numel()
                else None
            )
        stop_ids = set(packet["stop_ids"].tolist())
        target_conditioned_accepted = (
            accepted_prefix(target_conditioned_proposal, reference, stop_ids)
            if target_conditioned_proposal is not None
            else None
        )
        if (
            target_conditioned_accepted is not None
            and 1 <= target_conditioned_accepted < len(reference)
        ):
            first_conditioned_error_target_in_topk = bool(
                prepared.candidate_ids[0, target_conditioned_accepted]
                .eq(reference[target_conditioned_accepted])
                .any()
            )
            first_conditioned_error_target_in_diagnostic_topk = bool(
                diagnostic_candidates[target_conditioned_accepted]
                .eq(reference[target_conditioned_accepted])
                .any()
            )
        rows.append(
            {
                "ordinal": ordinal,
                "record_id": packet["record_id"],
                "prompt_tokens": prompt_tokens,
                "visible_tokens": positions.numel(),
                "actual_fraction": positions.numel() / prompt_tokens,
                "proposal": proposal,
                "reference": reference,
                "accepted": accepted_prefix(proposal, reference, stop_ids),
                "target_conditioned_proposal": target_conditioned_proposal,
                "target_conditioned_accepted": target_conditioned_accepted,
                "suffix_target_in_base_topk": suffix_target_in_base_topk,
                "suffix_target_in_diagnostic_topk": (suffix_target_in_diagnostic_topk),
                "first_conditioned_error_target_in_topk": (
                    first_conditioned_error_target_in_topk
                ),
                "first_conditioned_error_target_in_diagnostic_topk": (
                    first_conditioned_error_target_in_diagnostic_topk
                ),
                "draft_ms": draft_ms,
                "repair_ms": repair_ms,
            }
        )
    accepted = [row["accepted"] for row in rows]
    conditioned = [
        row["target_conditioned_accepted"]
        for row in rows
        if row["target_conditioned_accepted"] is not None
    ]
    return {
        "contract": (
            "immutable packet seed/reference and exact whole-chunk runtime Anchor "
            "layout; this does not emulate a differently truncated live prompt"
        ),
        "packets": str(packet_root),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "chunk_tokens": args.chunk_tokens,
        "anchor_fraction": args.fraction,
        "anchor_mode": args.mode,
        "draft_tokens": args.draft_tokens,
        "summary": {
            "requests": len(rows),
            "mean_accepted": statistics.fmean(accepted),
            "accepted_mean_95ci": bootstrap_delta(accepted),
            "first_token_match_rate": sum(value > 0 for value in accepted)
            / len(accepted),
            "zero_acceptance_rate": sum(value == 0 for value in accepted)
            / len(accepted),
            "mean_target_conditioned_accepted": (
                statistics.fmean(conditioned) if conditioned else None
            ),
            "target_conditioned_accepted_mean_95ci": (
                bootstrap_delta(conditioned) if conditioned else None
            ),
            "mean_target_conditioned_suffix_accepted": (
                statistics.fmean(max(0, value - 1) for value in conditioned)
                if conditioned
                else None
            ),
            "mean_suffix_target_in_base_topk_rate": (
                statistics.fmean(
                    row["suffix_target_in_base_topk"]
                    for row in rows
                    if row["suffix_target_in_base_topk"] is not None
                )
                if conditioned
                else None
            ),
            "diagnostic_topk": args.diagnostic_topk,
            "mean_suffix_target_in_diagnostic_topk_rate": (
                statistics.fmean(
                    row["suffix_target_in_diagnostic_topk"]
                    for row in rows
                    if row["suffix_target_in_diagnostic_topk"] is not None
                )
                if conditioned
                else None
            ),
            "first_conditioned_error_target_in_topk_rate": (
                statistics.fmean(
                    float(row["first_conditioned_error_target_in_topk"])
                    for row in rows
                    if row["first_conditioned_error_target_in_topk"] is not None
                )
                if any(
                    row["first_conditioned_error_target_in_topk"] is not None
                    for row in rows
                )
                else None
            ),
            "first_conditioned_error_target_in_diagnostic_topk_rate": (
                statistics.fmean(
                    float(row["first_conditioned_error_target_in_diagnostic_topk"])
                    for row in rows
                    if row["first_conditioned_error_target_in_diagnostic_topk"]
                    is not None
                )
                if any(
                    row["first_conditioned_error_target_in_diagnostic_topk"] is not None
                    for row in rows
                )
                else None
            ),
            "mean_draft_ms": statistics.fmean(row["draft_ms"] for row in rows),
            "mean_repair_ms": (
                statistics.fmean(
                    row["repair_ms"] for row in rows if row["repair_ms"] is not None
                )
                if conditioned
                else None
            ),
            "mean_actual_fraction": statistics.fmean(
                row["actual_fraction"] for row in rows
            ),
        },
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", default="/data/models/qwen/Qwen3-8B")
    parser.add_argument("--packets", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--num-requests", type=int, default=0)
    parser.add_argument("--draft-tokens", type=int, default=7)
    parser.add_argument("--fraction", type=float, default=0.1)
    parser.add_argument("--chunk-tokens", type=int, default=256)
    parser.add_argument("--mode", default="protected_uniform")
    parser.add_argument("--diagnostic-topk", type=int, default=256)
    args = parser.parse_args()
    if (
        args.num_requests < 0
        or min(args.draft_tokens, args.chunk_tokens, args.diagnostic_topk) <= 0
    ):
        parser.error("invalid request, draft, or chunk count")
    result = run(args)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
