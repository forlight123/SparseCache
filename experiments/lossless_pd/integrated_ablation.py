"""Within-process 2x2 ablation for the structural SparseCache-PD revision.

The four arms rotate within every request/repetition so target-start effects
cannot be confused with a separate process, model load, or long-term GPU drift.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import torch

from experiments.compare_layer_ready_ablation import compare
from experiments.lossless_pd.core import page_order, visible_positions
from experiments.lossless_pd.integrated_probe import (
    LAYER_IDS,
    extend_baseline,
    load_drafter,
    run_condition,
    summarize,
)
from experiments.lossless_pd.pilot import load_target, write_json
from experiments.lossless_pd.reference_packets import load_packet, read_index
from experiments.lossless_pd.shape_invariant_attention import (
    register_shape_invariant_attention,
)

ARMS = (
    ("layer_ready", "baseline"),
    ("layer_ready", "speculative"),
    ("full_ready", "baseline"),
    ("full_ready", "speculative"),
)


def balanced_arms(request_index: int, repeat: int):
    offset = (request_index + repeat) % len(ARMS)
    return ARMS[offset:] + ARMS[:offset]


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--packets", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--target", default="/data/models/qwen/Qwen3-8B")
    parser.add_argument("--requests", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--fraction", type=float, default=0.1)
    parser.add_argument(
        "--order",
        choices=["priority", "random", "reverse_priority"],
        default="priority",
    )
    parser.add_argument("--proposal-tokens", type=int, default=7)
    parser.add_argument("--gbps", type=float, default=100)
    parser.add_argument(
        "--verifier-semantics", choices=["eager", "bitwise"], default="eager"
    )
    args = parser.parse_args()
    if args.requests <= 0 or args.repeats <= 0 or not 0 < args.fraction <= 1:
        parser.error("invalid request, repeat, or visibility limit")

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    packet_root = Path(args.packets).resolve()
    index = read_index(packet_root)
    target = load_target(args.target)
    target.config._attn_implementation = "eager"
    drafter = load_drafter(args.checkpoint)
    write_json(
        output / "manifest.json",
        {
            "arguments": {
                **vars(args),
                "packets": str(packet_root),
                "checkpoint": str(Path(args.checkpoint).resolve()),
                "output": str(output.resolve()),
                "target": str(Path(args.target).resolve()),
            },
            "drafter_config": asdict(drafter.base.config),
            "arm_order": "four-arm cyclic rotation by request_index + repeat",
            "measurement_contract": (
                "one process and one physical GPU; wait-full/layer-ready x "
                "no-draft/sparse-draft; all four arms rotate within each pair"
            ),
        },
    )

    rows = []
    for request_index, entry in enumerate(index["entries"][: args.requests]):
        packet = load_packet(packet_root, entry)
        ids = packet["prompt_ids"].to("cuda")
        target.config._attn_implementation = "sdpa"
        prompt = target.model(ids, use_cache=True)
        seed = target.lm_head(prompt.last_hidden_state[:, -1:]).argmax(-1)
        if int(seed.item()) != int(packet["seed"].item()):
            raise RuntimeError("fresh P seed differs from immutable packet")
        host = [
            (layer.keys.cpu().pin_memory(), layer.values.cpu().pin_memory())
            for layer in prompt.past_key_values.layers
        ]
        scores = packet["priority_scores"].to("cuda")
        order = page_order(
            ids.shape[1],
            index["page_size"],
            args.order,
            index["seed"] + request_index,
            scores,
        )
        positions = visible_positions(
            ids.shape[1],
            index["page_size"],
            args.fraction,
            order,
            torch.device("cuda"),
        )
        anchor_keys = torch.stack([host[layer][0] for layer in LAYER_IDS], dim=1)
        anchor_values = torch.stack([host[layer][1] for layer in LAYER_IDS], dim=1)
        cpu_positions = positions.cpu()
        anchor = (
            anchor_keys.index_select(-2, cpu_positions).pin_memory(),
            anchor_values.index_select(-2, cpu_positions).pin_memory(),
        )
        del prompt, anchor_keys, anchor_values
        target.config._attn_implementation = (
            register_shape_invariant_attention()
            if args.verifier_semantics == "bitwise"
            else "eager"
        )

        warm_k = anchor[0].to("cuda")
        warm_v = anchor[1].to("cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            drafter.propose(
                seed,
                target.model.embed_tokens,
                target.lm_head,
                warm_k,
                warm_v,
                positions,
                ids.shape[1],
                length=args.proposal_tokens,
            )
        torch.cuda.synchronize()
        del warm_k, warm_v

        for repeat in range(args.repeats):
            cells = {}
            for target_start_mode, condition in balanced_arms(request_index, repeat):
                cells[(target_start_mode, condition)] = run_condition(
                    target,
                    drafter,
                    host,
                    anchor,
                    seed,
                    ids.shape[1],
                    positions,
                    args.gbps,
                    args.proposal_tokens,
                    condition,
                    args.verifier_semantics,
                    target_start_mode,
                )

            mode_rows = {}
            for target_start_mode in ("layer_ready", "full_ready"):
                speculative = cells[(target_start_mode, "speculative")]
                baseline = cells[(target_start_mode, "baseline")]
                progress = len(speculative["committed"])
                tail_started = time.perf_counter()
                extend_baseline(target, baseline, progress)
                baseline["same_progress_ms"] += (
                    time.perf_counter() - tail_started
                ) * 1000
                baseline.pop("cache", None)
                baseline.pop("next_token", None)
                speculative.pop("cache", None)
                speculative.pop("next_token", None)
                row = {
                    "record_id": packet["record_id"],
                    "repeat": repeat,
                    "context": ids.shape[1],
                    "actual_fraction": positions.numel() / ids.shape[1],
                    "proposal_tokens": args.proposal_tokens,
                    "progress_tokens": progress,
                    "committed_output_equal": baseline["committed"]
                    == speculative["committed"],
                    "target_start_mode": target_start_mode,
                    "baseline": baseline,
                    "speculative": speculative,
                }
                rows.append(row)
                mode_rows[target_start_mode] = row
            if (
                mode_rows["layer_ready"]["speculative"]["committed"]
                != mode_rows["full_ready"]["speculative"]["committed"]
            ):
                raise RuntimeError("target start mode changed committed output")
            write_json(output / "progress.json", {"rows": rows})
        print(
            json.dumps(
                {
                    "event": "integrated_2x2",
                    "completed": request_index + 1,
                    "total": min(args.requests, len(index["entries"])),
                }
            ),
            flush=True,
        )

    documents = {}
    for mode in ("layer_ready", "full_ready"):
        mode_args = argparse.Namespace(**vars(args), target_start_mode=mode)
        mode_rows = [row for row in rows if row["target_start_mode"] == mode]
        documents[mode] = {
            "summary": summarize(mode_rows, mode_args),
            "rows": mode_rows,
        }
        write_json(output / f"{mode}_results.json", documents[mode])
    ablation = compare(documents["layer_ready"], documents["full_ready"])
    ablation["pairing_scope"] = (
        "all four arms order-balanced within one process and physical GPU"
    )
    summary = {
        "contract": (
            "within-process order-balanced 2x2 structural ablation; one-host "
            "paced-link proxy, not physical two-node evidence"
        ),
        "requests": args.requests,
        "paired_runs_per_target_start_mode": args.requests * args.repeats,
        "layer_ready": documents["layer_ready"]["summary"],
        "full_ready": documents["full_ready"]["summary"],
        "ablation": ablation,
    }
    write_json(output / "results.json", {"summary": summary, "rows": rows})
    write_json(output / "summary.json", summary)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
