"""Train block-wise sparse Target-KV cross-attention on immutable packets."""

from __future__ import annotations

import argparse
import json
import random
import statistics
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

from experiments.lossless_pd.reference_packets import digest_file, load_packet
from experiments.small_kv_adapter.cross_attention import (
    CrossAttentionConfig,
    SparseTargetKVCrossAttention,
)
from experiments.small_kv_adapter.train import (
    packet_entries,
    sparse_packet_view,
    validate_model_contract,
)


def train(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.set_num_threads(args.cpu_threads)
    roots = [Path(value) for value in args.packet_roots]
    entries = packet_entries(roots)
    random.Random(args.seed).shuffle(entries)

    config = CrossAttentionConfig(initial_gate=args.initial_gate)
    adapter = SparseTargetKVCrossAttention(config).to("cuda", dtype=torch.bfloat16)
    small = (
        AutoModelForCausalLM.from_pretrained(
            args.draft_model,
            local_files_only=True,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
        .to("cuda")
        .eval()
        .requires_grad_(False)
    )
    validate_model_contract(small, config)
    if int(small.config.hidden_size) != config.hidden_size:
        raise ValueError("small-model hidden width differs from cross-attention width")
    optimizer = torch.optim.AdamW(
        adapter.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )

    rows = []
    entry_cursor = 0
    consecutive_skipped = 0
    started = time.perf_counter()
    step = 0
    while step < args.steps:
        root, entry = entries[entry_cursor % len(entries)]
        entry_cursor += 1
        packet = load_packet(root, entry)
        reference = packet["reference"][: args.draft_tokens]
        if reference.numel() < args.draft_tokens:
            consecutive_skipped += 1
            if consecutive_skipped >= len(entries):
                raise RuntimeError("no packet has the requested draft horizon")
            continue
        consecutive_skipped = 0
        prompt_ids = packet["prompt_ids"].to("cuda", non_blocking=True)
        with torch.no_grad():
            prompt = small.model(prompt_ids, use_cache=True, return_dict=True)
        keys, values, positions = sparse_packet_view(
            packet, fraction=args.fraction, seed=args.seed + step
        )
        memory = adapter.prepare_memory(keys, values, positions)
        seed_token = packet["seed"].reshape(1, 1).to("cuda", non_blocking=True)
        labels = reference.reshape(1, -1).to("cuda", non_blocking=True)
        teacher_input = torch.cat((seed_token, labels[:, :-1]), dim=1)
        query_positions = torch.arange(
            int(packet["prompt_tokens"]),
            int(packet["prompt_tokens"]) + args.draft_tokens,
            device="cuda",
        )
        with adapter.activate(small, memory, query_positions):
            output = small.model(
                teacher_input,
                past_key_values=prompt.past_key_values,
                use_cache=False,
                return_dict=True,
            )
        logits = small.lm_head(output.last_hidden_state).float()
        per_token = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), reduction="none"
        ).reshape(1, -1)
        weights = args.survival_discount ** torch.arange(
            args.draft_tokens, device="cuda", dtype=torch.float32
        )
        weights = weights / weights.mean()
        loss = (per_token * weights).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            adapter.parameters(), args.max_grad_norm
        )
        optimizer.step()

        prediction = logits.argmax(dim=-1)
        matches = prediction.eq(labels)[0].tolist()
        teacher_prefix = next(
            (index for index, match in enumerate(matches) if not match), len(matches)
        )
        step += 1
        row = {
            "step": step,
            "record_id": packet["record_id"],
            "packet_root": str(root),
            "loss": float(loss.detach()),
            "grad_norm": float(grad_norm),
            "teacher_forced_accuracy": sum(matches) / len(matches),
            "teacher_forced_prefix": teacher_prefix,
            "visible_tokens": int(positions.numel()),
            "prompt_tokens": int(packet["prompt_tokens"]),
            "mean_abs_gate": statistics.fmean(
                float(bridge.gate.detach().abs()) for bridge in adapter.bridges
            ),
        }
        rows.append(row)
        if step == 1 or step % args.log_every == 0:
            print(json.dumps({"event": "train_cross_attention", **row}), flush=True)
        del packet, prompt_ids, prompt, keys, values, positions, memory, output, logits

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    output_root = args.output.resolve()
    output_root.mkdir(parents=True, exist_ok=False)
    adapter.save_checkpoint(output_root / "checkpoint")
    with (output_root / "train.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")
    summary = {
        "schema_version": 1,
        "architecture": "block_wise_sparse_target_kv_cross_attention",
        "draft_model": str(args.draft_model.resolve()),
        "packet_roots": [str(root.resolve()) for root in roots],
        "packet_index_sha256": {
            str(root.resolve()): digest_file(root.resolve() / "index.json")
            for root in roots
        },
        "steps": args.steps,
        "draft_tokens": args.draft_tokens,
        "fraction": args.fraction,
        "survival_discount": args.survival_discount,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "initial_gate": args.initial_gate,
        "seed": args.seed,
        "trainable_parameters": sum(
            parameter.numel() for parameter in adapter.parameters()
        ),
        "elapsed_seconds": elapsed,
        "gpu_hours": elapsed / 3600,
        "mean_loss_last_50": statistics.fmean(
            row["loss"] for row in rows[-min(50, len(rows)) :]
        ),
        "mean_teacher_prefix_last_50": statistics.fmean(
            row["teacher_forced_prefix"] for row in rows[-min(50, len(rows)) :]
        ),
        "final_mean_abs_gate": rows[-1]["mean_abs_gate"],
    }
    (output_root / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--draft-model", type=Path, required=True)
    parser.add_argument("--packet-roots", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--draft-tokens", type=int, default=8)
    parser.add_argument("--fraction", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--survival-discount", type=float, default=0.75)
    parser.add_argument("--initial-gate", type=float, default=0.05)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260910)
    parser.add_argument("--cpu-threads", type=int, default=8)
    parser.add_argument("--log-every", type=int, default=25)
    args = parser.parse_args()
    if min(args.steps, args.draft_tokens, args.cpu_threads) <= 0:
        parser.error("step/token/thread counts must be positive")
    if not 0.0 < args.fraction <= 1.0:
        parser.error("fraction must lie in (0, 1]")
    if not 0.0 < args.survival_discount <= 1.0:
        parser.error("survival discount must lie in (0, 1]")
    if not 0.0 <= args.initial_gate <= 1.0:
        parser.error("initial gate must lie in [0, 1]")
    return args


if __name__ == "__main__":
    train(parse_args())
