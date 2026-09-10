"""Train only the sparse Target-KV residual adapter on immutable packets."""

from __future__ import annotations

import argparse
import json
import random
import statistics
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

from experiments.lossless_pd.core import page_order, visible_positions
from experiments.lossless_pd.reference_packets import (
    digest_file,
    load_packet,
    read_index,
)
from experiments.small_kv_adapter.model import (
    SmallKVAdapterConfig,
    SparseTargetKVAdapter,
)


def packet_entries(roots: list[Path]) -> list[tuple[Path, dict[str, Any]]]:
    entries = []
    for root in roots:
        resolved = root.resolve()
        index = read_index(resolved)
        if tuple(index["layer_ids"]) != SmallKVAdapterConfig().target_layer_ids:
            raise ValueError(f"packet Target-layer contract differs: {resolved}")
        entries.extend((resolved, entry) for entry in index["entries"])
    if not entries:
        raise ValueError("training packet roots contain no entries")
    return entries


def sparse_packet_view(
    packet: dict[str, Any], *, fraction: float, seed: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    prompt_tokens = int(packet["prompt_tokens"])
    scores = packet["priority_scores"]
    order = page_order(
        prompt_tokens,
        int(packet["page_size"]),
        "priority",
        seed,
        scores,
    )
    positions = visible_positions(
        prompt_tokens,
        int(packet["page_size"]),
        fraction,
        order,
        torch.device("cpu"),
    )
    keys = packet["keys"].index_select(-2, positions).to("cuda", non_blocking=True)
    values = packet["values"].index_select(-2, positions).to("cuda", non_blocking=True)
    return keys, values, positions.to("cuda", non_blocking=True)


def validate_model_contract(model: torch.nn.Module, config: SmallKVAdapterConfig):
    if int(model.config.num_key_value_heads) != config.num_key_value_heads:
        raise ValueError("small model and adapter KV-head counts differ")
    if int(model.config.head_dim) != config.head_dim:
        raise ValueError("small model and adapter head dimensions differ")
    if max(config.draft_layer_ids) >= int(model.config.num_hidden_layers):
        raise ValueError("adapter draft layer lies outside the small model")
    rope = getattr(model.config, "rope_parameters", None) or {}
    model_rope_theta = float(
        rope.get(
            "rope_theta",
            getattr(model.config, "rope_theta", 10_000.0),
        )
    )
    if model_rope_theta != config.rope_theta:
        raise ValueError("small model and Target adapter RoPE theta differ")


def train(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.set_num_threads(args.cpu_threads)
    roots = [Path(value) for value in args.packet_roots]
    entries = packet_entries(roots)
    rng = random.Random(args.seed)
    rng.shuffle(entries)

    config = SmallKVAdapterConfig(bottleneck_size=args.bottleneck_size)
    adapter = SparseTargetKVAdapter(config).to("cuda", dtype=torch.bfloat16)
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
    optimizer = torch.optim.AdamW(
        adapter.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )

    rows: list[dict[str, Any]] = []
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
        cache = adapter(prompt.past_key_values, keys, values, positions)
        seed_token = packet["seed"].reshape(1, 1).to("cuda", non_blocking=True)
        labels = reference.reshape(1, -1).to("cuda", non_blocking=True)
        teacher_input = torch.cat((seed_token, labels[:, :-1]), dim=1)

        output = small.model(
            teacher_input,
            past_key_values=cache,
            use_cache=False,
            return_dict=True,
        )
        logits = small.lm_head(output.last_hidden_state).float()
        per_token = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), reduction="none"
        ).reshape(1, -1)
        # Earlier proposal positions dominate speculative survival.  The
        # normalized geometric weights optimize that prefix objective without
        # changing the fixed g=8 output contract.
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
        }
        rows.append(row)
        if step == 1 or step % args.log_every == 0:
            print(json.dumps({"event": "train", **row}), flush=True)
        del packet, prompt_ids, prompt, keys, values, positions, cache, output, logits

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    adapter.save_checkpoint(output / "checkpoint")
    with (output / "train.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")
    summary = {
        "schema_version": 1,
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
        "bottleneck_size": args.bottleneck_size,
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
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--draft-model", type=Path, required=True)
    parser.add_argument("--packet-roots", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--draft-tokens", type=int, default=8)
    parser.add_argument("--fraction", type=float, default=0.1)
    parser.add_argument("--bottleneck-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--survival-discount", type=float, default=0.9)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260910)
    parser.add_argument("--cpu-threads", type=int, default=8)
    parser.add_argument("--log-every", type=int, default=10)
    args = parser.parse_args()
    if min(args.steps, args.draft_tokens, args.bottleneck_size, args.cpu_threads) <= 0:
        parser.error("step/token/dimension/thread counts must be positive")
    if not 0.0 < args.fraction <= 1.0:
        parser.error("fraction must lie in (0, 1]")
    if not 0.0 < args.survival_discount <= 1.0:
        parser.error("survival discount must lie in (0, 1]")
    return args


if __name__ == "__main__":
    train(parse_args())
