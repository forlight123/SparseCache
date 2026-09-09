"""Train a causal proposal corrector from immutable long-prefix KV packets.

This is deliberately not EAGLE: the drafter receives a seed token, sparse
Target KV, original positions, and preceding proposal tokens only.  It never
receives live Target hidden states.  LongBench request packs are allowed for
mechanism rejection experiments only and must not be reported as clean task
evaluation after being used here.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections import OrderedDict
from dataclasses import asdict, replace
from pathlib import Path

import torch
import torch.nn.functional as F

from experiments.blockdraft.model import BlockKVDraft
from experiments.lossless_pd.core import (
    ProgressiveBlock,
    accepted_prefix,
    page_order,
    visible_positions,
)
from experiments.lossless_pd.lmcache_pd.eval_runtime_layout import (
    runtime_anchor_positions,
)
from experiments.lossless_pd.pilot import bootstrap_delta, load_target, write_json
from experiments.lossless_pd.reference_packets import (
    digest_file,
    load_packet,
    read_index,
)


class PacketPool:
    def __init__(self, roots: list[Path], cache_size: int):
        self.items = []
        self.cache_size = cache_size
        self.cache = OrderedDict()
        self.index_digests = {}
        for root in roots:
            index = read_index(root)
            self.index_digests[str(root)] = digest_file(root / "index.json")
            for entry in index["entries"]:
                if entry["reference_tokens"] >= 1:
                    self.items.append((root, index, entry))
        if not self.items:
            raise ValueError("no nonempty immutable packet targets")

    def get(self, item_index: int):
        root, index, entry = self.items[item_index]
        key = str(root / entry["packet"])
        if key not in self.cache:
            self.cache[key] = load_packet(root, entry)
            while len(self.cache) > self.cache_size:
                self.cache.popitem(last=False)
        self.cache.move_to_end(key)
        return index, entry, self.cache[key]


def convert_checkpoint(args) -> ProgressiveBlock:
    source = Path(args.base_checkpoint)
    source_model = BlockKVDraft.load_checkpoint(source)
    if source_model.config.use_seed_hidden:
        raise ValueError(
            "causal correction must not inherit Target hidden-state inputs"
        )
    config = replace(
        source_model.config,
        correction_hidden_size=args.correction_hidden,
        correction_bottleneck_size=args.correction_bottleneck,
        correction_mode=args.correction_mode,
        correction_topk=(
            args.correction_topk if args.correction_mode == "rerank" else 0
        ),
    )
    base = BlockKVDraft(config)
    correction_fields = (
        "correction_hidden_size",
        "correction_bottleneck_size",
        "correction_mode",
    )
    if all(
        getattr(config, field) == getattr(source_model.config, field)
        for field in correction_fields
    ):
        # Continue an already trained causal reranker without silently
        # reinitializing its correction head.
        base.load_state_dict(source_model.state_dict(), strict=True)
    else:
        base.initialize_from_blockdraft(source)
    model = ProgressiveBlock(base)
    stage = source / "stage.pt"
    if stage.exists():
        model.stage.load_state_dict(
            torch.load(stage, map_location="cpu", weights_only=True)
        )
    return model


def select_trainable_parameters(model: ProgressiveBlock, scope: str):
    """Select a precision-safe training scope for the direct-KV model."""

    if scope not in {"correction", "adapter", "all"}:
        raise ValueError(f"unknown train scope: {scope}")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if scope == "all":
        model.float()
        selected = list(model.parameters())
    else:
        modules = [model.base.correction_gru, model.base.correction_head]
        if any(module is None for module in modules):
            raise ValueError("correction training needs a causal correction head")
        if scope == "adapter":
            if model.base.target_hidden_projector is None:
                raise ValueError("adapter scope needs a Target-KV hidden projector")
            modules.extend((model.stage, model.base.target_hidden_projector))
        selected = []
        seen = set()
        for module in modules:
            module.float()
            for parameter in module.parameters():
                if id(parameter) not in seen:
                    selected.append(parameter)
                    seen.add(id(parameter))
    for parameter in selected:
        parameter.requires_grad_(True)
    return selected


def prefix_supervision_mask(
    correct: torch.Tensor,
    covered: torch.Tensor,
    loss_start_position: int,
) -> torch.Tensor:
    """Keep reachable prefix positions through the first greedy error.

    Once a speculative token is wrong, later teacher-forced positions cannot
    contribute to the accepted prefix for that draft.  An uncovered target is
    still a prefix-ending error, but is excluded from the loss because a
    candidate reranker cannot promote a token that is absent from its set.
    """

    if correct.shape != covered.shape or correct.ndim != 2:
        raise ValueError("correct and covered must have matching [B,H] shapes")
    horizon = correct.shape[1]
    if not 0 <= loss_start_position < horizon:
        raise ValueError("loss start must select a position in the horizon")
    positions = torch.arange(horizon, device=correct.device)[None, :]
    active = positions >= loss_start_position
    failures = active & ~correct
    sentinel = torch.full_like(positions.expand_as(correct), horizon)
    first_failure = torch.where(failures, positions, sentinel).amin(dim=1)
    reachable = active & (positions <= first_failure[:, None])
    return reachable & covered


def sparse_inputs(
    packet,
    index,
    fraction,
    order_mode,
    seed,
    *,
    runtime_chunk_tokens=0,
    runtime_anchor_mode="protected_uniform",
):
    prompt_tokens = int(packet["prompt_tokens"])
    if runtime_chunk_tokens:
        positions = runtime_anchor_positions(
            prompt_tokens,
            runtime_chunk_tokens,
            fraction,
            runtime_anchor_mode,
            torch.device("cpu"),
        )
    else:
        order = page_order(
            prompt_tokens,
            index["page_size"],
            order_mode,
            seed,
            packet["priority_scores"],
        )
        positions = visible_positions(
            prompt_tokens, index["page_size"], fraction, order, torch.device("cpu")
        )
    keys = packet["keys"].index_select(-2, positions).to("cuda", non_blocking=True)
    values = packet["values"].index_select(-2, positions).to("cuda", non_blocking=True)
    return positions.to("cuda"), keys, values


@torch.no_grad()
def evaluate(
    model,
    target,
    pool,
    requests,
    fraction,
    seed,
    *,
    runtime_chunk_tokens=0,
    runtime_anchor_mode="protected_uniform",
):
    model.eval()
    rows = []
    for item_index in range(min(requests, len(pool.items))):
        index, entry, packet = pool.get(item_index)
        length = min(model.base.config.block_size - 1, packet["reference"].numel())
        positions, keys, values = sparse_inputs(
            packet,
            index,
            fraction,
            "priority",
            seed + item_index,
            runtime_chunk_tokens=runtime_chunk_tokens,
            runtime_anchor_mode=runtime_anchor_mode,
        )
        reference = packet["reference"][:length].tolist()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            if (
                model.base.correction_gru is not None
                and model.base.config.correction_mode == "rerank"
            ):
                prepared = model.prepare_rerank_block(
                    packet["seed"].to("cuda"),
                    target.model.embed_tokens,
                    target.lm_head,
                    keys,
                    values,
                    positions,
                    int(packet["prompt_tokens"]),
                    length=length,
                )
                proposal = model.propose_prepared_rerank(
                    prepared, target.model.embed_tokens, target.lm_head
                )[0].tolist()
                repaired = model.propose_prepared_rerank(
                    prepared,
                    target.model.embed_tokens,
                    target.lm_head,
                    first_target_token=torch.tensor(
                        [[reference[0]]], device="cuda", dtype=torch.long
                    ),
                )[0].tolist()
                conditioned = [reference[0], *repaired]
            else:
                proposal = model.propose(
                    packet["seed"].to("cuda"),
                    target.model.embed_tokens,
                    target.lm_head,
                    keys,
                    values,
                    positions,
                    int(packet["prompt_tokens"]),
                    length=length,
                )[0].tolist()
                conditioned = None
        accepted = accepted_prefix(
            proposal, reference, set(packet["stop_ids"].tolist())
        )
        conditioned_accepted = (
            accepted_prefix(conditioned, reference, set(packet["stop_ids"].tolist()))
            if conditioned is not None
            else None
        )
        rows.append(
            {
                "record_id": packet["record_id"],
                "packet_sha256": entry["sha256"],
                "accepted": accepted,
                "target_conditioned_accepted": conditioned_accepted,
                "horizon": length,
                "proposal": proposal,
                "reference": reference,
            }
        )
    conditioned = [
        row["target_conditioned_accepted"]
        for row in rows
        if row["target_conditioned_accepted"] is not None
    ]
    return {
        "requests": len(rows),
        "mean_accepted": sum(row["accepted"] for row in rows) / len(rows),
        "zero_acceptance_rate": sum(row["accepted"] == 0 for row in rows) / len(rows),
        "mean_target_conditioned_accepted": (
            sum(conditioned) / len(conditioned) if conditioned else None
        ),
        "mean_target_conditioned_suffix_accepted": (
            sum(max(0, value - 1) for value in conditioned) / len(conditioned)
            if conditioned
            else None
        ),
        "rows": rows,
    }


def save_checkpoint(model, destination):
    destination.mkdir(parents=True, exist_ok=False)
    model.base.save_checkpoint(destination)
    torch.save(model.stage.state_dict(), destination / "stage.pt")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-packets", required=True)
    parser.add_argument("--eval-packets", required=True)
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--target", default="/data/models/qwen/Qwen3-8B")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--eval-requests", type=int, default=64)
    parser.add_argument("--cache-packets", type=int, default=12)
    parser.add_argument("--fractions", default="0.05,0.1,0.2,0.5")
    parser.add_argument("--orders", default="priority")
    parser.add_argument(
        "--runtime-chunk-tokens",
        type=int,
        default=0,
        help="use the exact whole-chunk runtime Anchor layout when nonzero",
    )
    parser.add_argument("--runtime-anchor-mode", default="protected_uniform")
    parser.add_argument(
        "--loss-start-position",
        type=int,
        default=0,
        help="zero-based first proposal position included in the loss",
    )
    parser.add_argument("--correction-hidden", type=int, default=256)
    parser.add_argument("--correction-bottleneck", type=int, default=256)
    parser.add_argument(
        "--correction-mode", choices=["vocab", "rerank"], default="rerank"
    )
    parser.add_argument("--correction-topk", type=int, default=64)
    parser.add_argument(
        "--train-scope",
        choices=["correction", "adapter", "all"],
        default="correction",
    )
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument(
        "--objective",
        choices=[
            "hard_ce",
            "error_correct",
            "prefix_hard_ce",
            "prefix_error_correct",
        ],
        default="error_correct",
    )
    parser.add_argument("--preserve-weight", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260909)
    args = parser.parse_args()
    if (
        min(
            args.steps,
            args.eval_requests,
            args.cache_packets,
            args.correction_hidden,
            args.correction_bottleneck,
            args.correction_topk,
        )
        <= 0
    ):
        parser.error("all resource and model limits must be positive")
    if args.preserve_weight < 0:
        parser.error("preservation weight cannot be negative")
    if args.runtime_chunk_tokens < 0 or args.loss_start_position < 0:
        parser.error("runtime chunk size and loss start must be nonnegative")
    fractions = [float(value) for value in args.fractions.split(",")]
    orders = [value for value in args.orders.split(",") if value]
    if not fractions or min(fractions) <= 0 or max(fractions) > 1:
        parser.error("fractions must be in (0,1]")
    if not orders or set(orders) - {"priority", "random", "reverse_priority"}:
        parser.error("invalid training order")

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    train_roots = [Path(value).resolve() for value in args.train_packets.split(",")]
    eval_roots = [Path(value).resolve() for value in args.eval_packets.split(",")]
    train_pool = PacketPool(train_roots, args.cache_packets)
    eval_pool = PacketPool(eval_roots, args.cache_packets)
    train_pool.items = [
        item
        for item in train_pool.items
        if item[2]["reference_tokens"] > args.loss_start_position
    ]
    if not train_pool.items:
        raise ValueError("no training packet reaches the requested loss start")
    train_hashes = {entry[2]["input_sha256"] for entry in train_pool.items}
    eval_hashes = {entry[2]["input_sha256"] for entry in eval_pool.items}
    overlap = train_hashes & eval_hashes
    if overlap:
        raise RuntimeError(f"train/eval immutable inputs overlap: {len(overlap)}")

    torch.manual_seed(args.seed)
    randomizer = random.Random(args.seed)
    target = load_target(args.target)
    model = convert_checkpoint(args).to("cuda", dtype=torch.bfloat16)
    # Frozen components stay compact BF16. Every trainable scope is promoted to
    # FP32 so AdamW moments and small updates are not quantized to BF16.
    trainable = select_trainable_parameters(model, args.train_scope)
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.01)
    manifest = {
        "arguments": vars(args),
        "base_checkpoint_sha256": digest_file(
            Path(args.base_checkpoint) / "block_kv_draft.pt"
        ),
        "train_packet_indices": train_pool.index_digests,
        "eval_packet_indices": eval_pool.index_digests,
        "train_eval_input_overlap": 0,
        "model_config": asdict(model.base.config),
        "trainable_parameters": sum(parameter.numel() for parameter in trainable),
        "input_contract": (
            "seed token + sparse Target KV + original KV positions + preceding proposal "
            "tokens; no EAGLE state and no live Target hidden state"
        ),
        "data_warning": (
            "LongBench-derived training packets are mechanism-only; tasks represented in "
            "train packets are contaminated and prohibited as paper evaluation."
        ),
        "status": "running",
    }
    write_json(output / "manifest.json", manifest)
    before = evaluate(
        model,
        target,
        eval_pool,
        args.eval_requests,
        0.1,
        args.seed,
        runtime_chunk_tokens=args.runtime_chunk_tokens,
        runtime_anchor_mode=args.runtime_anchor_mode,
    )
    write_json(output / "before.json", before)

    model.train()
    position_weights = torch.tensor(
        [0.9**position for position in range(model.base.config.block_size - 1)],
        device="cuda",
    )
    started = time.perf_counter()
    losses = []
    with (output / "train.jsonl").open("w") as log:
        for step in range(args.steps):
            item_index = randomizer.randrange(len(train_pool.items))
            index, _entry, packet = train_pool.get(item_index)
            horizon = min(model.base.config.block_size - 1, packet["reference"].numel())
            fraction = fractions[step % len(fractions)]
            order_mode = orders[(step // len(fractions)) % len(orders)]
            positions, keys, values = sparse_inputs(
                packet,
                index,
                fraction,
                order_mode,
                args.seed + step,
                runtime_chunk_tokens=args.runtime_chunk_tokens,
                runtime_anchor_mode=args.runtime_anchor_mode,
            )
            seed = packet["seed"].to("cuda")
            labels = packet["reference"][:horizon].to("cuda")[None]
            previous = torch.cat((seed, labels[:, :-1]), dim=1)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                if args.correction_mode == "rerank":
                    candidate_ids, base_scores, correction_scores = (
                        model.teacher_forced_rerank(
                            seed,
                            target.model.embed_tokens,
                            target.lm_head,
                            keys,
                            values,
                            positions,
                            int(packet["prompt_tokens"]),
                            previous_tokens=previous,
                        )
                    )
                    logits = base_scores + correction_scores
                    matches = candidate_ids.eq(labels[:, :, None])
                    covered = matches.any(-1)
                    candidate_labels = matches.float().argmax(-1)
                    cross_entropy = F.cross_entropy(
                        logits.float().reshape(-1, logits.shape[-1]),
                        candidate_labels.reshape(-1),
                        reduction="none",
                    ).reshape(1, -1)
                    base_correct = candidate_ids[:, :, 0].eq(labels)
                    base_reference = base_scores
                else:
                    base_logits, correction = model.forward_components(
                        seed,
                        target.model.embed_tokens,
                        target.lm_head,
                        keys,
                        values,
                        positions,
                        int(packet["prompt_tokens"]),
                        previous_tokens=previous,
                    )
                    logits = base_logits + correction
                    cross_entropy = F.cross_entropy(
                        logits.float().reshape(-1, logits.shape[-1]),
                        labels.reshape(-1),
                        reduction="none",
                    ).reshape(1, -1)
                    base_correct = base_logits.argmax(-1).eq(labels)
                    covered = torch.ones_like(base_correct)
                    base_reference = base_logits
                if args.objective in {"error_correct", "prefix_error_correct"}:
                    base_log_probability = base_reference.float().log_softmax(-1)
                    corrected_log_probability = logits.float().log_softmax(-1)
                    preservation = (
                        base_log_probability.exp()
                        * (base_log_probability - corrected_log_probability)
                    ).sum(-1)
                    per_position = torch.where(
                        base_correct,
                        args.preserve_weight * preservation,
                        cross_entropy,
                    )
                else:
                    per_position = cross_entropy
                if args.objective.startswith("prefix_"):
                    supervision = prefix_supervision_mask(
                        logits.argmax(-1).eq(candidate_labels)
                        if args.correction_mode == "rerank"
                        else logits.argmax(-1).eq(labels),
                        covered,
                        args.loss_start_position,
                    )
                else:
                    supervision = covered.clone()
                    supervision[:, : args.loss_start_position] = False
                weights = position_weights[:horizon][None] * supervision
                loss = (
                    per_position[:, :horizon] * weights
                ).sum() / weights.sum().clamp_min(1e-12)
            if not torch.isfinite(loss):
                raise RuntimeError("nonfinite causal-correction loss")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(
                trainable, 1.0, error_if_nonfinite=True
            )
            warmup = min(1.0, (step + 1) / 50)
            cosine = 0.5 * (1 + math.cos(math.pi * step / args.steps))
            for group in optimizer.param_groups:
                group["lr"] = args.lr * warmup * cosine
            optimizer.step()
            losses.append(float(loss.detach()))
            row = {
                "step": step + 1,
                "record_id": packet["record_id"],
                "fraction": fraction,
                "order": order_mode,
                "loss": losses[-1],
                "base_position_accuracy": float(base_correct.float().mean()),
                "target_in_candidate_rate": float(covered.float().mean()),
                "supervised_positions": int(supervision.sum()),
                "grad_norm": float(norm),
                "elapsed_s": time.perf_counter() - started,
            }
            log.write(json.dumps(row) + "\n")
            if step == 0 or (step + 1) % 25 == 0:
                log.flush()
                print(json.dumps(row), flush=True)

    model.eval()
    after = evaluate(
        model,
        target,
        eval_pool,
        args.eval_requests,
        0.1,
        args.seed,
        runtime_chunk_tokens=args.runtime_chunk_tokens,
        runtime_anchor_mode=args.runtime_anchor_mode,
    )
    write_json(output / "after.json", after)
    save_checkpoint(model, output / "checkpoint")
    if [row["record_id"] for row in before["rows"]] != [
        row["record_id"] for row in after["rows"]
    ]:
        raise RuntimeError("paired evaluation request order changed during training")
    accepted_deltas = [
        right["accepted"] - left["accepted"]
        for left, right in zip(before["rows"], after["rows"])
    ]
    conditioned_deltas = [
        right["target_conditioned_accepted"] - left["target_conditioned_accepted"]
        for left, right in zip(before["rows"], after["rows"])
        if right["target_conditioned_accepted"] is not None
        and left["target_conditioned_accepted"] is not None
    ]
    summary = {
        "before_mean_accepted": before["mean_accepted"],
        "after_mean_accepted": after["mean_accepted"],
        "accepted_delta": after["mean_accepted"] - before["mean_accepted"],
        "accepted_delta_ci95_request_bootstrap": bootstrap_delta(accepted_deltas),
        "better_same_worse_requests": [
            sum(delta > 0 for delta in accepted_deltas),
            sum(delta == 0 for delta in accepted_deltas),
            sum(delta < 0 for delta in accepted_deltas),
        ],
        "before_mean_target_conditioned_accepted": before[
            "mean_target_conditioned_accepted"
        ],
        "after_mean_target_conditioned_accepted": after[
            "mean_target_conditioned_accepted"
        ],
        "target_conditioned_accepted_delta": (
            sum(conditioned_deltas) / len(conditioned_deltas)
            if conditioned_deltas
            else None
        ),
        "target_conditioned_delta_ci95_request_bootstrap": (
            bootstrap_delta(conditioned_deltas) if conditioned_deltas else None
        ),
        "steps_logged": len(losses),
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "mean_last_50_loss": sum(losses[-50:]) / min(50, len(losses)),
        "elapsed_s": time.perf_counter() - started,
    }
    write_json(output / "summary.json", summary)
    manifest.update(status="completed", summary=summary)
    write_json(output / "manifest.json", manifest)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
