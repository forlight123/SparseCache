"""Real sparse-draft -> paced H2D -> layer-ready exact-verify pilot.

The rate limiter represents a serialized P/D link, while every payload still
performs a real pinned-CPU to H200 copy.  This is a one-host deployment proxy,
not NIC/RDMA evidence. Sparse anchor pages land in their authoritative target
slots and are excluded from the later residual transfer, so bytes are not duplicated.
"""

from __future__ import annotations

import argparse
import json
import math
import queue
import statistics
import threading
import time
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path

import torch

from experiments.blockdraft.model import BlockKVDraft
from experiments.lossless_pd.core import ProgressiveBlock, page_order, visible_positions
from experiments.lossless_pd.pilot import bootstrap_delta, load_target, write_json
from experiments.lossless_pd.reference_packets import load_packet, read_index
from experiments.lossless_pd.sequence_equivalence import greedy_commit
from experiments.lossless_pd.shape_invariant_attention import (
    register_shape_invariant_attention,
    row_invariant_verifier_ops,
)
from experiments.lossless_pd.verifier_probe import cache_with_buffers


LAYER_IDS = [1, 9, 17, 25, 33]


def load_drafter(checkpoint):
    checkpoint = Path(checkpoint)
    model = ProgressiveBlock(BlockKVDraft.load_checkpoint(checkpoint))
    stage = checkpoint / "stage.pt"
    if stage.exists():
        model.stage.load_state_dict(torch.load(stage, map_location="cpu", weights_only=True))
    return model.to("cuda", dtype=torch.bfloat16).eval()


def wait_until(deadline):
    while True:
        remaining = deadline - time.perf_counter()
        if remaining <= 0:
            return
        if remaining > .0003:
            time.sleep(remaining - .00015)


def producer(copy_stream, origin, started, gbps, transfers, output_queue, scatter=None):
    """Pace serialized wire completion and publish guarded CUDA events."""

    bytes_per_second = gbps * 1e9 / 8
    cumulative = 0
    try:
        for name, destinations, sources in transfers:
            wait_until(started + cumulative / bytes_per_second)
            with torch.cuda.stream(copy_stream):
                copy_stream.wait_event(origin)
                for destination, source in zip(destinations, sources):
                    destination.copy_(source, non_blocking=True)
                ready = torch.cuda.Event(enable_timing=True)
                ready.record(copy_stream)
                if name == "anchor" and scatter is not None:
                    target_buffers, anchor_buffers, positions = scatter
                    for sampled, layer_id in enumerate(LAYER_IDS):
                        target_buffers[layer_id][0].index_copy_(
                            -2, positions, anchor_buffers[0][:, sampled]
                        )
                        target_buffers[layer_id][1].index_copy_(
                            -2, positions, anchor_buffers[1][:, sampled]
                        )
            payload = sum(source.numel() * source.element_size() for source in sources)
            cumulative += payload
            # Publish logical readiness only after the last bit could have
            # crossed the emulated wire. The CUDA event separately guards a
            # real H2D copy that is slower than the configured link.
            wait_until(started + cumulative / bytes_per_second)
            output_queue.put((name, ready, payload, None))
    except BaseException as error:  # propagate instead of deadlocking the consumer
        output_queue.put(("error", None, 0, error))


def next_event(output_queue, expected):
    name, event, payload, error = output_queue.get()
    if error is not None:
        raise error
    if name != expected:
        raise RuntimeError(f"transfer order mismatch: expected {expected}, got {name}")
    return event, payload


def target_block(target, block, cache, output_queue, bitwise_verifier=False):
    """Run exact target layers, waiting only for each layer's full prompt KV."""

    n = cache.layers[0].keys.shape[-2]
    length = block.shape[1]
    hidden = target.model.embed_tokens(block)
    positions = torch.arange(n, n + length, device="cuda")
    rotary = target.model.rotary_emb(hidden, positions[None])
    allowed = torch.arange(n + length, device="cuda")[None, :] <= positions[:, None]
    mask = torch.zeros(1, 1, length, n + length, device="cuda", dtype=hidden.dtype)
    mask.masked_fill_(~allowed, torch.finfo(hidden.dtype).min)
    compute = torch.cuda.current_stream()
    ready_events = []
    bytes_received = 0
    exact_ops = row_invariant_verifier_ops(target) if bitwise_verifier else nullcontext()
    with exact_ops:
        for layer_id, layer in enumerate(target.model.layers):
            ready, payload = next_event(output_queue, f"layer{layer_id}")
            ready_events.append(ready)
            bytes_received += payload
            compute.wait_event(ready)
            hidden = layer(hidden, attention_mask=mask, position_ids=positions[None],
                           past_key_values=cache, use_cache=True, cache_position=positions,
                           position_embeddings=rotary)
        logits = target.lm_head(target.model.norm(hidden)).float()
    return logits, cache, ready_events, bytes_received


def contiguous_ranges(positions, length, selected):
    chosen = set(int(value) for value in positions.cpu().tolist())
    flags = [(index in chosen) == selected for index in range(length)]
    ranges, start = [], None
    for index, include in enumerate(flags + [False]):
        if include and start is None:
            start = index
        elif not include and start is not None:
            ranges.append((start, index))
            start = None
    return ranges


def sliced_pairs(destination, source, ranges):
    destinations, sources = [], []
    for begin, end in ranges:
        for target_tensor, host_tensor in zip(destination, source):
            destinations.append(target_tensor[..., begin:end, :])
            sources.append(host_tensor[..., begin:end, :])
    return tuple(destinations), tuple(sources)


@torch.no_grad()
def run_condition(target, drafter, host, anchor, seed, prompt_tokens, positions,
                   gbps, proposal_tokens, condition, verifier_semantics="eager"):
    target_buffers = [(torch.empty_like(key, device="cuda"),
                       torch.empty_like(value, device="cuda")) for key, value in host]
    cache = cache_with_buffers(target, target_buffers)
    copy_stream = torch.cuda.Stream()
    compute = torch.cuda.current_stream()
    origin = torch.cuda.Event(enable_timing=True)
    events = queue.Queue()
    transfers = []
    anchor_buffers = None
    scatter = None
    if condition == "speculative":
        anchor_buffers = (torch.empty_like(anchor[0], device="cuda"),
                          torch.empty_like(anchor[1], device="cuda"))
        transfers.append(("anchor", anchor_buffers, anchor))
        missing_ranges = contiguous_ranges(positions, prompt_tokens, False)
        scatter = (target_buffers, anchor_buffers, positions)
        for layer_id in range(len(host)):
            if layer_id in LAYER_IDS:
                destinations, sources = sliced_pairs(
                    target_buffers[layer_id], host[layer_id], missing_ranges
                )
            else:
                destinations, sources = target_buffers[layer_id], host[layer_id]
            transfers.append((f"layer{layer_id}", destinations, sources))
    else:
        transfers.extend((f"layer{layer_id}", target_buffers[layer_id], host[layer_id])
                         for layer_id in range(len(host)))
    torch.cuda.synchronize()
    origin.record(compute)
    started = time.perf_counter()
    worker = threading.Thread(target=producer, args=(
        copy_stream, origin, started, gbps, transfers, events, scatter
    ), daemon=True)
    worker.start()
    bytes_sent = 0
    draft_ms = 0.0
    if condition == "speculative":
        anchor_ready, payload = next_event(events, "anchor")
        bytes_sent += payload
        compute.wait_event(anchor_ready)
        draft_start = torch.cuda.Event(enable_timing=True)
        draft_end = torch.cuda.Event(enable_timing=True)
        draft_start.record(compute)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            proposals = drafter.propose(
                seed, target.model.embed_tokens, target.lm_head,
                anchor_buffers[0], anchor_buffers[1], positions, prompt_tokens,
                length=proposal_tokens,
            )
            proposals = proposals[0]
        draft_end.record(compute)
        block = torch.cat((seed, proposals[None]), dim=1)
    else:
        proposals = None
        block = seed
    logits, cache, layer_events, target_bytes = target_block(
        target, block, cache, events,
        bitwise_verifier=condition == "speculative" and verifier_semantics == "bitwise",
    )
    bytes_sent += target_bytes
    torch.cuda.synchronize()
    first_commit_ms = (time.perf_counter() - started) * 1000
    worker.join()
    releases = [origin.elapsed_time(event) for event in layer_events]
    if condition == "speculative":
        draft_ms = draft_start.elapsed_time(draft_end)
        committed, accepted = greedy_commit(proposals.tolist(), logits.argmax(-1)[0].tolist())
        same_progress_ms = first_commit_ms
    else:
        accepted = None
        token = logits[:, -1].argmax(-1, keepdim=True)
        committed = [int(token.item())]
        same_progress_ms = first_commit_ms
    return {
        "condition": condition,
        "first_commit_ms": first_commit_ms,
        "same_progress_ms": same_progress_ms,
        "committed": committed,
        "accepted": accepted,
        "draft_ms": draft_ms,
        "bytes_sent": bytes_sent,
        "layer_release_ms": releases,
        "cache": cache,
        "next_token": None if condition == "speculative" else token,
    }


@torch.no_grad()
def extend_baseline(target, result, progress):
    """Finish the baseline to the speculative condition's exact token count."""

    token = result.pop("next_token")
    cache = result.pop("cache")
    for _ in range(progress - 1):
        output = target.model(token, past_key_values=cache, use_cache=True)
        token = target.lm_head(output.last_hidden_state[:, -1:]).argmax(-1)
        result["committed"].append(int(token.item()))
        cache = output.past_key_values
    torch.cuda.synchronize()
    # ``first_commit_ms`` was measured from a condition-local origin. Extend
    # it by the elapsed decode tail, measured separately to avoid retaining
    # a wall-clock start across the caller's bookkeeping.
    return result


def clustered_values(rows, value):
    grouped = {}
    for row in rows:
        grouped.setdefault(row["record_id"], []).append(value(row))
    return [statistics.mean(values) for values in grouped.values()]


def summarize(rows, args):
    baseline_same = clustered_values(
        rows, lambda row: row["baseline"]["same_progress_ms"]
    )
    speculative_same = clustered_values(
        rows, lambda row: row["speculative"]["same_progress_ms"]
    )
    deltas = clustered_values(
        rows, lambda row: row["baseline"]["same_progress_ms"]
        - row["speculative"]["same_progress_ms"]
    )
    first = clustered_values(
        rows, lambda row: row["speculative"]["first_commit_ms"]
        - row["baseline"]["first_commit_ms"]
    )
    baseline_first = clustered_values(
        rows, lambda row: row["baseline"]["first_commit_ms"]
    )
    speculative_first = clustered_values(
        rows, lambda row: row["speculative"]["first_commit_ms"]
    )
    mismatch_records = {row["record_id"] for row in rows
                        if not row["committed_output_equal"]}
    return {
        "requests": len({row["record_id"] for row in rows}),
        "paired_runs": len(rows),
        "gbps_decimal_bits_per_second": args.gbps,
        "fraction": args.fraction,
        "order": args.order,
        "proposal_tokens": args.proposal_tokens,
        "verifier_semantics": args.verifier_semantics,
        "mean_progress_tokens": statistics.mean(row["progress_tokens"] for row in rows),
        "mean_baseline_same_progress_ms": statistics.mean(baseline_same),
        "mean_speculative_same_progress_ms": statistics.mean(speculative_same),
        "mean_same_progress_saving_ms": statistics.mean(deltas),
        "same_progress_saving_ci95_request_bootstrap_ms": bootstrap_delta(deltas),
        "mean_first_commit_delta_spec_minus_baseline_ms": statistics.mean(first),
        "mean_baseline_first_commit_ms": statistics.mean(baseline_first),
        "mean_speculative_first_commit_ms": statistics.mean(speculative_first),
        "first_commit_delta_ci95_request_bootstrap_ms": bootstrap_delta(first),
        "mean_draft_ms": statistics.mean(row["speculative"]["draft_ms"] for row in rows),
        "fraction_requests_faster_at_same_progress": sum(value > 0 for value in deltas)
        / len(deltas),
        "committed_output_mismatch_runs": sum(not row["committed_output_equal"] for row in rows),
        "committed_output_mismatch_requests": len(mismatch_records),
        "committed_output_mismatch_record_ids": sorted(mismatch_records),
        "mean_baseline_bytes": statistics.mean(row["baseline"]["bytes_sent"] for row in rows),
        "mean_speculative_bytes": statistics.mean(row["speculative"]["bytes_sent"] for row in rows),
        "contract": (
            "one-host paced-link proxy plus real pinned H2D; actual sparse drafter; "
            f"actual all-layer {args.verifier_semantics} verifier; anchor bytes reused in full "
            "target cache; P prefill, "
            "host pinning, allocations, and NIC/RDMA excluded"
        ),
    }


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--packets", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--target", default="/data/models/qwen/Qwen3-8B")
    parser.add_argument("--requests", type=int, default=16)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--fraction", type=float, default=.1)
    parser.add_argument("--order", choices=["priority", "random", "reverse_priority"],
                        default="priority")
    parser.add_argument("--proposal-tokens", type=int, default=15)
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
    write_json(output / "manifest.json", {
        "arguments": {
            **vars(args),
            "packets": str(packet_root),
            "checkpoint": str(Path(args.checkpoint).resolve()),
            "output": str(output.resolve()),
            "target": str(Path(args.target).resolve()),
        },
        "drafter_config": asdict(drafter.base.config),
        "measurement_contract": (
            "one-host paced-link proxy with real pinned H2D; run timing cells "
            "serially because concurrent GPUs share host-memory and PCIe resources"
        ),
    })
    rows = []
    for request_index, entry in enumerate(index["entries"][:args.requests]):
        packet = load_packet(packet_root, entry)
        ids = packet["prompt_ids"].to("cuda")
        target.config._attn_implementation = "sdpa"
        prompt = target.model(ids, use_cache=True)
        seed = target.lm_head(prompt.last_hidden_state[:, -1:]).argmax(-1)
        if int(seed.item()) != int(packet["seed"].item()):
            raise RuntimeError("fresh P seed differs from immutable packet")
        host = [(layer.keys.cpu().pin_memory(), layer.values.cpu().pin_memory())
                for layer in prompt.past_key_values.layers]
        scores = packet["priority_scores"].to("cuda")
        order = page_order(ids.shape[1], index["page_size"], args.order,
                           index["seed"] + request_index, scores)
        positions = visible_positions(ids.shape[1], index["page_size"], args.fraction,
                                      order, torch.device("cuda"))
        anchor_keys = torch.stack([host[layer][0] for layer in LAYER_IDS], dim=1)
        anchor_values = torch.stack([host[layer][1] for layer in LAYER_IDS], dim=1)
        cpu_positions = positions.cpu()
        anchor = (anchor_keys.index_select(-2, cpu_positions).pin_memory(),
                  anchor_values.index_select(-2, cpu_positions).pin_memory())
        del prompt, anchor_keys, anchor_values
        target.config._attn_implementation = (
            register_shape_invariant_attention()
            if args.verifier_semantics == "bitwise" else "eager"
        )
        # Warm only model kernels; timed runs always allocate fresh target cache.
        warm_k = anchor[0].to("cuda")
        warm_v = anchor[1].to("cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            drafter.propose(
                seed, target.model.embed_tokens, target.lm_head,
                warm_k, warm_v, positions, ids.shape[1], length=args.proposal_tokens,
            )
        torch.cuda.synchronize()
        del warm_k, warm_v
        for repeat in range(args.repeats):
            order_conditions = ("baseline", "speculative") if repeat % 2 == 0 else (
                "speculative", "baseline"
            )
            pair = {}
            for condition in order_conditions:
                pair[condition] = run_condition(
                    target, drafter, host, anchor, seed, ids.shape[1], positions,
                    args.gbps, args.proposal_tokens, condition, args.verifier_semantics
                )
            progress = len(pair["speculative"]["committed"])
            baseline = pair["baseline"]
            tail_started = time.perf_counter()
            extend_baseline(target, baseline, progress)
            baseline["same_progress_ms"] += (time.perf_counter() - tail_started) * 1000
            # Drop non-serializable authoritative state after the decode tail.
            baseline.pop("cache", None)
            baseline.pop("next_token", None)
            pair["speculative"].pop("cache", None)
            pair["speculative"].pop("next_token", None)
            row = {
                "record_id": packet["record_id"],
                "repeat": repeat,
                "context": ids.shape[1],
                "actual_fraction": positions.numel() / ids.shape[1],
                "proposal_tokens": args.proposal_tokens,
                "progress_tokens": progress,
                "committed_output_equal": baseline["committed"] == pair["speculative"]["committed"],
                **pair,
            }
            rows.append(row)
            write_json(output / "progress.json", {"rows": rows})
        print(json.dumps({"event": "integrated", "completed": request_index + 1,
                          "total": min(args.requests, len(index["entries"]))}), flush=True)
    summary = summarize(rows, args)
    write_json(output / "results.json", {"summary": summary, "rows": rows})
    write_json(output / "summary.json", summary)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
