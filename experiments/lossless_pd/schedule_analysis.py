"""Calibrated lossless P/D schedule model from measured verifier traces.

The model is deliberately small: one serialized link, five drafter KV layers,
anchor-first transfer, and the exact layer dependency recurrence.  It does not
claim a deployed network measurement.  Its purpose is to select bandwidth,
visibility, and block-length cells for the next integrated experiment.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
from pathlib import Path


TARGET_LAYERS = 36
DRAFT_LAYERS = {1, 9, 17, 25, 33}


def read(path):
    return json.loads(Path(path).read_text())


def bootstrap(values, seed=20260909, repeats=5000):
    rng = random.Random(seed)
    means = sorted(statistics.mean(rng.choices(values, k=len(values)))
                   for _ in range(repeats))
    return [means[int(.025 * repeats)], means[int(.975 * repeats)]]


def average_traces(document):
    grouped = {}
    for row in document["rows"]:
        grouped.setdefault(row["record_id"], []).append(row)
    result = {}
    for record_id, rows in grouped.items():
        if not all(row["serial_streamed_bitwise_equal"] for row in rows):
            raise RuntimeError(f"non-bitwise verifier trace: {record_id}")
        costs = [statistics.mean(row["serial"]["layers"][layer]["cost"]
                                 for row in rows)
                 for layer in range(TARGET_LAYERS)]
        payload = rows[0]["serial"]["payload_bytes"]
        terminal = statistics.mean(
            row["serial"]["wall_ms"]
            - row["serial"]["release_ms"][-1]
            - sum(layer["cost"] for layer in row["serial"]["layers"])
            for row in rows
        )
        result[record_id] = {
            "costs": costs,
            "payload_bytes": payload,
            "terminal_ms": max(0.0, terminal),
        }
    return result


def switch_arrivals(payload_bytes, fraction, gbps, prefix_layers):
    """Prefix -> remaining anchors -> remaining layers one-switch schedule.

    Target layers before ``prefix_layers`` are transferred completely.  Their
    draft anchors, if any, are already satisfied.  Remaining draft anchors are
    sent next; then target-layer residuals continue in dependency order.  No
    byte is duplicated and the final layer is ready after exactly one payload.
    """

    if not 0 <= prefix_layers <= TARGET_LAYERS:
        raise ValueError("prefix layer count is outside the target")
    layer_bytes = payload_bytes / TARGET_LAYERS
    bytes_per_ms = gbps * 1e6 / 8
    ready = []
    now = 0.0
    for _ in range(prefix_layers):
        now += layer_bytes / bytes_per_ms
        ready.append(now)
    remaining_anchors = sum(layer >= prefix_layers for layer in DRAFT_LAYERS)
    now += remaining_anchors * fraction * layer_bytes / bytes_per_ms
    anchor_ready = now
    for layer in range(prefix_layers, TARGET_LAYERS):
        residual = layer_bytes * (1 - fraction if layer in DRAFT_LAYERS else 1)
        now += residual / bytes_per_ms
        ready.append(now)
    return ready, anchor_ready


def arrivals(payload_bytes, fraction, gbps, anchor_first):
    """Compatibility wrapper for pure layer-order or anchor-first schedules."""

    if anchor_first:
        return switch_arrivals(payload_bytes, fraction, gbps, 0)
    layer_bytes = payload_bytes / TARGET_LAYERS
    bytes_per_ms = gbps * 1e6 / 8
    return [(layer + 1) * layer_bytes / bytes_per_ms
            for layer in range(TARGET_LAYERS)], 0.0


def finish(ready, costs, initial, terminal):
    current = initial
    for release, cost in zip(ready, costs):
        current = max(current, release) + cost
    return current + terminal


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--draft-eval", required=True)
    parser.add_argument("--g1", required=True,
                        help="results.json for seed-only one-token verifier")
    parser.add_argument("--block", action="append", required=True,
                        help="PROPOSALS:results.json, e.g. 3:path")
    parser.add_argument("--output", required=True)
    parser.add_argument("--fractions", default="0.05,0.1,0.2")
    parser.add_argument("--gbps", default="10,25,50,100,200,400")
    args = parser.parse_args()

    baseline_doc = read(args.g1)
    if not baseline_doc["summary"]["serial_streamed_bitwise_equal"]:
        raise RuntimeError("one-token baseline is not bitwise stable")
    backend = baseline_doc["summary"].get("verification_attention", "unknown")
    baseline = average_traces(baseline_doc)
    blocks = {}
    for item in args.block:
        proposals, path = item.split(":", 1)
        document = read(path)
        if document["summary"].get("verification_attention", "unknown") != backend:
            raise RuntimeError("verifier traces use different attention backends")
        if document["summary"]["proposal_tokens"] != int(proposals):
            raise RuntimeError("block trace proposal length is mislabeled")
        blocks[int(proposals)] = average_traces(document)

    draft = read(args.draft_eval)
    # The first timed cell includes compilation/warmup. Use a per-cell median
    # as a calibrated constant and retain it in the output.
    draft_cells = {}
    for row in draft["rows"]:
        if row["order"] == "priority" and row["memory_mode"] == "exact":
            draft_cells.setdefault(float(row["fraction"]), []).append(row)
    fractions = [float(value) for value in args.fractions.split(",")]
    bandwidths = [float(value) for value in args.gbps.split(",")]
    rows = []
    for fraction in fractions:
        cell = draft_cells[fraction]
        draft_ms = statistics.median(
            row["draft_ms_unfused_with_projection"] for row in cell
        )
        proposals_by_request = {row["record_id"]: row for row in cell}
        for g, block in sorted(blocks.items()):
            for gbps in bandwidths:
                deltas, spec_times, base_times, first_times, progresses = [], [], [], [], []
                for record_id, proposal in proposals_by_request.items():
                    if record_id not in baseline or record_id not in block:
                        continue
                    one = baseline[record_id]
                    many = block[record_id]
                    if one["payload_bytes"] != many["payload_bytes"]:
                        raise RuntimeError("verifier payload changed across block sizes")
                    progress = min(int(proposal["accepted"]), g) + 1
                    base_ready, _ = arrivals(one["payload_bytes"], 0, gbps, False)
                    base_first = finish(base_ready, one["costs"], 0, one["terminal_ms"])
                    # All cache is ready once the first target token crosses
                    # the last layer. Later tokens therefore pay pure decode.
                    base = base_first + (progress - 1) * (
                        sum(one["costs"]) + one["terminal_ms"]
                    )
                    alternatives = []
                    for prefix_layers in range(TARGET_LAYERS + 1):
                        spec_ready, anchor_ready = switch_arrivals(
                            many["payload_bytes"], fraction, gbps, prefix_layers
                        )
                        spec_finish = finish(
                            spec_ready, many["costs"], anchor_ready + draft_ms,
                            many["terminal_ms"]
                        )
                        alternatives.append((spec_finish, prefix_layers))
                    spec, best_prefix_layers = min(alternatives)
                    deltas.append(base - spec)
                    spec_times.append(spec)
                    base_times.append(base)
                    first_times.append(base_first)
                    progresses.append(progress)
                    # The per-request switch point is retained before summary.
                    proposal.setdefault("_best_prefix_layers", {})[
                        (g, gbps)
                    ] = best_prefix_layers
                switch_points = [
                    proposal["_best_prefix_layers"][(g, gbps)]
                    for proposal in proposals_by_request.values()
                    if (g, gbps) in proposal.get("_best_prefix_layers", {})
                ]
                rows.append({
                    "fraction": fraction,
                    "proposal_tokens": g,
                    "gbps_decimal_bits_per_second": gbps,
                    "requests": len(deltas),
                    "median_measured_draft_ms": draft_ms,
                    "mean_exact_output_tokens": statistics.mean(progresses),
                    "mean_layer_ready_no_draft_ms_to_same_progress": statistics.mean(base_times),
                    "mean_layer_ready_no_draft_first_token_ms": statistics.mean(first_times),
                    "mean_anchor_draft_streamed_verify_ms": statistics.mean(spec_times),
                    "mean_first_commit_delta_ms_spec_minus_no_draft": statistics.mean(
                        spec - first for spec, first in zip(spec_times, first_times)
                    ),
                    "first_commit_delta_ci95_request_bootstrap_ms": bootstrap(
                        [spec - first for spec, first in zip(spec_times, first_times)]
                    ),
                    "mean_saving_ms": statistics.mean(deltas),
                    "saving_ci95_request_bootstrap_ms": bootstrap(deltas),
                    "fraction_faster": sum(value > 0 for value in deltas) / len(deltas),
                    "median_optimal_prefix_layers_before_anchors": statistics.median(
                        switch_points
                    ),
                    "anchor_first_optimal_fraction": sum(value == 0 for value in switch_points)
                    / len(switch_points),
                })
    result = {
        "contract": (
            "trace-calibrated single-link model, not a network benchmark; exact layer "
            "recurrence; O(L) enumeration of prefix->anchors->residual switch points; "
            "anchor bytes are part of rather than additional to full KV; "
            "uncontended draft compute; same-progress comparison includes verifier correction"
        ),
        "attention_backend": backend,
        "target_layers": TARGET_LAYERS,
        "draft_kv_layers": sorted(DRAFT_LAYERS),
        "rows": rows,
    }
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"output": args.output, "best": sorted(
        rows, key=lambda row: row["mean_saving_ms"], reverse=True
    )[:8]}, indent=2))


if __name__ == "__main__":
    main()
