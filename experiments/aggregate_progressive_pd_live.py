"""Fail-closed paper aggregation for the live progressive P/D experiment."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
from itertools import pairwise
from pathlib import Path
from typing import Any

try:
    from experiments.longbench_metrics import score_prediction
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from longbench_metrics import score_prediction

VALID_ARMS = ("baseline", "fixed_s1", "continuous")


def parse_token_ids(value: str) -> tuple[int, ...]:
    tokens = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if len(tokens) != len(set(tokens)) or any(token < 0 for token in tokens):
        raise argparse.ArgumentTypeError(
            "token IDs must be a unique comma-separated list of non-negative integers"
        )
    return tokens


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-jsonl", type=Path, required=True)
    parser.add_argument("--metadata-jsonl", type=Path, required=True)
    parser.add_argument("--scheduler-stats-jsonl", type=Path, required=True)
    parser.add_argument("--attention-stats-jsonl", type=Path, required=True)
    parser.add_argument("--link-stats-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-requests", type=int, required=True)
    parser.add_argument("--arms", default=",".join(VALID_ARMS))
    parser.add_argument("--fixed-output-tokens", type=int)
    parser.add_argument("--protected-prefix-tokens", type=int, default=0)
    parser.add_argument("--protected-suffix-tokens", type=int, default=0)
    parser.add_argument("--priority-jsonl", type=Path)
    parser.add_argument("--priority-chunk-tokens", type=int, default=256)
    parser.add_argument("--stop-token-ids", type=parse_token_ids, default=())
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260827)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise TypeError(f"{path}:{line_number} is not a JSON object")
            rows.append(row)
    if not rows:
        raise ValueError(f"no JSONL rows in {path}")
    return rows


def _priority_by_request(
    rows: list[dict[str, Any]] | None,
) -> dict[int, tuple[int, ...]] | None:
    """Validate an optional request-scoped full chunk permutation sidecar."""
    if rows is None:
        return None
    priorities = {}
    for row in rows:
        index = row.get("request_index")
        priority = row.get("priority_chunks")
        if (
            not isinstance(index, int)
            or isinstance(index, bool)
            or index < 0
            or index in priorities
        ):
            raise ValueError("priority request indices must be unique integers")
        if (
            not isinstance(priority, list)
            or not priority
            or any(
                not isinstance(chunk, int) or isinstance(chunk, bool) or chunk < 0
                for chunk in priority
            )
            or sorted(priority) != list(range(len(priority)))
        ):
            raise ValueError("priority_chunks must be a complete chunk permutation")
        priorities[index] = tuple(priority)
    return priorities


def percentile(values: list[float], quantile: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(float(value) for value in values)
    position = quantile * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def bootstrap_mean_ci(values: list[float], *, samples: int, seed: int) -> list[float]:
    if not values:
        return [math.nan, math.nan]
    rng = random.Random(seed)
    estimates = [
        statistics.fmean(rng.choices(values, k=len(values))) for _ in range(samples)
    ]
    return [percentile(estimates, 0.025), percentile(estimates, 0.975)]


def describe(
    values: list[float], *, bootstrap_samples: int, seed: int
) -> dict[str, Any]:
    numeric = [float(value) for value in values]
    return {
        "mean": statistics.fmean(numeric),
        "mean_bootstrap_ci95": bootstrap_mean_ci(
            numeric, samples=bootstrap_samples, seed=seed
        ),
        "p50": percentile(numeric, 0.50),
        "p95": percentile(numeric, 0.95),
        "p99": percentile(numeric, 0.99),
        "min": min(numeric),
        "max": max(numeric),
    }


def _gate(passed: bool, **details: Any) -> dict[str, Any]:
    return {"passed": bool(passed), **details}


def _request_id_matches(request_id: str, proxy_request_id: str) -> bool:
    return bool(proxy_request_id) and proxy_request_id in request_id


def _validate_scheduler_record(
    record: dict[str, Any], result: dict[str, Any]
) -> tuple[bool, list[str]]:
    errors = []
    arm = result["arm"]
    draft_ids = record.get("draft_token_ids")
    verified_ids = record.get("verified_output_token_ids")
    client_ids = result["token_ids"]
    if not isinstance(draft_ids, list) or not isinstance(verified_ids, list):
        return False, ["scheduler token IDs are not lists"]
    draft_tokens = record.get("draft_tokens")
    draft_completed = record.get("draft_completed_at_ns")
    accepted = record.get("accepted_tokens")
    maximum = record.get("max_draft_tokens")
    if draft_tokens != len(draft_ids):
        errors.append("draft_tokens does not match draft_token_ids")
    if not isinstance(draft_completed, list) or len(draft_completed) != len(draft_ids):
        errors.append("draft completion telemetry does not cover draft tokens")
    if not isinstance(accepted, int) or not 0 <= accepted <= len(draft_ids):
        errors.append("accepted_tokens is outside the draft prefix")
    if not isinstance(maximum, int) or not 0 <= len(draft_ids) <= maximum:
        errors.append("draft token count exceeds max_draft_tokens")
    if record.get("visibility_mode") != arm:
        errors.append("scheduler visibility_mode does not match arm")
    if record.get("seed_token_id") != result.get("seed_token_id"):
        errors.append("scheduler seed does not match paired producer seed")
    if not client_ids or client_ids[0] != record.get("seed_token_id"):
        errors.append("hidden producer seed is not the first committed token")
    if isinstance(accepted, int) and accepted >= 0:
        if client_ids[1 : 1 + accepted] != draft_ids[:accepted]:
            errors.append("accepted draft prefix does not match client output")
        if verified_ids[:accepted] != draft_ids[:accepted]:
            errors.append("verifier output does not retain the accepted prefix")
    if client_ids[1 : 1 + len(verified_ids)] != verified_ids:
        errors.append("verified batch is not the committed client prefix")

    started = record.get("started_at_ns")
    full = record.get("full_arrival_at_ns")
    verify_started = record.get("verify_started_at_ns")
    verify_completed = record.get("verify_completed_at_ns")
    if not all(
        isinstance(value, int)
        for value in (started, full, verify_started, verify_completed)
    ):
        errors.append("scheduler transition timestamps are incomplete")
    elif not started <= full <= verify_started <= verify_completed:
        errors.append("full KV did not arrive before immutable verification")
    last_draft = record.get("last_draft_at_ns")
    if last_draft is not None and (
        not isinstance(last_draft, int) or not started <= last_draft <= full
    ):
        errors.append("last draft timestamp is outside the transfer interval")
    if (
        isinstance(draft_completed, list)
        and all(isinstance(value, int) for value in draft_completed)
        and isinstance(started, int)
        and isinstance(full, int)
    ):
        if draft_completed != sorted(draft_completed):
            errors.append("draft completion timestamps are not monotonic")
        if draft_completed and (
            draft_completed[0] < started
            or draft_completed[-1] > full
            or draft_completed[-1] != last_draft
        ):
            errors.append("draft completion timestamps are outside the draft window")
    return not errors, errors


def _link_summary(
    rows: list[dict[str, Any]], *, arm: str
) -> tuple[dict[str, Any], list[str]]:
    errors = []
    ordered = sorted(rows, key=lambda row: int(row.get("bundle_index", -1)))
    expected_mode = "monolithic" if arm == "baseline" else "progressive_bundle"
    if any(row.get("mode") != expected_mode for row in ordered):
        errors.append("controlled-link mode does not match benchmark arm")
    if arm == "baseline" and len(ordered) != 1:
        errors.append("baseline must have exactly one monolithic link record")
    indices = [row.get("bundle_index") for row in ordered]
    if arm != "baseline" and indices != list(range(len(ordered))):
        errors.append("progressive bundle indices are not contiguous")
    prior_fraction = 0.0
    prior_reported = None
    for row in ordered:
        try:
            fraction = float(row["completed_fraction"])
            logical_tokens = int(row["logical_tokens"])
            bytes_per_token = float(row["bytes_per_token"])
            link_gbps = float(row["link_gbps"])
            submitted = float(row["submitted_at_s"])
            not_before = float(row["not_before_s"])
            reported = float(row["reported_at_s"])
            modeled_ms = float(row["modeled_wire_ms"])
        except (KeyError, TypeError, ValueError) as error:
            errors.append(f"invalid controlled-link record: {error}")
            continue
        if not 0.0 < fraction <= 1.0 or fraction < prior_fraction:
            errors.append("controlled-link completion fractions are not monotonic")
        prior_fraction = fraction
        if logical_tokens <= 0 or bytes_per_token <= 0 or link_gbps <= 0:
            errors.append("controlled-link byte/rate fields must be positive")
        if not submitted <= not_before <= reported + 1e-9:
            errors.append("retrieve was reported before its serialized deadline")
        expected_ms = logical_tokens * bytes_per_token * 8.0 / (link_gbps * 1e6)
        if not math.isclose(modeled_ms, expected_ms, rel_tol=1e-9, abs_tol=1e-6):
            errors.append("modeled wire time does not match logical payload")
        ranges = row.get("token_ranges")
        try:
            range_tokens = (
                sum(int(end) - int(start) for start, end in ranges)
                if isinstance(ranges, list)
                else -1
            )
        except (TypeError, ValueError):
            range_tokens = -1
        if range_tokens != logical_tokens:
            errors.append("controlled-link exact token ranges do not match token count")
        if row.get("succeeded") is not True:
            errors.append("controlled-link retrieve did not succeed")
        if prior_reported is not None and submitted + 1e-9 < prior_reported:
            errors.append("progressive bundles overlapped on the serialized wire")
        prior_reported = reported
    if ordered and float(ordered[-1].get("completed_fraction", 0.0)) != 1.0:
        errors.append("controlled-link trace does not end at full KV")
    total_tokens = sum(int(row.get("logical_tokens", 0)) for row in ordered)
    total_payload = sum(float(row.get("logical_payload_bytes", 0.0)) for row in ordered)
    total_wire_ms = sum(float(row.get("modeled_wire_ms", 0.0)) for row in ordered)
    first_bundle_wire_ms = (
        float(ordered[0].get("modeled_wire_ms", 0.0)) if ordered else 0.0
    )
    return (
        {
            "request_id": str(ordered[0].get("request_id", "")) if ordered else "",
            "mode": expected_mode,
            "bundles": len(ordered),
            "logical_tokens": total_tokens,
            "logical_payload_bytes": total_payload,
            "link_gbps": float(ordered[0].get("link_gbps", 0.0)) if ordered else 0.0,
            "bytes_per_token": float(ordered[0].get("bytes_per_token", 0.0))
            if ordered
            else 0.0,
            "modeled_wire_ms": total_wire_ms,
            "first_bundle_modeled_wire_ms": first_bundle_wire_ms,
            "residual_modeled_wire_ms": total_wire_ms - first_bundle_wire_ms,
            "observed_chain_ms": (
                float(ordered[-1]["reported_at_s"])
                - float(ordered[0]["submitted_at_s"])
            )
            * 1000
            if ordered
            else 0.0,
            "completed_fractions": [
                float(row.get("completed_fraction", 0.0)) for row in ordered
            ],
            "first_bundle_token_ranges": ordered[0].get("token_ranges", [])
            if ordered
            else [],
            "bundle_token_ranges": [row.get("token_ranges", []) for row in ordered],
            "retrieve_start_token": min(
                (
                    int(start)
                    for row in ordered
                    for start, _ in (row.get("token_ranges") or [])
                ),
                default=0,
            ),
            "retrieve_end_token": max(
                (
                    int(end)
                    for row in ordered
                    for _, end in (row.get("token_ranges") or [])
                ),
                default=0,
            ),
        },
        errors,
    )


def _ranges_cover(
    ranges: list[list[int]] | list[tuple[int, int]], start: int, end: int
) -> bool:
    """Return whether half-open ranges completely cover one target interval."""
    if end <= start:
        return True
    cursor = start
    for lower, upper in sorted((int(a), int(b)) for a, b in ranges):
        if upper <= cursor:
            continue
        if lower > cursor:
            return False
        cursor = max(cursor, upper)
        if cursor >= end:
            return True
    return False


def _observed_bundle_chunks(
    link_summary: dict[str, Any], chunk_tokens: int
) -> tuple[list[set[int]], list[str]]:
    """Recover logical chunk sets from coalesced exact-range link records."""
    errors = []
    retrieve_start = int(link_summary["retrieve_start_token"])
    retrieve_end = int(link_summary["retrieve_end_token"])
    if chunk_tokens <= 0:
        raise ValueError("priority chunk size must be positive")
    if (retrieve_end - retrieve_start) % chunk_tokens:
        return [], ["retrieve range is not aligned to the priority chunk size"]
    bundles = []
    for bundle_index, ranges in enumerate(link_summary["bundle_token_ranges"]):
        chunks = set()
        if not isinstance(ranges, list):
            errors.append(f"bundle {bundle_index} token ranges are not a list")
            bundles.append(chunks)
            continue
        for raw_range in ranges:
            if not isinstance(raw_range, list | tuple) or len(raw_range) != 2:
                errors.append(f"bundle {bundle_index} has an invalid token range")
                continue
            start, end = map(int, raw_range)
            if (
                start < retrieve_start
                or end > retrieve_end
                or end <= start
                or (start - retrieve_start) % chunk_tokens
                or (end - retrieve_start) % chunk_tokens
            ):
                errors.append(f"bundle {bundle_index} has a misaligned priority range")
                continue
            first = (start - retrieve_start) // chunk_tokens
            stop = (end - retrieve_start) // chunk_tokens
            selected = set(range(first, stop))
            if chunks & selected:
                errors.append(f"bundle {bundle_index} contains duplicate chunks")
            chunks.update(selected)
        bundles.append(chunks)
    return bundles, errors


def _validate_priority_schedule(
    link_summary: dict[str, Any],
    expected_priority: tuple[int, ...],
    chunk_tokens: int,
) -> list[str]:
    """Prove that every serialized link bundle follows its request sidecar."""
    bundles, errors = _observed_bundle_chunks(link_summary, chunk_tokens)
    if errors:
        return errors
    expected_chunks = set(range(len(expected_priority)))
    observed_chunks = set().union(*bundles) if bundles else set()
    if observed_chunks != expected_chunks:
        errors.append("link bundles do not cover the sidecar chunk domain exactly")
        return errors
    cursor = 0
    for bundle_index, observed in enumerate(bundles):
        expected = set(expected_priority[cursor : cursor + len(observed)])
        if observed != expected:
            errors.append(
                f"bundle {bundle_index} does not match its priority sidecar slice"
            )
        cursor += len(observed)
    if cursor != len(expected_priority):
        errors.append("link bundle sizes do not consume the complete priority sidecar")
    return errors


def _attention_summary(
    rows: list[dict[str, Any]], *, scheduler: dict[str, Any], arm: str
) -> tuple[dict[str, Any], list[str]]:
    errors = []
    expected_request_id = scheduler["request_id"]
    if any(row.get("request_id") != expected_request_id for row in rows):
        errors.append("attention request_id does not match scheduler request_id")
    if any(row.get("visibility_mode") != arm for row in rows):
        errors.append("attention visibility_mode does not match arm")

    steps: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        step = row.get("decode_step")
        if not isinstance(step, int) or step < 0:
            errors.append("attention decode_step is invalid")
            continue
        steps.setdefault(step, []).append(row)
    expected_steps = set(range(int(scheduler["draft_tokens"])))
    if set(steps) != expected_steps:
        errors.append("attention steps do not exactly cover scheduler draft tokens")

    step_summaries = []
    prior_ranges: set[tuple[int, int]] = set()
    prior_fraction = 0.0
    prior_visible_pages = 0
    fixed_partial_signatures = set()
    for step, layer_rows in sorted(steps.items()):
        signatures = {
            (
                float(row["visible_fraction"]),
                json.dumps(row.get("visible_token_ranges"), sort_keys=True),
                int(row["candidate_pages"]),
                int(row["visible_pages"]),
            )
            for row in layer_rows
        }
        if len(signatures) != 1:
            errors.append(f"layers disagree on the visibility snapshot at step {step}")
        fraction, raw_ranges, candidate_pages, visible_pages = next(iter(signatures))
        ranges = {tuple(item) for item in json.loads(raw_ranges)}
        if not 0.0 < fraction <= 1.0:
            errors.append(f"visible fraction is invalid at step {step}")
        if not 0 <= visible_pages <= candidate_pages or candidate_pages <= 0:
            errors.append(f"visible page count is invalid at step {step}")
        if arm == "continuous":
            if fraction < prior_fraction or visible_pages < prior_visible_pages:
                errors.append(f"continuous visibility regressed at step {step}")
            if not prior_ranges.issubset(ranges):
                errors.append(f"continuous exact token ranges regressed at step {step}")
        elif fraction < 1.0:
            fixed_partial_signatures.add((fraction, raw_ranges, visible_pages))
        prior_fraction = fraction
        prior_visible_pages = visible_pages
        prior_ranges = ranges
        step_summaries.append(
            {
                "decode_step": step,
                "layers": len(layer_rows),
                "visible_fraction": fraction,
                "candidate_pages": candidate_pages,
                "visible_pages": visible_pages,
                "sparse_page_fraction": visible_pages / candidate_pages,
            }
        )
    if arm == "fixed_s1" and len(fixed_partial_signatures) > 1:
        errors.append("fixed-S1 changed its partial visibility snapshot")
    sparse_steps = sum(
        item["visible_pages"] < item["candidate_pages"] for item in step_summaries
    )
    distinct_fractions = len({item["visible_fraction"] for item in step_summaries})
    return (
        {
            "steps": len(step_summaries),
            "layer_rows": len(rows),
            "sparse_steps": sparse_steps,
            "sparse_step_rate": sparse_steps / len(step_summaries)
            if step_summaries
            else 0.0,
            "distinct_visibility_fractions": distinct_fractions,
            "first_visible_fraction": step_summaries[0]["visible_fraction"]
            if step_summaries
            else None,
            "last_visible_fraction": step_summaries[-1]["visible_fraction"]
            if step_summaries
            else None,
            "step_summaries": step_summaries,
        },
        errors,
    )


def _validate_token_arrivals(result: dict[str, Any]) -> list[str]:
    arrivals = result.get("token_arrival_ms")
    token_ids = result.get("token_ids")
    if not isinstance(arrivals, list) or not isinstance(token_ids, list):
        return ["token arrival telemetry is not a list"]
    if len(arrivals) != len(token_ids):
        return ["token arrival telemetry does not cover returned token IDs"]
    if any(
        not isinstance(value, int | float)
        or isinstance(value, bool)
        or not math.isfinite(value)
        or value < 0
        for value in arrivals
    ):
        return ["token arrival telemetry contains an invalid timestamp"]
    if arrivals != sorted(arrivals):
        return ["token arrival telemetry is not monotonic"]
    return []


def _request_latency_model(
    pair: dict[str, Any], method_arm: str
) -> tuple[dict[str, float] | None, list[str]]:
    """Predict one method's gain from independently observed critical-path terms."""
    baseline = pair["arms"].get("baseline")
    method = pair["arms"].get(method_arm)
    if baseline is None or method is None:
        return None, ["latency model requires baseline and method arms"]
    scheduler = method.get("scheduler")
    link = method.get("link")
    if not isinstance(scheduler, dict) or not isinstance(link, dict):
        return None, ["latency model requires scheduler and link telemetry"]

    arrivals = baseline.get("token_arrival_ms")
    if not isinstance(arrivals, list):
        return None, ["latency model requires baseline token arrival telemetry"]
    positive_intervals = [
        float(right) - float(left)
        for left, right in pairwise(arrivals)
        if float(right) > float(left)
    ]
    if not positive_intervals:
        return None, ["latency model has no positive baseline decode intervals"]

    draft_completed = scheduler.get("draft_completed_at_ns")
    started = scheduler.get("started_at_ns")
    verify_ns = scheduler.get("verify_ns")
    accepted = scheduler.get("accepted_tokens")
    if (
        not isinstance(draft_completed, list)
        or not all(isinstance(value, int) for value in draft_completed)
        or not isinstance(started, int)
        or not isinstance(verify_ns, int | float)
        or not isinstance(accepted, int)
    ):
        return None, ["latency model scheduler timing fields are incomplete"]

    target_step_ms = statistics.fmean(positive_intervals)
    draft_span_ms = (draft_completed[-1] - started) / 1e6 if draft_completed else 0.0
    residual_wire_ms = float(link.get("residual_modeled_wire_ms", math.nan))
    verify_ms = float(verify_ns) / 1e6
    if not all(
        math.isfinite(value) and value >= 0
        for value in (target_step_ms, draft_span_ms, residual_wire_ms, verify_ms)
    ):
        return None, ["latency model contains an invalid duration"]

    accepted_decode_value_ms = accepted * target_step_ms
    exposed_draft_ms = max(0.0, draft_span_ms - residual_wire_ms)
    predicted_gain_ms = accepted_decode_value_ms - verify_ms - exposed_draft_ms
    observed_gain_ms = float(baseline["decode_completion_ms"]) - float(
        method["decode_completion_ms"]
    )
    prediction_error_ms = observed_gain_ms - predicted_gain_ms
    return (
        {
            "target_step_ms": target_step_ms,
            "target_step_intervals": float(len(positive_intervals)),
            "accepted_tokens": float(accepted),
            "accepted_decode_value_ms": accepted_decode_value_ms,
            "verify_ms": verify_ms,
            "sparse_draft_span_ms": draft_span_ms,
            "residual_modeled_wire_ms": residual_wire_ms,
            "exposed_draft_ms": exposed_draft_ms,
            "predicted_gain_ms": predicted_gain_ms,
            "observed_gain_ms": observed_gain_ms,
            "prediction_error_ms": prediction_error_ms,
            "absolute_prediction_error_ms": abs(prediction_error_ms),
            "absolute_error_over_observed_gain": abs(prediction_error_ms)
            / max(abs(observed_gain_ms), 1.0),
            "gain_sign_matches": float(
                (predicted_gain_ms > 0) == (observed_gain_ms > 0)
            ),
        },
        [],
    )


def _pair_comparison(
    pairs: list[dict[str, Any]],
    left: str,
    right: str,
    *,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    left_rows = [pair["arms"][left] for pair in pairs]
    right_rows = [pair["arms"][right] for pair in pairs]
    completion_gain = [
        left_row["decode_completion_ms"] - right_row["decode_completion_ms"]
        for left_row, right_row in zip(left_rows, right_rows, strict=True)
    ]
    ttft_gain = [
        left_row["decode_ttft_ms"] - right_row["decode_ttft_ms"]
        for left_row, right_row in zip(left_rows, right_rows, strict=True)
    ]
    quality_delta = [
        right_row["quality_score"] - left_row["quality_score"]
        for left_row, right_row in zip(left_rows, right_rows, strict=True)
    ]
    equality = [
        left_row["token_ids"] == right_row["token_ids"]
        for left_row, right_row in zip(left_rows, right_rows, strict=True)
    ]
    speedups = [
        left_row["decode_completion_ms"] / right_row["decode_completion_ms"]
        for left_row, right_row in zip(left_rows, right_rows, strict=True)
    ]
    comparison = {
        "left": left,
        "right": right,
        "latency_gain_definition": "left minus right; positive favors right",
        "quality_delta_definition": "right minus left; positive favors right",
        "token_equality_rate": statistics.fmean(equality),
        "decode_completion_gain_ms": describe(
            completion_gain, bootstrap_samples=bootstrap_samples, seed=seed
        ),
        "decode_ttft_gain_ms": describe(
            ttft_gain, bootstrap_samples=bootstrap_samples, seed=seed + 1
        ),
        "decode_completion_speedup": describe(
            speedups, bootstrap_samples=bootstrap_samples, seed=seed + 2
        ),
        "quality_score_delta": describe(
            quality_delta, bootstrap_samples=bootstrap_samples, seed=seed + 3
        ),
        "right_faster_completion": sum(value > 0 for value in completion_gain),
        "right_faster_ttft": sum(value > 0 for value in ttft_gain),
    }
    if left == "baseline" and right != "baseline":
        models = [pair.get("latency_models", {}).get(right) for pair in pairs]
        models = [model for model in models if model is not None]
        if models:
            comparison["latency_model"] = {
                key: describe(
                    [model[key] for model in models],
                    bootstrap_samples=bootstrap_samples,
                    seed=seed + 10 + index,
                )
                for index, key in enumerate(
                    (
                        "target_step_ms",
                        "accepted_decode_value_ms",
                        "verify_ms",
                        "sparse_draft_span_ms",
                        "residual_modeled_wire_ms",
                        "exposed_draft_ms",
                        "predicted_gain_ms",
                        "observed_gain_ms",
                        "prediction_error_ms",
                        "absolute_prediction_error_ms",
                    )
                )
            }
            comparison["latency_model"]["gain_sign_agreement_rate"] = statistics.fmean(
                model["gain_sign_matches"] for model in models
            )
            comparison["latency_model"]["requests"] = len(models)
    return comparison


def aggregate_live_run(
    results: list[dict[str, Any]],
    metadata_rows: list[dict[str, Any]],
    scheduler_rows: list[dict[str, Any]],
    attention_rows: list[dict[str, Any]],
    link_rows: list[dict[str, Any]],
    *,
    arms: tuple[str, ...],
    expected_requests: int,
    fixed_output_tokens: int | None,
    bootstrap_samples: int,
    seed: int,
    protected_prefix_tokens: int = 0,
    protected_suffix_tokens: int = 0,
    priority_rows: list[dict[str, Any]] | None = None,
    priority_chunk_tokens: int = 256,
    enforce_paired_prefill: bool = True,
    stop_token_ids: tuple[int, ...] = (),
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if expected_requests <= 0 or bootstrap_samples <= 0:
        raise ValueError("expected requests and bootstrap samples must be positive")
    if protected_prefix_tokens < 0 or protected_suffix_tokens < 0:
        raise ValueError("protected token counts must be non-negative")
    if priority_chunk_tokens <= 0:
        raise ValueError("priority chunk size must be positive")
    if len(arms) < 2 or len(arms) != len(set(arms)):
        raise ValueError("arms must contain at least two unique values")
    if any(arm not in VALID_ARMS for arm in arms):
        raise ValueError(f"invalid arms: {arms}")
    metadata_by_index = {}
    for metadata in metadata_rows:
        index = metadata.get("request_index")
        if not isinstance(index, int) or index in metadata_by_index:
            raise ValueError("metadata request indices must be unique integers")
        metadata_by_index[index] = metadata
    priorities = _priority_by_request(priority_rows)

    matrix: dict[int, dict[str, dict[str, Any]]] = {}
    proxy_ids = set()
    raw_errors = []
    for result in results:
        request_index = result.get("request_index")
        source_index = result.get("source_request_index")
        arm = result.get("arm")
        if not isinstance(request_index, int) or request_index < 0:
            raise ValueError("result request_index must be a non-negative integer")
        if not isinstance(source_index, int) or source_index not in metadata_by_index:
            raise ValueError("result source_request_index has no metadata row")
        if arm not in arms:
            raise ValueError(f"unexpected result arm: {arm}")
        pair = matrix.setdefault(request_index, {})
        if arm in pair:
            raise ValueError(f"duplicate result for request {request_index}/{arm}")
        token_ids = result.get("token_ids")
        if not isinstance(token_ids, list) or any(
            not isinstance(token, int) or isinstance(token, bool) for token in token_ids
        ):
            raise TypeError("result token_ids must be an integer list")
        raw_errors.extend(
            f"{request_index}/{arm}: {error}"
            for error in _validate_token_arrivals(result)
        )
        for metric in (
            "header_ms",
            "ttft_ms",
            "decode_ttft_ms",
            "completion_ms",
            "decode_completion_ms",
        ):
            value = result.get(metric)
            if (
                not isinstance(value, int | float)
                or not math.isfinite(value)
                or value < 0
            ):
                raise ValueError(f"invalid result latency {metric}")
        proxy_id = result.get("proxy_request_id")
        if not isinstance(proxy_id, str) or not proxy_id:
            raw_errors.append(f"missing proxy request ID for {request_index}/{arm}")
        elif proxy_id in proxy_ids:
            raw_errors.append(f"duplicate proxy request ID {proxy_id}")
        proxy_ids.add(proxy_id)
        if result.get("warmup") is not False:
            raw_errors.append(
                f"warmup record leaked into results for {request_index}/{arm}"
            )
        expected_runtime = "baseline" if arm == "baseline" else "progressive"
        if result.get("runtime_mode") != expected_runtime:
            raw_errors.append(f"runtime mode mismatch for {request_index}/{arm}")
        prefill_group = result.get("prefill_group")
        if not isinstance(prefill_group, str) or not prefill_group:
            raw_errors.append(f"missing prefill group for {request_index}/{arm}")
        if not isinstance(result.get("prefill_reused"), bool):
            raw_errors.append(f"invalid prefill reuse flag for {request_index}/{arm}")
        seed_token_id = result.get("seed_token_id")
        if (
            not isinstance(seed_token_id, int)
            or isinstance(seed_token_id, bool)
            or seed_token_id < 0
        ):
            raw_errors.append(f"invalid producer seed for {request_index}/{arm}")
        pair[arm] = dict(result)

    expected_indices = set(range(expected_requests))
    complete_matrix = set(matrix) == expected_indices and all(
        set(pair) == set(arms) for pair in matrix.values()
    )

    scheduler_matches: dict[str, list[dict[str, Any]]] = {}
    for result in results:
        if result["arm"] == "baseline":
            continue
        scheduler_matches[result["proxy_request_id"]] = [
            row
            for row in scheduler_rows
            if _request_id_matches(
                str(row.get("request_id", "")), result["proxy_request_id"]
            )
        ]
    attention_by_request_id: dict[str, list[dict[str, Any]]] = {}
    for row in attention_rows:
        request_id = row.get("request_id")
        if isinstance(request_id, str) and request_id:
            attention_by_request_id.setdefault(request_id, []).append(row)
    link_matches = {
        result["proxy_request_id"]: [
            row
            for row in link_rows
            if _request_id_matches(
                str(row.get("request_id", "")), result["proxy_request_id"]
            )
        ]
        for result in results
    }

    paired_rows = []
    scheduler_errors = []
    attention_errors = []
    link_errors = []
    method_records = 0
    method_records_with_one_scheduler = 0
    method_records_with_attention = 0
    inferred_fixed = []
    paired_prefill_errors = []
    observed_prefill_groups: set[str] = set()
    for request_index in sorted(matrix):
        pair = matrix[request_index]
        source_indices = {row["source_request_index"] for row in pair.values()}
        if len(source_indices) != 1:
            raw_errors.append(f"arms do not share metadata for request {request_index}")
            continue
        source_index = next(iter(source_indices))
        if enforce_paired_prefill:
            prefill_groups = {row.get("prefill_group") for row in pair.values()}
            seeds = {row.get("seed_token_id") for row in pair.values()}
            fresh_prefills = sum(
                row.get("prefill_reused") is False for row in pair.values()
            )
            if len(prefill_groups) != 1:
                paired_prefill_errors.append(
                    f"{request_index}: arms do not share one prefill group"
                )
            else:
                prefill_group = next(iter(prefill_groups))
                if not isinstance(prefill_group, str) or not prefill_group:
                    paired_prefill_errors.append(
                        f"{request_index}: paired prefill group is invalid"
                    )
                elif prefill_group in observed_prefill_groups:
                    paired_prefill_errors.append(
                        f"{request_index}: prefill group was reused across executions"
                    )
                else:
                    observed_prefill_groups.add(prefill_group)
            if len(seeds) != 1:
                paired_prefill_errors.append(
                    f"{request_index}: arms do not share one producer seed"
                )
            if fresh_prefills != 1:
                paired_prefill_errors.append(
                    f"{request_index}: expected one producer prefill, got "
                    f"{fresh_prefills}"
                )
        metadata = metadata_by_index[source_index]
        if metadata.get("fixed_output_horizon") is True:
            inferred_fixed.append(int(metadata["output_tokens"]))
        enriched_arms = {}
        for arm, result in pair.items():
            metric_name, exact_match, quality_score = score_prediction(
                str(metadata["dataset"]),
                str(result.get("text", "")),
                list(metadata["answers"]),
                metadata.get("all_classes"),
            )
            enriched = {
                **result,
                "metric_name": metric_name,
                "exact_match": exact_match,
                "quality_score": quality_score,
            }
            matched_link = link_matches[result["proxy_request_id"]]
            if matched_link:
                link_summary, errors = _link_summary(matched_link, arm=arm)
                link_errors.extend(
                    f"{request_index}/{arm}: {error}" for error in errors
                )
                enriched["link"] = link_summary
            else:
                link_errors.append(
                    f"{request_index}/{arm}: no request-attributed link trace"
                )
                enriched["link"] = None
            if arm != "baseline":
                method_records += 1
                matches = scheduler_matches[result["proxy_request_id"]]
                if len(matches) == 1:
                    method_records_with_one_scheduler += 1
                    scheduler = dict(matches[0])
                    valid, errors = _validate_scheduler_record(scheduler, result)
                    if not valid:
                        scheduler_errors.extend(
                            f"{request_index}/{arm}: {error}" for error in errors
                        )
                    draft_tokens = int(scheduler["draft_tokens"])
                    accepted_tokens = int(scheduler["accepted_tokens"])
                    useful_draft_tokens = draft_tokens
                    token_ids = list(result["token_ids"])
                    if stop_token_ids:
                        if token_ids and token_ids[0] in stop_token_ids:
                            useful_draft_tokens = 0
                        else:
                            for output_index, token_id in enumerate(
                                token_ids[1 : draft_tokens + 1], start=1
                            ):
                                if token_id in stop_token_ids:
                                    useful_draft_tokens = output_index
                                    break
                    useful_accepted_tokens = min(
                        accepted_tokens, useful_draft_tokens
                    )
                    scheduler["useful_draft_tokens"] = useful_draft_tokens
                    scheduler["useful_accepted_tokens"] = useful_accepted_tokens
                    scheduler["accepted_tokens_after_first_stop"] = (
                        accepted_tokens - useful_accepted_tokens
                    )
                    attention = attention_by_request_id.get(
                        str(scheduler["request_id"]), []
                    )
                    if attention:
                        method_records_with_attention += 1
                        attention_summary, errors = _attention_summary(
                            attention, scheduler=scheduler, arm=arm
                        )
                        attention_errors.extend(
                            f"{request_index}/{arm}: {error}" for error in errors
                        )
                    else:
                        attention_summary = None
                        attention_errors.append(
                            f"{request_index}/{arm}: no request-attributed attention rows"
                        )
                    enriched["scheduler"] = scheduler
                    enriched["attention"] = attention_summary
                else:
                    scheduler_errors.append(
                        f"{request_index}/{arm}: expected one scheduler row, got {len(matches)}"
                    )
                    enriched["scheduler"] = None
                    enriched["attention"] = None
            enriched_arms[arm] = enriched
        paired_rows.append(
            {
                "request_index": request_index,
                "source_request_index": source_index,
                "metadata": metadata,
                "arms": enriched_arms,
            }
        )

    for pair in paired_rows:
        link_summaries = [pair["arms"][arm]["link"] for arm in arms]
        if any(summary is None for summary in link_summaries):
            continue
        reference = link_summaries[0]
        for arm, summary in zip(arms[1:], link_summaries[1:], strict=True):
            for key in (
                "logical_tokens",
                "logical_payload_bytes",
                "link_gbps",
                "bytes_per_token",
                "modeled_wire_ms",
            ):
                if not math.isclose(
                    float(reference[key]),
                    float(summary[key]),
                    rel_tol=1e-9,
                    abs_tol=1e-6,
                ):
                    link_errors.append(
                        f"{pair['request_index']}/{arm}: fair-link {key} mismatch"
                    )

    latency_model_errors = []
    latency_model_records = 0
    if "baseline" in arms:
        for pair in paired_rows:
            pair["latency_models"] = {}
            for arm in arms:
                if arm == "baseline":
                    continue
                model, errors = _request_latency_model(pair, arm)
                if model is not None:
                    latency_model_records += 1
                pair["latency_models"][arm] = model
                latency_model_errors.extend(
                    f"{pair['request_index']}/{arm}: {error}" for error in errors
                )

    inferred_fixed_set = set(inferred_fixed)
    if fixed_output_tokens is None and len(inferred_fixed_set) == 1:
        fixed_output_tokens = next(iter(inferred_fixed_set))
    fixed_output_errors = []
    if fixed_output_tokens is not None:
        for pair in paired_rows:
            for arm, row in pair["arms"].items():
                if len(row["token_ids"]) != fixed_output_tokens:
                    fixed_output_errors.append(
                        f"{pair['request_index']}/{arm}: "
                        f"{len(row['token_ids'])} != {fixed_output_tokens}"
                    )

    equality_errors = []
    for pair in paired_rows:
        reference = pair["arms"][arms[0]]["token_ids"]
        for arm in arms[1:]:
            if pair["arms"][arm]["token_ids"] != reference:
                equality_errors.append(f"{pair['request_index']}: {arms[0]} != {arm}")

    protected_anchor_errors = []
    priority_schedule_errors = []
    for pair in paired_rows:
        source_index = int(pair["source_request_index"])
        expected_priority = None
        if priorities is not None:
            expected_priority = priorities.get(source_index)
            if expected_priority is None:
                priority_schedule_errors.append(
                    f"{pair['request_index']}: no priority row for source request "
                    f"{source_index}"
                )
        for arm in arms:
            if arm == "baseline":
                continue
            link_summary = pair["arms"][arm].get("link")
            if not isinstance(link_summary, dict):
                continue
            first_ranges = link_summary["first_bundle_token_ranges"]
            retrieve_start = int(link_summary["retrieve_start_token"])
            retrieve_end = int(link_summary["retrieve_end_token"])
            prefix_end = min(retrieve_end, retrieve_start + protected_prefix_tokens)
            suffix_start = max(retrieve_start, retrieve_end - protected_suffix_tokens)
            if not _ranges_cover(first_ranges, retrieve_start, prefix_end):
                protected_anchor_errors.append(
                    f"{pair['request_index']}/{arm}: protected prefix not in S1"
                )
            if not _ranges_cover(first_ranges, suffix_start, retrieve_end):
                protected_anchor_errors.append(
                    f"{pair['request_index']}/{arm}: protected suffix not in S1"
                )
            if expected_priority is not None:
                priority_schedule_errors.extend(
                    f"{pair['request_index']}/{arm}: {error}"
                    for error in _validate_priority_schedule(
                        link_summary,
                        expected_priority,
                        priority_chunk_tokens,
                    )
                )

    gates = {
        "complete_paired_matrix": _gate(
            complete_matrix,
            expected_requests=expected_requests,
            observed_request_indices=sorted(matrix),
        ),
        "request_protocol": _gate(not raw_errors, errors=raw_errors),
        "single_paired_prefill": _gate(
            not enforce_paired_prefill or not paired_prefill_errors,
            applicable=enforce_paired_prefill,
            expected_fresh_prefills=(
                expected_requests if enforce_paired_prefill else None
            ),
            observed_prefill_groups=len(observed_prefill_groups),
            errors=paired_prefill_errors,
        ),
        "fixed_output_horizon": _gate(
            not fixed_output_errors,
            expected_output_tokens=fixed_output_tokens,
            errors=fixed_output_errors,
        ),
        "one_scheduler_transition_per_method": _gate(
            method_records_with_one_scheduler == method_records,
            expected=method_records,
            observed=method_records_with_one_scheduler,
        ),
        "scheduler_transition_invariants": _gate(
            not scheduler_errors, errors=scheduler_errors
        ),
        "request_attributed_sparse_attention": _gate(
            method_records_with_attention == method_records and not attention_errors,
            expected=method_records,
            observed=method_records_with_attention,
            errors=attention_errors,
        ),
        "fair_controlled_link": _gate(not link_errors, errors=link_errors),
        "protected_anchor_arrival": _gate(
            not protected_anchor_errors,
            protected_prefix_tokens=protected_prefix_tokens,
            protected_suffix_tokens=protected_suffix_tokens,
            errors=protected_anchor_errors,
        ),
        "request_scoped_priority_schedule": _gate(
            priorities is None or not priority_schedule_errors,
            applicable=priorities is not None,
            priority_chunk_tokens=priority_chunk_tokens,
            sidecar_requests=len(priorities) if priorities is not None else 0,
            errors=priority_schedule_errors,
        ),
        "latency_model_telemetry": _gate(
            "baseline" not in arms
            or (latency_model_records == method_records and not latency_model_errors),
            applicable="baseline" in arms,
            expected=method_records if "baseline" in arms else 0,
            observed=latency_model_records,
            errors=latency_model_errors,
        ),
        "exact_greedy_output_equivalence": _gate(
            not equality_errors, errors=equality_errors
        ),
    }
    gates["overall"] = _gate(all(item["passed"] for item in gates.values()))

    arm_summaries = {}
    for arm in arms:
        rows = [pair["arms"][arm] for pair in paired_rows]
        arm_summary = {
            "requests": len(rows),
            "metric_names": sorted({row["metric_name"] for row in rows}),
            "exact_match": statistics.fmean(row["exact_match"] for row in rows),
            "quality_score": describe(
                [row["quality_score"] for row in rows],
                bootstrap_samples=bootstrap_samples,
                seed=seed,
            ),
        }
        for metric in (
            "ttft_ms",
            "decode_ttft_ms",
            "completion_ms",
            "decode_completion_ms",
        ):
            arm_summary[metric] = describe(
                [row[metric] for row in rows],
                bootstrap_samples=bootstrap_samples,
                seed=seed + len(arm_summary),
            )
        if arm != "baseline":
            schedulers = [row["scheduler"] for row in rows if row["scheduler"]]
            attentions = [row["attention"] for row in rows if row["attention"]]
            arm_summary["draft"] = {
                "draft_tokens": describe(
                    [row["draft_tokens"] for row in schedulers],
                    bootstrap_samples=bootstrap_samples,
                    seed=seed + 20,
                ),
                "accepted_tokens": describe(
                    [row["accepted_tokens"] for row in schedulers],
                    bootstrap_samples=bootstrap_samples,
                    seed=seed + 21,
                ),
                "acceptance_rate": (
                    sum(row["accepted_tokens"] for row in schedulers)
                    / sum(row["draft_tokens"] for row in schedulers)
                    if sum(row["draft_tokens"] for row in schedulers)
                    else 0.0
                ),
                "useful_draft_tokens": describe(
                    [row["useful_draft_tokens"] for row in schedulers],
                    bootstrap_samples=bootstrap_samples,
                    seed=seed + 24,
                ),
                "useful_accepted_tokens": describe(
                    [row["useful_accepted_tokens"] for row in schedulers],
                    bootstrap_samples=bootstrap_samples,
                    seed=seed + 25,
                ),
                "effective_acceptance_rate": (
                    sum(row["useful_accepted_tokens"] for row in schedulers)
                    / sum(row["useful_draft_tokens"] for row in schedulers)
                    if sum(row["useful_draft_tokens"] for row in schedulers)
                    else 0.0
                ),
                "accepted_tokens_after_first_stop": describe(
                    [
                        row["accepted_tokens_after_first_stop"]
                        for row in schedulers
                    ],
                    bootstrap_samples=bootstrap_samples,
                    seed=seed + 26,
                ),
                "transition_ms": describe(
                    [row["transition_ns"] / 1e6 for row in schedulers],
                    bootstrap_samples=bootstrap_samples,
                    seed=seed + 22,
                ),
                "verify_ms": describe(
                    [row["verify_ns"] / 1e6 for row in schedulers],
                    bootstrap_samples=bootstrap_samples,
                    seed=seed + 23,
                ),
                "sparse_step_rate": (
                    statistics.fmean(row["sparse_step_rate"] for row in attentions)
                    if attentions
                    else math.nan
                ),
                "requests_with_progressive_visibility": sum(
                    row["distinct_visibility_fractions"] > 1 for row in attentions
                ),
            }
        links = [row["link"] for row in rows if row["link"]]
        arm_summary["link"] = {
            "modeled_wire_ms": describe(
                [row["modeled_wire_ms"] for row in links],
                bootstrap_samples=bootstrap_samples,
                seed=seed + 30,
            ),
            "observed_chain_ms": describe(
                [row["observed_chain_ms"] for row in links],
                bootstrap_samples=bootstrap_samples,
                seed=seed + 31,
            ),
            "logical_payload_bytes_mean": statistics.fmean(
                row["logical_payload_bytes"] for row in links
            ),
        }
        arm_summaries[arm] = arm_summary

    comparisons = {}
    if "baseline" in arms:
        for arm_index, arm in enumerate(arms):
            if arm == "baseline":
                continue
            comparisons[f"baseline_vs_{arm}"] = _pair_comparison(
                paired_rows,
                "baseline",
                arm,
                bootstrap_samples=bootstrap_samples,
                seed=seed + 100 * (arm_index + 1),
            )
    if "fixed_s1" in arms and "continuous" in arms:
        comparisons["fixed_s1_vs_continuous"] = _pair_comparison(
            paired_rows,
            "fixed_s1",
            "continuous",
            bootstrap_samples=bootstrap_samples,
            seed=seed + 999,
        )
    summary = {
        "schema_version": 2,
        "status": "passed" if gates["overall"]["passed"] else "failed_gates",
        "arms": list(arms),
        "requests": len(paired_rows),
        "datasets": sorted({pair["metadata"]["dataset"] for pair in paired_rows}),
        "fixed_output_tokens": fixed_output_tokens,
        "stop_token_ids": list(stop_token_ids),
        "bootstrap": {"samples": bootstrap_samples, "seed": seed},
        "priority_schedule": {
            "applicable": priorities is not None,
            "chunk_tokens": priority_chunk_tokens,
            "sidecar_requests": len(priorities) if priorities is not None else 0,
        },
        "gates": gates,
        "arm_summaries": arm_summaries,
        "comparisons": comparisons,
    }
    return summary, paired_rows


def _write_markdown(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Progressive P/D live result",
        "",
        f"Gate status: **{summary['status']}**; paired requests: {summary['requests']}.",
        "",
        "| Arm | Quality | Decode TTFT p50 (ms) | Decode completion p50 (ms) |",
        "|---|---:|---:|---:|",
    ]
    for arm, item in summary["arm_summaries"].items():
        lines.append(
            f"| {arm} | {item['quality_score']['mean']:.4f} | "
            f"{item['decode_ttft_ms']['p50']:.3f} | "
            f"{item['decode_completion_ms']['p50']:.3f} |"
        )
    lines.extend(
        [
            "",
            (
                "| Comparison | Token equality | Completion gain mean [95% CI] "
                "(ms) | Predicted gain (ms) | Model MAE (ms) |"
            ),
            "|---|---:|---:|---:|---:|",
        ]
    )
    for name, item in summary["comparisons"].items():
        gain = item["decode_completion_gain_ms"]
        ci = gain["mean_bootstrap_ci95"]
        model = item.get("latency_model")
        predicted = f"{model['predicted_gain_ms']['mean']:.3f}" if model else "-"
        mae = f"{model['absolute_prediction_error_ms']['mean']:.3f}" if model else "-"
        lines.append(
            f"| {name} | {item['token_equality_rate']:.4f} | "
            f"{gain['mean']:.3f} [{ci[0]:.3f}, {ci[1]:.3f}] | "
            f"{predicted} | {mae} |"
        )
    lines.extend(["", "## Gates", ""])
    for name, gate in summary["gates"].items():
        lines.append(f"- {name}: {'PASS' if gate['passed'] else 'FAIL'}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_csv(path: Path, summary: dict[str, Any]) -> None:
    fieldnames = [
        "kind",
        "name",
        "requests",
        "quality_mean",
        "token_equality_rate",
        "decode_ttft_p50_ms",
        "decode_completion_p50_ms",
        "decode_completion_gain_mean_ms",
        "decode_completion_gain_ci95_low_ms",
        "decode_completion_gain_ci95_high_ms",
        "predicted_gain_mean_ms",
        "latency_model_mae_ms",
        "latency_model_gain_sign_agreement_rate",
    ]
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        for arm, item in summary["arm_summaries"].items():
            writer.writerow(
                {
                    "kind": "arm",
                    "name": arm,
                    "requests": item["requests"],
                    "quality_mean": item["quality_score"]["mean"],
                    "decode_ttft_p50_ms": item["decode_ttft_ms"]["p50"],
                    "decode_completion_p50_ms": item["decode_completion_ms"]["p50"],
                }
            )
        for name, item in summary["comparisons"].items():
            gain = item["decode_completion_gain_ms"]
            model = item.get("latency_model", {})
            writer.writerow(
                {
                    "kind": "comparison",
                    "name": name,
                    "requests": summary["requests"],
                    "token_equality_rate": item["token_equality_rate"],
                    "decode_completion_gain_mean_ms": gain["mean"],
                    "decode_completion_gain_ci95_low_ms": gain["mean_bootstrap_ci95"][
                        0
                    ],
                    "decode_completion_gain_ci95_high_ms": gain["mean_bootstrap_ci95"][
                        1
                    ],
                    "predicted_gain_mean_ms": model.get("predicted_gain_ms", {}).get(
                        "mean"
                    ),
                    "latency_model_mae_ms": model.get(
                        "absolute_prediction_error_ms", {}
                    ).get("mean"),
                    "latency_model_gain_sign_agreement_rate": model.get(
                        "gain_sign_agreement_rate"
                    ),
                }
            )


def write_live_artifacts(
    output_dir: Path,
    summary: dict[str, Any],
    paired_rows: list[dict[str, Any]],
) -> None:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {output_dir}")
    output_dir.mkdir(parents=True)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    with (output_dir / "paired_rows.jsonl").open("w", encoding="utf-8") as output:
        for row in paired_rows:
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
    _write_markdown(output_dir / "summary.md", summary)
    _write_csv(output_dir / "paper_table.csv", summary)


def main() -> None:
    args = parse_args()
    arms = tuple(item.strip() for item in args.arms.split(",") if item.strip())
    summary, paired_rows = aggregate_live_run(
        read_jsonl(args.results_jsonl),
        read_jsonl(args.metadata_jsonl),
        read_jsonl(args.scheduler_stats_jsonl),
        read_jsonl(args.attention_stats_jsonl),
        read_jsonl(args.link_stats_jsonl),
        arms=arms,
        expected_requests=args.expected_requests,
        fixed_output_tokens=args.fixed_output_tokens,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
        protected_prefix_tokens=args.protected_prefix_tokens,
        protected_suffix_tokens=args.protected_suffix_tokens,
        priority_rows=(
            read_jsonl(args.priority_jsonl) if args.priority_jsonl is not None else None
        ),
        priority_chunk_tokens=args.priority_chunk_tokens,
        stop_token_ids=args.stop_token_ids,
    )
    write_live_artifacts(args.output_dir, summary, paired_rows)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if not summary["gates"]["overall"]["passed"]:
        raise SystemExit("one or more live-run validity gates failed")


if __name__ == "__main__":
    main()
