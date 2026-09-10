"""Measure the finite-candidate scheduling headroom for lossless P/D drafting.

Every candidate must contain the same request IDs and output horizon.  Candidate
selection uses the request-paired saving against the full-transfer run measured
inside that candidate's process, which removes most cross-run drift.  The
per-request maximum is deliberately labelled a finite-candidate oracle: it is
an optimistic selection diagnostic, not a deployable policy or global bound.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
from pathlib import Path


def bootstrap_mean_ci(
    values: list[float], *, samples: int = 10_000, seed: int = 20260910
) -> list[float]:
    if not values:
        raise ValueError("cannot bootstrap an empty sample")
    generator = random.Random(seed)
    means = sorted(
        statistics.fmean(generator.choices(values, k=len(values)))
        for _ in range(samples)
    )
    return [means[int(0.025 * samples)], means[int(0.975 * samples)]]


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _extract(
    label: str, path: Path, *, require_exposed_equality: bool = True
) -> dict[str, dict]:
    extracted = {}
    rows = _read_jsonl(path)
    # Older benchmark records did not persist tokenizer EOS IDs.  Recover the
    # terminal-token set from quality trajectories that ended before the fixed
    # timing horizon.  This is exact for these greedy records and prevents the
    # controller from learning from intentionally generated post-EOS fillers.
    terminal_token_ids = set()
    for row in rows:
        chain = row["schedules"]["query"]["chains"]["inf"]
        max_new_tokens = int(row["protocol"]["max_new_tokens"])
        if chain["token_ids"] and len(chain["token_ids"]) < max_new_tokens:
            terminal_token_ids.add(int(chain["token_ids"][-1]))

    for row in rows:
        full = row["schedules"]["query"]["full_target"]
        chain = row["schedules"]["query"]["chains"]["inf"]
        pipeline = chain["pipeline"]
        verified = [item for item in pipeline["trace"] if item["verified"]]
        drafted = [item for item in pipeline["trace"] if not item["verified"]]
        if len(verified) != 1:
            raise ValueError(f"{label}/{row['id']} does not have one final verifier")
        if len(drafted) != 1:
            raise ValueError(f"{label}/{row['id']} does not have one draft block")
        if require_exposed_equality and not chain["token_match_target"]:
            raise ValueError(f"{label}/{row['id']} changed the exposed output")
        protocol = row["protocol"]
        seed_tokens = int(protocol["seed_tokens_from_prefill_side"])
        max_new_tokens = int(protocol["max_new_tokens"])
        configured = int(chain["configured_draft_tokens"])
        quality_tokens = [int(value) for value in chain["token_ids"]]
        quality_terminated = len(quality_tokens) < max_new_tokens
        raw_margins = [
            float(value) for value in drafted[0].get("draft_top1_margins", [])
        ]
        raw_token_ids = [
            int(value) for value in drafted[0].get("draft_token_ids", [])
        ]
        # The benchmark intentionally keeps decoding after EOS to hold the
        # timed output horizon constant.  Those filler tokens are useful for
        # latency accounting, but a causal horizon controller could never use
        # their identities or margins.  Include the first draft EOS itself and
        # discard everything after it.  Also handle the (rare) case where P's
        # seed token already terminates the request.
        seed_terminated = bool(
            quality_tokens
            and seed_tokens > 0
            and quality_tokens[0] in terminal_token_ids
        )
        draft_eos_position = next(
            (
                index
                for index, token_id in enumerate(raw_token_ids)
                if token_id in terminal_token_ids
            ),
            None,
        )
        if seed_terminated:
            meaningful_draft_count = 0
        elif draft_eos_position is not None:
            meaningful_draft_count = draft_eos_position + 1
        else:
            meaningful_draft_count = configured
        meaningful_margins = raw_margins[:meaningful_draft_count]
        meaningful_token_ids = raw_token_ids[:meaningful_draft_count]
        extracted[row["id"]] = {
            "dataset_index": int(row["dataset_index"]),
            "prompt_tokens": int(row["prompt_tokens"]),
            "accepted": int(verified[0]["accepted_pending"]),
            "accepted_meaningful": min(
                int(verified[0]["accepted_pending"]), meaningful_draft_count
            ),
            "configured": configured,
            "timing_token_match_target": bool(chain["timing_token_match_target"]),
            "token_match_target": bool(chain["token_match_target"]),
            "first_draft_ms": float(pipeline["first_draft_batch_ms"]),
            "verify_ms": float(pipeline["verify_ms"]),
            "draft_min_margin": (
                min(meaningful_margins) if meaningful_margins else float("inf")
            ),
            "draft_margins": meaningful_margins,
            "draft_token_ids": meaningful_token_ids,
            "timing_draft_margins": raw_margins,
            "timing_draft_token_ids": raw_token_ids,
            "meaningful_draft_count": meaningful_draft_count,
            "quality_terminated": quality_terminated,
            "draft_reached_termination": seed_terminated
            or draft_eos_position is not None,
            "inferred_terminal_token_ids": sorted(terminal_token_ids),
            "pipeline_ms": float(pipeline["response_ms"]),
            "full_ms": float(full["response_ms"]),
            "paired_saving_ms": float(full["response_ms"])
            - float(pipeline["response_ms"]),
            "full_ready_ms": float(full["response_ms"])
            - float(full["seed_forward_ms"])
            - float(full["decode_ms"]),
            "seed_forward_ms": float(full["seed_forward_ms"]),
            "decode_ms": float(full["decode_ms"]),
            "tail_steps": max_new_tokens - seed_tokens,
            "max_new_tokens": max_new_tokens,
            "logical_kv_bytes": int(row["logical_bf16_kv_bytes"]),
        }
    if not extracted:
        raise ValueError(f"{path} has no records")
    return extracted


def _describe(values: list[float]) -> dict:
    return {
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "bootstrap_95ci": bootstrap_mean_ci(values),
        "minimum": min(values),
        "maximum": max(values),
    }


def apply_adjudications(
    actions: dict[str, dict[str, dict]],
    specifications: list[str],
    *,
    require_exposed_equality: bool,
) -> list[dict]:
    """Replace explicitly named stalled measurements and retain an audit log."""

    audit = []
    for specification in specifications:
        label, path = specification.split(":", 1)
        if label not in actions:
            raise ValueError(f"adjudication names an unknown candidate: {label}")
        replacement = _extract(
            label,
            Path(path),
            require_exposed_equality=require_exposed_equality,
        )
        if len(replacement) != 1:
            raise ValueError("each adjudication file must contain exactly one request")
        request_id, new_row = next(iter(replacement.items()))
        if request_id not in actions[label]:
            raise ValueError(
                f"adjudication request is absent from {label}: {request_id}"
            )
        old_row = actions[label][request_id]
        if (old_row["max_new_tokens"], old_row["logical_kv_bytes"]) != (
            new_row["max_new_tokens"],
            new_row["logical_kv_bytes"],
        ):
            raise ValueError("adjudication changed the request payload or horizon")
        actions[label][request_id] = new_row
        audit.append(
            {
                "candidate": label,
                "request_id": request_id,
                "source": path,
                "original_paired_saving_ms": old_row["paired_saving_ms"],
                "replacement_paired_saving_ms": new_row["paired_saving_ms"],
            }
        )
    return audit


def analyze(
    actions: dict[str, dict[str, dict]],
    *,
    preselected: str,
    p_runahead_budgets: tuple[int, ...] = (1, 4, 8, 16),
) -> dict:
    if preselected not in actions:
        raise ValueError("preselected action is missing")
    ids = set(actions[preselected])
    if any(set(rows) != ids for rows in actions.values()):
        raise ValueError("candidate request IDs differ")
    if any(budget <= 0 for budget in p_runahead_budgets):
        raise ValueError("P run-ahead budgets must be positive")

    ordered_ids = sorted(ids, key=lambda item: actions[preselected][item]["dataset_index"])
    reference_shape = {
        request_id: (
            actions[preselected][request_id]["max_new_tokens"],
            actions[preselected][request_id]["logical_kv_bytes"],
        )
        for request_id in ordered_ids
    }
    for label, rows in actions.items():
        shape = {
            request_id: (
                rows[request_id]["max_new_tokens"],
                rows[request_id]["logical_kv_bytes"],
            )
            for request_id in ordered_ids
        }
        if shape != reference_shape:
            raise ValueError(f"candidate payload or output horizon differs: {label}")

    fixed = {}
    for label, rows in actions.items():
        accepted = [rows[request_id]["accepted"] for request_id in ordered_ids]
        accepted_meaningful = [
            rows[request_id].get(
                "accepted_meaningful", rows[request_id]["accepted"]
            )
            for request_id in ordered_ids
        ]
        savings = [rows[request_id]["paired_saving_ms"] for request_id in ordered_ids]
        fixed[label] = {
            "configured_draft_tokens": rows[ordered_ids[0]]["configured"],
            "exposed_outputs_equal": sum(
                rows[request_id]["token_match_target"]
                for request_id in ordered_ids
            ),
            "post_eos_timing_trajectories_equal": sum(
                rows[request_id]["timing_token_match_target"]
                for request_id in ordered_ids
            ),
            "accepted_prefix": _describe(accepted),
            "accepted_pre_eos_prefix": _describe(accepted_meaningful),
            "draft_reached_eos": sum(
                rows[request_id].get("draft_reached_termination", False)
                for request_id in ordered_ids
            ),
            "first_draft_ms": _describe(
                [rows[request_id]["first_draft_ms"] for request_id in ordered_ids]
            ),
            "verify_ms": _describe(
                [rows[request_id]["verify_ms"] for request_id in ordered_ids]
            ),
            "paired_saving_ms": _describe(savings),
            "faster_requests": sum(value > 0 for value in savings),
        }

    best_fixed = max(
        fixed,
        key=lambda label: fixed[label]["paired_saving_ms"]["mean"],
    )
    selected = []
    oracle_savings = []
    advantage_best = []
    advantage_preselected = []
    selection_rows = []
    for request_id in ordered_ids:
        label = max(
            actions,
            key=lambda item: actions[item][request_id]["paired_saving_ms"],
        )
        selected.append(label)
        oracle = actions[label][request_id]["paired_saving_ms"]
        oracle_savings.append(oracle)
        advantage_best.append(
            oracle - actions[best_fixed][request_id]["paired_saving_ms"]
        )
        advantage_preselected.append(
            oracle - actions[preselected][request_id]["paired_saving_ms"]
        )
        selection_rows.append(
            {
                "request_id": request_id,
                "dataset_index": actions[preselected][request_id]["dataset_index"],
                "prompt_tokens": actions[preselected][request_id]["prompt_tokens"],
                "selected": label,
                "selected_output_equal": actions[label][request_id][
                    "token_match_target"
                ],
                "paired_saving_ms": {
                    item: actions[item][request_id]["paired_saving_ms"]
                    for item in sorted(actions)
                },
                "accepted_prefix": {
                    item: actions[item][request_id]["accepted"]
                    for item in sorted(actions)
                },
                "draft_min_margin": {
                    item: actions[item][request_id]["draft_min_margin"]
                    for item in sorted(actions)
                },
            }
        )

    reference = actions[best_fixed]
    p_runahead = {}
    for budget in p_runahead_budgets:
        totals = []
        savings = []
        p_gpu = []
        versus_best_pipeline = []
        for request_id in ordered_ids:
            row = reference[request_id]
            produced = min(budget, row["tail_steps"])
            token_ms = row["decode_ms"] / row["tail_steps"]
            occupied = produced * token_ms
            # Optimistic low-load control: P keeps its exact prompt state,
            # generates ``produced`` tail tokens while prompt KV moves, and
            # sends their much smaller generated KV before D resumes.  NIC/GPU
            # contention and generated-KV handoff are intentionally omitted.
            total = max(row["full_ready_ms"], occupied) + (
                row["tail_steps"] - produced
            ) * token_ms
            totals.append(total)
            savings.append(row["full_ms"] - total)
            p_gpu.append(occupied)
            versus_best_pipeline.append(total - row["pipeline_ms"])
        p_runahead[str(budget)] = {
            "modeled_completion_ms": _describe(totals),
            "saving_vs_full_transfer_ms": _describe(savings),
            "p_gpu_occupancy_ms": _describe(p_gpu),
            "completion_minus_best_fixed_pipeline_ms": _describe(
                versus_best_pipeline
            ),
        }

    return {
        "contract": {
            "requests": len(ordered_ids),
            "preselected_action": preselected,
            "finite_candidate_oracle": (
                "per-request maximum paired saving among measured candidates; "
                "optimistic selection diagnostic, not a deployable policy or "
                "global upper bound"
            ),
            "p_runahead": (
                "optimistic low-load model; exact P tail decoding overlaps prompt-KV "
                "transfer; generated-KV handoff and P/NIC contention omitted"
            ),
        },
        "fixed_actions": fixed,
        "best_fixed_action_on_this_split": best_fixed,
        "candidate_oracle": {
            "selected_counts": {
                label: selected.count(label) for label in sorted(actions)
            },
            "paired_saving_ms": _describe(oracle_savings),
            "advantage_vs_best_fixed_ms": _describe(advantage_best),
            "advantage_vs_preselected_ms": _describe(advantage_preselected),
            "per_request": selection_rows,
        },
        "p_exact_runahead": p_runahead,
    }


def evaluate_frozen_selection(result: dict, evaluation: dict[str, dict[str, dict]]) -> dict:
    selection = result["candidate_oracle"]["per_request"]
    ids = {row["request_id"] for row in selection}
    if any(set(rows) != ids for rows in evaluation.values()):
        raise ValueError("evaluation candidate request IDs differ")
    missing = {row["selected"] for row in selection} - set(evaluation)
    if missing:
        raise ValueError(f"evaluation is missing selected candidates: {sorted(missing)}")

    fixed_means = {
        label: statistics.fmean(rows[request_id]["paired_saving_ms"] for request_id in ids)
        for label, rows in evaluation.items()
    }
    source_best = result["best_fixed_action_on_this_split"]
    if source_best not in evaluation:
        raise ValueError("evaluation is missing the source-selected fixed action")
    evaluation_best = max(fixed_means, key=fixed_means.get)
    frozen = []
    versus_source_fixed = []
    versus_evaluation_fixed = []
    repeated = 0
    for row in selection:
        request_id = row["request_id"]
        selected = row["selected"]
        saving = evaluation[selected][request_id]["paired_saving_ms"]
        frozen.append(saving)
        versus_source_fixed.append(
            saving - evaluation[source_best][request_id]["paired_saving_ms"]
        )
        versus_evaluation_fixed.append(
            saving - evaluation[evaluation_best][request_id]["paired_saving_ms"]
        )
        observed_best = max(
            evaluation,
            key=lambda label: evaluation[label][request_id]["paired_saving_ms"],
        )
        repeated += observed_best == selected
    return {
        "contract": (
            "candidate choice is frozen from the selection replicate; all latency "
            "values come from the independent evaluation replicate"
        ),
        "fixed_mean_paired_saving_ms": fixed_means,
        "source_selected_best_fixed": source_best,
        "evaluation_best_fixed_descriptive": evaluation_best,
        "frozen_selector_paired_saving_ms": _describe(frozen),
        "advantage_vs_source_best_fixed_ms": _describe(versus_source_fixed),
        "advantage_vs_evaluation_best_fixed_ms": _describe(versus_evaluation_fixed),
        "per_request_choice_repeated": repeated,
        "requests": len(selection),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--candidate", action="append", required=True, help="LABEL:results.jsonl"
    )
    parser.add_argument(
        "--evaluation-candidate",
        action="append",
        help="LABEL:independent-results.jsonl for a frozen-selection replication",
    )
    parser.add_argument(
        "--adjudication",
        action="append",
        default=[],
        help="LABEL:single-request-remeasurement.jsonl; raw files remain unchanged",
    )
    parser.add_argument("--preselected", required=True)
    parser.add_argument(
        "--diagnostic-allow-output-mismatch",
        action="store_true",
        help="report numerical mismatches instead of failing; never use as exact evidence",
    )
    parser.add_argument("--p-runahead-budgets", default="1,4,8,16")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    actions = {}
    for item in args.candidate:
        label, path = item.split(":", 1)
        if label in actions:
            raise ValueError(f"duplicate candidate label: {label}")
        actions[label] = _extract(
            label,
            Path(path),
            require_exposed_equality=not args.diagnostic_allow_output_mismatch,
        )
    budgets = tuple(int(value) for value in args.p_runahead_budgets.split(","))
    adjudications = apply_adjudications(
        actions,
        args.adjudication,
        require_exposed_equality=not args.diagnostic_allow_output_mismatch,
    )
    result = analyze(actions, preselected=args.preselected, p_runahead_budgets=budgets)
    result["adjudications"] = adjudications
    if args.evaluation_candidate:
        evaluation = {}
        for item in args.evaluation_candidate:
            label, path = item.split(":", 1)
            if label in evaluation:
                raise ValueError(f"duplicate evaluation label: {label}")
            evaluation[label] = _extract(
                label,
                Path(path),
                require_exposed_equality=not args.diagnostic_allow_output_mismatch,
            )
        result["independent_selection_evaluation"] = evaluate_frozen_selection(
            result, evaluation
        )
    write_json(Path(args.output), result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
