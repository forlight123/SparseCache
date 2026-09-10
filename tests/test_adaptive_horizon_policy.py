import math

from experiments.analyze_adaptive_horizon_policy import analyze, choose, fit_thresholds


def row(horizon, margin, saving, tokens):
    return {
        "configured": horizon,
        "draft_min_margin": margin,
        "draft_margins": [margin] * horizon,
        "draft_token_ids": tokens[:horizon],
        "paired_saving_ms": saving,
        "token_match_target": True,
        "draft_reached_termination": False,
    }


def actions():
    result = {label: {} for label in ("g4", "g6", "g8", "g12")}
    tokens = list(range(12))
    for index in range(8):
        request_id = str(index)
        high = index % 2 == 0
        for label, horizon in (("g4", 4), ("g6", 6), ("g8", 8), ("g12", 12)):
            margin = 4.0 if high else 0.25
            saving = horizon if high else 10 - horizon
            result[label][request_id] = row(horizon, margin, saving, tokens)
    return result


def test_threshold_policy_uses_only_current_or_earlier_margin():
    candidates = actions()
    labels = ["g4", "g6", "g8", "g12"]
    thresholds = {"g4": 1.0, "g6": math.inf, "g8": math.inf}
    assert choose(candidates, labels, thresholds, "0") == "g6"
    assert choose(candidates, labels, thresholds, "1") == "g4"


def test_backward_fit_learns_to_continue_high_margin_requests():
    candidates = actions()
    labels = ["g4", "g6", "g8", "g12"]
    thresholds = fit_thresholds(
        candidates,
        labels,
        list(candidates["g4"]),
        candidates=(0.0, 1.0, math.inf),
        min_leaf=2,
    )
    assert choose(candidates, labels, thresholds, "0") == "g12"
    assert choose(candidates, labels, thresholds, "1") == "g4"


def test_train_and_evaluation_are_disjoint_and_fixed_comparator_is_frozen():
    result = analyze(actions(), train_count=4, min_leaf=1)
    assert result["train"]["requests"] == 4
    assert result["evaluation"]["requests"] == 4
    assert result["frozen_fixed_label"] in {"g4", "g6", "g8", "g12"}


def test_eos_forces_stop_before_margin_policy_continues():
    candidates = actions()
    candidates["g4"]["0"]["draft_reached_termination"] = True
    labels = ["g4", "g6", "g8", "g12"]
    thresholds = {"g4": 0.0, "g6": 0.0, "g8": 0.0}
    assert choose(candidates, labels, thresholds, "0") == "g4"


def test_validation_accepts_only_post_eos_timing_prefix_differences():
    candidates = actions()
    for label in candidates:
        candidates[label]["0"]["draft_token_ids"] = [0, 1]
        candidates[label]["0"]["draft_margins"] = [4.0, 4.0]
        candidates[label]["0"]["meaningful_draft_count"] = 2
    result = analyze(candidates, train_count=4, min_leaf=1)
    assert result["evaluation"]["requests"] == 4
