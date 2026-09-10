from experiments.analyze_joint_pd_policy import analyze, evaluate_frozen_selection


def action(savings, accepted, *, full=100.0, pipeline=None):
    rows = {}
    for index, (saving, acceptance) in enumerate(zip(savings, accepted)):
        rows[str(index)] = {
            "dataset_index": index,
            "prompt_tokens": 100,
            "accepted": acceptance,
            "configured": 4,
            "timing_token_match_target": True,
            "token_match_target": True,
            "first_draft_ms": 10.0,
            "verify_ms": 5.0,
            "draft_min_margin": 1.0,
            "pipeline_ms": full - saving if pipeline is None else pipeline,
            "full_ms": full,
            "paired_saving_ms": saving,
            "full_ready_ms": 40.0,
            "seed_forward_ms": 10.0,
            "decode_ms": 50.0,
            "tail_steps": 10,
            "max_new_tokens": 11,
            "logical_kv_bytes": 1024,
        }
    return rows


def test_candidate_oracle_uses_paired_saving_and_reports_selection_headroom():
    result = analyze(
        {
            "a": action([10.0, 1.0], [2, 3]),
            "b": action([2.0, 9.0], [4, 1]),
        },
        preselected="a",
        p_runahead_budgets=(4,),
    )

    assert result["best_fixed_action_on_this_split"] == "a"
    assert result["candidate_oracle"]["selected_counts"] == {"a": 1, "b": 1}
    assert result["candidate_oracle"]["paired_saving_ms"]["mean"] == 9.5
    assert result["candidate_oracle"]["advantage_vs_best_fixed_ms"]["mean"] == 4.0


def test_p_runahead_is_labelled_optimistic_and_charges_p_gpu_time():
    result = analyze(
        {"a": action([0.0], [4])},
        preselected="a",
        p_runahead_budgets=(4,),
    )

    row = result["p_exact_runahead"]["4"]
    assert row["p_gpu_occupancy_ms"]["mean"] == 20.0
    assert row["modeled_completion_ms"]["mean"] == 70.0
    assert row["saving_vs_full_transfer_ms"]["mean"] == 30.0
    assert "optimistic low-load" in result["contract"]["p_runahead"]


def test_frozen_selection_is_evaluated_on_independent_measurements():
    selected = analyze(
        {
            "a": action([10.0, 1.0], [2, 3]),
            "b": action([2.0, 9.0], [4, 1]),
        },
        preselected="a",
        p_runahead_budgets=(1,),
    )
    result = evaluate_frozen_selection(
        selected,
        {
            "a": action([5.0, 6.0], [2, 3]),
            "b": action([4.0, 3.0], [4, 1]),
        },
    )

    assert result["frozen_selector_paired_saving_ms"]["mean"] == 4.0
    assert result["source_selected_best_fixed"] == "a"
    assert result["advantage_vs_source_best_fixed_ms"]["mean"] == -1.5
    assert result["per_request_choice_repeated"] == 1
