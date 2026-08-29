from experiments.eval_sparse_kv_boundaries import select_offsets, summarize


def test_boundary_offsets_exclude_initial_and_are_bounded() -> None:
    assert select_offsets(1, 8) == ()
    assert select_offsets(5, 8) == (1, 2, 3, 4)
    offsets = select_offsets(64, 8)
    assert len(offsets) == 8
    assert 0 not in offsets
    assert tuple(sorted(set(offsets))) == offsets


def test_boundary_summary_weights_tokens_and_windows_separately() -> None:
    summary = summarize(
        [
            {"tokens": 2, "correct": 1, "accepted": 1},
            {"tokens": 4, "correct": 3, "accepted": 4},
        ]
    )
    assert summary["teacher_forced_top1"] == 4 / 6
    assert summary["mean_accepted_prefix"] == 2.5
    assert summary["full_accept_rate"] == 0.5
