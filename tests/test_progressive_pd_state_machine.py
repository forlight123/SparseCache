import pytest

from experiments.progressive_pd_state_machine import (
    ProgressivePDPhase,
    ProgressivePDRequestState,
)


def test_progressive_pd_commits_only_after_one_full_verifier() -> None:
    state = ProgressivePDRequestState(
        priority_pages=(4, 1, 3, 2),
        anchor_pages=1,
        max_draft_tokens=4,
    )
    with pytest.raises(RuntimeError, match="only after S1"):
        state.record_draft_token(10)

    state.record_page_completion({4})
    assert state.phase == ProgressivePDPhase.DRAFT_WHILE_LOADING
    state.record_draft_token(10)
    assert state.external_token_ids == []

    state.record_page_completion({1, 3})
    state.record_draft_token(11)
    assert state.proposals[0].visible_priority_pages == (4,)
    assert state.proposals[1].visible_priority_pages == (4, 1, 3)

    state.record_page_completion({2})
    assert state.phase == ProgressivePDPhase.READY_TO_VERIFY
    state.record_final_verification(accepted_prefix=1, correction_token=99)
    assert state.verifier_calls == 1
    assert state.external_token_ids == [10, 99]
    assert state.phase == ProgressivePDPhase.TARGET_DECODING


def test_out_of_order_completion_does_not_expose_a_priority_hole() -> None:
    state = ProgressivePDRequestState(
        priority_pages=(8, 2, 5),
        anchor_pages=2,
        max_draft_tokens=2,
    )
    state.record_page_completion({8, 5})
    assert state.visible_priority_pages == (8,)
    assert state.phase == ProgressivePDPhase.WAIT_ANCHOR

    state.record_page_completion({2})
    assert state.visible_priority_pages == (8, 2, 5)
    assert state.phase == ProgressivePDPhase.READY_TO_VERIFY


def test_progressive_pd_rejects_duplicate_and_unknown_completions() -> None:
    state = ProgressivePDRequestState(
        priority_pages=(0, 1),
        anchor_pages=1,
        max_draft_tokens=2,
    )
    state.record_page_completion({0})
    with pytest.raises(ValueError, match="twice"):
        state.record_page_completion({0})
    with pytest.raises(ValueError, match="unknown"):
        state.record_page_completion({9})
