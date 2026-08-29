import copy

import pytest

from experiments.verify_progressive_pd_input_trace import verify_input_trace


def _records() -> tuple[list[dict], list[dict]]:
    scheduler = [
        {
            "request_id": "request-1",
            "seed_token_id": 10,
            "max_draft_tokens": 3,
            "draft_token_ids": [20, 21, 22],
            "draft_tokens": 3,
            "started_at_ns": 100,
            "last_draft_at_ns": 190,
            "full_arrival_at_ns": 250,
            "verify_started_at_ns": 300,
            "verify_completed_at_ns": 400,
        }
    ]
    inputs = [
        {
            "request_id": "unrelated-prefiller-request",
            "input_token_ids": [99],
            "positions": [8],
            "num_scheduled_tokens": 1,
            "recorded_at_ns": 110,
        },
        {
            "request_id": "request-1",
            "input_token_ids": [10],
            "positions": [65536],
            "num_scheduled_tokens": 1,
            "recorded_at_ns": 120,
        },
        {
            "request_id": "request-1",
            "input_token_ids": [20],
            "positions": [65537],
            "num_scheduled_tokens": 1,
            "recorded_at_ns": 150,
        },
        {
            "request_id": "request-1",
            "input_token_ids": [21],
            "positions": [65538],
            "num_scheduled_tokens": 1,
            "recorded_at_ns": 180,
        },
        {
            "request_id": "request-1",
            "input_token_ids": [10, 20, 21, 22],
            "positions": [65536, 65537, 65538, 65539],
            "num_scheduled_tokens": 4,
            "recorded_at_ns": 350,
        },
        {
            "request_id": "request-1",
            "input_token_ids": [30],
            "positions": [65540],
            "num_scheduled_tokens": 1,
            "recorded_at_ns": 450,
        },
    ]
    return scheduler, inputs


def test_verify_input_trace_proves_sparse_chain_and_full_verify() -> None:
    scheduler, inputs = _records()

    report = verify_input_trace(scheduler, inputs, expected_requests=1)

    assert report["status"] == "passed"
    assert report["verified_requests"] == 1
    assert report["total_sparse_forwards"] == 3
    assert report["total_verify_forwards"] == 1
    assert report["requests"][0]["prompt_tokens"] == 65536


def test_verify_input_trace_rejects_replayed_seed() -> None:
    scheduler, inputs = _records()
    inputs[2]["input_token_ids"] = [10]

    with pytest.raises(ValueError, match="sparse token chain mismatch"):
        verify_input_trace(scheduler, inputs, expected_requests=1)


def test_verify_input_trace_rejects_wrong_verify_batch() -> None:
    scheduler, inputs = _records()
    inputs[4]["input_token_ids"] = [10, 20, 21, 21]

    with pytest.raises(ValueError, match="verify token batch mismatch"):
        verify_input_trace(scheduler, inputs, expected_requests=1)


def test_verify_input_trace_rejects_position_drift() -> None:
    scheduler, inputs = _records()
    inputs[3]["positions"] = [65539]

    with pytest.raises(ValueError, match="sparse positions are not contiguous"):
        verify_input_trace(scheduler, inputs, expected_requests=1)


def test_verify_input_trace_rejects_missing_request() -> None:
    scheduler, inputs = _records()

    with pytest.raises(ValueError, match="scheduler request count mismatch"):
        verify_input_trace(scheduler, inputs, expected_requests=2)


def test_verify_input_trace_rejects_duplicate_scheduler_request() -> None:
    scheduler, inputs = _records()
    scheduler.append(copy.deepcopy(scheduler[0]))

    with pytest.raises(ValueError, match="duplicate scheduler request_id"):
        verify_input_trace(scheduler, inputs, expected_requests=2)
