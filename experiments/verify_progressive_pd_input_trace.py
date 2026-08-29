"""Fail-closed audit for scheduler-injected progressive P/D inputs.

The scheduler trace records the exact producer seed, the sparse draft outputs,
and the full-KV verification interval.  The worker trace records the token IDs
and positions that the GPU model runner actually consumed.  This audit joins
the two traces by request ID and proves that the sparse autoregressive chain
and the final verifier consumed the intended sequences.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scheduler-stats-jsonl", type=Path, required=True)
    parser.add_argument("--input-stats-jsonl", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-requests", type=int, required=True)
    return parser.parse_args()


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as source:
        for line_number, raw_line in enumerate(source, start=1):
            line = raw_line.strip()
            if not line:
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            records.append(value)
    if not records:
        raise ValueError(f"{path} contains no records")
    return records


def _integer_list(record: dict[str, Any], key: str) -> list[int]:
    value = record.get(key)
    if not isinstance(value, list) or any(
        not isinstance(item, int) or isinstance(item, bool) for item in value
    ):
        raise ValueError(f"{key} must be a list of integers")
    return value


def _integer(record: dict[str, Any], key: str) -> int:
    value = record.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{key} must be an integer")
    return value


def _contiguous(values: list[int]) -> bool:
    return all(right == left + 1 for left, right in zip(values, values[1:]))


def verify_input_trace(
    scheduler_records: list[dict[str, Any]],
    input_records: list[dict[str, Any]],
    *,
    expected_requests: int,
) -> dict[str, Any]:
    """Verify sparse draft and final verifier inputs for every request.

    Args:
        scheduler_records: Completed progressive scheduler records.
        input_records: Small-forward GPU model-runner input records.
        expected_requests: Exact number of progressive request records required.

    Returns:
        A JSON-serializable passing audit report.

    Raises:
        ValueError: If a trace is incomplete, ambiguous, or inconsistent.
    """
    if expected_requests <= 0:
        raise ValueError("expected_requests must be positive")
    if len(scheduler_records) != expected_requests:
        raise ValueError(
            "scheduler request count mismatch: "
            f"{len(scheduler_records)} != {expected_requests}"
        )

    inputs_by_request: dict[str, list[dict[str, Any]]] = {}
    for record in input_records:
        request_id = record.get("request_id")
        if not isinstance(request_id, str) or not request_id:
            raise ValueError("input trace contains an invalid request_id")
        _integer(record, "recorded_at_ns")
        tokens = _integer_list(record, "input_token_ids")
        positions = _integer_list(record, "positions")
        count = _integer(record, "num_scheduled_tokens")
        if count <= 0 or len(tokens) != count or len(positions) != count:
            raise ValueError(f"input trace shape mismatch for {request_id}")
        inputs_by_request.setdefault(request_id, []).append(record)
    for records in inputs_by_request.values():
        records.sort(key=lambda item: _integer(item, "recorded_at_ns"))

    seen_request_ids: set[str] = set()
    request_reports: list[dict[str, Any]] = []
    total_sparse_forwards = 0
    total_verify_forwards = 0
    for scheduler_record in scheduler_records:
        request_id = scheduler_record.get("request_id")
        if not isinstance(request_id, str) or not request_id:
            raise ValueError("scheduler trace contains an invalid request_id")
        if request_id in seen_request_ids:
            raise ValueError(f"duplicate scheduler request_id: {request_id}")
        seen_request_ids.add(request_id)

        seed = _integer(scheduler_record, "seed_token_id")
        drafts = _integer_list(scheduler_record, "draft_token_ids")
        draft_tokens = _integer(scheduler_record, "draft_tokens")
        max_draft_tokens = _integer(scheduler_record, "max_draft_tokens")
        if not drafts or len(drafts) != draft_tokens:
            raise ValueError(f"incomplete draft sequence for {request_id}")
        if draft_tokens != max_draft_tokens:
            raise ValueError(f"draft did not reach its fixed horizon for {request_id}")

        started_at = _integer(scheduler_record, "started_at_ns")
        last_draft_at = _integer(scheduler_record, "last_draft_at_ns")
        full_arrival_at = _integer(scheduler_record, "full_arrival_at_ns")
        verify_started_at = _integer(scheduler_record, "verify_started_at_ns")
        verify_completed_at = _integer(scheduler_record, "verify_completed_at_ns")
        if not (
            started_at
            <= last_draft_at
            <= full_arrival_at
            <= verify_started_at
            <= verify_completed_at
        ):
            raise ValueError(f"non-monotonic scheduler timestamps for {request_id}")

        request_inputs = inputs_by_request.get(request_id, [])
        sparse_records = [
            record
            for record in request_inputs
            if started_at
            <= _integer(record, "recorded_at_ns")
            <= last_draft_at
        ]
        verify_records = [
            record
            for record in request_inputs
            if verify_started_at
            <= _integer(record, "recorded_at_ns")
            <= verify_completed_at
        ]
        if len(sparse_records) != draft_tokens:
            raise ValueError(
                f"sparse forward count mismatch for {request_id}: "
                f"{len(sparse_records)} != {draft_tokens}"
            )
        expected_sparse_inputs = [seed, *drafts[:-1]]
        actual_sparse_inputs: list[int] = []
        sparse_positions: list[int] = []
        for record in sparse_records:
            tokens = _integer_list(record, "input_token_ids")
            positions = _integer_list(record, "positions")
            if len(tokens) != 1 or len(positions) != 1:
                raise ValueError(
                    f"sparse forward was not single-token for {request_id}"
                )
            actual_sparse_inputs.append(tokens[0])
            sparse_positions.append(positions[0])
        if actual_sparse_inputs != expected_sparse_inputs:
            raise ValueError(
                f"sparse token chain mismatch for {request_id}: "
                f"{actual_sparse_inputs} != {expected_sparse_inputs}"
            )
        if not _contiguous(sparse_positions):
            raise ValueError(f"sparse positions are not contiguous for {request_id}")

        if len(verify_records) != 1:
            raise ValueError(
                f"verify forward count mismatch for {request_id}: "
                f"{len(verify_records)} != 1"
            )
        verify_tokens = _integer_list(verify_records[0], "input_token_ids")
        verify_positions = _integer_list(verify_records[0], "positions")
        expected_verify_tokens = [seed, *drafts]
        if verify_tokens != expected_verify_tokens:
            raise ValueError(
                f"verify token batch mismatch for {request_id}: "
                f"{verify_tokens} != {expected_verify_tokens}"
            )
        if not _contiguous(verify_positions):
            raise ValueError(f"verify positions are not contiguous for {request_id}")
        if verify_positions[0] != sparse_positions[0]:
            raise ValueError(f"verify rewind position mismatch for {request_id}")

        total_sparse_forwards += len(sparse_records)
        total_verify_forwards += 1
        request_reports.append(
            {
                "request_id": request_id,
                "draft_tokens": draft_tokens,
                "prompt_tokens": sparse_positions[0],
                "sparse_input_chain_exact": True,
                "sparse_positions_contiguous": True,
                "verify_input_batch_exact": True,
                "verify_positions_contiguous": True,
                "verify_rewinds_to_seed_position": True,
            }
        )

    return {
        "schema_version": 1,
        "status": "passed",
        "expected_requests": expected_requests,
        "verified_requests": len(request_reports),
        "total_sparse_forwards": total_sparse_forwards,
        "total_verify_forwards": total_verify_forwards,
        "requests": request_reports,
    }


def main() -> None:
    args = parse_args()
    report = verify_input_trace(
        _load_jsonl(args.scheduler_stats_jsonl),
        _load_jsonl(args.input_stats_jsonl),
        expected_requests=args.expected_requests,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
