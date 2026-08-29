import json

import pytest

from experiments.benchmark_sparse_draft_crossover import (
    aggregate,
    balanced_fraction_schedule,
    measurement_seed_token,
    parse_fractions,
)


def test_fraction_schedule_is_complete_and_rotated() -> None:
    fractions = parse_fractions("1,.1,.02")
    schedule = balanced_fraction_schedule(fractions, 4)
    assert len(schedule) == 12
    for repeat in range(4):
        assert {
            fraction for row_repeat, _, fraction in schedule if row_repeat == repeat
        } == set(fractions)
    assert [item[2] for item in schedule[:3]] != [
        item[2] for item in schedule[3:6]
    ]
    with pytest.raises(ValueError, match="include"):
        parse_fractions(".02,.1")
    assert measurement_seed_token(128_256, 0) == 1024
    assert measurement_seed_token(128_256, 1) == 1025


def test_aggregate_fails_closed_and_reports_paired_crossover(tmp_path) -> None:
    prompt = {
        "effective_prompt_tokens": 1024,
        "source_prompt_tokens": 1024,
        "logical_bf16_kv_bytes_per_token": 131072,
    }
    fractions = (0.1, 1.0)
    progressive = []
    attention = []
    for repeat in range(3):
        for order_index, fraction in enumerate(fractions):
            request_id = f"progressive-r{repeat:03d}-o{order_index:02d}-f{fraction:.6f}"
            progressive.append(
                {
                    "request_id": request_id,
                    "repeat": repeat,
                    "order_index": order_index,
                    "visible_fraction": fraction,
                    "elapsed_ms": 8.0 if fraction == 0.1 else 12.0,
                    "num_cached_tokens": 1024,
                    "draft_tokens": 4,
                    "token_ids": [1, 2, 3, 4],
                }
            )
    for fraction in fractions:
        for step in range(4):
            attention.append(
                {
                    "request_id": f"mechanism-audit-f{fraction:.6f}",
                    "decode_step": step,
                    "trace_scope": "representative_layer_0",
                    "visible_logical_blocks_sha256": "a" * 64,
                    "visible_pages": 1 if fraction == 0.1 else 10,
                    "candidate_pages": 10,
                    "sparse_seq_len": 128 if fraction == 0.1 else 1024,
                    "full_seq_len": 1024,
                }
            )
    attention_path = tmp_path / "attention.jsonl"
    attention_path.write_text(
        "".join(json.dumps(row) + "\n" for row in attention), encoding="utf-8"
    )
    flash = [
        {
            "repeat": repeat,
            "elapsed_ms": 11.5,
            "num_cached_tokens": 1024,
            "draft_tokens": 4,
            "token_ids": [1, 2, 3, 4],
        }
        for repeat in range(3)
    ]

    summary = aggregate(
        prompt=prompt,
        progressive_rows=progressive,
        flash_rows=flash,
        attention_path=attention_path,
        fractions=fractions,
        repeats=3,
        block_size=64,
    )
    assert summary["status"] == "valid"
    assert summary["decision_metrics"][
        "economic_crossover_observed_at_or_below_10pct"
    ]
    assert summary["progressive"]["0.100000"][
        "paired_saved_vs_100pct_ms"
    ]["bootstrap_95pct_ci"] == [4.0, 4.0]
    assert summary["progressive"]["0.100000"]["logical_kv_payload"][
        "addressed_fraction"
    ] == 0.125
    assert summary["gates"]["timing_trace_isolation"]

    progressive[0]["num_cached_tokens"] = 0
    invalid = aggregate(
        prompt=prompt,
        progressive_rows=progressive,
        flash_rows=flash,
        attention_path=attention_path,
        fractions=fractions,
        repeats=3,
        block_size=64,
    )
    assert invalid["status"] == "invalid"
    assert not invalid["gates"]["prefix_cache_coverage"]
