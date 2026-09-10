import pytest

from experiments.lossless_pd.lmcache_pd.benchmark_shape_invariant_flashinfer import (
    parse_positive_ints,
    summarize,
)


def test_parse_positive_ints():
    assert parse_positive_ints("4,8,16") == (4, 8, 16)
    with pytest.raises(ValueError):
        parse_positive_ints("")
    with pytest.raises(ValueError):
        parse_positive_ints("4,0")


def test_summarize_shape_invariant_cells():
    summary = summarize(
        [
            {
                "fixed_split_pages": 64,
                "bitwise_equal": True,
                "speedup_vs_serial_q1": 2.0,
            },
            {
                "fixed_split_pages": 64,
                "bitwise_equal": True,
                "speedup_vs_serial_q1": 8.0,
            },
        ]
    )
    assert summary["64"]["exact_cells"] == 2
    assert summary["64"]["min_speedup_vs_serial_q1"] == 2.0
    assert summary["64"]["geomean_speedup_vs_serial_q1"] == pytest.approx(4.0)
