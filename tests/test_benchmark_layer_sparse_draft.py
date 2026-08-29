from experiments.benchmark_layer_sparse_draft import (
    common_prefix_length,
    fraction_order,
    parse_fractions,
    parse_ints,
)


def test_common_prefix_length_stops_at_first_mismatch() -> None:
    assert common_prefix_length([1, 2, 9, 4], [1, 2, 3, 4]) == 2
    assert common_prefix_length([1, 2], [1, 2, 3]) == 2
    assert common_prefix_length([], [1]) == 0


def test_fraction_order_balances_first_position() -> None:
    fractions = (0.05, 1.0)
    assert fraction_order(fractions, 0) == (0.05, 1.0)
    assert fraction_order(fractions, 1) == (1.0, 0.05)
    assert fraction_order(fractions, 2) == (1.0, 0.05)
    assert fraction_order(fractions, 3) == (0.05, 1.0)


def test_parsers_sort_and_validate_controls() -> None:
    assert parse_ints("32,8,16", field="depths") == (8, 16, 32)
    assert parse_fractions("1.0,0.05") == (0.05, 1.0)
