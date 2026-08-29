import argparse
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))

from evaluate_eagle3_quality import bootstrap_mean_ci, percentile


def test_percentile_and_bootstrap_are_deterministic():
    assert percentile([0.0, 10.0], 0.25) == pytest.approx(2.5)
    assert bootstrap_mean_ci([1.0], samples=10, seed=7) == [1.0, 1.0]
    assert bootstrap_mean_ci([0.0, 1.0], samples=100, seed=7) == (
        bootstrap_mean_ci([0.0, 1.0], samples=100, seed=7)
    )


def test_percentile_rejects_empty_input():
    with pytest.raises(ValueError, match="non-empty"):
        percentile([], 0.5)
