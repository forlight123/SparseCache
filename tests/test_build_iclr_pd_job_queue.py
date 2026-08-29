from experiments.build_iclr_pd_job_queue import fractions_for_bundles


def test_fractions_for_bundles_is_cumulative_and_exact_at_boundaries():
    values = [float(value) for value in fractions_for_bundles(0.02, 4).split(",")]
    assert len(values) == 5
    assert values[0] == 0.02
    assert values[-1] == 1.0
    assert all(left < right for left, right in zip(values, values[1:]))
