from pathlib import Path

import pytest

from experiments.run_sparse_draft_crossover_job import (
    ROOT,
    command_for_gpu,
    load_job,
    validate_inputs,
)


QUEUE = ROOT / "outputs/progressive_kv/sparse_draft_crossover_queue_20260827.json"
JOB_ID = "crossover_llama31_8b_c16384_g8_r30"


def test_sparse_crossover_job_hashes_and_gpu_substitution() -> None:
    job, queue_hash = load_job(QUEUE, JOB_ID)
    observed = validate_inputs(job)
    command = command_for_gpu(job, 2)

    assert len(queue_hash) == 64
    assert observed == job["input_hashes"]
    assert command[command.index("--cuda-visible-devices") + 1] == "2"
    assert "--require-exclusive-gpu" in command
    assert not Path(job["output_dir"]).is_absolute()


def test_sparse_crossover_job_rejects_negative_gpu() -> None:
    job, _ = load_job(QUEUE, JOB_ID)
    with pytest.raises(ValueError, match="non-negative"):
        command_for_gpu(job, -1)
