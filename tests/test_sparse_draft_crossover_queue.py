import json
from pathlib import Path

from experiments.build_sparse_draft_crossover_queue import build_queue

ROOT = Path(__file__).resolve().parents[1]


def test_crossover_queue_covers_context_and_gamma_without_duplicates() -> None:
    matrix = json.loads(
        (ROOT / "configs/iclr_pd_experiment_matrix.json").read_text(
            encoding="utf-8"
        )
    )
    queue = build_queue(matrix)
    jobs = queue["jobs"]
    cells = {(job["context_tokens"], job["draft_tokens"]) for job in jobs}

    assert len(jobs) == len(cells) == 7
    assert {(16384, 8), (32768, 8), (65536, 8), (131008, 8)} <= cells
    assert {(65536, 4), (65536, 16), (65536, 32)} <= cells
    assert all(job["exclusive_gpu_slots"] == 1 for job in jobs)
    assert all("--require-exclusive-gpu" in job["command"] for job in jobs)
    assert all(job["repeats_per_fraction"] == 30 for job in jobs)
    assert queue["confirmation_template"]["repeats_per_fraction"] == 100
    assert queue["confirmation_template"]["fresh_process_required"]
    assert len(queue["profiler_jobs"]) == 1
    profiler = queue["profiler_jobs"][0]
    assert "--emit-nvtx" in profiler["command"]
    assert not profiler["latency_from_this_run_is_publishable"]
    assert "nvtx_gpu_proj_sum,nvtx_kern_sum" in profiler["stats_commands"][0]
