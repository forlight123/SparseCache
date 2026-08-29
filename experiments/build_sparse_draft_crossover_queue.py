# SPDX-License-Identifier: Apache-2.0
"""Build the frozen exact sparse-draft crossover execution queue."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

VLLM_PYTHON = "/home/ytm/algorithm/kvreuse/LMCache/.venv-vllm/bin/python"
RUNNER = "experiments/benchmark_sparse_draft_crossover.py"
PACK_ROOT = Path("outputs/progressive_kv/pd_packs/llama31_8b/controlled/ruler_qa2")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def crossover_cells(config: dict) -> list[tuple[int, int]]:
    """Return unique (context, gamma) selection cells in frozen order."""
    primary_gamma = int(config["primary_draft_tokens"])
    cells = [
        (int(context), primary_gamma) for context in config["context_tokens"]
    ]
    sweep_context = int(config["gamma_sweep_context_tokens"])
    cells.extend(
        (sweep_context, int(gamma)) for gamma in config["draft_token_sweep"]
    )
    return list(dict.fromkeys(cells))


def build_queue(matrix: dict) -> dict:
    config = matrix["sparse_draft_crossover"]
    model = next(
        item
        for item in matrix["models"]
        if item["name"] == config["primary_model"]
    )
    fractions = ",".join(str(value) for value in config["visible_fraction"])
    jobs = []
    for context, gamma in crossover_cells(config):
        nominal = 131072 if context == 131008 else context
        request_pack = PACK_ROOT / str(nominal) / "evidence_first" / "requests.jsonl"
        if not request_pack.is_file():
            raise FileNotFoundError(f"missing audited crossover pack: {request_pack}")
        repeats = config["selection_repeats_per_fraction"]
        job_id = f"crossover_llama31_8b_c{context}_g{gamma}_r{repeats}"
        output_dir = Path("outputs/progressive_kv/sparse_draft_crossover") / job_id
        jobs.append(
            {
                "id": job_id,
                "phase": "sparse_draft_crossover_selection",
                "status": "ready",
                "purpose": "paired dense-vs-exact-sparse full-model draft cost",
                "context_tokens": context,
                "draft_tokens": gamma,
                "visibility_fractions": config["visible_fraction"],
                "repeats_per_fraction": repeats,
                "exclusive_gpu_slots": 1,
                "input_hashes": {
                    "requests_sha256": sha256_file(request_pack),
                    "model_config_sha256": sha256_file(
                        Path(model["path"]) / "config.json"
                    ),
                },
                "output_dir": str(output_dir),
                "expected_artifacts": [
                    str(output_dir / "summary.json"),
                    str(output_dir / "summary.md"),
                    str(output_dir / "paper_table.csv"),
                ],
                "command": [
                    VLLM_PYTHON,
                    RUNNER,
                    "--model",
                    model["path"],
                    "--requests-jsonl",
                    str(request_pack),
                    "--output-dir",
                    str(output_dir),
                    "--max-context-tokens",
                    str(context),
                    "--visibility-fractions",
                    fractions,
                    "--draft-tokens",
                    str(gamma),
                    "--block-size",
                    str(config["page_tokens"]),
                    "--warmup-repeats",
                    str(config["warmup_repeats"]),
                    "--measurement-repeats",
                    str(repeats),
                    "--cuda-visible-devices",
                    "0",
                    "--require-exclusive-gpu",
                ],
            }
        )
    profile_context = 65536
    profile_gamma = 8
    profile_pack = (
        PACK_ROOT / str(profile_context) / "evidence_first" / "requests.jsonl"
    )
    profile_id = "nsys_crossover_llama31_8b_c65536_g8_r3"
    profile_output = Path("outputs/progressive_kv/sparse_draft_profile") / profile_id
    trace_prefix = Path("outputs/progressive_kv") / profile_id
    profiler_jobs = [
        {
            "id": profile_id,
            "phase": "sparse_draft_kernel_attribution",
            "status": "ready_after_matching_latency_cell_is_valid",
            "latency_from_this_run_is_publishable": False,
            "purpose": "NVTX-projected CUDA kernel attribution for 2/10/100% pages",
            "command": [
                "nsys",
                "profile",
                "--trace=cuda,nvtx,osrt",
                "--sample=none",
                "--cpuctxsw=none",
                "--trace-fork-before-exec=true",
                "--wait=all",
                "--force-overwrite=false",
                "--output",
                str(trace_prefix),
                VLLM_PYTHON,
                RUNNER,
                "--model",
                model["path"],
                "--requests-jsonl",
                str(profile_pack),
                "--output-dir",
                str(profile_output),
                "--max-context-tokens",
                str(profile_context),
                "--visibility-fractions",
                "0.02,0.1,1.0",
                "--draft-tokens",
                str(profile_gamma),
                "--block-size",
                str(config["page_tokens"]),
                "--warmup-repeats",
                "1",
                "--measurement-repeats",
                "3",
                "--cuda-visible-devices",
                "0",
                "--require-exclusive-gpu",
                "--emit-nvtx",
            ],
            "stats_commands": [
                [
                    "nsys",
                    "stats",
                    "--report",
                    "nvtx_gpu_proj_sum,nvtx_kern_sum",
                    "--format",
                    "csv",
                    str(trace_prefix) + ".nsys-rep",
                ]
            ],
            "expected_artifacts": [
                str(trace_prefix) + ".nsys-rep",
                str(profile_output / "summary.json"),
            ],
        }
    ]
    return {
        "schema_version": 1,
        "status": "pre-registered selection queue; not results",
        "protocol": config["protocol"],
        "jobs": jobs,
        "profiler_jobs": profiler_jobs,
        "confirmation_template": {
            "status": "blocked_on_selection_policy_freeze",
            "repeats_per_fraction": config["confirmation_repeats_per_fraction"],
            "contexts": [32768, 65536, 131008],
            "freeze_fields": ["draft_tokens", "visible_fraction_operating_point"],
            "fresh_process_required": True,
            "selection_rounds_must_not_be_reused": True,
        },
        "counts": {
            "ready": len(jobs),
            "profiler_jobs": len(profiler_jobs),
            "confirmation_templates": 1,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", default="configs/iclr_pd_experiment_matrix.json")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    matrix = json.loads(Path(args.matrix).read_text(encoding="utf-8"))
    queue = build_queue(matrix)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(queue, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(queue["counts"]))


if __name__ == "__main__":
    main()
