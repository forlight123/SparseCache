# SPDX-License-Identifier: Apache-2.0
"""Build the pre-registered, executable P/D experiment queue.

The queue deliberately separates selection from confirmation.  Jobs in the
confirmation template must not be materialized until one policy has been
frozen from the selection split.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNNER = "experiments/pd_progressive_kv_pipeline.py"
PYTHON = ".venv/bin/python"
VLLM_PYTHON = "/home/ytm/algorithm/kvreuse/LMCache/.venv-vllm/bin/python"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", default="configs/iclr_pd_experiment_matrix.json")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def fractions_for_bundles(first_fraction, residual_bundles):
    """Return cumulative fractions with one seed and N residual bundles."""
    values = [first_fraction]
    for index in range(1, residual_bundles + 1):
        values.append(
            first_fraction + (1.0 - first_fraction) * index / residual_bundles
        )
    values[-1] = 1.0
    return ",".join(f"{value:.6f}" for value in values)


def common_method_args(*, bandwidth, drafts, fractions, fixed_horizon):
    args = [
        "--transport-gbps",
        str(bandwidth),
        "--eager-serial-wire",
        "--stage-fractions",
        fractions,
        "--draft-cache-mode",
        "continuous",
        "--draft-tokens",
        str(drafts),
        "--verification-mode",
        "final_only",
        "--commit-windows",
        "inf",
        "--draft-stage-policy",
        "first",
        "--reuse-final-verify",
        "--schedules",
        "query",
        "--skip-serial",
        "--require-exclusive-gpu",
    ]
    if fixed_horizon:
        args.append("--fixed-token-horizon")
    return args


def command(*, dataset, dataset_format, model, output, sample_count, count):
    return [
        PYTHON,
        RUNNER,
        "--dataset",
        dataset,
        "--dataset-format",
        dataset_format,
        "--model",
        model,
        "--output",
        output,
        "--sample-count",
        str(sample_count),
        "--count",
        str(count),
    ]


def add_job(jobs, *, job_id, phase, purpose, command_args, **metadata):
    jobs.append(
        {
            "id": job_id,
            "phase": phase,
            "purpose": purpose,
            "status": "ready",
            "command": command_args,
            **metadata,
        }
    )


def main():
    args = parse_args()
    matrix_path = Path(args.matrix)
    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    datasets = {item["name"]: item for item in matrix["datasets"]}
    primary, second = matrix["models"][:2]
    jobs = []

    # Gate 0.5: replace the contaminated QMSum result with one isolated,
    # fixed-horizon, paired full-target comparison.
    qmsum = datasets["qmsum"]
    gate_output = (
        "outputs/progressive_kv/iclr_queue/gate0p5_llama_qmsum_n100_bw100_s002_g6.jsonl"
    )
    gate_command = command(
        dataset=qmsum["path"],
        dataset_format="longbench",
        model=primary["path"],
        output=gate_output,
        sample_count=100,
        count=100,
    )
    gate_command += [
        "--max-new-tokens",
        "64",
        "--context-page-tokens",
        "256",
        "--transport-gbps",
        "100",
        "--eager-serial-wire",
        "--stage-fractions",
        "0.02,1.0",
        "--draft-cache-mode",
        "s1",
        "--draft-tokens",
        "6",
        "--verification-mode",
        "final_only",
        "--commit-windows",
        "inf",
        "--draft-stage-policy",
        "first",
        "--reuse-final-verify",
        "--schedules",
        "query",
        "--fixed-token-horizon",
        "--paired-measurement",
        "--skip-serial",
        "--require-exclusive-gpu",
    ]
    add_job(
        jobs,
        job_id="gate0p5_llama_qmsum_n100_bw100_s002_g6",
        phase="measurement_integrity",
        purpose="replace contaminated fixed-S1 latency cell",
        command_args=gate_command,
        validator=[
            PYTHON,
            "experiments/validate_pd_latency_cell.py",
            gate_output,
            "--expected-count",
            "100",
        ],
    )

    # Kernel gate: target attention stays full while the independent
    # same-checkpoint draft instance uses a genuinely compact block table.
    # This is a mechanism/compute cell, not end-to-end P/D transport timing.
    kernel_job_id = "kernel_llama_qmsum_8k_g8_progressive"
    add_job(
        jobs,
        job_id=kernel_job_id,
        phase="sparse_kernel_mechanism",
        purpose=(
            "measure real sparse draft attention, completion events, and "
            "external immutable-verifier handoff"
        ),
        command_args=[
            VLLM_PYTHON,
            "experiments/vllm_progressive_sparse_draft_probe.py",
            "--dataset",
            qmsum["path"],
            "--model",
            primary["path"],
            "--output-dir",
            f"outputs/progressive_kv/iclr_queue/{kernel_job_id}",
            "--offset",
            "0",
            "--max-prompt-tokens",
            "8192",
            "--max-new-tokens",
            "9",
            "--draft-tokens",
            "8",
            "--block-size",
            "64",
            "--visible-fractions",
            "0.02,0.25,0.5,0.75,1.0",
            "--completion-trace-ms",
            "0,20,40,60,80",
            "--page-order",
            "uniform",
            "--gpu-memory-utilization",
            "0.9",
            "--require-exclusive-gpu",
        ],
    )

    # Gate 2.5 selection: continuous visibility versus an in-process fixed-S1
    # control.  This is the only grid allowed to select gamma/bundle policy.
    focus = matrix["selection_split"]["progressive_focus_cells"]
    for model in (primary, second):
        model_slug = model["name"].lower().replace("-", "_").replace(".", "")
        for bandwidth in focus["transport_gbps"]:
            for drafts in focus["draft_tokens"]:
                for bundles in focus["arrival_bundles"]:
                    fractions = fractions_for_bundles(0.02, bundles)
                    job_id = (
                        f"select_{model_slug}_qmsum_n30_bw{bandwidth}_"
                        f"g{drafts}_b{bundles}"
                    )
                    output = f"outputs/progressive_kv/iclr_queue/{job_id}.jsonl"
                    run = command(
                        dataset=qmsum["path"],
                        dataset_format="longbench",
                        model=model["path"],
                        output=output,
                        sample_count=30,
                        count=30,
                    )
                    run += [
                        "--max-new-tokens",
                        "64",
                        "--context-page-tokens",
                        "256",
                        *common_method_args(
                            bandwidth=bandwidth,
                            drafts=drafts,
                            fractions=fractions,
                            fixed_horizon=True,
                        ),
                        "--paired-fixed-control",
                    ]
                    add_job(
                        jobs,
                        job_id=job_id,
                        phase="continuous_policy_selection",
                        purpose="compare continuous page visibility with fixed S1",
                        command_args=run,
                        model=model["name"],
                        bandwidth_gbps=bandwidth,
                        draft_tokens=drafts,
                        residual_bundles=bundles,
                    )

    # The full quality and controlled-context matrices are emitted as frozen
    # templates.  Their policy fields are intentionally null so a selection
    # result cannot be silently chosen after looking at the confirmation data.
    confirmation_templates = []
    for model in (primary, second):
        for dataset in matrix["datasets"]:
            confirmation_templates.append(
                {
                    "id": f"quality_{model['name']}_{dataset['name']}",
                    "dataset": dataset["path"],
                    "dataset_format": "longbench",
                    "model": model["path"],
                    "sample_count": dataset["available"],
                    "official_output_length": True,
                    "max_context_tokens": 32768,
                }
            )
    for nominal, dataset in matrix["controlled_context"]["paths"].items():
        for placement in matrix["controlled_context"]["evidence_placement"]:
            confirmation_templates.append(
                {
                    "id": f"ruler_llama_{nominal}_{placement}",
                    "dataset": dataset,
                    "dataset_format": "ruler",
                    "model": primary["path"],
                    "sample_count": matrix["controlled_context"]["requests_per_cell"],
                    "ruler_placement": (
                        f"evidence_{placement}"
                        if placement in {"first", "uniform", "last"}
                        else placement
                    ),
                    "official_output_length": True,
                    "max_prompt_tokens": matrix["controlled_context"][
                        "primary_model_max_prompt_tokens"
                    ],
                }
            )
    confirmation_templates.append(
        {
            "id": "longbench_v2_llama_all",
            "dataset": matrix["longbench_v2"]["path"],
            "dataset_format": "longbench_v2",
            "model": primary["path"],
            "sample_count": matrix["longbench_v2"]["available"],
            "official_output_length": True,
            "max_context_tokens": matrix["longbench_v2"][
                "llama31_8b_max_context_tokens"
            ],
        }
    )

    payload = {
        "schema_version": 1,
        "matrix": str(matrix_path.resolve()),
        "selection_is_preregistered": True,
        "ready_jobs": jobs,
        "confirmation_policy": {
            "status": "blocked_on_continuous_policy_selection",
            "freeze_fields": [
                "transport_gbps",
                "draft_tokens",
                "arrival_bundles",
                "priority_fraction",
                "page_tokens",
                "schedule",
            ],
            "templates": confirmation_templates,
        },
        "counts": {
            "ready": len(jobs),
            "measurement_integrity": sum(
                item["phase"] == "measurement_integrity" for item in jobs
            ),
            "continuous_policy_selection": sum(
                item["phase"] == "continuous_policy_selection" for item in jobs
            ),
            "sparse_kernel_mechanism": sum(
                item["phase"] == "sparse_kernel_mechanism" for item in jobs
            ),
            "confirmation_templates": len(confirmation_templates),
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(payload["counts"], ensure_ascii=False))


if __name__ == "__main__":
    main()
