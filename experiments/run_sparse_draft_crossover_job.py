# SPDX-License-Identifier: Apache-2.0
"""Fail-closed launcher for one frozen sparse-draft crossover job."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time
from typing import Any

try:
    from experiments.progressive_pd_resource_gate import (
        evaluate_exclusivity,
        inspect_gpus,
    )
except ModuleNotFoundError:  # Direct script execution from experiments/.
    from progressive_pd_resource_gate import evaluate_exclusivity, inspect_gpus


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_QUEUE = ROOT / "outputs/progressive_kv/sparse_draft_crossover_queue_20260827.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_job(queue_path: Path, job_id: str) -> tuple[dict[str, Any], str]:
    queue_path = queue_path.resolve()
    queue = json.loads(queue_path.read_text(encoding="utf-8"))
    matches = [job for job in queue.get("jobs", []) if job.get("id") == job_id]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one selection job named {job_id!r}")
    job = matches[0]
    if job.get("status") != "ready":
        raise ValueError(f"job is not executable: {job.get('status')!r}")
    return job, sha256_file(queue_path)


def _option(command: list[str], name: str) -> str:
    try:
        return command[command.index(name) + 1]
    except (ValueError, IndexError) as error:
        raise ValueError(f"frozen command is missing {name}") from error


def validate_inputs(job: dict[str, Any]) -> dict[str, str]:
    command = list(job["command"])
    request_path = (ROOT / _option(command, "--requests-jsonl")).resolve()
    model_config = Path(_option(command, "--model")) / "config.json"
    output_dir = (ROOT / job["output_dir"]).resolve()
    command_output = (ROOT / _option(command, "--output-dir")).resolve()
    if output_dir != command_output:
        raise ValueError("job output_dir and command --output-dir differ")
    expected = job.get("input_hashes", {})
    observed = {
        "requests_sha256": sha256_file(request_path),
        "model_config_sha256": sha256_file(model_config),
    }
    if observed != expected:
        raise ValueError(f"input hash mismatch: expected={expected}, observed={observed}")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {output_dir}")
    existing = [
        str((ROOT / path).resolve())
        for path in job["expected_artifacts"]
        if (ROOT / path).exists()
    ]
    if existing:
        raise FileExistsError(f"refusing to overwrite expected artifacts: {existing}")
    return observed


def command_for_gpu(job: dict[str, Any], gpu: int) -> list[str]:
    if gpu < 0:
        raise ValueError("GPU index must be non-negative")
    command = list(job["command"])
    index = command.index("--cuda-visible-devices") + 1
    command[index] = str(gpu)
    if "--require-exclusive-gpu" not in command:
        raise ValueError("frozen command must retain its internal exclusivity gate")
    return command


def validate_outputs(job: dict[str, Any]) -> dict[str, str]:
    observed = {}
    for raw_path in job["expected_artifacts"]:
        path = (ROOT / raw_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"expected artifact was not produced: {path}")
        observed[raw_path] = sha256_file(path)
    summary_path = (ROOT / job["expected_artifacts"][0]).resolve()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("status") != "valid":
        raise RuntimeError(f"crossover summary failed validity gates: {summary_path}")
    return observed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queue", type=Path, default=DEFAULT_QUEUE)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-resource-gate", action="store_true")
    return parser.parse_args()


def execute(args: argparse.Namespace) -> None:
    if args.skip_resource_gate and not args.dry_run:
        raise ValueError("--skip-resource-gate is permitted only with --dry-run")
    job, queue_hash = load_job(args.queue, args.job_id)
    input_hashes = validate_inputs(job)
    command = command_for_gpu(job, args.gpu)
    resource_gate = None
    if not args.skip_resource_gate:
        resource_gate = evaluate_exclusivity(
            inspect_gpus(),
            (args.gpu,),
            max_used_memory_mib=1024.0,
            max_utilization_pct=5.0,
        )
        if resource_gate["status"] != "passed":
            print(json.dumps(resource_gate, indent=2, ensure_ascii=False))
            raise SystemExit("exclusive-GPU resource gate failed")
    record = {
        "schema_version": 1,
        "status": "dry_run" if args.dry_run else "running",
        "job_id": job["id"],
        "queue": str(args.queue.resolve()),
        "queue_sha256": queue_hash,
        "input_hashes_observed": input_hashes,
        "resource_gate": resource_gate,
        "command": command,
        "started_at_unix_ns": time.time_ns(),
    }
    if args.dry_run:
        print(json.dumps(record, indent=2, ensure_ascii=False))
        return
    subprocess.run(command, cwd=ROOT, check=True)
    record["artifacts_sha256"] = validate_outputs(job)
    record["status"] = "completed_valid"
    record["finished_at_unix_ns"] = time.time_ns()
    manifest = (ROOT / job["output_dir"]).resolve() / "execution_manifest.json"
    manifest.write_text(
        json.dumps(record, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(record, indent=2, ensure_ascii=False))


def main() -> None:
    execute(parse_args())


if __name__ == "__main__":
    main()
