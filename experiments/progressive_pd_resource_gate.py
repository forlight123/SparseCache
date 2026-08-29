"""Exclusive-GPU resource gate for latency-valid progressive P/D runs."""

from __future__ import annotations

import argparse
import csv
import io
import json
import subprocess
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

GPU_QUERY_FIELDS = (
    "index",
    "uuid",
    "name",
    "memory.used",
    "memory.total",
    "utilization.gpu",
    "pstate",
    "temperature.gpu",
    "power.draw",
)


@dataclass(frozen=True)
class GPUState:
    index: int
    uuid: str
    name: str
    memory_used_mib: float
    memory_total_mib: float
    utilization_pct: float
    pstate: str
    temperature_c: float
    power_w: float
    compute_processes: tuple[dict[str, Any], ...]


def parse_gpu_query(raw: str) -> list[dict[str, Any]]:
    rows = []
    for row in csv.reader(io.StringIO(raw.strip())):
        if not row:
            continue
        if len(row) != 9:
            raise ValueError(f"unexpected nvidia-smi GPU row: {row}")
        rows.append(
            {
                "index": int(row[0].strip()),
                "uuid": row[1].strip(),
                "name": row[2].strip(),
                "memory_used_mib": float(row[3].strip()),
                "memory_total_mib": float(row[4].strip()),
                "utilization_pct": float(row[5].strip()),
                "pstate": row[6].strip(),
                "temperature_c": float(row[7].strip()),
                "power_w": float(row[8].strip()),
            }
        )
    if not rows:
        raise ValueError("nvidia-smi returned no GPUs")
    return rows


def parse_compute_query(raw: str) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    cleaned = raw.strip()
    if not cleaned or cleaned.startswith("No running processes found"):
        return result
    for row in csv.reader(io.StringIO(cleaned)):
        if not row:
            continue
        if len(row) != 4:
            raise ValueError(f"unexpected nvidia-smi process row: {row}")
        uuid = row[0].strip()
        used = row[3].strip()
        result.setdefault(uuid, []).append(
            {
                "pid": int(row[1].strip()),
                "process_name": row[2].strip(),
                "used_gpu_memory_mib": (
                    None if used in {"N/A", "[N/A]"} else float(used)
                ),
            }
        )
    return result


def _run(command: list[str]) -> str:
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    return result.stdout


def inspect_gpus(
    *, command_runner: Callable[[list[str]], str] = _run
) -> list[GPUState]:
    gpu_rows = parse_gpu_query(
        command_runner(
            [
                "nvidia-smi",
                f"--query-gpu={','.join(GPU_QUERY_FIELDS)}",
                "--format=csv,noheader,nounits",
            ]
        )
    )
    process_rows = parse_compute_query(
        command_runner(
            [
                "nvidia-smi",
                "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory",
                "--format=csv,noheader,nounits",
            ]
        )
    )
    return [
        GPUState(
            **row,
            compute_processes=tuple(process_rows.get(row["uuid"], [])),
        )
        for row in gpu_rows
    ]


def evaluate_exclusivity(
    states: list[GPUState],
    requested_indices: tuple[int, ...],
    *,
    max_used_memory_mib: float,
    max_utilization_pct: float,
) -> dict[str, Any]:
    if not requested_indices or len(requested_indices) != len(set(requested_indices)):
        raise ValueError("requested GPU indices must be a non-empty unique list")
    by_index = {state.index: state for state in states}
    missing = [index for index in requested_indices if index not in by_index]
    if missing:
        raise ValueError(f"requested GPUs do not exist: {missing}")
    failures = []
    selected = []
    for index in requested_indices:
        state = by_index[index]
        selected.append(asdict(state))
        if state.compute_processes:
            failures.append(
                {
                    "gpu": index,
                    "reason": "compute_processes_present",
                    "processes": list(state.compute_processes),
                }
            )
        if state.memory_used_mib > max_used_memory_mib:
            failures.append(
                {
                    "gpu": index,
                    "reason": "memory_used_above_threshold",
                    "observed_mib": state.memory_used_mib,
                    "threshold_mib": max_used_memory_mib,
                }
            )
        if state.utilization_pct > max_utilization_pct:
            failures.append(
                {
                    "gpu": index,
                    "reason": "utilization_above_threshold",
                    "observed_pct": state.utilization_pct,
                    "threshold_pct": max_utilization_pct,
                }
            )
    return {
        "schema_version": 1,
        "status": "passed" if not failures else "failed",
        "checked_at_unix_ns": time.time_ns(),
        "requested_gpu_indices": list(requested_indices),
        "thresholds": {
            "max_used_memory_mib": max_used_memory_mib,
            "max_utilization_pct": max_utilization_pct,
            "compute_processes_allowed": 0,
        },
        "selected_gpus": selected,
        "failures": failures,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", required=True)
    parser.add_argument("--max-used-memory-mib", type=float, default=1024.0)
    parser.add_argument("--max-utilization-pct", type=float, default=5.0)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    requested = tuple(
        int(item.strip()) for item in args.gpus.split(",") if item.strip()
    )
    result = evaluate_exclusivity(
        inspect_gpus(),
        requested,
        max_used_memory_mib=args.max_used_memory_mib,
        max_utilization_pct=args.max_utilization_pct,
    )
    if args.output is not None:
        if args.output.exists():
            raise FileExistsError(f"refusing to overwrite {args.output}")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if result["status"] != "passed":
        raise SystemExit("exclusive-GPU resource gate failed")


if __name__ == "__main__":
    main()
