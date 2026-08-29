"""Safely launch one preregistered live SparseCache-PD job."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import socket
import subprocess
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from experiments.progressive_pd_resource_gate import (
        evaluate_exclusivity,
        inspect_gpus,
    )
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from progressive_pd_resource_gate import evaluate_exclusivity, inspect_gpus

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_QUEUE = ROOT / "outputs/progressive_kv/live_queue_llama31_8b_20260827.json"
DEFAULT_MODEL = "/data/models/llama/Llama-3.1-8B-Instruct"
DEFAULT_PYTHON = "/home/ytm/algorithm/kvreuse/LMCache/.venv-vllm/bin/python"
DEFAULT_BIN = "/home/ytm/algorithm/kvreuse/LMCache/.venv-vllm/bin"


@dataclass
class ManagedProcess:
    name: str
    process: subprocess.Popen
    log_handle: Any
    log_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queue", type=Path, default=DEFAULT_QUEUE)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--prefill-gpu", type=int, required=True)
    parser.add_argument("--decode-gpu", type=int, required=True)
    parser.add_argument(
        "--model",
        help="optional checkpoint override; defaults to the queue's frozen model path",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--lmcache-port", type=int, default=5555)
    parser.add_argument("--prefill-port", type=int, default=8100)
    parser.add_argument("--decoder-port", type=int, default=8200)
    parser.add_argument("--proxy-port", type=int, default=8000)
    parser.add_argument("--telemetry-port", type=int, default=5768)
    parser.add_argument("--l1-size-gb", type=int, default=128)
    parser.add_argument("--startup-timeout-seconds", type=float, default=900.0)
    parser.add_argument("--shutdown-timeout-seconds", type=float, default=30.0)
    parser.add_argument("--resource-monitor-interval-seconds", type=float, default=1.0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-resource-gate", action="store_true")
    parser.add_argument(
        "--audit-input-trace",
        action="store_true",
        help=(
            "record decoder GPU inputs and fail unless the sparse draft and "
            "full-KV verify token/position chains are exact; diagnostic runs only"
        ),
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_job(queue_path: Path, job_id: str) -> tuple[dict[str, Any], str]:
    queue = json.loads(queue_path.read_text(encoding="utf-8"))
    matches = [job for job in queue["jobs"] if job["id"] == job_id]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one queue job {job_id}, got {len(matches)}")
    job = matches[0]
    if job["status"] != "ready_when_two_gpus_are_exclusive":
        raise ValueError(f"job is not ready: {job['status']}")
    if not isinstance(job["bandwidth_gbps"], int):
        raise TypeError("ready job requires a concrete integer bandwidth")
    return job, sha256(queue_path)


def validate_pack_hashes(job: dict[str, Any]) -> dict[str, str]:
    pack_dir = ROOT / job["pack_dir"]
    paths = {
        "requests": pack_dir / "requests.jsonl",
        "metadata": pack_dir / "metadata.jsonl",
    }
    observed = {name: sha256(path) for name, path in paths.items()}
    expected = job["pack_hashes"]
    if observed["requests"] != expected["requests_sha256"]:
        raise ValueError("queued request-pack hash does not match current file")
    if observed["metadata"] != expected["metadata_sha256"]:
        raise ValueError("queued metadata hash does not match current file")
    for mode, sidecar in job.get("priority_sidecars", {}).items():
        path = ROOT / sidecar["path"]
        observed_hash = sha256(path)
        if observed_hash != sidecar["sha256"]:
            raise ValueError(
                f"queued {mode} priority-sidecar hash does not match current file"
            )
        observed[f"priority:{mode}"] = observed_hash
    return observed


def _connector_config(extra: dict[str, Any], role: str) -> str:
    if role not in {"kv_producer", "kv_consumer"}:
        raise ValueError(f"invalid disaggregated KV role: {role}")
    config = {
        "kv_connector": "LMCacheMPConnector",
        "kv_role": role,
        "kv_connector_extra_config": extra,
    }
    return json.dumps(config, separators=(",", ":"))


def build_launch_spec(
    job: dict[str, Any],
    *,
    model: str,
    host: str,
    lmcache_port: int,
    prefill_port: int,
    decoder_port: int,
    proxy_port: int,
    telemetry_port: int,
    prefill_gpu: int,
    decode_gpu: int,
    l1_size_gb: int,
    audit_input_trace: bool = False,
) -> dict[str, Any]:
    if prefill_gpu == decode_gpu:
        raise ValueError("prefill and decode require distinct GPUs")
    if l1_size_gb <= 0:
        raise ValueError("LMCache L1 size must be positive")
    run_dir = ROOT / job["run_dir"]
    completion = run_dir / "completion.json"
    attention = run_dir / "attention.jsonl"
    scheduler = run_dir / "scheduler.jsonl"
    input_trace = run_dir / "input_tokens.jsonl"
    input_audit = run_dir / "input_audit.json"
    prefiller_extra = {
        "lmcache.mp.host": f"tcp://{host}",
        "lmcache.mp.port": lmcache_port,
    }
    decoder_extra = {
        **prefiller_extra,
        **job["decoder_connector_extra_config"],
        "lmcache.mp.progressive_completion_path": str(completion),
        "lmcache.mp.controlled_link_stats_path": str(run_dir / "link.jsonl"),
    }
    queued_model = job.get("model", {})
    served_model_name = queued_model.get(
        "served_model_name", "sparsecache-llama31-8b"
    )
    max_model_len = int(queued_model.get("max_position_embeddings", 131072))
    common_vllm = [
        "--served-model-name",
        served_model_name,
        "--host",
        host,
        "--block-size",
        "64",
        "--max-model-len",
        str(max_model_len),
        "--max-num-seqs",
        "1",
        "--no-async-scheduling",
        "--no-enable-prefix-caching",
        "--enforce-eager",
        "--dtype",
        "bfloat16",
    ]
    lmcache = [
        f"{DEFAULT_BIN}/lmcache",
        "server",
        "--host",
        host,
        "--port",
        str(lmcache_port),
        "--l1-size-gb",
        str(l1_size_gb),
        "--eviction-policy",
        "LRU",
    ]
    prefiller = [
        f"{DEFAULT_BIN}/vllm",
        "serve",
        model,
        "--port",
        str(prefill_port),
        *common_vllm,
        "--kv-transfer-config",
        _connector_config(prefiller_extra, job["pd_roles"]["prefiller"]),
    ]
    decoder = [
        f"{DEFAULT_BIN}/vllm",
        "serve",
        model,
        "--port",
        str(decoder_port),
        *common_vllm,
        "--attention-backend",
        "PROGRESSIVE_KV",
        "--speculative-config",
        json.dumps(
            {
                "method": "custom_class",
                "model": (
                    "vllm.v1.spec_decode.progressive_shared_proposer."
                    "ProgressiveSharedKVProposer"
                ),
                "num_speculative_tokens": 32,
            },
            separators=(",", ":"),
        ),
        "--kv-transfer-config",
        _connector_config(decoder_extra, job["pd_roles"]["decoder"]),
    ]
    proxy = [
        DEFAULT_PYTHON,
        "experiments/progressive_pd_proxy.py",
        "--host",
        host,
        "--port",
        str(proxy_port),
        "--telemetry-port",
        str(telemetry_port),
        "--prefiller-url",
        f"http://{host}:{prefill_port}",
        "--decoder-url",
        f"http://{host}:{decoder_port}",
        "--default-mode",
        "progressive",
        "--max-draft-tokens",
        str(job["proxy_args"]["max_draft_tokens"]),
        "--start-fraction",
        str(job["proxy_args"]["start_fraction"]),
    ]
    benchmark = list(job["benchmark_command"])
    benchmark.extend(["--url", f"http://{host}:{proxy_port}"])
    aggregate = list(job["aggregate_command"])
    base_env = {
        "PYTHONPATH": f"{ROOT / 'vllm'}:{ROOT / 'LMCache'}",
        "LMCACHE_DISABLE_BANNER": "1",
    }
    cpu_env = {**base_env, "CUDA_VISIBLE_DEVICES": ""}
    # The LMCache server reconstructs CUDA IPC handles and resolves the
    # producer/consumer device UUIDs. It therefore must see both worker GPUs
    # even though it does not own model execution on either device.
    lmcache_env = {
        **base_env,
        "CUDA_VISIBLE_DEVICES": f"{prefill_gpu},{decode_gpu}",
    }
    decoder_env = {
        **base_env,
        "CUDA_VISIBLE_DEVICES": str(decode_gpu),
        "VLLM_PROGRESSIVE_KV_DOC_START_TOKEN": "0",
        "VLLM_PROGRESSIVE_KV_DOC_END_TOKEN": str(max_model_len),
        "VLLM_PROGRESSIVE_KV_PROTECTED_PREFIX_TOKENS": str(
            job["decoder_environment"].get(
                "VLLM_PROGRESSIVE_KV_PROTECTED_PREFIX_TOKENS", "0"
            )
        ),
        "VLLM_PROGRESSIVE_KV_PROTECTED_SUFFIX_TOKENS": str(
            job["decoder_environment"].get(
                "VLLM_PROGRESSIVE_KV_PROTECTED_SUFFIX_TOKENS", "0"
            )
        ),
        "VLLM_PROGRESSIVE_KV_COMPLETION_PATH": str(completion),
        "VLLM_PROGRESSIVE_KV_STATS_PATH": str(attention),
        "VLLM_PROGRESSIVE_PD_STATS_PATH": str(scheduler),
    }
    if audit_input_trace:
        decoder_env["VLLM_PROGRESSIVE_PD_INPUT_STATS_PATH"] = str(input_trace)
    progressive_arms = sum(arm != "baseline" for arm in job["arms"])
    schedule_conditions = len(job.get("schedule_modes", ())) or 1
    expected_progressive_requests = (
        int(job["requests"]) + int(job["warmup_requests"])
    ) * progressive_arms * schedule_conditions
    return {
        "run_dir": str(run_dir),
        "commands": {
            "lmcache": lmcache,
            "prefiller": prefiller,
            "decoder": decoder,
            "proxy": proxy,
            "benchmark": benchmark,
            "aggregate": aggregate,
            "input_audit": [
                DEFAULT_PYTHON,
                "experiments/verify_progressive_pd_input_trace.py",
                "--scheduler-stats-jsonl",
                str(scheduler),
                "--input-stats-jsonl",
                str(input_trace),
                "--output",
                str(input_audit),
                "--expected-requests",
                str(expected_progressive_requests),
            ],
        },
        "environments": {
            "lmcache": lmcache_env,
            "prefiller": {
                **base_env,
                "CUDA_VISIBLE_DEVICES": str(prefill_gpu),
                "LMCACHE_REQUEST_TELEMETRY_TYPE": "fastapi",
                "LMCACHE_REQUEST_TELEMETRY_ENDPOINT": (
                    f"http://{host}:{telemetry_port}/api/v1/telemetry"
                ),
            },
            "decoder": decoder_env,
            "proxy": cpu_env,
            "benchmark": cpu_env,
            "aggregate": cpu_env,
            "input_audit": cpu_env,
        },
        "health": {
            "lmcache_tcp": [host, lmcache_port],
            "prefiller": f"http://{host}:{prefill_port}/health",
            "decoder": f"http://{host}:{decoder_port}/health",
            "proxy": f"http://{host}:{proxy_port}/v1/models",
        },
        "ports": [lmcache_port, prefill_port, decoder_port, proxy_port, telemetry_port],
    }


def assert_ports_free(host: str, ports: list[int]) -> None:
    if len(ports) != len(set(ports)):
        raise ValueError("service ports must be unique")
    for port in ports:
        with socket.socket() as candidate:
            candidate.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                candidate.bind((host, port))
            except OSError as error:
                raise RuntimeError(f"service port is unavailable: {host}:{port}") from error


def _spawn(
    name: str,
    command: list[str],
    environment: dict[str, str],
    run_dir: Path,
) -> ManagedProcess:
    log_path = run_dir / f"{name}.log"
    log_handle = log_path.open("w", encoding="utf-8")
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        env={**os.environ, **environment},
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    return ManagedProcess(name, process, log_handle, log_path)


def _tail(path: Path, characters: int = 4000) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    return text[-characters:]


def _wait_tcp(
    host: str,
    port: int,
    process: ManagedProcess,
    timeout: float,
) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.process.poll() is not None:
            raise RuntimeError(
                f"{process.name} exited during startup:\n{_tail(process.log_path)}"
            )
        with socket.socket() as client:
            client.settimeout(1.0)
            if client.connect_ex((host, port)) == 0:
                return
        time.sleep(1.0)
    raise TimeoutError(f"timed out waiting for {process.name} at {host}:{port}")


_FATAL_STARTUP_MARKERS = (
    "Error handling request RequestType.REGISTER_KV_CACHE",
    "not found in the discovered devices",
)


def _raise_on_fatal_startup_log(processes: list[ManagedProcess]) -> None:
    for managed in processes:
        tail = _tail(managed.log_path, characters=16_000)
        marker = next(
            (candidate for candidate in _FATAL_STARTUP_MARKERS if candidate in tail),
            None,
        )
        if marker is not None:
            raise RuntimeError(
                f"{managed.name} reported fatal startup marker {marker!r}:\n{tail}"
            )


def _wait_http(
    url: str,
    process: ManagedProcess,
    timeout: float,
    dependencies: list[ManagedProcess] | None = None,
) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        startup_processes = [process, *(dependencies or [])]
        _raise_on_fatal_startup_log(startup_processes)
        if process.process.poll() is not None:
            raise RuntimeError(
                f"{process.name} exited during startup:\n{_tail(process.log_path)}"
            )
        try:
            with urllib.request.urlopen(url, timeout=2.0) as response:
                if 200 <= response.status < 300:
                    return
        except (urllib.error.URLError, TimeoutError):
            pass
        time.sleep(2.0)
    raise TimeoutError(f"timed out waiting for {process.name} at {url}")


def _stop(processes: list[ManagedProcess], timeout: float) -> None:
    for managed in reversed(processes):
        if managed.process.poll() is None:
            os.killpg(managed.process.pid, signal.SIGTERM)
    deadline = time.monotonic() + timeout
    for managed in reversed(processes):
        remaining = max(0.0, deadline - time.monotonic())
        try:
            managed.process.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            if managed.process.poll() is None:
                os.killpg(managed.process.pid, signal.SIGKILL)
                managed.process.wait(timeout=5)
        managed.log_handle.close()


def _parent_pid(pid: int) -> int | None:
    try:
        raw = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        return None
    closing = raw.rfind(")")
    if closing < 0:
        return None
    fields = raw[closing + 2 :].split()
    return int(fields[1]) if len(fields) > 1 else None


def _is_descendant_process(
    pid: int,
    roots: set[int],
    *,
    parent_lookup: Callable[[int], int | None] = _parent_pid,
) -> bool:
    seen = set()
    current = pid
    while current > 1 and current not in seen:
        if current in roots:
            return True
        seen.add(current)
        parent = parent_lookup(current)
        if parent is None:
            return False
        current = parent
    return current in roots


def _foreign_compute_processes(
    states: list[Any],
    requested_indices: tuple[int, ...],
    service_roots: set[int],
    *,
    parent_lookup: Callable[[int], int | None] = _parent_pid,
) -> list[dict[str, Any]]:
    selected = {state.index: state for state in states}
    foreign = []
    for index in requested_indices:
        state = selected.get(index)
        if state is None:
            foreign.append({"gpu": index, "reason": "gpu_disappeared"})
            continue
        for process in state.compute_processes:
            pid = int(process["pid"])
            if not _is_descendant_process(
                pid, service_roots, parent_lookup=parent_lookup
            ):
                foreign.append(
                    {
                        "gpu": index,
                        "reason": "foreign_compute_process",
                        **process,
                    }
                )
    return foreign


def _run_logged(
    name: str,
    command: list[str],
    environment: dict[str, str],
    run_dir: Path,
    *,
    monitor: Callable[[], dict[str, Any] | None] | None = None,
    monitor_interval_seconds: float = 1.0,
) -> None:
    log_path = run_dir / f"{name}.log"
    with log_path.open("w", encoding="utf-8") as output:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env={**os.environ, **environment},
            stdout=output,
            stderr=subprocess.STDOUT,
            text=True,
        )
        monitor_failure = None
        while process.poll() is None:
            if monitor is not None and (monitor_failure := monitor()) is not None:
                process.terminate()
                try:
                    process.wait(timeout=10.0)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                break
            time.sleep(monitor_interval_seconds)
        returncode = process.wait()
    if monitor_failure is not None:
        raise RuntimeError(
            "runtime exclusive-GPU monitor detected interference: "
            f"{json.dumps(monitor_failure, ensure_ascii=False)}"
        )
    if returncode:
        raise RuntimeError(f"{name} failed:\n{_tail(log_path)}")


def _write_manifest(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def execute(args: argparse.Namespace) -> None:
    job, queue_hash = load_job(args.queue, args.job_id)
    pack_hashes = validate_pack_hashes(job)
    spec = build_launch_spec(
        job,
        model=args.model or job.get("model", {}).get("path", DEFAULT_MODEL),
        host=args.host,
        lmcache_port=args.lmcache_port,
        prefill_port=args.prefill_port,
        decoder_port=args.decoder_port,
        proxy_port=args.proxy_port,
        telemetry_port=args.telemetry_port,
        prefill_gpu=args.prefill_gpu,
        decode_gpu=args.decode_gpu,
        l1_size_gb=args.l1_size_gb,
        audit_input_trace=args.audit_input_trace,
    )
    if args.skip_resource_gate and not args.dry_run:
        raise ValueError("--skip-resource-gate is allowed only with --dry-run")
    resource_gate = None
    if not args.skip_resource_gate:
        resource_gate = evaluate_exclusivity(
            inspect_gpus(),
            (args.prefill_gpu, args.decode_gpu),
            max_used_memory_mib=1024,
            max_utilization_pct=5,
        )
        if resource_gate["status"] != "passed":
            print(json.dumps(resource_gate, indent=2, ensure_ascii=False))
            raise RuntimeError("exclusive-GPU resource gate failed")
    assert_ports_free(args.host, spec["ports"])
    run_dir = Path(spec["run_dir"])
    if run_dir.exists():
        raise FileExistsError(f"refusing to reuse run directory {run_dir}")
    launch_record = {
        "schema_version": 1,
        "status": "dry_run" if args.dry_run else "starting",
        "job": job,
        "queue": str(args.queue.resolve()),
        "queue_sha256": queue_hash,
        "pack_hashes_observed": pack_hashes,
        "resource_gate": resource_gate,
        "launch_spec": spec,
        "started_at_unix_ns": time.time_ns(),
    }
    if args.dry_run:
        print(json.dumps(launch_record, indent=2, ensure_ascii=False))
        return

    run_dir.mkdir(parents=True)
    manifest_path = run_dir / "run_manifest.json"
    _write_manifest(manifest_path, launch_record)
    managed: list[ManagedProcess] = []
    try:
        managed.append(
            _spawn(
                "lmcache",
                spec["commands"]["lmcache"],
                spec["environments"]["lmcache"],
                run_dir,
            )
        )
        _wait_tcp(args.host, args.lmcache_port, managed[-1], args.startup_timeout_seconds)
        for name in ("prefiller", "decoder"):
            managed.append(
                _spawn(
                    name,
                    spec["commands"][name],
                    spec["environments"][name],
                    run_dir,
                )
            )
        _wait_http(
            spec["health"]["prefiller"],
            managed[-2],
            args.startup_timeout_seconds,
            dependencies=managed,
        )
        _wait_http(
            spec["health"]["decoder"],
            managed[-1],
            args.startup_timeout_seconds,
            dependencies=managed,
        )
        managed.append(
            _spawn(
                "proxy",
                spec["commands"]["proxy"],
                spec["environments"]["proxy"],
                run_dir,
            )
        )
        _wait_http(spec["health"]["proxy"], managed[-1], 60.0)
        launch_record["status"] = "benchmarking"
        launch_record["service_pids"] = {
            item.name: item.process.pid for item in managed
        }
        runtime_resource_monitor = {
            "schema_version": 1,
            "status": "monitoring",
            "requested_gpu_indices": [args.prefill_gpu, args.decode_gpu],
            "interval_seconds": args.resource_monitor_interval_seconds,
            "service_roots": sorted(item.process.pid for item in managed),
            "checks": 0,
            "failures": [],
        }

        def monitor_resources() -> dict[str, Any] | None:
            runtime_resource_monitor["checks"] += 1
            checked_at = time.time_ns()
            runtime_resource_monitor["last_checked_at_unix_ns"] = checked_at
            foreign = _foreign_compute_processes(
                inspect_gpus(),
                (args.prefill_gpu, args.decode_gpu),
                set(runtime_resource_monitor["service_roots"]),
            )
            if not foreign:
                return None
            failure = {
                "checked_at_unix_ns": checked_at,
                "foreign_processes": foreign,
            }
            runtime_resource_monitor["failures"].append(failure)
            runtime_resource_monitor["status"] = "failed"
            return failure

        _write_manifest(manifest_path, launch_record)
        _run_logged(
            "benchmark",
            spec["commands"]["benchmark"],
            spec["environments"]["benchmark"],
            run_dir,
            monitor=monitor_resources,
            monitor_interval_seconds=args.resource_monitor_interval_seconds,
        )
        runtime_resource_monitor["status"] = "passed"
        (run_dir / "resource_monitor.json").write_text(
            json.dumps(runtime_resource_monitor, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        launch_record["runtime_resource_monitor"] = runtime_resource_monitor
        launch_record["status"] = "benchmark_complete"
        _write_manifest(manifest_path, launch_record)
    except BaseException as error:
        if "runtime_resource_monitor" in locals():
            (run_dir / "resource_monitor.json").write_text(
                json.dumps(
                    runtime_resource_monitor, indent=2, ensure_ascii=False
                )
                + "\n",
                encoding="utf-8",
            )
            launch_record["runtime_resource_monitor"] = runtime_resource_monitor
        launch_record["status"] = "failed"
        launch_record["error"] = repr(error)
        _write_manifest(manifest_path, launch_record)
        raise
    finally:
        _stop(managed, args.shutdown_timeout_seconds)

    if args.audit_input_trace:
        try:
            _run_logged(
                "input_audit",
                spec["commands"]["input_audit"],
                spec["environments"]["input_audit"],
                run_dir,
            )
        except BaseException as error:
            launch_record["status"] = "input_audit_failed"
            launch_record["error"] = repr(error)
            _write_manifest(manifest_path, launch_record)
            raise

    try:
        _run_logged(
            "aggregate",
            spec["commands"]["aggregate"],
            spec["environments"]["aggregate"],
            run_dir,
        )
    except BaseException as error:
        launch_record["status"] = "aggregation_failed"
        launch_record["error"] = repr(error)
        _write_manifest(manifest_path, launch_record)
        raise
    launch_record["status"] = "complete"
    launch_record["completed_at_unix_ns"] = time.time_ns()
    artifact_names = [
        "results.jsonl",
        "scheduler.jsonl",
        "attention.jsonl",
        "link.jsonl",
        "aggregate/summary.json",
        "aggregate/paper_table.csv",
        "resource_monitor.json",
    ]
    if args.audit_input_trace:
        artifact_names.extend(("input_tokens.jsonl", "input_audit.json"))
    launch_record["artifacts"] = {
        name: sha256(run_dir / name)
        for name in artifact_names
    }
    _write_manifest(manifest_path, launch_record)
    print(json.dumps(launch_record, indent=2, ensure_ascii=False))


def main() -> None:
    args = parse_args()
    if min(
        args.startup_timeout_seconds,
        args.shutdown_timeout_seconds,
        args.resource_monitor_interval_seconds,
        args.l1_size_gb,
    ) <= 0:
        raise ValueError("timeouts and L1 size must be positive")
    execute(args)


if __name__ == "__main__":
    main()
