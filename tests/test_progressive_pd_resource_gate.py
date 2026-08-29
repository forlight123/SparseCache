from experiments.progressive_pd_resource_gate import (
    evaluate_exclusivity,
    inspect_gpus,
    parse_compute_query,
    parse_gpu_query,
)

GPU_QUERY = """0, GPU-a, NVIDIA H200 NVL, 12, 143771, 0, P8, 31, 70.5
1, GPU-b, NVIDIA H200 NVL, 120000, 143771, 99, P0, 72, 650.0
"""
PROCESS_QUERY = "GPU-b, 1234, VLLM::EngineCore, 119000\n"


def test_resource_query_parsers_and_inspection() -> None:
    commands = []

    def runner(command: list[str]) -> str:
        commands.append(command)
        return GPU_QUERY if "--query-gpu=" in command[1] else PROCESS_QUERY

    states = inspect_gpus(command_runner=runner)

    assert len(parse_gpu_query(GPU_QUERY)) == 2
    assert parse_compute_query(PROCESS_QUERY)["GPU-b"][0]["pid"] == 1234
    assert len(commands) == 2
    assert not states[0].compute_processes
    assert states[1].compute_processes[0]["process_name"] == "VLLM::EngineCore"


def test_exclusive_gate_passes_only_idle_selected_gpus() -> None:
    outputs = iter((GPU_QUERY, PROCESS_QUERY))
    states = inspect_gpus(command_runner=lambda _: next(outputs))

    passed = evaluate_exclusivity(
        states,
        (0,),
        max_used_memory_mib=1024,
        max_utilization_pct=5,
    )
    failed = evaluate_exclusivity(
        states,
        (1,),
        max_used_memory_mib=1024,
        max_utilization_pct=5,
    )

    assert passed["status"] == "passed"
    assert failed["status"] == "failed"
    assert {item["reason"] for item in failed["failures"]} == {
        "compute_processes_present",
        "memory_used_above_threshold",
        "utilization_above_threshold",
    }
