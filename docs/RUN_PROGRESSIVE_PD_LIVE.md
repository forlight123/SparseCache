# Live SparseCache-PD deployment

This is the first 1P1D correctness and latency gate. It uses one LMCache
server, one prefill GPU, one decode GPU, and the project-local vLLM/LMCache
sources. Do not interpret a paced one-host result as a real network result;
the final system gate must repeat the trend on two physical nodes.

## 1. Fixed assumptions

- Model: `/data/models/llama/Llama-3.1-8B-Instruct`.
- Cache dtype: BF16, 131,072 prompt-KV bytes per token.
- One request at a time, prefix caching off, synchronous scheduler.
- Greedy decode only for the first equality gate.
- S1 atomically contains the protected 256-token prefix and 512-token suffix.
  Fractional S1 is rounded upward when necessary; report actual bundle bytes.
- LMCache chunks are 256 tokens; vLLM attention pages are 64 tokens.
- Progressive fractions are cumulative on one serialized wire.
- The first gate uses `uniform` page order. Query-aware ordering is a later
  ablation, not silently substituted into the initial result.

For this model, exact prompt KV is 2/4/8/16 GiB at
16K/32K/64K/128K tokens. The ideal 25 Gbps wire times are approximately
0.687/1.374/2.749/5.498 seconds; 50 Gbps halves them.

## 2. Environment

Run all processes with the project sources before the installed packages:

```bash
export SPARSECACHE_ROOT=/home/ytm/algorithm/kvreuse/SparseCache
export SPARSECACHE_PY=/home/ytm/algorithm/kvreuse/LMCache/.venv-vllm/bin/python
export SPARSECACHE_BIN=/home/ytm/algorithm/kvreuse/LMCache/.venv-vllm/bin
export PYTHONPATH="$SPARSECACHE_ROOT/vllm:$SPARSECACHE_ROOT/LMCache"
export SPARSECACHE_MODEL=/data/models/llama/Llama-3.1-8B-Instruct
export SPARSECACHE_COMPLETION=/tmp/sparsecache-pd-completion.json
export SPARSECACHE_RUN=$SPARSECACHE_ROOT/outputs/progressive_kv/live_25gbps_n100
```

Use a fresh run directory because both decoder traces are append-only:

```bash
test ! -e "$SPARSECACHE_RUN"
mkdir -p "$SPARSECACHE_RUN"
```

Verify the source paths before launching:

```bash
$SPARSECACHE_PY -c 'import inspect,vllm,lmcache; from vllm.v1.core.sched.scheduler import Scheduler; print(inspect.getsourcefile(Scheduler)); print(lmcache.__file__)'
```

The paths must resolve under `SparseCache/vllm` and `SparseCache/LMCache`.

## 3. Launch the four services

Terminal 1, LMCache server:

```bash
$SPARSECACHE_BIN/lmcache server --l1-size-gb 128 --eviction-policy LRU
```

Use 128 GiB for the 64K campaign. A 40 GiB run exposed an L1-eviction race
between producer-store completion and the paired decoder retrieve and was
invalidated; it is retained only as an eviction stress artifact. The runtime
also requires a full remote prompt-KV hit before progressive drafting starts,
so a partial hit fails closed instead of silently recomputing the missing
prefix.

Terminal 2, prefiller on an exclusive GPU:

```bash
CUDA_VISIBLE_DEVICES=0 \
LMCACHE_REQUEST_TELEMETRY_TYPE=fastapi \
LMCACHE_REQUEST_TELEMETRY_ENDPOINT=http://127.0.0.1:5768/api/v1/telemetry \
$SPARSECACHE_BIN/vllm serve "$SPARSECACHE_MODEL" \
  --served-model-name sparsecache-llama31-8b \
  --port 8100 \
  --block-size 64 \
  --max-num-seqs 1 \
  --no-async-scheduling \
  --no-enable-prefix-caching \
  --enforce-eager \
  --kv-transfer-config '{"kv_connector":"LMCacheMPConnector","kv_role":"kv_producer","kv_connector_extra_config":{"lmcache.mp.host":"tcp://127.0.0.1","lmcache.mp.port":5555}}'
```

Terminal 3, decoder. This example emulates a 25 Gbps serialized link; use
`50` for the second preregistered cell and `0`/omit both pacing fields for the
native transport control.

Despite the legacy `progressive_link_gbps` key name, the controlled-link rate
is charged to both the baseline monolithic retrieve and every progressive
bundle. The `controlled_link_stats_path` trace is a mandatory aggregation gate:
all arms must report identical logical tokens, payload bytes, link rate, and
total modeled wire time.

```bash
CUDA_VISIBLE_DEVICES=1 \
VLLM_PROGRESSIVE_KV_DOC_START_TOKEN=0 \
VLLM_PROGRESSIVE_KV_DOC_END_TOKEN=131072 \
VLLM_PROGRESSIVE_KV_PROTECTED_PREFIX_TOKENS=256 \
VLLM_PROGRESSIVE_KV_PROTECTED_SUFFIX_TOKENS=512 \
VLLM_PROGRESSIVE_KV_COMPLETION_PATH="$SPARSECACHE_COMPLETION" \
VLLM_PROGRESSIVE_KV_STATS_PATH="$SPARSECACHE_RUN/attention.jsonl" \
VLLM_PROGRESSIVE_PD_STATS_PATH="$SPARSECACHE_RUN/scheduler.jsonl" \
$SPARSECACHE_BIN/vllm serve "$SPARSECACHE_MODEL" \
  --served-model-name sparsecache-llama31-8b \
  --port 8200 \
  --attention-backend PROGRESSIVE_KV \
  --block-size 64 \
  --max-num-seqs 1 \
  --no-async-scheduling \
  --no-enable-prefix-caching \
  --enforce-eager \
  --speculative-config '{"method":"custom_class","model":"vllm.v1.spec_decode.progressive_shared_proposer.ProgressiveSharedKVProposer","num_speculative_tokens":32}' \
  --kv-transfer-config "{\"kv_connector\":\"LMCacheMPConnector\",\"kv_role\":\"kv_consumer\",\"kv_connector_extra_config\":{\"lmcache.mp.host\":\"tcp://127.0.0.1\",\"lmcache.mp.port\":5555,\"lmcache.mp.progressive_retrieve_fractions\":\"0.05,0.25,0.5,0.75,1.0\",\"lmcache.mp.progressive_retrieve_priority\":\"uniform\",\"lmcache.mp.progressive_protected_prefix_tokens\":256,\"lmcache.mp.progressive_protected_suffix_tokens\":512,\"lmcache.mp.progressive_completion_path\":\"$SPARSECACHE_COMPLETION\",\"lmcache.mp.controlled_link_stats_path\":\"$SPARSECACHE_RUN/link.jsonl\",\"lmcache.mp.progressive_link_gbps\":25,\"lmcache.mp.progressive_bytes_per_token\":131072}}"
```

Terminal 4, proxy and store-completion telemetry receiver:

```bash
$SPARSECACHE_PY experiments/progressive_pd_proxy.py \
  --prefiller-url http://127.0.0.1:8100 \
  --decoder-url http://127.0.0.1:8200 \
  --default-mode progressive \
  --max-draft-tokens 8 \
  --start-fraction 0.05
```

## 4. Three-arm paired gate

Use the audited exact 16,384-token RULER pack. The first four shapes are replayed
as warmups, then rows 0--99 are measured. Supporting documents remain intact; only
distractors are repeated or trimmed to fill the exact LMCache-aligned budget.
The pack and its metadata must already have passed
`experiments/audit_ruler_pd_packs.py`:

```bash
export SPARSECACHE_PACK=$SPARSECACHE_ROOT/outputs/progressive_kv/pd_packs/llama31_8b/controlled/ruler_qa2/16384/evidence_first
```

The benchmark input contains complete OpenAI request bodies, one per line.
The driver rotates all three execution orders, performs warmups, requests token
IDs, and stores the source-row index needed for an exact metadata join:

```bash
$SPARSECACHE_PY experiments/benchmark_progressive_pd_live.py \
  --endpoint /v1/completions \
  --requests-jsonl "$SPARSECACHE_PACK/requests.jsonl" \
  --output-jsonl "$SPARSECACHE_RUN/results.jsonl" \
  --num-requests 100 \
  --warmup-requests 4 \
  --fixed-output-tokens 32 \
  --arms baseline,fixed_s1,continuous
```

Selection jobs use the default `--request-offset 0`; confirmation jobs in the
machine queue explicitly use `--request-offset 100`. Warmups replay rows inside
their own split and never consume or cross the selection/confirmation boundary.

Aggregate only after the decoder has flushed both trace files:

```bash
$SPARSECACHE_PY experiments/aggregate_progressive_pd_live.py \
  --results-jsonl "$SPARSECACHE_RUN/results.jsonl" \
  --metadata-jsonl "$SPARSECACHE_PACK/metadata.jsonl" \
  --scheduler-stats-jsonl "$SPARSECACHE_RUN/scheduler.jsonl" \
  --attention-stats-jsonl "$SPARSECACHE_RUN/attention.jsonl" \
  --link-stats-jsonl "$SPARSECACHE_RUN/link.jsonl" \
  --output-dir "$SPARSECACHE_RUN/aggregate" \
  --expected-requests 100 \
  --fixed-output-tokens 32 \
  --protected-prefix-tokens 256 \
  --protected-suffix-tokens 512
```

The aggregator exits nonzero if any validity gate fails. It requires a complete
three-arm matrix, unique request IDs, exactly 32 tokens per arm, one scheduler
transition per method request, full arrival before one immutable verifier,
request-attributed sparse attention at every draft step, and exact greedy
token equality. The `protected_anchor_arrival` gate also requires S1 to cover
the complete prefix and question suffix before drafting. It emits
`summary.json`, `summary.md`, `paper_table.csv`, and
enriched `paired_rows.jsonl`, including 10,000-sample paired bootstrap CIs.
Report client TTFT separately from `decode_ttft_ms`; the former includes P
prefill/store orchestration.

The attention JSONL intentionally contains one representative layer-zero row
per draft step, not one row per transformer layer. This keeps telemetry off the
critical path while preserving exact range/page counts and the selected-page
set digest. All-layer snapshot agreement is checked in the backend tests.

The same aggregation reconstructs the latency equation independently for
every request. `results.jsonl` contains per-token client arrival times,
`scheduler.jsonl` contains every hidden draft completion plus immutable-
verifier timing, and `link.jsonl` separates the first bundle from the modeled
residual-wire window. The JSON, Markdown, and CSV artifacts report predicted
versus observed gain, model MAE, and gain-sign agreement. Any missing timing
series fails the `latency_model_telemetry` gate.

For the full campaign, use the preregistered machine-readable queue at
`outputs/progressive_kv/live_queue_llama31_8b_20260827.json`. Each job embeds
the audited input hashes, fresh artifact paths, decoder environment, connector
pacing configuration, benchmark command, aggregate command, and two-exclusive-
GPU requirement.

The preferred execution path is the fail-closed launcher. It validates pack
hashes, refuses occupied GPU/port/run-directory state, assigns producer and
consumer roles, waits for service health, runs the client, stops only its own
process group, and then applies all aggregation gates:

```bash
$SPARSECACHE_PY experiments/run_progressive_pd_live_job.py \
  --queue outputs/progressive_kv/live_queue_llama31_8b_20260827.json \
  --job-id core_ruler_65536_evidence_first_bw25_n100 \
  --prefill-gpu 0 \
  --decode-gpu 1 \
  --lmcache-port 15555 \
  --prefill-port 18100 \
  --decoder-port 18200 \
  --proxy-port 18000 \
  --telemetry-port 15768 \
  --l1-size-gb 128
```

Before a large run, add `--audit-input-trace` to a short diagnostic cell. This
enables a decoder-only trace and runs
`experiments/verify_progressive_pd_input_trace.py` automatically after service
shutdown. For every progressive request it requires exactly `gamma`
single-token inputs `[seed, draft_0, ..., draft_{gamma-2}]` at contiguous
positions, followed by one full-verifier batch
`[seed, draft_0, ..., draft_{gamma-1}]` rewound to the seed position. Any
missing, replayed, or shifted token fails the whole run. The trace performs
synchronous JSONL I/O and is therefore a mechanism diagnostic, not a latency
measurement.

The semantic scheduling selection uses the same launcher with
`outputs/progressive_kv/live_scheduling_queue_llama31_8b_20260827.json`.
For example, the 25 Gbps selection cell is
`scheduling_selection_ruler_65536_original_bw25_n100_offset0`. The launcher
hash-validates all five priority sidecars before starting services. Its single
benchmark process rotates five schedules by two visibility arms for each
request. Each source request performs exactly one P prefill; all ten
schedule-by-visibility conditions reuse that same exact KV object and producer
seed. This is a cross-schedule pairing requirement, not merely a two-arm
pairing inside each schedule. The scheduling aggregator rejects a second fresh
prefill, a changed `prefill_group` or seed, missing conditions, output drift,
or any link bundle whose reconstructed chunk set differs from the planned
sidecar slice. Confirmation jobs remain non-runnable until their queue status
is changed after the selection policy is frozen.

To inspect exclusivity without launching anything:

```bash
$SPARSECACHE_PY experiments/progressive_pd_resource_gate.py \
  --gpus 0,1 \
  --max-used-memory-mib 1024 \
  --max-utilization-pct 5
```
