# Online lossless progressive-KV deployment

This directory contains the real LMCache/vLLM P/D integration for SparseCache.
It leaves the adjacent LMCache checkout unchanged and installs all runtime
patches through `runtime_site/sitecustomize.py`.

The tested one-host topology is:

```text
GPU 0  Qwen3-8B prefiller + LMCache NIXL sender
GPU 1  Qwen3-8B decoder + LMCache NIXL receiver + sparse-KV drafter
GPU 2  monolithic Qwen3-8B paired correctness/latency control
CPU    LMCache disaggregated-prefill request proxy
```

The active path is not EAGLE and does not run a partial Target prefill. The
drafter directly consumes five layers of Target KV from the arrived Anchor,
plus the exact first token sampled by the P Target. No proposed token is exposed
to the client before the normal full-KV Target verifies it.

## Runtime sequence

1. P samples the exact seed through `SeedSignalProposer`. The proposer itself
   predicts zero tokens.
2. `anchor_runtime.py` partitions the normal 256-token LMCache chunks. In
   `protected_uniform` mode it always keeps the first and last chunks and spaces
   the remaining Anchor chunks uniformly through the prompt.
3. P gathers and submits the Anchor first. The Residual gather and NIXL write
   continue without changing LMCache's final completion notification.
4. After remote completion, D resolves the advertised NIXL keys to registered
   CUDA `MemoryObj`s, pins them, and creates basic-slice zero-copy views for
   Target layers `[1,9,17,25,33]`.
5. `online_drafter.py` packs those views and asynchronously precomputes sparse-KV
   hidden states, base top-64 candidates, and the causal rerank state on a
   private CUDA stream.
6. When the authoritative full-Target token `t1` becomes available, the cheap
   target-conditioned repair starts from `t1` and proposes only the suffix
   `q2..qg`. In `inject` mode vLLM verifies that suffix against complete Target
   KV; in `observe` mode it executes the identical draft/repair path but returns
   no proposals.

The expensive sparse-KV work is therefore overlapped with KV movement even when
the first raw draft token is wrong. Only the approximately 3-ms causal repair is
on the critical path after `t1`.

## Required launch contract

Use the adjacent LMCache environment:

```bash
/home/ytm/algorithm/kvreuse/LMCache/.venv-vllm/bin/python
```

The validated stack is Python 3.12, vLLM 0.23.0, PyTorch 2.11 + CUDA 13.0,
LMCache from the adjacent checkout, and NIXL 1.4.1. Both P and D must receive:

```text
PYTHONHASHSEED=0
PYTHONPATH=<SparseCache>:<SparseCache>/experiments/lossless_pd/lmcache_pd/runtime_site:<LMCache>
SPARSECACHE_ANCHOR_PATCH=1              # P
SPARSECACHE_GATHER_FIRST=1              # P
SPARSECACHE_RECEIVER_PATCH=1            # D
SPARSECACHE_ANCHOR_NOTIFY=udp://HOST:PORT       # P
SPARSECACHE_ANCHOR_LISTEN=udp://HOST:PORT       # D
SPARSECACHE_DRAFT_LAYERS=1,9,17,25,33           # D
SPARSECACHE_DRAFTER_CHECKPOINT=<checkpoint>      # D
SPARSECACHE_DRAFT_TOKENS=7                       # D
SPARSECACHE_ONLINE_DRAFT_MODE=observe|inject     # D
SPARSECACHE_DRAFT_TRACE=<trace.jsonl>            # D
```

P must use `kv_role=kv_producer`; D must use `kv_role=kv_consumer`. The model,
dtype, LMCache chunk size, hash policy, and maximum context must match. Using
`kv_both` on D or omitting `PYTHONHASHSEED=0` can silently turn the experiment
into local recomputation instead of transferred-KV reuse.

The online proposer class is
`experiments.lossless_pd.lmcache_pd.online_drafter.OnlineSparseKVProposer`.
This prototype's losslessness contract is greedy decoding only. Sampling needs
the repaired proposal probabilities and a sampling-aware acceptance rule before
it may be claimed lossless.

For exact frozen-packet replay, launch the CPU proxy through:

```bash
PYTHONPATH="$PWD:/home/ytm/algorithm/kvreuse/LMCache" \
  /home/ytm/algorithm/kvreuse/LMCache/.venv-vllm/bin/python -m \
  experiments.lossless_pd.lmcache_pd.direct_token_proxy \
  --host 127.0.0.1 --port 19100 \
  --prefiller-host 127.0.0.1 --prefiller-port <P_PORT> \
  --decoder-host 127.0.0.1 --decoder-port <D_PORT> \
  --decoder-init-port <D_INIT_PORT> --decoder-alloc-port <D_ALLOC_PORT>
```

`direct_token_proxy.py` changes only the upstream proxy's `/tokenize` handling
for integer-array prompts. Completion routing remains the upstream LMCache
implementation. This is necessary because token-to-text-to-token round trips
changed prompt boundaries and caused five false correctness mismatches in an
earlier run.

## Frozen 64-request result

The final run sends packet token IDs directly, retains each prompt's complete
length (56 are 8,192 tokens; the others span 2,560--6,656), uses one request at
a time, greedy decoding, and produces eight output tokens. The nominal 10%
whole-chunk Anchor has mean actual token fraction 12.557% (range 10--20%) due to
chunk rounding and first/last protection.

An observe--inject--observe sandwich uses fresh servers for all three runs.
Every treatment request is compared with the mean of its two surrounding
observe-only measurements:

| metric | inject minus sandwich observe | request-bootstrap 95% CI | wins |
|---|---:|---:|---:|
| TTFT | -0.635 ms | [-1.579, +0.353] ms | 37/64 |
| total latency | **-24.407 ms** | **[-29.671, -19.513] ms** | **63/64** |
| monolithic-adjusted total | **-24.187 ms** | **[-29.399, -19.336] ms** | -- |

All three raw streamed P/D outputs equal one another and the paired monolithic
output on 64/64 requests. TTFT is unchanged because proposals accelerate tokens
after the first authoritative token. Total latency falls because full-KV
verification accepts a mean 2.438-token prefix, including a mean 1.438 injected
suffix tokens per request. The raw first-token sparse proposal matches 54/64
requests, but target-conditioned repair injects a block on every request.

Hot sparse-KV draft time is 24.86 ms, repair time is 3.035 ms on GPU, and the
completed draft leads its D-side handoff by 236.25 ms on average. This proves
real concurrent Anchor consumption, proposal injection, full-KV verification,
exact greedy output, and a statistically positive post-first-token latency
effect. It does not yet prove multi-request throughput, cross-node behavior,
sampling correctness, task-unseen quality, or an optimized end-to-end P/D
service beating monolithic serving.

The fixed paper-facing report is
`docs/ICLR2027_ONLINE_LMCACHE_RESULT_20260909.md`. Raw traces and checkpoints
stay under ignored `outputs/`.

## Analysis tools

- `benchmark.py`: paired P/D versus monolithic replay; use
  `--packet-token-ids --max-input-tokens 8192` for strict packet identity.
- `direct_token_proxy.py`: integer-token-compatible wrapper around the upstream
  LMCache proxy.
- `analyze_trace.py`: P gather/write and AnchorReady--FullReady timeline.
- `analyze_receiver_trace.py`: D object-resolution and zero-copy-view audit.
- `analyze_online_draft.py`: live proposal, repair, acceptance, and lead-time
  statistics.
- `compare_online_modes.py`: paired or observe--inject--observe comparisons.
- `eval_runtime_layout.py`: offline evaluation under the exact runtime
  whole-chunk Anchor layout.

The 24-GiB receiver CUDA arena is required for the 64-request run because
transferred chunks remain resident long enough to exhaust a 2-GiB arena.
