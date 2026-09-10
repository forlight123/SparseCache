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

### External cheap-model proposal bridge

The runtime also supports a lossless external Qwen3-4B proposal sidecar.  This
is the post-stop-loss main candidate after both learned sparse-Target-KV
adapter revisions failed their frozen task-unseen acceptance gate.  It does
not claim that the 4B model itself consumes sparse Target KV.

The proxy starts a `g+1` seed branch on the sidecar when the request arrives,
in parallel with P.  When P returns its exact seed, a matching branch reuses
the remaining `g` tokens.  A mismatch triggers a prefix-cached request
conditioned on the exact P seed.  The proposal is sent to D over a fail-closed
UDP join; D acknowledges only after publishing it to the existing custom
proposer registry.  Every returned suffix is still verified by the ordinary
complete-KV Target before commitment.

Start the sidecar with prefix caching enabled, for example:

```bash
CUDA_VISIBLE_DEVICES=2 \
/home/ytm/algorithm/kvreuse/LMCache/.venv-vllm/bin/vllm serve \
  /data/models/qwen/Qwen3-4B --host 127.0.0.1 --port 18320 \
  --dtype bfloat16 --max-model-len 8304 --gpu-memory-utilization 0.30 \
  --enable-prefix-caching --enforce-eager --disable-log-stats
```

Add these variables to D:

```text
SPARSECACHE_DRAFTER_KIND=external
SPARSECACHE_DRAFT_TOKENS=8
SPARSECACHE_EXTERNAL_DRAFT_LISTEN=udp://127.0.0.1:17620
SPARSECACHE_DRAFT_NOTIFY=udp://127.0.0.1:17610
SPARSECACHE_ONLINE_DRAFT_MODE=observe|inject
```

Add these variables to `layer_ready_proxy.py`:

```text
SPARSECACHE_EXTERNAL_DRAFT_URL=http://127.0.0.1:18320
SPARSECACHE_EXTERNAL_DRAFT_MODEL=/data/models/qwen/Qwen3-4B
SPARSECACHE_EXTERNAL_DRAFT_TOKENS=8
SPARSECACHE_EXTERNAL_DRAFT_NOTIFY=udp://127.0.0.1:17620
SPARSECACHE_DRAFT_LISTEN=udp://127.0.0.1:17610
```

The first live gate uses a third GPU so drafter compute cannot perturb D.  A
paper result must additionally report resource-normalized throughput and a
co-located D-side configuration; an uncharged extra GPU is not a valid win.

That gate has now completed on a fresh-server 64-request
observe--inject--observe sandwich.  Greedy live output IDs match on 64/64,
accepted injected suffix is 3.953 tokens/request, and injection saves 30.690 ms
(95% CI 21.814--39.479 ms), but total speedup is only 1.0379x and TTFT regresses
by 3.192 ms.  It therefore fails the pre-registered 1.10x stop-loss threshold;
the dedicated-4B portfolio is retained as a positive mechanism baseline, not
the ICLR main line.  See `docs/ICLR2027_DRAFTER_STOPLOSS_PIVOT_20260910.md`.

`benchmark_pd_packets.py` replays packet token IDs without occupying the third
GPU with a monolithic control.  `analyze_external_draft_gate.py` joins its three
arms with the proxy/draft traces and evaluates the frozen gate.  Same-stack
observe output is the executable losslessness reference because vLLM and the
Transformers packet builder diverge on a small, identical set of low-margin
requests.

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
SPARSECACHE_ANCHOR_MODE=protected_uniform|nested_protected_uniform  # P
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

### Direct Target self-draft gate

The learned five-layer block drafter above is retained as a baseline, but its
task-unseen accepted length is too low for the main SparseCache route.  The
runtime also supports an untrained, exact-weight Target self-drafter that reads
all Target layers from the compact Anchor KV and is verified by the ordinary
full-KV vLLM Target:

```text
SPARSECACHE_DRAFTER_KIND=target
SPARSECACHE_TARGET_DRAFTER_MODEL=/data/models/qwen/Qwen3-8B
SPARSECACHE_DRAFT_LAYERS=0,1,2,...,35
SPARSECACHE_DRAFT_TOKENS=9  # one t1 alignment + at most eight injected tokens
SPARSECACHE_TARGET_ROOT_TOPK=1  # >1 is an experimental negative ablation
SPARSECACHE_TARGET_DRAFTER_WARMUP_TOKENS=1024
SPARSECACHE_ONLINE_DRAFT_MODE=observe|inject
```

Do not set `SPARSECACHE_DRAFTER_CHECKPOINT` for this mode.  The current
prototype loads a second Hugging Face copy of the Target on D after vLLM has
initialized, so the D launch must reserve roughly another model-weight-sized
GPU allocation.  It is an integration/correctness gate, not the final memory
architecture; production code should share vLLM weights and use a paged sparse
attention kernel.  At the default root top-k of one, a first-proposal mismatch
fails closed (no suffix is injected); larger values precompute several
first-token-conditioned branches but did not improve strict all-request
progress in the initial online screen.  Every injected suffix is still checked
by vLLM against complete KV before commitment.  For the example above, launch
the D custom proposer with `num_speculative_tokens=8`; paper metrics must report
only accepted injected suffix tokens and must not count the `t1` alignment
position.

`nested_protected_uniform` is the monotone systems-control schedule.  It
preserves the validated 10% chunk set and only appends chunks at larger
fractions.  It fixes set replacement but is not a substitute for a
query-dependent relevance order.

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
after the first authoritative token. The legacy progress counter is 2.438, but
it includes one already-authoritative Target token; the standard accepted
speculative suffix is only 1.438 tokens/request. This learned branch is an
online-mechanism proof, not the primary paper method. The raw first-token sparse
proposal matches 54/64 requests, but target-conditioned repair injects a block
on every request.

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

## Exact P-runahead baseline (experimental)

`runahead_token_proxy.py` is a bounded source-side continuation control. It
asks the exact P Target for more than the usual one token, requires vLLM to
return their integer IDs, appends those IDs to the D request, and reduces D's
remaining output budget. It never retokenizes generated text. Launch it with
the same arguments as `direct_token_proxy.py`:

```bash
SPARSECACHE_P_RUNAHEAD_TOKENS=4 \
PYTHONPATH="$PWD:/home/ytm/algorithm/kvreuse/LMCache" \
  /home/ytm/algorithm/kvreuse/LMCache/.venv-vllm/bin/python -m \
  experiments.lossless_pd.lmcache_pd.runahead_token_proxy \
  --host 127.0.0.1 --port 19100 \
  --prefiller-host 127.0.0.1 --prefiller-port <P_PORT> \
  --decoder-host 127.0.0.1 --decoder-port <D_PORT> \
  --decoder-init-port <D_INIT_PORT> --decoder-alloc-port <D_ALLOC_PORT>
```

Benchmark it with two discarded warm-up requests and exact output IDs:

```bash
PYTHONPATH="$PWD:/home/ytm/algorithm/kvreuse/LMCache" \
  /home/ytm/algorithm/kvreuse/LMCache/.venv-vllm/bin/python -m \
  experiments.lossless_pd.lmcache_pd.benchmark_runahead \
  --packets <PACKET_ROOT> --output <RESULT.json> \
  --num-requests 32 --warmup-requests 2 \
  --max-input-tokens 7680 --max-new-tokens 8 --runahead-tokens 4
```

The measured one-host result is a negative control. At 256-token-aligned prompt
boundaries, `k=4` returns identical integer output IDs on 32/32 requests but is
7.05 ms slower in total, CI [5.33, 8.65], and 52.53 ms slower to first token,
CI [51.25, 53.74], than the normal one-token handoff. D recomputes the short
P-generated suffix and that cost is charged. With a 7,800-token unaligned
boundary, D retrieves only 7,680 tokens and output IDs match 7/8, demonstrating
that exact Target weights do not imply bitwise equality across a recomputed BF16
state boundary. Use aligned boundaries for the bitwise baseline and retain the
unaligned result as an exactness diagnostic.
