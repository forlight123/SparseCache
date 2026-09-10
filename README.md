# SparseCache

> Cloud training handoff: read [TRAIN.md](TRAIN.md) before spending GPU time.
> The current model path is a direct Target-KV block drafter plus a causal
> hidden-space top-64 reranker; EAGLE expansion has been stopped. The direct
> drafter now runs online from real LMCache Anchor buffers while Residual KV is
> in flight. On 64 exact-token QMSum requests, full-KV verification preserves
> 64/64 monolithic outputs and reduces total latency by 24.41 ms versus an
> observe-only sandwich control (95% CI [19.51, 29.67] ms saved).

SparseCache is a reference framework for reusable RAG document KV. The current
benchmark evaluates CacheBlend on its MuSiQue subset under a strict
independent-document protocol: document KV is produced without the target
system prompt or question, while system/question tokens are always computed
online. The repository also retains an earlier portable `cpukv/` prototype.

The two online controls are:

- `direct`: concatenate the cached chunks and prefill only the question.
- `cacheblend`: recompute the full prompt through an early check layer, rank
  document tokens by new-versus-cached value deviation, and update only the top
  fraction in later layers.

KV remains native BF16. The current experiment establishes a corrected
CacheBlend baseline under independent-document reuse before adding sparse
decode-time paging.

## Repository layout

```text
SparseCache/
  TRAIN.md          cloud drafter-training handoff
  configs/          frozen experiment configurations
  docs/             method, baseline, and experiment protocols
  experiments/      data builders, trainers, evaluators, and live runners
  src/sparsecache/  producer, cache composer, and QA consumers
  tests/            unit and protocol gates
```

The local development machine additionally has physical `CacheBlend/`,
`LMCache/`, and `vllm/` working copies plus raw `datasets/`, `cpukv/`, and
`outputs/`. They are deliberately excluded from Git because they contain
third-party histories, generated KV, large datasets, profiler traces, or model
artifacts. Clone compatible third-party revisions and mount data separately
when reproducing the systems experiments below.

The current lossless P/D algorithm and the real LMCache gate are documented in
[`docs/ICLR2027_ALGORITHM.md`](docs/ICLR2027_ALGORITHM.md) and
[`docs/ICLR2027_ONLINE_LMCACHE_RESULT_20260909.md`](docs/ICLR2027_ONLINE_LMCACHE_RESULT_20260909.md).
The earlier receiver-only milestone remains in
[`docs/ICLR2027_LMCACHE_SYSTEM_GATE_20260909.md`](docs/ICLR2027_LMCACHE_SYSTEM_GATE_20260909.md).

The copied CacheBlend tree has its own uv environment and no longer depends
on the sibling repository's virtual environment:

```bash
cd /home/ytm/algorithm/kvreuse/SparseCache
uv venv --python 3.10 CacheBlend/.venv
uv pip install --python CacheBlend/.venv/bin/python \
  -r CacheBlend/requirements-local.txt
```

The local requirements keep the vLLM 0.4.1 CUDA-extension ABI on Python
3.10, Torch 2.2.1, xFormers 0.0.25, Transformers 4.40.2, and NumPy 1.26.4.
Experiment commands set `PYTHONPATH` to the copied `vllm_blend` source; the
created environment also contains a local `.pth` entry for interactive use.

## Strict independent-document MuSiQue experiment

The input is CacheBlend's checked-in `CacheBlend/inputs/musique_s.json`: 150
examples with ten documents each, for 1,500 document uses and 1,255 unique
token sequences.

The runner enforces these invariants:

- Every document is tokenized independently. Exact token sequences are
  deduplicated and each unique document is sent through the producer once.
- Producer input contains only that document: no benchmark question, system
  prompt, other documents, or target-specific population prompt.
- At target time, documents are concatenated in each sample's `ctxs` order.
- The system prefix and question suffix have zero placeholder KV in the
  offline cache. Their real tokens are evaluated online and forced into the
  active set at every layer after the CacheBlend check layer.
- CacheBlend ranks only document positions with its value-deviation rule.
  Online system/question positions do not consume the document budget. Two
  different `15%` conventions are reported below; they must not be mixed.

The reported full run uses three disjoint 50-example shards so the three H200s
can produce and evaluate in parallel:

| GPU | offset/count | persistent cache suffix | result suffix |
|---:|---:|---|---|
| 0 | 0/50 | `n150_shard000_050` | `shard000_050.json` |
| 1 | 50/50 | `n150_shard050_100` | `shard050_100.json` |
| 2 | 100/50 | `n150_shard100_150` | `shard100_150.json` |

For example, produce and evaluate the first shard as follows. Repeat with the
two other GPU/offset/cache/output tuples from the table:

```bash
cd /home/ytm/algorithm/kvreuse/SparseCache

CUDA_VISIBLE_DEVICES=0 PYTHONHASHSEED=0 \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
PYTHONPATH="$PWD/CacheBlend/vllm_blend" \
$PWD/CacheBlend/.venv/bin/python \
  CacheBlend/example/blend_musique_independent.py \
  --phase produce \
  --cache-dir cpukv/cacheblend_musique_llama31_8b_n150_shard000_050 \
  --dataset CacheBlend/inputs/musique_s.json \
  --model /data/models/llama/Llama-3.1-8B-Instruct \
  --offset 0 --count 50 --max-new-tokens 24 \
  --recompute-ratio 0.15 --blend-check-layer 1 \
  --gpu-memory-utilization 0.5 --max-model-len 20000

CUDA_VISIBLE_DEVICES=0 PYTHONHASHSEED=0 \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
PYTHONPATH="$PWD/CacheBlend/vllm_blend" \
$PWD/CacheBlend/.venv/bin/python \
  CacheBlend/example/blend_musique_independent.py \
  --phase evaluate \
  --cache-dir cpukv/cacheblend_musique_llama31_8b_n150_shard000_050 \
  --dataset CacheBlend/inputs/musique_s.json \
  --model /data/models/llama/Llama-3.1-8B-Instruct \
  --output outputs/vllm_cacheblend/musique_independent_fixed_n150_r015_shard000_050.json \
  --offset 0 --count 50 --max-new-tokens 24 \
  --recompute-ratio 0.15 --blend-check-layer 1 \
  --gpu-memory-utilization 0.5 --max-model-len 20000

# Repeat evaluate with --recompute-ratio 1.0 and an r100 output name for the
# correctness gate. Both consumers read the same persistent cache.

$PWD/CacheBlend/.venv/bin/python \
  CacheBlend/example/aggregate_musique_shards.py \
  --output outputs/vllm_cacheblend/musique_independent_fixed_n150_r015.json \
  outputs/vllm_cacheblend/musique_independent_fixed_n150_r015_shard000_050.json \
  outputs/vllm_cacheblend/musique_independent_fixed_n150_r015_shard050_100.json \
  outputs/vllm_cacheblend/musique_independent_fixed_n150_r015_shard100_150.json
```

The original copied implementation applied a bottom-right triangular mask to
the selected query rows. That mask is only causal for a contiguous suffix;
CacheBlend selects arbitrary original prompt positions, so some rows could
attend to future KV. The repaired path materializes causal rows from each
query's original position, carries those positions across layers, and uses the
native full-causal kernel for the 100% correctness gate. A unit test covers
the sparse-position mask directly.

Corrected released-code-style results on all 150 examples with greedy decoding:

| path | configured/final ratio | actual layer-average document compute | EM | F1 | mean TTFT | raw answer match vs full |
|---|---:|---:|---:|---:|---:|---:|
| full prefill | 100% | 100% | 0.187 | 0.348 | 177.0 ms | 150/150 |
| CacheBlend released-code style | 15% | 20.3% | 0.133 | 0.278 | 76.9 ms | 65/150 |
| CacheBlend correctness gate | 100% | 100% | 0.187 | 0.348 | 184.2 ms | 150/150 |

The released-code-style 15% run averages 5,684 document tokens and about 87 online
system/question tokens per prompt. Later layers recompute about 852 document
tokens and keep about 939 active tokens in total, giving a 2.30x mean model
TTFT reduction versus full prefill. Its F1 is 0.070 lower than full (0.278
versus 0.348); per-example F1 has 16 wins, 104 ties, and 30 losses. Raw answers
match on 65 cases, or 74 after answer normalization. The earlier ten-example
slice happened to overstate 15% quality and is not representative. The 100%
gate produces exactly the same raw answer as full prefill on all 150 cases,
validating prompt partitioning, causal attention, and the Llama-3.1 RoPE
adaptation across the complete checked-in dataset.

### What `15% recompute` means

The public CacheBlend command uses one check at layer 1 and
`--recompute-ratio 0.15`. Layers 0 and 1 have already processed every document
token before the filter takes effect, and only layers 2--31 process 15%.
Therefore, on a 32-layer model its actual layer-average document QKV work is
`(2 * 100% + 30 * 15%) / 32 = 20.3125%`. The configured 15% is a retained
token ratio, not a strict end-to-end compute budget.

The runner also supports audited gradual filtering. This schedule has a true
layer-average document budget of 15%:

```bash
--blend-check-layers 0,1,2 \
--recompute-ratios 0.30,0.20,0.11379310344827587 \
--target-average-recompute-ratio 0.15
```

Its layer trace is 100% on layer 0, 30% on layer 1, 20% on layer 2, and
11.379% on layers 3--31. Each ratio is measured against the original document
token count; online system/question positions are always retained and are
reported separately. `--deviation-metric` accepts squared-L2 `value` (the
released-code default), `key`, or `kv`, plus the corresponding `*_l1` modes.
Every result row records the actual per-layer document-token trace and its
mean, and `--target-average-recompute-ratio` fails the run if the measured
budget misses the target by more than 0.002.

### Paper-alignment experiments

The poor Llama-3.1 result above is not directly comparable with the paper:
it uses ten documents averaging 5,684 document tokens, while the paper uses
Mistral-7B/Yi-34B/Llama-70B and reports its main result on six retrieved
chunks. We therefore also evaluated the locally available
Mistral-7B-Instruct-v0.2 on the same 150 public MuSiQue examples, using the
first six complete checked-in chunks and appending `Answer within 5 words.`.
The checked-in JSON does not contain the paper's SentenceTransformer scores or
source vector database, so this is a controlled paper-like comparison, not an
exact reconstruction of its top-6 retrieval. The six complete chunks average
3,891 Mistral tokens per request.

| Mistral-7B path | configured/final ratio | actual layer-average document compute | EM | F1 | F1 delta vs paired FullKV | mean TTFT | speedup |
|---|---:|---:|---:|---:|---:|---:|---:|
| paired FullKV | 100% | 100% | 0.067 | 0.240 | -- | 115.2 ms | 1.00x |
| CacheBlend released-code style, Value-L2 | 15% | 20.30% | 0.060 | 0.233 | -0.0064 | 57.2 ms | 2.02x |
| CacheBlend gradual, strict average, Value-L2 | 11.38% final | 14.99% | 0.007 | 0.174 | -0.0659 | 46.9 ms | 2.45x |
| CacheBlend gradual, strict average, Key-L2 | 11.38% final | 14.99% | 0.007 | 0.171 | -0.0691 | 47.4 ms | 2.43x |

The released-code-style result has 24 per-example F1 wins, 101 ties, and 25
losses against paired FullKV; 43 raw answers match, or 59 after normalization.
Its absolute F1 loss of 0.0064 is within the paper's reported 0.02 bound. This
shows that the repaired implementation can reproduce paper-like quality when
`15%` uses the public implementation's retained-ratio convention. Enforcing a
literal 15% average compute budget is a materially harder operating point and
must be reported as a separate baseline. On this run, switching from the
public code's Value-L2 score to the current LMCache Key-L2 score did not recover
the strict-budget quality.

The primary Mistral artifacts are
`musique_mistral7b_first6_fullchunks_released_r015_value_n150.json`,
`musique_mistral7b_first6_fullchunks_gradavg015_value_n150.json`, and
`musique_mistral7b_first6_fullchunks_gradavg015_key_n150.json` under
`outputs/vllm_cacheblend/`. Each contains all 150 rows, the paired FullKV
outputs, protocol fields, actual compute traces, and aggregate diagnostics.
An additional Mistral 100% smoke gate matches the paired FullKV raw answer on
all 3/3 checked cases; the full Llama gate above remains the exhaustive
150/150 correctness check.

The three persistent shard caches contain 466, 422, and 424 safetensors files
and occupy about 33, 30, and 30 GiB. Both consumers record
`producer_calls_this_run = 0`. The expanded run was launched concurrently
against newly written pageable files, so its assembly/H2D measurement includes
cold page faults and disk contention: mean 3.40 s, median 2.36 s, and p95
8.29 s. These numbers are not comparable to the earlier warm ten-example
measurement and are not a serving-latency claim. They do show that the current
full-cache assembly remains unsuitable as the final SparseCache data path;
sparse/asynchronous transfer is still required. Model TTFT is measured
separately above.

The corrected primary artifacts are
`musique_independent_fixed_n150_r015.json` and
`musique_independent_fixed_native_n150_r100.json`; they contain all 150 rows,
aggregate diagnostics, and the validated source-shard paths. The older
`musique_original_independent_n10_r015.json` and
`musique_original_independent_n10_r100.json` were generated before the causal
mask repair and must not be used as baseline results.

The local LMCache integration is not used for this strict run. Its current
scheduler lookup walks a contiguous prefix and stops at the first miss; an
online system prompt therefore prevents it from discovering cached document
segments later in the request. Pre-populating that system prefix would violate
the protocol. The original CacheBlend model path can inject arbitrary
independent document segments, so it is the correctness baseline until
LMCache gains non-prefix segment lookup and scheduler-visible block assembly.

## Real asynchronous SSD CacheBlend deployment

The independent-document CacheBlend path now has a physical, layer-ready
storage pipeline. This is the copied `CacheBlend/` implementation in this
repository and runs in `CacheBlend/.venv`; it does not import the parent
repository's CacheBlend or virtual environment. The persistent document KV is
stored under `cpukv/` on `/dev/nvme0n1p1`, a 7 TB SOLIDIGM NVMe SSD mounted as
ext4 at `/home/ytm`.

For every cold request, the implementation does the following:

1. It advises the kernel to evict the ten document safetensors from the page
   cache with `POSIX_FADV_DONTNEED`. This control is measured but performed
   before the request timer starts.
2. A background worker reads each requested layer from the real safetensors,
   composes benchmark-order document slices into one of two pinned CPU buffers,
   and enqueues nonblocking H2D on a dedicated CUDA stream.
3. The CacheBlend model starts immediately in `disk_async` mode. At each layer
   it waits on that layer's recorded CUDA event, rather than waiting for the
   complete cache. `disk_sync` uses exactly the same physical path but joins
   the worker before model execution, providing the control.

The implementation is in
`CacheBlend/example/layerwise_disk_kv.py`; the request driver is
`CacheBlend/example/blend_musique_independent.py`, and the layer-ready wait is
integrated into the copied vLLM Llama model. Every result row records physical
file bytes, logical KV bytes, SSD-to-pinned composition time, CUDA H2D time,
first-layer readiness, and end-to-end time from cache-transfer start.

The primary experiment is a 150-request, single-H200, paired cold-cache run.
Each mode processes the same three 50-request cache shards sequentially, so it
does not compare a contended SSD run with an uncontended one. Requests average
5,684 document tokens, 745.0 MB of safetensors input, and 732.7 MB of logical
BF16 KV transferred for cached layers 1--31.

| physical path, 150 requests | mean cache-start TTFT | p50 | p95 | mean completion | SSD/compose to pinned | H2D |
|---|---:|---:|---:|---:|---:|---:|
| `disk_sync`, cold | 1264.8 ms | 1244.3 ms | 1457.4 ms | 1361.2 ms | 1148.6 ms | 17.5 ms |
| `disk_async`, cold | 1249.8 ms | 1219.5 ms | 1452.4 ms | 1371.0 ms | 1199.7 ms | 18.7 ms |

The paired first-token saving is 15.0 ms on average and 32.0 ms at the median.
Async wins 88/150 requests. The mean per-request speedup ratio is 1.017x, while
the reduction computed from the two aggregate means is 1.19%. The paired mean
saving has a bootstrap 95% confidence interval of `[-5.0, 34.5]` ms, so this
deployment does **not** establish a statistically stable latency improvement.
Generation completion is 9.8 ms slower on average. All 150/150 raw answers and
all 150/150 per-layer recompute traces match between modes; asynchronous
transport introduces no quality change.

This small net result has a concrete explanation. The synchronous CacheBlend
model compute to first token is only 76.1 ms, which is the approximate maximum
work available to hide. SSD read plus safetensors extraction and CPU pinned
composition takes about 1.15 seconds. During overlap, that worker becomes about
60 ms slower, consuming most of the hidden compute. The GPU copy itself is not
the bottleneck. Moreover, ordinary full prefill takes about 177 ms on this
5.7K-token workload, so transferring the complete native BF16 document KV from
SSD is much slower end to end than recomputation even though CacheBlend's
compute-only TTFT is 2.3x faster.

A second 150-request control launched three synchronous consumers on three
H200s against the same NVMe. Mean cache-start TTFT rose to 6301.6 ms and the
per-consumer effective file rate fell from 629 MB/s to 120 MB/s. Its aggregate
rate is only about 359 MB/s, showing severe document-major random-read and CPU
composition contention. It is retained as a serving-contention diagnostic,
not mixed into the paired speedup above.

One shard can be reproduced by changing only `--transfer-mode` between
`disk_sync` and `disk_async`:

```bash
CUDA_VISIBLE_DEVICES=2 PYTHONHASHSEED=0 \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
PYTHONPATH="$PWD/CacheBlend/vllm_blend" \
$PWD/CacheBlend/.venv/bin/python \
  CacheBlend/example/blend_musique_independent.py \
  --phase evaluate --transfer-mode disk_async --cache-state cold \
  --cache-dir cpukv/cacheblend_musique_llama31_8b_n150_shard000_050 \
  --dataset CacheBlend/inputs/musique_s.json \
  --model /data/models/llama/Llama-3.1-8B-Instruct \
  --output outputs/vllm_cacheblend/musique_disk_async_cold_single_gpu_shard000_050.json \
  --offset 0 --count 50 --max-new-tokens 24 \
  --recompute-ratio 0.15 --blend-check-layer 1 \
  --gpu-memory-utilization 0.5 --max-model-len 20000

$PWD/CacheBlend/.venv/bin/python \
  CacheBlend/example/aggregate_disk_transfer.py \
  --sync outputs/vllm_cacheblend/musique_disk_sync_cold_single_gpu_shard*.json \
  --async outputs/vllm_cacheblend/musique_disk_async_cold_single_gpu_shard*.json \
  --output outputs/vllm_cacheblend/musique_disk_cold_single_gpu_n150_paired_summary.json
```

The result changes the next SparseCache systems priority: merely streaming the
full CacheBlend KV layer by layer is correct but not enough. The next data path
must reduce bytes before transfer (page/token sparsity) and should replace the
document-major per-layer safetensors reads with a contiguous layer/page layout.
Only after reducing the roughly 745 MB request payload is it useful to optimize
H2D further or connect this transport to the progressive draft/verify policy.

## Real P/D draft--progressive KV--verify pipeline

The P/D path has now also been executed with no SSD and no CacheBlend. P places
one correct contiguous full-prompt KV in pinned CPU memory; D receives real
priority tranches through a dedicated CUDA copy stream, drafts four tokens,
waits on per-stage CUDA events, refreshes generated-token representations, and
verifies under each newly arrived KV view. A same-token serial-stage control
isolates the transport overlap, while full KV H2D plus normal decode is the
end-to-end baseline.

On 150 natural MuSiQue requests (2,533 mean prompt tokens), query `W=1`
reduces the same-chain serial latency from 179.6 to 173.9 ms. The 5.8 ms paired
saving has 95% CI `[1.0, 12.2]` ms, so the physical overlap works. However,
full KV H2D plus normal decode takes only 113.0 ms: repeated draft/refresh and
verify make the progressive path 60.8 ms slower. Query `W=2` takes 206.3 ms
and transfers 97.7% of the KV.

Real 16K, 32K, and 64K timing points do not cross over. At 64K, the pipeline
hides 61.1 ms relative to its serialized form but still takes 429.1 ms versus
270.0 ms for the full-transfer baseline. The next gate is therefore bounded
representation refresh, information-driven verification, and smaller
cancelable transfer pages—not another fixed five-stage run. Full protocol,
quality results, scale data, limitations, and reproduction commands are in
[`outputs/progressive_kv/PD_REAL_PIPELINE_REPORT.md`](outputs/progressive_kv/PD_REAL_PIPELINE_REPORT.md).

The follow-up implements a physically compact fixed-S1 draft view, four
`10/40/70/100%` stages, final-verifier cache reuse, and paced P/D wire arrival
while retaining real pinned-memory H2D. On the matched 30-case 64K set, the new
pipeline is still 140.5 ms slower than full transfer on native H2D, but saves
216.5 ms at 100 Gbps and 406.6 ms at 50 Gbps. The 100 Gbps paired CI is
`[148.1, 287.4]` ms and 23/30 requests improve. Fixed-S1 drafting alone reduces
draft compute by 27.2% and response by 28.6 ms versus cumulative-KV drafting.
The timing direction is now feasible. On 150 natural MuSiQue requests, `W=1`
has 86.7% target-token match and F1 delta -0.0157 with 95% CI
`[-0.0369, 0.0020]`; at this short 2.5K context it remains 56.1 ms slower than
full transfer. Adaptive commitment quality is the next gate. See
[`outputs/progressive_kv/PD_SPARSE_DRAFT_BANDWIDTH_REPORT.md`](outputs/progressive_kv/PD_SPARSE_DRAFT_BANDWIDTH_REPORT.md).

### Exact final-verifier direction

The current mainline removes finite-window commitment from the performance
claim.  In `final_only` mode, D transfers an exact BF16 priority-page seed,
drafts against its physically compact KV, transfers the exact residual in the
background, and runs one immutable full-KV verifier before exposing any token.
Intermediate arrivals do not trigger verification.  This is the `W=inf`
endpoint; the earlier `W=1` numbers remain a separately labeled lossy
ablation.  The intended progressive variant changes only the draft page table:
new exact pages become visible as their transfer events complete, with no
replay, stage barrier, or intermediate verifier.  Fixed S1 is its robust
two-level control and fallback.

The mechanism runner implements this policy in `continuous` draft-cache mode:
it polls async H2D completion before every proposal and records the visible
arrival stage per token.  The Hugging Face legacy cache still physically
concatenates the enlarged prompt view and approximate generated tail, so this
is an acceptance/mechanism implementation; the final systems path requires a
vLLM metadata-only page table.

A preliminary paired LongBench QMSum run (`n=10`, 13.0K mean prompt, 64-token
output limit) establishes the latency mechanism.  At 25 Gbps, fixed 16-token
drafting saves 110.4 ms (1.075x, paired 95% CI `[58.4, 164.7]`).  At 100 Gbps
that draft is too long, but eight drafts save 40.3 ms (1.036x, CI
`[11.8, 66.5]`).  A predeclared residual-window scheduler with a 16-token cap
saves 97.0 and 31.5 ms at 25 and 100 Gbps respectively, with no estimated
draft compute exposed beyond the transfer window.  These are preliminary
break-even results, not benchmark-quality paper claims.  A follow-up `n=30`
adaptive run at 100 Gbps saves 58.9 ms (1.053x, CI `[34.5, 92.4]`, 27/30
faster) with ROUGE-L delta +0.00046 and CI `[-0.00334, 0.00405]`.  Raw greedy
token equality is only 17/30: the divergence usually occurs well after the
verified block because batched and token-at-a-time K/V differ numerically, and
verifier margin does not predict it.  The method therefore has a mathematical
speculative/distributional guarantee, not current bitwise greedy equality.
Deterministic kernels or fully charged sequential accepted-prefix
rematerialization and >=100-request natural-task runs are the next gates.
The same scheduler structure also gives a positive Qwen3-8B QMSum signal with
an independently profiled 22.5 ms sparse TPOT: `n=10` at 100 Gbps saves
79.7 ms (1.050x, CI `[57.3, 102.9]`, 10/10 faster), with no detected ROUGE-L
change.  A larger sample is still required.

A 100-request audit found that natural-EOS completion timing can compare
different output lengths.  The corrected fixed-64-token run then suffered
external GPU scheduling interference: its mean saving is 8.0 ms with CI
`[-34.6, 39.8]`, and 15/100 requests contain more than 100 ms of unattributed
wall time.  It is retained as a contaminated audit artifact, not a speedup
claim.  The runner now supports `--paired-measurement`, which warms both paths,
alternates their order, omits the interposed serial run, and pairs with
`--fixed-token-horizon`.  The primary QMSum cell must be rerun on one
exclusively owned GPU before the latency gate is considered passed.

See the [method specification](docs/METHOD_PROGRESSIVE_SPARSE_PD.md), the
[ICLR experiment matrix](docs/ICLR_EXPERIMENT_MATRIX.md), and the
[machine-readable experiment matrix](configs/iclr_pd_experiment_matrix.json).
The pre-registered executable selection queue is
[`outputs/progressive_kv/iclr_job_queue_20260826.json`](outputs/progressive_kv/iclr_job_queue_20260826.json),
generated by `experiments/build_iclr_pd_job_queue.py`; confirmation jobs stay
as blocked templates until the selection policy is frozen.

## Earlier portable KV prototype

The following `datasets/`, `cpukv/`, and `src/sparsecache/` workflow predates
the strict MuSiQue runner above. In particular, it stores a reusable system
prefix and is not the protocol used for the reported MuSiQue numbers.

### Data preparation

The checked-in normalized case uses row zero of the HotpotQA-E file. Its ten
chunks are full Wikipedia-like passages, not fixed-size KV pages.

```bash
PYTHONPATH=src /home/ytm/algorithm/kvreuse/CrossKV/.venv/bin/python \
  -m sparsecache.data \
  --dataset hotpotqa \
  --input datasets/hotpotqa/hotpotqa_e_disjoint_from_standard.jsonl \
  --output datasets/cases/hotpotqa_e_case0_10chunks.json \
  --index 0 \
  --max-chunks 10
```

The same adapter accepts an official MuSiQue JSONL file with
`--dataset musique`.

### Offline producer

The producer evaluates `S + Di` independently for every document and stores
only the `Di` cache. The system cache is stored once. Each safetensors file
contains `layer_NNN.key` and `layer_NNN.value` tensors in
`[1, kv_heads, tokens, head_dim]` layout.

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src \
  /home/ytm/algorithm/kvreuse/CrossKV/.venv/bin/python \
  -m sparsecache.precompute \
  --case datasets/cases/hotpotqa_e_case0_10chunks.json \
  --model /data/models/llama/Llama-3.1-8B-Instruct \
  --output-dir cpukv/llama31_8b_hotpotqa_e_case0 \
  --device cuda:0 \
  --dtype bfloat16
```

The manifest records the exact model fingerprint, token IDs, source RoPE
positions, tensor geometry, and logical KV byte count. A consumer refuses a
cache produced by a different model configuration or weight index.

### Online QA

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src \
  /home/ytm/algorithm/kvreuse/CrossKV/.venv/bin/python \
  -m sparsecache.qa \
  --manifest cpukv/llama31_8b_hotpotqa_e_case0/manifest.json \
  --modes baseline direct cacheblend \
  --check-layer 1 \
  --recompute-ratio 0.15 \
  --max-new-tokens 16 \
  --device cuda:0 \
  --output outputs/llama31_8b_hotpotqa_e_case0.json
```

`baseline` is an ordinary full prefill and is included only as a control.
`direct` loads the complete reusable document KV from CPU and performs zero
document-token prefill. `cacheblend` follows the layerwise CacheBlend
mechanism; it is not an oracle replacement with a separately computed full
cache.

The JSON reports CPU file loading plus RoPE composition as `compose_ms`, full
prefix host-to-device time as `prefix_h2d_ms`, and model execution separately.
These Python timings are correctness instrumentation, not optimized serving
latency.

## Earlier vLLM + LMCache integration smoke

This older systems smoke test uses the first HotpotQA-E example, all ten passages,
Llama-3.1-8B-Instruct, greedy decoding, and GPU 1. The population request
stores the passages in reverse order; the target request restores their
original order and changes the question. This forces segment reuse rather than
ordinary prefix reuse. `# #` is the LMCache segment boundary marker.

The compatible pre-existing Python environment supplies compiled CUDA
dependencies, but both imported Python packages are forced to the physical
copies in this directory through `PYTHONPATH`:

```bash
cd /home/ytm/algorithm/kvreuse/SparseCache

CUDA_VISIBLE_DEVICES=1 \
VLLM_ENABLE_V1_MULTIPROCESSING=0 \
VLLM_USE_V2_MODEL_RUNNER=0 \
PYTHONHASHSEED=0 \
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
PYTHONPATH="$PWD/vllm:$PWD/LMCache" \
/home/ytm/algorithm/kvreuse/LMCache/.venv-vllm/bin/python \
  LMCache/examples/blend_kv_v1/hotpot_same_task.py \
  --phase cold \
  --dataset datasets/hotpotqa/hotpotqa_e_disjoint_from_standard.jsonl \
  --model /data/models/llama/Llama-3.1-8B-Instruct \
  --output outputs/vllm_cacheblend/llama31_8b_hotpotqa_e_case0_cold.jsonl \
  --offset 0 --count 1 --max-new-tokens 24

CUDA_VISIBLE_DEVICES=1 \
VLLM_ENABLE_V1_MULTIPROCESSING=0 \
VLLM_USE_V2_MODEL_RUNNER=0 \
PYTHONHASHSEED=0 \
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
PYTHONPATH="$PWD/vllm:$PWD/LMCache" \
/home/ytm/algorithm/kvreuse/LMCache/.venv-vllm/bin/python \
  LMCache/examples/blend_kv_v1/hotpot_same_task.py \
  --phase blend \
  --dataset datasets/hotpotqa/hotpotqa_e_disjoint_from_standard.jsonl \
  --model /data/models/llama/Llama-3.1-8B-Instruct \
  --output outputs/vllm_cacheblend/llama31_8b_hotpotqa_e_case0_blend_r015.jsonl \
  --offset 0 --count 1 --max-new-tokens 24 \
  --recompute-ratio 0.15 --blend-check-layer 1
```

The 100% correctness gate uses the same blend command with output suffix
`blend_r100.jsonl`, `--recompute-ratio 1.0`, and `--blend-check-layer 0`.

Observed smoke-test results:

| mode | LMCache hit | TTFT | wall time | answer | F1 |
|---|---:|---:|---:|---|---:|
| cold full prefill | n/a | 247.3 ms | 298.3 ms | `Yanzhou District` | 0.667 |
| CacheBlend, 15% recompute | 6114/6157 (99.30%) | 166.0 ms | 341.5 ms | repetitive incorrect text | 0.000 |
| CacheBlend, 100% recompute | 6114/6157 (99.30%) | 283.9 ms | 335.3 ms | `Yanzhou District` | 0.667 |

This is a one-example integration smoke test, not a quality benchmark and not
the strict protocol above: its reverse-order population request is
target-specific. It
establishes that cache population, segment hits, reordered retrieval, vLLM
decode, and full-recompute correctness all work. At 15%, TTFT is 32.9% below
the cold run but the answer is unusable, so it is not yet a valid
speed/quality win. The failure is localized to selective recomputation (and
its current memory-lifecycle warnings), rather than to a total cache miss.

The copied vLLM needed one compatibility hook in
`vllm/vllm/v1/worker/gpu_worker.py`: after loading the model, it registers the
module with LMCache's `VLLMModelTracker`. Without that hook this particular
vLLM/LMCache revision pair aborts before the first blended request.

Raw outputs and summaries are under `outputs/vllm_cacheblend/`.

## Current systems boundary

The Transformers runner now implements BM25-ranked exact page seeds,
physically compact sparse drafting, paced asynchronous arrival, real
pinned-memory H2D, residual-window draft sizing, zero-model-replay grafting,
tentative intermediate verification, and final-verifier state reuse.  The
last two mechanisms are implemented as ablations: mechanical multi-stage
grafting has weak task-dependent acceptance gains, and verifying every stage
is compute-negative.  Producer last-query and question-token attention page
rankings were also weaker than BM25 on held-out requests.

The current positive mainline is two-level: a tiny exact BM25 priority set
starts sparse self-drafting while the full BF16 residual transfers, followed
by one immutable full-KV verifier.  The runtime remains single-host: network
arrival is paced rather than sent by a remote P worker, Hugging Face legacy
cache grafting physically concatenates tensors, and SDPA is not a production
paged sparse kernel.  Real two-node transport, concurrent scheduling, bounded
HBM page pools, deterministic verifier/decode kernels, and vLLM integration
remain before the final serving claim.

## Earlier independent-reuse progressive experiment (not P/D)

A 150-request Llama-3.1-8B pre-experiment tested five-stage KV arrival with
query, oracle, random, and supporting-evidence-last schedules. The current
formulation is not ready for systems integration: independent FullReuse loses
0.154 mean F1 against a contiguous full-context cache, W=1 matches only
31%--39% of final-verifier token sequences, and W=2 matches 72%--79% while
waiting for over 92% of the KV. On the local H200, a 311 MiB pinned H2D copy
takes 5.76 ms, much less than the 214--417 ms draft/verify chains.

The useful positive result is that scheduling strongly controls lossy answer
quality: oracle W=1 reaches 0.332 F1, versus 0.248 for lexical query order and
0.107 when supporting evidence arrives last. The complete methodology,
confidence intervals, timing bounds, and next experimental gate are in
[`outputs/progressive_kv/PRELIMINARY_REPORT.md`](outputs/progressive_kv/PRELIMINARY_REPORT.md).

This result must not be used to judge the P/D proposal: it composes document
KV produced independently and therefore changes the representation before the
draft/verify chain starts.

## P/D progressive-transfer draft/verify pre-experiment

The corrected first pre-experiment uses a real P/D boundary. The P worker
prefills the complete prompt once and produces one contiguous full-context KV
cache. The D worker receives the first generated token plus the small
system/query anchor, then observes document-token KV in five cumulative
priority stages. No independently produced or composed document cache is used.

On 150 stratified official MuSiQue requests, the query schedule reaches 91.2%
teacher-forced top-1 agreement with the complete verifier after 26.4% of total
KV has arrived; the oracle supporting-first schedule reaches 95.4%. The
supporting-evidence-last control reaches only 79.2%, showing that transmission
order is a first-class correctness variable.

| order | W | exact output vs complete P/D target | normalized output match | F1 delta vs target | total KV required |
|---|---:|---:|---:|---:|---:|
| query score | 1 | 81.3% | 82.7% | -0.0268 | 67.0% |
| query score | 2 | 91.3% | 92.0% | -0.0209 | 86.5% |
| oracle supporting-first | 1 | 87.3% | 88.0% | -0.0025 | 64.7% |
| oracle supporting-first | 2 | 92.0% | 92.7% | -0.0051 | 84.6% |
| supporting-last | 1 | 42.0% | 47.3% | -0.0833 | 69.3% |
| supporting-last | 2 | 58.7% | 62.7% | -0.0520 | 88.7% |

`W=inf` never exposes a provisional token. After complete KV arrival it uses
the fixed full-cache target and performs an exact replay when BF16 batched
verification disagrees with token-at-a-time greedy decoding. This gate is
100% target-equivalent on all 150 requests; replay is needed on 2.7%--3.3% of
schedule runs and its measured cost is included.

The semantic mechanism therefore passes the first P/D feasibility gate, but
W=1 is not a safe default and W=2 still falls short of 95% exact-sequence
agreement. At the current 2.5K-token/316.7-MiB scale, the unoptimized HF chain
is compute-dominated: optimistic overlap gives query W=2 a 1.33x speedup only
at 1 GiB/s, and 0.86x at 3 GiB/s. The next gate is an adaptive commitment rule
with a stronger page scorer on 16K--128K contexts or remote/cold KV transport,
followed by a real asynchronous P/D runtime only if that gate passes.

Full protocol, confidence intervals, timing assumptions, limitations, and
reproduction commands are in
[`outputs/progressive_kv/PD_PRELIMINARY_REPORT.md`](outputs/progressive_kv/PD_PRELIMINARY_REPORT.md).

## True reuse progressive draft/verify experiment

The follow-up now uses actual query-agnostic independent-document KV stored in
pinned CPU memory.  It computes system/query tokens online, performs real
asynchronous H2D, relocates document keys with RoPE, repairs each arrived view
with CacheBlend 15%, and then runs draft/verify.  There is no SSD, simulated
network, or P/D full-cache handoff in this path.

On 150 natural-length MuSiQue requests, one-shot CacheBlend reduces TTFT from
82.00 ms to 56.09 ms (paired saving 25.91 ms, 95% CI
[24.21, 27.66]).  The four-stage progressive chain takes 348.43 ms versus
157.07 ms for CacheBlend because repeated repair, verify, and full-model draft
far exceed the approximately 22 ms H2D window.

The best ablation uses two stages and one draft token.  It is CacheBlend-token
equivalent on 30/30 synchronized natural requests, but remains 33.65 ms slower.
At 64K timing scale (`n=5`), it overlaps 75.82 ms and beats no-reuse full
prefill by 1.62 seconds, yet remains 22.84 ms slower than one-shot CacheBlend.
Thus KV reuse is beneficial; the current progressive increment is not.

The implementation is in
`experiments/reuse_cacheblend_pipeline.py`, aggregation is in
`experiments/aggregate_reuse_cacheblend_pipeline.py`, and the complete report
is [`outputs/reuse_pipeline/TRUE_REUSE_PROGRESSIVE_REPORT.md`](outputs/reuse_pipeline/TRUE_REUSE_PROGRESSIVE_REPORT.md).

## Lossless verifier status

The real LMCache oracle ceiling now reaches 64/64 same-stack token-ID equality
and 1.1065x total speedup for an eight-token block after request-scoped prefix
caching and rare canonical replay. A stronger 32-token continuation test is
only 58/64 exact: fully accepted BF16 block representations can later diverge
from ordinary `q=1` decoding. Backend, horizon, and logit-margin scans do not
provide a lossless fix. The next gated mainline is a shape-invariant block
verifier; see
[`docs/ICLR2027_SHAPE_INVARIANT_VERIFIER_20260910.md`](docs/ICLR2027_SHAPE_INVARIANT_VERIFIER_20260910.md).
