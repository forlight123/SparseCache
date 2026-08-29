# SparseCache-PD ICLR experiment matrix

Status: execution plan, 2026-08-26.  No row is a paper result until its stated
sample, correctness, and statistical gates pass.

## 1. Research questions

1. Can exact page-progressive sparse self-speculation reduce stage-transition
   latency relative to full KV transfer?
2. Which context, bandwidth, batch, and model regimes satisfy the measured
   break-even conditions?
3. How much does page ordering affect sparse-draft acceptance at a fixed byte
   budget?
4. Does continuous, metadata-only expansion of the sparse draft mask improve
   long-window acceptance without replay or intermediate verification?
5. Can the system preserve full-target task quality and distribution while
   scaling to a real P/D runtime with concurrent requests?

## 2. Evaluation tracks

### Track A: controlled mechanism

- Context: 16K, 32K, 64K, 128K.
- Output: fixed 32 target tokens are materialized; the fixed-64 replication is
  still pending.
- Evidence placement: first, uniform, last, and adversarially split.
- Purpose: acceptance curves, page-order upper/lower bounds, verifier
  correctness, and latency-model validation without benchmark confounders.
- Dataset: RULER/needle and passage-retrieval variants; add a synthetic
  long-output continuation after the retrieval answer to expose TT32/TT64.

### Track B: natural long-context quality and latency

Use official prompts and metrics for each task.

The runner vendors the public dataset templates and output limits.  Primary
quality runs use `--longbench-prompt-mode official_chat
--official-output-length` with natural EOS.  System cells use the same public
template but a declared fixed 32/64-token horizon.  The earlier generic-chat
prompt is retained only to reproduce exploratory artifacts and must not be
mixed into the paper quality table.

| capability | datasets | primary metric | output role |
|---|---|---|---|
| multi-hop QA | LongBench 2WikiMQA, HotpotQA, MuSiQue | QA F1/EM | quality gate |
| document QA | Qasper, MultiFieldQA-en, NarrativeQA | task F1 | quality/scheduling |
| summarization | QMSum, GovReport, MultiNews | ROUGE-L | long-output latency |
| code | LCC, RepoBench-P | edit similarity | long-output latency |
| controlled retrieval | PassageRetrieval-en, PassageCount | accuracy | evidence order |
| hard long context | LongBench-v2 | accuracy by length/difficulty | 32K--128K quality |

At least 200 examples per quality dataset should be used when available.
System latency distributions require at least 100 requests per cell after
warmup; high-load throughput cells require >=1,000 completed requests or a
stable >=5 minute measurement window.

Local-data audit (the LongBench `length` field, before model tokenization):

| dataset | available | mean | p95 | official output cap |
|---|---:|---:|---:|---:|
| 2WikiMQA | 200 | 4.9K | 10.5K | 32 |
| HotpotQA | 200 | 9.1K | 12.2K | 32 |
| MuSiQue | 200 | 11.0K | 12.3K | 32 |
| Qasper | 200 | 3.6K | 6.4K | 128 |
| NarrativeQA | 200 | 18.4K | 31.7K | 128 |
| QMSum | 200 | 10.5K | 20.3K | 512 |
| GovReport | 200 | 8.2K | 15.6K | 512 |
| PassageRetrieval-en | 200 | 9.3K | 10.6K | 32 |
| LCC / RepoBench-P | 500 / 500 | 1.2K / 4.2K | 2.9K / 9.0K | 64 |

Natural LongBench alone therefore cannot substantiate 64K--128K scaling.
Track A and LongBench-v2 are mandatory rather than optional appendices.
Quality uses each dataset's official output cap; fixed 32/64-token runs answer
the systems-latency question and must be labeled separately.

Every fractional priority seed is subject to a fixed atomic boundary anchor:
the first 256 and last 512 prompt tokens arrive before drafting and remain
dense. Report the rounded observed S1 fraction, not merely its configured
percentage. This prevents a sparse draft from reading an untransferred system
or question block.

Track A is materialized locally as 20 exact-length RULER `qa_2` cells: 200
requests at 16,384, 32,768, 65,536, and 130,816 prompt tokens, crossed with
original, evidence-first, evidence-uniform, evidence-last, and
adversarial-split layouts. Every request contains exactly two HotpotQA
supporting documents. Length normalization changes only distractors: it repeats
them for short rows and trims them for 10/200 rows in the 130,816-token cell.
The five layouts at a given length have identical sample IDs, order, answers,
and output protocol. The fail-closed audit passed all 4,000 requests and
245,504,000 prompt tokens; its machine-readable result is
`outputs/progressive_kv/pd_packs/llama31_8b/controlled/ruler_qa2_audit.json`.

Track B request packs are also materialized. Thirteen LongBench datasets
contain 3,150 official-template requests and 31,560,704 aligned prompt tokens;
LongBench-v2 adds 503 requests and 40,719,872 prompt tokens up to a 120,064-token
model-safe cap. Hash, pairing, alignment, answer, and decoding-protocol audits
passed. The roots and audit files are under
`outputs/progressive_kv/pd_packs/llama31_8b/quality`, with aggregate audits at
`outputs/progressive_kv/pd_packs/llama31_8b/quality_audit.json` and
`outputs/progressive_kv/pd_packs/llama31_8b/longbench_v2_audit.json`.

The second-model Qwen3-8B packs are independently tokenized with its native
chat template and thinking disabled. They contain the same 3,150 LongBench
requests (32,232,192 prompt tokens), all 503 LongBench-v2 requests (15,155,712
prompt tokens), and ten 16K/32K RULER cells (2,000 requests and 49,152,000
prompt tokens). All three fail-closed audits passed. Qwen is capped at 32,768
prompt tokens here; it does not contribute to the 64K/128K scaling claim.

The audited live execution queue is
`outputs/progressive_kv/live_queue_llama31_8b_20260827.json`. It contains 16
ready core cells (four lengths at evidence-first plus the five-way 64K
scheduling slice, each at 25/50 Gbps), 40 blocked full controlled-confirmation
cells, and 14 natural-quality templates. The latter two phases remain blocked
until the core gate freezes the method; this prevents choosing bandwidth,
draft length, or schedule after seeing confirmation quality.
Core selection uses source rows 0--99. Every controlled-confirmation job is
hard-coded to `request_offset=100, n=100`, so the confirmatory confidence
interval never reuses a selection request. A later descriptive all-200 table
may combine the frozen-policy splits, but it is not the confirmatory test.

The corresponding Qwen3-8B queue is
`outputs/progressive_kv/live_queue_qwen3_8b_20260827.json`: 12 ready core cells,
20 controlled-confirmation cells, and 14 natural-quality cells. Its model shape
is frozen at 40,960 positions and 147,456 logical BF16 KV bytes/token; the same
25/50 Gbps link protocol and disjoint 100/100 selection-confirmation split are
used.

The scheduling contribution has a separate paired queue at
`outputs/progressive_kv/live_scheduling_queue_llama31_8b_20260827.json`.
At each of 25 and 50 Gbps, one deployment rotates all
`{sequential, uniform, random, BM25, oracle} x {fixed-S1, continuous}`
conditions request by request on the same 64K RULER prompts. The two selection
jobs use rows 0--99; two blocked confirmation jobs use rows 100--199. Every
sidecar is hash-pinned, every runtime bundle must match its sidecar slice, and
oracle is an analysis upper bound rather than an online-policy result.

Before these end-to-end cells are interpreted, the compute-only exact-page
sparse-draft crossover in `docs/SPARSE_DRAFT_CROSSOVER_PROTOCOL.md` scans
16K--128K with a paired 100%-visible control in the same engine. Ordinary
FlashAttention is a secondary metadata-overhead control. This prevents P/D
wire overlap from concealing a draft path which is not itself cheaper.

### Track C: production-style serving

- Real two-node RDMA/TCP P/D deployment.
- Open-loop Poisson arrivals and trace replay.
- Low/medium/high load including the saturation knee.
- Mixed 16K/32K/64K/128K lengths and 16/64/256 output lengths.
- Report goodput under TTFT/TT32/TPOT SLOs, not throughput alone.
- Include destination HBM pressure and multiple P-to-one-D fan-in.

## 3. Models and hardware

Minimum model matrix:

| model | reason |
|---|---|
| Llama-3.1-8B-Instruct | current validated implementation and long context |
| Qwen3-8B | second architecture on natural <=32K LongBench cells |
| DeepSeek-R1-Distill-Qwen-7B | Qwen2-family 128K long-context replication |
| Qwen3-Coder-30B-A3B-Instruct | 262K MoE scale replication |

The local Qwen3-8B config is capped at 40,960 positions and must not be used to
claim 64K/128K generalization.  Llama-3.1-8B and the two long-context
replication models cover those cells without unreported RoPE scaling.

Use H200 for the primary system implementation and at least one second GPU
class if available.  Record GPU SKU, clock policy, CUDA/Torch/Transformers or
vLLM commit, NIC, PCIe/NVLink topology, CPU NUMA placement, and KV precision.

## 4. Independent variables

| axis | values |
|---|---|
| network bandwidth | native H2D, 25, 50, 100, 200, 400 Gbps |
| context length | 16K, 32K, 64K, 128K |
| batch size | 1, 4, 8, 16 and saturation sweep |
| page size | 128, 256, 512, 1024 tokens |
| priority seed | 2%, 5%, 10%, 20%; fixed 1K/2K/4K budgets |
| draft length | 4, 8, 16, 32, 64 |
| schedule | sequential, uniform, random, BM25, P-attention (negative control), adaptive, oracle |
| draft state | fixed-seed, continuous arrival mask, coarse graft, representation refresh |
| verifier | final-only; finite-W is a separately labeled lossy ablation |

Use the same sampled request IDs for all cells in a sweep.  Alternate method
execution order per request and warm every shape before measurement.  Timed
completion comparisons use a fixed token horizon; EOS-truncated generations
are retained only for task scoring.  Natural-EOS completion latency is valid
only when the paired paths stop at the same output length.

## 5. Baselines

Mandatory:

1. Full exact KV monolithic transfer plus ordinary decode.
2. Existing layer-wise P/D transfer pipeline.
3. Full transfer plus MagicDec/TriForce-style local sparse self-speculation.
   QuantSpec hierarchical quantized self-drafting is a separate cheap-draft
   baseline because its official implementation is available. Its source audit
   and required common-harness gates are frozen in
   `docs/QUANTSPEC_BASELINE_AUDIT.md`; it is not a P/D progressive-transfer
   baseline.
4. Lynx under the evidence labels and endpoint separation frozen in
   `docs/LYNX_BASELINE_PROTOCOL.md`. Until author code is available, cite its
   paper results separately and call any local model an oracle or surrogate,
   never a reproduction.
5. SmartGen-style selective transfer/on-demand fetch.
6. OasisKV-style sparse prefetch if code is reproducible.
7. Dustin sparse verification and SpecPV partial verification as explicitly
   lossy latency/quality controls where reproduction is possible.

Context-reuse comparison, kept separate from the P/D main table:

8. Full prefill.
9. Corrected CacheBlend-15 and strict 15% compute.
10. EPIC/LegoLink if reproduced.

Ablations must include no overlap, random ordering, fixed seed, no final-state
reuse, dense-mask draft, and finite-W commitment.

The main exactness table contains only methods whose completed target is the
original BF16/FP16 KV state. Compression methods whose verifier targets a
reconstructed INT8/INT4 cache belong in a separate accuracy/latency Pareto
table even if their task score is statistically indistinguishable from BF16.
The complete role/code/endpoint map is frozen in
`docs/RELATED_WORK_POSITIONING.md`.

## 6. Metrics

Correctness and quality:

- raw token-sequence equality with token-at-a-time full target;
- speculative accepted-prefix length and per-position acceptance probability;
- low-margin verifier fallback rate;
- official task metric and paired delta to full target;
- for sampling, distributional tests under shared seeds rather than raw string
  equality alone.

Use the public LongBench dataset-to-metric mapping: QA F1, ROUGE-L,
retrieval/count accuracy, classification accuracy, and code similarity.  The
runner stores raw predictions, answers, classes, and per-request local scores;
paper tables must also be rescored with the unmodified official evaluator.

Latency:

- TTFT/TTST, TT8, TT16, TT32, TT64;
- time to first committed block;
- request completion latency and TPOT after transition;
- P50/P95/P99 and paired per-request deltas;
- exposed seed, residual, draft, verify, correction, and normal-decode time.

Resources:

- bytes sent before first commit and total bytes;
- useful/cancelled/prefetched bytes;
- HBM and host-memory footprint;
- NIC utilization, GPU SM utilization, and copy/compute overlap;
- page-selection and metadata overhead;
- throughput and SLO goodput per P/D GPU.

Report 95% paired bootstrap confidence intervals and positive-case counts.
For multiple quality datasets, also report macro averages and per-dataset
results; do not hide regressions behind a single aggregate.

## 7. Execution gates

### Gate 0: implementation correctness

- final-only trace contains exactly one verifier invocation;
- every output waits for the full verifier;
- serial and overlapped schedules produce identical tokens;
- full final cache is reused without prompt/history replay;
- missing pages are never read before their completion events.

### Gate 0.5: measurement integrity

- one latency process owns the measured GPU; unrelated compute processes are
  absent for the complete cell;
- both baseline and method shapes are warmed, and which path is timed first
  alternates by request ID;
- baseline and method generate the same fixed number of timed tokens;
- the extra serial/no-overlap diagnostic is not inserted between the paired
  latency paths;
- report a lower bound on unattributed wall time.  A cell is rerun if any
  request has more than 100 ms that cannot be attributed even after summing
  all measured compute, wire wait, and H2D components;
- do not trim outliers from the primary result.  Diagnose and repeat the whole
  contaminated cell under isolation.
- paper quality cells record the public prompt-template mode and official
  generation cap; generic prompts fail this gate.

### Gate 1: mechanism feasibility

- report raw sequence equality and locate numerical divergence;
- obtain 100% equality only with a deterministic verifier/decode kernel or a
  fully charged sequential rematerialization control; margin-only fallback is
  not considered sufficient;
- no statistically significant task-quality loss;
- positive accepted-prefix length on all three task families.

The cheap-draft component additionally requires a valid crossover surface:
all timed prompts are prefix-cache hits except for a changing one-token seed;
request-attributed physical page counts prove actual block-table compaction;
and a <=10%-visible point reaches >=1.10x full-model draft speedup with its
paired saved-time CI above zero. Missing this threshold is retained as a
negative result and triggers the pre-registered cheaper-draft pivot; it is not
reclassified as a measurement failure.

### Gate 2: latency feasibility

- >=1.10x TT32 or response speedup at a declared realistic bandwidth cell;
- paired 95% CI excludes zero;
- no-overlap control is slower, proving gains come from the pipeline;
- measured break-even error is within 10% of the latency model.
- method bypasses to the full-transfer baseline when predicted accepted-decode
  work cannot amortize verification plus exposed drafting.

The live artifact must expose the full request-level decomposition used for
this gate: baseline target-token interval, accepted-decode value, verifier
time, sparse-draft span, modeled residual-wire window, exposed draft time,
predicted gain, observed gain, absolute error, and gain-sign agreement. The
aggregator rejects missing or non-monotonic token/draft timing telemetry before
computing the surface.

### Gate 2.5: progressive-mask contribution

Compare fixed S1 and continuous arrival masks on identical request IDs, page
order, total proposal cap, wire trace, and final verifier.  Record the visible
page set at every proposal.  The progressive component passes only if, on a
fresh 100-request confirmation split at 25 or 50 Gbps:

- accepted prefix improves by at least 2 tokens on average or by 20% relative;
- the paired latency CI versus fixed S1 excludes zero after charging every
  page-table update and completion poll;
- total bytes, final task quality, and verifier count remain matched;
- the result holds for two model families.

If this gate fails, the publishable method is the two-level fixed-seed
arrival-aware pipeline; the word "progressive" refers only to the explored
design space, not a claimed main contribution.

### Gate 3: system result

- real two-node transport reproduces the paced trend;
- positive P95 latency and SLO-goodput results under concurrent load;
- benefits persist on at least two model families and three task families.

## 8. Immediate experiment queue

The machine-readable queue is generated by
`experiments/build_iclr_pd_job_queue.py` and stored at
`outputs/progressive_kv/iclr_job_queue_20260826.json`.  It currently contains
26 executable jobs (one measurement-integrity replacement, one real vLLM
sparse-kernel mechanism probe, and 24 continuous policy-selection cells).  The
43 quality, LongBench-v2, and controlled RULER
cells remain templates until a single policy is frozen from the selection
split; they must not be materialized by inspecting confirmation outcomes.

A separate seven-cell compute queue is generated by
`experiments/build_sparse_draft_crossover_queue.py` and stored at
`outputs/progressive_kv/sparse_draft_crossover_queue_20260827.json`. It covers
four context lengths at gamma 8 plus gamma 4/16/32 at 64K, with 30 balanced
repeats per visible fraction. Its fresh 100-repeat confirmation template stays
blocked until one operating point is frozen.

The request-paired scheduling queue is
`outputs/progressive_kv/live_scheduling_queue_llama31_8b_20260827.json`.
Its 25/50 Gbps selection jobs each interleave five transfer schedules by two
visibility arms over rows 0--99 in one deployment; the matching confirmation
jobs reserve rows 100--199. The benchmark and fail-closed cross-schedule
aggregator are `experiments/benchmark_progressive_pd_scheduling_live.py` and
`experiments/aggregate_progressive_pd_scheduling_live.py`.

The sparse-kernel mechanism job has now passed on one exclusive H200.  On an
8,192-token QMSum request, the seven draft positions observed exact page counts
`3, 32, 63, 95, 126, 126, 126` out of 126 candidates; all seven proposals were
accepted and the immutable full target exactly reproduced the nine-token full
greedy output.  This closes only the single-request mechanism gate.  Its result
is `outputs/progressive_kv/iclr_queue/kernel_llama_qmsum_8k_g8_progressive_gpu1_20260827_fix2/summary.json`.

The next live gate no longer uses the external-proposer bridge. The scheduler
now shares target weights and physical prompt blocks for hidden sparse
drafting, consumes exact LMCache token-range completions, pauses at its
proposal cap, and invokes one ordinary full-KV verifier. A paced serialized
link control and 1P1D proxy are implemented. The fixed input contains four
warmups plus 100 paired, exact 8,192-token RULER requests at
`outputs/progressive_kv/live_gate_requests.jsonl`; its manifest freezes the
hash, placement, and token horizon. This is deployment readiness, not a new
latency result.

1. Rerun the same 100-request QMSum 100-Gbps cell on one isolated H200 using
   `--fixed-token-horizon --paired-measurement`.  This replaces, rather than
   supplements, the contaminated three-process run.
2. Confirm the latency surface on matched request IDs: 25/50/100/200/400 Gbps,
   `gamma` 2/4/6/8/12, and priority budgets 2%/5%/10%.  Use 30 requests for
   selection and a fresh 100-request shard for the chosen policy.
3. Scale quality to at least 200 examples each on 2WikiMQA, HotpotQA, MuSiQue,
   Qasper, QMSum, GovReport, and PassageRetrieval-en.  Short-answer datasets
   are correctness/acceptance tests; QMSum/GovReport are completion-latency
   tests.
4. Repeat the frozen policy on Qwen3-8B and one 20--32B model.  Hyperparameters
   selected on Llama are not retuned on the test split.
5. Run controlled 16K/32K/64K/128K evidence-first, uniform, evidence-last, and
   adversarial-split cases.  This is the direct falsification test for the
   page-scheduling hypothesis.
6. Run the shared-paged-KV scheduler against a live LMCache 1P1D deployment.
   The code path and paired input are ready, but all local H200s are currently
   occupied by unrelated jobs. Until an isolated run completes, report only
   the earlier event replay as mechanism evidence and no end-to-end speedup.
7. Move stable cells to real two-node RDMA/TCP, then run open-loop concurrency
   and SLO-goodput sweeps.  Paced single-host numbers are not two-node claims.
8. At 25/50 Gbps, compare fixed S1 with continuous page-table expansion for
   16/32 proposals and 4/8/16 arrival bundles.  Keep mechanical coarse graft
   and per-stage verification as negative controls; no replay cost may be
   omitted from an end-to-end result.

## 9. Preliminary decision surface (2026-08-26)

The original 16-draft QMSum smoke was negative at 100 Gbps because drafting
exceeded the residual window.  A six-proposal, configured-2% BM25 seed makes
the exploratory natural-EOS means positive on QMSum, 2Wiki, HotpotQA, and
MuSiQue.  The larger QMSum run exposed a protocol flaw: paired paths sometimes
generated different token counts, and a later fixed-horizon three-GPU run had
15/100 requests with more than 100 ms of unattributed pipeline wall time.
That run saves 8.0 ms on average with a CI of `[-34.6, 39.8]` and is not a
paper latency result.  It is being replaced by the isolated paired protocol
specified in Gate 0.5.

Multi-stage graft increases acceptance only on a minority of 2Wiki requests
and has no significant latency advantage.  Mechanical intermediate tentative
verification and producer-attention page ranking are negative ablations.  Do
not scale those cells before an online trigger and in-process paired runner
exist.
