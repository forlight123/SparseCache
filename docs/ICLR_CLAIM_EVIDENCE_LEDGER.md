# SparseCache-PD claim/evidence ledger

Status: preregistered working ledger, updated 2026-09-10. A row marked `missing` or
`preliminary` is not a paper claim.

## 1. Main claim and boundary

The intended claim is narrowly scoped:

> In transfer-bound P/D disaggregation, an exact token-sparse KV Anchor can be
> used by a genuinely sparse draft path while
> the exact residual is still in flight. The immutable full-Target verifier
> executes layer `l` only after that layer's complete exact KV arrives, while
> later-layer KV may remain in flight. This preserves the Target endpoint and
> can reduce transition/completion latency.

The method does not claim that independent-document KV composition is repaired,
that partial KV may be committed without verification, or that mixed-precision
KV is novel. CacheBlend remains a separate context-reuse baseline. The exact
policy is `W=infinity`: no draft token is externally committed before all exact
Target layers and the final acceptance step complete.

## 2. Claim ledger

| ID | prospective claim | required evidence | present evidence | status |
|---|---|---|---|---|
| C0 | the shared P/D state machine is exact by construction | hidden producer seed; no draft output; every Target layer waits for its complete original KV and exact predecessor hidden state; one immutable acceptance step; worker input chain matches scheduler state | real LMCache P/D supplies exact P seed and registered Anchor views to an online direct-KV drafter; repaired suffixes are committed only after full-KV Target verification; exact token-ID replay matches monolithic on 64/64 greedy requests across observe--inject--observe. The new all-layer probe is bitwise identical between wait-full and layer-ready execution for the same eager block arithmetic. A separate P-runahead control is 32/32 equal only at chunk-aligned boundaries and 7/8 at an unaligned boundary, confirming that a bitwise claim also requires canonical transferred state/numerics | structural mechanism passed; deployed layer-ready state machine, sampling, and cross-shape bitwise contract pending |
| C1 | draft attention is genuinely sparse and becomes cheaper with context | request-attributed physical page counts, full-model draft time, profiler memory traffic/kernel attribution, and dense-vs-sparse crossover at 16K--128K | the valid trace-isolated 128K `n=30` cell addresses 5.04% of logical BF16 KV but takes 168.92 ms versus 161.05 ms at 100% (`0.953x`, paired saved-time CI `[-16.25, 0.43]` ms). Compact-table reuse improves the implementation, but the preregistered `>=1.10x` crossover is absent. Nsight attributes about 32 ms less attention-kernel work at 5%, yet the generic FA3 compact path adds 256 per-layer scheduling kernels and CPU launch gaps; an AOT-schedule attempt removed those kernels but introduced a roughly 40-ms synchronous schedule build and was reverted | current same-model/page-only path falsified; cheaper-draft pivot required |
| C2 | sparse drafting and layer-ready exact verification jointly improve latency in a realistic transfer-bound regime | n=100 paired cells at 25/50 Gbps; wait-full/layer-ready x no-draft/sparse-draft; P50/P95/P99; paired 95% CI; >=1.10x primary speedup | a within-process, order-balanced 100-Gbps 2x2 H200 probe hides 48.83 ms in the sparse arm, CI [47.26,49.91], and reaches 1.733x mean per-request equal-progress speedup. Its difference-in-differences is 0.099 ms, CI [-0.545,0.681], showing that layer-ready and proposal-progress gains are not double-counted. Preliminary 25/50-Gbps layer-ready cells reach 1.226x/1.419x ratio-of-means and zero draft overrun. All cells have one eager block-vs-sequential mismatch request and only n=64 | positive structural mechanism; n=100 exact-output and real two-node gates pending |
| C3 | continuous page arrival is better than fixed S1 | fresh confirmation split; >=2 accepted tokens or >=20% relative acceptance gain; paired latency CI excludes zero; matched bytes/quality | the valid shared-prefill 64K/25-Gbps `n=30` screen found zero useful-acceptance differences in all 150 schedule-request pairs; all five latency CIs cross zero. A follow-up 1%-tranche, `gamma=16`, `n=10` screen made 6--8 visibility levels observable in every continuous request and changed 8/50 draft token sequences, but changed useful accepted tokens in 0/50 pairs. Its 100-bundle transfer also accumulated about 0.23--1.33 s of observed fragmentation/control overhead above the 2.749-s modeled wire | multi-level continuous contribution falsified at both coarse and fine granularity; fixed-anchor pivot |
| C4 | page order controls proposal quality | five matched RULER layouts and request-paired sequential/uniform/random/BM25/oracle schedules at fixed bytes/deadlines; acceptance and latency interaction | all gates pass on the shared-prefill 64K/25-Gbps `n=30` screen. EOS-aware useful acceptance is 39.67% sequential, 2.48% uniform, 4.96% random, and 7.44% for both BM25 and oracle. Offline evidence coverage therefore does not predict sparse next-token alignment; semantic ordering is not presently supported as a contribution | valid negative screen; contribution at risk |
| C5 | the exact endpoint preserves task quality | 100% greedy token equality primary gate; official LongBench/LongBench-v2 metrics; per-dataset paired deltas | strict native-length QMSum packet replay gives 64/64 monolithic greedy equality; an earlier 59/64 text replay is superseded because retokenization changed the input; official task-diverse quality remains unrun | exact pilot passed; paper quality missing |
| C6 | the result generalizes | two model families; 16K/32K/64K/128K; second GPU class if available; real two-node transport | Qwen3-8B packs and 46-job queue materialized at <=32K; all live Qwen, long-context replication, second-GPU-class, and two-node evidence still missing | missing |
| C7 | gains are not an artifact of an unfair link or P-side cache hit | P=`kv_producer`, D=`kv_consumer`; identical total logical bytes/rate/wire time for every arm; protected S1 bytes charged; exclusive GPUs | real LMCache roles are fixed; observe and inject execute identical transfer/draft/repair work; exact prompt IDs and full lengths are shared; a fresh-server sandwich plus paired monolithic endpoint controls run drift; multi-request/two-node confirmation remains pending | preliminary live proof |
| C8 | comparison with Lynx does not conflate endpoints or unsupported reproductions | separate exact-BF16 and accuracy/latency tables; author-paper, official-reproduction, oracle, and surrogate labels | paper configuration and public reproducibility gaps audited in `docs/LYNX_BASELINE_PROTOCOL.md` | protocol frozen, live baseline missing |
| C9 | adjacent sparse-transfer/draft/verification systems are compared at the bottleneck and guarantee they actually target | endpoint, code, hardware, and experiment-role ledger for SmartGen, OasisKV, MagicDec, TriForce, QuantSpec, Dustin, and SpecPV | `docs/RELATED_WORK_POSITIONING.md`; executable-source status frozen as of 2026-08-27 | protocol frozen, reproductions pending |

## 3. Frozen primary decision rules

All primary latency comparisons use fixed 32 output tokens, greedy decoding,
one request at a time, alternating three-arm order, and no outlier removal.

1. A cell is invalid unless every request passes the complete paired matrix,
   unique request ID, natural output horizon, one immutable verifier, complete
   same-layer KV before each Target layer, request-attributed sparse attention,
   fair-link, and exact-output gates.
2. Primary feasibility passes only when `baseline - continuous` completion
   latency has a positive 10,000-sample paired-bootstrap 95% CI and mean
   speedup is at least 1.10x at 25 or 50 Gbps.
3. Continuous visibility passes its own contribution gate only when the fresh
   confirmation split improves accepted prefix by at least 2 tokens or 20%
   relative to fixed S1 and its paired latency CI excludes zero.
4. The first 100 RULER rows are selection only. Rows 100--199 are confirmation
   only. An all-200 descriptive table may be produced after policy freeze, but
   it is not the confirmatory test.
5. Natural-quality cells use their materialized official output protocol.
   Raw greedy token equality is the primary exactness gate. If numerical kernel
   order prevents bitwise equality, the paper must report that boundary and a
   fully charged deterministic/sequential-rematerialization control; task score
   alone cannot replace the exactness claim.
6. A secondary non-inferiority summary may use a -0.005 absolute task-score
   margin, but every dataset and its CI remains visible; macro averaging cannot
   hide a task regression.

## 4. Required ablations

| ablation | isolates | frozen comparison |
|---|---|---|
| baseline monolithic transfer | unavoidable P/D wait | full exact retrieve + ordinary decode |
| fixed S1 | overlap without progressive masks | identical anchor, bytes, draft cap, verifier |
| continuous arrival mask | incremental metadata-only visibility | fixed S1 vs continuous |
| dense draft mask | sparse-kernel contribution | same state machine with full prompt pages |
| wait-full verifier | layer-ready exact-verifier contribution | same proposal, Target arithmetic, bytes, progress, requests, and four-arm rotated order |
| random/sequential/uniform/BM25/query/oracle order | scheduling contribution and upper bound | fixed anchor bytes and wire trace |
| gamma 4/8/16/32 | proposal/verification tradeoff | matched requests and bandwidth |
| anchor 2/5/10/20% | first-visible working set | matched total bytes |
| native/25/50/100 Gbps | break-even surface | same logical KV payload |
| final verifier removed or finite W | explicitly lossy control | never mixed into exact tables |

## 5. Planned paper figures and tables

1. System timeline: monolithic wait versus exact-anchor/sparse-draft/residual/
   immutable-verify pipeline.
2. Break-even surface: measured net gain over context length, bandwidth, gamma,
   and anchor fraction, with the analytic boundary overlaid.
3. Fixed S1 versus continuous: accepted-prefix CDF and paired latency deltas.
4. Scheduling stress test: evidence-first/uniform/last/adversarial acceptance
   and quality at identical bytes.
5. Sparse-kernel scaling: per-token draft time and HBM bytes versus visible-page
   fraction and full context length.
6. Main latency table: P50/P95/P99, mean paired gain with CI, speedup, verifier
   time, acceptance, and total bytes.
7. Quality table: official per-dataset metric, baseline, method, paired delta,
   CI, and raw token equality.
8. Two-node validation: paced one-node prediction error and real network result.

## 6. Falsification and pivot rules

- If sparse draft plus verification exceeds the hideable residual window in all
  realistic cells, do not claim a latency system win; report the break-even
  analysis and move to a cheaper draft model/path.
- If continuous visibility fails the fresh fixed-S1 confirmation gate, the main
  method becomes a two-level fixed-anchor pipeline; “progressive mask” moves to
  a negative ablation.
- If online scheduling does not outperform uniform/random at matched bytes,
  remove scheduling as a contribution.
- If exact greedy equality fails, locate whether the cause is implementation
  state contamination or numerical verifier/decode order. A lossy quality result
  cannot be presented as exact speculative decoding.
- If paced one-node gains do not reproduce on two physical nodes, the paper may
  claim a mechanism/break-even result only, not production P/D speedup.

## 7. Authoritative artifacts

- Method: `docs/METHOD_PROGRESSIVE_SPARSE_PD.md`
- Full matrix: `docs/ICLR_EXPERIMENT_MATRIX.md`
- Live runbook: `docs/RUN_PROGRESSIVE_PD_LIVE.md`
- Machine queue: `outputs/progressive_kv/live_queue_llama31_8b_20260827.json`
- Second-model queue:
  `outputs/progressive_kv/live_queue_qwen3_8b_20260827.json`
- Paired semantic-scheduling queue:
  `outputs/progressive_kv/live_scheduling_queue_llama31_8b_20260827.json`
- Shared-prefill scheduling mechanism audit:
  `outputs/progressive_kv/live_scheduling_debug_shared_prefill/llama31_8b/scheduling_selection_ruler_65536_original_bw25_n2_offset0/input_audit.json`
- Valid shared-prefill 64K/25-Gbps scheduling screen (`n=30`, EOS-aware):
  `outputs/progressive_kv/live_scheduling_screen_s5_n30_shared_prefill/llama31_8b/scheduling_selection_ruler_65536_original_bw25_n30_offset0/aggregate_eos_aware/summary.json`
- Valid 128K exact sparse-draft crossover after compact-table reuse and timing/
  trace isolation:
  `outputs/progressive_kv/sparse_draft_crossover/crossover_llama31_8b_c131008_g8_r30_compactreuse_traceisolated/summary.json`
- Matching 128K Nsight mechanism trace (profiler wall time is non-publishable):
  `outputs/progressive_kv/nsys_crossover_llama31_8b_c131008_g8_compactreuse_traceisolated.nsys-rep`
- Fine 1%-tranche, `gamma=16`, `n=10` scheduling screen:
  `outputs/progressive_kv/live_scheduling_screen_s5_t1_g16_n10/llama31_8b/scheduling_selection_ruler_65536_original_bw25_n10_offset0_tranche100bp/aggregate/summary.json`
- The first scheduling `n=30` screen is quarantined under the suffix
  `invalid_independent_producer_prefills_20260827`: five independent P
  prefills per source request caused three cross-schedule producer-seed drifts,
  so none of its acceptance or latency values are paper evidence.
- Worker-input mechanism audit:
  `outputs/progressive_kv/live_matrix_debug_input_trace/llama31_8b/core_ruler_65536_evidence_first_bw25_n2/input_audit.json`
- RULER audit:
  `outputs/progressive_kv/pd_packs/llama31_8b/controlled/ruler_qa2_audit.json`
- LongBench audit:
  `outputs/progressive_kv/pd_packs/llama31_8b/quality_audit.json`
- LongBench-v2 audit:
  `outputs/progressive_kv/pd_packs/llama31_8b/longbench_v2_audit.json`
- Qwen3-8B audits: `outputs/progressive_kv/pd_packs/qwen3_8b/quality_audit.json`,
  `outputs/progressive_kv/pd_packs/qwen3_8b/longbench_v2_audit.json`, and
  `outputs/progressive_kv/pd_packs/qwen3_8b/controlled/ruler_qa2_audit.json`
- Runtime validity aggregator: `experiments/aggregate_progressive_pd_live.py`
- Exclusive resource gate: `experiments/progressive_pd_resource_gate.py`
- Fail-closed launcher: `experiments/run_progressive_pd_live_job.py`
