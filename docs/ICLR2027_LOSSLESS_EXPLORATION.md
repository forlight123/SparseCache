# SparseCache: lossless progressive P/D exploration

Date: 2026-09-09. Status: executable pilot and gated research plan.
This supersedes the EAGLE-adapter scale-up recommendation in TRAIN.md.
It does not claim paper-scale training, cross-node deployment, or ICLR acceptance.

## 1. Fixed research contract

The target is the ORIGINAL model with its complete, contextualized BF16/FP16
prompt KV and the declared decoding policy. Approximation is confined to the
proposer. No finite-W commitment is allowed. A candidate is externally visible
only after the fixed target's exact decision, or a sound certificate for that
same decision. Greedy equality, distributional exactness, and floating-point
kernel reproducibility are separate gates.

The deployment is long-context P/D handoff. It does not concatenate independently
prefilled documents or invoke CacheBlend repair. Train/test teacher states must
match this exact-context endpoint. The P-side seed is available equally to every
baseline. Record post-seed first commit, TT8/TT32/TT64 and end-to-end latency;
never count tentative internal drafts as client-visible TTFT.

Research hypotheses:

1. Training a KV-input block drafter on nested arrival views increases useful
   accepted prefixes per unit compute versus full-view-only training.
2. Scheduling layer/page transfers jointly with the exact verifier's dependency
   graph reduces exposed verification time beyond draft/transfer overlap alone.
3. Strict bounds on missing KV can sometimes resolve the SAME target decision
   before all KV arrives. This is a separate high-risk extension, not a premise
   of the lossless baseline or a confidence-threshold approximation.

## 2. Available hardware and allocation

Read-only audit found three idle NVIDIA H200 NVL GPUs, each 143771 MiB, with no
compute processes. GPU 0/1 are on NUMA 0 and linked by NV18; GPU 2 is NUMA 1,
with SYS paths to GPU 0/1. Available filesystem space was about 933 GiB.
Use SparseCache/.venv: Python 3.13, torch 2.13.0+cu130, transformers 4.57.6.
Do not use or mutate the adjacent CacheDraft virtual environment.

| Period | GPU 0 | GPU 1 | GPU 2 |
|---|---|---|---|
| Correctness smoke | imported KVShot P/D seed adaptation | KV block forward/backward and sparse positions | exact attention merge + fixed-query bound |
| Initial pilot | KVShot acceptance, then full-view block continued training | matched nested-view block continued training | real-query bound scan + local async-copy mechanism |
| Model scaling | independent training run/seed | independent training run/seed | online frozen teacher, data production, held-out audit |
| Final P/D timing | D worker | idle or compute-isolated diagnostics | P worker |
| Native-link control | P or D worker | paired D or P worker | idle |

Three independent workers are preferable to initial DDP: the model fits on one
GPU and GPU 2 is not part of the GPU 0/1 NVLink pair. DDP becomes an option only
after architecture selection and communication profiling. Training may run
concurrently; publication latency runs must isolate shared CPU/memory/storage
and communication resources, not just GPU SM utilization.

## 3. Imported baselines and exact input alignment

See experiments/kvshot/PORT_PROVENANCE.md for working-tree source hashes.

* KVShot: local clean-room AR implementation, NOT author code. The available
  checkpoint is TWO layers with a 32K vocabulary and historical EAGLE weight
  initialization. No runtime target hidden state is supplied. Its P/D wrapper
  uses only prompt KV and a known seed; seed KV is created by the drafter.
* Block control: five-layer, full-vocabulary, DFlash-initialized KV-input model
  with KV layers 1,9,17,25,33. KV is projected to internal memory; no target
  hidden-history tensor or seed-hidden feature is passed at inference.
* Progressive variant: same block weights and training data, physically compact
  received KV, original absolute positions, nested view budgets, and a
  zero-initialized stage projection. MASK positions contain no future tokens.

The selected block warm start is
../CacheDraft/results/blockdraft/qwen3_8b_dflash_b16_exactkv_hiddenadapter_joint2500.
Its metadata reports 4946 train IDs and 550 validation IDs. Continued training
intersects the new corpus with parent train IDs and excludes parent validation
IDs. Checkpoint filenames are not evidence of completed steps or architecture.
The current probe is a model-family comparison, not a parameter-, vocabulary-,
or KV-layer-matched architecture claim. Those controls are required later.

## 4. Phase 0: bounded pilot, three parallel lanes

First run a 2-request smoke, then an initial 64-request model screen. QMSum
indices 0..63 are DEVELOPMENT data for this pilot, never the final blind set.
Use the exact same native-Qwen request token IDs and P-side seed semantics.
Inputs above 8192 tokens use explicitly recorded head/tail truncation. This
is a mechanism screen, NOT official QMSum task-quality evaluation or a 64K result.

Model screen:

* Qwen3-8B frozen target, BF16, greedy, block 16 = known seed + 15 proposals.
* Visibility 5/10/20/50/100%; priority and random order. First/last pages are
  protected; record actual fractions after rounding. Reverse-priority is an
  optional stress test, not a ground-truth evidence-last oracle.
* Priority comes from the P-side LAST PROMPT query; no answer/future query is
  used. Layer selection differs for the historical KVShot and block models.
* Zero and other-request shuffled KV at 10%, preserving shape and original
  positional frame. Shuffled control omits request 0, for which no donor exists.
* Reference suffix is independently autoregressive full-target greedy decoding,
  stopping at EOS. Report longest accepted prefix, zero-acceptance probability,
  paired request-bootstrap CI, original/actual context and bytes, draft time
  including memory projection. The seed does not count as an accepted draft.

Matched continued-training screen:

* GPU 0 full-only versus GPU 1 nested views, identical block initialization,
  record sequence, cut RNG, optimizer, 256 parent-train records, 1000 updates,
  <=2048 training tokens, full-vocabulary teacher distillation, all block
  parameters trainable. This is a bounded adaptation, NOT large-scale training.
* Sample only assistant windows. At cut c, the draft sees KV at positions <c,
  token c as known seed, and MASKs for unknown positions. Target future states
  exist solely for loss computation; P priority query is c-1, not the final
  teacher-forward position. Each step supervises 15 future positions.
* Stage views are prefixes of one page ordering. This first test trains view
  snapshots; it does NOT yet model a continuously changing view within a block.
* Evaluate identical development requests before and after training. Persist
  record IDs, cut positions, fractions, loss, gradients, checkpoint hash and
  teacher-data hash. No EAGLE-adapter training job is launched.

GPU 2 fixed-query bound/streaming screen:

* 64 real requests, target layers 0/17/35, exact P-prefill last query, the same
  five page budgets. Measure omitted attention mass, its strict mathematical
  center/radius upper bound, attention output error and bound tightness.
* Compute bounds in FP64 and audit every inequality. This is NOT an
  outward-rounded implementation or an end-to-end logit certificate. Queries
  from deep layers are an oracle-ready-query diagnostic, not cheap D-side data.
* On layer 17 compare serial-copy-then-incremental-attention against real pinned
  H2D with a separate copy stream and per-tile CUDA events. Tile sizes
  256/1024/4096, warm both paths, alternate execution order, five repetitions.
* Full SDPA must be added as a serving-performance control before any speedup
  claim. The pilot compares identical incremental arithmetic to isolate overlap.
  The transfer is ONE LAYER's real local H2D, not all-model KV or network P/D.

Pilot decision gates are operational thresholds, not theoretical constants:

* All availability, position, EOS and finite-gradient checks pass.
* Full-view mean accepted prefix >=2 is an initial model-readiness target;
  sparse 10/20% should exceed 1 and beat zero/shuffled with a positive paired
  CI. Failure triggers model/data diagnosis before scaling.
* Nested training must beat full-only at common sparse budgets without hiding
  a full-view regression. Primary endpoint: priority 10%, accepted prefix.
* Bound violations must be zero within the explicitly declared numerical
  tolerance. If missing-mass upper bounds stay near 1, abandon claims of cheap
  early commitment and continue the exact final-verifier path.
* Kernel equality or better acceptance alone does not pass an E2E speed gate.

## 5. Phase 1: train a usable block drafter

After pilot correctness, use 10K -> 70K unique target-regenerated conversations
plus a disjoint long-context training mixture. The 70K figure is a scaling
target, not a corpus currently ready on disk. Audit availability and lineage
before data generation. Train/validation/test split by source document and
conversation, not by boundary/window; near-duplicate documents are grouped.
Do not train on the LongBench evaluation split.

Use online or bounded-LRU teacher extraction. Store token IDs, sampled layer
KV only when justified, sparse selection metadata, teacher distributions and
manifests. Do not persist every full-layer KV state for every training window.
Report unique records, unique supervised positions, processed positions and
FLOPs/GPU-hours separately; optimizer steps do not measure data scale.

For H200 utilization, pack 16/64 independent assistant anchors from one teacher
prefill into a block-training pass, with per-anchor prefix/arrival masks and
strict inter-block isolation. Sweep packing size against real tokens/s,
memory use and optimizer-update cost. Use measured matmul/attention occupancy
to choose microbatch size; the initial one-anchor pilot is not a throughput
optimized trainer. Keep sampler RNG and total supervised positions matched
between full-only and nested-view controls.

Start with the two best configurations rather than a full Cartesian sweep:

1. Full-only KV block versus nested-view KV block, matched initialization.
2. Stage input on/off; fixed-view versus view changes BETWEEN sealed blocks.
3. Full-vocabulary AR versus block training with matched KV layers and depth.
4. Two/four/five layers after the learning gate; compare acceptance per ms.
5. Prefix-weighted distillation versus uniform weights and on-policy boundaries.

Three seeds for finalists. First milestone: a >=1000-request held-out model
evaluation across chat, summarization, QA and code, clustered by request.
Teacher-forced accuracy is diagnostic only; candidate acceptance uses live
target verification. Adding KV need not improve every individual example.

## 6. Phase 2: exact verifier and theory-driven scheduling

Implement a verifier whose partial work is valid only for a fixed candidate
block and an EXACT target query. Layer l+1 cannot use the hidden state of an
unfinished layer l. Draft-generated KV never enters authoritative target state.

For the simple layer-complete, uncontended model:

    F[0] = draft_ready
    F[l] = max(F[l-1], KV_layer_ready[l]) + verify_cost[l, block_length]

Measure every release and completion timestamp. Validate model prediction error
before using it for online scheduling. Extend to page accumulation only after
the full-layer dependency gate passes. Page scores must account for both
proposal improvement and completion of verifier-critical layers.

Choose a transfer schedule pi and block length g to maximize the calibrated
finite-candidate estimate:

    ordinary_decode_ms * E[accepted(g)+1]
      - (verify_finish(pi,g) - baseline_full_transfer_finish)
      - extra_cost_not_already_in_verify_finish

This is a restricted optimization under measured assumptions, not a theorem of
global optimality for arbitrary load. Acceptance lengths use actual prefix
survival probabilities; do not multiply unconditional per-stage agreement rates.

Required ablations: full-transfer/no-draft; layer-wise exact/no-draft;
fixed-anchor block/final verify; progressive block/final verify;
fixed-anchor/streamed verify; joint scheduling/streamed verify; serial-overlap
disabled control with identical computations. Add the official-author Lynx
baseline only if reproducible; otherwise use its documented evidence label.
Its quantized target endpoint must not be conflated with original BF16 KV.

## 7. Phase 3: strict early-decision certificates (separate go/no-go)

Fixed-query page metadata: key centers/radii, value-norm maxima, page counts,
original positions; charge P-side extraction, metadata bytes and D computation.
For a full-model certificate, propagate intervals through actual target
normalization, attention, MLP, residuals and LM head. Empirical confidence,
learned bounds, a few unchanged stages, and an oracle full-target query do not
prove an early output is correct.

Greedy criterion: lower_logit[winner] > max upper_logit[other].
Sampling criterion for a frozen proposal y~q and independent U:
U*q(y) <= lower_p(y) guarantees the SAME exact acceptance decision. Unresolved
tokens wait; rejection still requires exact residual sampling. A token decision
does not make its approximate KV authoritative: exact representation completion
must be scheduled before normal target continuation.

Start with tiny one/two-layer models where all compatible toy completions can
be enumerated, then one real target layer, then full Qwen3-8B. Report certificate
coverage vs transferred bytes, runtime vs saved wait, false certificates (=0),
and worst-case evidence-last requests. Stop this extension if sound bounds are
too loose or more costly than exact computation.

## 8. Final experiment matrix and statistical discipline

Qwen3-8B main: native contexts 4/8/16/32K. A separately configured native-long
model (e.g. the existing Llama-3.1-8B) supplies 64/128K. Never extrapolate Qwen's
position limit without declaring and separately validating a scaling change.
P context = full prompt; D = known seed plus 32/64/128/256 generated tokens.
Natural EOS for task scores; fixed horizon is a separately labeled timing probe.

Transport regimes: native GPU0<->1 NVLink, actual GPU0<->2 SYS path, measured
local pinned H2D, and explicitly labeled link-paced PD emulation at
10/25/50/100/200 Gbps (1.25/3.125/6.25/12.5/25 GB/s before overhead).
Rate emulation must pace serialized payload bytes on the transfer producer,
keep a continuous wire, and publish readiness only AFTER real copies finish.
A one-host experiment is not real inter-node NIC/RDMA evidence.

Start with 32K x 25/100 Gbps x fixed32/128 outputs x 100 requests/cell;
expand only passing configurations. For each final cell use >=200 distinct
requests where available, paired ordering, warmup exclusion, request-clustered
bootstrap, three independent run repetitions and natural-length controls.
Final QMSum blind holdout excludes pilot indices 0..63; document any remaining
benchmark reuse and keep an additional untouched task set.

Multi-request: concurrency 1/4/8/16, controlled arrival process, >=1000 completed
requests AND a stable >=5-minute measurement interval for throughput/P99 claims.
Count P compute, draft/verify compute, actual wire bytes, metadata, all copied
pages, cancellation waste, memory projection and GPU contention. Include
unaccepted proposals and recovery/rematerialization in latency.

Exactness gates: sequential reference greedy trajectories, batched verification
logits and generated KV audits, deterministic comparison controls, tiny-vocab
exact probability enumeration, stochastic distribution tests, EOS/stop/top-p
handling, no premature publication, correct rollback and branch-state removal.
Statistical sampling tests supplement the proof; they cannot prove exactness.

## 9. Artifacts and budget discipline

Outputs live in outputs/progressive_kv/iclr2027_20260909/ and are ignored by Git.
Code, provenance, this protocol and a compact factual report are reviewable in
the repository. Each run keeps arguments, checkpoint hash, lineage, actual input
hashes, software versions, counts and completion state. Do not overwrite a run.

Budget the initial pilot at a maximum of 1000 updates per training arm and
64 development requests, then use measured time/step to estimate larger runs.
Do not promise a calendar completion date before profiling. Large training
queues are released only after model/causality gates. In the timing phase, idle
an otherwise free GPU when its work would contaminate the shared interconnect;
valid measurements take precedence over 100% utilization.

The launched pilot can be reproduced with a fresh output directory:

```bash
.venv/bin/python -m experiments.lossless_pd.launch \
  --output outputs/progressive_kv/iclr2027_20260909/pilot_n64_s1000 \
  --steps 1000 --requests 64
.venv/bin/python -m experiments.lossless_pd.report \
  --root outputs/progressive_kv/iclr2027_20260909/pilot_n64_s1000
```

The reporter audits shared initialization, identical training records/cuts and
identical target references before publishing the paired training comparison.
If reference replay drifts, it reports the failed gate without dropping those
requests or silently substituting reference tokens. The next paired run must
share an immutable P-prefill state and reference packet across conditions.
Concrete single-worker entry: python -m experiments.lossless_pd.pilot --help.
The program includes no automatic external upload, package installation,
unbounded training, network modification or process killing.

The immutable-packet and integrated second round uses these entry points:

```bash
.venv/bin/python -m experiments.lossless_pd.reference_packets build --help
.venv/bin/python -m experiments.lossless_pd.packet_eval evaluate --help
.venv/bin/python -m experiments.lossless_pd.verifier_probe --help
.venv/bin/python -m experiments.lossless_pd.sequence_equivalence --help
.venv/bin/python -m experiments.lossless_pd.schedule_analysis --help
.venv/bin/python -m experiments.lossless_pd.integrated_probe --help
```

The paced-link probe launches a CPU producer thread, delays logical readiness
according to serialized payload bits, and still performs every pinned H2D copy.
Selected anchor pages are scattered into their authoritative full-cache slots
and excluded from residual transfer. Its evidence label remains a one-host link
proxy until a real two-process NIC/RDMA or LMCache transport reproduces it.

An additional full-model verifier probe was implemented after the attention
smoke. It uses the real block proposals from the completed model pilot, all 36
Qwen target layers, one shared P-prefill per request, and exact per-layer CUDA
copy events. It compares native HF, same-work serial, and same-work streamed
copy+verification, auditing logits AND generated KV for bitwise equality:

```bash
CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=8 \
numactl --cpunodebind=1 --membind=1 .venv/bin/python \
  -m experiments.lossless_pd.verifier_probe \
  --proposals outputs/progressive_kv/iclr2027_20260909/pilot_n64_s1000/nested_block_gpu1/before.json \
  --requests 64 --repeats 3 \
  --output outputs/progressive_kv/iclr2027_20260909/verifier_n64_gpu2
```

This is a copy+verify subgraph experiment, not end-to-end PD. Its setup excludes
P prefill, host pinning and draft generation; all three timed arms use the same
preallocated input buffers and warmup policy. The native control is warmed but
not execution-order-balanced, so the paired serial-versus-streamed comparison
is primary. See ICLR2027_PILOT_REPORT_20260909.md for measured outcomes.
