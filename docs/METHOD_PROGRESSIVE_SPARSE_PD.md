# SparseCache-PD: exact-page progressive sparse self-speculation

Status: working method specification, 2026-08-27.

## 1. Target setting

SparseCache-PD targets prefill/decode disaggregation when the complete,
contextualized KV cache already exists on a P worker or remote KV service but
cannot reach the D worker instantaneously.  The initial method does not depend
on independent-document KV composition and does not repair CacheBlend-style
position/context errors.  Every transmitted K/V element remains BF16/FP16.

The baseline blocks D until all prompt KV arrives and then performs ordinary
autoregressive decoding.  SparseCache-PD transmits exact KV pages in priority
order, uses the arrived subset as a cheap self-draft context, and validates the
entire tentative sequence once the complete prompt KV is available.

## 2. State and invariants

Let the full prompt cache be partitioned into immutable logical pages
`P = {p_1, ..., p_m}`.  A schedule `pi` produces cumulative available sets
`A_0 subset ... subset A_J = P`.

The following invariants are mandatory for the exact endpoint.

1. A page is visible to attention only after its transfer completion event.
2. K retains the RoPE coordinate of its original prompt position.  Physical
   page order at D may differ because attention over historical K/V is
   permutation invariant once positions are encoded in K.
3. Draft tokens are never exposed before the immutable full-KV verifier.
4. The verifier always starts from the complete exact prompt KV and recomputes
   all generated-token representations.  Draft representations never enter
   the target continuation cache.
5. Verification runs once per request transition, not once per arrival level.

The current exact policy is therefore equivalent to `W = infinity`.  Finite-W
early commitment is a separate lossy operating point and must never be mixed
with exact results.

## 3. Pipeline

```text
P / remote KV service                    D worker

score pages and form A0
priority-send A0 ----------------------> install exact sparse prompt KV
send P \ A0 in background --------------+ draft gamma tokens with sparse KV
                                         |
full exact KV completion --------------> one parallel full-KV verification
                                         |
                                         + accept longest prefix
                                         + correct first rejection
                                         + continue ordinary target decoding
```

The first implementation uses two transfer classes: an exact sparse seed and
the exact residual.  The multi-stage implementation subdivides the residual
into cancelable pages, but page arrivals do not create new verifiers.
The first bundle is an atomic correctness anchor: it always contains the first
256 prompt tokens and final 512 prompt tokens, including the system boundary,
question, and generation boundary. These chunks are installed before drafting
and are never candidates for sparse-attention removal. If they exceed a
configured fractional S1 budget, S1 is rounded up and the actual bytes and
completion fraction are reported; later fraction targets below that rounded
anchor are skipped rather than creating a regressing level.
All tranche deadlines are cumulative on one continuously occupied wire: D-side
draft computation never delays the start of the next network tranche.  A page
is installed by real H2D only after its modeled wire completion event.

Three draft policies are intentionally separated:

- `fixed-seed`: every draft token attends to `A_0`.  This is the lowest-cost
  and cleanest two-stream baseline.
- `continuous-arrival-mask`: every completed exact page is inserted into the
  draft page table before the next token step.  There is no stage barrier,
  replay, or intermediate verifier.  Earlier generated-token K/V remains an
  approximate tail until the single final verifier.  This is the intended
  progressive system method.
- `coarse progressive-graft`: later draft tokens attend newly arrived tranches while
  retaining earlier draft-token K/V as approximate state.  The final verifier
  refreshes every generated representation.  This Hugging Face implementation
  is a functional acceptance probe and negative control; it physically
  concatenates legacy caches and is not the intended systems implementation.

A replay-based four-stage control has already been measured: it improves the
accepted prefix but is slower than the two-stage fixed-seed policy.  Therefore
"progressive" must not be implemented by replaying all earlier draft tokens at
every page boundary; zero-replay, metadata-only page-table expansion is a
method requirement.

The Hugging Face mechanism runner now implements the continuous policy.  It
pre-schedules cumulative wire deadlines, launches H2D asynchronously for every
wire-complete tranche, polls copy completion before individual proposal steps,
and records the visible stage for every draft token.  Its paired fixed-S1
control uses the same arrival polling and stops drafting when full KV is ready,
so neither arm receives an artificial proposal-time advantage.  Because the
legacy cache backend still concatenates prompt and generated-tail tensors when
the view expands, this path is valid for acceptance and end-to-end mechanism
tests but is not the final metadata-only systems implementation.

The local vLLM systems track now provides a `PROGRESSIVE_KV` draft attention
backend.  It retains exact K/V at their original physical block IDs, compacts
the draft block table to a monotonically expanding visible-page set, and calls
the unchanged FlashAttention kernel.  Non-document prompt blocks and the
generated approximate tail are always retained.  Full prompt passes and the
immutable target verifier use ordinary full attention.  The backend supports
both a proposal-indexed schedule for kernel isolation and a connector-owned,
atomically replaced completion snapshot.  Completion is read once per draft
proposal and shared by all layers, so a page arriving midway through a forward
cannot create a layer-dependent mask.  The probe can replay a wall-clock event
trace.  The LMCache multiprocess connector now has an opt-in progressive
retrieve path: it converts one retrieve into priority bundles, submits only one
bundle at a time, and publishes the exact completed token intervals after all
futures in that bundle resolve. Fraction-only reconstruction is deliberately
not used: LMCache transfer chunks and sparse-attention pages may have different
sizes, so it could expose a page that had not arrived. A controlled one-host
mode additionally withholds each completion until the serialized wire deadline
for a declared link rate, while still executing the real retrieve.

The scheduler now implements the decode-side state machine
`WAIT_ANCHOR -> DRAFT_WHILE_LOADING -> READY_TO_VERIFY -> TARGET_DECODING`.
An opt-in request leaves ordinary remote-KV waiting when S1 is complete, adds
the exact P-side seed internally, and runs single-token target-model forwards
through `PROGRESSIVE_KV`. Those forwards share the target weights and the same
physical paged prompt blocks; only their compact block table is sparse. Draft
tokens are accumulated internally and never returned. If the proposal cap is
reached first, the request pauses without consuming GPU until full completion.
At full arrival the scheduler keeps only the exact prompt prefix, rewinds the
generated tail, injects the tentative tokens into vLLM's ordinary speculative
verification path, and exposes the seed plus verified batch. A no-op custom
proposer enables the standard rejection sampler without loading a second
model. Target verification and every later decode step see full attention.
The state-machine unit gate covers S1 start, hidden drafting, pause, full-KV
rewind, one verify, and first visible output.

A decoder-only diagnostic trace closes the gap between scheduler intent and
worker execution. For every progressive request it records the small forwards
after GPU-model input preparation and joins them to the completed scheduler
record by exact request ID and monotonic timestamps. A 64K, 25-Gbps `n=2`
mechanism run passed all 4 method chains: 32 single-token sparse inputs were
exactly `[seed, draft_0, ..., draft_6]` at positions 65536--65543, and all four
full-verifier inputs were exactly `[seed, draft_0, ..., draft_7]` rewound to
position 65536. Thus the repeated-token patterns observed with uniform 5%
visibility are model behavior under a weak sparse context, not replay of the
producer seed. Because this trace performs synchronous JSONL writes, it is a
correctness artifact and its latency is excluded from performance tables.

Scheduling comparisons use an even stronger pairing rule. A source request is
prefilled exactly once at P, and sequential, uniform, random, BM25, and oracle
page orders--each under fixed-S1 and continuous visibility--all reuse that same
exact prompt KV and producer seed at D. The top-level validity gate requires
one non-reused record across the ten conditions, one shared `prefill_group`,
and one shared seed. A 64K `n=2` diagnostic passed this gate for all 20
conditions and also passed 160 sparse single-token forwards plus 20 final
verifier batches in the decoder-input audit. An earlier `n=30` screen that
performed a separate producer prefill per schedule is invalidated rather than
interpreted: three requests changed seed before D-side scheduling began.

The corrected 64K/25-Gbps `n=30` screen passes every runtime, pairing, link,
priority, and exact-output gate, but falsifies the initial 5%-tranche,
`gamma=8` progressive contribution. Across 150 request-schedule pairs,
continuous visibility changes the accepted prefix zero times and every paired
latency confidence interval crosses zero. The measured timing explains why:
one 5% residual tranche takes about 137.3 ms on the modeled link, while eight
sparse proposal steps span about 172.4 ms, so only the final proposal positions
can see even one expansion. A 1%-tranche, `gamma=16`, 64K `n=2` mechanism gate
then makes 6--8 distinct visibility levels observable in every continuous
draft and passes all 320 sparse-input plus 20 verifier-input checks. This
repairs the overlap geometry but is not yet quality evidence.

The matching 1%-tranche, `gamma=16`, `n=10` screen makes the negative result
stronger. Every continuous request observes 6--8 distinct page-visibility
levels, and progressive arrival changes 8 of 50 draft token sequences, so the
mask is genuinely live rather than accidentally fixed. Nevertheless useful
accepted tokens change in 0 of 50 fixed-versus-continuous pairs. Splitting the
same 2.749-second modeled wire into 100 bundles also adds roughly 0.23--1.33
seconds of observed chain overhead, depending on schedule. The current main
system should therefore use a two-level fixed anchor; fine continuous arrival
is retained as a negative ablation rather than a claimed contribution.

Fixed-horizon timing must also be separated from natural completion. Llama
3.1 uses stop IDs 128001, 128008, and 128009; `ignore_eos=true` can repeatedly
accept stop tokens after the response has already ended. Aggregation therefore
reports both raw acceptance and an effective metric truncated at the first
model stop token. In the valid `n=30` screen, sequential raw acceptance is
45.0% but effective acceptance is 39.67%, with two accepted tokens per request
on average occurring after the first stop. Paper quality and production
latency conclusions use natural stopping; fixed 32-token results remain a
controlled stress surface only.

The external-orchestrator bridge is implemented as vLLM's
`ProgressiveExternalDraftProposer`.  The P side supplies one exact seed token;
the sparse draft worker atomically publishes
`{state, epoch, seed_token_id, draft_token_ids, created_at_ns}` while the target
request is still waiting for full KV.  Once full KV arrives, the target computes
the seed itself, checks it against the snapshot, and hands the suffix to the
ordinary vLLM rejection/verification path.  The handoff never waits: if the
snapshot is absent or not ready at that exact boundary it returns no draft and
the request follows the baseline target path.  A seed mismatch is a hard error,
not a lossy fallback. It is retained as a mechanism control; it duplicates
draft model state and is not the main deployment path now that the
shared-paged-KV scheduler exists.

An exclusive-H200 mechanism gate on one 8K QMSum request now exercises the
event-driven compact block table and the external immutable-verifier handoff.
Across seven proposal positions, exact visible document pages expanded as
`3/126 -> 32/126 -> 63/126 -> 95/126 -> 126/126`; all seven draft tokens were
accepted and the nine-token verified output exactly matched full-target greedy
decode.  This is a single-request correctness gate, not a quality or latency
claim.  The recorded artifact is
`outputs/progressive_kv/iclr_queue/kernel_llama_qmsum_8k_g8_progressive_gpu1_20260827_fix2/summary.json`.

## 4. Correctness boundary

For greedy decoding, the final verifier computes target logits for all draft
positions from the complete prompt KV.  It accepts the longest prefix whose
tokens equal the target argmax, emits the target argmax at the first mismatch,
and continues from the target cache.  In exact arithmetic this produces the
same sequence as ordinary greedy target decoding, independent of how draft
page masks changed.

For sampling, each proposal must retain its draft probability under the page
set used at that step.  Standard rejection sampling against the final target
probability then preserves the target distribution even if proposal
distributions differ across draft positions.

BF16 batched verification and token-at-a-time decoding can create different
accepted-token K/V, causing later top-1 differences even when the verified
positions have comfortable margins.  Experiments must report both the
algorithmic exact policy and raw token equality.  An `n=30` diagnostic found
that a low-margin fallback cannot separate matches from mismatches.  Bitwise
greedy equality therefore requires a deterministic verifier/decode kernel or
sequential rematerialization of the accepted prefix; the latter must be
charged to latency.  Task-quality equivalence alone is not evidence of
token-level equality, while mathematical speculative equivalence must be
phrased as a distributional guarantee rather than a bitwise one.

## 5. Latency model

Define:

- `T0`: priority seed transfer and installation;
- `Tr`: residual transfer and installation;
- `Cs(gamma)`: sparse drafting time for `gamma` tokens;
- `V(gamma)`: one full-KV verification;
- `Cf`: full-KV target decode time per token;
- `A(gamma)`: accepted draft prefix length.

The verifier also produces the first correction token when a proposal is
rejected.  To keep the latency model conservative and independent of that
one-token bonus, the equations below charge `gamma - A(gamma)` ordinary
target steps after verification.

For a response region containing at least `gamma` target tokens:

```text
T_baseline = T0 + Tr + gamma * Cf
T_sparse   = T0 + max(Tr, Cs(gamma)) + V(gamma)
             + (gamma - A(gamma)) * Cf
```

This gives the conservative predicted saving:

```text
predicted_gain(gamma) = A(gamma) * Cf
                        - V(gamma)
                        - max(0, Cs(gamma) - Tr)
```

`Cs(gamma) <= Tr` and `V(gamma) < A(gamma) * Cf` are useful screening
conditions, but the first is not necessary: a small amount of exposed draft
work can still be profitable when acceptance is high.  The online
enable/disable decision must therefore use the joint score:

```text
predicted_gain(gamma) = A_hat(gamma) * Cf
                        - V(gamma)
                        - max(0, Cs(gamma) - Tr)
enable iff max_gamma predicted_gain(gamma) > safety_margin
```

Sizing `gamma` from `Cs <= Tr` alone is insufficient: it can eliminate exposed
drafting while leaving too few accepted tokens to amortize the fixed verifier.

If drafting exceeds the residual window, only its excess is exposed.  Sparse
attention reduces long-context KV reads but not model-weight, projection, or
MLP work; consequently the first gate depends on context length, batch size,
model architecture, and link bandwidth.  Strict TTFT need not improve because
the first committed token waits for final verification.  TT8/TT16/TT32,
time-to-first-committed-block, stage-transition stall, and request completion
are the primary latency metrics.

The live runtime evaluates this equation per request rather than fitting it to
aggregate latency. The baseline target-step cost comes from positive
successive client token-arrival intervals; sparse draft span comes from every
scheduler-recorded draft completion; residual time comes from the serialized
controlled-link trace after the first bundle; and verification is timed by the
immutable-verifier transition. The paper aggregator reports predicted gain,
observed paired gain, absolute error, and gain-sign agreement. Missing timing
coverage invalidates the cell. This decomposition is intentionally
conservative: the exact producer seed is not counted as an accepted-token
benefit.

Latency-valid live attention telemetry is emitted only by transformer layer zero.
All layers consume the same proposal-boundary visibility snapshot and execute
the same compact-block-table path; repeating long page/range lists for every
layer caused measurement-path file I/O to scale with `layers x proposals`.
All-layer agreement is therefore a mechanism/unit-test invariant, while the
representative live record contains the exact visible ranges, page counts, and
a SHA-256 identity of the selected logical block set. The compute-only
crossover uses a stricter protocol: timed requests disable page hashing and
trace file I/O completely, then one post-timing prefix-cache-hit audit request
per visibility fraction proves physical compaction. Any timed request found in
that trace invalidates the cell.

At 128K, compact block-table reuse is effective for seven of the eight draft
steps and removes repeated 2K-page table construction. In the valid
trace-isolated `n=30` cell, however, a 5% exact-page draft still takes 168.92 ms
versus 161.05 ms for 100% visibility (`0.953x`, paired saved-time 95% CI
`[-16.25, 0.43]` ms), despite addressing only 5.04% of the logical BF16 KV.
Nsight shows why: the sparse attention kernel itself saves about 32 ms across
eight tokens, but the unchanged full 8B projections/MLPs remain and the generic
compact FA3 path performs per-layer dynamic scheduling. Preparing one compact
FA3 schedule ahead of the layers removes 248 GPU operations, but its in-forward
synchronization costs roughly 40 ms once per request and is slower; that
implementation was rejected and reverted. Page sparsity alone is therefore not
a genuinely cheap draft path on this H200 setup. The pre-registered pivot is a
two-dimensional anchor that sparsifies both prompt pages and draft layers (or
another independently cheap proposer), while retaining the same one-shot full
verifier.

The live validity gate additionally checks that the first connector bundle
completely covers both protected prompt boundaries for every method request.
Keeping an unarrived query block in the block table is invalid even if later
output happens to match; such a cell is rejected before latency aggregation.

Semantic transfer schedules are request scoped. The benchmark removes its
`sparsecache_priority_chunks` field before sending the prompt to the OpenAI
prefill endpoint, then carries the complete chunk permutation only in the
decoder's KV-transfer metadata. The connector re-applies the protected anchor,
serializes the resulting bundles, and records their exact token intervals.
The paper aggregator reverses coalesced intervals back into chunk sets and
requires every observed bundle to equal the corresponding sidecar slice. Thus
merely generating a BM25 schedule without using it on the wire cannot pass the
runtime gate.

## 6. Joint page scheduling objective

The core research problem is not merely selecting high-attention pages.  The
schedule should maximize verified work produced before a network deadline:

```text
maximize_{pi, A0, gamma}
    E[accepted_target_tokens(pi, A0, gamma)]
    - lambda_1 * exposed_draft_time
    - lambda_2 * priority_bytes
```

subject to page-availability, HBM-capacity, and transfer-credit constraints.
The practical score is expected acceptance gain per transferred byte.  Inputs
may include BM25/retrieval score, prompt position, P-side last-query attention
summaries, prior-request verifier feedback, layer/head identity, and current
network backlog.  Oracle support/attention schedules are analysis upper bounds
and cannot be used in the online method.

The frozen 64K scheduling experiment compares sequential, uniform, stable
random, per-request BM25, and oracle-support order. It interleaves all five
schedules with fixed-S1 and continuous masks inside each request and model
deployment. Selection uses source rows 0--99; confirmation uses disjoint rows
100--199 after the policy is frozen. BM25 is the primary realizable policy;
oracle is labeled only as an analysis upper bound.

## 7. Distinction from adjacent work

- Lynx progressively transfers high- and low-significance bits and drafts over
  every token at lower precision.  SparseCache-PD progressively reveals exact
  token/page K/V and drafts over a sparse working set.
- MagicDec and TriForce establish sparse-KV self-drafting when the working set
  is already local.  SparseCache-PD makes exact-page availability and transfer
  order part of the draft/verification scheduler.
- SmartGen selectively transfers and remotely fetches pages for sparse target
  attention.  Its foreground decode is approximate sparse attention rather
  than a tentative proposal checked once by exact full attention.
- OasisKV uses lookahead tokens to prefetch a sparse working set for ongoing
  decode.  SparseCache-PD uses progressively arrived prompt pages to create an
  exact-verifiable transition batch.
- Dustin sparsifies the target verification path, while SpecPV periodically
  applies full verification after partial checks.  They are useful lossy
  latency/quality controls; SparseCache-PD keeps one immutable full-KV verifier
  and never commits its sparse-path tokens directly.

The publishable contribution must therefore be the co-design of arrival-aware
sparse self-speculation and page scheduling, not the individual use of sparse
attention, prioritized transfer, or speculative verification.

## 8. Testable hypotheses

- H1: At 25--100 Gbps and >=32K contexts, priority seed transfer exposes enough
  residual time to hide sparse drafting.
- H2: Query-aware page ordering increases accepted prefix length at a fixed
  priority-byte budget relative to sequential, uniform, and random order.
- H3: One final verifier removes task-quality degradation seen under finite-W
  progressive commitment.
- H4: Continuous page arrival improves the accepted prefix for long draft
  windows (16--32 tokens at 25--50 Gbps) without adding replay or verification
  compute.  Fixed-seed drafting is the fallback when the predicted window is
  too short for later arrivals to affect proposals.
- H5: Page sizes <=256 tokens recover most scheduling quality while keeping
  RDMA/launch metadata overhead bounded.
- H6: Benefits grow with context length and request batch size, and shrink with
  link bandwidth; the measured break-even surface matches the latency model.

## 9. Current evidence and method decision

The repository establishes real pinned-memory H2D overlap, serial 25--200 Gbps
wire pacing, physically compact sparse draft caches, and one immutable final
verifier.  No token is externally committed before the complete BF16 KV is
installed.  At 100 Gbps, a six-proposal fixed-seed policy with a configured 2%
BM25 priority set gives the following exploratory screening results.  Page and
anchor rounding make the actual first transfer roughly 3.7%--5.0%.

| task | n | mean prompt | accepted / 6 | saved vs full transfer | paired 95% CI |
|---|---:|---:|---:|---:|---:|
| QMSum | 10 | 13.0K | 5.1 | 52.3 ms | `[34.0, 70.5]` |
| 2WikiMultiHopQA | 30 | 7.9K | 4.57 | 23.8 ms | `[9.5, 37.1]` |
| HotpotQA | 10 | 13.9K | 5.0 | 43.5 ms | `[19.8, 63.6]` |
| MuSiQue (LongBench) | 10 | 15.5K | 4.1 | 31.8 ms | `[4.9, 55.4]` |

These are single-H200, paced-wire, natural-EOS results rather than real
two-node or paper-table claims.  They motivate the two-level mechanism:
transfer a tiny exact priority set, self-draft with compact sparse attention
while the exact residual moves, and perform one full-KV verification.
They also used the runner's earlier generic chat prompt.  All confirmation
cells restart from the vendored public LongBench template; these exploratory
numbers cannot be copied into the official quality table.

On QMSum `n=30`, fixed `gamma=6` gives the measured bandwidth boundary below.

| paced bandwidth | saved | paired 95% CI | speedup | exposed draft |
|---:|---:|---:|---:|---:|
| 100 Gbps | 45.5 ms | `[28.1, 62.7]` | 1.040x | 4.8 ms |
| 200 Gbps | 21.0 ms | `[4.6, 36.7]` | 1.020x | 25.5 ms |
| 300 Gbps | -2.9 ms | `[-20.2, 13.9]` | 0.997x | 48.1 ms |
| 400 Gbps | -17.9 ms | `[-35.6, -2.2]` | 0.983x | 60.0 ms |

This screening sweep places a provisional break-even region near 250--300
Gbps for this model/workload.  A residual-window-only scheduler at 200 Gbps reduces mean
`gamma` from 6 to 3.3 and exposed work from 25.5 to 0.5 ms, but acceptance falls
from 4.77 to 2.87 and mean saving falls from 21.0 to 1.8 ms with CI crossing
zero.  The deployment policy should retain an acceptance-calibrated horizon
and bypass speculation when predicted net gain is negative.

A later 100-request QMSum audit found that natural EOS made 4/100 baseline and
pipeline pairs generate different token counts, invalidating their completion
latency comparison.  A fixed-64-token rerun removes that confound but was run
as three concurrent GPU processes: 15/100 method requests contain more than
100 ms of unattributed wall time, including one 1.69-second scheduling gap.
Its untrimmed mean saving is only 8.0 ms with a 95% CI of
`[-34.6, 39.8]`.  This contaminated result is retained as an audit artifact,
not evidence of a speedup.  The primary QMSum cell must be replaced by a
single-GPU run that warms both paths, alternates measurement order, omits the
interposed serial-chain control, and uses a fixed token horizon.  No outlier
trimming is permitted.

The same fixed-horizon artifact is still usable for acceptance analysis because
GPU scheduling changes latency, not deterministic proposal identities.  Final
accepted-prefix counts for the six fixed-S1 proposals are
`{0:4, 1:8, 2:11, 3:6, 4:6, 5:8, 6:57}`; the probability of surviving through
positions 1--6 is `96%, 88%, 77%, 71%, 65%, 57%`.  Thus a 16--32-token fixed-S1
draft is unlikely to remain useful without later information.  This is the
specific headroom tested by continuous arrival masks at 25--50 Gbps; it is not
evidence that those masks will succeed.

Coarse mechanical multi-stage expansion at 100 Gbps is not supported as the
default.  With six
total proposals, `2% -> 20% -> 50% -> 100%` grafting changes accepted length by
`+0.0` on HotpotQA and MuSiQue, `+0.1` on QMSum, and `+0.37` on 2Wiki (`4/30`
requests improve and none regress).  In-process alternating repeats confirm
the acceptance gain but put the latency advantage near zero with CIs crossing
zero.  On a later disjoint shard only 1/30 improves and unconditional graft is
7.69 ms slower with CI excluding zero.  A Hugging Face graft must physically
concatenate legacy cache tensors; production paged metadata would remove that
copy but not create missing acceptance gain.

Verifying the tentative prefix at every intermediate arrival is also rejected:
on ten 2Wiki requests it raises final acceptance only from 5.0 to 5.2 proposals,
adds about 41 ms of verification, and changes a +23.6 ms saving into a
-17.3 ms slowdown.  Producer last-token attention and question-token attention
page rankings are rejected as well; on a disjoint ten-request QMSum validation
set, question-token attention accepts 4.9/6 versus BM25's 5.4/6 and never wins
per request.

Static BM25/request features and a frozen first-two-token margin trigger do not
generalize.  A two-token S2 stability probe identifies the sole improving
request on one held-out shard but has an uncharged probe cost and a latency CI
crossing zero.  Triggered multi-stage execution remains an open ablation rather
than the current method.

The robust fallback is two-level exact priority-page sparse self-speculation,
pending the clean fixed-horizon latency gate.  The paper candidate adds
continuous, metadata-only expansion of the sparse draft mask during long
transfer windows.  It must beat fixed S1 at 25--50 Gbps and 16--32 proposals
under matched acceptance, latency, and page-arrival traces.  Coarse graft and
per-arrival verification remain negative controls rather than substitutes for
that experiment.
