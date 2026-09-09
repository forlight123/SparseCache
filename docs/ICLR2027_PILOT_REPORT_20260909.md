# Lossless P/D initial exploration: measured results

Date: 2026-09-09. All jobs below have completed. These are development/mechanism
results, not a paper-ready end-to-end system or a claim of sampling exactness.

Protocol: [ICLR2027_LOSSLESS_EXPLORATION.md](ICLR2027_LOSSLESS_EXPLORATION.md).
Artifacts: outputs/progressive_kv/iclr2027_20260909/.

## What was run

Three idle H200 NVLs were used concurrently with one worker per physical GPU,
using SparseCache/.venv and explicit NUMA placement. No EAGLE-adapter training
was resumed. KVShot and the KV-input block model were imported into this repo
with source hashes. Weights and training data remain external local assets.

Main pilot: pilot_n64_s1000/. GPU 0 ran KVShot-PD and full-view block adaptation;
GPU 1 ran nested-view block adaptation; GPU 2 checked fixed-query bounds and
incremental attention. GPU 2 then ran the all-layer verifier probe separately.

Qwen3-8B, BF16, 64 distinct QMSum development requests, 15 proposals after one
known P seed. Mean prompt length 7844, range 2560..8192. **55/64 prompts are
head/tail shortened**, so these are not official QMSum quality results. Each
model evaluation contains 767 rows (2 orders x 5 fractions x 64, 64 zero-KV,
63 other-request-KV). Repeated cells are not independent requests.

## 1. Direct KV carries useful draft information

The imported five-layer KV-input block checkpoint, before new adaptation:

| Memory condition | Mean accepted draft prefix |
|---|---:|
| Priority, 5% requested | 1.921875 |
| Priority, 10% requested | 1.906250 |
| Priority, 20% requested | 1.953125 |
| Priority, 50% requested | 1.281250 |
| Full KV | 0.937500 |
| Zero KV, 10% shape | 0.000000 |
| Other-request KV, 10% shape, 63 donors | 0.015873 |

At 10%, exact-minus-zero paired request-bootstrap CI is [1.46875, 2.359375]
tokens (64 requests); exact-minus-shuffled is [1.50794, 2.39683] (63 paired
requests). These comparisons reuse the SAME target reference per request.
The model reads KV content. More KV does not monotonically improve this
short-context-trained checkpoint on long-context QMSum; that is a measured
problem for the proposed training curriculum, not evidence of monotonicity.

This model is a KV-input DFlash-backbone reference, not our newly trained
progressive algorithm. No target hidden-history/seed-hidden input is provided.
The historical EAGLE 0.42 value uses a different model/data/protocol and is not
a matched speedup or acceptance comparison.

The imported KVShot checkpoint obtains 0 accepted tokens at 10% and full KV in
the SAME seed-only P/D adaptation. This is not a reproduction of the paper's
metric: its original AR contract includes exact current-token KV, whereas the
P/D seed-only wrapper must synthesize seed KV in the draft. Its vocabulary,
selected layers and depth also differ. Reproduce the original contract and
train the P/D-aligned AR control before interpreting this as an architecture
comparison. The original reference code remains unchanged.

## 2. Small continued training does not yet demonstrate a progressive gain

Both arms start from the same checkpoint, use identical 1000 record/cut draws,
256 eligible parent-train records (246 actually sampled), and 15000 supervised
future positions. All 1,006,695,680 block/stage parameters are trainable. The
measured training loops take 155.43 s (full-only) and 142.79 s (nested), excluding
model load, evaluation and checkpoint I/O. This is a small adaptation pilot.

| After adaptation | 10% priority, raw mean prefix | Full KV, raw mean prefix |
|---|---:|---:|
| Full-only training | 1.609375 | 1.000000 |
| Nested-view training | 1.531250 | 0.609375 |

These raw values do not demonstrate an improvement. **A formal paired claim
is blocked:** independently repeated full-target greedy trajectories differ on
4/64 requests after training (IDs 19,44,52,61), despite identical prompts and
P seed tokens. One before-training reference also differs (ID 59). The reporter
retains every row and publishes no paired training CI or significance claim.
The cause of replay drift has not yet been localized; do not silently attribute
it to a particular kernel. The next comparison must share immutable P-prefill
state and target-reference packets and audit deterministic numerical behavior.

The actual mean visibility during nested training is 52.07%, because short
assistant cuts and protected first/last pages round nominal small budgets up.
This is not adequate exposure to real 5--10% visibility at 8--32K context.
Next data should provide real long prefixes and packed independent anchors,
not merely more optimizer updates on these 256 short conversations.

## 3. Simple strict missing-KV bounds are too loose

64 requests x 3 target layers x 5 visibility budgets = 960 fixed-query bound
cases. Both output-error-bound and missing-mass-bound violations are zero under
the declared FP64 tolerance. However, bounds on missing mass are approximately
1 at every incomplete visibility level.

At 10% visibility:

| Target layer | Actual mean missing attention mass | Upper bound | Error bound / full output norm |
|---|---:|---:|---:|
| 0 | 0.2828 | ~1 | 7.64 |
| 17 | 0.0676 | ~1 | 45.83 |
| 35 | 0.0478 | ~1 | 29.69 |

This center/radius construction does not support useful early token commitment.
It assumes exact queries computed at P and does not propagate uncertainty
through the full transformer. It is not an end-to-end certificate or a machine
rounding proof. Keep early certification as a separately gated research branch.

The single-layer Python incremental-attention microbenchmark is also negative:
async averages 3.06 ms slower than serial. Maximum merge error is 2.33e-6.
Tiny transfers, FP32 intermediate materialization and per-tile Python/kernel
overheads are included in this reference; it cannot justify serving speedups.

## 4. Positive result: overlap exact all-layer verification with transfer

Artifact: verifier_n64_gpu2/summary.json and results.json.

We then tested a complete 36-layer Qwen target verifier using real sparse-block
proposals. One P-prefill supplies the exact same cache to every condition within
a request. Pinned CPU KV transfers on a dedicated copy stream; each target
layer waits for its own completion event. Layers always use exact predecessor
hidden states. No target computations consume partial or draft-generated KV.

64 requests, three alternating paired repetitions = 192 paired runs. Mean
transferred payload is 1,156,644,864 bytes/request (about 1.16 GB decimal).

| Copy + verify path | Mean wall time |
|---|---:|
| Native HF after full copy, warmed diagnostic control | 49.352 ms |
| Same-work serial copy then manual target layers | 49.018 ms |
| Same-work layer-ready asynchronous pipeline | 31.182 ms |

Paired mean saving: **17.835 ms**, request-clustered 95% bootstrap CI
**[16.905,18.561] ms**. Ratio of mean serial/streamed times: **1.572x** for this
subgraph. 190/192 paired runs are faster.

Every serial/streamed comparison is BITWISE equal for logits and generated KV.
Both also match the native HF block forward bitwise on all tested requests.
The simple release-time recurrence underpredicts measured final-layer finish
by 0.739 ms on average; GPU event traces are retained for further analysis.

Scope: real local CPU-pinned H2D plus block verification only. P prefill, host
pinning, draft generation, deployment transport and later decode are outside
the timing region. It is not a 1.572x end-to-end speedup, a cross-node network
experiment, or proof that block and sequential BF16 generation agree forever.
The native control is warmed but not order-balanced; the primary statistic
uses the same-work, alternating serial/streamed pair.

## Decisions from this first round

1. Continue direct-KV block drafting: the KV-content ablation is strong.
2. Prioritize exact layer/page dependency scheduling: positive measured
   copy+verification overlap with preserved state is now available.
3. Do not scale this short-conversation adaptation unchanged. First freeze
   shared reference packets, supply long-prefix training, and pack anchors.
4. Do not claim early commitment from the tested norm bounds. They are valid
   at their limited scope but operationally uninformative.
5. Next integrated test must combine real sparse proposals, measured layer
   transfer readiness and exact verification, counting all components against
   both full-transfer and layer-wise/no-draft baselines.

## 5. Second round: frozen references and integrated pipeline

The second round generated 64 immutable P-side packets under
`reference_packets_n64/`. They contain exact prompt IDs, five selected Target-KV
layers, P-side page scores, one known seed and one canonical 15-token greedy
suffix. Every packet and the completed index have SHA-256 identities. The set is
10,285,622,720 bytes. Full-only and nested arms now read the same packet; a paired
comparison is rejected if any packet, seed or reference differs.

The corrected paired result confirms that the 1000-step short-context nested
adaptation is not an improvement:

| Cell | Full-only | Nested | Nested - full, paired 95% CI |
|---|---:|---:|---:|
| Priority 5% | 1.5000 | 1.5781 | +0.0781 [-0.0781, 0.2813] |
| Priority 10% | 1.6094 | 1.5312 | -0.0781 [-0.2344, 0.1094] |
| Priority 20% | 1.5156 | 1.5312 | +0.0156 [-0.0938, 0.1719] |
| Priority 50% | 1.2500 | 1.1094 | -0.1406 [-0.2969,-0.0156] |
| Full KV | 0.9688 | 0.6094 | -0.3594 [-0.5781,-0.1563] |

At priority 10%, both models still produce zero accepted tokens with zero KV and
about 0.0159 with other-request KV. The input-dependence conclusion survives;
the proposed curriculum does not. Longer real prefixes, matched effective stage
budgets and a frozen teacher corpus are required before scaling updates.

A fresh deterministic replay of the packet builder's full Target trajectory on
the same physical H200 changed 5/64 suffixes, while all 64 seed tokens remained
equal. First divergences occurred after equal prefixes at low BF16 logit margins.
`torch.use_deterministic_algorithms(True)` is therefore insufficient to define a
stable long-context greedy label. Frozen target packets are now mandatory.

### Exact verifier length and numerical audit

Using eager attention for the D verifier restored bitwise equality between
serial-transfer and layer-ready-transfer executions for every tested request at
all block lengths. Three repetitions per 64 requests gave:

| Proposal tokens | Verifier input | Serial copy+verify | Streamed | Saving |
|---:|---:|---:|---:|---:|
| 0 | 1 | 40.634 ms | 30.220 ms | 10.415 ms |
| 3 | 4 | 46.350 ms | 30.713 ms | 15.637 ms |
| 7 | 8 | 44.720 ms | 31.300 ms | 13.420 ms |
| 15 | 16 | 45.994 ms | 31.688 ms | 14.306 ms |

This is exact for the same eager block arithmetic and cache. It does not imply
that batched block arithmetic equals sequential one-token arithmetic bitwise.
With identical prompt KV and teacher-forced candidate tokens, eager batch versus
sequential execution changed some argmaxes on 2/64, 7/64 and 10/64 requests for
proposal lengths 3, 7 and 15. Under the actual greedy speculative commit rule,
external output changed on 1/64, 1/64 and 2/64 respectively. This is a numerical
shape effect, not a failure of the real-arithmetic speculative-decoding proof.
See `ICLR2027_ALGORITHM.md` for the two exactness contracts and the proposed
shape-invariant verifier requirement.

A layer-major rowwise oracle then forced every token row to use the endpoint's
`qlen=1` eager arithmetic while retaining the layer ordering needed by progressive
transfer. Across 64 requests x 16 verifier inputs, it matches sequential logits
and generated KV bitwise with maximum error exactly zero and no committed-output
mismatch. Its unfused Python implementation takes 448.50 ms versus 461.24 ms for
the sequential control, far above the 31.69-ms batched copy+verify path. This
establishes a correct numerical target for a future shape-invariant fused kernel;
it is not yet the serving path.

### Real integrated paced-link result

`integrated_scatter_100g_n64_r3/` runs the actual five-layer sparse drafter after
its 10% anchor arrives, continues a serialized paced transfer, scatters anchor
pages into their authoritative full-cache positions, and starts each of all 36
eager target layers on its CUDA readiness event. Every payload performs a real
pinned-CPU to H200 copy. This is a one-host 100 Gbps link proxy, not NIC/RDMA.
P prefill, host pinning and allocation are excluded equally.

Across 64 requests x 3 alternating paired runs:

* baseline and speculative transfer the same mean 1,156,644,864 bytes;
* the speculative path commits 2.625 tokens per first block on average;
* time to the same number of target tokens improves by **48.328 ms**, with
  request-bootstrap 95% CI **[36.706,61.241] ms**;
* first committed token is delayed by only **0.207 ms**, CI **[0.003,0.560] ms**;
* sparse anchor packing plus draft takes 10.430 ms and is almost fully hidden;
* batched versus sequential numerical semantics differ on the same two requests
  identified by the separate sequence audit (6/192 runs). Do not describe this
  implementation as bitwise lossless.

At 400 Gbps, the equal-progress saving remains 22.439 ms, but first commit is
delayed by 25.722 ms and measured draft rises to 23.654 ms under copy/scatter
contention. The simple no-contention recurrence is therefore inadequate at the
fast-link endpoint. A runtime admission rule must disable drafting when predicted
first-commit delay exceeds its SLO; 400 Gbps is a no-draft region for TTFT-sensitive
requests under the current kernel.

The completed byte-conserving cells are:

| Link proxy / view | Exact progress | Baseline same-progress | Pipeline | Saving | First-commit delta | Requests faster |
|---|---:|---:|---:|---:|---:|---:|
| 25 Gbps / 10% | 2.625 | 436.342 ms | 372.978 ms | 63.364 ms | +0.187 ms | 93.75% |
| 100 Gbps / 5% | 2.516 | 139.810 ms | 94.666 ms | 45.144 ms | +0.284 ms | 89.06% |
| 100 Gbps / 10% | 2.625 | 142.892 ms | 94.565 ms | 48.328 ms | +0.207 ms | 93.75% |
| 100 Gbps / 20% | 2.531 | 140.424 ms | 94.633 ms | 45.791 ms | +0.242 ms | 92.19% |
| 100 Gbps / 10% random | 2.469 | 138.264 ms | 94.704 ms | 43.560 ms | +0.339 ms | 87.50% |
| 400 Gbps / 10% | 2.625 | 84.193 ms | 61.754 ms | 22.439 ms | +25.722 ms | 76.56% |

Every row is paired within request and condition. Rows across bandwidths ran on
different physical GPUs/NUMA nodes and are not themselves paired, so the larger
25-Gbps absolute saving must not be interpreted as a monotonic link-rate effect.
The 10% stage dominates 5% at 100 Gbps for equal first-commit cost in this model.
It also beats 20%; more visible KV is not monotonically helpful for the current
short-context-trained checkpoint. Priority ordering raises mean accepted prefix
over random at 10% by 0.1563 token, but its paired 95% CI [-0.0313,0.3906]
includes zero. Thus the present P-query score is not yet a demonstrated algorithmic
contribution; it remains a training/scheduler target.
Shortening the candidate from 15 to 7 proposals at 100 Gbps yields 2.578 tokens,
46.449 ms same-progress saving (CI [35.556,58.323]) and +0.189 ms first-commit
delta. It removes one of the two numerical commit-mismatch requests, but gives
up about 1.88 ms mean saving. This is an implementation robustness tradeoff, not
a proof that length 7 is bitwise safe.

The measured outcome is conditional but positive: around 100 Gbps, sparse draft
work is nearly free in TTFT terms and returns multiple exact-arithmetic target
tokens at first commit. Model acceptance and bitwise shape invariance, not the
basic overlap mechanism, are now the two primary blockers.

Latest validation is 137 tests passed, 3 skipped; the current focused suite has
19 tests. It covers original-position preservation,
full-view equivalence,
unavailable-KV gradient isolation, future-position rejection, EOS, exact GQA
attention merge, fixed-query bounds, byte-conserving anchor schedules and the
layer recurrence. Real-GPU probes add the equality audits described above.

## 6. Third round: bitwise verifier and causal direct-KV head

### Minimal bitwise oracle

`numerical_localization.py` traced batched and sequential module outputs for the
requests whose committed token changed. Ordinary eager verification first
diverged at layer-0 attention output. A row-invariant attention implementation
moved the first divergence to `down_proj`; after row-invariant down projections,
four residual non-committing differences first appeared at RMSNorm. Keeping only
these three operator families in `qlen=1` arithmetic shape produced bitwise-equal
logits and generated KV on all 64 requests, maximum error 0.0.

| Verifier implementation, g=15 | Mean time | Logits/KV bitwise |
|---|---:|---:|
| ordinary eager block | about 31.69 ms | no |
| minimal row-invariant Python oracle | 219.14 ms | 64/64 |
| whole-layer rowwise oracle | 475.16 ms | 64/64 |

The minimal oracle is 2.17x faster than whole-layer rowwise execution, but its
Python per-row launches remain too expensive. In an integrated 16-request x 2
paired scan it transfers exactly the same bytes and has zero output mismatches,
yet loses 54.93, 152.19 and 182.72 ms at 25, 50 and 100 Gbps. This is a concrete
fused-kernel target, not a positive deployment result.

### Causal correction screen

The new correction head is not EAGLE. Its inputs are sparse Target KV, absolute
positions, the known seed and preceding proposal tokens. The five-layer block
backbone computes all proposal features once; a small causal GRU adds residual
proposal logits. Train/evaluation packet SHA sets are checked for overlap.

MultiNews/GovReport-only training reduced QMSum acceptance from 1.531 to 1.375,
despite lowering training loss. On the same training documents it raised
MultiNews acceptance from 1.000 to 3.063, proving capacity but also strong task
shift. This arm is rejected as a universal correction head.

Using QMSum requests 64..127 for training and frozen requests 0..63 for evaluation
gave 1.797 accepted tokens after 300 steps, a paired +0.266 improvement with 95%
bootstrap CI [0.094,0.438]. Expanding the disjoint training side to requests
64..191 and 1000 steps gave:

| Head/curriculum | Accepted @ priority 10% | Delta vs inherited base | Paired 95% CI |
|---|---:|---:|---:|
| 128 hidden, fixed 10% | 1.984 | +0.453 | [0.234,0.688] |
| 128 hidden, 5/10/20% | **2.016** | **+0.484** | **[0.281,0.688]** |
| 256 hidden, fixed 10% | 1.953 | +0.422 | [0.234,0.625] |

The best head improves 24/64 requests and hurts 3/64. At 10%, priority order
beats random by 0.281 token in the final matrix; the earlier 300-step checkpoint
already showed a paired priority-minus-random CI [0.063,0.484]. Zero KV remains
0 accepted and other-request KV remains 0.016, so the gain still depends on the
correct request KV. Visibility is nonmonotonic: the best model obtains
2.000/2.016/1.875/1.375/0.766 tokens at priority 5/10/20/50/100%.

The useful proposal horizon is seven: g=3 accepts 1.719, g=7 accepts 2.000, and
g=15 accepts 2.016. However the first full-vocabulary causal implementation
raises measured draft time and, under real copy contention, makes the 100-Gbps
integrated pipeline slower by 36.70 ms at g=7 and 39.87 ms at g=15. Accuracy
alone is not enough.

### Top-K reranker and clean serial timing

The hidden-space reranker gathers only selected Target LM-head rows and removes
the repeated full-vocabulary scans. On the same 128 disjoint QMSum development
requests and frozen 64-request evaluation set, 1000 steps give:

| Candidate set | Mean accepted, g=15 | Delta from inherited base | Paired 95% CI |
|---:|---:|---:|---:|
| top 16 | 2.109 | +0.578 | [0.375,0.813] |
| top 64 | **2.188** | **+0.656** | **[0.359,0.953]** |
| top 256 | 2.031 | +0.500 | [0.109,0.844] |

At the latency-oriented horizon `g=7`, top-64 advances 3.141 exact target
tokens per verified block at the 10% view, counting the target correction/bonus.
The inherited drafter advances 2.531. A first attempt to time three GPU cells in
parallel was invalid: all three arms, including the unchanged control, became
slower because the GPUs shared host-memory/PCIe resources. Those runs are retained
only as a contamination diagnostic and excluded from claims.

Clean runs use GPU 2 alone, NUMA node 1, 64 requests x 3 alternating pairs. The
top-64 result is:

| Link/view/order | Progress | Baseline same-progress | Pipeline | Saving (95% CI) | First-commit delta |
|---|---:|---:|---:|---:|---:|
| 25 Gbps / 10% / priority | 3.141 | 453.321 ms | 372.996 ms | 80.325 ms [66.659,93.983] | +0.123 ms |
| 100 Gbps / 5% / priority | 3.016 | 153.613 ms | 94.394 ms | 59.219 ms [47.898,71.023] | +0.041 ms |
| 100 Gbps / 10% / priority | **3.141** | 156.727 ms | 94.379 ms | **62.348 ms [50.209,74.642]** | +0.032 ms |
| 100 Gbps / 20% / priority | 2.953 | 151.651 ms | 94.322 ms | 57.328 ms [45.019,69.354] | -0.048 ms |
| 100 Gbps / 10% / random | 2.938 | 152.586 ms | 94.370 ms | 58.215 ms [46.904,70.704] | -0.020 ms |
| 400 Gbps / 10% / priority | 3.141 | 98.677 ms | 64.451 ms | 34.226 ms [21.897,46.922] | +28.591 ms |

The 100-Gbps 10% top-64 arm beats the clean inherited control's 45.471-ms
saving while preserving essentially zero first-commit delay. At 400 Gbps the
same-progress mean remains positive, but only 68.75% of requests are faster and
TTFT rises by 28.59 ms; a near-zero-TTFT controller must reject this cell. The
5/10/20% and priority/random paired contrasts favor 10% priority, but their
64-request confidence intervals still touch zero, so neither the visibility
optimum nor page scorer is yet a statistically established contribution.

A first cross-task check rejects the interpretation that the small QMSum-only
checkpoint is universal: it changes held-out MultiNews from 1.016 to 1.000
accepted token and leaves GovReport at 1.000. A subsequent controlled screen
trained three top-64 models for 3000 steps: eight tasks with hidden size 128,
five long-summary packet sets with hidden size 128, and the same five sets with
hidden size 256. All train/evaluation documents and immutable input hashes are
disjoint. These task rows are development-contaminated, so this measures
within-task held-out-document generalization, not unseen-task or final-benchmark
generalization.

The five-set hidden-256 arm wins the aggregate screen:

| Held-out requests, 10% priority, g=7 | Inherited base | Trained top-64 | Paired delta (95% CI) | Base top-64 coverage | First-error coverage |
|---|---:|---:|---:|---:|---:|
| QMSum 0..63 | 1.531 | 2.313 | +0.781 [0.453,1.125] | 83.0% | 88.7% |
| MultiNews 64..127 | 1.016 | 2.891 | +1.875 [1.672,2.078] | 71.2% | 98.4% |
| GovReport 64..127 | 1.000 | 4.547 | +3.547 [3.219,3.875] | 90.2% | 100.0% |

Zero-KV acceptance remains zero on all three sets. Other-request KV is 0.016,
0.048 and 0.000 respectively, versus 2.313/2.891/4.547 with the correct request
KV. The first-error diagnostic shows that most initial failures are reachable by
top-64 reranking; later-position coverage is a separate ceiling and is lowest on
MultiNews.

Finally, the winning checkpoint was rerun alone on GPU 2/NUMA node 1 in the
100-Gbps integrated proxy. It advances 3.312 verified output tokens per block,
with 163.099-ms no-draft time versus 94.312-ms pipeline time at equal progress:
**68.787 ms saving**, request-bootstrap 95% CI **[56.626,81.581] ms**. Mean
first-commit delta is -0.028 ms, and 98.44% of requests are faster. The 13.952-ms
draft remains hidden. Ordinary eager block verification still has one numerical
committed-output mismatch request (3/192 runs); the distributional-lossless
algorithm and bitwise-oracle results have not changed.

All LongBench-derived training in this round is mechanism-only and contaminates
those task rows. Final paper quality evaluation requires a separately sourced
training corpus and untouched task test splits.
