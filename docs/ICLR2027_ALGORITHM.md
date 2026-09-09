# SparseCache lossless algorithm contract

Date: 2026-09-09. This document specifies the intended algorithm, proof
obligations, scheduler family, and the boundary between mathematical losslessness
and floating-point reproducibility. It is a research design, not a claim that the
current prototype already satisfies every systems condition.

## 1. Endpoint and state

The authoritative target is the original model with complete contextual prompt
KV, denoted by conditional distribution `p`. Sparse or progressively arriving KV
is used only by a proposer `q_s`, where `s` is an arrival epoch. Draft hidden
states and draft-generated KV never enter authoritative target state.

Each candidate has an immutable epoch record:

```
E = (prompt_id, sparse_KV_digest, original_positions, stage, candidate_tokens,
     proposal_probabilities, decoding_policy, RNG_counter)
```

A later KV stage may replace a candidate only before target verification starts.
After verifier layer 0 begins, the epoch is sealed. Late sparse views are saved
for a later block. This avoids the ill-defined case where different target layers
verify different proposals.

“Progressive” therefore means that proposal opportunities improve as KV arrives;
it does not mean that partial-KV target decisions are externally committed. The
main algorithm has no finite-W commitment and no lossy verifier.

## 2. Pipeline

For a fixed visibility stage `s` and proposal length `g`:

1. P computes and publishes the exact first Target token (the draft seed), then
   sends the scheduled anchor pages, with their original positions, for the
   Target-KV layers consumed by the drafter.
2. D samples or greedily constructs a candidate block from `q_s`. This work is
   internal and may overlap residual KV transfer.
3. P continues sending every byte required by the original full target. A target
   layer becomes runnable only after that layer's exact prompt K and V are ready.
4. D verifies the sealed candidate layer by layer. Layer `l+1` consumes only the
   exact hidden output of completed layer `l`; draft states are never substituted.
5. Only the final target decision is committed. Rejected branch KV is discarded;
   accepted/correction KV is retained according to standard speculative decoding.

For layer-ready time `R_l`, proposal-ready time `D_s`, and measured block cost
`c_l(g)`, verifier completion obeys

```
F_-1 = D_s
F_l  = max(F_{l-1}, R_l) + c_l(g),       l = 0..L-1.
```

This recurrence is both the online controller's prediction model and an audit
invariant: measured layer start must be no earlier than both predecessors.

## 3. Losslessness statements

### Greedy theorem

Let `T(x)` be the full-target greedy next-token function. For any proposal block
`y_1..y_g`, compare `y_i` with `T(x,y_<i)` in order. Emit the longest equal prefix
and then the first target token, or emit all proposals plus the target bonus token
if all match. The emitted sequence is exactly the prefix produced by repeated
applications of `T`, independent of how inaccurate `q_s` is.

The proof is induction on accepted positions. Every accepted proposal equals the
authoritative target token under an already proven target prefix. At the first
mismatch, the emitted correction is the authoritative next token. The complete
target KV requirement is essential; progressive sparse KV affects only runtime.

### Sampling theorem

At position `i`, record the actual conditional proposal probability
`q_i(y_i | y_<i, E)` used to sample the sealed candidate. With the full target
probability `p_i`, accept using `min(1, p_i(y_i)/q_i(y_i))`; on rejection draw
from the normalized positive residual `(p_i-q_i)_+`. The resulting next-token
law is `p_i`, so induction yields the original target sequence distribution.

The drafter need not share the target architecture and may use any sparse KV
stage. A parallel block proposer is valid if its actual factorization is declared
and those probabilities—not a surrogate confidence—are used. Candidate selection
based on sampled token values changes the proposal law and is forbidden unless
the resulting selection probability is included. A stage choice based only on
arrival/compute state is safe. Greedy experiments do not establish this sampling
implementation; sampling is a separate gate.

## 4. Joint transfer scheduling

Under a deliberately restricted model—one serialized preemptible link, one fixed
anchor set, no link/compute interference, and a target layer chain—an optimal
schedule exists in the following one-switch family:

```
complete target layers 0..k-1
send all still-unsatisfied draft anchors
send residual target layers k..L-1 in dependency order
```

Before the draft is ready, a non-prefix target layer unlocks neither the drafter
nor the verifier and can be exchanged with an anchor or a missing prefix byte.
After the draft is ready, an inversion in target-layer order cannot make an
earlier dependent layer ready and can be exchanged without increasing makespan.
Thus all `k=0..L` can be enumerated in `O(L)` for each `(s,g)` under these
assumptions. Contention, multiple requests, packet granularity, or multiple links
invalidate the reduction and require measured scheduling rather than a stronger
claim.

For each request, the controller estimates

```
gain(s,g,k) = time_layer_ready_decode(E[min(A_s,g)+1]) - F_{L-1}(s,g,k),
```

where `A_s` is the accepted-prefix random variable calibrated on held-out live
verification. Prefix survival, not the product of unrelated stage agreement
rates, determines `E[min(A_s,g)]`. The controller may launch a candidate only if
its predicted gain is positive and its first-commit delay is within policy.

`D_s` and `c_l` must be conditioned on link regime and concurrent copy/scatter
work. The H200 pilot measured a 10% draft at about 10.4 ms in the useful 100-Gbps
cell but 23.7 ms in the 400-Gbps cell; treating compute and transport as
independent would select the wrong action. A conservative admission rule is

```
launch iff lower_CI(equal_progress_gain) > 0
       and upper_CI(first_commit_delay) <= TTFT_budget.
```

With the current kernel and a near-zero TTFT budget, 25/100 Gbps pass while
400 Gbps does not. These are empirical controller parameters, not universal
bandwidth thresholds.

Multiple sparse stages form an optimal-stopping problem. A practical first policy
evaluates the finite set of measured `(s,g,k)` choices whenever a stage seals,
replacing an unverified candidate only when predicted final completion improves.
This is content-independent at runtime and preserves the simple sampling proof.

## 5. Two different meanings of “exact”

Distributional exactness is the usual speculative-decoding guarantee in real
arithmetic: the final full target, not the drafter, defines the output law.
Bitwise endpoint equivalence additionally requires that batched verification and
the declared sequential target execute shape-invariant floating-point reductions.
The latter does not follow from the former.

The current H200 probe found three concrete numerical effects:

* repeated BF16 SDPA greedy replay changed 5/64 suffixes despite identical seeds;
* concurrent copies changed one-token SDPA logits/KV, although tested argmaxes
  remained equal in the first diagnostic;
* eager batched versus eager sequential verification changed the externally
  committed greedy block on 1/64 requests at `g=3` and `g=7`, and 2/64 at `g=15`.

Therefore “eager” or “deterministic algorithms” alone is not a bitwise proof. A
paper may make the standard distributional-lossless claim with this limitation
reported explicitly. A stronger bitwise claim requires a shape-invariant
verifier kernel: per-row normalization, projection, attention and MLP reductions
must use the same tiling/order as the declared one-token endpoint, regardless of
block length. A serial decision replay is a correctness oracle but removes the
parallel-verification benefit; margin thresholds without a sound rounding bound
are not a proof.

The initial rowwise oracle implements the same idea without fusion: it executes
the model layer-major but invokes each token row with `qlen=1`. On 64 Qwen3-8B
requests and 16 verifier inputs it reproduces sequential logits and generated KV
bitwise with zero maximum error. It costs 448.50 ms versus 461.24 ms sequential
and 31.69 ms for ordinary batched copy+verify, so the next kernel question is
whether fixed-row reduction semantics can be parallelized without giving up the
batched path's order-of-magnitude advantage.

## 6. Current go/no-go conditions

The method proceeds only if all three gates pass:

1. **Model:** sparse-KV proposals beat zero/other-request KV and produce enough
   accepted progress per millisecond on held-out long prompts.
2. **Schedule:** an integrated paced-link run beats layer-ready no-draft at equal
   exact output progress, while reporting first-commit latency and every byte.
3. **Exactness:** the standard greedy/sampling proof is implemented correctly;
   any additional bitwise claim is conditioned on an audited shape-invariant
   kernel or an explicitly charged fallback.

The strict early-decision certificate remains a separate high-risk extension.
The tested missing-KV norm bound is too loose and is not part of the main method.

## 7. Direct-KV causal correction without EAGLE

The current proposal model first computes a parallel sparse-memory block

```
h_1..h_g = B(seed, MASK_1..g, K_Es, V_Es, absolute_positions, stage).
```

An optional causal residual head then restores dependence among proposal tokens:

```
r_i = GRU(embed(y_{i-1}), r_{i-1}),   y_0 := seed
q_i = softmax(W h_i + C(h_i,r_i)).
```

At inference `y_i` is sampled or chosen from `q_i`; under teacher forcing the
head sees `[seed,y_1,..,y_{g-1}]`. It receives no EAGLE feature and no live
Target hidden state. The long-context state is still only the arrived compact
Target KV and its original positions. This preserves the deployment boundary:
the P node may serialize the same anchors needed by the drafter, while the D node
does not run a partial Target prefill.

A preservation-aware objective treats the inherited direct-KV block as a base
proposal `q^0`. Where its top-1 equals the frozen teacher token, training
minimizes `KL(q^0 || q)`; where it is wrong, training minimizes teacher CE. This
prevents a small task adapter from spending most of its capacity perturbing
already-correct positions. The teacher trajectory and every sparse view are
immutable packets, and train/evaluation prompt hashes must be disjoint.

The full-vocabulary residual improves proposal accuracy but performs one
vocabulary projection per autoregressive proposal, which competes with KV copies
for HBM bandwidth. The latency-oriented variant instead freezes the parallel
base top-K candidate set and predicts a hidden-space residual `delta_i`:

```
score_i(v) = (W h_i)_v + <delta_i, W_v>,   v in TopK(W h_i).
```

Only `K` rows of the existing Target LM head are gathered; there is no learned
vocabulary-sized correction matrix. This is exact as a proposal distribution
over its declared candidate set—Target verification still supplies output
losslessness—but its acceptance ceiling is the Target-token top-K coverage of
the base block. Coverage and accepted tokens per draft millisecond are therefore
joint model-selection metrics.

The first top-K screen validates this systems/model coupling. On 64 frozen QMSum
requests, top-16/64/256 rerankers accept 2.109/2.188/2.031 proposals at horizon
15, versus 1.531 for the inherited block. Top-64 at horizon 7 advances 3.141
verified output tokens and saves 62.348 ms to equal progress in a serial
100-Gbps one-host proxy, compared with 45.471 ms for the inherited block. Its
cross-task result is not yet sufficient: a QMSum-trained head gives no gain on
MultiNews or GovReport. Every evaluation must therefore report both
`target_in_base_topk_rate` and first-error top-K coverage. The former measures
the average candidate ceiling; the latter directly distinguishes an unreachable
first mismatch from a trainable ranking failure.

A second 3000-step screen trains top-64 on five long-summary packet sets. The
hidden-256 model improves g=7 accepted prefix by 0.781/1.875/3.547 tokens on
held-out QMSum/MultiNews/GovReport documents, with every paired 95% interval
strictly positive. Its first-error top-64 coverage is 88.7%/98.4%/100.0%.
In the clean 100-Gbps proxy it advances 3.312 verified output tokens and saves
68.787 ms at equal progress with a near-zero first-commit delta. This promotes
top-K reranking as the current model path, while leaving task-unseen data and
real two-node transport as mandatory external-validity gates.

## 8. Shape-invariant verifier localization

Module tracing on the two QMSum requests where ordinary eager block verification
changed the committed result localized the first drift to layer-0 attention
reduction. Making only attention row-invariant moved the first drift to the MLP
`down_proj`; after fixing those two, four remaining non-committing argmax drifts
first appeared at RMSNorm. The current minimal Python oracle therefore keeps
attention reductions, MLP down projections and RMSNorm calls in the endpoint's
`qlen=1` shape, while Q/K/V, output/gate/up projections and the LM head remain
block-parallel.

On 64 requests x 16 verifier inputs this hybrid is bitwise equal to sequential
logits and generated KV with maximum error zero. Mean verification is 219.14 ms,
versus 475.16 ms for fully rowwise sequential execution and about 31.69 ms for
ordinary eager batched copy+verify. Thus the semantic construction is validated,
but the Python loop is not a serving implementation. A fused row-program kernel
must preserve each row's reduction order while scheduling rows concurrently.

In the current one-host integrated proxy the unfused bitwise path remains slower
than baseline at 25/50/100 Gbps by 54.9/152.2/182.7 ms on 16 requests x 2 paired
runs, with zero committed-output mismatches. The performance gate for a bitwise
claim is therefore explicit: reclaim at least 183 ms at the 100-Gbps endpoint,
then rerun the same paired test. Standard speculative distributional losslessness
does not depend on this stronger engineering contract.

## 9. Real LMCache decoder data-plane gate

The one-host 1P1D deployment now carries the progressive phase boundary into
the decoder. P gathers four uniformly spaced token chunks first, waits for their
actual NIXL completion, and sends the exact seed, cache keys, original chunk
indices and completion timestamp to a D-side mailbox. D resolves those keys in
LMCache's registered CUDA arena; the Residual continues through the unchanged
full-KV path and remains the only phase allowed to emit FullReady.

For the current drafter layers `[1,9,17,25,33]`, D pins the four owning Anchor
objects and creates basic-slice tensor views. This is a representation contract,
not a copy: 1,220/1,220 views across 64 QMSum requests alias their original CUDA
storage. All 244 advertised objects and 9,210,691,584 resident bytes resolve.
The complete receiver control path costs 0.911 ms on average while the 7.8K
AnchorReady--FullReady window is 107.968 ms.

The integration now consumes `ClaimedLayerViews` in the trained direct-KV block
drafter on a private D-side CUDA stream while Residual movement continues. Exact
token ranges and the original prompt length accompany the objects, preventing
the live runtime from silently evaluating a different sparse position map than
the frozen training packet. `PYTHONHASHSEED=0`, P=`kv_producer`, and
D=`kv_consumer` remain mandatory; a missing hash seed can turn a supposed
KV-reuse test into decoder recomputation.

Strict evaluation submits immutable packet token IDs directly and preserves the
native prompt length. This supersedes the earlier text-round-trip observation:
observe, inject, and monolithic paths now agree on 64/64 greedy outputs. The
earlier five discrepancies came from token/text protocol and boundary changes,
not from progressive KV.

## 10. Target-conditioned suffix repair

The first online implementation admitted a block only if its first sparse-KV
proposal matched the first authoritative D-side Target token. That gate wastes
all overlapped long-context work on a first-token mismatch. The corrected
factorization computes the expensive state before FullReady:

```text
Z(A,s) = H_sparse(A,s)
       + TopK(W H_sparse)
       + GRUState(s),
```

where `A` is the arrived Anchor and `s` is the exact P seed. When the full Target
produces `t1`, the reranker advances the cached state with `t1` and produces
only `q2..qg`. Thus sparse attention and the parallel vocabulary projection
remain overlapped; only the small causal top-K repair follows `t1`.

For greedy decoding, let the full-KV verifier accept the longest suffix prefix
whose tokens equal its own transitions from the authoritative state after
`t1`, and let it supply the first mismatch or bonus token. By induction over the
accepted positions, every committed token equals sequential full-KV Target
decoding. The proposal may be arbitrary and the conditioning changes only its
acceptance rate. There is still no early commitment and no verifier drift.

This gives a useful separation:

```text
information arrival controls proposal readiness;
full-KV Target state controls commitment.
```

The current runtime contract is greedy only. Distributional losslessness under
sampling additionally requires the repaired proposal probabilities and an exact
sampling-aware acceptance/rejection rule.

On 64 native-length frozen QMSum packets, the nominal 10% protected whole-chunk
Anchor exposes 12.557% of tokens on average. Raw first-token agreement is
84.375%. Target-conditioned blocks are submitted for all 64 requests and the
full verifier accepts 2.438 tokens per request, including 1.438 useful suffix
tokens beyond `t1`. Hot sparse precomputation takes 24.860 ms and repair takes
3.035 ms on GPU.

A fresh-server observe--inject--observe sandwich keeps the complete online
draft/repair cost in both treatment and controls. Injection leaves TTFT
unchanged (-0.635 ms, 95% CI [-1.579,+0.353]) and reduces total latency by
24.407 ms (95% CI [19.513,29.671] ms saved), with 63/64 requests faster and
64/64 outputs equal to monolithic. The fixed system result and its limitations
are in `ICLR2027_ONLINE_LMCACHE_RESULT_20260909.md`.
