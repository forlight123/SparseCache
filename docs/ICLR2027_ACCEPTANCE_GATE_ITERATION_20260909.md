# SparseCache acceptance-gate iteration (2026-09-09)

> **Protocol correction (2026-09-10).** The pilot horizon table reused a
> `g=6` artifact sampled from a 200-row pool while `g=4/8/12` used a 150-row
> pool; 5/30 request IDs differ. The table remains a historical acceptance
> screen, but its latency optimum is not confirmatory. The corrected
> identical-request held-out sweep selects the leaner `g=6` descriptively;
> `g=6` and `g=8` differ by +2.07 ms, 95% CI [-7.52, 11.64].

## Outcome

The short-draft objection is resolved, but the paper gate is not.  With a
realistic fixed horizon of eight speculative tokens, direct sparse Target
self-drafting reaches more than four useful tokens on unseen RULER-QA2
requests.  The current live LMCache bridge remains below four because its
uniform Anchor is not relevance-ranked, the unoptimized Hugging Face drafter
sometimes misses the D handoff, and the custom-proposer callback loses the
first proposal before verification.

No number in this report counts an already-authoritative token as useful work:

```text
A = full speculative prefix accepted by the final Target
U = max(A - 1, 0) = suffix usable by the current late-callback vLLM bridge
```

All online values use every attempted request as the denominator.  A draft
that finishes too late to reach D contributes zero.

## Protocol

- Target: Qwen3-8B, BF16, greedy decoding.
- Offline model gate: RULER-QA2 16K, model-native chat template, query-ranked
  pages, 100-Gbps serial link pacing plus real pinned-CPU-to-HBM copies, one
  immutable full-KV verifier, fixed 32-token output horizon.
- Pilot: 30 frozen requests from sampled offsets 0--29.
- Held-out validation: 30 disjoint requests from offsets 30--59.  The Anchor
  policy selected on the pilot was frozen before validation.
- Live gate: real LMCache 0.5.4rc5.dev3, vLLM 0.23.0, NIXL 1.4.1, GPU 0 as P,
  GPU 1 as D plus the temporary second HF Target copy, 7,800-token cap and 16
  output tokens.  P sends whole chunks in Anchor then Residual phases; vLLM
  verifies every injected token against complete transferred KV.

The three-H200 fraction scans initially oversubscribed CPU worker pools.  Only
the first two pilot rows were affected; they were rerun with disjoint CPU
affinity and 32 threads/process before aggregation.  The contaminated latency
rows are not used below.

## 1. A realistic draft horizon works offline

At 10% query-ranked KV on the first 30 requests:

| horizon `g` | mean `A` | mean useful `U` | full block | latency saved vs full transfer |
|---:|---:|---:|---:|---:|
| 4 | 3.400 | 2.533 | 83.3% | 47.87 ms [31.85, 63.21] |
| 6 | 4.800 | 3.833 | 56.7% | 81.98 ms [59.38, 102.06] |
| **8** | **5.833** | **4.933** | 53.3% | **93.31 ms [62.45, 122.64]** |
| 12 | 8.267 | 7.367 | 56.7% | 47.57 ms [-0.47, 93.58] |

`g=8` is the operating point: it advances about 5.8 Target tokens per final
verification pass and retains a statistically positive latency result.  `g=12`
accepts more tokens but its serial draft cost removes the significant gain.

## 2. More Anchor KV is not automatically better

With `g=8`, the paired 30-request pilot gives:

| query-ranked Anchor | mean `A` | mean `U` | sparse draft | zero-acceptance |
|---:|---:|---:|---:|---:|
| 10% | 5.833 | 4.933 | 218.42 ms | 10.0% |
| 15% | 5.933 | 5.067 | 233.61 ms | 13.3% |
| 20% | 5.900 | 5.000 | 239.51 ms | 10.0% |
| 30% | 6.500 | 5.500 | 269.17 ms | 0.0% |

The paired `U(30%)-U(10%)` delta is only +0.567, bootstrap 95% CI
[-0.400, +1.567], while draft time increases by 50.75 ms [43.22, 57.99].
Fixed 30% is therefore not justified by this pilot.

The disjoint validation is stronger evidence that 15% is enough in the
query-ranked setting:

| query-ranked Anchor | mean `A` | mean `U` | zero/full blocks | exposed output equality |
|---:|---:|---:|---:|---:|
| 10% | 5.767 | 4.800 | 1 / 15 | 30/30 |
| 15% | 6.067 | 5.100 | 1 / 17 | 30/30 |
| 30% | 6.067 | 5.100 | 1 / 17 | 30/30 |

The acceptance intervals still overlap; this is an operating-point screen,
not a claim that 15% statistically dominates 10%.

## 3. Three tempting adaptations are rejected

### Root top-k branches

The sparse Target can expand the first-token top-k and generate all continuations
as a batch.  When D supplies its authoritative first token, the matching branch
is selected and the suffix still goes through the full-KV verifier.  The
implementation is lossless under the same greedy verifier contract.

On the same eight QMSum and eight MultiFieldQA requests, strict all-request
useful progress is identical for K=1, K=2, and K=4: 51/16 = **3.1875** tokens.
K=2 and K=4 increase root coverage but lose a handoff to extra compute.  Hot
draft time rises from about 315 ms (K=1) to 380 ms (K=2) and 364 ms (K=4).
Root branching is retained only as an ablation.

### Margin-triggered 10% to 30% escalation

On the pilot, escalating when the first 10%-view logit margin is at most 0.5
looked attractive: only 4/30 requests escalated, the nominal mean Anchor was
12.7%, and `U` increased from 4.933 to 5.467.  The frozen rule failed on the
disjoint validation: it escalated 1/30, that request already accepted all eight
tokens, and it caught none of the three requests that benefited from 30%.
First-token confidence is not a usable information-arrival trigger.

### Fixed 15% runtime Anchor

The live system uses whole 256-token `protected_uniform` chunks rather than the
offline query ranking.  Raising its nominal fraction from 10% to 15% changes
the selected uniform set and makes the draft slower.  On QMSum, useful progress
falls from 2.750 to 0.500 token/request; MultiFieldQA remains 3.625.  Across all
16 requests, only 11 drafts reach handoff and strict useful progress is 2.0625.

A new `nested_protected_uniform` mode fixes the set-replacement bug: for 31
chunks, 10% is `{0,10,20,30}`, 15% appends chunk 15, and every larger set is a
strict superset.  This recovers QMSum to 1.875 useful tokens/request, but costs
392 ms/draft and remains below the original 10% result.  Nestedness is a
necessary scheduler invariant, not a sufficient relevance policy.

## 4. Correctness status

The real LMCache runs preserve the same 16 output hashes under K=1, K=2, K=4,
fixed 15%, and nested 15%; every injected suffix is accepted or corrected by
vLLM's complete-KV Target before exposure.

The offline block verifier is mathematically the same greedy Target but can
differ from sequential BF16 decoding near ties because the attention reduction
shape changes.  The pilot has 30/30 exposed equality at 10% and 15%, 29/30 at
20%, and 28/30 at 30%; the disjoint validation has 30/30 for all three tested
fractions.  Every exposed mismatch has verifier minimum top-1 margin at most
0.25.  A paper claim should be phrased as standard speculative-decoding
losslessness unless the implementation adds a shape-invariant verifier or a
canonical single-token replay guard.  A heuristic margin threshold alone is
not a proof.

## 5. Main algorithm after this gate

The next mainline is no longer a fraction search:

1. **Nested relevance priority.** P computes one chunk ordering `pi(q)`; every
   transmission stage is a prefix of that same order.  Arrival can only add KV,
   never replace an Anchor.  Uniform nested order remains a systems control.
2. **Cheap direct-KV execution.** Share Target weights with vLLM and implement
   paged sparse attention over arrived chunks.  The immediate target is a hot
   eight-token draft below 250--280 ms, with no second 8B checkpoint.
3. **Pre-verification rendezvous.** The D scheduler allocates speculative slots
   after full KV readiness but before its first decode forward, so `q1..q8` are
   verified together.  The current post-sampling callback wastes `q1` and is
   not the final architecture.
4. **Immutable final verifier.** No draft token is externally committed before
   complete-KV verification.  Multiple arrival stages may refresh proposals,
   but they never replace the final verifier.
5. **Numerical guard.** Use a shape-invariant verification kernel, or replay
   uncertified low-margin positions with the canonical single-token execution
   shape before commitment.

The central publishable hypothesis is now precise: a relevance-ordered nested
KV prefix can produce a 4+ token block before full KV readiness, and a cheap
direct-KV drafter plus pre-verification rendezvous can turn that block into
latency hidden behind transfer while retaining the immutable full-KV Target
distribution.  Acceptance is established offline; the optimized online
latency and task-diverse relevance scheduler are still open gates.

## Artifacts

Raw JSONL and summaries are under the ignored directory
`outputs/progressive_kv/iclr2027_20260909/`:

- `direct_sparse_target_qwen_ruler16k_g{4,6,8,12}_n30*`
- `direct_sparse_target_qwen_ruler16k_g8_f{15,20,30}_n30*`
- `direct_sparse_target_qwen_ruler16k_g8_f{10,15,30}_val_n30*`
- `lmcache_target_rootk{2,4}_*`
- `lmcache_target_f15_rootk1_*`
- `lmcache_target_nested_f15_rootk1_*`
