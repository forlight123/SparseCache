# EAGLE3 learned-draft selection protocol

Status: pre-registered selection screen. This experiment evaluates a learned,
cheap draft path before it is connected to the LMCache P/D transport. It is
not an end-to-end progressive-transfer latency claim.

## Question

Does an exact-match EAGLE3 checkpoint for the frozen Llama-3.1-8B-Instruct
target provide enough accepted speculative work, and enough real decode
speedup, to replace the failed untrained layer-skip drafts?

## Frozen screen

- Target: local Llama-3.1-8B-Instruct, BF16, greedy decoding.
- Draft: local EAGLE3-LLaMA3.1-Instruct-8B checkpoint; no retraining.
- Data: two paired 30-request screens: fixed-length RULER QA2 at 16K for
  short-answer service, and the existing official-chat MultiNews pack for
  long-form generation. Variable MultiNews prompt lengths are retained.
- Output: EOS-aware greedy completion, capped at 32 tokens for RULER and 128
  tokens for MultiNews. Post-EOS tokens are never measured.
- Speculative horizons: 2, 3, 4, and 8 tokens, plus ordinary non-speculative
  vLLM. Horizon 3 is the checkpoint default; 8 tests the longer proposal needed
  by a transport-overlap design.
- Serving geometry: batch size one, one exclusive H200, the same vLLM checkout,
  default FlashAttention, prefix caching enabled, and CUDA graphs enabled.
- Timing: an untimed one-token prefix fill and one full-horizon calibration run
  precede measurement. vLLM deliberately recomputes a trailing cache block and
  speculative lookahead can retain one additional trailing block; each timed
  request must therefore reuse all but at most two 64-token blocks of the long
  prompt. GPU synchronization brackets wall time.
- Acceptance evidence: vLLM Prometheus counters are differenced immediately
  around every timed request. Draft cycles, drafted tokens, accepted tokens,
  and per-position acceptance counts are stored in raw JSONL.
- Verification: vLLM's native EAGLE rejection path verifies every emitted token
  with the target. We additionally compare every EAGLE output token-for-token
  with an independently executed ordinary target and report common-prefix and
  exact-match diagnostics. The latter is not a gate: different target forward
  shapes can diverge at a low-margin greedy argmax through floating-point
  non-associativity even though both paths execute the same target verifier.
  Dataset-level quality non-inferiority is a separate scale-up gate.

The ordinary and EAGLE engines run in separate processes on the same physical
GPU. Model loading and graph capture are outside request latency. Candidate
latency is paired by request with the ordinary target; the report includes a
deterministic bootstrap confidence interval for mean milliseconds saved.

## Pre-registered decision rule

A speculative horizon passes this compute selection screen only if all of the
following hold:

1. Native target verification is active and speculative counters are valid.
2. Every paired request performs the same number of output-token steps.
3. Mean acceptance length, including the target bonus token, is at least 2.0.
4. Ratio-of-means decode speedup is at least 1.10x.
5. The 95% paired bootstrap interval for milliseconds saved is strictly above
   zero.

Using mean acceptance length avoids a biased rule where a longer speculative
horizon is penalized merely because it proposes more tail positions. If
acceptance passes but native speculative speedup does not, EAGLE3 remains a
possible transport-overlap mechanism, but it cannot be reported as a standalone
decode acceleration. Exact greedy divergence remains visible and triggers the
dataset-quality audit, but is not mislabeled as unverified decoding.

## Boundary with the proposed P/D method

Native EAGLE3 has local access to target hidden features and target KV. The
proposed P/D experiment must instead account for a small target-feature anchor
sent by P before the residual full target KV, execute EAGLE drafting on D while
that residual KV is in flight, and verify with the final target view. Results
from this screen therefore establish only draft viability and a lossless local
control; they do not establish hidden transfer time or superiority to Lynx.
