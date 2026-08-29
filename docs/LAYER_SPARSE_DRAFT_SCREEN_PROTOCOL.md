# Layer-sparse × page-sparse draft selection protocol

Status: pre-registered mechanism screen. This is not an end-to-end P/D latency
claim and is not evidence that an untrained early exit preserves answer quality.

## Question

Can the draft path become materially cheaper by executing only the first N
frozen target layers while attending to only the currently visible prompt
pages, without destroying speculative acceptance?

## Frozen screen

- Target: Llama-3.1-8B-Instruct, 32 layers, BF16, greedy decoding.
- Context: fixed-length RULER QA2 requests, initially 16K smoke and then 64K.
- Proposal horizon: 8 tokens.
- Depths: 8, 16, 24, and 32 layers.
- Visible prompt pages: 5% and 100%, uniform page order.
- Reference: a full-depth pass first emits one greedy committed seed token.
  Every timed candidate starts from that seed, and the 32-layer/100%-visible
  verifier replay defines the next eight target tokens. Agreement with the
  unused monolithic `seed+8` suffix is reported only as a determinism diagnostic.
- Draft exit: first N original target layers followed by the original final norm
  and language-model head. No early-exit weights are trained in this screen.
- Timing: one exclusive H200, eager vLLM. Every timed request must reuse the
  entire block-aligned document prefix and compute only the one-token committed
  seed plus draft horizon. Trace I/O runs only after timing.

For each cell we report real wall latency, speedup over the dense reference,
the longest greedy token prefix accepted before the first mismatch, zero-accept
fraction, and the physical page count observed by PROGRESSIVE_KV.

## Pre-registered decision rule

The raw untrained exit passes only if at least one sub-32-layer cell reaches
both 50% token acceptance and 1.5x draft speedup. The combined page/layer path
passes only if the same thresholds hold at no more than 10% visible pages.

Failure rejects only the free, untrained early-exit construction. It does not
reject a trained early-exit head, EAGLE-style feature draft, or separate small
draft model; those become the next branch if compute speedup is present but
acceptance is absent.
