# Producer-side EAGLE3 token-handoff screen

Status: frozen before the formal `n=30` source-side run. This gate measures
whether a learned proposal can be created on P while target KV is transferred,
without sending draft KV or hidden states to D. It is not an LMCache
end-to-end latency claim and does not include D-side verifier savings.

## Frozen hypothesis and topology

P executes the ordinary full target prefill and obtains the exact first target
token. At the same transition where full target KV becomes transferable, the
one-layer EAGLE3 model consumes the target auxiliary hidden states and creates
one eight-token proposal. The proposal seed and token IDs become host-visible
and are atomically exported. Only these token IDs, not EAGLE KV, would be sent
to D. D can consume them through the already implemented external proposer
after full target KV arrives.

The relevant exposed source cost is

```text
T_source_extra = T(P target prefill -> proposal ready)
               - T(P target prefill -> first exact token).
```

The internal `propose` wall span is retained as a diagnostic, but it is not
called isolated draft compute: target CUDA kernels are asynchronous and may
still be queued when the Python proposer begins.

## Formal source-side cell

- Target: local Llama-3.1-8B-Instruct, BF16, greedy decoding.
- Draft: the exact-match local RedHatAI EAGLE3 checkpoint, BF16.
- Data: QMSum requests 0--29 from the frozen LongBench request pack.
- Prompt lengths: retain the audited block-aligned natural lengths, bounded by
  32K; no runtime truncation and no prefix-cache reuse.
- Proposal horizon: eight tokens. This is the longest horizon that still
  accelerated the prior 128-token local screen and is long enough to expose
  the observed acceptance tail.
- Output used for the audit: at most nine natural-EOS tokens, so native target
  verification reveals the first proposal's useful accepted prefix.
- Serving geometry: batch size one, one exclusive H200, CUDA graphs enabled,
  one short untimed shape warmup, then paired dense and EAGLE workers in
  isolated processes.
- Modeled target-KV links: 25, 50, and 100 Gbps. Transfer time is derived from
  the target config and exact prompt token count; Llama-3.1-8B BF16 is 128 KiB
  of prompt KV per token.
- Every row stores the prompt hash, cold-cache audit, exact seed token, raw
  proposal, proposal-ready monotonic timestamp, native verifier output, first
  proposal accepted prefix, and speculative counters.

## Source gate

The topology advances to D-side replay only if all of the following hold:

1. all 30 traces are present and temporally contained in their request;
2. every dense/EAGLE prompt hash and first exact seed token matches;
3. every timed request reports zero cached prompt tokens;
4. the mean first-proposal accepted prefix is at least one token;
5. at 25 Gbps, at least 95% of proposals become ready before complete target
   KV transfer would finish.

Passing is deliberately weak: it establishes that token-only handoff is
feasible. It does not establish a net latency gain. The mandatory next gate
replays the frozen raw proposal on a separate target D engine and directly
measures verifier latency against ordinary full-KV decode. P throughput is
then charged separately because EAGLE occupies the producer after target
prefill even when its latency is hidden from D.
