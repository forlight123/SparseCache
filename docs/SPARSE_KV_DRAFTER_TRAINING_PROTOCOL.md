# Sparse target-KV drafter training protocol

Status: frozen Phase-A smoke protocol. This validates the training/data path;
it is not an end-to-end latency or generalization claim.

## Deployment contract

The P side commits one exact greedy seed token and sends three last-token
boundary hidden states. The D-side drafter receives only:

- the committed seed token and its own draft-token history;
- exact target K/V from layers 8, 20, and 31 for pages that have arrived;
- boundary hidden states from target layers 2, 16, and 29;
- absolute RoPE positions, prompt length, and the visible-token fraction.

It never receives missing K/V, full-prompt hidden states, future verifier
states, answers, or oracle relevance labels. The target embedding and LM head
are frozen and shared. The trainable drafter is one 512-wide block with four
128-dimensional self-attention heads and eight direct cross-attention heads
matching the target K/V layout.

## Frozen Phase-A smoke split

- Target: local Llama-3.1-8B-Instruct, BF16, greedy decoding.
- Data: QMSum requests 0, 1, and 5 for training; request 8 held out.
- Horizon: eight tokens after the exact P-side seed token.
- Page size: 64 tokens.
- Sparse views: independently sampled 5%, 10%, and 20% page budgets, always
  retaining the first and final page. The realized fraction is recorded.
- Teacher: full target model under the same exact prompt; top-64 distribution
  values and greedy labels are materialized once.
- Objective: prefix-weighted hard-token cross entropy plus renormalized top-64
  distillation. The v2 scale-up additionally matches the verifier's final
  normalized hidden state, drops the seed feature on 25% of updates, and uses
  mismatched-request KV as a contrastive negative. Earlier positions receive
  higher weight because the first rejection terminates a speculative block.
- Optimizer: AdamW, 120 steps, learning rate 3e-4, batch size one.

The selection seed, request IDs, artifact hashes, teacher manifest, training
history, and pre/post metrics are persisted. The evaluation page selection is
fixed and differs from each step's training selection.

## What constitutes success

Phase A passes when all artifacts validate, loss is finite and decreases, the
training requests gain nonzero autoregressive accepted-prefix length, and the
held-out path runs without leakage. Held-out improvement on one request is only
a diagnostic. It does not justify a quality claim.

After Phase A, scale to disjoint multi-dataset train/validation/test splits and
train with nested arrival masks and student on-policy rollouts. The go/no-go
metric for that phase is verifier-accepted tokens per millisecond at fixed
answer quality, compared with no-draft waiting and the existing EAGLE screen.

## Static-mask scale-up following the smoke

Before introducing nested masks, run all 200 QMSum requests through the same
teacher contract. Indices divisible by five form a frozen 40-request
validation split; the remaining 160 requests train the drafter. This isolates
generalization from the three-request smoke while changing no model or input
contract. It is still an in-domain pilot, not the final multi-dataset result.
The positive-KV model must outperform zeroed and request-shuffled KV on the
same 40 requests; otherwise the drafter has learned a boundary-feature shortcut
and does not pass as a sparse-KV-guided model.

The next controlled variant replaces random pages with a deployable P-side
priority: the final prompt query at target layers 8, 20, and 31 scores every
64-token page by exact attention mass. The first and final pages remain
protected; all other pages arrive in descending score order, so 5%, 10%, and
20% views are nested prefixes of one transfer schedule. No answer or verifier
future token is used to construct this order.
