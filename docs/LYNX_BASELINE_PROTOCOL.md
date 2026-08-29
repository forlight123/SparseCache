# Lynx baseline protocol and reproducibility boundary

Status: frozen comparison protocol, 2026-08-27. This document is an evidence
policy, not a claim that Lynx has been reproduced.

## 1. Why Lynx is the nearest baseline

Lynx and SparseCache-PD both begin tentative generation before the P/D KV
transfer finishes and apply one immutable verifier after complete target state
arrives. The independent variable differs:

| property | Lynx | SparseCache-PD |
|---|---|---|
| transferred first view | top four bits for every KV coordinate | exact BF16/FP16 KV for selected token pages |
| residual | bottom four bits for every coordinate | exact pages not yet present |
| draft attention | approximate dense context | exact sparse context |
| priority | bit significance | page semantics and network deadline |
| completed target state | hierarchical INT8 `Q_full` | original BF16/FP16 prompt KV |
| exact verification endpoint | `Q_full` distribution | original full-KV target distribution |

Consequently the primary novelty claim cannot be “decoding before all KV has
arrived.” It is exact-page arrival-aware sparse self-speculation and semantic
page scheduling under a stronger, uncompressed endpoint.

## 2. Authoritative paper facts

Source: [arXiv 2607.01831](https://arxiv.org/abs/2607.01831), PDF SHA-256
`f918a1f951293c488e63a576f6464a416dce9d59275b22e31a6c4615ea507299` as
downloaded on 2026-08-27.

- Prototype: approximately 2K lines of Ascend-C and 2K lines of Python in
  vLLM-Ascend with LMCache-Ascend.
- Hardware: two Atlas A2 servers, eight Ascend 910B4 NPUs with 32 GB HBM per
  server; experiments rate-limit the inter-server connector to 10--50 Gbps.
- Quantization: page size 256 tokens, per-channel chunk size 32, logarithmic
  transform, four-bit Anchor plus four-bit Residual, plus page/chunk scalars.
- Execution: Anchor drains before Residual, drafting polls residual completion,
  and one parallel `Q_full` verifier accepts a prefix and corrects the first
  divergence.
- Maximum proposal horizon: 64 tokens.
- Models: Llama-3.1-8B-Instruct, Qwen3-32B, and Mistral-3-24B-Instruct.
- Datasets: MMLU-Pro (512), multilingual Needle (512), and QMSum (200).
- Reported acceptance on MMLU-Pro/Qwen: 21.43 proposed and 19.38 accepted on
  average; 64.8% of proposed sequences are fully accepted.
- The total split payload is eight bits per coordinate. The verifier preserves
  generation under reconstructed hierarchical INT8 `Q_full`; the reported
  relationship to BF16 is task-score equivalence, not zero quantization error.

## 3. What is not reproducible from the public artifact

As of the status date, no author implementation or artifact link was found.
The paper does not provide all constants and layouts needed for a faithful
replacement:

- the logarithmic transform parameter `alpha` is not specified;
- epsilon, scalar precision, scalar amortization, and exact byte accounting are
  not fully specified;
- the precise inverse/sign reconstruction and packed metadata layout are not
  sufficient to reproduce the reported vNMSE unambiguously;
- Ascend-C serialization, dequantization, paged-attention, and verification
  kernels are not available on the local H200 testbed.

Ordinary INT4/INT8 KV quantization, truncating BF16 mantissas, or assuming the
paper's acceptance distribution is therefore not a Lynx reproduction.

## 4. Allowed evidence labels

Use exactly one of the following labels in tables and captions.

1. **Lynx (authors, paper)**: numbers copied from the paper, with its Ascend
   hardware, model, context, bandwidth, and metric visible. Never place these
   numbers in a same-hardware speedup column.
2. **Lynx (official reproduction)**: allowed only if author code becomes
   available, its quantization error/acceptance is reproduced, and the commit
   and modifications are recorded.
3. **split-INT8 matched-byte oracle**: an analytic upper bound with four bits
   arriving first, zero SerDes overhead, and declared acceptance. It must not
   be named Lynx and cannot support a system-comparison claim.
4. **hierarchical split-INT8 surrogate**: allowed only after every chosen
   missing constant is disclosed, quantization error is reported, and the
   implementation is kept out of the official-reproduction column.

The present status is label 1 only. Labels 2 and 4 are unavailable; label 3 is
an optional ceiling analysis.

## 5. Paper table separation

Two tables are mandatory because the endpoint guarantees differ.

### Exact BF16 endpoint table

- monolithic BF16 KV transfer plus ordinary decode;
- layer-wise BF16 transfer;
- SparseCache-PD fixed seed;
- SparseCache-PD continuous page visibility;
- no-overlap and dense-draft ablations.

Every row must pass raw greedy equality or the declared fully charged
deterministic-rematerialization control. Lynx and any lossy compression method
do not enter this table.

### Accuracy/latency Pareto table

- BF16, ordinary INT8, and ordinary INT4 transfer;
- Lynx author-reported values in a visually separate hardware block;
- an official Lynx reproduction if it later becomes available;
- SparseCache-PD exact rows;
- explicitly lossy sparse-transfer or finite-W controls.

Report total bytes, bytes before drafting, target endpoint precision, task
score, raw token equality, TT1/TT32/TT64, hardware, and link rate. A task-score
tie must not be described as a token-level exactness tie.

## 6. Same-testbed comparison if official code appears

Freeze the following before execution:

- common model: Llama-3.1-8B-Instruct;
- common contexts: 32K, 64K, and 128K;
- common links: 10, 25, and 50 Gbps;
- common fixed horizons: 32 and 64 output tokens;
- common request IDs: RULER controlled rows and QMSum official prompts;
- page size 256 and Lynx chunk size 32;
- Lynx proposal cap 64 and SparseCache caps 8/16/32/64;
- identical one-request load, warmup, execution-order rotation, and exclusive
  GPU policy;
- task quality plus raw token equality against each row's declared target.

Charge quantization, scalar packing, serialization, dequantization, transfer,
verification, correction, and all page-table work. Match physical bytes rather
than nominal bit labels. Do not transfer BF16 for one row and report only an
INT8-equivalent modeled deadline for another.

## 7. Orthogonal combination

Page scheduling and bit-plane scheduling are orthogonal, but their combination
is not part of the primary method. It may be evaluated only after the exact
page method passes its own latency gate. The combination would send low-bit
representations of all pages plus exact high-priority pages, draft over the
global approximate view, and verify after exact residual completion. It belongs
in future-work/ablation unless its additional endpoint and byte accounting are
implemented and measured.
