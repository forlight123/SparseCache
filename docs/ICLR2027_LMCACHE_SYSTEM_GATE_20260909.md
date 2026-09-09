# Real LMCache lossless P/D systems gate

Date: 2026-09-09

## Outcome

The real LMCache/vLLM deployment now exposes a large enough physical interval
for the trained direct-KV drafter. For 56 steady-state QMSum requests truncated
to 7,800 tokens, a whole-chunk 12.903% Anchor is remotely complete after
19.876 ms and full KV is complete after 129.126 ms. The resulting 109.250-ms
draft window has request-bootstrap 95% CI [109.139, 109.366] ms and minimum
108.312 ms.

The separately measured winning `g=7` direct-KV drafter takes 13.952 ms on
average, 15.317 ms at p95, and 19.547 ms at maximum over 192 runs. These are
not concurrent measurements and therefore do not yet prove end-to-end speedup,
but even the observed maximum is 5.5x below the minimum real transport window.
The systems feasibility gate passes; integration, not break-even arithmetic,
is now the next blocker.

## Experimental contract

The topology is one host and three H200 NVL GPUs:

```text
GPU 0  Qwen3-8B vLLM prefiller + LMCache NIXL sender
GPU 1  Qwen3-8B vLLM decoder   + LMCache NIXL receiver
GPU 2  monolithic Qwen3-8B paired latency control
CPU    official LMCache disaggregated-prefill proxy
```

LMCache uses its asynchronous P/D backend, UCX `cuda_ipc,cuda_copy,tcp`, CUDA
staging on both endpoints, 256-token chunks, and a 24-GiB registered buffer.
vLLM 0.23.0 runs BF16 eager Qwen3-8B at maximum model length 8,192 with prefix
caching disabled. The workload is 64 immutable QMSum packet prompts selected
deterministically by prompt length, capped at 7,800 tokens, one request at a
time, greedy, with eight output tokens. Endpoint order alternates within each
paired request.

The runtime hook is opt-in and lives entirely in SparseCache. It does not edit
the adjacent LMCache checkout. It partitions the normal LMCache token chunks
into an Anchor and Residual whose disjoint union is the original batch. The
Anchor is gathered from vLLM paged KV and submitted to NIXL first; residual
gather starts immediately after submission. Only the residual phase carries
the original `is_last_prefill`, so LMCache's existing ProxyNotif cannot fire
until all original chunks complete.

## Measurements

The upstream full-KV 1P1D control over 64 requests has mean TTFT 385.571 ms,
versus 245.927 ms for monolithic vLLM. Its paired overhead is 139.644 ms,
95% CI [135.386, 143.177] ms, and outputs are identical for 64/64 requests.

Gather-first by itself is not expected to reduce visible TTFT because no D-side
drafter consumes AnchorReady yet. Over the same 64-request protocol it records
387.946-ms P/D TTFT, 247.102-ms monolithic TTFT, and 140.843-ms paired overhead,
95% CI [136.400, 144.422] ms. Outputs remain identical for 64/64 requests.

The physical timeline for the 56 7,800-token requests is:

| event | mean | 95% CI |
|---|---:|---:|
| Anchor gather, 4/31 chunks | 16.402 ms | [16.364, 16.438] |
| Anchor NIXL write, 150,994,944 bytes | 1.375 ms | [1.366, 1.384] |
| AnchorReady from store start | 19.876 ms | [19.842, 19.911] |
| Residual gather, 27/31 chunks | 106.261 ms | [106.158, 106.372] |
| Residual NIXL write, 1,019,215,872 bytes | 3.699 ms | [3.684, 3.714] |
| FullReady from store start | 129.126 ms | [129.018, 129.242] |
| AnchorReady to FullReady | **109.250 ms** | **[109.139, 109.366]** |

This corrects an earlier interpretation of LMCache's reported roughly 120-ms
offload interval. The actual local NIXL transfer is only about five milliseconds;
most time is paged-KV gather on P and, in the unmodified decoder, full-cache
scatter on D. A paper design that prioritizes only network packets misses the
dominant local movement. Gather, wire, scatter, draft, and verify must be one
joint schedule.

## Exact seed control path

The drafter requires the first full-Target token as well as sparse KV. Upstream
vLLM normally invokes LMCache `wait_for_save` before sampling that token. The
new `SeedSignalProposer` uses vLLM's custom-proposer lifecycle only to make
sampling precede store. It queues the exact sampled token and returns an empty
proposal list; it has no model weights, consumes no EAGLE feature, and makes no
prediction.

In a second real run, the seed is sampled 0.299 ms before store starts. Across
14 steady 7,800-token requests, seed-to-AnchorReady is 19.982 ms, 95% CI
[19.874, 20.086] ms; the subsequent draft window is 106.624 ms, CI
[106.455, 106.803] ms. All 16 length-stratified endpoint outputs are identical.
The online vLLM seed can differ from the offline Hugging Face packet seed, which
rules out using prerecorded proposals in the final deployment.

## What is and is not established

Established:

* actual LMCache allocation and NIXL transfer, not a bandwidth sleep model;
* early exact seed plus early remotely resident Anchor for the same request;
* byte conservation: every full-Target chunk is sent exactly once;
* unchanged FullReady notification and 64/64 exact endpoint outputs;
* a physical window comfortably larger than the measured sparse drafter cost.

Not yet established:

* D-side consumption of Anchor KV without waiting for FullReady;
* concurrent online sparse drafting and residual gather/write;
* progressive scatter into the target cache and layer-ready verification;
* end-to-end latency reduction on the real LMCache deployment;
* bitwise-equivalent fast block verification. Standard speculative decoding is
  distributionally lossless, while the shape-invariant fast-kernel gate remains
  open.

The next implementation milestone is therefore a receiver-side AnchorReady
mailbox, direct access or scatter of only the five drafter-layer pages, proposal
epoch sealing, and one full-Target verification after exact KV readiness. No
draft token may be externally committed. Evaluation must retain the full-KV
1P1D and monolithic controls, use at least 64 paired requests, and separately
report first committed token, time to equal exact progress, total bytes, and
output/distributional correctness.

## Reproduction surface

The deployment files are in `experiments/lossless_pd/lmcache_pd/`:

* `prefiller.yaml` and `decoder.yaml`: LMCache P/D configuration;
* `anchor_runtime.py`: opt-in gather-first and AnchorReady instrumentation;
* `seed_signal_proposer.py`: zero-proposal exact-seed lifecycle hook;
* `benchmark.py`: immutable length-stratified paired endpoint benchmark;
* `analyze_trace.py`: monotonic-clock reconstruction and bootstrap intervals;
* `runtime_site/sitecustomize.py`: environment-gated installation.

Raw traces and model artifacts remain under ignored `outputs/`; the numerical
results above are fixed here so the Git repository retains the research record.
