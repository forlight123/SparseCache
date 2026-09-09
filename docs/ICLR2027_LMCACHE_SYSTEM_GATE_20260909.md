# Real LMCache lossless P/D systems gate

Date: 2026-09-09

## Outcome

The real LMCache/vLLM deployment now delivers the early Anchor into the decoder
process, not merely into a P-side trace. For 56 steady-state QMSum requests
truncated to 7,800 tokens, a whole-chunk 12.903% Anchor is remotely complete
after 20.043 ms and full KV is complete after 128.010 ms. The resulting
107.968-ms draft window has request-bootstrap 95% CI [107.738, 108.205] ms.

Across the full 64-request run, the decoder resolves 244/244 advertised CUDA KV
objects and 9,210,691,584/9,210,691,584 resident bytes. P-side NIXL completion
to D-side control-message receipt is 0.603 ms; object lookup is 0.228 ms. The
mailbox creates 1,220 views of Target layers `[1,9,17,25,33]`; every view aliases
the original CUDA allocation. The complete key-resolution, pin, view, alias
check, and release path costs 0.911 ms, 95% CI [0.883, 0.934] ms.

The separately measured winning `g=7` direct-KV drafter takes 13.952 ms on
average, 15.317 ms at p95, and 19.547 ms at maximum over 192 runs. These are
not concurrent measurements and therefore do not yet prove end-to-end speedup,
but even the observed maximum is 5.5x below the minimum real transport window.
The systems feasibility and decoder-layout gates pass. The remaining integration
blocker is invoking the trained direct-KV drafter from this mailbox while the
Residual transfer continues, then submitting its sealed proposal block to the
unchanged full-KV Target verifier.

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
caching disabled. `PYTHONHASHSEED=0` is fixed before both processes start; P
uses `kv_producer` and D uses `kv_consumer`. The workload is 64 immutable QMSum
packet prompts selected deterministically by prompt length, capped at 7,800
tokens, one request at a time, greedy, with eight output tokens. Endpoint order
alternates within each paired request.

The runtime hook is opt-in and lives entirely in SparseCache. It does not edit
the adjacent LMCache checkout. It partitions the normal LMCache token chunks
into an Anchor and Residual whose disjoint union is the original batch. The
Anchor is gathered from vLLM paged KV and submitted to NIXL first; residual
gather starts immediately after submission. Only the residual phase carries
the original `is_last_prefill`, so LMCache's existing ProxyNotif cannot fire
until all original chunks complete.

## Measurements

The correctly configured upstream full-KV 1P1D control over 64 requests has
mean TTFT 388.300 ms versus 247.515 ms for monolithic vLLM. Its paired overhead
is 140.785 ms, 95% CI [136.451, 145.649] ms. The progressive run with live
mailbox and five-layer views records 389.606-ms P/D TTFT, 247.350-ms monolithic
TTFT, and 142.256-ms paired overhead, CI [137.236, 148.768] ms. These are
separate runs, so their small difference is not an end-to-end speedup result.
No online drafter consumes the window yet.

The correctness reference is the deployed full-KV P/D path, not a different
monolithic execution shape. Progressive output equals full-KV P/D for 64/64
requests. Both P/D variants equal monolithic output for 59/64 requests and
diverge on exactly the same five ordinals. Thus progressive transfer adds zero
observed output drift; the five greedy differences already belong to LMCache's
full-P/D versus monolithic numerical path.

An earlier 64/64 monolithic-equality result was obtained without a fixed builtin
hash. Decoder logs now show that this can miss transferred keys and fall back to
local recomputation, so that result is superseded and must not be cited as a KV
reuse correctness result.

The physical timeline for the 56 7,800-token requests in the final run is:

| event | mean | 95% CI |
|---|---:|---:|
| Anchor gather, 4/31 chunks | 16.203 ms | [16.165, 16.242] |
| Anchor NIXL write, 150,994,944 bytes | 1.395 ms | [1.384, 1.407] |
| AnchorReady from store start | 20.043 ms | [19.966, 20.123] |
| Residual gather, 27/31 chunks | 105.204 ms | [104.965, 105.446] |
| Residual NIXL write, 1,019,215,872 bytes | 3.592 ms | [3.581, 3.603] |
| FullReady from store start | 128.010 ms | [127.750, 128.277] |
| AnchorReady to FullReady | **107.968 ms** | **[107.738, 108.205]** |

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

In the final run, the seed is sampled 0.304 ms before store starts. Across
56 steady 7,800-token requests, seed-to-AnchorReady is 20.346 ms, 95% CI
[20.270, 20.427] ms; the subsequent draft window is 107.968 ms, CI
[107.738, 108.205] ms.
The online vLLM seed can differ from the offline Hugging Face packet seed, which
rules out using prerecorded proposals in the final deployment.

## What is and is not established

Established:

* actual LMCache allocation and NIXL transfer, not a bandwidth sleep model;
* early exact seed plus 64/64 decoder-resolved Anchors for the same request;
* partition conservation: every requested chunk belongs to exactly one
  Anchor/Residual phase, while receiver-resident keys send no duplicate bytes;
* ordered zero-copy views of the five Target-KV drafter layers;
* unchanged FullReady notification and 64/64 equality to full-KV P/D;
* a physical window comfortably larger than the measured sparse drafter cost.

Not yet established:

* an actual model forward consuming the claimed Anchor views;
* concurrent online sparse drafting and residual gather/write;
* progressive scatter into the target cache and layer-ready verification;
* end-to-end latency reduction on the real LMCache deployment;
* bitwise-equivalent fast block verification. Standard speculative decoding is
  distributionally lossless, while the shape-invariant fast-kernel gate remains
  open.

The next implementation milestone is therefore wiring the trained direct-KV
block and top-64 reranker to the receiver mailbox, proposal epoch sealing, and
one full-Target verification after exact KV readiness. No
draft token may be externally committed. Evaluation must retain the full-KV
1P1D and monolithic controls, use at least 64 paired requests, and separately
report first committed token, time to equal exact progress, total bytes, and
output/distributional correctness.

## Reproduction surface

The deployment files are in `experiments/lossless_pd/lmcache_pd/`:

* `prefiller.yaml` and `decoder.yaml`: LMCache P/D configuration;
* `anchor_runtime.py`: opt-in gather-first and AnchorReady instrumentation;
* `receiver_runtime.py`: decoder mailbox, pin/release API, and zero-copy layers;
* `seed_signal_proposer.py`: zero-proposal exact-seed lifecycle hook;
* `benchmark.py`: immutable length-stratified paired endpoint benchmark;
* `analyze_trace.py`: monotonic-clock reconstruction and bootstrap intervals;
* `analyze_receiver_trace.py`: decoder object/view integrity and latency;
* `compare_pd_outputs.py`: progressive versus authoritative full-P/D equality;
* `runtime_site/sitecustomize.py`: environment-gated installation.

Raw traces and model artifacts remain under ignored `outputs/`; the numerical
results above are fixed here so the Git repository retains the research record.
