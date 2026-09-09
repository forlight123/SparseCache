"""All-layer exact-verifier dependency probe with real asynchronous H2D.

Uses fixed proposals from the KV block pilot. This does not simulate a network
or include draft generation/P prefill in its timed region. It measures the
copy+verification subgraph, including Python scheduling, under identical work.
"""

import argparse
import json
from pathlib import Path
import random
import statistics
import time

import torch
from transformers import DynamicCache

from experiments.lossless_pd.pilot import load_target, load_requests, write_json


def cache_with_buffers(target, buffers):
    # Merely install buffer references. DynamicCache.update would read/copy
    # unready tensors here, so it must NOT be used before the completion event.
    cache = DynamicCache(config=target.config)
    for layer, (key,value) in zip(cache.layers,buffers):
        layer.keys,layer.values = key,value
        layer.dtype,layer.device = key.dtype,key.device
        layer.is_initialized = True
    return cache


@torch.no_grad()
def run(target, block, host, mode, order):
    buffers=[(torch.empty_like(k,device="cuda"),torch.empty_like(v,device="cuda"))
             for k,v in host]
    cache=cache_with_buffers(target,buffers)
    n=host[0][0].shape[-2]
    g=block.shape[1]
    copies=torch.cuda.Stream()
    compute=torch.cuda.current_stream()
    origin=torch.cuda.Event(enable_timing=True)
    ready={}
    layer_times=[]
    torch.cuda.synchronize()
    started=time.perf_counter()
    origin.record(compute)
    copies.wait_event(origin)
    for layer_id in order:
        with torch.cuda.stream(copies):
            for destination,source in zip(buffers[layer_id],host[layer_id]):
                destination.copy_(source,non_blocking=True)
            event=torch.cuda.Event(enable_timing=True)
            event.record(copies)
            ready[layer_id]=event
    if mode != "streamed":
        compute.wait_event(ready[order[-1]])
    if mode == "native":
        hidden=target.model(block,past_key_values=cache,use_cache=True).last_hidden_state
    else:
        hidden=target.model.embed_tokens(block)
        positions=torch.arange(n,n+g,device=block.device)
        rotary=target.model.rotary_emb(hidden,positions[None])
        allowed=torch.arange(n+g,device=block.device)[None,:] <= positions[:,None]
        mask=torch.zeros(1,1,g,n+g,device=block.device,dtype=hidden.dtype)
        mask.masked_fill_(~allowed,torch.finfo(hidden.dtype).min)
        for layer_id,layer in enumerate(target.model.layers):
            if mode == "streamed":
                compute.wait_event(ready[layer_id])
            begin,end=(torch.cuda.Event(enable_timing=True) for _ in range(2))
            begin.record(compute)
            hidden=layer(hidden,attention_mask=mask,position_ids=positions[None],
                         past_key_values=cache,use_cache=True,cache_position=positions,
                         position_embeddings=rotary)
            end.record(compute)
            layer_times.append((begin,end))
        hidden=target.model.norm(hidden)
    logits=target.lm_head(hidden)
    torch.cuda.synchronize()
    ms=(time.perf_counter()-started)*1000
    tail=[(layer.keys[...,-g:,:].clone(),layer.values[...,-g:,:].clone())
          for layer in cache.layers]
    releases=[origin.elapsed_time(ready[i]) for i in range(len(host))]
    times=[{"start":origin.elapsed_time(begin),"finish":origin.elapsed_time(end),
            "cost":begin.elapsed_time(end)} for begin,end in layer_times]
    predicted=0.
    for release,timing in zip(releases,times):
        predicted=max(predicted,release)+timing["cost"]
    return logits,tail,{"mode":mode,"wall_ms":ms,"release_ms":releases,
                        "layers":times,"ideal_layer_finish_ms":predicted,
                        "payload_bytes":sum(t.numel()*t.element_size() for pair in host for t in pair)}


@torch.no_grad()
def main():
    p=argparse.ArgumentParser()
    p.add_argument("--proposals",required=True)
    p.add_argument("--output",required=True)
    p.add_argument("--target",default="/data/models/qwen/Qwen3-8B")
    p.add_argument("--requests",type=int,default=64)
    p.add_argument("--repeats",type=int,default=3)
    p.add_argument("--proposal-tokens",type=int,default=15,
                   help="number of proposal tokens after the known seed")
    p.add_argument("--verification-attention",choices=["sdpa","eager"],default="sdpa",
                   help="attention backend used only by the timed D verifier")
    args=p.parse_args()
    if args.proposal_tokens < 0:
        p.error("proposal-tokens cannot be negative")
    output=Path(args.output)
    output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(8)
    target=load_target(args.target)
    data=json.loads(Path(args.proposals).read_text())
    proposals=[r for r in data["rows"] if r["order"]=="priority" and
               r["fraction"]==.1 and r["memory_mode"]=="exact"][:args.requests]
    if len(proposals)!=args.requests:
        raise ValueError("not enough distinct proposals")
    rows=[]
    for index,proposal in enumerate(proposals):
        request=load_requests(proposal["source"],proposal["source_index"]+1,
                              proposal["prompt_tokens"])[-1]
        if request["input_sha256"] != proposal["input_sha256"]:
            raise ValueError("proposal and verifier prompt differ")
        ids=torch.tensor([request["tokens"]],device="cuda")
        # Keep the long P-side prefill on SDPA.  The D-side verifier backend is
        # an explicit experimental contract because one-token SDPA may use a
        # split reduction whose BF16 result changes with concurrent copy work.
        target.config._attn_implementation="sdpa"
        prefill=target.model(ids,use_cache=True)
        seed=int(target.lm_head(prefill.last_hidden_state[:,-1]).argmax(-1).item())
        if seed != proposal["seed"]:
            raise ValueError("P seed differs from fixed draft source")
        host=[(layer.keys.cpu().pin_memory(),layer.values.cpu().pin_memory())
              for layer in prefill.past_key_values.layers]
        del prefill
        target.config._attn_implementation=args.verification_attention
        if len(proposal["proposal"]) < args.proposal_tokens:
            raise ValueError("proposal row is shorter than --proposal-tokens")
        block=torch.tensor(
            [[seed]+proposal["proposal"][:args.proposal_tokens]], device="cuda"
        )
        order=list(range(len(host)))
        run(target,block,host,"native",order)
        native,native_tail,native_time=run(target,block,host,"native",order)
        for mode in ("serial","streamed"):
            run(target,block,host,mode,order)
        for repeat in range(args.repeats):
            modes=("serial","streamed") if repeat%2==0 else ("streamed","serial")
            pair={}
            outputs={}
            for mode in modes:
                logits,tail,timing=run(target,block,host,mode,order)
                outputs[mode]=(logits,tail)
                pair[mode]=timing
            left,lt=outputs["serial"]
            right,rt=outputs["streamed"]
            logit_equal=torch.equal(left,right)
            tail_pairs=[(a,b) for x,y in zip(lt,rt) for a,b in zip(x,y)]
            tail_equal=all(torch.equal(a,b) for a,b in tail_pairs)
            if not logit_equal or not tail_equal:
                logit_error=float((left-right).abs().max())
                tail_error=max(float((a-b).abs().max()) for a,b in tail_pairs)
                argmax_equal=bool(torch.equal(left.argmax(-1),right.argmax(-1)))
                raise AssertionError(
                    "same-work serial/streamed drift: "
                    f"record={proposal['record_id']} repeat={repeat} "
                    f"logit_equal={logit_equal} tail_equal={tail_equal} "
                    f"max_logit_error={logit_error} max_tail_error={tail_error} "
                    f"argmax_equal={argmax_equal}"
                )
            native_match=torch.equal(left,native)
            native_tail_match=all(torch.equal(a,b) for x,y in zip(lt,native_tail) for a,b in zip(x,y))
            row={"record_id":proposal["record_id"],"repeat":repeat,"context":ids.shape[1],
                 "block_tokens":block.shape[1],"serial_streamed_bitwise_equal":True,
                 "native_logits_bitwise_equal":native_match,
                 "native_generated_kv_bitwise_equal":native_tail_match,
                 "native_argmax_equal":bool(torch.equal(left.argmax(-1),native.argmax(-1))),
                 "max_native_logit_error":float((left-native).abs().max()),
                 "native":native_time,**pair}
            rows.append(row)
        print(json.dumps({"event":"full_verifier","completed":index+1,"total":len(proposals)}),flush=True)
        write_json(output/"progress.json",{"rows":rows})
        del host,block,native,native_tail,left,right,lt,rt,outputs
    request_deltas={}
    for row in rows:
        request_deltas.setdefault(row["record_id"],[]).append(
            row["serial"]["wall_ms"]-row["streamed"]["wall_ms"])
    deltas=[statistics.mean(x) for x in request_deltas.values()]
    rng=random.Random(20260909)
    boot=sorted(statistics.mean(rng.choices(deltas,k=len(deltas))) for _ in range(5000))
    summary={"requests":len(proposals),"paired_runs":len(rows),
             "proposal_tokens":args.proposal_tokens,
             "verifier_input_tokens":args.proposal_tokens+1,
             "verification_attention":args.verification_attention,
             "mean_serial_minus_streamed_ms":statistics.mean(deltas),"paired_ci95_ms":[boot[125],boot[4875]],
             "serial_streamed_bitwise_equal":all(r["serial_streamed_bitwise_equal"] for r in rows),
             "native_logits_bitwise_equal":all(r["native_logits_bitwise_equal"] for r in rows),
             "native_generated_kv_bitwise_equal":all(r["native_generated_kv_bitwise_equal"] for r in rows),
             "native_argmax_equal":all(r["native_argmax_equal"] for r in rows),
             "mean_payload_bytes":statistics.mean(r["serial"]["payload_bytes"] for r in rows),
             "mean_native_wall_ms":statistics.mean(r["native"]["wall_ms"] for r in rows),
             "mean_serial_wall_ms":statistics.mean(r["serial"]["wall_ms"] for r in rows),
             "mean_streamed_wall_ms":statistics.mean(r["streamed"]["wall_ms"] for r in rows),
             "contract":"all 36 target layers; real pinned H2D; fixed sparse-block proposals; copy+verify only; no network/E2E/sequence-level exactness claim"}
    write_json(output/"summary.json",summary)
    write_json(output/"results.json",{"summary":summary,"rows":rows})
    print(json.dumps(summary),flush=True)


if __name__=="__main__":
    main()
