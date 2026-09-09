"""Bounded three-GPU exploration; all paths use the local SparseCache imports.

Acceptance is against an independently autoregressive, full-KV greedy target.
These are model/mechanism pilots, NOT end-to-end PD or sampling guarantees.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from experiments.blockdraft.model import BlockKVDraft
from experiments.kvshot.chat_data import load_regenerated_chat
from experiments.kvshot.model import KVShotDraft, apply_rope, stack_sampled_target_kv
from experiments.lossless_pd.core import (
    ProgressiveBlock, accepted_prefix, attention_state, attention_value,
    expand_gqa, fixed_query_bound, kvshot_pd_propose, merge_attention,
    page_order, visible_positions,
)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def digest_file(path):
    sha = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            sha.update(chunk)
    return sha.hexdigest()


def sync():
    torch.cuda.synchronize()


def load_target(path):
    target = AutoModelForCausalLM.from_pretrained(
        path, local_files_only=True, dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).to("cuda").eval().requires_grad_(False)
    if target.config.model_type != "qwen3" or target.config.rope_scaling:
        raise ValueError("this fixed-query reference is audited for unscaled Qwen3 RoPE")
    return target


@torch.no_grad()
def context_forward(target, ids, layer_ids, query_index=-1):
    captured = {}
    n = ids.shape[1]
    index = query_index % n
    handles = []
    for layer_id in layer_ids:
        def capture(module, args, kwargs, layer_id=layer_id):
            hidden = kwargs.get("hidden_states", args[0] if args else None)
            hidden = hidden[:, index:index + 1]
            q = module.q_norm(module.q_proj(hidden).view(
                1, 1, target.config.num_attention_heads, target.config.head_dim
            )).transpose(1, 2)
            captured[layer_id] = apply_rope(
                q, ids.new_tensor([[index]]), target.config.rope_theta
            ).detach()
        handles.append(target.model.layers[layer_id].self_attn.register_forward_pre_hook(
            capture, with_kwargs=True
        ))
    try:
        output = target.model(ids, use_cache=True, return_dict=True)
    finally:
        for handle in handles:
            handle.remove()
    return output, captured


@torch.no_grad()
def priority_scores(keys, queries, layer_ids, page_size):
    n = keys.shape[-2]
    mass = torch.zeros(n, device=keys.device, dtype=torch.float32)
    for index, layer_id in enumerate(layer_ids):
        q = queries[layer_id]
        key = expand_gqa(keys[:, index], q.shape[1])
        score = q.float() @ key.float().transpose(-1, -2) / math.sqrt(q.shape[-1])
        mass += score.softmax(-1).mean(dim=(0, 1, 2)) / len(layer_ids)
    padded = F.pad(mass, (0, (-n) % page_size))
    return padded.reshape(-1, page_size).sum(-1)


def load_requests(paths, count, max_context):
    rows = []
    for path in paths.split(","):
        with Path(path).open() as handle:
            for index, line in enumerate(handle):
                raw = json.loads(line)
                tokens = raw["prompt"]
                if not isinstance(tokens, list):
                    raise ValueError("request packs must already contain model-specific token IDs")
                original = len(tokens)
                if original > max_context:
                    half = max_context // 2
                    tokens = tokens[:half] + tokens[-(max_context - half):]
                rows.append({
                    "record_id": f"{Path(path).parent.name}:{index}",
                    "source": str(Path(path).resolve()),
                    "source_index": index,
                    "original_tokens": original,
                    "prompt_tokens": len(tokens),
                    "truncated": original > max_context,
                    "tokens": tokens,
                    "input_sha256": hashlib.sha256(json.dumps(tokens).encode()).hexdigest(),
                })
                if len(rows) == count:
                    return rows
    if len(rows) < count:
        raise ValueError(f"requested {count} distinct requests, found {len(rows)}")
    return rows


@torch.no_grad()
def greedy_suffix(target, cache, seed, length, stop_ids):
    if int(seed.item()) in stop_ids:
        return []
    token = seed
    tokens = []
    for _ in range(length):
        output = target.model(token, past_key_values=cache, use_cache=True)
        token = target.lm_head(output.last_hidden_state[:, -1:]).argmax(-1)
        tokens.append(int(token.item()))
        cache = output.past_key_values
        if tokens[-1] in stop_ids:
            break
    return tokens


def bootstrap_delta(values, seed=971, repeats=2000):
    rng = random.Random(seed)
    means = sorted(statistics.mean(rng.choices(values, k=len(values)))
                   for _ in range(repeats))
    return [means[int(repeats * .025)], means[int(repeats * .975)]]


@torch.no_grad()
def evaluate(args, target, model, kind, output_dir, label):
    model.eval()
    layer_ids = [1, 17, 32] if kind == "kvshot" else [1, 9, 17, 25, 33]
    requests = load_requests(args.requests, args.num_requests, args.max_context)
    eos = target.generation_config.eos_token_id
    stop_ids = {eos} if isinstance(eos, int) else set(eos or [])
    fractions = [float(x) for x in args.fractions.split(",")]
    cells = [(order, frac, "exact") for order in args.orders.split(",") for frac in fractions]
    cells += [("priority", .1, "zero"), ("priority", .1, "shuffled")]
    rows, grouped = [], {}
    previous = None
    for request_index, request in enumerate(requests):
        ids = torch.tensor([request["tokens"]], device="cuda", dtype=torch.long)
        out, queries = context_forward(target, ids, layer_ids)
        keys, values = stack_sampled_target_kv(out.past_key_values, layer_ids)
        scores = priority_scores(keys, queries, layer_ids, args.page_size)
        seed = target.lm_head(out.last_hidden_state[:, -1:]).argmax(-1)
        reference = greedy_suffix(target, out.past_key_values, seed, args.draft_tokens, stop_ids)
        del out, queries
        for order_mode, fraction, memory_mode in cells:
            order = page_order(ids.shape[1], args.page_size, order_mode,
                               args.seed + request_index, scores)
            pos = visible_positions(ids.shape[1], args.page_size, fraction, order, ids.device)
            selected_k = keys.index_select(-2, pos)
            selected_v = values.index_select(-2, pos)
            if memory_mode == "zero":
                selected_k = torch.zeros_like(selected_k)
                selected_v = torch.zeros_like(selected_v)
            elif memory_mode == "shuffled":
                if previous is None:
                    # No same-request substitute: omit until a real other request exists.
                    continue
                pk, pv = previous
                # Put other-request content in the same original-position frame.
                idx = torch.arange(pos.numel(), device=ids.device) % pk.shape[-2]
                src_pos = idx[None]
                selected_k = pk.index_select(-2, idx)
                b, layers, heads, t, dim = selected_k.shape
                selected_k = apply_rope(selected_k.reshape(b * layers, heads, t, dim),
                                       src_pos.repeat_interleave(layers, 0),
                                       target.config.rope_theta, inverse=True)
                selected_k = apply_rope(selected_k, pos[None].repeat_interleave(layers, 0),
                                       target.config.rope_theta).reshape(b, layers, heads, t, dim)
                selected_v = pv.index_select(-2, idx)
            sync()
            start = time.perf_counter()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                if kind == "kvshot":
                    proposal = kvshot_pd_propose(model, seed, target.model.embed_tokens,
                                                selected_k, selected_v, pos, ids.shape[1],
                                                args.draft_tokens)[0].tolist()
                else:
                    proposal = model.propose(
                        seed, target.model.embed_tokens, target.lm_head,
                        selected_k, selected_v, pos, ids.shape[1],
                        length=args.draft_tokens,
                    )[0].tolist()
            sync()
            elapsed = (time.perf_counter() - start) * 1000
            accepted = accepted_prefix(proposal, reference, stop_ids)
            row = {k: v for k, v in request.items() if k != "tokens"}
            row.update(label=label, kind=kind, order=order_mode, fraction=fraction,
                       actual_fraction=pos.numel() / ids.shape[1], memory_mode=memory_mode,
                       seed=int(seed.item()), reference=reference, proposal=proposal,
                       accepted=accepted, draft_ms_unfused_with_projection=elapsed,
                       generated_positions=(args.draft_tokens if kind == "kvshot"
                                            else model.base.config.block_size - 1))
            rows.append(row)
            key = f"{order_mode}/f{fraction:g}/{memory_mode}"
            grouped.setdefault(key, []).append(row)
        previous = (keys, values)
        print(json.dumps({"event":"evaluation", "label": label,
                          "completed": request_index + 1, "total": len(requests)}), flush=True)
    summary = {}
    for key, cell in grouped.items():
        accepted = [r["accepted"] for r in cell]
        summary[key] = {
            "requests": len(cell), "mean_accepted": statistics.mean(accepted),
            "accepted_mean_ci95_request_bootstrap": bootstrap_delta(accepted),
            "zero_acceptance_rate": sum(x == 0 for x in accepted) / len(cell),
            "mean_draft_ms_unfused_with_projection": statistics.mean(
                r["draft_ms_unfused_with_projection"] for r in cell),
            "mean_actual_fraction": statistics.mean(r["actual_fraction"] for r in cell),
        }
    paired = {}
    exact = {r["record_id"]: r["accepted"] for r in grouped.get("priority/f0.1/exact", [])}
    for mode in ("zero", "shuffled"):
        other = grouped.get(f"priority/f0.1/{mode}", [])
        delta = [exact[r["record_id"]] - r["accepted"] for r in other
                 if r["record_id"] in exact]
        if delta:
            paired[mode] = {"pairs": len(delta), "mean_delta": statistics.mean(delta),
                            "ci95_request_bootstrap": bootstrap_delta(delta)}
    result = {"label":label, "cells":summary, "paired_memory_delta":paired,
              "metric_contract": "greedy accepted prefix excluding P seed; natural EOS; no online speedup claim",
              "context_contract": "head-tail truncation where flagged; mechanism only, not official task quality",
              "rows":rows}
    write_json(output_dir / f"{label}.json", result)
    return {k:v for k,v in result.items() if k != "rows"}


def train_block(args, target, model, output_dir):
    tokenizer = AutoTokenizer.from_pretrained(args.target, local_files_only=True)
    length = model.base.config.block_size - 1
    if args.draft_tokens > length:
        raise ValueError("requested evaluation horizon exceeds block output positions")
    records = load_regenerated_chat(
        args.dataset, tokenizer, max_seq_len=args.train_context, ttt_length=length,
        cache_path=output_dir / "tokenized.pt",
    )
    metadata = json.loads((Path(args.block_checkpoint) / "run_metadata.json").read_text())
    allowed = set(metadata["train_ids"])
    forbidden = set(metadata["validation_ids"])
    records = [r for r in records if r.record_id in allowed and r.record_id not in forbidden]
    rng = random.Random(args.seed)
    rng.shuffle(records)
    records = records[:args.train_records]
    if len(records) < args.train_records:
        raise ValueError("not enough lineage-audited train records")
    write_json(output_dir / "training_split.json", {
        "train_ids": [r.record_id for r in records],
        "excluded_parent_validation_ids": sorted(forbidden),
        "parent_metadata": str(Path(args.block_checkpoint) / "run_metadata.json"),
        "dataset_sha256": digest_file(args.dataset),
        "training_window_contract": "teacher-forced assistant cuts; KV and priority query end BEFORE known seed",
    })
    model.float().train()
    for name, p in model.named_parameters():
        p.requires_grad_(args.train_scope == "all" or name.startswith("stage.")
                         or "target_hidden_projector" in name or "prefix_projectors" in name)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                  lr=args.lr, weight_decay=.01)
    start = time.perf_counter()
    fractions = [float(x) for x in args.fractions.split(",")]
    layer_ids = [1, 9, 17, 25, 33]
    position_weights = torch.tensor([.9 ** j for j in range(length)], device="cuda")
    with (output_dir / "train.jsonl").open("w") as log:
        for step in range(args.steps):
            record = records[rng.randrange(len(records))]
            cut = record.candidate_cuts[rng.randrange(len(record.candidate_cuts))]
            ids = record.input_ids[:cut + length].unsqueeze(0).to("cuda")
            with torch.no_grad():
                out, queries = context_forward(target, ids, layer_ids, query_index=cut - 1)
                all_k, all_v = stack_sampled_target_kv(out.past_key_values, layer_ids)
                keys, values = all_k[..., :cut, :], all_v[..., :cut, :]
                scores = priority_scores(keys, queries, layer_ids, args.page_size)
                teacher = target.lm_head(out.last_hidden_state[:, cut:cut + length]).float().softmax(-1)
            order = page_order(cut, args.page_size, "priority", args.seed + step, scores)
            fraction = 1.0 if args.train_views == "full" else fractions[step % len(fractions)]
            pos = visible_positions(cut, args.page_size, fraction, order, ids.device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(ids[:, cut:cut + 1], target.model.embed_tokens, target.lm_head,
                               keys.index_select(-2, pos), values.index_select(-2, pos), pos, cut,
                               previous_tokens=(
                                   ids[:, cut:cut + length]
                                   if model.base.correction_gru is not None else None
                               ))
                losses = -(teacher * logits.float().log_softmax(-1)).sum(-1)[0]
                loss = (losses * position_weights).sum() / position_weights.sum()
            if not torch.isfinite(loss):
                raise RuntimeError("nonfinite training loss")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            scale = min(1., (step + 1) / 25) * .5 * (1 + math.cos(math.pi * step / args.steps))
            for group in optimizer.param_groups:
                group["lr"] = args.lr * scale
            optimizer.step()
            row = {"step":step + 1,"record_id":record.record_id,"cut":cut,
                   "fraction":fraction,"actual_fraction":pos.numel()/cut,
                   "teacher_forced_distill_loss":float(loss.detach()),
                   "grad_norm":float(norm),"elapsed_s":time.perf_counter()-start}
            log.write(json.dumps(row) + "\n")
            if step == 0 or (step + 1) % 25 == 0:
                log.flush()
                print(json.dumps(row), flush=True)
            del out, all_k, all_v, keys, values, teacher, logits, loss, queries
    model.eval()
    model.base.save_checkpoint(output_dir / "checkpoint")
    torch.save(model.stage.state_dict(), output_dir / "checkpoint" / "stage.pt")
    write_json(output_dir / "training_summary.json", {
        "steps":args.steps,"distinct_train_records_available":len(records),
        "supervised_positions":args.steps * length,
        "trainable_parameters":sum(p.numel() for p in model.parameters() if p.requires_grad),
        "elapsed_seconds":time.perf_counter()-start,
        "label":"bounded continued-training pilot, not a paper-scale model",
    })


def transfer_attention(query, cpu_k, cpu_v, chunk, asynchronous):
    gpu_k = torch.empty_like(cpu_k, device="cuda")
    gpu_v = torch.empty_like(cpu_v, device="cuda")
    copy_stream = torch.cuda.Stream()
    compute = torch.cuda.current_stream()
    completions = []
    sync()
    started = time.perf_counter()
    if asynchronous:
        for begin in range(0, cpu_k.shape[-2], chunk):
            end = min(begin + chunk, cpu_k.shape[-2])
            with torch.cuda.stream(copy_stream):
                gpu_k[..., begin:end, :].copy_(cpu_k[..., begin:end, :], non_blocking=True)
                gpu_v[..., begin:end, :].copy_(cpu_v[..., begin:end, :], non_blocking=True)
                event = torch.cuda.Event()
                event.record(copy_stream)
            completions.append((begin, end, event))
    else:
        gpu_k.copy_(cpu_k, non_blocking=True)
        gpu_v.copy_(cpu_v, non_blocking=True)
        compute.synchronize()
        completions = [(begin, min(begin + chunk, cpu_k.shape[-2]), None)
                       for begin in range(0, cpu_k.shape[-2], chunk)]
    state = None
    for begin, end, event in completions:
        if event is not None:
            compute.wait_event(event)
        state = merge_attention(state, attention_state(query, gpu_k[..., begin:end, :],
                                                       gpu_v[..., begin:end, :]))
    output = attention_value(state)
    sync()
    return output, (time.perf_counter()-started)*1000


@torch.no_grad()
def attention_pilot(args, target, output_dir):
    requests = load_requests(args.requests, args.num_requests, args.max_context)
    layer_ids = [0, 17, 35]
    bounds, timings = [], []
    for request_index, request in enumerate(requests):
        ids = torch.tensor([request["tokens"]], device="cuda", dtype=torch.long)
        out, queries = context_forward(target, ids, layer_ids)
        keys, values = stack_sampled_target_kv(out.past_key_values, layer_ids)
        scores = priority_scores(keys, queries, layer_ids, args.page_size)
        order = page_order(ids.shape[1], args.page_size, "priority", args.seed, scores)
        del out
        for index, layer_id in enumerate(layer_ids):
            query, key, value = queries[layer_id], keys[:, index], values[:, index]
            for fraction in map(float, args.fractions.split(",")):
                pos = visible_positions(ids.shape[1], args.page_size, fraction, order, ids.device)
                row = fixed_query_bound(query, key, value, pos, args.page_size)
                row.update(record_id=request["record_id"],layer=layer_id,fraction=fraction,
                           actual_fraction=pos.numel()/ids.shape[1],context=ids.shape[1])
                bounds.append(row)
            if layer_id == 17:
                cpu_k, cpu_v = key.cpu().pin_memory(), value.cpu().pin_memory()
                reference = attention_value(attention_state(query, key, value))
                for chunk in (256, 1024, 4096):
                    transfer_attention(query, cpu_k, cpu_v, chunk, False)
                    transfer_attention(query, cpu_k, cpu_v, chunk, True)
                    for repeat in range(args.repeats):
                        pair = {}
                        for asynchronous in ((False,True) if repeat % 2 == 0 else (True,False)):
                            result, ms = transfer_attention(query, cpu_k, cpu_v, chunk, asynchronous)
                            error = float((result-reference).abs().max())
                            if not torch.allclose(result, reference, atol=1e-5, rtol=1e-4):
                                raise AssertionError(f"streaming merge mismatch {error}")
                            pair["async" if asynchronous else "serial"] = ms
                            pair["max_abs_error"] = max(pair.get("max_abs_error", 0), error)
                        timings.append(dict(record_id=request["record_id"],chunk=chunk,
                                            repeat=repeat,**pair))
        print(json.dumps({"event":"attention", "completed":request_index+1,
                          "total":len(requests)}),flush=True)
        write_json(output_dir / "attention_progress.json", {"bounds":bounds,"timings":timings})
    summary = {"bound_violations":sum(r["bound_violations_fp64_tolerance_1e_9"] for r in bounds),
               "mass_bound_violations":sum(r["mass_violations_fp64_tolerance_1e_9"] for r in bounds),
               "max_streaming_abs_error":max(r["max_abs_error"] for r in timings),
               "mean_async_minus_serial_ms":statistics.mean(r["async"]-r["serial"] for r in timings),
               "contract":"fixed exact P-prefill query, one-head-group attention bound; NOT full-model token certification; real local pinned H2D only, no network emulation"}
    write_json(output_dir / "attention.json", {"summary":summary,"bounds":bounds,"timings":timings})
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode",choices=["kvshot","block","attention"],required=True)
    p.add_argument("--target",default="/data/models/qwen/Qwen3-8B")
    p.add_argument("--block-checkpoint",default="../CacheDraft/results/blockdraft/qwen3_8b_dflash_b16_exactkv_hiddenadapter_joint2500")
    p.add_argument("--kvshot-checkpoint",default="../CacheDraft/results/kvshot/qwen3_8b_pure_kv_regen_full_dense4_10k_s512")
    p.add_argument("--requests",default="outputs/progressive_kv/pd_packs/qwen3_8b/quality/qmsum/requests.jsonl")
    p.add_argument("--dataset",default="../CacheDraft/datasets/kvshot/sharegpt10k/sharegpt_train_regen_qwen3_8b_t0_n1000.jsonl")
    p.add_argument("--output",required=True)
    p.add_argument("--num-requests",type=int,default=64)
    p.add_argument("--max-context",type=int,default=8192)
    p.add_argument("--draft-tokens",type=int,default=15)
    p.add_argument("--page-size",type=int,default=64)
    p.add_argument("--fractions",default="0.05,0.1,0.2,0.5,1")
    p.add_argument("--orders",default="priority,random")
    p.add_argument("--seed",type=int,default=20260909)
    p.add_argument("--steps",type=int,default=0)
    p.add_argument("--train-views",choices=["full","nested"],default="nested")
    p.add_argument("--train-records",type=int,default=256)
    p.add_argument("--train-context",type=int,default=2048)
    p.add_argument("--train-scope",choices=["all","memory"],default="all")
    p.add_argument("--lr",type=float,default=1e-5)
    p.add_argument("--repeats",type=int,default=5)
    args = p.parse_args()
    if min(args.num_requests,args.max_context,args.draft_tokens,args.page_size,args.repeats) <= 0 or args.steps < 0:
        p.error("invalid resource/shape limit")
    output = Path(args.output)
    output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(8)
    torch.manual_seed(args.seed)
    metadata = {"arguments":vars(args),"torch":torch.__version__,
                "gpu":torch.cuda.get_device_name(),"started_unix":time.time(),
                "no_eagle_adapter_training":True,"status":"running"}
    write_json(output / "manifest.json",metadata)
    target = load_target(args.target)
    if args.mode == "attention":
        result = attention_pilot(args,target,output)
    else:
        checkpoint = Path(args.kvshot_checkpoint if args.mode == "kvshot" else args.block_checkpoint)
        weight = checkpoint / ("kvshot_draft.pt" if args.mode == "kvshot" else "block_kv_draft.pt")
        metadata["initial_checkpoint_sha256"] = digest_file(weight)
        metadata["initial_checkpoint"] = str(checkpoint.resolve())
        write_json(output / "manifest.json",metadata)
        if args.mode == "kvshot":
            model = KVShotDraft.load_checkpoint(checkpoint).to("cuda",dtype=torch.bfloat16)
        else:
            model = ProgressiveBlock(BlockKVDraft.load_checkpoint(checkpoint)).to("cuda",dtype=torch.bfloat16)
        result = {"before":evaluate(args,target,model,args.mode,output,"before")}
        if args.steps:
            if args.mode != "block":
                raise ValueError("only the KV-input block model is trained in this experiment")
            train_block(args,target,model,output)
            result["after"] = evaluate(args,target,model,args.mode,output,"after")
    write_json(output / "summary.json",result)
    metadata.update(status="completed",finished_unix=time.time())
    write_json(output / "manifest.json",metadata)
    print(json.dumps({"event":"completed","output":str(output)}),flush=True)


if __name__ == "__main__":
    main()
