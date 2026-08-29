# SPDX-License-Identifier: Apache-2.0
"""P/D progressive-transfer draft/verify feasibility experiment.

The prefill side produces one correct contiguous cache for the complete prompt.
The decode side always receives the small system/question regions, while
document regions become visible in priority order.  A seed token sampled by the
prefill side starts decode-side self-speculation.  No independent-document KV
reuse is involved in this experiment.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from progressive_kv_feasibility import (
    answer_scores,
    clean_answer,
    continue_from_view,
    decode_text,
    encode_contiguous_cache,
    eos_token_ids,
    full_prefill_generate,
    level_metrics,
    load_jsonl,
    mean,
    normalize_answer,
    parse_fractions,
    parse_schedules,
    parse_windows,
    progressive_chain,
    ranked_document_indices,
    stage_document_sets,
    stratified_indices,
    synchronized_call,
    teacher_predictions,
    tokenize_prompt,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--sample-count", type=int, default=150)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--count", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--draft-tokens", type=int, default=4)
    parser.add_argument("--seed-tokens", type=int, default=1)
    parser.add_argument("--answer-word-limit", type=int, default=5)
    parser.add_argument("--stage-fractions", default="0.2,0.4,0.6,0.8,1.0")
    parser.add_argument(
        "--schedules", default="query,oracle,random,support_last"
    )
    parser.add_argument("--commit-windows", default="1,2,inf")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    args.stage_fractions = parse_fractions(args.stage_fractions)
    args.schedules = parse_schedules(args.schedules)
    args.commit_windows = parse_windows(args.commit_windows)
    if min(args.sample_count, args.count, args.max_new_tokens, args.draft_tokens) <= 0:
        parser.error("sample/count/generation lengths must be positive")
    if not 0 < args.seed_tokens < args.max_new_tokens:
        parser.error("seed-tokens must lie between zero and max-new-tokens")
    if args.offset < 0 or args.offset + args.count > args.sample_count:
        parser.error("invalid sample shard")
    return args


def pd_visibility_mask(prefix_tokens, documents, suffix_tokens, selected, device):
    document_end = prefix_tokens + sum(len(document.token_ids) for document in documents)
    total = document_end + suffix_tokens
    mask = torch.zeros((1, total), dtype=torch.long, device=device)
    mask[:, :prefix_tokens] = 1
    for index in selected:
        document = documents[index]
        mask[:, document.start:document.end] = 1
    # System/chat prefix and the fully contextualized query region form the
    # mandatory small P/D anchor. Only document pages are progressively exposed.
    mask[:, document_end:] = 1
    return mask


def evaluate_example(model, tokenizer, row, dataset_index, args, device):
    prefix, documents, question_suffix = tokenize_prompt(
        tokenizer, row, args.answer_word_limit
    )
    full_prompt = tuple(prefix)
    for document in documents:
        full_prompt += document.token_ids
    full_prompt += tuple(question_suffix)
    eos_ids = eos_token_ids(model, tokenizer)

    p_tokens, p_prefill_ms, p_decode_ms = full_prefill_generate(
        model,
        full_prompt,
        max_new_tokens=args.max_new_tokens,
        eos_ids=eos_ids,
        device=device,
    )
    full_cache, cache_producer_ms = synchronized_call(
        device, lambda: encode_contiguous_cache(model, full_prompt, device)
    )
    all_documents = set(range(len(documents)))
    full_mask = pd_visibility_mask(
        len(prefix), documents, len(question_suffix), all_documents, device
    )
    seed_ids = p_tokens[:args.seed_tokens]
    pd_tail, pd_seed_forward_ms, pd_decode_ms = continue_from_view(
        model,
        full_cache,
        full_mask,
        seed_ids,
        (),
        additional_tokens=args.max_new_tokens - len(seed_ids),
        eos_ids=eos_ids,
        device=device,
    )
    pd_target_ids = list(seed_ids) + list(pd_tail)
    pd_target_tail = pd_target_ids[len(seed_ids):]
    final_teacher = teacher_predictions(
        model,
        full_cache,
        full_mask,
        seed_ids,
        pd_target_tail,
        device=device,
    )
    final_teacher_self_agreement = mean(
        predicted == generated
        for predicted, generated in zip(
            final_teacher["predictions"], pd_target_tail
        )
    )

    golds = [row["answer"], *row.get("answer_aliases", [])]
    p_text = decode_text(tokenizer, p_tokens)
    pd_target_text = decode_text(tokenizer, pd_target_ids)
    p_em, p_f1 = answer_scores(p_text, golds)
    target_em, target_f1 = answer_scores(pd_target_text, golds)

    schedules = {}
    prompt_tokens = len(full_prompt)
    anchor_tokens = len(prefix) + len(question_suffix)
    for schedule in args.schedules:
        ranking = ranked_document_indices(
            schedule,
            question=row["question"],
            documents=documents,
            random_seed=args.seed + dataset_index,
        )
        selected_sets = stage_document_sets(
            ranking, documents, args.stage_fractions
        )
        masks = [
            pd_visibility_mask(
                len(prefix), documents, len(question_suffix), selected, device
            )
            for selected in selected_sets
        ]
        teacher_rows = []
        for stage_index, mask in enumerate(masks):
            teacher_rows.append(
                final_teacher
                if stage_index == len(masks) - 1
                else teacher_predictions(
                    model,
                    full_cache,
                    mask,
                    seed_ids,
                    pd_target_tail,
                    device=device,
                )
            )
        chains = {}
        for commit_window in args.commit_windows:
            chain = progressive_chain(
                model,
                full_cache,
                masks,
                seed_ids,
                commit_window=commit_window,
                draft_tokens=args.draft_tokens,
                max_new_tokens=args.max_new_tokens - len(seed_ids),
                eos_ids=eos_ids,
                device=device,
            )
            chain_tail = chain.pop("token_ids")
            if commit_window is None:
                # Batched verification and token-at-a-time greedy decoding can
                # choose different tokens for near-tied BF16 logits.  W=inf is
                # the exactness endpoint, so audit the speculative result and,
                # when needed, charge a final fixed-target replay before exposing
                # the output.  The target continuation was measured above with
                # the same full P-side cache and deterministic decode path.
                chain["pre_exact_replay_token_match"] = (
                    list(chain_tail) == list(pd_target_tail)
                )
                chain["exact_replay_required"] = float(
                    not chain["pre_exact_replay_token_match"]
                )
                chain["exact_replay_ms"] = (
                    pd_seed_forward_ms + pd_decode_ms
                    if chain["exact_replay_required"]
                    else 0.0
                )
                chain["draft_and_refresh_ms"] += chain["exact_replay_ms"]
                if chain["exact_replay_required"]:
                    chain_tail = list(pd_target_tail)
            chain_ids = list(seed_ids) + list(chain_tail)
            chain_text = decode_text(tokenizer, chain_ids)
            em, f1 = answer_scores(chain_text, golds)
            chain["answer"] = clean_answer(chain_text)
            chain["em"] = em
            chain["f1"] = f1
            chain["token_match_pd_target"] = chain_ids == pd_target_ids
            chain["raw_match_pd_target"] = chain_text == pd_target_text
            chain["normalized_match_pd_target"] = (
                normalize_answer(chain_text) == normalize_answer(pd_target_text)
            )
            name = "inf" if commit_window is None else str(commit_window)
            if commit_window is None:
                chain["lossless_gate_pass"] = chain["token_match_pd_target"]
            chains[name] = chain
        levels = level_metrics(
            teacher_rows,
            pd_target_tail,
            selected_sets,
            documents,
            args.stage_fractions,
        )
        for level, selected in zip(levels, selected_sets):
            selected_document_tokens = sum(
                len(documents[index].token_ids) for index in selected
            )
            level["actual_total_kv_fraction"] = (
                anchor_tokens + selected_document_tokens
            ) / prompt_tokens
        schedules[schedule] = {
            "ranking": ranking,
            "levels": levels,
            "chains": chains,
        }

    result = {
        "dataset_index": dataset_index,
        "id": row.get("id"),
        "hop_count": len(row.get("question_decomposition", [])),
        "question": row["question"],
        "gold_answers": golds,
        "document_count": len(documents),
        "prompt_tokens": prompt_tokens,
        "anchor_tokens": anchor_tokens,
        "document_tokens": sum(len(document.token_ids) for document in documents),
        "supporting_document_tokens": sum(
            len(document.token_ids) for document in documents if document.supporting
        ),
        "logical_bf16_kv_bytes": sum(
            key.numel() * key.element_size() + value.numel() * value.element_size()
            for key, value in full_cache
        ),
        "prefill_side": {
            "answer": clean_answer(p_text),
            "token_ids": p_tokens,
            "em": p_em,
            "f1": p_f1,
            "prefill_ms": p_prefill_ms,
            "decode_ms": p_decode_ms,
            "cache_producer_ms": cache_producer_ms,
        },
        "pd_target": {
            "answer": clean_answer(pd_target_text),
            "token_ids": pd_target_ids,
            "seed_token_ids": list(seed_ids),
            "em": target_em,
            "f1": target_f1,
            "seed_forward_ms": pd_seed_forward_ms,
            "decode_ms": pd_decode_ms,
            "token_match_prefill_side": pd_target_ids == p_tokens,
            "normalized_match_prefill_side": (
                normalize_answer(pd_target_text) == normalize_answer(p_text)
            ),
            "teacher_self_top1_agreement": final_teacher_self_agreement,
        },
        "schedules": schedules,
    }
    del full_cache, full_mask
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main():
    args = parse_args()
    output_path = Path(args.output)
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"{output_path} exists; pass --overwrite")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = load_jsonl(args.dataset)
    selected = stratified_indices(rows, args.sample_count, args.seed)
    shard_indices = selected[args.offset:args.offset + args.count]

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        local_files_only=True,
        torch_dtype=getattr(torch, args.dtype),
        attn_implementation="sdpa",
    ).eval().to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    with output_path.open("w", encoding="utf-8") as output:
        for local_index, dataset_index in enumerate(shard_indices):
            result = evaluate_example(
                model, tokenizer, rows[dataset_index], dataset_index, args, device
            )
            result["protocol"] = {
                "experiment": "pd-progressive-full-context-kv",
                "model": str(Path(args.model).resolve()),
                "dataset": str(Path(args.dataset).resolve()),
                "sample_count": args.sample_count,
                "sample_seed": args.seed,
                "shard_offset": args.offset,
                "shard_count": args.count,
                "stage_fractions": args.stage_fractions,
                "schedules": args.schedules,
                "commit_windows": [
                    "inf" if item is None else item for item in args.commit_windows
                ],
                "draft_tokens": args.draft_tokens,
                "seed_tokens_from_prefill_side": args.seed_tokens,
                "cache_source": "single contiguous full-prompt prefill",
                "mandatory_first_stream": "system/chat prefix plus query KV",
                "progressive_stream": "document token regions",
                "static_global_positions": True,
                "representation_refresh": "all generated token identities",
                "decoding": "greedy",
            }
            output.write(json.dumps(result, ensure_ascii=False) + "\n")
            output.flush()
            print(
                json.dumps(
                    {
                        "event": "example_complete",
                        "local": local_index + 1,
                        "count": len(shard_indices),
                        "dataset_index": dataset_index,
                        "hop_count": result["hop_count"],
                        "pd_target_gold_f1": result["pd_target"]["f1"],
                    }
                ),
                flush=True,
            )
    print(json.dumps({"event": "complete", "output": str(output_path)}))


if __name__ == "__main__":
    main()
