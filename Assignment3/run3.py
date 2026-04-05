'''
Part 3: Retrieval Heads

Goal:
    - Select a subset of attention heads using training data
    - Use ONLY those heads to rank tools at test time
'''

import os
# os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
import argparse
import time
import random
import numpy as np
import torch
import gc
from tqdm import tqdm

from utils import load_model_tokenizer, PromptUtils, get_queries_and_items

from code3 import select_retrieval_heads

# -------------------------
# Do NOT change
# -------------------------
def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def query_to_docs_attention_heads(attentions, query_span, doc_spans, selected_heads, target_device):
    # TODO 2: Head-based scoring
    doc_scores = torch.zeros(len(doc_spans), device=target_device)
    query_start, query_end = query_span
 
    for (layer_idx, head_idx) in selected_heads:
        # attentions[layer_idx]: [1, num_heads, N, N]
        head_attn = attentions[layer_idx][0, head_idx].to(target_device)   # [N, N]
 
        for doc_idx, (doc_start, doc_end) in enumerate(doc_spans):
            # sum attention from all query tokens to all tokens of this doc
            doc_scores[doc_idx] += head_attn[query_start:query_end, doc_start:doc_end].sum()
 
    # average over the number of selected heads
    if len(selected_heads) > 0:
        doc_scores /= len(selected_heads)
 
    return doc_scores


def get_query_span(putils, question, input_ids):
    # TODO 3: Query span
    query_prompt    = f"Query: {question}\nCorrect tool_id:"
    query_token_len = len(
        putils.tokenizer(query_prompt, add_special_tokens=False).input_ids
    )
    total_len   = input_ids.shape[0]
    query_end   = total_len - putils.prompt_suffix_length
    query_start = query_end - query_token_len
    return (query_start, query_end)


parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=64)
parser.add_argument('--model', type=str, default="unsloth/Llama-3.2-1B-Instruct")
parser.add_argument('--max_heads', type=int, default=20)
parser.add_argument('--train_samples', type=int, default=200)
parser.add_argument("--debug", action="store_true")
args = parser.parse_args()


if __name__ == '__main__':

    seed_all(args.seed)
    model_name = args.model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer, model = load_model_tokenizer(
        model_name=model_name,
        device=device,
        dtype=torch.float16,
        device_map="auto" if device.startswith("cuda") else None,
    )
    device = model.device

    train_queries, test_queries, tools = get_queries_and_items()
    print("\n[Phase 1] Selecting retrieval heads...")

    selected_heads = select_retrieval_heads(
        train_queries=train_queries[:args.train_samples],
        model=model,
        tokenizer=tokenizer,
        tools=tools,
        device=device,
        max_heads=args.max_heads
    )

    print(f"Selected {len(selected_heads)} heads")
    print(selected_heads)

    print("\n[Phase 2] Evaluating on test set...")
    recall_at_1 = 0
    correct_at_1 = 0
    correct_at_5 = 0
    total = 0

    for qix in tqdm(range(len(test_queries))):

        sample = test_queries[qix]
        question = sample["text"]
        gold_tool_name = sample["gold_tool_name"]

        # --------------------
        # Do Not change the shuffling here
        # --------------------
        shuffled_keys = list(tools.keys())
        random.shuffle(shuffled_keys)

        putils = PromptUtils(
            tokenizer=tokenizer,
            doc_ids=shuffled_keys,
            dict_all_docs=tools,
        )

        item_spans = putils.doc_spans
        map_docname_id = putils.dict_doc_name_id

        gold_tool_id = map_docname_id[gold_tool_name]

        prompt = putils.create_prompt(query=question)
        inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        inputs = inputs.to(device)

        input_ids = inputs.input_ids[0]
        query_span = get_query_span(putils, question, input_ids)

        with torch.no_grad():
            attentions = model(**inputs).attentions


        doc_scores = query_to_docs_attention_heads(
            attentions,
            query_span,
            item_spans,
            selected_heads,
            target_device=device,
        )

        if device.type == "cuda":
            del attentions, inputs
            torch.cuda.empty_cache()
            gc.collect()


        # rank docs by descending score; gold_rank is 1-indexed
        ranked_docs = torch.argsort(doc_scores, descending=True).tolist()
        gold_rank   = ranked_docs.index(gold_tool_id) + 1   # 1-indexed
 
        # TODO: measure the recall@1, recall@5
        total        += 1
        correct_at_1 += 1 if gold_rank == 1 else 0
        correct_at_5 += 1 if gold_rank <= 5 else 0
 
    recall_at_1 = correct_at_1 / total
    recall_at_5 = correct_at_5 / total
    print(f"\nRecall@1 (selected heads): {recall_at_1:.4f}")
    print(f"Recall@5 (selected heads): {recall_at_5:.4f}")