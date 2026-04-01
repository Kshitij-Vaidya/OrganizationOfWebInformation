'''
Part 2: are we lost in the middle?

Goal:
    - visualize the attention from the query to gold document based on the distance between them
    - use attention as a metric to rank documents for a query 
'''
import gc
import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"
import argparse
import json 
import time
import pandas as pd
from tqdm import tqdm
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
import os
from utils import load_model_tokenizer, PromptUtils, get_queries_and_items

# -------------------------
# Do NOT change
# -------------------------
def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed) 
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def query_to_docs_attention(attentions, query_span, doc_spans):
    """
    attentions: tuple(num_layers) of [1, heads, N, N]
    query_span: (start, end)
    doc_spans: list of (start, end)
    """
    doc_scores = torch.zeros(len(doc_spans), device=attentions[0].device)
    
    # TODO 1: implement to get final query to doc attention stored in doc_scores
    query_start, query_end = query_span
    
    for layer_idx, attention_matrix in enumerate(attentions):
        # attention_matrix: [1, heads, N, N]
        attention_matrix = attention_matrix[0]  # [heads, N, N]
        
        # Average attention across all heads
        avg_attention = attention_matrix.mean(dim=0)  # [N, N]
        
        # Get attention from query tokens to each document
        for doc_idx, (doc_start, doc_end) in enumerate(doc_spans):
            # Sum attention from query span to this document span
            query_to_doc_attn = avg_attention[query_start:query_end, doc_start:doc_end].sum()
            doc_scores[doc_idx] += query_to_doc_attn
    
    # Average across layers
    doc_scores /= len(attentions)
    
    return doc_scores


def analyze_gold_attention(result, save_path="plot2/gold_attention_plot.png"):
    # TODO 2: visualize graph
    """
    input -> result: list of dicts with keys:
                        - gold_position
                        - gold_score
                        - gold_rank
    GOAL: Using the results data, generate a visualization that shows how attention to the gold tool varies with its position in the prompt.
    """
 
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    positions = [r["gold_position"] for r in result]
    scores    = [r["gold_score"]    for r in result]
    ranks     = [r["gold_rank"]     for r in result]
 
    # --- bin-level aggregation for cleaner trend lines ---
    max_pos = max(positions) + 1
    bin_size = max(1, max_pos // 20)          # ~20 bins regardless of pool size
    bins = list(range(0, max_pos + bin_size, bin_size))
 
    def bin_mean(values, pos_list, bins):
        means, centers = [], []
        for i in range(len(bins) - 1):
            lo, hi = bins[i], bins[i + 1]
            vals = [v for v, p in zip(values, pos_list) if lo <= p < hi]
            if vals:
                means.append(np.mean(vals))
                centers.append((lo + hi) / 2)
        return centers, means
 
    bin_centers_s, bin_scores = bin_mean(scores, positions, bins)
    bin_centers_r, bin_ranks  = bin_mean(ranks,  positions, bins)
 
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Lost-in-the-Middle: Attention vs Gold Tool Position", fontsize=13)
 
    # --- Plot 1: attention score vs position ---
    axes[0].scatter(positions, scores, alpha=0.25, s=20, color='steelblue', label='per-query')
    axes[0].plot(bin_centers_s, bin_scores, color='navy', linewidth=2, marker='o', label='bin mean')
    axes[0].set_xlabel("Gold Tool Position in Prompt")
    axes[0].set_ylabel("Attention Score (query → gold tool)")
    axes[0].set_title("Attention Score vs Position")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
 
    # --- Plot 2: rank vs position ---
    axes[1].scatter(positions, ranks, alpha=0.25, s=20, color='darkorange', label='per-query')
    axes[1].plot(bin_centers_r, bin_ranks, color='saddlebrown', linewidth=2, marker='o', label='bin mean')
    axes[1].set_xlabel("Gold Tool Position in Prompt")
    axes[1].set_ylabel("Rank of Gold Tool (lower = better)")
    axes[1].set_title("Gold Tool Rank vs Position")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
 
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()
    print(f"Plot saved to {save_path}")

def get_query_span(putils, question, input_ids):
    # TODO 3: Query span
    """
    Identify the token span corresponding to the query in the full tokenised prompt.
 
    The prompt structure (from PromptUtils.create_prompt) is:
        prompt_prefix | all_docs | sep | add_text1 | sep | query_prompt | prompt_suffix
 
    The query_prompt is: "Query: {question}\\nCorrect tool_id:"
    It sits just before the assistant header (prompt_suffix).
 
    Strategy: measure the length of query_prompt tokens, then count back
    from the end of the sequence (minus the suffix).
    """
    query_prompt = f"Query: {question}\nCorrect tool_id:"
    query_token_len = len(
        putils.tokenizer(query_prompt, add_special_tokens=False).input_ids
    )
    total_len  = input_ids.shape[0]
    query_end   = total_len - putils.prompt_suffix_length
    query_start = query_end - query_token_len
    return (query_start, query_end)

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=64)
parser.add_argument('--model', type=str, default="meta-llama/Llama-3.2-1B-Instruct")
parser.add_argument('--top_heads', type=int, default=20)
parser.add_argument("--debug", action="store_true", help="Enable debug mode")
args = parser.parse_args()


if __name__ == '__main__':
    seed_all(seed=args.seed)
    model_name = args.model
    device = "cuda:0"
    
    tokenizer, model = load_model_tokenizer(model_name=model_name, device=device, dtype=torch.float16)
    num_heads = model.config.num_attention_heads
    num_layers = model.config.num_hidden_layers
    d = getattr(model.config, "head_dim", model.config.hidden_size // model.config.num_attention_heads)
    num_key_value_groups = num_heads//model.config.num_key_value_heads
    softmax_scaling=d**-0.5
    train_queries, test_queries, tools = get_queries_and_items()
 

    print("---- debug print start ----")
    print(f"seed: {args.seed}, model: {model_name}")
    print("model.config._attn_implementation: ", model.config._attn_implementation)

    dict_head_freq = {}
    df_data = []
    avg_latency = []
    count = 0
    start_time = time.time()
    results = []
    for qix in tqdm(range(len(test_queries))):
        sample =  test_queries[qix]
        qid = sample["qid"]
        question = sample["text"]
        gold_tool_name = sample["gold_tool_name"]

        # --------------------
        # Do Not change the shuffling here
        # --------------------
        num_dbs = len(tools)
        shuffled_keys = list(tools.keys())
        random.shuffle(shuffled_keys)

        putils = PromptUtils(
            tokenizer=tokenizer, 
            doc_ids=shuffled_keys, 
            dict_all_docs=tools,
            )
        item_spans = putils.doc_spans
        doc_lengths = putils.doc_lengths
        map_docname_id = putils.dict_doc_name_id
        map_id_docname = {v:k for k, v in map_docname_id.items()}
        db_lengths_pt = torch.tensor(doc_lengths, device=device)
        
        gold_tool_id = map_docname_id[gold_tool_name]

        prompt = putils.create_prompt(query=question)
        inputs = tokenizer(prompt, return_tensors = "pt", add_special_tokens = False).to(device)

        if args.debug and qix < 5:
            ip_ids = inputs.input_ids[0].cpu()
            print("-------"*5)
            print(prompt)
            print("-------"*5)
            print("---- doc1 ----")
            print(tokenizer.decode(ip_ids[item_spans[0][0]: item_spans[0][1]]))
            print("---- lastdoc ----")
            print(tokenizer.decode(ip_ids[item_spans[-1][0]: item_spans[-1][1]]))
            print("-------"*5)


        with torch.no_grad():
            attentions = model(**inputs).attentions
            '''
                attentions - tuple of length = # layers
                attentions[0].shape - [1, h, N, N] : first layer's attention matrix for h heads
            '''
        
        input_ids = inputs.input_ids[0]
        query_span = get_query_span(putils, question, input_ids)

        doc_scores = query_to_docs_attention(attentions, query_span, item_spans)

        # TODO: find gold_rank- rank of gold tool in doc_scores
        # TODO: find gold_score - score of gold tool
        gold_score = doc_scores[gold_tool_id].item()
        gold_rank = (doc_scores > gold_score).sum().item() + 1
        
        results.append({
            "qid": qid,
            "gold_position": gold_tool_id,
            "gold_score": gold_score,
            "gold_rank": gold_rank
        })

        # TODO: calucalte recall@1, recall@5 metric and print at end of loop
        if qix == len(test_queries) - 1:
            recall_at_1 = sum(1 for r in results if r["gold_rank"] == 1) / len(results)
            recall_at_5 = sum(1 for r in results if r["gold_rank"] <= 5) / len(results)
            print(f"Recall@1: {recall_at_1:.4f}")
            print(f"Recall@5: {recall_at_5:.4f}")

    analyze_gold_attention(results)

    