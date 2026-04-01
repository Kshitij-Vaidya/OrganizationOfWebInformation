'''
Part 1: Independent Retrieval

Goal:
    - Encode queries and tools independently
    - Compute similarity and retrieve top-k tools
    - Report Recall@1 and Recall@5
'''
# import os
# os.environ["TRANSFORMERS_OFFLINE"] = "1"

import argparse
import random
import numpy as np
import torch
from tqdm import tqdm

from utils import (
    get_queries_and_items,
    load_dense_encoder,
    encode_texts,
    build_bm25_index,
    bm25_rank,
    compute_recall_at_k,
)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate_bm25(test_queries, tools):
    tool_ids = list(tools.keys())
    tool_texts = [tools[t] for t in tool_ids]
    bm25 = build_bm25_index(tool_texts)
    tool_id_to_idx = {t: i for i, t in enumerate(tool_ids)}

    recall_1 = 0.0
    recall_5 = 0.0

    for sample in tqdm(test_queries, desc="BM25"):
        query = sample["text"]
        gold_tool = sample["gold_tool_name"]
        gold_idx = tool_id_to_idx[gold_tool]

        ranked, _ = bm25_rank(query, bm25)
        recalls = compute_recall_at_k(ranked, gold_idx, k_list=(1, 5))
        recall_1 += recalls[1]
        recall_5 += recalls[5]

    total = max(len(test_queries), 1)
    return recall_1 / total, recall_5 / total


def evaluate_dense(model_name, test_queries, tools, device, batch_size=32, dtype=torch.float32):
    tool_ids = list(tools.keys())
    tool_texts = [tools[t] for t in tool_ids]
    tool_id_to_idx = {t: i for i, t in enumerate(tool_ids)}

    tokenizer, model = load_dense_encoder(model_name=model_name, device=device, dtype=dtype)

    tool_embs = encode_texts(tool_texts, tokenizer, model, device, batch_size=batch_size)
    tool_embs = tool_embs.to(device)

    recall_1 = 0.0
    recall_5 = 0.0

    for sample in tqdm(test_queries, desc=model_name):
        query = sample["text"]
        gold_tool = sample["gold_tool_name"]
        gold_idx = tool_id_to_idx[gold_tool]

        query_emb = encode_texts([query], tokenizer, model, device, batch_size=1)
        query_emb = query_emb.to(device)
        scores = torch.matmul(tool_embs, query_emb[0])
        ranked = torch.argsort(scores, descending=True).tolist()

        recalls = compute_recall_at_k(ranked, gold_idx, k_list=(1, 5))
        recall_1 += recalls[1]
        recall_5 += recalls[5]

    total = max(len(test_queries), 1)
    return recall_1 / total, recall_5 / total


parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=64)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--model_msmarco', type=str, default='msmarco-MiniLM')
parser.add_argument('--model_uae', type=str, default='UAE-large-v1')
parser.add_argument('--device', type=str, default=None)
args = parser.parse_args()


if __name__ == '__main__':
    seed_all(args.seed)
    device = args.device
    if device is None:
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    _, test_queries, tools = get_queries_and_items()

    r1, r5 = evaluate_bm25(test_queries, tools)
    print(f"BM25 Recall@1: {r1:.4f} | Recall@5: {r5:.4f}")

    r1, r5 = evaluate_dense(
        model_name=args.model_msmarco,
        test_queries=test_queries,
        tools=tools,
        device=device,
        batch_size=args.batch_size,
        dtype=torch.float32,
    )
    print(f"msmarco-MiniLM Recall@1: {r1:.4f} | Recall@5: {r5:.4f}")

    r1, r5 = evaluate_dense(
        model_name=args.model_uae,
        test_queries=test_queries,
        tools=tools,
        device=device,
        batch_size=args.batch_size,
        dtype=torch.float32,
    )
    print(f"UAE-large-v1 Recall@1: {r1:.4f} | Recall@5: {r5:.4f}")
