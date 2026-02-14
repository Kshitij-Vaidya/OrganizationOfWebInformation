from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence
import numpy as np
from scipy import sparse
# from sklearn.decomposition import TruncatedSVD

from utils import (
    build_vocabulary_index,
    find_nearest_neighbours,
    load_data,
    tokenize_corpus,
    gen_term_document_matrix,
    truncated_svd,
    save_svd_embeddings,
)

DATA_PATH = "data/updated_vocab_document_dict.json"
OUTPUT_DIR = "output/task2"
NEIGHBOURS = ["data", "cities", "temperature"]
SEED = 17
VECTOR_SIZE = 200
TOP_K = 5

def run_svd() -> None:
    print("Loading data and building vocabulary...")
    vocab, corpus = load_data(DATA_PATH)
    vocab_list, vocab_to_id = build_vocabulary_index(vocab)
    token_sequences = tokenize_corpus(corpus, set(vocab_list))

    print("Constructing term-document matrix...")
    # Term-document matrix construction and SVD-based embedding learning
    term_document_matrix: sparse.csr_matrix = gen_term_document_matrix(token_sequences, vocab_to_id)

    print("Performing truncated SVD...")
    embeddings = truncated_svd(term_document_matrix, vector_size=VECTOR_SIZE, seed=SEED)

    save_svd_embeddings(output_dir=OUTPUT_DIR, vocab=vocab_list, embeddings=embeddings, config={"vector_size": VECTOR_SIZE}, prefix=f"svd_d{VECTOR_SIZE}")
    print(f"SVD embeddings saved to {OUTPUT_DIR}")

    
    if NEIGHBOURS:
        neighbor_report = compile_neighbors(NEIGHBOURS, vocab_to_id, embeddings, TOP_K)
        output_path = Path(OUTPUT_DIR)
        with (output_path / f"svd_d{VECTOR_SIZE}_neighbors.json").open("w", encoding="utf-8") as handle:
            json.dump(neighbor_report, handle, indent=2)
        print("Nearest neighbor report written to", output_path / f"svd_d{VECTOR_SIZE}_neighbors.json")

def compile_neighbors(
    query_tokens: Sequence[str],
    vocab_to_id: dict[str, int],
    embeddings,
    top_k: int,
) -> dict[str, list[tuple[str, float]]]:
    present_queries = [token for token in query_tokens if token in vocab_to_id]
    missing_queries = [token for token in query_tokens if token not in vocab_to_id]

    if missing_queries:
        print("Tokens not found in vocabulary:", ", ".join(sorted(set(missing_queries))))

    result = find_nearest_neighbours(present_queries, vocab_to_id, embeddings, top_k=top_k)
    for token in missing_queries:
        result[token] = []
    return result


if __name__ == "__main__":
    run_svd()