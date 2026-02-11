from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence
import numpy as np
from scipy import sparse
# from sklearn.decomposition import TruncatedSVD

from utils import (
    build_vocabulary_index,
    find_nearest_neighbors,
    load_data,
    save_embeddings,
    tokenize_corpus,
    gen_term_document_matrix,
    truncated_svd,
    save_svd_embeddings,
)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train GloVe embeddings for Task 1")
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/updated_vocab_document_dict.json",
        help="Path to the CC-News vocabulary-document mapping JSON.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="artifacts/task1",
        help="Directory where embeddings and metadata will be stored.",
    )
    parser.add_argument(
        "--d",
        type=int,
        default=200,
        help="Dimensionality of the learned embeddings.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=17,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--neighbours",
        nargs="*",
        default=None,
        help="Optional list of vocabulary tokens for nearest neighbour inspection.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of neighbors to report when --neighbours is provided.",
    )
    return parser.parse_args()

def run_svd(args: argparse.Namespace) -> None:
    # vocab, corpus = load_data(args.data_path)
    # vocab_list, vocab_to_id = build_vocabulary_index(vocab)
    # tokenized_corpus = tokenize_corpus(corpus, vocab_filter=set(vocab_list))

    # # Placeholder for co-occurrence matrix construction and SVD
    # # cooccurrence_matrix = build_cooccurrence_matrix(tokenized_corpus, vocab_to_id, window_size=args.window_size)
    # # embeddings = perform_svd(cooccurrence_matrix, vector_size=args.d)

    # # For demonstration, we'll create random embeddings
    # np.random.seed(args.seed)
    # embeddings = np.random.rand(len(vocab_list), args.d)

    # output_dir = Path(args.output_dir)
    # output_dir.mkdir(parents=True, exist_ok=True)
    # save_embeddings(embeddings, vocab_list, output_dir / "embeddings.json")

    # if args.neighbours:
    #     for token in args.neighbours:
    #         if token in vocab_to_id:
    #             token_id = vocab_to_id[token]
    #             neighbors = find_nearest_neighbors(embeddings, token_id, top_k=args.top_k)
    #             print(f"Nearest neighbors for '{token}': {[vocab_list[n] for n in neighbors]}")
    #         else:
    #             print(f"Token '{token}' not found in vocabulary.")
    print("Loading data and building vocabulary...")
    vocab, corpus = load_data(args.data_path)
    vocab_list, vocab_to_id = build_vocabulary_index(vocab)
    token_sequences = tokenize_corpus(corpus, set(vocab_list))

    print("Constructing term-document matrix...")
    # Term-document matrix construction and SVD-based embedding learning
    term_document_matrix: sparse.csr_matrix = gen_term_document_matrix(token_sequences, vocab_to_id)

    print("Performing truncated SVD...")
    embeddings = truncated_svd(term_document_matrix, vector_size=args.d, seed=args.seed)

    save_svd_embeddings(output_dir=args.output_dir, vocab=vocab_list, embeddings=embeddings, config={"vector_size": args.d}, prefix=f"svd_d{args.d}")
    print(f"SVD embeddings saved to {args.output_dir}")

    
    if args.neighbours:
        neighbor_report = compile_neighbors(args.neighbours, vocab_to_id, embeddings, args.top_k)
        output_path = Path(args.output_dir)
        with (output_path / f"svd_d{args.d}_neighbors.json").open("w", encoding="utf-8") as handle:
            json.dump(neighbor_report, handle, indent=2)
        print("Nearest neighbor report written to", output_path / f"svd_d{args.d}_neighbors.json")

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

    result = find_nearest_neighbors(present_queries, vocab_to_id, embeddings, top_k=top_k)
    for token in missing_queries:
        result[token] = []
    return result


if __name__ == "__main__":
    args = parse_args()
    run_svd(args)