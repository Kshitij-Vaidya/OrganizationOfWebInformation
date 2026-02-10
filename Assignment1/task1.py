from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from utils import (
    GloVeConfig,
    GloVeModel,
    build_cooccurrence_matrix,
    build_vocabulary_index,
    cooccurrence_items,
    find_nearest_neighbors,
    load_data,
    save_embeddings,
    tokenize_corpus,
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
        "--window-size",
        type=int,
        default=5,
        help="Number of tokens to consider on each side during co-occurrence counting.",
    )
    parser.add_argument(
        "--vector-size",
        type=int,
        default=200,
        help="Dimensionality of the learned embeddings.",
    )
    parser.add_argument(
        "--x-max",
        type=float,
        default=100.0,
        help="Cut-off threshold for the GloVe weighting function.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.75,
        help="Exponent applied to (x/x_max) when computing the weighting function.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.05,
        help="Learning rate used by the optimizer.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=25,
        help="Number of passes over the co-occurrence statistics.",
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
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Disable per-epoch loss logging during training.",
    )
    return parser.parse_args()


def run_training(args: argparse.Namespace) -> None:
    vocab, corpus = load_data(args.data_path)
    vocab_list, vocab_to_id = build_vocabulary_index(vocab)

    token_sequences = tokenize_corpus(corpus, set(vocab_list))
    cooccurrence = build_cooccurrence_matrix(token_sequences, vocab_to_id, args.window_size)
    cooccurrence_triplets = cooccurrence_items(cooccurrence)

    config = GloVeConfig(
        vector_size=args.vector_size,
        x_max=args.x_max,
        alpha=args.alpha,
        learning_rate=args.learning_rate,
        seed=args.seed,
    )
    model = GloVeModel(vocab_size=len(vocab_list), config=config)
    history = model.fit(cooccurrence_triplets, epochs=args.epochs, verbose=not args.quiet)

    embeddings = model.get_embeddings()
    save_embeddings(args.output_dir, vocab_list, embeddings, history, config)

    if args.neighbors:
        neighbor_report = compile_neighbors(args.neighbors, vocab_to_id, embeddings, args.top_k)
        output_path = Path(args.output_dir)
        with (output_path / "glove_neighbors.json").open("w", encoding="utf-8") as handle:
            json.dump(neighbor_report, handle, indent=2)
        print("Nearest neighbor report written to", output_path / "glove_neighbors.json")


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
    run_training(parse_args())
