from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np

from utils import (
    GloVeConfig,
    GloVeModel,
    build_cooccurrence_matrix,
    build_vocabulary_index,
    cooccurrence_items,
    find_nearest_neighbours,
    load_data,
    save_embeddings,
    tokenize_corpus,
)

DATA_PATH = "data/updated_vocab_document_dict.json"
OUTPUT_DIR = "output/task1"
NEIGHBOURS = ["data", "cities", "temperature"]
SEED = 17
WINDOW_SIZE = 5
VECTOR_SIZE = 200
LEARNING_RATE = 0.05
EPOCHS = 25
TOP_K = 5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train GloVe embeddings for Task 1")

    parser.add_argument(
        "--run",
        action="store_true",
        help="Execute training only when this flag is provided.",
    )
    return parser.parse_args()


def run_training(args: argparse.Namespace) -> None:
    vocab, corpus = load_data(DATA_PATH)
    vocab_list, vocab_to_id = build_vocabulary_index(vocab)

    token_sequences = tokenize_corpus(corpus, set(vocab_list))
    cooccurrence = build_cooccurrence_matrix(token_sequences, vocab_to_id, WINDOW_SIZE)
    cooccurrence_triplets = cooccurrence_items(cooccurrence)

    config = GloVeConfig(
        vector_size=VECTOR_SIZE,
        x_max=100,
        alpha=0.75,
        learning_rate=LEARNING_RATE,
        seed=SEED,
    )
    model = GloVeModel(vocab_size=len(vocab_list), config=config)
    history = model.fit(cooccurrence_triplets, epochs=EPOCHS)

    embeddings = model.get_embeddings()
    save_embeddings(OUTPUT_DIR, vocab_list, embeddings, history, config)

    # if NEIGHBOURS:
    #     neighbor_report = compile_neighbours(NEIGHBOURS, vocab_to_id, embeddings, TOP_K)
    #     output_path = Path(OUTPUT_DIR)
    #     with (output_path / "glove_neighbours.json").open("w", encoding="utf-8") as handle:
    #         json.dump(neighbor_report, handle, indent=2)
    #     print("Nearest neighbor report written to", output_path / "glove_neighbours.json")


def compile_neighbours(
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


def run_experiments(args: argparse.Namespace) -> None:
    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(parents=True, exist_ok=True)

    vocab, corpus = load_data(DATA_PATH)
    vocab_list, vocab_to_id = build_vocabulary_index(vocab)
    token_sequences = tokenize_corpus(corpus, set(vocab_list))

    window_sizes = [2, 5, 10]
    learning_rates = [0.05, 0.1]

    sweep_dimension = 200
    experiments: list[dict[str, float | int | str]] = []
    histories: list[dict[str, object]] = []
    # For the fixed dimension of 200, run a grid search over the window sizes and learning rates to determine the best hyperparameters
    for window_size in window_sizes:
        cooccurrence = build_cooccurrence_matrix(token_sequences, vocab_to_id, window_size)
        cooccurrence_triplets = cooccurrence_items(cooccurrence)

        for learning_rate in learning_rates:
            config = GloVeConfig(
                vector_size=sweep_dimension,
                x_max=100,
                alpha=0.75,
                learning_rate=learning_rate,
                seed=SEED,
            )
            model = GloVeModel(vocab_size=len(vocab_list), config=config)

            start_time = time.perf_counter()
            history = model.fit(cooccurrence_triplets, epochs=25)
            latency = time.perf_counter() - start_time

            embeddings = model.get_embeddings()
            run_id = f"w{window_size}_lr{learning_rate}_d{sweep_dimension}"
            save_embeddings(output_path, vocab_list, embeddings, history, config, prefix=f"glove_{run_id}")

            experiments.append(
                {
                    "run_id": run_id,
                    "window_size": window_size,
                    "learning_rate": learning_rate,
                    "epochs": 25,
                    "vector_size": sweep_dimension,
                    "final_loss": float(history[-1]),
                    "latency_seconds": float(latency),
                }
            )
            histories.append({"run_id": run_id, "loss_history": history})

    summary_path = output_path / "task1_experiments_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump({"experiments": experiments}, handle, indent=2)

    plt.figure(figsize=(10, 6))
    for record in histories:
        plt.plot(record["loss_history"], label=record["run_id"])
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("GloVe Training Loss Curves (Task 1)")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path / "task1_loss_curves_d200.png", dpi=200)
    plt.close()

    # Select the best hyperparameters from above and run a search over the dimension of the glove vectors

    best_run = min(experiments, key=lambda item: item["final_loss"])
    best_window = int(best_run["window_size"])
    best_learning_rate = float(best_run["learning_rate"])
    best_epochs = int(best_run["epochs"])

    selected_config_path = output_path / "task1_selected_hyperparameters.json"
    with selected_config_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "window_size": best_window,
                "learning_rate": best_learning_rate,
                "epochs": best_epochs,
                "x_max": 100,
                "alpha": 0.75,
            },
            handle,
            indent=2,
        )

    final_dimensions = [50, 100, 200, 300]
    dimension_histories: list[dict[str, object]] = []

    cooccurrence = build_cooccurrence_matrix(token_sequences, vocab_to_id, best_window)
    cooccurrence_triplets = cooccurrence_items(cooccurrence)

    for dimension in final_dimensions:
        config = GloVeConfig(
            vector_size=dimension,
            x_max=100,
            alpha=0.75,
            learning_rate=best_learning_rate,
            seed=SEED,
        )
        model = GloVeModel(vocab_size=len(vocab_list), config=config)

        start_time = time.perf_counter()
        history = model.fit(cooccurrence_triplets, epochs=best_epochs)
        latency = time.perf_counter() - start_time

        embeddings = model.get_embeddings()
        run_id = f"w{best_window}_lr{best_learning_rate}_e{best_epochs}_d{dimension}"
        save_embeddings(output_path, vocab_list, embeddings, history, config, prefix=f"glove_{run_id}")

        experiments.append(
            {
                "run_id": run_id,
                "window_size": best_window,
                "learning_rate": best_learning_rate,
                "epochs": best_epochs,
                "vector_size": dimension,
                "final_loss": float(history[-1]),
                "latency_seconds": float(latency),
            }
        )
        dimension_histories.append({"run_id": run_id, "loss_history": history})

        if dimension == 200:
            neighbour_vocab_path = output_path / f"glove_{run_id}_vocab.json"
            neighbour_vectors_path = output_path / f"glove_{run_id}_vectors.npy"
            if neighbour_vocab_path.exists() and neighbour_vectors_path.exists():
                with neighbour_vocab_path.open("r", encoding="utf-8") as handle:
                    best_vocab_list = json.load(handle)
                embeddings = np.load(neighbour_vectors_path)
                best_vocab_to_id = {token: idx for idx, token in enumerate(best_vocab_list)}

                if NEIGHBOURS:
                    neighbor_queries = list(NEIGHBOURS)
                else:
                    neighbor_queries = best_vocab_list[:3]

                neighbour_report = compile_neighbours(
                    neighbor_queries, best_vocab_to_id, embeddings, TOP_K
                )
                with (output_path / "glove_neighbours.json").open("w", encoding="utf-8") as handle:
                    json.dump(neighbour_report, handle, indent=2)

    plt.figure(figsize=(10, 6))
    for record in dimension_histories:
        plt.plot(record["loss_history"], label=record["run_id"])
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("GloVe Training Loss Curves (Best Hyperparameters, Varying d)")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path / "task1_loss_curves_by_dimension.png", dpi=200)
    plt.close()

    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump({"experiments": experiments}, handle, indent=2)



if __name__ == "__main__":
    args = parse_args()
    # Training argument was 
    if args.run:
        run_training(args)
    else:
        run_experiments(args)

