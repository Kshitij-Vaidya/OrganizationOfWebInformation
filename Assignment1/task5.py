from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from datasets import load_dataset
from sklearn.metrics import classification_report, f1_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from utils import (
    build_vocabulary_index,
    find_nearest_neighbours,
    gen_term_document_matrix,
    gen_tfidf_term_document_matrix,
    load_data,
    save_svd_embeddings,
    tokenize_corpus,
    truncated_svd,
)

DATA_PATH = "data/updated_vocab_document_dict.json"
OUTPUT_DIR = "output/task5"
NEIGHBOURS = ["data", "temperature", "cities", "apple", "fast"]
TOP_K = 5
DATASET = "conll2003"
EPOCHS = 10
BATCH_SIZE = 1024
HIDDEN_SIZE = 128
LEARNING_RATE = 1e-3
SEED = 17
DIM = 200

class TokenMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_labels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_labels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def build_token_dataset(
    split,
    vocab_to_id: dict[str, int],
    label_map: dict[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    if "<UNK>" not in vocab_to_id:
        raise ValueError("<UNK> missing from vocab_to_id. Ensure OOV handling before dataset build.")
    token_ids: list[int] = []
    label_ids: list[int] = []
    for record in split:
        for token, ner in zip(record["tokens"], record["ner_tags"]):
            token_ids.append(vocab_to_id.get(token.lower(), vocab_to_id["<UNK>"]))
            label_ids.append(label_map[int(ner)])
    return np.array(token_ids, dtype=np.int64), np.array(label_ids, dtype=np.int64)


def train_and_eval_mlp(
    embeddings: np.ndarray,
    vocab_to_id: dict[str, int],
    output_dir: Path,
    prefix: str,
) -> dict[str, object]:
    if "<UNK>" not in vocab_to_id or vocab_to_id["<UNK>"] >= embeddings.shape[0]:
        mean = embeddings.mean(axis=0)
        std = embeddings.std(axis=0)
        oov_vec = mean + 0.05 * std * np.random.randn(*mean.shape)
        vocab_to_id["<UNK>"] = embeddings.shape[0]
        embeddings = np.vstack([embeddings, oov_vec])
    dataset = load_dataset(DATASET)
    ner_feature = dataset["train"].features["ner_tags"]
    label_names = ner_feature.feature.names
    label_map = {idx: idx for idx in range(len(label_names))}

    train_ids, train_labels = build_token_dataset(dataset["train"], vocab_to_id, label_map)
    dev_ids, dev_labels = build_token_dataset(dataset["validation"], vocab_to_id, label_map)
    test_ids, test_labels = build_token_dataset(dataset["test"], vocab_to_id, label_map)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    embedding_tensor = torch.tensor(embeddings, dtype=torch.float32, device=device)
    x_train = embedding_tensor[torch.tensor(train_ids, device=device)]
    y_train = torch.tensor(train_labels, dtype=torch.long, device=device)
    x_dev = embedding_tensor[torch.tensor(dev_ids, device=device)]
    y_dev = torch.tensor(dev_labels, dtype=torch.long, device=device)
    x_test = embedding_tensor[torch.tensor(test_ids, device=device)]
    y_test = torch.tensor(test_labels, dtype=torch.long, device=device)

    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    dev_loader = DataLoader(TensorDataset(x_dev, y_dev), batch_size=BATCH_SIZE)
    test_loader = DataLoader(TensorDataset(x_test, y_test), batch_size=BATCH_SIZE)

    model = TokenMLP(embeddings.shape[1], HIDDEN_SIZE, len(label_names)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    for _ in range(1, EPOCHS + 1):
        model.train()
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

    def evaluate(loader: DataLoader) -> tuple[list[int], list[int]]:
        model.eval()
        preds: list[int] = []
        labels: list[int] = []
        with torch.no_grad():
            for batch_x, batch_y in loader:
                logits = model(batch_x)
                batch_preds = logits.argmax(dim=-1)
                preds.extend(batch_preds.detach().cpu().tolist())
                labels.extend(batch_y.detach().cpu().tolist())
        return labels, preds

    dev_true, dev_pred = evaluate(dev_loader)
    test_true, test_pred = evaluate(test_loader)

    dev_report = classification_report(
        dev_true,
        dev_pred,
        labels=list(range(len(label_names))),
        target_names=label_names,
        output_dict=True,
        zero_division=0,
    )
    test_report = classification_report(
        test_true,
        test_pred,
        labels=list(range(len(label_names))),
        target_names=label_names,
        output_dict=True,
        zero_division=0,
    )

    metrics = {
        "embedding_prefix": prefix,
        "embedding_dimension": embeddings.shape[1],
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "hidden_size": HIDDEN_SIZE,
        "learning_rate": LEARNING_RATE,
        "dev": {"weighted_f1": f1_score(dev_true, dev_pred, average="weighted"), "classification_report": dev_report},
        "test": {"weighted_f1": f1_score(test_true, test_pred, average="weighted"), "classification_report": test_report},
    }

    metrics_path = output_dir / f"task5_mlp_{prefix}_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    return metrics


def compile_neighbors(
    query_tokens: Sequence[str],
    vocab_to_id: dict[str, int],
    embeddings: np.ndarray,
    top_k: int,
) -> dict[str, list[tuple[str, float]]]:
    return find_nearest_neighbours(query_tokens, vocab_to_id, embeddings, top_k=top_k)


def main() -> None:
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    vocab, corpus = load_data(DATA_PATH)
    vocab_list, vocab_to_id = build_vocabulary_index(vocab)
    token_sequences = tokenize_corpus(corpus, set(vocab_list))

    raw_matrix = gen_term_document_matrix(token_sequences, vocab_to_id)
    tfidf_matrix = gen_tfidf_term_document_matrix(token_sequences, vocab_to_id)

    raw_embeddings = truncated_svd(raw_matrix, vector_size=DIM, seed=SEED)
    tfidf_embeddings = truncated_svd(tfidf_matrix, vector_size=DIM, seed=SEED)

    save_svd_embeddings(
        output_dir=output_dir,
        vocab=vocab_list,
        embeddings=raw_embeddings,
        config={"vector_size": DIM, "weighting": "raw"},
        prefix=f"svd_raw_d{DIM}",
    )
    save_svd_embeddings(
        output_dir=output_dir,
        vocab=vocab_list,
        embeddings=tfidf_embeddings,
        config={"vector_size": DIM, "weighting": "tfidf"},
        prefix=f"svd_tfidf_d{DIM}",
    )

    neighbor_report = {
        "tokens": list(NEIGHBOURS),
        "raw": compile_neighbors(NEIGHBOURS, vocab_to_id, raw_embeddings, TOP_K),
        "tfidf": compile_neighbors(NEIGHBOURS, vocab_to_id, tfidf_embeddings, TOP_K),
    }
    with (output_dir / f"task5_neighbor_comparison_d{DIM}.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(neighbor_report, handle, indent=2)

    mlp_raw = train_and_eval_mlp(raw_embeddings, vocab_to_id, output_dir, f"svd_raw_d{DIM}")
    mlp_tfidf = train_and_eval_mlp(tfidf_embeddings, vocab_to_id, output_dir, f"svd_tfidf_d{DIM}")

    summary = {
        "best_d": DIM,
        "neighbors_report": f"task5_neighbor_comparison_d{DIM}.json",
        "mlp_raw_metrics": f"task5_mlp_svd_raw_d{DIM}_metrics.json",
        "mlp_tfidf_metrics": f"task5_mlp_svd_tfidf_d{DIM}_metrics.json",
        "comparison": {
            "dev_weighted_f1_raw": mlp_raw["dev"]["weighted_f1"],
            "dev_weighted_f1_tfidf": mlp_tfidf["dev"]["weighted_f1"],
            "test_weighted_f1_raw": mlp_raw["test"]["weighted_f1"],
            "test_weighted_f1_tfidf": mlp_tfidf["test"]["weighted_f1"],
        },
    }
    with (output_dir / "task5_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print("Task 5 complete. Outputs written to", output_dir)


if __name__ == "__main__":
    main()