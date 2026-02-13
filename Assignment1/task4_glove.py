from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from datasets import load_dataset
from sklearn.metrics import classification_report, f1_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Task 4: MLP NER with pretrained embeddings")
    # Keeping default parameters here for Task 1 results, make this always required to accomodate both task 1 and task 2 result files accordingly
    parser.add_argument(
        "--embedding-vectors",
        type=str,
        default="output/task1/glove_w10_lr0.05_e25_d200_vectors.npy",
        help="Path to embeddings .npy (e.g., glove_w10_lr0.05_e25_d200_vectors.npy).",
    )
    parser.add_argument(
        "--embedding-vocab",
        type=str,
        default="output/task1/glove_w10_lr0.05_e25_d200_vocab.json",
        help="Path to vocab .json matching the embeddings.",
    )
    parser.add_argument(
        "--task1-dir",
        type=str,
        default="output/task1",
        help="Directory containing Task 1 embedding artifacts.",
    )
    parser.add_argument(
        "--task1-selection",
        type=str,
        default="output/task1/task1_selected_hyperparameters.json",
        help="Path to Task 1 selected hyperparameters JSON.",
    )
    parser.add_argument(
        "--dims",
        type=int,
        nargs="*",
        default=[50, 100, 200, 300],
        help="Embedding dimensions to sweep when not using --single.",
    )
    parser.add_argument(
        "--single",
        action="store_true",
        help="Train only with --embedding-vectors/--embedding-vocab (skip sweep).",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="conll2003",
        help="Hugging Face dataset name to load.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Training epochs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1024,
        help="Batch size for token classification.",
    )
    parser.add_argument(
        "--hidden-size",
        type=int,
        default=128,
        help="Hidden layer size for the MLP.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output/task4",
        help="Directory for metrics output.",
    )
    return parser.parse_args()


def load_embeddings(vectors_path: Path, vocab_path: Path) -> tuple[np.ndarray, dict[str, int]]:
    vectors = np.load(vectors_path)
    with vocab_path.open("r", encoding="utf-8") as handle:
        vocab = json.load(handle)
    vocab_to_id = {token: idx for idx, token in enumerate(vocab)}
    return vectors, vocab_to_id


def build_token_dataset(
    split: Iterable[dict[str, object]],
    vocab_to_id: dict[str, int],
    ner_label_map: dict[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    token_ids: list[int] = []
    label_ids: list[int] = []

    for record in split:
        tokens = record["tokens"]
        ner_tags = record["ner_tags"]
        for token, ner in zip(tokens, ner_tags):
            token_ids.append(vocab_to_id.get(token.lower(), vocab_to_id["<UNK>"]))
            label_ids.append(ner_label_map[int(ner)])

    return np.array(token_ids, dtype=np.int64), np.array(label_ids, dtype=np.int64)


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


def format_lr(value: float) -> str:
    text = f"{value}"
    if text.endswith(".0"):
        return text[:-2]
    return text


def train_and_evaluate(
    vectors_path: Path,
    vocab_path: Path,
    dataset,
    label_names: list[str],
    ner_label_map: dict[int, int],
    args: argparse.Namespace,
) -> dict[str, object]:
    vectors, vocab_to_id = load_embeddings(vectors_path, vocab_path)
    if "<UNK>" not in vocab_to_id:
        unk_vector = np.zeros((1, vectors.shape[1]), dtype=vectors.dtype)
        vocab_to_id["<UNK>"] = vectors.shape[0]
        vectors = np.vstack([vectors, unk_vector])

    train_ids, train_labels = build_token_dataset(
        dataset["train"], vocab_to_id, ner_label_map
    )
    dev_ids, dev_labels = build_token_dataset(
        dataset["validation"], vocab_to_id, ner_label_map
    )
    test_ids, test_labels = build_token_dataset(
        dataset["test"], vocab_to_id, ner_label_map
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    embedding_tensor = torch.tensor(vectors, dtype=torch.float32, device=device)
    x_train = embedding_tensor[torch.tensor(train_ids, device=device)]
    y_train = torch.tensor(train_labels, dtype=torch.long, device=device)
    x_dev = embedding_tensor[torch.tensor(dev_ids, device=device)]
    y_dev = torch.tensor(dev_labels, dtype=torch.long, device=device)
    x_test = embedding_tensor[torch.tensor(test_ids, device=device)]
    y_test = torch.tensor(test_labels, dtype=torch.long, device=device)

    train_loader = DataLoader(
        TensorDataset(x_train, y_train), batch_size=args.batch_size, shuffle=True
    )
    dev_loader = DataLoader(TensorDataset(x_dev, y_dev), batch_size=args.batch_size)
    test_loader = DataLoader(TensorDataset(x_test, y_test), batch_size=args.batch_size)

    model = TokenMLP(
        input_dim=vectors.shape[1],
        hidden_dim=args.hidden_size,
        num_labels=len(label_names),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch_x.size(0)

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_x, batch_y in dev_loader:
                logits = model(batch_x)
                preds = logits.argmax(dim=-1)
                correct += (preds == batch_y).sum().item()
                total += batch_y.numel()

        avg_loss = train_loss / max(len(train_loader.dataset), 1)
        accuracy = correct / max(total, 1)
        print(f"Epoch {epoch}: loss={avg_loss:.4f} | dev_acc={accuracy:.4f}")

    def evaluate_split(loader: DataLoader) -> tuple[list[int], list[int]]:
        model.eval()
        all_preds: list[int] = []
        all_labels: list[int] = []
        with torch.no_grad():
            for batch_x, batch_y in loader:
                logits = model(batch_x)
                preds = logits.argmax(dim=-1)
                all_preds.extend(preds.detach().cpu().tolist())
                all_labels.extend(batch_y.detach().cpu().tolist())
        return all_labels, all_preds

    dev_true, dev_pred = evaluate_split(dev_loader)
    test_true, test_pred = evaluate_split(test_loader)

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
    dev_f1 = f1_score(dev_true, dev_pred, average="weighted")
    test_f1 = f1_score(test_true, test_pred, average="weighted")

    return {
        "embedding_vectors": str(vectors_path),
        "embedding_vocab": str(vocab_path),
        "num_labels": len(label_names),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "hidden_size": args.hidden_size,
        "learning_rate": args.lr,
        "dev": {"weighted_f1": dev_f1, "classification_report": dev_report},
        "test": {"weighted_f1": test_f1, "classification_report": test_report},
    }


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(args.dataset)
    ner_feature = dataset["train"].features["ner_tags"]
    label_names = ner_feature.feature.names
    ner_label_map = {idx: idx for idx in range(len(label_names))}

    if args.single:
        vectors_path = Path(args.embedding_vectors)
        vocab_path = Path(args.embedding_vocab)
        metrics = train_and_evaluate(
            vectors_path,
            vocab_path,
            dataset,
            label_names,
            ner_label_map,
            args,
        )
        metrics_path = output_dir / "task4_metrics.json"
        with metrics_path.open("w", encoding="utf-8") as handle:
            json.dump(metrics, handle, indent=2)
        print("Task 4 complete. Metrics written to", metrics_path)
        return

    selection_path = Path(args.task1_selection)
    with selection_path.open("r", encoding="utf-8") as handle:
        selected = json.load(handle)

    window_size = int(selected["window_size"])
    learning_rate = float(selected["learning_rate"])
    epochs = int(selected["epochs"])
    lr_text = format_lr(learning_rate)

    metrics_summary: list[dict[str, object]] = []
    for dim in args.dims:
        vectors_name = f"glove_w{window_size}_lr{lr_text}_e{epochs}_d{dim}_vectors.npy"
        vocab_name = f"glove_w{window_size}_lr{lr_text}_e{epochs}_d{dim}_vocab.json"
        vectors_path = Path(args.task1_dir) / vectors_name
        vocab_path = Path(args.task1_dir) / vocab_name

        print(f"\nTraining MLP for d={dim} using {vectors_name}")
        metrics = train_and_evaluate(
            vectors_path,
            vocab_path,
            dataset,
            label_names,
            ner_label_map,
            args,
        )
        metrics["embedding_dimension"] = dim
        metrics["task1_selected"] = selected

        metrics_path = output_dir / f"task4_metrics_d{dim}.json"
        with metrics_path.open("w", encoding="utf-8") as handle:
            json.dump(metrics, handle, indent=2)
        metrics_summary.append({"dimension": dim, "metrics_path": str(metrics_path)})

    summary_path = output_dir / "task4_metrics_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump({"runs": metrics_summary}, handle, indent=2)

    print("Task 4 complete. Metrics written to", output_dir)


if __name__ == "__main__":
    main()
