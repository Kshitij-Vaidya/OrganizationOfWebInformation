from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from datasets import load_dataset
from sklearn.metrics import classification_report, f1_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

EMBEDDING_VECTORS = "output/task1/glove_w10_lr0.05_e25_d200_vectors.npy"
EMBEDDING_VOCAB = "output/task1/glove_w10_lr0.05_e25_d200_vocab.json"
TASK1_DIR = "output/task1"
TASK1_HYPERPARAMETERS = "output/task1/task1_selected_hyperparameters.json"
DIMS = [50, 100, 200, 300]
DATASET = "conll2003"
EPOCHS = 10
BATCH_SIZE = 1024
HIDDEN_SIZE = 128
LEARNING_RATE = 1e-3
OUTPUT_DIR = "output/task4"

# def parse_args() -> argparse.Namespace:
#     parser = argparse.ArgumentParser(description="Task 4: MLP NER with pretrained embeddings")
#     # Argument for smaller testing to validate program correctness
#     parser.add_argument(
#         "--single",
#         action="store_true",
#         help="Train only with --embedding-vectors/--embedding-vocab (skip sweep).",
#     )
#     return parser.parse_args()


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
) -> dict[str, object]:
    vectors, vocab_to_id = load_embeddings(vectors_path, vocab_path)
    # Add an <UNK> token to the vocabulary list to handle OOV instances in the test dataset
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
        TensorDataset(x_train, y_train), batch_size=BATCH_SIZE, shuffle=True
    )
    dev_loader = DataLoader(TensorDataset(x_dev, y_dev), batch_size=BATCH_SIZE)
    test_loader = DataLoader(TensorDataset(x_test, y_test), batch_size=BATCH_SIZE)

    model = TokenMLP(
        input_dim=vectors.shape[1],
        hidden_dim=HIDDEN_SIZE,
        num_labels=len(label_names),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, EPOCHS + 1):
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
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "hidden_size": HIDDEN_SIZE,
        "learning_rate": LEARNING_RATE,
        "dev": {"weighted_f1": dev_f1, "classification_report": dev_report},
        "test": {"weighted_f1": test_f1, "classification_report": test_report},
    }


def main() -> None:
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(DATASET)
    ner_feature = dataset["train"].features["ner_tags"]
    label_names = ner_feature.feature.names
    ner_label_map = {idx: idx for idx in range(len(label_names))}

    # Run a single experiment over the model hyperparameters without any dimension sweeps
    # if args.single:
    #     vectors_path = Path(EMBEDDING_VECTORS)
    #     vocab_path = Path(EMBEDDING_VOCAB)
    #     metrics = train_and_evaluate(
    #         vectors_path,
    #         vocab_path,
    #         dataset,
    #         label_names,
    #         ner_label_map,
    #         args,
    #     )
    #     metrics_path = output_dir / "task4_metrics.json"
    #     with metrics_path.open("w", encoding="utf-8") as handle:
    #         json.dump(metrics, handle, indent=2)
    #     print("Task 4 complete. Metrics written to", metrics_path)
    #     return

    selection_path = Path(TASK1_HYPERPARAMETERS)
    with selection_path.open("r", encoding="utf-8") as handle:
        selected = json.load(handle)

    window_size = int(selected["window_size"])
    learning_rate = float(selected["learning_rate"])
    epochs = int(selected["epochs"])
    lr_text = format_lr(learning_rate)

    metrics_summary: list[dict[str, object]] = []
    for dim in DIMS:
        vectors_name = f"glove_w{window_size}_lr{lr_text}_e{epochs}_d{dim}_vectors.npy"
        vocab_name = f"glove_w{window_size}_lr{lr_text}_e{epochs}_d{dim}_vocab.json"
        vectors_path = Path(TASK1_DIR) / vectors_name
        vocab_path = Path(TASK1_DIR) / vocab_name

        print(f"\nTraining MLP for d={dim} using {vectors_name}")
        metrics = train_and_evaluate(
            vectors_path,
            vocab_path,
            dataset,
            label_names,
            ner_label_map,
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
