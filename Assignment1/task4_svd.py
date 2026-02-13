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


# -----------------------------
# CLI
# -----------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Task 4: MLP NER with SVD embeddings")

    parser.add_argument("--svd-dir", type=str, default="output/task2")
    parser.add_argument("--dims", type=int, nargs="*", default=[50, 100, 200, 300])

    parser.add_argument("--dataset", type=str, default="conll2003")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)

    parser.add_argument("--output-dir", type=str, default="output/task4")

    return parser.parse_args()


# -----------------------------
# Embeddings
# -----------------------------

def load_embeddings(vectors_path: Path, vocab_path: Path):
    vectors = np.load(vectors_path)

    # normalize — helps SVD a lot
    vectors = vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-7)

    with vocab_path.open("r") as f:
        vocab = json.load(f)

    vocab_to_id = {t: i for i, t in enumerate(vocab)}

    # OOV strategy — mean + small noise
    if "<UNK>" not in vocab_to_id:
        mean = vectors.mean(axis=0)
        std = vectors.std(axis=0)
        oov_vec = mean + 0.05 * std * np.random.randn(*mean.shape)

        vocab_to_id["<UNK>"] = len(vectors)
        vectors = np.vstack([vectors, oov_vec])

    return vectors, vocab_to_id


# -----------------------------
# Dataset builder
# -----------------------------

def build_token_dataset(split, vocab_to_id, ner_label_map):
    token_ids = []
    label_ids = []

    for record in split:
        for tok, ner in zip(record["tokens"], record["ner_tags"]):
            token_ids.append(vocab_to_id.get(tok.lower(), vocab_to_id["<UNK>"]))
            label_ids.append(ner_label_map[int(ner)])

    return np.array(token_ids, dtype=np.int64), np.array(label_ids, dtype=np.int64)


# -----------------------------
# Model
# -----------------------------

class TokenMLP(nn.Module):
    def __init__(self, d, hidden, num_labels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, num_labels),
        )

    def forward(self, x):
        return self.net(x)


# -----------------------------
# Train + evaluate
# -----------------------------

def train_and_evaluate(vectors_path, vocab_path, dataset, label_names, args):

    vectors, vocab_to_id = load_embeddings(vectors_path, vocab_path)

    ner_label_map = {i: i for i in range(len(label_names))}

    tr_ids, tr_labels = build_token_dataset(dataset["train"], vocab_to_id, ner_label_map)
    dv_ids, dv_labels = build_token_dataset(dataset["validation"], vocab_to_id, ner_label_map)
    te_ids, te_labels = build_token_dataset(dataset["test"], vocab_to_id, ner_label_map)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    emb_tensor = torch.tensor(vectors, dtype=torch.float32, device=device)

    Xtr = emb_tensor[torch.tensor(tr_ids, device=device)]
    Xdv = emb_tensor[torch.tensor(dv_ids, device=device)]
    Xte = emb_tensor[torch.tensor(te_ids, device=device)]

    ytr = torch.tensor(tr_labels, device=device, dtype=torch.long)
    ydv = torch.tensor(dv_labels, device=device, dtype=torch.long)
    yte = torch.tensor(te_labels, device=device, dtype=torch.long)

    train_loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=args.batch_size, shuffle=True)
    dev_loader = DataLoader(TensorDataset(Xdv, ydv), batch_size=args.batch_size)
    test_loader = DataLoader(TensorDataset(Xte, yte), batch_size=args.batch_size)

    model = TokenMLP(vectors.shape[1], args.hidden_size, len(label_names)).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    # ---- training ----
    for epoch in range(args.epochs):
        model.train()
        for xb, yb in train_loader:
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()

    # ---- eval helper ----
    def eval_loader(loader):
        model.eval()
        preds, gold = [], []
        with torch.no_grad():
            for xb, yb in loader:
                out = model(xb)
                p = out.argmax(dim=1)
                preds.extend(p.cpu().tolist())
                gold.extend(yb.cpu().tolist())

        return {
            "weighted_f1": float(f1_score(gold, preds, average="weighted")),
            "classification_report": classification_report(
                gold,
                preds,
                target_names=label_names,
                output_dict=True,
                zero_division=0,
            ),
        }

    return eval_loader(dev_loader), eval_loader(test_loader)


# -----------------------------
# Main
# -----------------------------

def main():
    args = parse_args()
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(args.dataset)
    label_names = dataset["train"].features["ner_tags"].feature.names

    for d in args.dims:

        vec_path = Path(args.svd_dir) / f"svd_d{d}_vectors.npy"
        vocab_path = Path(args.svd_dir) / f"svd_d{d}_vocab.json"

        if not vec_path.exists():
            print("Missing:", vec_path)
            continue

        print(f"\nTraining SVD-MLP for d={d}")

        dev_metrics, test_metrics = train_and_evaluate(
            vec_path, vocab_path, dataset, label_names, args
        )

        result = {
            "embedding_vectors": str(vec_path),
            "embedding_vocab": str(vocab_path),
            "num_labels": len(label_names),
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "hidden_size": args.hidden_size,
            "learning_rate": args.lr,
            "dev": dev_metrics,
            "test": test_metrics,
            "embedding_dimension": d,
        }

        out_file = outdir / f"task4_svd_metrics_d{d}.json"
        json.dump(result, open(out_file, "w"), indent=2)

        print("Saved:", out_file)


if __name__ == "__main__":
    main()
