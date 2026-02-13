from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from datasets import load_dataset
from sklearn.metrics import f1_score, accuracy_score

TASK1_SELECTION = "output/task1/task1_selected_hyperparameters.json"

def parse_args():
    p = argparse.ArgumentParser("Task4 MLP NER")
    p.add_argument(
        "--embedding-vectors",
        type=str,
        default="output/task1/glove_w10_lr0.05_e25_d200_vectors.npy",
    )
    p.add_argument(
        "--embedding-vocab",
        type=str,
        default="output/task1/glove_w10_lr0.05_e25_d200_vocab.json",
    )
    p.add_argument("--task1-dir", type=str, default="output/task1")
    p.add_argument("--dims", type=int, nargs="*", default=[50, 100, 200, 300])
    p.add_argument("--single", action="store_true")
    p.add_argument("--svd-dir", type=str, default="output/task2") # Update this accordingly
    p.add_argument("--output-dir", type=str, default="output/task4") 
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--seed", type=int, default=17)
    return p.parse_args()


class TokenMLP(nn.Module):
    def __init__(self, d_in: int, hidden: int, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, 2*hidden),
            nn.LayerNorm(2*hidden),
            nn.ReLU(),
            nn.Dropout(0.4),

            nn.Linear(2*hidden, hidden),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(hidden, num_classes)
        )

    def forward(self, x):
        return self.net(x)


class TokenDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]



def load_embedding_set(prefix: Path) -> Tuple[List[str], np.ndarray]:
    vocab = json.load(open(prefix.with_suffix("_vocab.json"), "r"))
    vectors = np.load(prefix.with_suffix("_vectors.npy"))
    return vocab, vectors


def build_lookup(vocab: List[str], vectors: np.ndarray):
    token_to_id = {t: i for i, t in enumerate(vocab)}

    # ---- OOV STRATEGY ----
    # Use mean embedding vector
    oov_vector = vectors.mean(axis=0)

    def lookup(token: str):
        idx = token_to_id.get(token.lower())
        if idx is None:
            return oov_vector
        return vectors[idx]

    return lookup


def load_conll_flat():
    trust_remote_code = True
    ds = load_dataset("conll2003", trust_remote_code=trust_remote_code)
    ner_feature = ds["train"].features["ner_tags"]
    label_names = ner_feature.feature.names

    def flatten(split):
        tokens = []
        labels = []
        for rec in split:
            for tok, lab in zip(rec["tokens"], rec["ner_tags"]):
                tokens.append(tok)
                labels.append(lab)
        return tokens, np.array(labels)

    tr = flatten(ds["train"])
    dv = flatten(ds["validation"])
    te = flatten(ds["test"])

    return tr, dv, te, label_names



def tokens_to_matrix(tokens: List[str], lookup):
    X = np.stack([lookup(t) for t in tokens])
    return X


def format_lr(value: float) -> str:
    text = f"{value}"
    if text.endswith(".0"):
        return text[:-2]
    return text



def train_eval(name, Xtr, ytr, Xdv, ydv, Xte, yte, d, args, num_classes):

    model = TokenMLP(d, args.hidden, num_classes)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    train_loader = DataLoader(TokenDataset(Xtr, ytr), batch_size=args.batch_size, shuffle=True)
    dev_loader = DataLoader(TokenDataset(Xdv, ydv), batch_size=args.batch_size)
    test_loader = DataLoader(TokenDataset(Xte, yte), batch_size=args.batch_size)

    for epoch in range(args.epochs):
        model.train()
        for xb, yb in train_loader:
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()

    def evaluate(loader):
        model.eval()
        preds, gold = [], []
        with torch.no_grad():
            for xb, yb in loader:
                out = model(xb)
                p = out.argmax(dim=1)
                preds.extend(p.numpy())
                gold.extend(yb.numpy())
        acc = accuracy_score(gold, preds)
        mf1 = f1_score(gold, preds, average="macro")
        return acc, mf1

    dev_acc, dev_f1 = evaluate(dev_loader)
    test_acc, test_f1 = evaluate(test_loader)

    return {
        "name": name,
        "dim": d,
        "dev_acc": dev_acc,
        "dev_macro_f1": dev_f1,
        "test_acc": test_acc,
        "test_macro_f1": test_f1,
    }



def main():
    args = parse_args()
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    (tr_tok, ytr), (dv_tok, ydv), (te_tok, yte), label_names = load_conll_flat()
    num_classes = len(label_names)
    print(f"Loaded CoNLL-2003 with {len(tr_tok)} train, {len(dv_tok)} dev, {len(te_tok)} test tokens")
    print(f"Number of classes: {num_classes}")
    results = []

    dims = [50, 100, 200, 300]

    # -------- GLOVE --------
    if args.single:
        vectors_path = Path(args.embedding_vectors)
        vocab_path = Path(args.embedding_vocab)
        vocab = json.load(open(vocab_path))
        vecs = np.load(vectors_path)
        vecs = vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-7)
        d = vecs.shape[1]

        lookup = build_lookup(vocab, vecs)
        Xtr = tokens_to_matrix(tr_tok, lookup)
        Xdv = tokens_to_matrix(dv_tok, lookup)
        Xte = tokens_to_matrix(te_tok, lookup)

        results.append(train_eval("GloVe-MLP", Xtr, ytr, Xdv, ydv, Xte, yte, d, args, num_classes))
    else:
        selection_path = Path(TASK1_SELECTION)
        with selection_path.open("r", encoding="utf-8") as handle:
            selected = json.load(handle)

        window_size = int(selected["window_size"])
        learning_rate = float(selected["learning_rate"])
        epochs = int(selected["epochs"])
        lr_text = format_lr(learning_rate)

        for d in args.dims:
            print(f"\n=== Evaluating GloVe with d={d} ===")
            vectors_name = f"glove_w{window_size}_lr{lr_text}_e{epochs}_d{d}_vectors.npy"
            vocab_name = f"glove_w{window_size}_lr{lr_text}_e{epochs}_d{d}_vocab.json"
            vectors_path = Path(args.task1_dir) / vectors_name
            vocab_path = Path(args.task1_dir) / vocab_name
            if not vectors_path.exists() or not vocab_path.exists():
                continue

            vocab = json.load(open(vocab_path))
            vecs = np.load(vectors_path)
            vecs = vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-7)

            lookup = build_lookup(vocab, vecs)
            Xtr = tokens_to_matrix(tr_tok, lookup)
            Xdv = tokens_to_matrix(dv_tok, lookup)
            Xte = tokens_to_matrix(te_tok, lookup)

            results.append(train_eval("GloVe-MLP", Xtr, ytr, Xdv, ydv, Xte, yte, d, args, num_classes))

    # -------- SVD --------
    # for d in dims:
    #     print(f"\n=== Evaluating SVD with d={d} ===")
    #     base = Path(args.svd_dir) / f"svd_d{d}"
    #     if not (base.with_name(base.name + "_vectors.npy")).exists():
    #         continue

    #     vocab = json.load(open(str(base)+"_vocab.json"))
    #     vecs = np.load(str(base)+"_vectors.npy")
    #     vecs = vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-7) 

    #     lookup = build_lookup(vocab, vecs)

    #     Xtr = tokens_to_matrix(tr_tok, lookup)
    #     Xdv = tokens_to_matrix(dv_tok, lookup)
    #     Xte = tokens_to_matrix(te_tok, lookup)

    #     results.append(train_eval(f"SVD-MLP", Xtr, ytr, Xdv, ydv, Xte, yte, d, args, num_classes))

    json.dump(results, open(outdir/"task4_results.json","w"), indent=2)

    print("\n=== RESULTS ===")
    for r in results:
        print(r)


if __name__ == "__main__":
    main()
