from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, Sequence

import numpy as np
# from sklearn.kernel_approximation import svd
from tqdm import tqdm
from scipy import sparse
from scipy.sparse.linalg import svds


ArrayLike = np.ndarray


def load_data(json_path: str | Path) -> tuple[list[str], list[str]]:
    """Load vocabulary and documents from the provided JSON artifact."""
    json_path = Path(json_path)
    print(f"Loading data from {json_path}")
    with json_path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    vocab = list(data.keys())
    docs_map: Dict[int, str] = {}

    print("Reconstructing corpus")
    for entries in tqdm(data.values(), desc="Parsing keys"):
        for index, text in entries:
            if index not in docs_map:
                docs_map[index] = text

    sorted_indices = sorted(docs_map.keys())
    corpus = [docs_map[i] for i in sorted_indices]
    print(f"Loaded {len(vocab)} vocab tokens and {len(corpus)} documents")
    return vocab, corpus


def tokenizer(text: str) -> list[str]:
    """Lowercase tokenize by word boundaries."""
    return re.findall(r"\b\w+\b", text.lower())


def build_vocabulary_index(vocab: Iterable[str]) -> tuple[list[str], dict[str, int]]:
    """Return a sorted vocabulary list and mapping to contiguous ids."""
    vocab_list = sorted(set(vocab))
    vocab_to_id = {token: idx for idx, token in enumerate(vocab_list)}
    return vocab_list, vocab_to_id


def tokenize_corpus(corpus: Sequence[str], vocab_filter: set[str] | None = None) -> list[list[str]]:
    """Tokenize each document while optionally restricting to a known vocabulary."""
    filtered_tokens: list[list[str]] = []
    for document in corpus:
        tokens = tokenizer(document)
        if vocab_filter is not None:
            tokens = [token for token in tokens if token in vocab_filter]
        if tokens:
            filtered_tokens.append(tokens)
    return filtered_tokens


def build_cooccurrence_matrix(
    token_sequences: Sequence[Sequence[str]],
    vocab_to_id: dict[str, int],
    window_size: int,
) -> dict[tuple[int, int], float]:
    """Construct a symmetric co-occurrence dictionary using inverse distance weights."""
    cooccurrence: dict[tuple[int, int], float] = defaultdict(float)
    for tokens in tqdm(token_sequences, desc="Building co-occurrence"):
        token_ids = [vocab_to_id[token] for token in tokens if token in vocab_to_id]
        for center_pos, center_id in enumerate(token_ids):
            start = max(center_pos - window_size, 0)
            end = min(center_pos + window_size + 1, len(token_ids))
            for context_pos in range(start, end):
                if context_pos == center_pos:
                    continue
                context_id = token_ids[context_pos]
                distance = abs(center_pos - context_pos)
                cooccurrence[(center_id, context_id)] += 1.0 / distance
    return cooccurrence


def cooccurrence_items(cooccurrence: dict[tuple[int, int], float]) -> list[tuple[int, int, float]]:
    """Convert co-occurrence mapping into a list of triplets."""
    return [(i, j, value) for (i, j), value in cooccurrence.items()]


@dataclass(slots=True)
class GloVeConfig:
    vector_size: int
    x_max: float = 100.0
    alpha: float = 0.75
    learning_rate: float = 0.05
    seed: int | None = None


class GloVeModel:
    """Minimal GloVe implementation for Task 1."""

    def __init__(self, vocab_size: int, config: GloVeConfig) -> None:
        self.vocab_size = vocab_size
        self.config = config
        rng = np.random.default_rng(config.seed)
        scale = 0.5 / config.vector_size
        self.target_embeddings = rng.uniform(-scale, scale, (vocab_size, config.vector_size))
        self.context_embeddings = rng.uniform(-scale, scale, (vocab_size, config.vector_size))
        self.target_biases = np.zeros(vocab_size, dtype=np.float64)
        self.context_biases = np.zeros(vocab_size, dtype=np.float64)

    def fit(
        self,
        cooccurrence: Sequence[tuple[int, int, float]],
        epochs: int,
        verbose: bool = True,
    ) -> list[float]:
        if not cooccurrence:
            raise ValueError("Co-occurrence data is empty")

        history: list[float] = []
        rng = np.random.default_rng(self.config.seed)
        entries = list(cooccurrence)

        for epoch in range(1, epochs + 1):
            rng.shuffle(entries)
            total_loss = 0.0

            for i_idx, j_idx, x_ij in entries:
                weight = 1.0 if x_ij >= self.config.x_max else (x_ij / self.config.x_max) ** self.config.alpha
                log_x = math.log(x_ij)
                target_vec = self.target_embeddings[i_idx]
                context_vec = self.context_embeddings[j_idx]
                interaction = float(np.dot(target_vec, context_vec)) + self.target_biases[i_idx] + self.context_biases[j_idx] - log_x
                loss = weight * interaction * interaction * 0.5
                total_loss += loss

                grad_scale = weight * interaction
                target_snapshot = target_vec.copy()
                context_snapshot = context_vec.copy()

                self.target_embeddings[i_idx] -= self.config.learning_rate * grad_scale * context_snapshot
                self.context_embeddings[j_idx] -= self.config.learning_rate * grad_scale * target_snapshot
                self.target_biases[i_idx] -= self.config.learning_rate * grad_scale
                self.context_biases[j_idx] -= self.config.learning_rate * grad_scale

            average_loss = total_loss / len(entries)
            history.append(average_loss)
            if verbose:
                print(f"Epoch {epoch}: loss={average_loss:.4f}")

        return history

    def get_embeddings(self) -> ArrayLike:
        """Combine target and context embeddings."""
        return self.target_embeddings + self.context_embeddings


def save_embeddings(
    output_dir: str | Path,
    vocab: Sequence[str],
    embeddings: ArrayLike,
    loss_history: Sequence[float],
    config: GloVeConfig,
    prefix: str = "glove",
) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    np.save(output_path / f"{prefix}_vectors.npy", embeddings)
    with (output_path / f"{prefix}_vocab.json").open("w", encoding="utf-8") as handle:
        json.dump(list(vocab), handle)
    metadata = {"loss_history": list(loss_history), "config": asdict(config)}
    with (output_path / f"{prefix}_training.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

def save_svd_embeddings(
    output_dir: str | Path,
    vocab: Sequence[str],
    embeddings: ArrayLike,
    config: dict,
    prefix: str = "svd",
) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    np.save(output_path / f"{prefix}_vectors.npy", embeddings)
    with (output_path / f"{prefix}_vocab.json").open("w", encoding="utf-8") as handle:
        json.dump(list(vocab), handle)
    with (output_path / f"{prefix}_config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)


def find_nearest_neighbors(
    query_tokens: Sequence[str],
    vocab_to_id: dict[str, int],
    embeddings: ArrayLike,
    top_k: int = 5,
) -> dict[str, list[tuple[str, float]]]:
    """Return cosine nearest neighbors for given tokens."""
    normed = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12)
    results: dict[str, list[tuple[str, float]]] = {}
    vocab_list = [None] * len(vocab_to_id)
    for token, idx in vocab_to_id.items():
        vocab_list[idx] = token

    for token in query_tokens:
        if token not in vocab_to_id:
            results[token] = []
            continue
        idx = vocab_to_id[token]
        similarities = normed @ normed[idx]
        ranked = np.argsort(-similarities)
        neighbors: list[tuple[str, float]] = []
        for candidate in ranked:
            if candidate == idx:
                continue
            neighbors.append((vocab_list[candidate], float(similarities[candidate])))
            if len(neighbors) == top_k:
                break
        results[token] = neighbors
    return results

def gen_term_document_matrix(token_sequences: Sequence[Sequence[str]], vocab_to_id: dict[str, int]) -> np.ndarray:
    """Construct a term-document matrix from token sequences."""
    num_docs = len(token_sequences)
    vocab_size = len(vocab_to_id)
    matrix = sparse.lil_matrix((num_docs, vocab_size), dtype=np.float64)

    for doc_idx, tokens in enumerate(tqdm(token_sequences, desc="Building term-document matrix")):
        for token in tokens:
            if token in vocab_to_id:
                token_id = vocab_to_id[token]
                matrix[doc_idx, token_id] += 1.0

    matrix = matrix.tocsr()
    return matrix

def truncated_svd(matrix: sparse.csr_matrix, vector_size: int, seed: int | None = None) -> np.ndarray:
    """Perform truncated SVD on the given matrix."""
    # u, s, vt = svds(matrix, k=vector_size, random_state=seed)
    # return u @ np.diag(s)
    
    u, s, vt = svds(matrix, k=vector_size, random_state=seed)

    # svds returns singular values in ascending order → reverse them
    idx = np.argsort(-s)
    s = s[idx]
    vt = vt[idx, :]

    # Word embeddings = V Σ
    embeddings = vt.T * s

    return embeddings
