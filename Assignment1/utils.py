from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass, asdict
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, Sequence

import numpy as np
from tqdm import tqdm


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


def find_nearest_neighbours(
    query_tokens: Sequence[str],
    vocab_to_id: dict[str, int],
    embeddings: ArrayLike,
    top_k: int = 5,
) -> dict[str, list[tuple[str, float]]]:
    """Return cosine nearest neighbours for given tokens."""
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
        neighbours: list[tuple[str, float]] = []
        for candidate in ranked:
            if candidate == idx:
                continue
            neighbours.append((vocab_list[candidate], float(similarities[candidate])))
            if len(neighbours) == top_k:
                break
        results[token] = neighbours
    return results


@dataclass(slots=True)
class CRFFeatureConfig:
    include_lexical: bool = True
    include_shape: bool = True
    include_subword: bool = True
    window_size: int = 2


@lru_cache(maxsize=10000)
def word_shape(token: str) -> str:
    """Map token to a compact shape signature (e.g., 'McDonalds' -> 'XxXx')."""
    shape_chars = []
    for char in token:
        if char.isupper():
            shape_chars.append("X")
        elif char.islower():
            shape_chars.append("x")
        elif char.isdigit():
            shape_chars.append("d")
        else:
            shape_chars.append("_")
    if not shape_chars:
        return ""
    compressed = [shape_chars[0]]
    for char in shape_chars[1:]:
        if char != compressed[-1]:
            compressed.append(char)
    return "".join(compressed)


def sentence_to_features(
    sentence: Sequence[tuple[str, str, str, str]],
    config: CRFFeatureConfig,
) -> list[dict[str, object]]:
    """Convert a sentence into a list of CRF feature dicts."""
    features: list[dict[str, object]] = []
    for index in range(len(sentence)):
        features.append(token_to_features(sentence, index, config))
    return features


def sentence_to_labels(sentence: Sequence[tuple[str, str, str, str]]) -> list[str]:
    return [label for (_, _, _, label) in sentence]


def token_to_features(
    sentence: Sequence[tuple[str, str, str, str]],
    index: int,
    config: CRFFeatureConfig,
) -> dict[str, object]:
    word, pos, chunk, _ = sentence[index]
    features: dict[str, object] = {
        "bias": 1.0,
    }

    if config.include_lexical:
        features.update(
            {
                "word.lower": word.lower(),
                "pos": pos,
                "chunk": chunk,
                "pos[:2]": pos[:2],
            }
        )

    if config.include_shape:
        features.update(
            {
                "word.isupper": word.isupper(),
                "word.istitle": word.istitle(),
                "word.isdigit": word.isdigit(),
                "word.shape": word_shape(word),
                "word.has_digit": any(char.isdigit() for char in word),
                "word.has_hyphen": "-" in word,
                "word.has_period": "." in word,
            }
        )

    if config.include_subword:
        lowered = word.lower()
        for length in (2, 3, 4):
            if len(lowered) >= length:
                features[f"pref{length}"] = lowered[:length]
                features[f"suf{length}"] = lowered[-length:]

    window = config.window_size
    for offset in range(1, window + 1):
        prev_index = index - offset
        if prev_index >= 0:
            prev_word, prev_pos, prev_chunk, _ = sentence[prev_index]
            prefix = f"- {offset}"
            if config.include_lexical:
                features[f"{prefix}:word.lower"] = prev_word.lower()
                features[f"{prefix}:pos"] = prev_pos
                features[f"{prefix}:chunk"] = prev_chunk
            if config.include_shape:
                features[f"{prefix}:word.isupper"] = prev_word.isupper()
                features[f"{prefix}:word.istitle"] = prev_word.istitle()
                features[f"{prefix}:word.isdigit"] = prev_word.isdigit()
                features[f"{prefix}:word.shape"] = word_shape(prev_word)
        else:
            features[f"BOS{offset}"] = True

        next_index = index + offset
        if next_index < len(sentence):
            next_word, next_pos, next_chunk, _ = sentence[next_index]
            prefix = f"+ {offset}"
            if config.include_lexical:
                features[f"{prefix}:word.lower"] = next_word.lower()
                features[f"{prefix}:pos"] = next_pos
                features[f"{prefix}:chunk"] = next_chunk
            if config.include_shape:
                features[f"{prefix}:word.isupper"] = next_word.isupper()
                features[f"{prefix}:word.istitle"] = next_word.istitle()
                features[f"{prefix}:word.isdigit"] = next_word.isdigit()
                features[f"{prefix}:word.shape"] = word_shape(next_word)
        else:
            features[f"EOS{offset}"] = True

    return features