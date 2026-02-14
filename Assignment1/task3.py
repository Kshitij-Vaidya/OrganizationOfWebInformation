from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

from tqdm import tqdm

from datasets import load_dataset

from utils import CRFFeatureConfig, sentence_to_features, sentence_to_labels

import sklearn_crfsuite
from sklearn_crfsuite import metrics

DATASET = "conll2003"
OUTPUT_DIR = "output/task3"
C1 = 0.1
C2 = 0.1
MAX_ITERATIONS = 200
WINDOW_SIZE = 2
TOP_K = 10

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CRF NER model for CoNLL-2003")
    parser.add_argument(
        "--run",
        action="store_true",
        help="Train only the full feature model (skip experiments).",
    )
    return parser.parse_args()


def build_dataset(
    sentences: Iterable[list[tuple[str, str, str, str]]],
    config: CRFFeatureConfig,
) -> tuple[list[list[dict[str, object]]], list[list[str]]]:
    features: list[list[dict[str, object]]] = []
    labels: list[list[str]] = []
    for sentence in tqdm(list(sentences), desc="Extracting features"):
        features.append(sentence_to_features(sentence, config))
        labels.append(sentence_to_labels(sentence))
    return features, labels


def train_crf(
    x_train: list[list[dict[str, object]]],
    y_train: list[list[str]],
) -> sklearn_crfsuite.CRF:
    model = sklearn_crfsuite.CRF(
        algorithm="lbfgs",
        c1=C1,
        c2=C2,
        max_iterations=MAX_ITERATIONS,
        all_possible_transitions=True,
        all_possible_states=True,
    )
    model.fit(x_train, y_train)
    return model


def evaluate_model(
    model: sklearn_crfsuite.CRF,
    x_data: list[list[dict[str, object]]],
    y_data: list[list[str]],
    labels: list[str],
) -> dict[str, object]:
    y_pred = model.predict(x_data)
    f1 = metrics.flat_f1_score(y_data, y_pred, average="weighted", labels=labels)
    report = metrics.flat_classification_report(
        y_data,
        y_pred,
        labels=labels,
        digits=4,
        zero_division=0,
        output_dict=True,
    )
    return {"weighted_f1": float(f1), "classification_report": report}

# Feature templates were determined and compiled using Copilot
def collect_feature_templates(config: CRFFeatureConfig) -> list[str]:
    templates = ["bias"]
    if config.include_lexical:
        templates += ["word.lower", "pos", "chunk", "pos[:2]"]
    if config.include_shape:
        templates += [
            "word.isupper",
            "word.istitle",
            "word.isdigit",
            "word.shape",
            "word.has_digit",
            "word.has_hyphen",
            "word.has_period",
        ]
    if config.include_subword:
        templates += ["pref2", "pref3", "pref4", "suf2", "suf3", "suf4"]

    for offset in range(1, config.window_size + 1):
        if config.include_lexical:
            templates += [
                f"- {offset}:word.lower",
                f"- {offset}:pos",
                f"- {offset}:chunk",
                f"+ {offset}:word.lower",
                f"+ {offset}:pos",
                f"+ {offset}:chunk",
            ]
        if config.include_shape:
            templates += [
                f"- {offset}:word.isupper",
                f"- {offset}:word.istitle",
                f"- {offset}:word.isdigit",
                f"- {offset}:word.shape",
                f"+ {offset}:word.isupper",
                f"+ {offset}:word.istitle",
                f"+ {offset}:word.isdigit",
                f"+ {offset}:word.shape",
            ]
        templates += [f"BOS{offset}", f"EOS{offset}"]
    return templates


def extract_feature_importance(
    model: sklearn_crfsuite.CRF,
    top_k: int,
) -> dict[str, list[tuple[str, str, float]]]:
    state_features = sorted(
        ((attr, label, weight) for (attr, label), weight in model.state_features_.items()),
        key=lambda item: abs(item[2]),
        reverse=True,
    )
    transition_features = sorted(
        ((from_tag, to_tag, weight) for (from_tag, to_tag), weight in model.transition_features_.items()),
        key=lambda item: abs(item[2]),
        reverse=True,
    )
    return {
        "state_features": state_features[:top_k],
        "transition_features": transition_features[:top_k],
    }


def run_single_experiment(
    name: str,
    config: CRFFeatureConfig,
    train_sentences: list[list[tuple[str, str, str, str]]],
    dev_sentences: list[list[tuple[str, str, str, str]]],
    test_sentences: list[list[tuple[str, str, str, str]]],
    labels: list[str],
) -> dict[str, object]:
    x_train, y_train = build_dataset(train_sentences, config)
    x_dev, y_dev = build_dataset(dev_sentences, config)
    x_test, y_test = build_dataset(test_sentences, config)

    start = time.perf_counter()
    model = train_crf(x_train, y_train)
    latency = time.perf_counter() - start

    dev_metrics = evaluate_model(model, x_dev, y_dev, labels)
    test_metrics = evaluate_model(model, x_test, y_test, labels)

    return {
        "name": name,
        "config": asdict(config),
        "training_seconds": float(latency),
        "dev": dev_metrics,
        "test": test_metrics,
        "model": model,
    }


def main() -> None:
    args = parse_args()
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(DATASET)
    ner_feature = dataset["train"].features["ner_tags"]
    pos_feature = dataset["train"].features["pos_tags"]
    chunk_feature = dataset["train"].features["chunk_tags"]

    def split_to_sentences(split):
        sentences = []
        for record in split:
            tokens = record["tokens"]
            pos_tags = [pos_feature.feature.names[idx] for idx in record["pos_tags"]]
            chunk_tags = [chunk_feature.feature.names[idx] for idx in record["chunk_tags"]]
            ner_tags = [ner_feature.feature.names[idx] for idx in record["ner_tags"]]
            sentences.append(list(zip(tokens, pos_tags, chunk_tags, ner_tags)))
        return sentences

    train_sentences = split_to_sentences(dataset["train"])
    dev_sentences = split_to_sentences(dataset["validation"])
    test_sentences = split_to_sentences(dataset["test"])

    labels = [label for label in ner_feature.feature.names if label != "O"]
    # Run experiments using partial or complete set of CRF features to check for relative feature importances
    experiment_configs = [
        (
            "lexical_only",
            CRFFeatureConfig(
                include_lexical=True,
                include_shape=False,
                include_subword=False,
                window_size=WINDOW_SIZE,
            ),
        ),
        (
            "lexical_shape",
            CRFFeatureConfig(
                include_lexical=True,
                include_shape=True,
                include_subword=False,
                window_size=WINDOW_SIZE,
            ),
        ),
        (
            "lexical_shape_subword",
            CRFFeatureConfig(
                include_lexical=True,
                include_shape=True,
                include_subword=True,
                window_size=WINDOW_SIZE,
            ),
        ),
    ]

    results: list[dict[str, object]] = []
    for name, config in experiment_configs:
        if args.run and name != "lexical_shape_subword":
            continue
        result = run_single_experiment(
            name,
            config,
            train_sentences,
            dev_sentences,
            test_sentences,
            labels,
        )
        results.append(result)

    summary = []
    for result in results:
        summary.append(
            {
                "name": result["name"],
                "config": result["config"],
                "training_seconds": result["training_seconds"],
                "dev": result["dev"],
                "test": result["test"],
            }
        )

    with (output_dir / "task3_experiments_summary.json").open("w", encoding="utf-8") as handle:
        json.dump({"experiments": summary}, handle, indent=2)

    best = max(results, key=lambda item: item["dev"]["weighted_f1"])
    best_model = best["model"]
    best_config = CRFFeatureConfig(**best["config"])

    feature_report = {
        "feature_templates": collect_feature_templates(best_config),
        "top_features": extract_feature_importance(best_model, TOP_K),
    }

    with (output_dir / "task3_feature_report.json").open("w", encoding="utf-8") as handle:
        json.dump(feature_report, handle, indent=2)

    with (output_dir / "task3_best_config.json").open("w", encoding="utf-8") as handle:
        json.dump(best["config"], handle, indent=2)

    print("Task 3 complete. Results written to", output_dir)


if __name__ == "__main__":
    main()
