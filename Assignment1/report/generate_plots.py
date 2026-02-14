from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt


def read_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def plot_task1_loss_latency(result_dir: Path, plot_dir: Path) -> None:
    summary = read_json(result_dir / "task1_experiments_summary.json")
    experiments = summary["experiments"]
    run_ids = [exp["run_id"] for exp in experiments]
    losses = [exp["final_loss"] for exp in experiments]
    latencies = [exp["latency_seconds"] for exp in experiments]

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    axes[0].plot(run_ids, losses, marker="o")
    axes[0].set_ylabel("Final Loss")
    axes[0].set_title("Task 1: Final Loss by Run")
    axes[0].tick_params(axis="x", rotation=45)

    axes[1].plot(run_ids, latencies, marker="o", color="tab:orange")
    axes[1].set_ylabel("Latency (s)")
    axes[1].set_title("Task 1: Latency by Run")
    axes[1].set_xlabel("Run ID")
    axes[1].tick_params(axis="x", rotation=45)

    fig.tight_layout()
    fig.savefig(plot_dir / "task1_loss_latency.png", dpi=200)
    plt.close(fig)


def plot_task1_loss_curves(output_task1_dir: Path, plot_dir: Path) -> None:
    training_files = list(output_task1_dir.glob("glove_*_training.json"))
    if not training_files:
        return

    fig = plt.figure(figsize=(10, 6))
    for path in sorted(training_files):
        data = read_json(path)
        history = data.get("loss_history")
        if not history:
            continue
        label = path.stem.replace("_training", "")
        plt.plot(history, label=label)

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Task 1: Loss Curves")
    plt.legend(fontsize=7)
    plt.tight_layout()
    fig.savefig(plot_dir / "task1_loss_curves.png", dpi=200)
    plt.close(fig)


def plot_task4_metrics(result_dir: Path, plot_dir: Path) -> None:
    glove_metrics = []
    svd_metrics = []
    dims = [50, 100, 200, 300]

    for dim in dims:
        glove_path = result_dir / f"task4_metrics_d{dim}.json"
        svd_path = result_dir / f"task4_svd_metrics_d{dim}.json"
        if glove_path.exists():
            data = read_json(glove_path)
            test = data["test"]["classification_report"]
            glove_metrics.append(
                {
                    "dim": dim,
                    "accuracy": test["accuracy"],
                    "macro_f1": test["macro avg"]["f1-score"],
                }
            )
        if svd_path.exists():
            data = read_json(svd_path)
            test = data["test"]["classification_report"]
            svd_metrics.append(
                {
                    "dim": dim,
                    "accuracy": test["accuracy"],
                    "macro_f1": test["macro avg"]["f1-score"],
                }
            )

    if glove_metrics:
        fig = plt.figure(figsize=(8, 5))
        plt.plot([m["dim"] for m in glove_metrics], [m["macro_f1"] for m in glove_metrics], marker="o", label="GloVe")
        if svd_metrics:
            plt.plot([m["dim"] for m in svd_metrics], [m["macro_f1"] for m in svd_metrics], marker="o", label="SVD")
        plt.xlabel("Embedding Dimension")
        plt.ylabel("Test Macro-F1")
        plt.title("Task 4: Macro-F1 vs Dimension")
        plt.legend()
        plt.tight_layout()
        fig.savefig(plot_dir / "task4_macro_f1.png", dpi=200)
        plt.close(fig)

        fig = plt.figure(figsize=(8, 5))
        plt.plot([m["dim"] for m in glove_metrics], [m["accuracy"] for m in glove_metrics], marker="o", label="GloVe")
        if svd_metrics:
            plt.plot([m["dim"] for m in svd_metrics], [m["accuracy"] for m in svd_metrics], marker="o", label="SVD")
        plt.xlabel("Embedding Dimension")
        plt.ylabel("Test Accuracy")
        plt.title("Task 4: Accuracy vs Dimension")
        plt.legend()
        plt.tight_layout()
        fig.savefig(plot_dir / "task4_accuracy.png", dpi=200)
        plt.close(fig)


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    result_dir = base_dir / "result_json"
    plot_dir = base_dir / "plots"
    ensure_dir(plot_dir)

    plot_task1_loss_latency(result_dir, plot_dir)
    plot_task1_loss_curves(base_dir.parent / "output" / "task1", plot_dir)
    plot_task4_metrics(result_dir, plot_dir)

    print("Plots written to", plot_dir)


if __name__ == "__main__":
    main()
