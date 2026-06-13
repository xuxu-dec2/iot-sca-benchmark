"""Visualization for DL-SCA experiment results."""

import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import RESULTS_DIR, DATASET_CONFIGS, MODEL_CONFIGS, EVAL_CONFIG

sns.set_style("whitegrid")
plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "figure.dpi": 150,
})


def load_summary():
    """Load the merged experiment summary."""
    path = os.path.join(RESULTS_DIR, "summary.json")
    if not os.path.exists(path):
        # Load individual run results
        return _load_from_runs()
    with open(path) as f:
        return json.load(f)


def _load_from_runs():
    """Load and merge individual run files."""
    from src.config import DATASET_CONFIGS, MODEL_CONFIGS
    summaries = []
    import glob
    for model in MODEL_CONFIGS:
        for dataset in DATASET_CONFIGS:
            pattern = os.path.join(RESULTS_DIR,
                                   f"{model}_{dataset}_run*.json")
            files = glob.glob(pattern)
            if not files:
                continue
            all_metrics = []
            for fp in files:
                with open(fp) as f:
                    all_metrics.extend(json.load(f))

            ttrs = [m["traces_to_recovery"] for m in all_metrics
                    if m["traces_to_recovery"] is not None]
            ge_len = len(all_metrics[0]["ge_curve"])
            ge_matrix = [m["ge_curve"] for m in all_metrics]

            summaries.append({
                "model": model,
                "dataset": dataset,
                "description": DATASET_CONFIGS[dataset]["description"],
                "n_runs": len(files),
                "success_count": f"{sum(1 for m in all_metrics if m['success'])}/{len(files)}",
                "ttr_mean": sum(ttrs) / len(ttrs) if ttrs else None,
                "ttr_min": min(ttrs) if ttrs else None,
                "ttr_max": max(ttrs) if ttrs else None,
                "ge_curve_mean": [float(np.mean([g[i] for g in ge_matrix]))
                                   for i in range(ge_len)],
            })
    return summaries


def plot_ge_comparison(summaries, output_path=None):
    """Plot GE curves comparing all models on a given dataset."""
    if output_path is None:
        output_path = os.path.join(RESULTS_DIR, "ge_comparison.png")

    datasets = sorted(set(s["dataset"] for s in summaries))
    models = sorted(set(s["model"] for s in summaries))
    eval_cfg = EVAL_CONFIG

    fig, axes = plt.subplots(1, len(datasets), figsize=(5 * len(datasets), 4),
                              squeeze=False)

    for ax_idx, dataset in enumerate(datasets):
        ax = axes[0, ax_idx]
        dataset_sums = [s for s in summaries if s["dataset"] == dataset]

        for s in dataset_sums:
            ge = s["ge_curve_mean"]
            x = np.arange(1, len(ge) + 1) * eval_cfg["ge_step"]
            ax.semilogy(x, np.maximum(ge, 0.1), label=s["model"], linewidth=1.5)

        ax.axhline(y=1, color="gray", linestyle="--", alpha=0.5, label="GE=1")
        ax.set_xlabel("Attack Traces")
        ax.set_ylabel("Guessing Entropy")
        desc = DATASET_CONFIGS.get(dataset, {}).get("description", dataset)
        ax.set_title(desc, fontsize=11)
        ax.legend(fontsize=9)
        ax.set_xlim(0, eval_cfg["ge_max_traces"])

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"GE comparison saved to {output_path}")


def plot_ttr_heatmap(summaries, output_path=None):
    """Plot Traces-to-Recovery heatmap (model × dataset)."""
    if output_path is None:
        output_path = os.path.join(RESULTS_DIR, "ttr_heatmap.png")

    datasets = sorted(set(s["dataset"] for s in summaries))
    models = sorted(set(s["model"] for s in summaries))

    ttr_matrix = np.zeros((len(models), len(datasets)))
    annot = []

    for i, model in enumerate(models):
        row_annot = []
        for j, dataset in enumerate(datasets):
            s = [s_ for s_ in summaries
                 if s_["model"] == model and s_["dataset"] == dataset]
            if s and s[0]["ttr_mean"] is not None:
                ttr_matrix[i, j] = s[0]["ttr_mean"]
                row_annot.append(f"{s[0]['ttr_mean']:.0f}")
            else:
                ttr_matrix[i, j] = EVAL_CONFIG["ge_max_traces"]
                row_annot.append("N/A")
        annot.append(row_annot)

    fig, ax = plt.subplots(figsize=(len(datasets) * 1.5 + 2, len(models) * 0.8 + 1))
    im = ax.imshow(ttr_matrix, cmap="RdYlGn_r", aspect="auto")

    for i in range(len(models)):
        for j in range(len(datasets)):
            color = "white" if ttr_matrix[i, j] > EVAL_CONFIG["ge_max_traces"] / 2 else "black"
            ax.text(j, i, annot[i][j], ha="center", va="center", fontsize=11,
                    fontweight="bold", color=color)

    ax.set_xticks(range(len(datasets)))
    ax.set_xticklabels(datasets, rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models, fontsize=10)
    ax.set_title("Traces to Recovery (GE=1)\nLower = better", fontsize=13)
    plt.colorbar(im, ax=ax, label="TtR")

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"TtR heatmap saved to {output_path}")


def plot_all():
    """Generate all paper-ready figures."""
    summaries = load_summary()
    if not summaries:
        print("No results found. Run experiments first.")
        return

    plot_ge_comparison(summaries)
    plot_ttr_heatmap(summaries)


if __name__ == "__main__":
    plot_all()
