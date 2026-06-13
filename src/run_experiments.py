"""Batch experiment runner — iterates over all model × dataset combinations."""

import os
import sys
import json
import numpy as np
import subprocess
from itertools import product

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import DATASET_CONFIGS, MODEL_CONFIGS, TRAINING_CONFIG, RESULTS_DIR


# Datasets available for experiments (skip ascad_v2 if not downloaded)
DATASETS = [
    "ascad_v1_fixed",
    "ascad_v1_desync50",
    "ascad_v1_desync100",
    # "ascad_v2",  # uncomment after download
]

MODELS = ["mlp", "cnn", "resnet"]

NUM_RUNS = TRAINING_CONFIG["num_runs"]


def run_experiment(model, dataset, run_id):
    """Run a single training job via subprocess."""
    cmd = [
        "python", "-m", "src.train",
        "--model", model,
        "--dataset", dataset,
        "--runs", "1",
        "--output", f"results/{model}_{dataset}_run{run_id}.json",
    ]
    print(f"\n{'#' * 60}")
    print(f"# {model} × {dataset} (run {run_id + 1}/{NUM_RUNS})")
    print(f"{'#' * 60}")

    result = subprocess.run(
        cmd,
        cwd=os.path.dirname(os.path.dirname(__file__)),
        capture_output=False,
    )
    return result.returncode == 0


def merge_results(model, dataset):
    """Merge individual run results into a single summary."""
    all_metrics = []
    for run_id in range(NUM_RUNS):
        path = os.path.join(RESULTS_DIR,
                            f"{model}_{dataset}_run{run_id}.json")
        if os.path.exists(path):
            with open(path) as f:
                metrics = json.load(f)
                all_metrics.extend(metrics)

    if not all_metrics:
        return None

    # Compute summary statistics
    ttrs = [m["traces_to_recovery"] for m in all_metrics if m["traces_to_recovery"] is not None]
    success_count = sum(1 for m in all_metrics if m["success"])

    summary = {
        "model": model,
        "dataset": dataset,
        "description": DATASET_CONFIGS[dataset]["description"],
        "n_runs": NUM_RUNS,
        "success_count": f"{success_count}/{NUM_RUNS}",
        "ttr_mean": sum(ttrs) / len(ttrs) if ttrs else None,
        "ttr_min": min(ttrs) if ttrs else None,
        "ttr_max": max(ttrs) if ttrs else None,
        "avg_train_time_s": sum(m["train_time_s"] for m in all_metrics) / len(all_metrics),
        "avg_epochs": sum(m["epochs_trained"] for m in all_metrics) / len(all_metrics),
    }

    # Average GE curve across runs
    ge_len = len(all_metrics[0]["ge_curve"])
    ge_matrix = [m["ge_curve"] for m in all_metrics]
    summary["ge_curve_mean"] = [float(np.mean([g[i] for g in ge_matrix]))
                                 for i in range(ge_len)]

    return summary


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=" * 70)
    print("  EXPERIMENT MATRIX")
    print("=" * 70)
    print(f"  Models:  {MODELS}")
    print(f"  Datasets: {DATASETS}")
    print(f"  Runs per experiment: {NUM_RUNS}")
    print(f"  Total jobs: {len(MODELS) * len(DATASETS) * NUM_RUNS}")
    print("=" * 70)

    # Run all experiments
    for model, dataset in product(MODELS, DATASETS):
        for run_id in range(NUM_RUNS):
            success = run_experiment(model, dataset, run_id)
            if not success:
                print(f"  [FAILED] {model} × {dataset} run {run_id}")

    # Generate summary table
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)

    summaries = []
    for model, dataset in product(MODELS, DATASETS):
        s = merge_results(model, dataset)
        if s:
            summaries.append(s)

    # Print table
    print(f"\n{'Model':<10} {'Dataset':<25} {'Success':<10} {'TtR Mean':<12} {'TtR Min':<10} {'Avg Time (s)':<14}")
    print("-" * 85)
    for s in summaries:
        ttr_mean = f"{s['ttr_mean']:.0f}" if s['ttr_mean'] else "N/A"
        ttr_min = f"{s['ttr_min']:.0f}" if s['ttr_min'] else "N/A"
        print(f"{s['model']:<10} {s['dataset']:<25} {s['success_count']:<10} "
              f"{ttr_mean:<12} {ttr_min:<10} {s['avg_train_time_s']:<14.1f}")

    # Save summary
    summary_path = os.path.join(RESULTS_DIR, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summaries, f, indent=2)
    print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    main()
