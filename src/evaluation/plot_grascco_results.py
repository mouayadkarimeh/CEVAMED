from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt


METRIC_LABELS = {
    "iou_mean": "IoU",
    "recall_mean": "Recall",
    "precision_mean": "Precision",
    "precision_omega_mean": "Precision Omega",
    "f1_mean": "F1",
    "hit_at_k_mean": "Hit@k",
    "mrr_mean": "MRR",
    "ndcg_at_k_mean": "nDCG@k",
    "fragmentation_mean": "Fragmentation",
}


def _metric_keys(results: dict) -> list[str]:
    profiles = list(results["profiles"].keys())
    first_profile = profiles[0]
    first_chunker = list(results["profiles"][first_profile].keys())[0]
    all_keys = list(results["profiles"][first_profile][first_chunker].keys())
    return [key for key in all_keys if key.endswith("_mean")]


def _grid_dimensions(metric_count: int) -> tuple[int, int]:
    cols = 3
    rows = int(math.ceil(metric_count / cols))
    return rows, cols


def _load_results(results_path: Path) -> dict:
    with results_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _plot_grouped_bars_for_profile(profile_name: str, profile_data: dict, output_dir: Path, metric_keys: list[str]) -> None:
    chunkers = list(profile_data.keys())
    x_positions = list(range(len(chunkers)))

    rows, cols = _grid_dimensions(len(metric_keys))
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3.8 * rows), constrained_layout=True)
    axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for axis, metric in zip(axes_flat, metric_keys):
        values = [float(profile_data[chunker][metric]) for chunker in chunkers]
        display_values = [value * 100 for value in values]
        axis.bar(x_positions, display_values)
        axis.set_title(METRIC_LABELS.get(metric, metric))
        axis.set_xticks(x_positions)
        axis.set_xticklabels(chunkers, rotation=20, ha="right")
        y_upper = max(display_values) * 1.2 if display_values else 1.0
        axis.set_ylim(0, max(1.0, y_upper))
        axis.set_ylabel("Prozent")
        axis.grid(axis="y", linestyle="--", alpha=0.35)

        for idx, value in enumerate(display_values):
            axis.text(idx, value + max(y_upper * 0.02, 0.02), f"{value:.2f}%", ha="center", va="bottom", fontsize=8)

    for axis in axes_flat[len(metric_keys):]:
        axis.axis("off")

    fig.suptitle(f"GRASCCO Evaluation - {profile_name.upper()}", fontsize=14)
    output_file = output_dir / f"grascco_{profile_name}_metrics.png"
    fig.savefig(output_file, dpi=150)
    plt.close(fig)


def _plot_metric_heatmaps(results: dict, output_dir: Path, metric_keys: list[str]) -> None:
    profiles = list(results["profiles"].keys())
    chunkers = list(results["profiles"][profiles[0]].keys())

    rows, cols = _grid_dimensions(len(metric_keys))
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3.8 * rows), constrained_layout=True)
    axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for axis, metric in zip(axes_flat, metric_keys):
        matrix = []
        for profile in profiles:
            row = [float(results["profiles"][profile][chunker][metric]) for chunker in chunkers]
            matrix.append(row)

        image = axis.imshow(matrix, aspect="auto", cmap="YlGnBu")
        axis.set_title(METRIC_LABELS.get(metric, metric))
        axis.set_xticks(range(len(chunkers)))
        axis.set_xticklabels(chunkers, rotation=20, ha="right")
        axis.set_yticks(range(len(profiles)))
        axis.set_yticklabels(profiles)

        for y in range(len(profiles)):
            for x in range(len(chunkers)):
                axis.text(x, y, f"{matrix[y][x] * 100:.2f}%", ha="center", va="center", fontsize=8)

        fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)

    for axis in axes_flat[len(metric_keys):]:
        axis.axis("off")

    fig.suptitle("GRASCCO Evaluation - Metric Heatmaps", fontsize=14)
    output_file = output_dir / "grascco_metric_heatmaps.png"
    fig.savefig(output_file, dpi=150)
    plt.close(fig)


def build_plots() -> list[Path]:
    repo_root = Path(__file__).resolve().parents[2]
    results_path = repo_root / "data" / "processed" / "grascco_eval_results.json"
    output_dir = repo_root / "data" / "processed" / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    results = _load_results(results_path)
    metric_keys = _metric_keys(results)

    generated_files: list[Path] = []
    for profile_name, profile_data in results["profiles"].items():
        _plot_grouped_bars_for_profile(profile_name, profile_data, output_dir, metric_keys)
        generated_files.append(output_dir / f"grascco_{profile_name}_metrics.png")

    _plot_metric_heatmaps(results, output_dir, metric_keys)
    generated_files.append(output_dir / "grascco_metric_heatmaps.png")

    return generated_files


def main() -> None:
    files = build_plots()
    print("Generated plot files:")
    for file in files:
        print(file)


if __name__ == "__main__":
    main()
