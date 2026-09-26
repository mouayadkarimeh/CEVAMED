from __future__ import annotations

import json
import math
from pathlib import Path
import csv
import matplotlib.pyplot as plt
import numpy as np

# Alle möglichen Metriken mit benutzerfreundlichen Labels
METRIC_LABELS = {
    # Klassische Metriken
    "precision_at_k_global_mean": "Precision@K",
    "recall_at_k_global_mean": "Recall@K",

    # RAGAS Metriken
    "ragas_precision_global_mean": "RAGAS Precision",
    "ragas_recall_global_mean": "RAGAS Recall",

    # Strukturelle Metriken
    "block_integrity_global_mean": "Block Integrity",
    "reference_completeness_global_mean": "Reference Completeness",
    "context_compactness_global_mean": "Context Compactness",
    "intrachunk_cohesion_global_mean": "Intrachunk Cohesion (ICC)",
}

# Farbschema für die Plots
COLOR_PALETTE = {
    "character_text_splitter": "#1f77b4",  # Blau
    "transformer_token_chunker": "#ff7f0e",  # Orange
    "recursive_token_chunker": "#2ca02c",  # Grün
    "kamradt_semantic_chunker": "#d62728",  # Rot
    "recursive_semantic_chunker": "#9467bd",  # Lila
    "cluster_semantic_chunker": "#8c564b"   # Braun
}

CHUNKER_LABELS = {
    "character_text_splitter": "Character",
    "transformer_token_chunker": "Transformer Token",
    "recursive_token_chunker": "Recursive Token",
    "kamradt_semantic_chunker": "Kamradt Semantic",
    "recursive_semantic_chunker": "Recursive Semantic",
    "cluster_semantic_chunker": "Cluster Semantic",
}


def _style_percentage_axis(axis: plt.Axes) -> None:
    axis.set_xlim(0, 100)
    #axis.set_xlabel("Score (%)", fontsize=9)
    axis.set_axisbelow(True)
    axis.grid(axis="x", linestyle=":", linewidth=0.8, alpha=0.45)
    axis.spines[["top", "right", "left"]].set_visible(False)
    axis.tick_params(axis="y", length=0)


def _add_bar_labels(axis: plt.Axes, bars: object, values: list[float]) -> None:
    for bar, value in zip(bars, values):
        if not math.isfinite(value):
            axis.text(
                1.2,
                bar.get_y() + bar.get_height() / 2,
                "n/a",
                ha="left",
                va="center",
                fontsize=8,
                color="#666666",
                fontstyle="italic",
            )
            continue
        axis.text(
            min(value + 1.2, 98.5),
            bar.get_y() + bar.get_height() / 2,
            f"{value:.1f}%",
            ha="right" if value >= 96 else "left",
            va="center",
            fontsize=8,
            fontweight="bold",
        )

def _load_results(results_path: Path) -> dict:
    """Lädt die Evaluationsergebnisse aus JSON"""
    with results_path.open("r", encoding="utf-8") as file:
        results = json.load(file)

    if "profiles" in results:
        return results

    profiles: dict[str, dict[str, dict]] = {}
    for chunker_name, chunker_profiles in results.items():
        for profile_name, profile_data in chunker_profiles.items():
            profiles.setdefault(profile_name, {})[chunker_name] = profile_data.get(
                "global_metrics", profile_data
            )
    return {"profiles": profiles}

def _get_all_metric_keys(results: dict) -> list[str]:
    """
    Erhält ALLE Metrik-Schlüssel aus den Ergebnissen.
    Berücksichtigt sowohl klassische als auch strukturelle Metriken.
    """
    profiles = list(results["profiles"].keys())
    if not profiles:
        return []

    all_metrics = []

    # Durchsuche alle Metriken in den Daten
    for profile in results["profiles"].values():
        for chunker_data in profile.values():
            for metric_key in chunker_data.keys():
                if metric_key.endswith("_global_mean"):
                    if metric_key not in all_metrics:
                        all_metrics.append(metric_key)

    # Sortiere die Metriken nach Kategorie
    klassische = ["precision_at_k_global_mean", "recall_at_k_global_mean"]
    ragas = ["ragas_precision_global_mean", "ragas_recall_global_mean"]
    strukturell = ["block_integrity_global_mean", "reference_completeness_global_mean",
                  "context_compactness_global_mean", "intrachunk_cohesion_global_mean"]

    return [m for m in klassische + ragas + strukturell if m in all_metrics]

def _grid_dimensions(metric_count: int) -> tuple[int, int]:
    """Berechnet optimale Grid-Dimensionen für die Plots"""
    cols = 2  # 2 Spalten für bessere Lesbarkeit
    rows = int(math.ceil(metric_count / cols))
    return rows, cols

def _plot_grouped_bars_for_profile(
    profile_name: str,
    profile_data: dict,
    output_dir: Path,
    metric_keys: list[str]
) -> None:
    """
    Erstellt gruppierte Balkendiagramme für alle Metriken eines Profils.
    Jeder Balken repräsentiert einen Chunker.
    """
    chunkers = list(profile_data.keys())
    y_positions = list(range(len(chunkers)))
    chunker_labels = [CHUNKER_LABELS.get(chunker, chunker) for chunker in chunkers]

    rows, cols = _grid_dimensions(len(metric_keys))
    fig, axes = plt.subplots(
        rows, cols,
        figsize=(8 * cols, 2.8 * rows),
        constrained_layout=True
    )
    axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for axis, metric in zip(axes_flat, metric_keys):
        values = []
        for chunker in chunkers:
            value = float(profile_data[chunker].get(metric, 0.0))
            values.append(value * 100)  # Umrechnung in Prozent

        bars = axis.barh(
            y_positions,
            values,
            color=[COLOR_PALETTE.get(chunker, "#7f7f7f") for chunker in chunkers],
            height=0.62,
        )

        axis.set_title(METRIC_LABELS.get(metric, metric), fontsize=10, fontweight="bold")
        axis.set_yticks(y_positions, labels=chunker_labels, fontsize=8)
        axis.invert_yaxis()
        _style_percentage_axis(axis)
        _add_bar_labels(axis, bars, values)

    # Unnötige Subplots ausblenden
    for axis in axes_flat[len(metric_keys):]:
        axis.axis("off")

    fig.suptitle(
        f"GRASCCO Evaluation - {profile_name.upper()}",
        fontsize=14,
        fontweight="bold",
        y=1.02
    )
    output_file = output_dir / f"grascco_{profile_name}_all_metrics.png"
    fig.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close(fig)

def _plot_metric_comparison_heatmap(
    results: dict,
    output_dir: Path,
    metric_keys: list[str]
) -> None:
    """
    Erstellt eine Heatmap, die alle Chunker und Profile für jede Metrik vergleicht.
    Zeigt die Performance aller Chunker über alle Profile hinweg.
    """
    profiles = list(results["profiles"].keys())
    chunkers = list(results["profiles"][profiles[0]].keys())

    rows, cols = _grid_dimensions(len(metric_keys))
    fig, axes = plt.subplots(
        rows, cols,
        figsize=(8 * cols, 2.8 * rows),
        constrained_layout=True
    )
    axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for axis, metric in zip(axes_flat, metric_keys):
        # Matrix erstellen: Zeilen = Profile, Spalten = Chunker
        matrix = []
        for profile in profiles:
            row = []
            for chunker in chunkers:
                value = float(results["profiles"][profile][chunker].get(metric, 0.0))
                row.append(value * 100)  # Umrechnung in Prozent
            matrix.append(row)

        # Heatmap plotten
        image = axis.imshow(
            matrix,
            aspect="auto",
            cmap="YlGnBu",
            vmin=0,
            vmax=100
        )

        # Beschriftungen
        axis.set_title(
            METRIC_LABELS.get(metric, metric),
            fontsize=10,
            fontweight="bold"
        )
        axis.set_xticks(range(len(chunkers)))
        axis.set_xticklabels(
            [
                CHUNKER_LABELS.get(chunker, chunker).replace(" ", "\n")
                for chunker in chunkers
            ],
            rotation=0,
            ha="center",
            fontsize=7,
        )
        axis.set_yticks(range(len(profiles)))
        axis.set_yticklabels(
            [p.upper() for p in profiles],
            fontsize=9
        )

        # Prozentwerte in den Zellen anzeigen
        for y in range(len(profiles)):
            for x in range(len(chunkers)):
                value = matrix[y][x]
                axis.text(
                    x, y,
                    f"{value:.1f}%",
                    ha="center", va="center",
                    fontsize=8,
                    color="white" if value > 50 else "black"
                )

        fig.colorbar(
            image,
            ax=axis,
            fraction=0.046,
            pad=0.04,
            label="Prozent"
        )

    # Unnötige Subplots ausblenden
    for axis in axes_flat[len(metric_keys):]:
        axis.axis("off")

    fig.suptitle(
        "GRASCCO Evaluation - Metric Comparison Heatmaps",
        fontsize=14,
        fontweight="bold",
        y=1.02
    )
    output_file = output_dir / "grascco_all_metrics_heatmap.png"
    fig.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close(fig)

def _plot_aggregated_metrics(
    results: dict,
    output_dir: Path,
    metric_keys: list[str]
) -> None:
    """
    Erstellt aggregierte Plots, die alle Metriken über alle Chunker hinweg vergleichen.
    Zeigt die durchschnittliche Performance aller Chunker pro Metrik.
    """
    chunkers = list(results["profiles"][list(results["profiles"].keys())[0]].keys())

    # Daten vorbereiten
    aggregated_data = {metric: [] for metric in metric_keys}
    for profile_data in results["profiles"].values():
        for chunker in chunkers:
            for metric in metric_keys:
                value = float(profile_data[chunker].get(metric, 0.0))
                aggregated_data[metric].append(value * 100)

    # Durchschnittswerte berechnen
    avg_values = {
        metric: np.mean(values) if values else 0
        for metric, values in aggregated_data.items()
    }

    fig, axis = plt.subplots(figsize=(10, 7), constrained_layout=True)

    # Balken plotten
    labels = [METRIC_LABELS.get(metric, metric) for metric in avg_values]
    values = list(avg_values.values())
    y_positions = np.arange(len(labels))
    bars = axis.barh(
        y_positions,
        values,
        color="#2ca02c",  # Grün für aggregierte Werte
        height=0.62,
    )

    # Beschriftungen
    axis.set_title(
        "Durchschnittliche Performance aller Chunker",
        fontsize=12,
        fontweight="bold"
    )
    axis.set_yticks(y_positions, labels=labels, fontsize=9)
    axis.invert_yaxis()
    _style_percentage_axis(axis)
    _add_bar_labels(axis, bars, values)

    output_file = output_dir / "grascco_aggregated_metrics.png"
    fig.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close(fig)

def build_plots() -> list[Path]:
    """Hauptfunktion zum Erstellen aller Plots"""
    repo_root = Path(__file__).resolve().parents[2]
    results_path = repo_root / "data" / "processed" / "grascco_multi_5_docs_report.json"
    output_dir = repo_root / "data" / "processed" / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    results = _load_results(results_path)
    metric_keys = _get_all_metric_keys(results)

    if not metric_keys:
        raise ValueError("Keine Metriken gefunden in den Ergebnissen!")

    generated_files: list[Path] = []

    # 1. Balkendiagramme pro Profil
    for profile_name, profile_data in results["profiles"].items():
        _plot_grouped_bars_for_profile(profile_name, profile_data, output_dir, metric_keys)
        generated_files.append(output_dir / f"grascco_{profile_name}_all_metrics.png")

    # 2. Vergleichs-Heatmap
    _plot_metric_comparison_heatmap(results, output_dir, metric_keys)
    generated_files.append(output_dir / "grascco_all_metrics_heatmap.png")

    # 3. Aggregierte Metriken
    _plot_aggregated_metrics(results, output_dir, metric_keys)
    generated_files.append(output_dir / "grascco_aggregated_metrics.png")

    return generated_files




def plot_csv(datei: Path) -> None:
    x_chunker = []
    # Wörterbuch, das für jede Metrik eine Liste von Werten speichert
    metriken_daten = {}
    
    with open(datei, "r", encoding="utf-8") as f:
        text = csv.reader(f)
        header = next(text)  # Überspringt die Kopfzeile und speichert die Spaltennamen
        
        # Alle Metriken ab Index 2 dynamisch erfassen
        metriken_namen = header[2:]
        for name in metriken_namen:
            metriken_daten[name] = []
            
        for row in text:
            # Nur Zeilen verarbeiten, die vollständig befüllt sind
            if len(row) >= len(header):
                x_chunker.append(row[0])  # Spalte 0: chunker (z.B. character_text_splitter)
                
                # Alle Werte ab Spalte 2 durchgehen und als float speichern
                for i, name in enumerate(metriken_namen):
                    raw_value = row[2 + i].strip()
                    try:
                        wert = float(raw_value) if raw_value else math.nan
                    except ValueError:
                        wert = math.nan
                    metriken_daten[name].append(wert)

    anzahl_metriken = len(metriken_namen)
    rows, cols = _grid_dimensions(anzahl_metriken)
    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(7 * cols, 4 * rows),
        squeeze=False,
    )
    
    axes_flat = axes.flatten()
    
    chunker_labels = [CHUNKER_LABELS.get(chunker, chunker) for chunker in x_chunker]
    y_positions = np.arange(len(x_chunker))

    for i, name in enumerate(metriken_namen):
        ax = axes_flat[i]
        werte = [value * 100 for value in metriken_daten[name]]
        plot_werte = [value if math.isfinite(value) else 0.0 for value in werte]
        bar_colors = [
            COLOR_PALETTE.get(chunker, "#7f7f7f") if math.isfinite(value) else "#d6d6d6"
            for chunker, value in zip(x_chunker, werte)
        ]
        
        bars = ax.barh(
            y_positions,
            plot_werte,
            color=bar_colors,
            height=0.62,
        )
        
        # Titel sauber formatieren (Unterstriche durch Leerzeichen ersetzen)
        schoener_titel = name.replace("_global", "").replace("_", " ")
        ax.set_title(schoener_titel, fontsize=11, fontweight='bold', pad=10)
        ax.set_yticks(y_positions, labels=chunker_labels, fontsize=8)
        ax.invert_yaxis()
        _style_percentage_axis(ax)
        _add_bar_labels(ax, bars, werte)
    
    # Falls es weniger Metriken als Raster-Plätze gibt, ungenutzte Plots unsichtbar machen
    for j in range(i + 1, len(axes_flat)):
        fig.delaxes(axes_flat[j])
        
    # Layout optimieren, damit Titel und X-Achsen sich nicht überlappen
    plt.tight_layout()
    plt.show()


    

def main() -> None:
    """Hauptfunktion zum Ausführen der Plots"""
    print("📊 Erstelle Plots für GRASCCO Evaluation...")
    #plot_csv("data/processed/chunker_global_conmparsion_5_")
    files = build_plots()
    print("\n✅ Generierte Plot-Dateien:")
    for file in files:
        print(f"  • {file}")

if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parents[2]
    #print(Path(__file__).resolve().parents[2])
    file = repo_root / "data" / "processed" / "chunker_global_comparison_10_docs.csv"
    print(f"datei existiert richtig: {file}")
    plot_csv(file)