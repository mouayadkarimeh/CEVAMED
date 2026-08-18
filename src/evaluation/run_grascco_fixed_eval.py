from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import chromadb.utils.embedding_functions as embedding_functions

from src.chunking.fixedTokenChunker import FixedTokenChunker
from src.chunking.hierarchical import HierarchicalChunker
from src.chunking.recursive import RecursiveCharacterTextSplitter
from src.chunking.semantic import SemanticChunker
from src.evaluation.basic_evaluation import BaseEvaluation
from src.evaluation.fixed_question_dataset import build_fixed_grascco_eval_dataset


@dataclass
class EvaluationProfile:
    name: str
    retrieve: int


def _chunkers() -> Dict[str, object]:
    return {
        "fixed_token": FixedTokenChunker(chunk_size=256, chunk_overlap=32),
        "recursive": RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=80),
        "hierarchical": HierarchicalChunker(max_sentence_group_size=450),
        "semantic": SemanticChunker(avg_chunk_size=420, min_chunk_size=120, model_name="sentence-transformers/all-MiniLM-L6-v2"),
    }


def _profiles() -> list[EvaluationProfile]:
    return [
        EvaluationProfile(name="basic", retrieve=-1),
        EvaluationProfile(name="general", retrieve=5),
        EvaluationProfile(name="synthetic", retrieve=10),
    ]


def run() -> dict:
    repo_root = Path(__file__).resolve().parents[2]

    json_input_dir = repo_root / "data" / "data-json" / "grascco-json" / "grascco_phi_annotation_json"
    text_output_dir = repo_root / "data" / "processed" / "grascco_texts"
    questions_csv_output = repo_root / "data" / "processed" / "grascco_fixed_questions.csv"
    results_output = repo_root / "data" / "processed" / "grascco_eval_results.json"

    converted_count, rows_count = build_fixed_grascco_eval_dataset(
        json_dir=json_input_dir,
        text_dir=text_output_dir,
        questions_csv_path=questions_csv_output,
    )

    evaluator = BaseEvaluation(questions_csv_path=str(questions_csv_output))
    embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    results: dict[str, dict[str, dict[str, float]]] = {
        "dataset": {
            "converted_documents": converted_count,
            "question_rows": rows_count,
        },
        "profiles": {},
    }

    for profile in _profiles():
        profile_results: dict[str, dict[str, float]] = {}
        for chunker_name, chunker in _chunkers().items():
            metrics = evaluator.run(
                chunker=chunker,
                embedding_function=embedding_function,
                retrieve=profile.retrieve,
            )
            profile_results[chunker_name] = {
                "iou_mean": float(metrics["iou_mean"]),
                "recall_mean": float(metrics["recall_mean"]),
                "precision_omega_mean": float(metrics["precision_omega_mean"]),
                "precision_mean": float(metrics["precision_mean"]),
            }
        results["profiles"][profile.name] = profile_results

    results_output.parent.mkdir(parents=True, exist_ok=True)
    with results_output.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2, ensure_ascii=False)

    return results


def _print_summary(results: dict) -> None:
    print("GRASCCO converted documents:", results["dataset"]["converted_documents"])
    print("GRASCCO fixed question rows:", results["dataset"]["question_rows"])

    for profile_name, chunker_results in results["profiles"].items():
        print(f"\n=== {profile_name.upper()} ===")
        for chunker_name, metrics in chunker_results.items():
            print(
                f"{chunker_name:12s} | iou={metrics['iou_mean']:.4f} "
                f"recall={metrics['recall_mean']:.4f} "
                f"precision={metrics['precision_mean']:.4f} "
                f"precision_omega={metrics['precision_omega_mean']:.4f}"
            )


if __name__ == "__main__":
    _print_summary(run())
