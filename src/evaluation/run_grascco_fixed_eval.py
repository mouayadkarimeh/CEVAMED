from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Dict

import chromadb.utils.embedding_functions as embedding_functions

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from chunking.langchainSplitter import FixedTokenChunker
from chunking.recusrive_semantic_chunker import HierarchicalChunker
from chunking.recursive_token_chunker import RecursiveCharacterTextSplitter
from chunking.kamradt_semantic_chunker import SemanticChunker
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
        "semantic": SemanticChunker(avg_chunk_size=400, min_chunk_size=50),
    }

def _profiles() -> list[EvaluationProfile]:
    return [
        EvaluationProfile(name="minimal", retrieve=-1),
        EvaluationProfile(name="top_3", retrieve=3),
        EvaluationProfile(name="top_5", retrieve=5),
    ]


def run() -> dict:
    repo_root = REPO_ROOT

    json_input_path = repo_root / "data" / "data-json" / "grascco-test.json"
    text_output_dir = repo_root / "data" / "row"
    questions_csv_output = repo_root / "data" / "processed" / "grascco_fixed_questions.csv"
    results_output = repo_root / "data" / "processed" / "grascco_eval_results.json"

    converted_count, rows_count = build_fixed_grascco_eval_dataset(
        annotations_path=json_input_path,
        text_dir=text_output_dir,
        questions_csv_path=questions_csv_output,
    )

    evaluator = BaseEvaluation(questions_csv_path=str(questions_csv_output))
    embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    results: dict[str, dict[str, dict[str, float]]] = {
        "dataset": {
            "documents_with_text": converted_count,
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
                "f1_mean": float(metrics["f1_mean"]),
                "hit_at_k_mean": float(metrics["hit_at_k_mean"]),
                "mrr_mean": float(metrics["mrr_mean"]),
                "ndcg_at_k_mean": float(metrics["ndcg_at_k_mean"]),
                "fragmentation_mean": float(metrics["fragmentation_mean"]),
                "question_metrics": metrics["question_metrics"],
            }
        results["profiles"][profile.name] = profile_results

    results_output.parent.mkdir(parents=True, exist_ok=True)
    with results_output.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2, ensure_ascii=False)

    return results


def _print_summary(results: dict) -> None:
    print("GRASCCO documents with text:", results["dataset"]["documents_with_text"])
    print("GRASCCO fixed question rows:", results["dataset"]["question_rows"])

    for profile_name, chunker_results in results["profiles"].items():
        print(f"\n=== {profile_name.upper()} ===")
        for chunker_name, metrics in chunker_results.items():
            print(
                f"{chunker_name:12s} | iou={metrics['iou_mean']:.4f} "
                f"recall={metrics['recall_mean']:.4f} "
                f"precision={metrics['precision_mean']:.4f} "
                f"f1={metrics['f1_mean']:.4f} "
                f"hit@k={metrics['hit_at_k_mean']:.4f} "
                f"mrr={metrics['mrr_mean']:.4f} "
                f"ndcg@k={metrics['ndcg_at_k_mean']:.4f} "
                f"frag={metrics['fragmentation_mean']:.4f} "
                f"precision_omega={metrics['precision_omega_mean']:.4f}"
            )


if __name__ == "__main__":
    _print_summary(run())
