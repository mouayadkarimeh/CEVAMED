"""Run the fixed-question evaluation for a single document and write a
detailed, per-question report showing which chunks were returned for each
question, alongside the groundtruth answer and the resulting metrics.

No plots are generated here; this is only for inspecting one document at a
time (e.g. a single Entlassungsbrief). Plots for the full 60-document
evaluation are generated separately by ``src.evaluation.plot_grascco_results``.

Usage:
    python -m src.evaluation.report_single_document_question_chunks --document Albers.txt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import chromadb.utils.embedding_functions as embedding_functions
import pandas as pd
import tiktoken

from src.chunking.fixedTokenChunker import FixedTokenChunker
from src.chunking.hierarchical import HierarchicalChunker
from src.chunking.recursive import RecursiveCharacterTextSplitter
from src.chunking.semantic import SemanticChunker
from src.evaluation.basic_evaluation import BaseEvaluation
from src.evaluation.fixed_question_dataset import build_fixed_grascco_eval_dataset

REPO_ROOT = Path(__file__).resolve().parents[2]
RETRIEVE_PROFILES = [("minimal", -1), ("top_3", 3), ("top_5", 5)]
_TOKENIZER = tiktoken.get_encoding("cl100k_base")


def _count_tokens(text: str) -> int:
    """Token count used only for reporting, independent of the chosen chunker."""
    return len(_TOKENIZER.encode(text, disallowed_special=()))


ALL_CHUNKER_NAMES = ["fixed_token", "recursive", "hierarchical", "semantic"]


def _build_chunker(chunker_name: str):
    # NOTE: only fixed_token enforces a hard token ceiling. semantic's
    # avg_chunk_size is an approximate target (see split_text), so individual
    # chunks can be much larger than avg_chunk_size.
    chunkers = {
        "fixed_token": lambda: FixedTokenChunker(chunk_size=256, chunk_overlap=32),
        "recursive": lambda: RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=80),
        "hierarchical": lambda: HierarchicalChunker(max_sentence_group_size=450),
        "semantic": lambda: SemanticChunker(avg_chunk_size=400, min_chunk_size=50),
    }
    if chunker_name not in chunkers:
        raise ValueError(f"Unknown chunker '{chunker_name}'. Choose from: {list(chunkers)}")
    return chunkers[chunker_name]()


def _build_single_document_questions_csv(
    full_questions_csv: Path, document_name: str, output_csv: Path
) -> Path:
    """Filter the full groundtruth CSV down to only the target document's rows."""
    questions_df = pd.read_csv(full_questions_csv)
    document_rows = questions_df[questions_df["corpus_id"].str.endswith(document_name)]
    if document_rows.empty:
        raise ValueError(f"No groundtruth questions found for document: {document_name}")
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    document_rows.to_csv(output_csv, index=False)
    return output_csv


def _write_json_report(chunker_results: dict[str, dict[str, list[dict]]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(chunker_results, file, indent=2, ensure_ascii=False)


def _write_text_report(
    chunker_results: dict[str, dict[str, list[dict]]],
    document_name: str,
    output_path: Path,
) -> None:
    lines: list[str] = [
        f"Question-to-Chunk Report for {document_name}",
        f"Chunking strategies tested: {', '.join(chunker_results.keys())}",
        "=" * 60,
        "",
    ]

    for chunker_name, profile_results in chunker_results.items():
        lines.append("#" * 70)
        lines.append(f"# CHUNKER: {chunker_name}")
        lines.append("#" * 70)
        lines.append("")

        for profile_name, question_metrics in profile_results.items():
            lines.append(f"### PROFILE: {profile_name} (chunker: {chunker_name}) ###")
            lines.append("")

            for entry in question_metrics:
                lines.append(f"Question ({entry['question_key']}): {entry['question']}")
                lines.append("Groundtruth:")
                for reference in entry["groundtruth"]:
                    lines.append(
                        f"  - '{reference['content']}' "
                        f"[{reference['start_index']}-{reference['end_index']}]"
                    )

                lines.append(f"Retrieved chunks ({len(entry['retrieved_chunks'])}):")
                for chunk in entry["retrieved_chunks"]:
                    marker = "RELEVANT" if chunk["relevant"] else "not relevant"
                    char_length = chunk["end_index"] - chunk["start_index"]
                    token_count = _count_tokens(chunk["text"])
                    lines.append("  " + "-" * 70)
                    lines.append(
                        f"  CHUNK START (rank {chunk['rank']}, {marker}) "
                        f"| position [{chunk['start_index']}-{chunk['end_index']}] "
                        f"| chars={char_length} | tokens={token_count}"
                    )
                    lines.append(f"    {chunk['text']}")
                    lines.append(f"  CHUNK END (rank {chunk['rank']})")
                    lines.append("  " + "-" * 70)

                metrics = entry["metrics"]
                lines.append(
                    "Metrics: "
                    f"recall={metrics['recall']:.3f} precision={metrics['precision']:.3f} "
                    f"iou={metrics['iou']:.3f} hit_at_k={metrics['hit_at_k']:.3f} mrr={metrics['mrr']:.3f}"
                )
                lines.append("")

            lines.append("")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def run(document_name: str, chunker_names: list[str]) -> None:
    annotations_path = REPO_ROOT / "data" / "data-json" / "grascco-test.json"
    text_dir = REPO_ROOT / "data" / "row"
    full_questions_csv = REPO_ROOT / "data" / "processed" / "grascco_fixed_questions.csv"
    build_fixed_grascco_eval_dataset(annotations_path, text_dir, full_questions_csv)

    document_stem = Path(document_name).stem
    single_document_questions_csv = (
        REPO_ROOT / "data" / "processed" / f"{document_stem}_only_questions.csv"
    )
    _build_single_document_questions_csv(
        full_questions_csv, document_name, single_document_questions_csv
    )

    evaluator = BaseEvaluation(questions_csv_path=str(single_document_questions_csv))
    embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    chunker_results: dict[str, dict[str, list[dict]]] = {}
    for chunker_name in chunker_names:
        chunker = _build_chunker(chunker_name)
        profile_results: dict[str, list[dict]] = {}
        for profile_name, retrieve in RETRIEVE_PROFILES:
            metrics = evaluator.run(
                chunker=chunker,
                embedding_function=embedding_function,
                retrieve=retrieve,
            )
            profile_results[profile_name] = metrics["question_metrics"]
        chunker_results[chunker_name] = profile_results

    json_output = REPO_ROOT / "data" / "processed" / f"{document_stem}_question_chunk_report.json"
    text_output = REPO_ROOT / "data" / "processed" / f"{document_stem}_question_chunk_report.txt"
    _write_json_report(chunker_results, json_output)
    _write_text_report(chunker_results, document_name, text_output)

    for chunker_name, profile_results in chunker_results.items():
        for profile_name, question_metrics in profile_results.items():
            print(f"{chunker_name}/{profile_name}: {len(question_metrics)} questions analyzed")
    print(f"JSON report: {json_output}")
    print(f"Text report: {text_output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--document",
        default="Albers.txt",
        help="File name under data/row/ to analyze (e.g. Albers.txt)",
    )
    parser.add_argument(
        "--chunker",
        default="all",
        choices=["all"] + ALL_CHUNKER_NAMES,
        help="Chunking strategy to analyze. Use 'all' (default) to test every strategy.",
    )
    args = parser.parse_args()
    selected_chunkers = ALL_CHUNKER_NAMES if args.chunker == "all" else [args.chunker]
    run(args.document, selected_chunkers)
