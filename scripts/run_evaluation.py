"""End-to-end evaluation pipeline for GraSCCo clinical documents.

Steps performed automatically:
  1. Convert GRASCCO JSON → plain text (``data/grascco_texts/``)
  2. Chunk every document with the chosen chunker
  3. Run the evaluation framework (BM25 retrieval, IoU + Recall metrics)
  4. Print / save results

Usage::

    # Basic run (fixed-size chunker, 500 chars):
    python scripts/run_evaluation.py

    # Choose chunking strategy:
    python scripts/run_evaluation.py --chunker recursive --chunk-size 400

    # Save results to a JSON file:
    python scripts/run_evaluation.py --output results.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Make the project root importable when running as a script
_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from scripts.convert_grascco_json import convert
from src.evaluation.grascco_evaluator import GraSCCoEvaluation, FIXED_QUESTIONS


# ---------------------------------------------------------------------------
# Minimal built-in chunkers (no external dependencies)
# ---------------------------------------------------------------------------

class FixedSizeChunker:
    """Split text into non-overlapping fixed-size character windows."""

    def __init__(self, chunk_size: int = 500, overlap: int = 0) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap

    def split_text(self, text: str):
        step = max(1, self.chunk_size - self.overlap)
        return [text[i : i + self.chunk_size] for i in range(0, len(text), step) if text[i : i + self.chunk_size].strip()]


class RecursiveChunker:
    """Split text recursively on paragraph / sentence boundaries."""

    def __init__(self, chunk_size: int = 500, overlap: int = 50) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap

    def split_text(self, text: str):
        separators = ["\n\n", "\n", ". ", " "]
        return self._split(text.strip(), separators)

    def _split(self, text: str, separators: list):
        if len(text) <= self.chunk_size:
            return [text] if text.strip() else []
        sep = separators[0] if separators else " "
        parts = text.split(sep)
        chunks = []
        current = ""
        for part in parts:
            candidate = current + (sep if current else "") + part
            if len(candidate) <= self.chunk_size:
                current = candidate
            else:
                if current.strip():
                    chunks.append(current)
                current = part
        if current.strip():
            chunks.append(current)
        return chunks


_CHUNKERS = {
    "fixed": FixedSizeChunker,
    "recursive": RecursiveChunker,
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--chunker",
        choices=list(_CHUNKERS.keys()),
        default="fixed",
        help="Chunking strategy to evaluate (default: fixed)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=500,
        help="Target chunk size in characters (default: 500)",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=50,
        help="Character overlap between consecutive chunks (default: 50)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of chunks to retrieve per query (default: 5)",
    )
    parser.add_argument(
        "--input",
        default="data/data-json/grascco-json/grascco_phi_annotation_json",
        help="GRASCCO JSON source directory",
    )
    parser.add_argument(
        "--texts-dir",
        default="data/grascco_texts",
        help="Directory for intermediate plain-text files",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional path to save results as JSON",
    )
    args = parser.parse_args()

    input_dir = Path(args.input) if Path(args.input).is_absolute() else _ROOT / args.input
    texts_dir = Path(args.texts_dir) if Path(args.texts_dir).is_absolute() else _ROOT / args.texts_dir

    # ------------------------------------------------------------------ #
    # Step 1: Convert JSON → text                                         #
    # ------------------------------------------------------------------ #
    print("=" * 60)
    print("Step 1 – Converting GRASCCO JSON → plain text")
    print("=" * 60)
    n = convert(input_dir, texts_dir)
    print(f"→ {n} document(s) converted.\n")

    # ------------------------------------------------------------------ #
    # Step 2: Build chunker                                               #
    # ------------------------------------------------------------------ #
    chunker_cls = _CHUNKERS[args.chunker]
    chunker = chunker_cls(chunk_size=args.chunk_size, overlap=args.overlap)
    print(f"Step 2 – Using chunker: {args.chunker!r} "
          f"(chunk_size={args.chunk_size}, overlap={args.overlap})\n")

    # ------------------------------------------------------------------ #
    # Step 3: Run evaluation                                              #
    # ------------------------------------------------------------------ #
    print("=" * 60)
    print("Step 3 – Running evaluation")
    print(f"Questions evaluated ({len(FIXED_QUESTIONS)}):")
    for q in FIXED_QUESTIONS:
        print(f"  • {q}")
    print("=" * 60)

    evaluator = GraSCCoEvaluation(texts_dir=texts_dir, top_k=args.top_k)
    results = evaluator.run(chunker)

    # ------------------------------------------------------------------ #
    # Step 4: Output                                                      #
    # ------------------------------------------------------------------ #
    print("\nResults:")
    print(f"  IoU mean:     {results['iou_mean']:.4f}  (±{results['iou_std']:.4f})")
    print(f"  Recall mean:  {results['recall_mean']:.4f}  (±{results['recall_std']:.4f})")
    print(f"  Queries:      {results['num_queries']}")

    if args.output:
        out_path = Path(args.output)
        out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
