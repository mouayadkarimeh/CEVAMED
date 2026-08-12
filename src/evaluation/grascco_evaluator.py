"""GraSCCo Evaluation Framework – adapted from brandonstarxel/chunking_evaluation.

Changes vs. the original:
* **No OpenAI / no external embeddings** – retrieval uses a pure-Python BM25-
  inspired keyword scorer (no third-party dependencies beyond the stdlib).
* **Four fixed questions** derived from the PHI annotations already present in
  the GraSCCo dataset:
    1. "Wie heißt der Patient?"           → NAME_PATIENT span
    2. "Wann hat der Patient Geburtstag?" → DATE span closest to "geboren"
    3. "Wann wurde der Patient aufgenommen?" → earliest DATE in document
    4. "Wann wurde der Patient entlassen?" → latest DATE in document
* Metrics (iou_mean / recall_mean) follow the same definition as the original.

Public API::

    from src.evaluation.grascco_evaluator import GraSCCoEvaluation

    evaluator = GraSCCoEvaluation()
    results = evaluator.run(chunker)
    print(results)
"""

from __future__ import annotations

import json
import math
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Fixed questions (corrected spelling)
# ---------------------------------------------------------------------------

FIXED_QUESTIONS: List[str] = [
    "Wie heißt der Patient?",
    "Wann hat der Patient Geburtstag?",
    "Wann wurde der Patient aufgenommen?",
    "Wann wurde der Patient entlassen?",
]

# PHI annotation kinds mapped to each question
_QUESTION_KIND_MAP: Dict[str, str] = {
    "Wie heißt der Patient?": "NAME_PATIENT",
    "Wann hat der Patient Geburtstag?": "DATE_BIRTH",   # virtual – see _find_span
    "Wann wurde der Patient aufgenommen?": "DATE_ADMIT",  # virtual – see _find_span
    "Wann wurde der Patient entlassen?": "DATE_DISCHARGE",  # virtual
}

# Keywords that signal admission / discharge context
_ADMIT_KEYWORDS = re.compile(
    r"\b(aufnahme|aufgenommen|stationäre\s+aufnahme|einweisung|eingewiesen|stat\.\s*aufnahme)\b",
    re.IGNORECASE,
)
_DISCHARGE_KEYWORDS = re.compile(
    r"\b(entlassen|entlassung|nach\s+hause|entlassungsdatum|abschlussbericht|epikrise)\b",
    re.IGNORECASE,
)
_BIRTH_KEYWORDS = re.compile(
    r"\b(geboren\s*am|geburtsdatum|geb\.|geb\s+am|dob|geburtstag)\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class QARecord:
    """A single question/answer record linked to a document."""

    doc_id: str
    question: str
    excerpt: str       # ground-truth text span
    begin: int         # character offset in original text
    end: int           # character offset in original text


@dataclass
class GraSCCoEvaluation:
    """End-to-end evaluator for GraSCCo clinical documents.

    Parameters
    ----------
    texts_dir:
        Directory that contains ``<doc_id>.txt`` and ``<doc_id>.meta.json``
        files (output of ``scripts/convert_grascco_json.py``).
    questions:
        Override the default four fixed questions (optional).
    top_k:
        Number of chunks to retrieve per query when computing metrics.
    """

    texts_dir: str | Path = field(
        default_factory=lambda: Path(__file__).resolve().parents[2]
        / "data"
        / "grascco_texts"
    )
    questions: List[str] = field(default_factory=lambda: list(FIXED_QUESTIONS))
    top_k: int = 5

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def run(self, chunker) -> Dict[str, float]:
        """Evaluate *chunker* and return aggregated metrics.

        Parameters
        ----------
        chunker:
            Any object with a ``split_text(text: str) -> List[str]`` method
            (compatible with :class:`src.chunking.base.BaseChunker`).

        Returns
        -------
        dict with keys ``iou_mean``, ``iou_std``, ``recall_mean``, ``recall_std``.
        """
        texts_dir = Path(self.texts_dir)
        if not texts_dir.exists():
            raise FileNotFoundError(
                f"texts_dir '{texts_dir}' does not exist. "
                "Run 'python scripts/convert_grascco_json.py' first."
            )

        records = self._load_qa_records(texts_dir)
        if not records:
            raise RuntimeError(
                f"No QA records found in '{texts_dir}'. "
                "Make sure the directory contains .txt and .meta.json files."
            )

        iou_scores: List[float] = []
        recall_scores: List[float] = []

        for record in records:
            txt_path = texts_dir / f"{record.doc_id}.txt"
            if not txt_path.exists():
                continue
            full_text = txt_path.read_text(encoding="utf-8")

            chunks: List[str] = chunker.split_text(full_text)
            if not chunks:
                iou_scores.append(0.0)
                recall_scores.append(0.0)
                continue

            # Compute character offsets for every chunk (non-overlapping scan)
            chunk_spans = _build_chunk_spans(full_text, chunks)

            # Retrieve top-k chunks by BM25-like score
            retrieved_indices = _bm25_retrieve(record.question, chunks, top_k=self.top_k)

            gold_set = set(range(record.begin, record.end))
            best_iou = 0.0
            best_recall = 0.0

            for idx in retrieved_indices:
                c_begin, c_end = chunk_spans[idx]
                pred_set = set(range(c_begin, c_end))
                intersection = len(gold_set & pred_set)
                union = len(gold_set | pred_set)
                iou = intersection / union if union else 0.0
                recall = intersection / len(gold_set) if gold_set else 0.0
                best_iou = max(best_iou, iou)
                best_recall = max(best_recall, recall)

            iou_scores.append(best_iou)
            recall_scores.append(best_recall)

        return _aggregate(iou_scores, recall_scores)

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _load_qa_records(self, texts_dir: Path) -> List[QARecord]:
        records: List[QARecord] = []
        for meta_path in sorted(texts_dir.glob("*.meta.json")):
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            doc_id: str = meta["doc_id"]
            txt_path = texts_dir / f"{doc_id}.txt"
            if not txt_path.exists():
                continue
            full_text = txt_path.read_text(encoding="utf-8")
            phi_list: List[dict] = meta.get("phi_annotations", [])

            for question in self.questions:
                span = _find_span(question, phi_list, full_text)
                if span is not None:
                    begin, end = span
                    records.append(
                        QARecord(
                            doc_id=doc_id,
                            question=question,
                            excerpt=full_text[begin:end],
                            begin=begin,
                            end=end,
                        )
                    )
        return records


# ---------------------------------------------------------------------------
# Span extraction from PHI annotations
# ---------------------------------------------------------------------------

def _find_span(
    question: str, phi_list: List[dict], full_text: str
) -> Optional[Tuple[int, int]]:
    """Return (begin, end) character offsets for the answer to *question*."""
    kind = _QUESTION_KIND_MAP.get(question)

    if kind == "NAME_PATIENT":
        for phi in phi_list:
            if phi.get("kind") == "NAME_PATIENT":
                return phi["begin"], phi["end"]
        return None

    # For date-based questions we look at all DATE annotations
    date_spans = [
        (phi["begin"], phi["end"])
        for phi in phi_list
        if phi.get("kind") == "DATE"
    ]
    if not date_spans:
        return None

    if kind == "DATE_BIRTH":
        # Find the DATE annotation whose context contains birth keywords
        span = _date_near_keyword(date_spans, full_text, _BIRTH_KEYWORDS)
        return span if span else date_spans[0]

    if kind == "DATE_ADMIT":
        # Earliest DATE in document (admissions usually mentioned first)
        span = _date_near_keyword(date_spans, full_text, _ADMIT_KEYWORDS)
        return span if span else min(date_spans, key=lambda s: s[0])

    if kind == "DATE_DISCHARGE":
        # Latest DATE in document or one near discharge keywords
        span = _date_near_keyword(date_spans, full_text, _DISCHARGE_KEYWORDS)
        return span if span else max(date_spans, key=lambda s: s[0])

    return None


def _date_near_keyword(
    date_spans: List[Tuple[int, int]],
    full_text: str,
    keyword_re: re.Pattern,
    window: int = 150,
) -> Optional[Tuple[int, int]]:
    """Return the DATE span closest to a keyword match within *window* chars."""
    keyword_positions = [m.start() for m in keyword_re.finditer(full_text)]
    if not keyword_positions:
        return None
    best_span: Optional[Tuple[int, int]] = None
    best_dist = window + 1
    for kw_pos in keyword_positions:
        for begin, end in date_spans:
            dist = min(abs(begin - kw_pos), abs(end - kw_pos))
            if dist < best_dist:
                best_dist = dist
                best_span = (begin, end)
    return best_span


# ---------------------------------------------------------------------------
# Chunk-span reconstruction
# ---------------------------------------------------------------------------

def _build_chunk_spans(
    full_text: str, chunks: List[str]
) -> List[Tuple[int, int]]:
    """Map each chunk back to its (begin, end) character offsets in *full_text*."""
    spans: List[Tuple[int, int]] = []
    cursor = 0
    for chunk in chunks:
        idx = full_text.find(chunk, cursor)
        if idx == -1:
            # Fallback: search from start (handles overlapping chunkers)
            idx = full_text.find(chunk)
        if idx == -1:
            spans.append((cursor, cursor))
        else:
            spans.append((idx, idx + len(chunk)))
            cursor = idx + len(chunk)
    return spans


# ---------------------------------------------------------------------------
# BM25-inspired keyword retrieval (no external dependencies)
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> List[str]:
    return re.findall(r"\w+", text.lower())


def _bm25_retrieve(
    query: str, chunks: List[str], top_k: int = 5, k1: float = 1.5, b: float = 0.75
) -> List[int]:
    """Return indices of the *top_k* chunks most relevant to *query*."""
    query_tokens = _tokenize(query)
    if not query_tokens:
        return list(range(min(top_k, len(chunks))))

    # Build per-chunk term-frequency maps
    tf_maps: List[Dict[str, int]] = []
    for chunk in chunks:
        counts: Dict[str, int] = {}
        for tok in _tokenize(chunk):
            counts[tok] = counts.get(tok, 0) + 1
        tf_maps.append(counts)

    n = len(chunks)
    avg_len = sum(len(_tokenize(c)) for c in chunks) / max(n, 1)

    # IDF for each query token
    idf: Dict[str, float] = {}
    for tok in query_tokens:
        df = sum(1 for tf in tf_maps if tok in tf)
        idf[tok] = math.log((n - df + 0.5) / (df + 0.5) + 1.0)

    # Score each chunk
    scores: List[float] = []
    for i, tf in enumerate(tf_maps):
        doc_len = sum(tf.values())
        score = 0.0
        for tok in query_tokens:
            freq = tf.get(tok, 0)
            norm = freq * (k1 + 1) / (freq + k1 * (1 - b + b * doc_len / avg_len))
            score += idf.get(tok, 0.0) * norm
        scores.append(score)

    ranked = sorted(range(n), key=lambda j: scores[j], reverse=True)
    return ranked[:top_k]


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def _mean_std(values: List[float]) -> Tuple[float, float]:
    if not values:
        return 0.0, 0.0
    n = len(values)
    mean = sum(values) / n
    variance = sum((v - mean) ** 2 for v in values) / n
    return mean, math.sqrt(variance)


def _aggregate(
    iou_scores: List[float], recall_scores: List[float]
) -> Dict[str, float]:
    iou_mean, iou_std = _mean_std(iou_scores)
    recall_mean, recall_std = _mean_std(recall_scores)
    return {
        "iou_mean": round(iou_mean, 6),
        "iou_std": round(iou_std, 6),
        "recall_mean": round(recall_mean, 6),
        "recall_std": round(recall_std, 6),
        "num_queries": len(iou_scores),
    }
