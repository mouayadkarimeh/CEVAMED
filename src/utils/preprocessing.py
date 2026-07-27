from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
import re
import unicodedata

import numpy as np


@dataclass
class ClinicalDocument:
    """Container for GraSCCo-style clinical document text."""

    doc_id: str
    text: str
    metadata: Dict[str, Any]


def load_grascco_texts(data_dir: str | Path) -> List[ClinicalDocument]:
    """Load clinical texts from a directory of .txt/.json files."""
    directory = Path(data_dir)
    if not directory.exists():
        return []

    docs: List[ClinicalDocument] = []
    for path in sorted(directory.rglob("*")):
        if path.suffix.lower() == ".txt":
            docs.append(ClinicalDocument(doc_id=path.stem, text=path.read_text(encoding="utf-8"), metadata={"path": str(path)}))
        elif path.suffix.lower() == ".json":
            import json

            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, dict) and "text" in payload:
                docs.append(
                    ClinicalDocument(
                        doc_id=str(payload.get("id", path.stem)),
                        text=str(payload["text"]),
                        metadata={k: v for k, v in payload.items() if k != "text"},
                    )
                )
    return docs


def normalize_german_clinical_text(text: str) -> str:
    """Normalize whitespace, unicode form, and common chart punctuation."""
    normalized = unicodedata.normalize("NFKC", text)
    normalized = normalized.replace("\u00a0", " ")
    normalized = re.sub(r"\s+", " ", normalized)
    normalized = re.sub(r"\s*([,;:.!?])\s*", r"\1 ", normalized)
    return normalized.strip()


def split_sentences(text: str) -> List[str]:
    """Sentence splitter with regex fallback for German clinical text."""
    parts = re.split(r"(?<=[.!?])\s+(?=[A-ZÄÖÜ])", text.strip())
    return [p.strip() for p in parts if p.strip()]


def extract_medical_entities(text: str, model_name: Optional[str] = None) -> Dict[str, List[str]]:
    """Extract simple medical entities, with optional transformers NER when available."""
    entities = {
        "conditions": [],
        "medications": [],
        "procedures": [],
        "measurements": [],
    }

    # Lightweight clinical-pattern fallback
    med_pattern = re.compile(r"\b([A-Z][a-z]{2,}\s?(?:mg|ml|g|IE)?)\b")
    measurement_pattern = re.compile(r"\b\d+(?:[.,]\d+)?\s?(?:mg/dl|mmHg|bpm|%|°C)\b", re.IGNORECASE)
    for m in med_pattern.findall(text):
        if any(unit in m.lower() for unit in ("mg", "ml", "ie", "g")):
            entities["medications"].append(m)
    entities["measurements"] = measurement_pattern.findall(text)

    condition_keywords = ["diabetes", "hypertonie", "infarkt", "pneumonie", "schmerz"]
    procedure_keywords = ["ct", "mrt", "operation", "biopsie", "ekg"]
    lowered = text.lower()
    entities["conditions"] = [kw for kw in condition_keywords if kw in lowered]
    entities["procedures"] = [kw for kw in procedure_keywords if kw in lowered]

    if model_name:
        try:
            from transformers import pipeline

            ner = pipeline("ner", model=model_name, aggregation_strategy="simple")
            for ent in ner(text):
                group = str(ent.get("entity_group", "")).lower()
                token = str(ent.get("word", "")).strip()
                if not token:
                    continue
                if "med" in group:
                    entities["medications"].append(token)
                elif "proc" in group:
                    entities["procedures"].append(token)
                elif "dis" in group or "cond" in group:
                    entities["conditions"].append(token)
        except Exception:
            pass

    return {k: sorted(set(v)) for k, v in entities.items()}


def _hash_embedding(text: str, dimension: int = 64) -> np.ndarray:
    vec = np.zeros(dimension, dtype=float)
    for token in re.findall(r"\w+", text.lower()):
        vec[hash(token) % dimension] += 1.0
    norm = np.linalg.norm(vec)
    return vec / norm if norm else vec


def generate_embeddings(texts: Iterable[str], model_name: Optional[str] = None) -> np.ndarray:
    """Generate embeddings with sentence-transformers when available, else deterministic hash vectors."""
    texts = list(texts)
    if model_name:
        try:
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer(model_name)
            return np.asarray(model.encode(texts))
        except Exception:
            pass
    return np.asarray([_hash_embedding(t) for t in texts])
