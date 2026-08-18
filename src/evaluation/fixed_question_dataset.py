"""Create fixed-question groundtruth data from GRASCCO annotations."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


FIXED_QUESTIONS = {
    "patient_name": "Wie heißt der Patient?",
    "birth_date": "Wann hat der Patient Geburtsdatum?",
    "admission_date": "Wann wurde der Patient aufgenommen?",
    "discharge_date": "Wann wurde der Patient entlassen?",
}

LABEL_TO_QUESTION = {
    "PatientName": "patient_name",
    "PatientGeburtsdatum": "birth_date",
    "AufnahmeDatum": "admission_date",
    "EntlassDatum": "discharge_date",
}


@dataclass(frozen=True)
class Reference:
    content: str
    start_index: int
    end_index: int

    def to_dict(self) -> dict[str, str | int]:
        return {
            "content": self.content,
            "start_index": self.start_index,
            "end_index": self.end_index,
        }





def _load_json(path: Path):
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _extract_sofa_string(payload: dict) -> str:
    for feature in payload.get("%FEATURE_STRUCTURES", []):
        if feature.get("%TYPE") == "uima.cas.Sofa":
            return str(feature.get("sofaString", "")).lstrip("\ufeff")
    return ""


def _extract_annotations(payload: dict) -> list[dict[str, object]]:
    result = []
    for feature in payload.get("%FEATURE_STRUCTURES", []):
        if "PHI" in str(feature.get("%TYPE", "")):
            result.append({
                "kind": str(feature.get("kind", "")),
                "begin": feature.get("begin"),
                "end": feature.get("end"),
            })
    return result


def _extract_label_studio_annotations(entry: dict) -> list[dict[str, object]]:
    result = []
    for annotation in entry.get("annotations", []):
        for item in annotation.get("result", []):
            value = item.get("value", {})
            labels = value.get("labels", [])
            if labels:
                result.append({
                    "kind": str(labels[0]),
                    "begin": value.get("start"),
                    "end": value.get("end"),
                })
    return result


def _reference(text: str, annotation: dict[str, object]) -> Reference | None:
    try:
        begin = int(annotation["begin"])
        end = int(annotation["end"])
    except (KeyError, TypeError, ValueError):
        return None
    if not 0 <= begin < end <= len(text):
        return None
    return Reference(text[begin:end], begin, end)


def _date_question(text: str, reference: Reference) -> str | None:
    context = text[max(0, reference.start_index - 80):reference.end_index + 80].lower()
    if any(word in context for word in ("geburtsdatum", "geb.", "geburtstag", "geburts-")):
        return "birth_date"
    if any(word in context for word in ("aufnahme", "aufgenommen", "aufnahmedatum")):
        return "admission_date"
    if any(word in context for word in ("entlassung", "entlassungsdatum", "entlassen")):
        return "discharge_date"
    return None


def _question_rows_for_annotations(
    corpus_path: Path,
    text: str,
    annotations: Iterable[dict[str, object]],
) -> Iterable[dict[str, str]]:
    references: dict[str, list[Reference]] = {}
    for annotation in annotations:
        reference = _reference(text, annotation)
        if reference is None:
            continue

        kind = str(annotation.get("kind", ""))
        question_key = LABEL_TO_QUESTION.get(kind)
        if kind == "NAME_PATIENT":
            question_key = "patient_name"
        elif kind == "DATE":
            question_key = _date_question(text, reference)
        if question_key is not None:
            references.setdefault(question_key, []).append(reference)

    for question_key, question in FIXED_QUESTIONS.items():
        if references.get(question_key):
            yield {
                "question": question,
                "references": json.dumps(
                    [reference.to_dict() for reference in references[question_key]],
                    ensure_ascii=False,
                ),
                "corpus_id": str(corpus_path),
            }


def _question_rows_for_document_from_payload(
    corpus_path: Path,
    payload: dict,
) -> Iterable[dict[str, str]]:
    yield from _question_rows_for_annotations(
        corpus_path,
        _extract_sofa_string(payload),
        _extract_annotations(payload),
    )


def _question_rows_for_label_studio_entry(
    corpus_path: Path,
    entry: dict,
    text: str,
) -> Iterable[dict[str, str]]:
    yield from _question_rows_for_annotations(
        corpus_path,
        text,
        _extract_label_studio_annotations(entry),
    )


def _document_name(entry: dict) -> str:
    name = Path(str(entry.get("file_upload", ""))).name
    if "-" in name:
        name = name.split("-", 1)[1]
    if name:
        return name
    return Path(str(entry.get("data", {}).get("text", ""))).name


def _resolve_label_studio_text(entry: dict, text_dir: Path) -> Path | None:
    name = _document_name(entry)
    candidates = [text_dir / name]
    if name.endswith(".txt"):
        candidates.append(text_dir / name[:-4])
    return next((path for path in candidates if path.is_file()), None)


def _write_rows(rows: Iterable[dict[str, str]], csv_path: Path) -> int:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["question", "references", "corpus_id"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
            count += 1
    return count


def generate_fixed_questions_csv_from_json(
    json_files: Iterable[Path],
    csv_path: Path,
) -> int:
    rows = (
        row
        for path in json_files
        for row in _question_rows_for_document_from_payload(path, _load_json(path))
    )
    return _write_rows(rows, csv_path)


def generate_fixed_questions_csv(text_files: Iterable[Path], csv_path: Path) -> int:
    """Create the CSV header; plain text files contain no groundtruth spans."""
    return _write_rows([], csv_path)


def build_fixed_grascco_eval_dataset(
    annotations_path: Path,
    text_dir: Path,
    questions_csv_path: Path,
) -> tuple[int, int]:
    loaded = _load_json(annotations_path) if annotations_path.is_file() else []
    if not isinstance(loaded, list):
        return 0, 0

    rows = []
    documents = 0
    for entry in loaded:
        text_path = _resolve_label_studio_text(entry, text_dir)
        if text_path is None:
            continue
        text = text_path.read_text(encoding="utf-8")
        rows.extend(_question_rows_for_label_studio_entry(text_path, entry, text))
        documents += 1
    return documents, _write_rows(rows, questions_csv_path)


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[2]
    documents, rows = build_fixed_grascco_eval_dataset(
        root / "data" / "data-json" / "grascco-test.json",
        root / "data" / "row",
        root / "data" / "processed" / "grascco_fixed_questions.csv",
    )
    print(f"Documents with text: {documents}")
    print(f"Groundtruth rows: {rows}")
    print(f"Fixed questions CSV: {root / 'data' / 'processed' / 'grascco_fixed_questions.csv'}")
