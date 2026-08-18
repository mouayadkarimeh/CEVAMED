from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
import json
import re
from typing import Iterable


FIXED_QUESTIONS = {
    "patient_name": "Wie heißt der Patient?",
    "birth_date": "Wann hat der Patient Geburtsdatum?",
    "admission_date": "Wann wurde der Patient aufgenommen?",
    "discharge_date": "Wann wurde der Patient entlassen?",
}

DATE_RE = r"\d{1,2}\.\d{1,2}\.\d{2,4}"


@dataclass
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


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _extract_sofa_string(payload: dict) -> str:
    feature_structures = payload.get("%FEATURE_STRUCTURES", [])
    for item in feature_structures:
        if item.get("%TYPE") == "uima.cas.Sofa":
            return str(item.get("sofaString", "")).lstrip("\ufeff")
    return ""


def convert_grascco_json_to_text_files(json_dir: Path, text_dir: Path) -> list[Path]:
    text_dir.mkdir(parents=True, exist_ok=True)
    output_files: list[Path] = []

    for json_path in sorted(json_dir.glob("*_phi.json")):
        payload = _load_json(json_path)
        text = _extract_sofa_string(payload)
        if not text.strip():
            continue

        text_filename = json_path.stem.replace("_phi", "")
        text_path = text_dir / text_filename
        text_path.write_text(text, encoding="utf-8")
        output_files.append(text_path)

    return output_files


def _capture_group_reference(match: re.Match[str], group_index: int = 1) -> Reference:
    start_index, end_index = match.span(group_index)
    content = match.group(group_index).strip()
    return Reference(content=content, start_index=start_index, end_index=end_index)


def _extract_patient_name(text: str) -> Reference | None:
    patterns = [
        r"Patient\s*:\s*([^\n,]+)",
        r"Name\s*:\s*([^\n,]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return _capture_group_reference(match, 1)
    return None


def _extract_birth_date(text: str) -> Reference | None:
    patterns = [
        rf"(?:geb\.?|Geburtsdatum)\s*[:.]?\s*({DATE_RE})",
        rf"Geb\.?\s*:\s*({DATE_RE})",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return _capture_group_reference(match, 1)
    return None


def _extract_admission_and_discharge(text: str) -> tuple[Reference | None, Reference | None]:
    range_patterns = [
        rf"Station[aä]rer\s+Aufenthalt\s*:\s*({DATE_RE})\s*(?:bis|\-|–)\s*({DATE_RE})",
        rf"Aufenthalt\s*:\s*({DATE_RE})\s*(?:bis|\-|–)\s*({DATE_RE})",
    ]
    for pattern in range_patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return _capture_group_reference(match, 1), _capture_group_reference(match, 2)

    admission_patterns = [
        rf"Aufnahme(?:datum)?\s*:\s*({DATE_RE})",
        rf"aufgenommen\s+am\s+({DATE_RE})",
    ]
    discharge_patterns = [
        rf"Entlass(?:ung|ungsdatum)?\s*:\s*({DATE_RE})",
        rf"entlassen\s+am\s+({DATE_RE})",
    ]

    admission_ref = None
    discharge_ref = None

    for pattern in admission_patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            admission_ref = _capture_group_reference(match, 1)
            break

    for pattern in discharge_patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            discharge_ref = _capture_group_reference(match, 1)
            break

    return admission_ref, discharge_ref


def _question_rows_for_document(corpus_path: Path, text: str) -> Iterable[dict[str, str]]:
    patient_name = _extract_patient_name(text)
    birth_date = _extract_birth_date(text)
    admission_date, discharge_date = _extract_admission_and_discharge(text)

    mapping: list[tuple[str, Reference | None]] = [
        (FIXED_QUESTIONS["patient_name"], patient_name),
        (FIXED_QUESTIONS["birth_date"], birth_date),
        (FIXED_QUESTIONS["admission_date"], admission_date),
        (FIXED_QUESTIONS["discharge_date"], discharge_date),
    ]

    for question, reference in mapping:
        if reference is None:
            continue
        yield {
            "question": question,
            "references": json.dumps([reference.to_dict()], ensure_ascii=False),
            "corpus_id": str(corpus_path),
        }


def generate_fixed_questions_csv(text_files: Iterable[Path], csv_path: Path) -> int:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    row_count = 0

    with csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["question", "references", "corpus_id"])
        writer.writeheader()

        for text_file in text_files:
            text = text_file.read_text(encoding="utf-8")
            for row in _question_rows_for_document(text_file, text):
                writer.writerow(row)
                row_count += 1

    return row_count


def build_fixed_grascco_eval_dataset(
    json_dir: Path,
    text_dir: Path,
    questions_csv_path: Path,
) -> tuple[int, int]:
    text_files = convert_grascco_json_to_text_files(json_dir, text_dir)
    question_rows = generate_fixed_questions_csv(text_files, questions_csv_path)
    return len(text_files), question_rows


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parents[2]
    json_input_dir = repo_root / "data" / "data-json" / "grascco-json" / "grascco_phi_annotation_json"
    text_output_dir = repo_root / "data" / "processed" / "grascco_texts"
    questions_csv_output = repo_root / "data" / "processed" / "grascco_fixed_questions.csv"

    converted_count, rows_count = build_fixed_grascco_eval_dataset(
        json_dir=json_input_dir,
        text_dir=text_output_dir,
        questions_csv_path=questions_csv_output,
    )

    print(f"Converted JSON documents: {converted_count}")
    print(f"Generated fixed question rows: {rows_count}")
    print(f"Questions CSV: {questions_csv_output}")