import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.chunking.semantic import SemanticChunker
from src.evaluation.fixed_question_dataset import build_fixed_grascco_eval_dataset


def test_albers_four_questions_match_semantic_chunks():
    annotations_path = ROOT / "data" / "data-json" / "grascco-test.json"
    text_dir = ROOT / "data" / "row"
    questions_path = ROOT / "data" / "processed" / "grascco_fixed_questions.csv"
    build_fixed_grascco_eval_dataset(annotations_path, text_dir, questions_path)

    import pandas as pd

    text_path = text_dir / "Albers.txt"
    text = text_path.read_text(encoding="utf-8")
    rows = pd.read_csv(questions_path)
    rows = rows[rows["corpus_id"] == str(text_path)]

    assert set(rows["question_key"]) == {
        "patient_name",
        "birth_date",
        "admission_date",
        "discharge_date",
    }

    chunks = SemanticChunker(avg_chunk_size=400, min_chunk_size=50).split_text(text)
    assert chunks
    assert all(chunk in text for chunk in chunks)

    for row in rows.itertuples(index=False):
        references = json.loads(row.references)
        for reference in references:
            start = int(reference["start_index"])
            end = int(reference["end_index"])
            assert any(
                start >= text.find(chunk)
                and end <= text.find(chunk) + len(chunk)
                for chunk in chunks
            ), f"{row.question_key}: groundtruth is not fully contained in a chunk"


if __name__ == "__main__":
    test_albers_four_questions_match_semantic_chunks()
    print("test_albers_four_questions_match_semantic_chunks: PASSED")

