import json

import pandas as pd
import pytest

from src.evaluation.basic_evaluation import BaseEvaluation


def test_evaluator_keeps_loaded_questions_when_source_is_removed(tmp_path) -> None:
    questions_path = tmp_path / "questions.csv"
    pd.DataFrame(
        [
            {
                "question_key": "patient_name",
                "question": "Wie heißt der Patient?",
                "references": json.dumps([]),
                "corpus_id": "document.txt",
            }
        ]
    ).to_csv(questions_path, index=False)
    evaluator = BaseEvaluation(str(questions_path))

    questions_path.unlink()

    assert evaluator.questions_df["question"].tolist() == ["Wie heißt der Patient?"]
    assert evaluator.corpus_list == ["document.txt"]


def test_evaluator_rejects_missing_questions_csv(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="Questions CSV does not exist"):
        BaseEvaluation(str(tmp_path / "missing.csv"))