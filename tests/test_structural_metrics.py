import json
from types import SimpleNamespace

from src.evaluation.run_grascco_fixed_eval import (
    evaluate_context_compactness,
    evaluate_document_structural_metrics,
)


class FakeCompletions:
    def __init__(self, result: dict) -> None:
        self.result = result
        self.request = None

    def create(self, **kwargs):
        self.request = kwargs
        message = SimpleNamespace(content=json.dumps(self.result))
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def _client_with_result(result: dict):
    completions = FakeCompletions(result)
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    return client, completions


def test_document_metrics_are_position_aware_ratios() -> None:
    client, completions = _client_with_result(
        {
            "block_assessments": [
                {"block_id": 1, "chunk_ids": [1], "intact": True},
                {"block_id": 2, "chunk_ids": [1, 2], "intact": False},
            ],
            "reference_assessments": [
                {"reference_id": 1, "intact": True},
                {"reference_id": 2, "intact": False},
            ],
        }
    )
    chunks = [
        {"rank": 1, "start_index": 100, "end_index": 150, "text": "Zweiter Teil."},
        {"rank": 2, "start_index": 10, "end_index": 60, "text": "Erster Teil."},
    ]

    scores = evaluate_document_structural_metrics(chunks, client)

    assert scores == {
        "block_integrity": 0.5,
        "reference_completeness": 0.5,
        "block_count": 2,
        "split_block_count": 1,
        "reference_pair_count": 2,
        "split_reference_count": 1,
    }
    prompt = completions.request["messages"][1]["content"]
    assert prompt.index('start="10"') < prompt.index('start="100"')


def test_context_compactness_is_inverse_minimum_chunk_count() -> None:
    client, _ = _client_with_result(
        {"answer_supported": True, "minimum_chunks_needed": 2}
    )
    chunks = [
        {"rank": 1, "start_index": 0, "end_index": 10, "text": "Teil eins."},
        {"rank": 2, "start_index": 20, "end_index": 30, "text": "Teil zwei."},
    ]

    score = evaluate_context_compactness("Frage?", chunks, "Antwort", client)

    assert score == 0.5


def test_context_compactness_rejects_invalid_minimum_chunk_count() -> None:
    client, _ = _client_with_result(
        {
            "answer_supported": True,
            "minimum_chunks_needed": 2,
        }
    )
    chunks = [
        {"rank": 1, "start_index": 0, "end_index": 10, "text": "Antwort."},
    ]

    score = evaluate_context_compactness("Frage?", chunks, "Antwort", client)

    assert score is None


def test_context_compactness_ignores_count_when_answer_is_unsupported() -> None:
    client, _ = _client_with_result(
        {"answer_supported": False, "minimum_chunks_needed": 2}
    )
    chunks = [
        {"rank": 1, "start_index": 0, "end_index": 10, "text": "Teil eins."},
        {"rank": 2, "start_index": 20, "end_index": 30, "text": "Teil zwei."},
    ]

    score = evaluate_context_compactness("Frage?", chunks, "Fehlende Antwort", client)

    assert score == 0.0


def test_reference_completeness_is_missing_without_reference_pairs() -> None:
    client, _ = _client_with_result(
        {
            "block_assessments": [
                {"block_id": 1, "chunk_ids": [1], "intact": True}
            ],
            "reference_assessments": [],
        }
    )
    chunks = [
        {"start_index": 0, "end_index": 10, "text": "Vollständiger Satz."}
    ]

    scores = evaluate_document_structural_metrics(chunks, client)

    assert scores == {
        "block_integrity": 1.0,
        "reference_completeness": None,
        "block_count": 1,
        "split_block_count": 0,
        "reference_pair_count": 0,
        "split_reference_count": 0,
    }