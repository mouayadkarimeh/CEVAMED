import numpy as np

from src.evaluation.intrachunk_cohesion import intrachunk_cohesion, parse_text_blocks


class FakeEmbeddingModel:
    def __init__(self, embeddings: dict[str, list[float]]) -> None:
        self.embeddings = embeddings

    def encode(self, sentences: list[str], **_: object) -> np.ndarray:
        return np.asarray([self.embeddings[text] for text in sentences])


class FakeFastTokenizer:
    def __call__(self, text: str, **_: object) -> dict[str, list[tuple[int, int]]]:
        offsets = [
            (index, index + 1)
            for index, character in enumerate(text)
            if not character.isspace()
        ]
        return {"offset_mapping": offsets}


def test_intrachunk_cohesion_means_valid_chunks_and_ignores_single_blocks() -> None:
    chunks = [
        {"text": "chunk-a", "start_index": 0, "end_index": 10},
        {"text": "chunk-b", "start_index": 10, "end_index": 20},
        {"text": "chunk-c", "start_index": 20, "end_index": 30},
    ]
    blocks = [
        {"text": "a1", "start_index": 0},
        {"text": "a2", "start_index": 5},
        {"text": "b1", "start_index": 10},
        {"text": "b2", "start_index": 15},
        {"text": "ignored", "start_index": 20},
    ]
    model = FakeEmbeddingModel(
        {
            "chunk-a": [1.0, 0.0],
            "a1": [1.0, 0.0],
            "a2": [0.0, 1.0],
            "chunk-b": [0.0, 1.0],
            "b1": [0.0, 1.0],
            "b2": [0.0, 2.0],
        }
    )

    score = intrachunk_cohesion(chunks, blocks, embedding_model=model)

    assert score == 0.75


def test_intrachunk_cohesion_clips_negative_mean_to_zero() -> None:
    chunks = [{"content": "chunk", "start_index": 0, "end_index": 10}]
    blocks = [
        {"content": "one", "start_index": 1},
        {"content": "two", "start_index": 2},
    ]
    model = FakeEmbeddingModel(
        {"chunk": [1.0, 0.0], "one": [-1.0, 0.0], "two": [-1.0, 0.0]}
    )

    assert intrachunk_cohesion(chunks, blocks, embedding_model=model) == 0.0


def test_intrachunk_cohesion_returns_zero_without_valid_chunks() -> None:
    chunks = [{"text": "chunk", "start_index": 0, "end_index": 10}]
    blocks = [{"text": "only block", "start_index": 0}]

    assert intrachunk_cohesion(chunks, blocks) == 0.0


def test_parse_text_blocks_retains_sentence_start_indexes() -> None:
    text = "Erster Satz. Zweiter Satz!"

    assert parse_text_blocks(text, tokenizer=FakeFastTokenizer()) == [
        {"text": "Erster Satz.", "start_index": 0, "end_index": 12},
        {"text": "Zweiter Satz!", "start_index": 13, "end_index": 26},
    ]