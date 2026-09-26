from src.evaluation.basic_evaluation import (
    BaseEvaluation,
    _alnum_normalize_with_map,
    find_target_with_alnum_normalization,
    find_target_with_lexical_tokens,
    intersect_two_ranges,
)


def test_touching_half_open_ranges_do_not_overlap() -> None:
    assert intersect_two_ranges((0, 5), (5, 10)) is None
    assert intersect_two_ranges((0, 6), (5, 10)) == (5, 6)


def test_precision_and_recall_are_calculated_independently() -> None:
    metrics = BaseEvaluation._binary_metrics_from_confusion(
        {"k": 2, "tp": 1, "fp": 1, "tn": 3, "fn": 2}
    )

    assert metrics == {
        "k": 2,
        "tp": 1,
        "fp": 1,
        "tn": 3,
        "fn": 2,
        "precision_at_k": 0.5,
        "recall_at_k": 1 / 3,
    }
    assert "f1_at_k" not in metrics
    assert "mrr" not in metrics
    assert "ndcg" not in metrics


def test_alnum_normalization_maps_expanding_casefold_characters() -> None:
    document = "Die Straße enthält einen Befund."
    normalized, index_map = _alnum_normalize_with_map(document)

    assert len(normalized) == len(index_map)
    assert find_target_with_alnum_normalization(
        document,
        "DIE STRASSE ENTHÄLT EINEN BEFUND",
        normalized,
        index_map,
    ) == (0, 31)


def test_lexical_alignment_handles_spacing_and_unknown_tokens() -> None:
    document = "CT-Abdomen: 74.5 mg wurden verabreicht."
    reconstructed_chunk = "CT - Abdomen : 74 . 5 [UNK] wurden verabreicht ."

    assert find_target_with_lexical_tokens(document, reconstructed_chunk) == (
        0,
        len(document),
    )