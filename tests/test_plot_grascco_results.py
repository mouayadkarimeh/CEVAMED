import json

from src.evaluation.plot_grascco_results import _get_all_metric_keys, _load_results


def test_load_results_normalizes_chunker_first_report(tmp_path) -> None:
    report_path = tmp_path / "report.json"
    report_path.write_text(
        json.dumps(
            {
                "example_chunker": {
                    "adaptive_k": {
                        "global_metrics": {
                            "intrachunk_cohesion_global_mean": 0.8,
                        }
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    results = _load_results(report_path)

    assert results["profiles"]["adaptive_k"]["example_chunker"] == {
        "intrachunk_cohesion_global_mean": 0.8
    }
    assert "intrachunk_cohesion_global_mean" in _get_all_metric_keys(results)