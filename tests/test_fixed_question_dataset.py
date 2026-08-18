import json
from pathlib import Path

from src.evaluation.fixed_question_dataset import _question_rows_for_document_from_payload


def test_question_rows_from_annotations_use_json_offsets():
    payload = {
        "%FEATURE_STRUCTURES": [
            {
                "%TYPE": "uima.cas.Sofa",
                "sofaString": "Patient: Anna Müller\nGeburtsdatum: 05.07.1954\nAufnahme: 01.01.2024\nEntlassung: 15.01.2024",
            },
            {"%TYPE": "webanno.custom.PHI", "kind": "NAME_PATIENT", "begin": 9, "end": 19},
            {"%TYPE": "webanno.custom.PHI", "kind": "DATE", "begin": 33, "end": 43},
            {"%TYPE": "webanno.custom.PHI", "kind": "DATE", "begin": 55, "end": 65},
            {"%TYPE": "webanno.custom.PHI", "kind": "DATE", "begin": 78, "end": 88},
        ]
    }

    rows = list(_question_rows_for_document_from_payload(Path("dummy.json"), payload))

    assert len(rows) == 4

    references = [json.loads(row["references"]) for row in rows]
    assert references[0][0]["content"] == "Anna Müller"
    assert references[0][0]["start_index"] == 9
    assert references[0][0]["end_index"] == 19

    assert any(ref[0]["content"] == "05.07.1954" for ref in references)
    assert any(ref[0]["content"] == "01.01.2024" for ref in references)
    assert any(ref[0]["content"] == "15.01.2024" for ref in references)
