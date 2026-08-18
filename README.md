Chunking strategies benchmark

## Evaluation Framework (GraSCCo)

Adaptiert aus [brandonstarxel/chunking_evaluation](https://github.com/brandonstarxel/chunking_evaluation).
**Kein OpenAI-Modell erforderlich** – Retrieval erfolgt über einen reinen Python-BM25-Scorer.

### Ablauf lokal starten

```bash
# 1. Evaluation mit den vorhandenen Textdateien und Annotationen starten
python -m src.evaluation.run_grascco_fixed_eval

# Mit anderen Optionen:
python -m src.evaluation.run_grascco_fixed_eval

# Ergebnis als JSON speichern:
python -m src.evaluation.run_grascco_fixed_eval
```

### Evaluierte Fragen

| # | Frage | Datenquelle |
|---|-------|-------------|
| 1 | Wie heißt der Patient? | PHI-Annotation `NAME_PATIENT` |
| 2 | Wann hat der Patient Geburtstag? | `DATE`-Annotation nahe „geboren" |
| 3 | Wann wurde der Patient aufgenommen? | `DATE`-Annotation nahe Aufnahme-Kontext |
| 4 | Wann wurde der Patient entlassen? | `DATE`-Annotation nahe Entlassungs-Kontext |

### Metriken

* **IoU (Intersection over Union)** – Anteil überlappender Zeichen zwischen erwartetem Excerpt und gefundenem Chunk.
* **Recall** – Anteil der erwarteten Zeichen, der in mindestens einem der top-k abgerufenen Chunks enthalten ist.

### Verzeichnisstruktur

```
scripts/
  run_evaluation.py         # kompletter Pipeline-Run

src/evaluation/
  run_grascco_fixed_eval.py # Evaluation mit Annotationen und data/row

data/
  data-json/                 # originale GRASCCO Annotationen
  row/                       # vorhandene Dokumenttexte für Chunking
```
