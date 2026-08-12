Chunking strategies benchmark

## Evaluation Framework (GraSCCo)

Adaptiert aus [brandonstarxel/chunking_evaluation](https://github.com/brandonstarxel/chunking_evaluation).
**Kein OpenAI-Modell erforderlich** – Retrieval erfolgt über einen reinen Python-BM25-Scorer.

### Ablauf lokal starten

```bash
# 1. GRASCCO JSON → Textdateien konvertieren und Evaluation starten
python scripts/run_evaluation.py

# Mit anderen Optionen:
python scripts/run_evaluation.py --chunker recursive --chunk-size 400 --overlap 50

# Ergebnis als JSON speichern:
python scripts/run_evaluation.py --output results.json
```

Nur Konvertierung (ohne Evaluation):

```bash
python scripts/convert_grascco_json.py \
    --input  data/data-json/grascco-json/grascco_phi_annotation_json \
    --output data/grascco_texts
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
  convert_grascco_json.py   # JSON → TXT + meta.json
  run_evaluation.py         # kompletter Pipeline-Run

src/evaluation/
  grascco_evaluator.py      # adaptiertes Framework (kein OpenAI)

data/
  data-json/grascco-json/   # originale GRASCCO JSON-Dateien
  grascco_texts/            # generiert – nicht im Repo versioniert
```
