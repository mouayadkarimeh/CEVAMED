# Chunking evaluation of medical reports
CEVAMED is developed as bachelorwork and an  experimental library to investigate different chunking strategies in RAG(retrieval Augumented Generation) Pipeline to generate the relevant chunks from the reports based on questions so that the LLM (large language Modell) extract as possibl the correct Answer improving accuricy in critical medical documents domaäns.


## Dataset(GRASCCO)
GraSCCo  is a collection of artificially generated semi-structured and unstructured German-language clinical summaries. These summaries are formulated as letters from the hospital to the patient's GP after in-patient or out-patient care. Details:

## Chunking strategieis
testing 4 chunking strategies Fixed-token, recursive, semantic, architecutural


## Evaluation Metrics

- **Precision**: Measures the proportion of predicted chunks that contain ground-truth information, evaluating how many returned chunks are actually correct.

- **Recall**: Measures the proportion of ground-truth chunks correctly identified, evaluating how many relevant chunks the system finds.

- **F1-Score**: The harmonic mean of precision and recall, balancing both metrics into a single value to compare strategies.

- **nDCG (Normalized Discounted Cumulative Gain)**: Evaluates the ranking quality of chunks based on graded relevance, ensuring the most relevant chunks appear first.

- **hit@k**: Checks if at least one relevant chunk appears in the top-*k* results, measuring early retrieval success.

- **MRR (Mean Reciprocal Rank)**: Measures how early the first relevant chunk appears in the results, indicating retrieval speed.



## Quickstart
If you just want to it try out, you can clone the project and install dependencies with `pip`:

```shell
git clone https://github.com/mouayadkarimeh/CEVAMED.git
pip install -e ".[full]"
python src.evaluation.report_single_document_question_chunks.py
```

<details>
<summary>Source: src.evaluation.report_single_document_question_chunks.py</summary>
```python
--8<-- Source: src.evaluation.report_single_document_question_chunks.py ""
```
</details>




### Evaluated questions (german)

| # | Frage | Quelle|
|---|-------|-------------|
| 1 | Wie heißt der Patient? | PHI-Annotation `NAME_PATIENT` |
| 2 | Wann hat der Patient Geburtstag? | `DATE`-Annotation nahe „geboren" |
| 3 | Wann wurde der Patient aufgenommen? | `DATE`-Annotation nahe Aufnahme-Kontext |
| 4 | Wann wurde der Patient entlassen? | `DATE`-Annotation nahe Entlassungs-Kontext |



