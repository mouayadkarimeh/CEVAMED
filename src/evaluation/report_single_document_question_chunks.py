# ======================================================================
# VOLLSTÄNDIGER KORRIGIERTER CODE MIT ALLEN FIXES
# Löst VertexAI-Fehler und LM Studio Probleme
# ======================================================================



# 2. LM STUDIO KONFIGURATION (MUSS VOR allen anderen imports stehen!)
import os

os.environ["OPENAI_API_BASE"] = "https://chat.kiconnect.nrw/api/v1"
#os.environ["RAGAS_DO_NOT_TRACK"] = "true"

# ======================================================================

# 3. STANDARD IMPORTS
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd
from datasets import Dataset
import http
import logging


# 4. RAGAS IMPORTS
from ragas import evaluate
from ragas.metrics import ContextPrecision
from ragas.metrics  import ContextRecall
from ragas.llms import  llm_factory

from openai import OpenAI
from langchain_huggingface import HuggingFaceEmbeddings

# 6. SYSTEM IMPORTS
from transformers import AutoTokenizer
import chromadb.utils.embedding_functions as embedding_functions

# 7. LOKALE IMPORTS
from evaluation.fixed_question_dataset import build_fixed_grascco_eval_dataset
from evaluation.basic_evaluation import BaseEvaluation
from evaluation.intrachunk_cohesion import (
    intrachunk_cohesion,
    load_icc_embedding_model,
    parse_text_blocks,
)

# Chunker-Imports
from langchain_text_splitters import CharacterTextSplitter
from chunking.trasnformer_token_chunker import TransformerTokenChunker
from chunking.recursive_token_chunker import RecursiveTokenChunker
from chunking.kamradt_semantic_chunker import KamradtSemanticChunker
from chunking.recusrive_semantic_chunker import RecursiveSemanticChunker
from chunking.cluster_semantic_chunker import ClusterSemanticChunker

# System-Globale Variablen
REPO_ROOT = Path(__file__).resolve().parents[2]
RETRIEVE_PROFILES = [("top_3", 3)]

# Deutscher Tokenizer
_TOKENIZER = AutoTokenizer.from_pretrained("deutsche-telekom/gbert-large-paraphrase-cosine")


import http.client
import logging

# 1. Debug-Modus für HTTP-Verbindungen aktivieren
http.client.HTTPConnection.debuglevel = 1

# 2. Einen sauberen Logger erstellen, der NUR in eine Datei schreibt
http_logger = logging.getLogger("urllib3")
http_logger.setLevel(logging.DEBUG)

# Log-Datei definieren (wird im Projektordner erstellt)
log_file_path = "network_http_debug.log"
file_handler = logging.FileHandler(log_file_path, mode="w", encoding="utf-8")
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
http_logger.addHandler(file_handler)

# Verhindern, dass die HTTP-Ausgaben das Terminal fluten
# http_logger.propagate = 

def _count_tokens(text: str) -> int:
    """Token count used only for reporting."""
    return len(_TOKENIZER.tokenize(text))


def _format_score(value: object) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return "n/a"


def _valid_mean(values: list[object]) -> float | None:
    valid_values = [float(value) for value in values if value is not None]
    return sum(valid_values) / len(valid_values) if valid_values else None

ALL_CHUNKER_NAMES = [
    "charachter_text_splitter",
    "transformer_token_chunker",
    "recursive_token_chunker",
    "kamradt_semantic_chunker",
    "recursive_semantic_chunker",
    "cluster_semantic_chunker"
]

def _build_chunker(chunker_name: str):
    """Erzeugt optimierte Chunker für medizinische Texte."""
    chunkers = {
        "charachter_text_splitter": lambda: CharacterTextSplitter(
            chunk_size=256,
            chunk_overlap=64,
            separator="\n\n",
            length_function=lambda x: len(_TOKENIZER.tokenize(x))
        ),
        "transformer_token_chunker": lambda: TransformerTokenChunker(
            chunk_overlap=64,
            model_name="deutsche-telekom/gbert-large-paraphrase-cosine",
            tokens_per_chunk=256,
            
        ),
        "recursive_token_chunker": lambda: RecursiveTokenChunker(
            chunk_size=256,
            chunk_overlap=64
        ),
        "kamradt_semantic_chunker": lambda: KamradtSemanticChunker(
           
        ),
        "recursive_semantic_chunker": lambda: RecursiveSemanticChunker(
            avg_chunk_size=256,
            min_chunk_size=50
        ),
        "cluster_semantic_chunker": lambda: ClusterSemanticChunker(
            max_chunk_size=256,
            min_chunk_size=50
        ),
    }
    if chunker_name not in chunkers:
        raise ValueError(f"Unknown chunker '{chunker_name}'. Choose from: {list(chunkers)}")
    return chunkers[chunker_name]()


def _build_single_document_questions_csv(
    full_questions_csv: Path, document_name: str, output_csv: Path
) -> Path:
    """Filtert Fragen für ein einzelnes Dokument."""
    questions_df = pd.read_csv(full_questions_csv)
    document_rows = questions_df[questions_df["corpus_id"].str.endswith(document_name)]
    if document_rows.empty:
        raise ValueError(f"No groundtruth questions found for document: {document_name}")
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    document_rows.to_csv(output_csv, index=False)
    return output_csv

def _write_json_report(chunker_results: dict[str, dict[str, list[dict]]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(chunker_results, file, indent=2, ensure_ascii=False)

def _write_text_report(
    chunker_results: dict[str, dict[str, dict]],
    document_name: str,
    output_path: Path,
) -> None:
    lines: list[str] = [
        f"Question-to-Chunk Report for {document_name}",
        f"Chunking strategies tested: {', '.join(chunker_results.keys())}",
        "=" * 60,
        "",
    ]

    for chunker_name, profile_results in chunker_results.items():
        lines.append("#" * 70)
        lines.append(f"# CHUNKER: {chunker_name}")
        lines.append("#" * 70)
        lines.append("")

        for profile_name, profile_data in profile_results.items():
            question_metrics = profile_data["questions"]
            binary_document = profile_data["binary_document"]

            lines.append(f"### PROFILE: {profile_name} (chunker: {chunker_name}) ###")
            lines.append("")
            lines.append(
                "Dokument-Metriken (Makro-Mittel über alle Fragen):\n"
                f"  [Klassisch]   precision@k_doc={binary_document['precision_at_k']:.3f} "
                f"recall@k_doc={binary_document['recall_at_k']:.3f}\n"
              
                f"  [RAGAS Soft]  Context-Precision={_format_score(binary_document.get('ragas_precision_doc'))} "
                f"Context-Recall={_format_score(binary_document.get('ragas_recall_doc'))}\n"
                
                f"  [Strukturell] Block-Integrity={_format_score(binary_document.get('block_integrity_doc'))} "
                f"Reference-Completeness(RC)={_format_score(binary_document.get('reference_completeness_doc'))} "
                f"Context-Fragmentation-Rate(CFR)={_format_score(binary_document.get('context_fragmentation_doc'))} "
                f"Intrachunk-Cohesion(ICC)={_format_score(binary_document.get('intrachunk_cohesion_doc'))}\n"
                f"                (Fragen: {binary_document['num_questions']})"
            )
            lines.append("")

            for entry in question_metrics:
                lines.append(f"Question ({entry['question_key']}): {entry['question']}")
                lines.append("Groundtruth:")
                for reference in entry["groundtruth"]:
                    lines.append(
                        f"  - '{reference['content']}' "
                        f"[{reference['start_index']}-{reference['end_index']}]"
                    )

                lines.append(f"Retrieved chunks ({len(entry['retrieved_chunks'])}):")
                for chunk in entry["retrieved_chunks"]:
                    marker = "RELEVANT" if chunk["relevant"] else "not relevant"
                    char_length = chunk["end_index"] - chunk["start_index"]
                    token_count = _count_tokens(chunk["text"])
                    lines.append("  " + "-" * 70)
                    lines.append(
                        f"  CHUNK START (rank {chunk['rank']}, {marker}) "
                        f"| position [{chunk['start_index']}-{chunk['end_index']}] "
                        f"| chars={char_length} | tokens={token_count}"
                    )
                    lines.append(f"    {chunk['text'][:200]}...")  # Text gekürzt
                    lines.append(f"  CHUNK END (rank {chunk['rank']})")
                    lines.append("  " + "-" * 70)

                metrics = entry["metrics"]
                lines.append(
                    f"Metriken (K={metrics['k']}):\n"
                    f"  -> Klassisch:  tp={metrics['tp']} fp={metrics['fp']} tn={metrics['tn']} fn={metrics['fn']} "
                    f"| precision@k={metrics['precision_at_k']:.3f} recall@k={metrics['recall_at_k']:.3f}\n"
                    f"  -> RAGAS:      Context-Precision={_format_score(metrics.get('ragas_context_precision'))} "
                    f"Context-Recall={_format_score(metrics.get('ragas_context_recall'))}\n"
                    f" ragas_llm_explanation = {metrics.get('ragas_llm_explanations')}"
                    f"  -> Struktur:   Block-Integrity={_format_score(metrics.get('block_integrity'))} "
                    f"Reference-Completeness(RC)={_format_score(metrics.get('reference_completeness'))} "
                    f"Context-Fragmentation(CFR)={_format_score(metrics.get('context_fragmentation_rate'))}"
                )

                


                lines.append("")

            lines.append("")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")

def evaluate_custom_structural_metrics(question: str, chunks: list[str], ground_truth: str, client) -> dict:
    """
    Evaluiert strukturelle Metriken mit Mistral über den konfigurierten Client.
    """
    combined_context = " --NEW_CHUNK-- ".join(chunks)

    prompt = f"""
Du bist ein wissenschaftlicher Gutachter für Information-Retrieval-Systeme im medizinischen Kontext.
Bewerte den Kontext anhand dieser Kriterien (Skala 0.0-1.0):

1. block_integrity: Sind Sätze logisch intakt? (1.0 = perfekt, 0.0 = fragmentiert)
2. reference_completeness: Sind Pronomen/Bezüge im Chunk enthalten? (1.0 = ja, 0.0 = nein)
3. context_fragmentation_rate: Liegt die Information kompakt in 1 Chunk? (1.0 = ja, 0.0 = nein)

Frage: "{question}"
Ground Truth: "{ground_truth}"
Chunks: "{combined_context[:1000]}..."

Gib nur JSON zurück:
{{
  "block_integrity": 0.90,
  "reference_completeness": 1.00,
  "context_fragmentation_rate": 0.00
}}
"""
    try:
        response = client.chat.completions.create(
            model="Mistral Small 4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            #response_format={"type": "json_object"}  # Verhindert kaputtes JSON
        )
        print(response)
        res_text = response.choices[0].message.content.strip()
        
        # JSON-Codeblöcke säubern, falls das Modell welche generiert hat
        if "```" in res_text:
            res_text = res_text.split("```")[1].strip()
        if res_text.startswith("json"):
            res_text = res_text[4:].strip()

        # Das Return muss die LETZTE Aktion im try-Block sein
        return json.loads(res_text)

    except Exception as e:
        print(f"⚠️ Strukturelle Evaluation fehlgeschlagen: {e}")
        return {
            "block_integrity": None,
            "reference_completeness": None,
            "context_fragmentation_rate": None,
        }




def run(document_name: str, chunker_names: List[str]) -> None:
    # 1. Daten laden
    annotations_path = REPO_ROOT / "data" / "data-json" / "grascco-test.json"
    text_dir = REPO_ROOT / "data" / "row"
    full_questions_csv = REPO_ROOT / "data" / "processed" / "grascco_fixed_questions.csv"
    build_fixed_grascco_eval_dataset(annotations_path, text_dir, full_questions_csv)

    document_stem = Path(document_name).stem
    single_document_questions_csv = REPO_ROOT / "data" / "processed" / f"{document_stem}_only_questions.csv"
    _build_single_document_questions_csv(full_questions_csv, document_name, single_document_questions_csv)

    # 2. Evaluator und Embeddings
    evaluator = BaseEvaluation(questions_csv_path=str(single_document_questions_csv))
    embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="deutsche-telekom/gbert-large-paraphrase-cosine"
    )
    telekom_embeddings = HuggingFaceEmbeddings(
        model_name="deutsche-telekom/gbert-large-paraphrase-cosine",
        model_kwargs={'device': 'cpu'}
    )
    corpus_id = str(evaluator.questions_df.iloc[0]["corpus_id"])
    text_blocks = parse_text_blocks(Path(corpus_id).read_text(encoding="utf-8"))
    icc_embedding_model = load_icc_embedding_model()

    # API-Konfiguration
    api_key = os.environ.get("OPENAI_API_KEY")
    base_url = os.environ.get("OPENAI_API_BASE")

    if not api_key or not base_url:
        raise ValueError("OPENAI_API_KEY oder OPENAI_API_BASE nicht in Umgebungsvariablen gesetzt!")

    # Clients initialisieren
    kiconnect_client = OpenAI(api_key=api_key, base_url=base_url)
    # LLM-Konfiguration
    llm_kiconnect = llm_factory(
        model="Mistral Small 4",
        client=kiconnect_client,
        provider="openai",
        temperature=0.0,
        system_prompt="Return ONLY raw JSON. Never wrap the response in markdown code blocks like ```json."
        #response_format={"response_format": {"type": "json_object"}}
    )

    # Metriken initialisieren (nur einmal!)
    metrics = [
        ContextPrecision(llm=llm_kiconnect),
        ContextRecall(llm=llm_kiconnect),
        
    ]

    print("[RAGAS INIT] Erfolgreich geladen.\n")

    # 4. Chunker-Tests
    chunker_results: Dict[str, Dict[str, Any]] = {}
    for chunker_name in chunker_names:
        chunker = _build_chunker(chunker_name)
        profile_results: Dict[str, Any] = {}
        for profile_name, retrieve in RETRIEVE_PROFILES:
            # Klassische Evaluation
            metrics_eval = evaluator.run(
                chunker=chunker,
                embedding_function=embedding_function,
                retrieve=retrieve,
            )
            #print(f"the result od basicevaluation obj: {metrics_eval}")
            raw_eval_data = {
                "questions": metrics_eval["question_metrics"],
                "binary_document": next(iter(metrics_eval["per_document"].values())),
            }
            document_chunks = [
                chunk
                for chunk in metrics_eval["chunks"]
                if str(chunk["corpus_id"]) == corpus_id
            ]
            raw_eval_data["binary_document"]["intrachunk_cohesion_doc"] = intrachunk_cohesion(
                document_chunks,
                text_blocks,
                embedding_model=icc_embedding_model,
            )
            #print(f"raw eval data: {raw_eval_data['questions']}")
            print(f"[{chunker_name}] Evaluierung läuft...")

            # RAGAS-Dataset vorbereiten
            ragas_questions = []
            ragas_contexts = []
            ragas_ground_truths = []

            for entry in raw_eval_data["questions"]:
                ragas_questions.append(entry["question"])
                chunks_text = [chunk["text"] for chunk in entry["retrieved_chunks"]]
                ragas_contexts.append(chunks_text)

                g_truth_text = ""
                if entry.get("groundtruth"):
                    if isinstance(entry["groundtruth"], list):
                        g_truth_text = "\n".join(
                            str(reference.get("content", ""))
                            for reference in entry["groundtruth"]
                            if reference.get("content")
                        )
                    elif isinstance(entry["groundtruth"], dict):
                        g_truth_text = entry["groundtruth"].get("content", "")
                ragas_ground_truths.append(g_truth_text)

            eval_dataset = Dataset.from_dict({
                "question": ragas_questions,
                "contexts": ragas_contexts,
                "ground_truth": ragas_ground_truths
            })

            # Strukturelle Metriken berechnen
            sem_integrities = []
            ref_completenesses = []
            frag_rates = []

            for idx, entry in enumerate(raw_eval_data["questions"]):
                chunks_text = [chunk["text"] for chunk in entry["retrieved_chunks"]]
                custom_scores = evaluate_custom_structural_metrics(
                    question=entry["question"],
                    chunks=chunks_text,
                    ground_truth=ragas_ground_truths[idx],
                    client=kiconnect_client
                )
                entry["metrics"]["block_integrity"] = custom_scores["block_integrity"]
                entry["metrics"]["reference_completeness"] = custom_scores["reference_completeness"]
                entry["metrics"]["context_fragmentation_rate"] = custom_scores["context_fragmentation_rate"]

                sem_integrities.append(custom_scores["block_integrity"])
                ref_completenesses.append(custom_scores["reference_completeness"])
                frag_rates.append(custom_scores["context_fragmentation_rate"])

            # RAGAS-Evaluation mit Fehlerbehandlung
            try:
                ragas_output = evaluate(
                    dataset=eval_dataset,
                    metrics=metrics,
                    llm=llm_kiconnect,
                    embeddings=telekom_embeddings,
                    batch_size=1
                ).to_pandas()

                # Korrekte Spaltennamen verwenden
                for idx, entry in enumerate(raw_eval_data["questions"]):
                    entry["metrics"]["ragas_context_precision"] = float(ragas_output.iloc[idx]["context_precision"])
                    entry["metrics"]["ragas_context_recall"] = float(ragas_output.iloc[idx]["context_recall"])
                    #entry["metrics"]["ragas_context_relevancy"] = float(ragas_output.iloc[idx]["context_utilization"])
                    

                    reasons = {}
                    for col in ragas_output.columns:
                        if "reason" in col or "verdict" in col or "classification" in col:
                            reasons[col] = ragas_output.iloc[idx][col]
                    
                    # Speichere die Begründungen direkt im JSON-Eintrag der Frage ab!
                    entry["metrics"]["ragas_llm_explanations"] = reasons

                raw_eval_data["binary_document"]["ragas_precision_doc"] = float(ragas_output["context_precision"].mean())
                raw_eval_data["binary_document"]["ragas_recall_doc"] = float(ragas_output["context_recall"].mean())
            except Exception as e:
                print(f"⚠️ RAGAS-Fehler bei {chunker_name}: {str(e)}")
                for entry in raw_eval_data["questions"]:
                    entry["metrics"]["ragas_context_precision"] = None
                    entry["metrics"]["ragas_context_recall"] = None
                raw_eval_data["binary_document"]["ragas_precision_doc"] = None
                raw_eval_data["binary_document"]["ragas_recall_doc"] = None

            # Strukturelle Metriken speichern
            raw_eval_data["binary_document"]["block_integrity_doc"] = _valid_mean(sem_integrities)
            raw_eval_data["binary_document"]["reference_completeness_doc"] = _valid_mean(ref_completenesses)
            raw_eval_data["binary_document"]["context_fragmentation_doc"] = _valid_mean(frag_rates)

            profile_results[profile_name] = raw_eval_data

        chunker_results[chunker_name] = profile_results

    # Berichte generieren
    json_output = REPO_ROOT / "data" / "processed" / f"{document_stem}_question_chunk_report.json"
    text_output = REPO_ROOT / "data" / "processed" / f"{document_stem}_question_chunk_report.txt"
    _write_json_report(chunker_results, json_output)
    _write_text_report(chunker_results, document_name, text_output)

    print("\n" + "="*70)
    print("🎉 Evaluation erfolgreich abgeschlossen!")
    print(f"Bericht gespeichert unter: {text_output}")
    print("="*70 + "\n")









if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Multidimensionale Chunk-Evaluation")
    parser.add_argument("--document", default="Albers.txt", help="Zu analysierendes Dokument")
    parser.add_argument("--chunker", default="all", choices=["all"] + ALL_CHUNKER_NAMES,
                      help="Chunking-Strategie (Standard: alle)")
    args = parser.parse_args()

    print("="*70)
    print(f"Starte Evaluation für Dokument: {args.document}")
    print(f"Chunking-Strategien: {', '.join(ALL_CHUNKER_NAMES if args.chunker == 'all' else [args.chunker])}")
    print("="*70)

    
    run(args.document, ALL_CHUNKER_NAMES if args.chunker == "all" else [args.chunker])
    
    
    