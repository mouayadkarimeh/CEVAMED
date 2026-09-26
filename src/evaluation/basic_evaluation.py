import chromadb.utils.embedding_functions as embedding_functions
import os
import pandas as pd
import json
import chromadb
import numpy as np
from importlib import resources
import re
import unicodedata
from bisect import bisect_left


#MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
#MODEL_NAME = "deutsche-telekom/gbert-large-paraphrase-cosine"
MODEL_NAME = "codefuse-ai/F2LLM-v2-0.6B"



def rigorous_document_search(document: str, target: str):
        """Locate a target snippet in a document and return (match, start, end)."""
        start_index = document.find(target)
        if start_index != -1:
            end_index = start_index + len(target)
            return target, start_index, end_index

        normalized_target = " ".join(target.split())
        if not normalized_target:
            return None

        compact_document = " ".join(document.split())
        compact_start = compact_document.find(normalized_target)
        if compact_start == -1:
            return None

        # Fall back to first exact substring of the normalized target to recover offsets.
        anchor = normalized_target[: min(24, len(normalized_target))]
        anchor_start = document.find(anchor)
        if anchor_start == -1:
            return None
        return target, anchor_start, anchor_start + len(target)



def intersect_two_ranges(range1, range2):
    # Unpack the ranges
    start1, end1 = range1
    start2, end2 = range2
    
    # Calculate the maximum of the starting indices and the minimum of the ending indices
    intersect_start = max(start1, start2)
    intersect_end = min(end1, end2)
    
    # Character offsets are half-open intervals: touching boundaries do not overlap.
    if intersect_start <= intersect_end:
        return (intersect_start, intersect_end)
    else:
        return None  # Return an None if there is no intersection

def find_target_in_document(document, target):
    start_index = document.find(target)
    if start_index == -1:
        return None
    end_index = start_index + len(target)
    return start_index, end_index

def find_target_with_flexible_whitespace(document: str, target: str, start_cursor: int = 0):
    tokens = target.split()
    if not tokens:
        return None

    pattern = r"\s+".join(re.escape(token) for token in tokens)

    local_match = re.search(pattern, document[start_cursor:])
    if local_match is not None:
        return start_cursor + local_match.start(), start_cursor + local_match.end()

    global_match = re.search(pattern, document)
    if global_match is not None:
        return global_match.start(), global_match.end()

    return None

def _alnum_normalize_with_map(text: str):
    normalized_chars = []
    index_map = []
    for original_index, char in enumerate(text):
        normalized = unicodedata.normalize("NFKC", char).casefold()
        for norm_char in normalized:
            if norm_char.isalnum():
                normalized_chars.append(norm_char)
                index_map.append(original_index)
    return "".join(normalized_chars), index_map

def find_target_with_alnum_normalization(
    document: str,
    target: str,
    doc_alnum: str,
    doc_alnum_map: list[int],
    start_cursor: int = 0,
):
    target_alnum, _ = _alnum_normalize_with_map(target)
    if len(target_alnum) < 20:
        return None

    alnum_start_cursor = bisect_left(doc_alnum_map, start_cursor)
    start_index = doc_alnum.find(target_alnum, alnum_start_cursor)
    if start_index == -1:
        start_index = doc_alnum.find(target_alnum)
        if start_index == -1:
            return None

    end_index = start_index + len(target_alnum) - 1
    original_start = doc_alnum_map[start_index]
    original_end = doc_alnum_map[end_index] + 1
    return original_start, original_end


def find_target_with_lexical_tokens(
    document: str,
    target: str,
    start_cursor: int = 0,
):
    """Locate reconstructed chunk text while preserving source character offsets."""
    tokens = re.findall(r"\[UNK\]|[^\W_]+", target, flags=re.UNICODE | re.IGNORECASE)
    lexical_tokens = [token for token in tokens if token.upper() != "[UNK]"]
    if len(lexical_tokens) < 3:
        return None

    pattern_parts = []
    for token in tokens:
        if token.upper() == "[UNK]":
            pattern_parts.append(r".{0,120}?")
        else:
            if pattern_parts and not pattern_parts[-1].endswith("?"):
                pattern_parts.append(r"\W*")
            pattern_parts.append(re.escape(token))
    if target and not target.rstrip()[-1].isalnum():
        pattern_parts.append(r"[^\w\r\n]*")
    pattern = "".join(pattern_parts)

    local_match = re.search(
        pattern,
        document[start_cursor:],
        flags=re.IGNORECASE | re.DOTALL,
    )
    if local_match is not None:
        return start_cursor + local_match.start(), start_cursor + local_match.end()

    global_match = re.search(pattern, document, flags=re.IGNORECASE | re.DOTALL)
    if global_match is not None:
        return global_match.start(), global_match.end()
    return None

# Rate-valued metrics that are macro-averaged (question -> document -> corpus).
MACRO_METRIC_NAMES = ("precision_at_k", "recall_at_k")


class BaseEvaluation:
    def __init__(self, questions_csv_path: str, chroma_db_path=None, corpora_id_paths=None):
        self.corpora_id_paths = corpora_id_paths

        self.questions_csv_path = questions_csv_path

        self.corpus_list = []

        self._load_questions_df()

        # self.questions_df = pd.read_csv(questions_csv_path)
        # self.questions_df['references'] = self.questions_df['references'].apply(json.loads)

        if chroma_db_path is not None:
            self.chroma_client = chromadb.PersistentClient(path=chroma_db_path)
        else:
            self.chroma_client = chromadb.Client()

        self.is_general = False

    def _load_questions_df(self):
        if not os.path.exists(self.questions_csv_path):
            raise FileNotFoundError(
                f"Questions CSV does not exist: {self.questions_csv_path}"
            )

        self.questions_df = pd.read_csv(self.questions_csv_path)
        if self.questions_df.empty:
            raise ValueError(f"Questions CSV is empty: {self.questions_csv_path}")
        self.questions_df['references'] = self.questions_df['references'].apply(json.loads)
        
        self.corpus_list = self.questions_df['corpus_id'].unique().tolist()

    def _get_chunks_and_metadata(self, splitter):
        # Warning: metadata will be incorrect if a chunk is repeated since we use .find() to find the start index. 
        # This isn't pratically an issue for chunks over 1000 characters.
        documents = []
        metadatas = []
        skipped_chunks = 0
        for corpus_id in self.corpus_list:
            corpus_path = corpus_id
            if self.corpora_id_paths is not None:
                corpus_path = self.corpora_id_paths[corpus_id]
    
            # Check the operating system and use UTF-8 encoding on Windows
            # This prevents UnicodeDecodeError when reading files with non-ASCII characters
            import platform
            if platform.system() == 'Windows':
                with open(corpus_path, 'r', encoding='utf-8') as file:
                    corpus = file.read()
            else:
                # Use default encoding on other systems
                with open(corpus_path, 'r') as file:
                    corpus = file.read()
    
            current_documents = splitter.split_text(corpus)
            print(f"total chunk number: {len(current_documents)}")
            current_metadatas = []
            current_documents_found = []
            search_cursor = 0
            corpus_alnum, corpus_alnum_map = _alnum_normalize_with_map(corpus)
            for document in current_documents:
                try:
                    start_index = corpus.find(document, search_cursor)
                    if start_index == -1:
                        start_index = corpus.find(document)
                    if start_index == -1:
                        flexible_match = find_target_with_flexible_whitespace(corpus, document, search_cursor)
                        if flexible_match is not None:
                            start_index, end_index = flexible_match
                        else:
                            alnum_match = find_target_with_alnum_normalization(
                                corpus,
                                document,
                                corpus_alnum,
                                corpus_alnum_map,
                                start_cursor=search_cursor,
                            )
                            if alnum_match is not None:
                                start_index, end_index = alnum_match
                            else:
                                lexical_match = find_target_with_lexical_tokens(
                                    corpus,
                                    document,
                                    start_cursor=search_cursor,
                                )
                                if lexical_match is not None:
                                    start_index, end_index = lexical_match
                                else:
                                    result = rigorous_document_search(corpus, document)
                                    if result is None:
                                        raise ValueError("Could not locate chunk in corpus")
                                    _, start_index, end_index = result
                    else:
                        end_index = start_index + len(document)
                    search_cursor = max(search_cursor, end_index)
                except Exception as error:
                    skipped_chunks += 1
                    preview = " ".join(document.split())[:100]
                    print(
                        f"Warning: skipping unmatched chunk in {corpus_id}: "
                        f"{error}; chunk={preview!r}"
                    )
                    continue
                current_documents_found.append(document)
                current_metadatas.append({"start_index": start_index, "end_index": end_index, "corpus_id": corpus_id})
            documents.extend(current_documents_found)
            metadatas.extend(current_metadatas)
        if skipped_chunks > 0:
            print(f"Warning: skipped {skipped_chunks} unmatched chunk(s) during metadata alignment.")
        return documents, metadatas

    def _relevant_chunk_counts(self, chunk_metadatas):
        """Count, per question, how many corpus chunks overlap any groundtruth reference.

        Used only to size the 'minimal' retrieval profile (retrieve=-1): each question
        pulls back exactly as many chunks as there are relevant ones in its document.
        """
        highlighted_chunks_count = []

        for _, row in self.questions_df.iterrows():
            references = row['references']
            corpus_id = row['corpus_id']
            ref_ranges = [(int(x['start_index']), int(x['end_index'])) for x in references]

            highlighted_chunk_count = 0
            for metadata in chunk_metadatas:
                chunk_start, chunk_end, chunk_corpus_id = metadata['start_index'], metadata['end_index'], metadata['corpus_id']
                if chunk_corpus_id != corpus_id:
                    continue
                if any(intersect_two_ranges((chunk_start, chunk_end), ref_range) is not None for ref_range in ref_ranges):
                    highlighted_chunk_count += 1

            highlighted_chunks_count.append(highlighted_chunk_count)

        return highlighted_chunks_count

    def _build_question_reports(
        self,
        question_metadatas,
        highlighted_chunks_count,
        question_documents=None,
    ):
        """Build the retrieved-chunks report for each question (no scoring here).

        Binary TP/FP/TN/FN, precision and recall are computed separately in
        ``_binary_confusion_scores`` and merged into ``metrics`` by the caller.
        """
        question_metrics = []
        rows = zip(self.questions_df.iterrows(), highlighted_chunks_count, question_metadatas)
        for question_index, ((index, row), highlighted_chunk_count, metadatas) in enumerate(rows):
            question = row['question']
            references = row['references']
            corpus_id = row['corpus_id']

            k = max(0, int(highlighted_chunk_count))
            top_k_metadatas = metadatas[:k]
            documents = question_documents[question_index][:k] if question_documents is not None else [None] * len(top_k_metadatas)

            retrieved_chunks = []
            for rank_index, (metadata, document) in enumerate(zip(top_k_metadatas, documents), start=1):
                chunk_start, chunk_end, chunk_corpus_id = metadata['start_index'], metadata['end_index'], metadata['corpus_id']

                chunk_intersections = []
                if chunk_corpus_id == corpus_id:
                    for ref_obj in references:
                        ref_start, ref_end = int(ref_obj['start_index']), int(ref_obj['end_index'])
                        intersection = intersect_two_ranges((chunk_start, chunk_end), (ref_start, ref_end))
                        if intersection is not None:
                            chunk_intersections.append(intersection)

                retrieved_chunks.append({
                    "rank": rank_index,
                    "start_index": int(chunk_start),
                    "end_index": int(chunk_end),
                    "text": document,
                    "relevant": bool(chunk_intersections),
                    "intersections": [
                        {"start_index": int(start), "end_index": int(end)}
                        for start, end in chunk_intersections
                    ],
                })

            question_metrics.append({
                "question_index": int(index),
                "question_key": row.get("question_key", ""),
                "question": question,
                "corpus_id": corpus_id,
                "groundtruth": references,
                "retrieved_chunks": retrieved_chunks,
                "metrics": {},
            })

        return {"question_metrics": question_metrics}

    def _binary_confusion_scores(self, chunk_metadatas, question_metadatas, highlighted_chunks_count):
        """Per question: classify every chunk of its document as retrieved/relevant to derive
        TP/FP/TN/FN at the question's own K.

        K is adaptive: ``highlighted_chunks_count`` holds N (the number of chunks that actually
        overlap the groundtruth) for that question, so precision@K / recall@K are measured
        against exactly as many retrieved chunks as there are correct ones.
        """
        chunks_by_corpus: dict[str, list[dict]] = {}
        for metadata in chunk_metadatas:
            chunks_by_corpus.setdefault(metadata['corpus_id'], []).append(metadata)

        question_confusions = []

        rows = zip(self.questions_df.iterrows(), highlighted_chunks_count, question_metadatas)
        for (index, row), highlighted_chunk_count, metadatas in rows:
            references = row['references']
            corpus_id = row['corpus_id']
            ref_ranges = [(int(x['start_index']), int(x['end_index'])) for x in references]

            k = max(0, int(highlighted_chunk_count))
            top_k_metadatas = [m for m in metadatas[:k] if m['corpus_id'] == corpus_id]
            retrieved_keys = {(m['start_index'], m['end_index']) for m in top_k_metadatas}

            tp = fp = tn = fn = 0
            for chunk in chunks_by_corpus.get(corpus_id, []):
                key = (chunk['start_index'], chunk['end_index'])
                is_relevant = any(
                    intersect_two_ranges(key, ref_range) is not None for ref_range in ref_ranges
                )
                is_retrieved = key in retrieved_keys
                if is_retrieved and is_relevant:
                    tp += 1
                elif is_retrieved and not is_relevant:
                    fp += 1
                elif is_relevant:
                    fn += 1
                else:
                    tn += 1

            confusion = {
                "corpus_id": corpus_id,
                "k": k,
                "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            }
            question_confusions.append(confusion)

        return question_confusions

    @staticmethod
    def _binary_metrics_from_confusion(confusion: dict[str, int]) -> dict[str, float]:
        """Calculate Precision@K and Recall@K from one question's confusion counts."""
        tp, fp, tn, fn = confusion["tp"], confusion["fp"], confusion["tn"], confusion["fn"]
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        return {
            "k": confusion.get("k", tp + fp),
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "precision_at_k": precision,
            "recall_at_k": recall,
        }

    @staticmethod
    def _macro_average(metric_dicts: list[dict[str, float]]) -> dict[str, float]:
        """Unweighted mean of each rate over the given per-question (or per-document) metrics."""
        if not metric_dicts:
            return {name: 0.0 for name in MACRO_METRIC_NAMES}
        return {
            name: float(np.mean([entry[name] for entry in metric_dicts]))
            for name in MACRO_METRIC_NAMES
        }

    def _chunker_to_collection(self, chunker, embedding_function, chroma_db_path:str = None, collection_name:str = None):
        collection = None


        if chroma_db_path is not None:
            try:
                chunk_client = chromadb.PersistentClient(path=chroma_db_path)
                collection = chunk_client.create_collection(collection_name, 
                                                            embedding_function=embedding_function, 
                                                            metadata={"hnsw:search_ef":50},
                                                             configuration={"hnsw": {
                                                                "space": "cosine" , 
                                                                
                                                                }})
                print("Created collection: ", collection_name)
            except Exception as e:
                print("Failed to create collection: ", e)
                pass
                # This shouldn't throw but for whatever reason, if it does we will default to below.

        collection_name = "auto_chunk"
        if collection is None:
            try:
                self.chroma_client.delete_collection(collection_name)
            except Exception:
                pass
            collection = self.chroma_client.create_collection(collection_name, 
                                                              embedding_function=embedding_function, 
                                                              metadata={"hnsw:search_ef":50}, 
                                                              configuration={"hnsw": {
                                                                  "space": "cosine"}})

        docs, metas = self._get_chunks_and_metadata(chunker)

        BATCH_SIZE = 500
        for i in range(0, len(docs), BATCH_SIZE):
            batch_docs = docs[i:i+BATCH_SIZE]
            batch_metas = metas[i:i+BATCH_SIZE]
            batch_ids = [str(i) for i in range(i, i+len(batch_docs))]
            collection.add(
                documents=batch_docs,
                metadatas=batch_metas,
                ids=batch_ids
            )

            # print("Documents: ", batch_docs)
            # print("Metadatas: ", batch_metas)

        return collection
    
    def _convert_question_references_to_json(self):
        def safe_json_loads(row):
            try:
                return json.loads(row)
            except (TypeError, json.JSONDecodeError):
                pass

        self.questions_df['references'] = self.questions_df['references'].apply(safe_json_loads)












    def run(self, chunker, embedding_function=None, retrieve:int = 5, db_to_save_chunks: str = None):
        """
        This function runs the evaluation over the provided chunker.

        Parameters:
        chunker: The chunker to evaluate.
        embedding_function: The embedding function to use for calculating the nearest neighbours during the retrieval step. If not provided, the default OpenAI embedding function is used.
        retrieve: The number of chunks to retrieve per question. If set to -1, the function will retrieve the minimum number of chunks that contain excerpts for a given query. This is typically around 1 to 3 but can vary by question. By setting a specific value for retrieve, this number is fixed for all queries.
        """
        if embedding_function is None:
            embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
               model_name = MODEL_NAME
               
            )

        collection = None
        if db_to_save_chunks is not None:
            chunk_size = chunker._chunk_size if hasattr(chunker, '_chunk_size') else "0"
            chunk_overlap = chunker._chunk_overlap if hasattr(chunker, '_chunk_overlap') else "0"
            embedding_function_name = embedding_function.__class__.__name__
            if embedding_function_name == "SentenceTransformerEmbeddingFunction":
                embedding_function_name = "SentEmbFunc"
            collection_name = embedding_function_name + '_' + chunker.__class__.__name__ + '_' + str(int(chunk_size)) + '_' + str(int(chunk_overlap))
            try:
                chunk_client = chromadb.PersistentClient(path=db_to_save_chunks)
                collection = chunk_client.get_collection(collection_name, embedding_function=embedding_function)
            except Exception:
                # Get collection throws if the collection does not exist. We will create it below if it does not exist.
                collection = self._chunker_to_collection(chunker, embedding_function, chroma_db_path=db_to_save_chunks, collection_name=collection_name)

        if collection is None:
            collection = self._chunker_to_collection(chunker, embedding_function)

        question_collection = None

        if self.is_general:
            with resources.as_file(resources.files('chunking_evaluation.evaluation_framework') / 'general_evaluation_data') as general_benchmark_path:
                questions_client = chromadb.PersistentClient(path=os.path.join(general_benchmark_path, 'questions_db'))
                
                if embedding_function.__class__.__name__ == "SentenceTransformerEmbeddingFunction":
                    try:
                        question_collection = questions_client.get_collection("auto_questions_sentence_transformer", embedding_function=embedding_function)
                    except Exception as e:
                        print("Warning: Failed to use the frozen embeddings originally used in the paper. As a result, this package will now generate a new set of embeddings. The change should be minimal and only come from the noise floor of SentenceTransformer's embedding function. The error: ", e)
        
        if not self.is_general or question_collection is None:
            # if self.is_general:
            #     print("FAILED TO LOAD GENERAL EVALUATION")
            try:
                self.chroma_client.delete_collection("auto_questions")
            except Exception:
                pass
            question_collection = self.chroma_client.create_collection("auto_questions", 
                                                                       embedding_function=embedding_function, 
                                                                       metadata={"hnsw:search_ef":50},
                                                                       configuration={"hnsw": {
                                                                           "space": "cosine" }} )
            question_collection.add(
                documents=self.questions_df['question'].tolist(),
                metadatas=[{"corpus_id": x} for x in self.questions_df['corpus_id'].tolist()],
                ids=[str(i) for i in self.questions_df.index]
            )
        
        question_db = question_collection.get(include=['embeddings'])

        # Convert ids to integers for sorting
        question_db['ids'] = [int(id) for id in question_db['ids']]
        # Sort both ids and embeddings based on ids
        _, sorted_embeddings = zip(*sorted(zip(question_db['ids'], question_db['embeddings'])))

        # Sort questions_df in ascending order
        self.questions_df = self.questions_df.sort_index()

        highlighted_chunks_count = self._relevant_chunk_counts(collection.get()['metadatas'])

        if retrieve == -1:
            # Adaptive K: each question uses K = N (its own number of relevant chunks),
            # so precision@K is a fair per-question score instead of a fixed top-5 cut-off.
            highlighted_chunks_count = [max(1, int(count)) for count in highlighted_chunks_count]
            maximum_n = min(20, max(highlighted_chunks_count))
        else:
            highlighted_chunks_count = [retrieve] * len(highlighted_chunks_count)
            maximum_n = retrieve

        # Chroma requires n_results >= 1.
        maximum_n = max(1, int(maximum_n))

        # arr_bytes = np.array(list(sorted_embeddings)).tobytes()
        # print("Hash: ", hashlib.md5(arr_bytes).hexdigest())

        # Retrieve only within each question's source document. Questions such as
        # "Wie heißt der Patient?" are intentionally generic, so searching across
        # every corpus would rank chunks from unrelated documents above the answer.
        retrieval_metadatas = []
        retrieval_documents = []
        for (_, row), query_embedding in zip(self.questions_df.iterrows(), sorted_embeddings):
            retrieval = collection.query(
                query_embeddings=[query_embedding],
                n_results=maximum_n,
                where={"corpus_id": row["corpus_id"]},
            )
            retrieval_metadatas.append(retrieval["metadatas"][0])
            retrieval_documents.append(retrieval["documents"][0])

        report_dict = self._build_question_reports(
            retrieval_metadatas,
            highlighted_chunks_count,
            question_documents=retrieval_documents,
        )

        question_confusions = self._binary_confusion_scores(
            collection.get()['metadatas'],
            retrieval_metadatas,
            highlighted_chunks_count,
        )
        for question_entry, confusion in zip(report_dict["question_metrics"], question_confusions):
            question_entry["metrics"] = self._binary_metrics_from_confusion(confusion)

        # Level 1 -> 2: a document's score is the unweighted mean over its own questions.
        questions_by_document: dict[str, list[dict]] = {}
        for question_entry in report_dict["question_metrics"]:
            questions_by_document.setdefault(question_entry["corpus_id"], []).append(
                question_entry["metrics"]
            )

        per_document = {}
        for corpus_id, question_metric_dicts in questions_by_document.items():
            document_metrics = self._macro_average(question_metric_dicts)
            document_metrics["num_questions"] = len(question_metric_dicts)
            per_document[corpus_id] = document_metrics

        # Level 2 -> 3: the global score is mean/std over documents, so every file counts equally.
        document_metrics_list = list(per_document.values())
        global_metrics = {}
        for name in MACRO_METRIC_NAMES:
            values = [entry[name] for entry in document_metrics_list]
            global_metrics[f"{name}_global_mean"] = float(np.mean(values)) if values else 0.0
            global_metrics[f"{name}_global_std"] = float(np.std(values)) if values else 0.0

        chunk_collection = collection.get(include=["documents", "metadatas"])
        chunks = [
            {"text": document, **metadata}
            for document, metadata in zip(
                chunk_collection["documents"], chunk_collection["metadatas"]
            )
        ]

        return {
            "per_document": per_document,
            "global": global_metrics,
            "question_metrics": report_dict["question_metrics"],
            "chunks": chunks,
        }