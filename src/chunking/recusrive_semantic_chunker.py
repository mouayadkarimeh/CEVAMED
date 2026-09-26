from __future__ import annotations

import re
import sys
import numpy as np
from pathlib import Path
from typing import Any, List, Optional
from matplotlib import pyplot as plt
from transformers import AutoTokenizer
from chunking.base_chunker import TextSplitter
from chunking.recursive_token_chunker import RecursiveTokenChunker

#MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
MODEL_NAME = "deutsche-telekom/gbert-large-paraphrase-cosine"
#MODEL_NAME = "codefuse-ai/F2LLM-v2-0.6B"


# Interface aus der Vaterklasse laden
'''
try:
    from langchain_text_splitters import TextSplitter
except ImportError:
    from langchain_text_splitters import TextSplitter
'''



def results_plot(distances: list[float], breakpoint_distance_threshold: float) -> None:
    """
    Plottet die semantischen Distanzen und markiert die Chunks basierend auf dem optimalen Threshold.
    """
    if not distances:
        print("Keine Distanzen zum Plotten.")
        return

    y_upper_bound = .6 
    plt.plot(distances, marker="o", linestyle="-", color="b", label="Kosinus-Distanz")
    plt.ylim(0, y_upper_bound)
    plt.xlim(0, len(distances))
    
    plt.axhline(y=breakpoint_distance_threshold, color="r", linestyle="--", label="Dynamischer Threshold")

    num_distances_above_threshold = sum(x > breakpoint_distance_threshold for x in distances)
    plt.text(x=(len(distances) * 0.01), y=y_upper_bound / 50, s=f"{num_distances_above_threshold + 1} Chunks", fontsize=11, fontweight='bold')

    indices_above_thresh = [i for i, x in enumerate(distances) if x > breakpoint_distance_threshold]
    colors = ["b", "g", "r", "c", "m", "y", "k"]
    
    for i, breakpoint_index in enumerate(indices_above_thresh):
        start_index = 0 if i == 0 else indices_above_thresh[i - 1] + 1
        end_index = breakpoint_index + 1
        plt.axvspan(start_index, end_index, facecolor=colors[i % len(colors)], alpha=0.15)
        plt.text(
            x=np.average([start_index, end_index]),
            y=breakpoint_distance_threshold + (y_upper_bound / 20),
            s=f"Chunk #{i}",
            horizontalalignment="center",
            rotation="vertical",
        )

    if indices_above_thresh:
        last_breakpoint = indices_above_thresh[-1] + 1
        if last_breakpoint < len(distances):
            plt.axvspan(
                last_breakpoint,
                len(distances),
                facecolor=colors[len(indices_above_thresh) % len(colors)],
                alpha=0.15,
            )
            plt.text(
                x=np.average([last_breakpoint, len(distances)]),
                y=breakpoint_distance_threshold + (y_upper_bound / 20),
                s=f"Chunk #{len(indices_above_thresh)}",
                horizontalalignment="center",
                rotation="vertical",
            )
    else:
        plt.axvspan(0, len(distances), facecolor="b", alpha=0.15)

    plt.title("Rekursiv semantisches Chunking" , fontsize=15, pad=22)
    plt.suptitle("Analysierte Datei: Albers.txt", fontsize=12, x=0.5, y=0.92, color="green")
    plt.xlabel("Satz-Index")
    plt.ylabel("Kosinus-Distanze zwichen aufeinanderfolgenden Sätzen")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


class RecursiveSemanticChunker(TextSplitter):
    
    def __init__(
        self, 
        avg_chunk_size: int = 128, 
        min_chunk_size: int = 50, 
        embedding_function: Optional[Any] = None, 
        length_function: Optional[Any] = None,
        **kwargs: Any
    ):
        """
        Initialisiert den Chunker. Unterstützt direkte Übergabe oder 
        die Factory-Methode 'from_huggingface_tokenizer' der Vaterklasse.
        """
        # Falls von der Vaterklasse via from_huggingface_tokenizer aufgerufen,
        # ist die length_function bereits in kwargs definiert.
        self.length_function = length_function or kwargs.get("length_function")
        
        # Fallback falls der Chunker direkt ohne length_function instanziiert wurde
        if self.length_function is None:
            tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
           

            self.length_function = lambda text: len(tokenizer.tokenize(text))

        # Interner Token-Splitter für feine Vorspaltung
        self.splitter = RecursiveTokenChunker(
            chunk_size=min_chunk_size,
            chunk_overlap=0,
            length_function=self.length_function
        )
        
        self.avg_chunk_size = avg_chunk_size

        if embedding_function is None:
            from sentence_transformers import SentenceTransformer
            self.embedding_function = SentenceTransformer(MODEL_NAME)
            
        else:
            self.embedding_function = embedding_function

        # Reiche verbleibende Argumente und die length_function an die Basisklasse we iter
        kwargs["length_function"] = self.length_function
        super().__init__(**kwargs)

    def combine_sentences(self, sentences: List[dict], buffer_size: int = 1) -> List[dict]:
        for i in range(len(sentences)):
            combined_sentence = ''
            for j in range(i - buffer_size, i):
                if j >= 0:
                    combined_sentence += sentences[j]['sentence'] + ' '

            combined_sentence += sentences[i]['sentence']

            for j in range(i + 1, i + 1 + buffer_size):
                if j < len(sentences):
                    combined_sentence += ' ' + sentences[j]['sentence']

            sentences[i]['combined_sentence'] = combined_sentence
        return sentences

    def calculate_cosine_distances(self, sentences: List[dict]) -> tuple[List[float], List[dict]]:
        if len(sentences) <= 1:
            return [], sentences

        BATCH_SIZE = 500
        embedding_matrix = None
        
        for i in range(0, len(sentences), BATCH_SIZE):
            batch_sentences = sentences[i:i+BATCH_SIZE]
            batch_texts = [sentence['combined_sentence'] for sentence in batch_sentences]
            
            if hasattr(self.embedding_function, "encode"):
                embeddings = self.embedding_function.encode(batch_texts, show_progress_bar=False)
            else:
                embeddings = self.embedding_function(batch_texts)
                
            batch_embedding_matrix = np.array(embeddings)

            if embedding_matrix is None:
                embedding_matrix = batch_embedding_matrix
            else:
                embedding_matrix = np.concatenate((embedding_matrix, batch_embedding_matrix), axis=0)

        norms = np.linalg.norm(embedding_matrix, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1e-9, norms)
        embedding_matrix = embedding_matrix / norms

        similarity_matrix = np.dot(embedding_matrix, embedding_matrix.T)
        
        distances = []
        for i in range(len(sentences) - 1):
            similarity = similarity_matrix[i, i + 1]
            distance = 1.0 - similarity
            distances.append(float(distance))
            sentences[i]['distance_to_next'] = distance

        sentences[-1]['distance_to_next'] = None
        return distances, sentences

    def _find_optimal_threshold(self, distances: List[float], total_tokens: int) -> float:
        number_of_cuts = total_tokens // self.avg_chunk_size
        lower_limit = 0.0
        upper_limit = 1.0
        #threshold = 0.5
        distances_np = np.array(distances)

        while upper_limit - lower_limit > 1e-6:
            threshold = (upper_limit + lower_limit) / 2.0
            num_points_above_threshold = np.sum(distances_np > threshold)
            
            if num_points_above_threshold > number_of_cuts:
                lower_limit = threshold
            else:
                upper_limit = threshold
        return threshold

    def split_text(self, text: str) -> List[str]:
        sentences_strips = self.splitter.split_text(text)
        if not sentences_strips:
            return []

        sentences = [{'sentence': x, 'index': i} for i, x in enumerate(sentences_strips)]
        sentences = self.combine_sentences(sentences, buffer_size=1)
        distances, sentences = self.calculate_cosine_distances(sentences)
        if not distances:
            return [text]

        total_tokens = sum(self.length_function(s['sentence']) for s in sentences)
        threshold = self._find_optimal_threshold(distances, total_tokens)

        indices_above_thresh = [i for i, x in enumerate(distances) if x > threshold] 
        
        start_index = 0
        chunks = []

        for index in indices_above_thresh:
            end_index = index
            group = sentences[start_index:end_index + 1]
            combined_text = ' '.join([d['sentence'] for d in group])
            chunks.append(combined_text)
            start_index = index + 1

        if start_index < len(sentences):
            combined_text = ' '.join([d['sentence'] for d in sentences[start_index:]])
            chunks.append(combined_text)

        return chunks


if __name__ == "__main__":
    from transformers import AutoTokenizer

    
    hf_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    


    with open("data/row/Albers.txt", "r", encoding="utf-8") as f:
        text = f.read()
        
        # 2. NUTZUNG DER VATERKLASSEN-METHODE:
        # Erstellt das Objekt dynamisch mit der Längenfunktion des Tokenizers
        semantic_chunker = RecursiveSemanticChunker.from_huggingface_tokenizer(
            hf_tokenizer,
            avg_chunk_size=200,
            min_chunk_size=50
        )
        
        # 3. Teilschritte für den Plot ausführen
        sentences_strips = semantic_chunker.splitter.split_text(text)
        splitted = [{'sentence': x, 'index': i} for i, x in enumerate(sentences_strips)]
        combined_sen = semantic_chunker.combine_sentences(splitted, buffer_size=1)
        distances, sentences = semantic_chunker.calculate_cosine_distances(combined_sen)
        
        total_tokens = sum(semantic_chunker.length_function(s['sentence']) for s in sentences)
        print(f"Total tokens: {total_tokens}")
        dynamic_threshold = semantic_chunker._find_optimal_threshold(distances, total_tokens)
        results_plot(distances, dynamic_threshold)

         # 4. Splitten & echten Token-Count im Terminal ausgeben
        chunks = semantic_chunker.split_text(text)
        print(f"\n--- Auswertung mit hf_tokenizer ---")
        for i, chunk in enumerate(chunks, 1):
            # Hier greift nun die saubere Methode aus der Vaterklasse zum Zählen der Tokens!
            token_count = semantic_chunker.length_function(chunk)
            print(f"Chunk {i}: len_tokens={token_count} | (len_zeichen={len(chunk)})")
        
        
