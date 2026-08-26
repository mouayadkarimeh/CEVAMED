"""Semantic chunking based on sentence embeddings and distance breakpoints."""

from __future__ import annotations

import sys
import re
from typing import Dict, List
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
from langchain_text_splitters import (
    TextSplitter,
)

sys.path.append(str(Path(__file__).resolve().parents[2]))



DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_EPSILON = 1e-6
BREAKPOINT_PERCENTILE_THRESHOLD = 90 # Default percentile for determining breakpoints

def results_plot(distances: list[float], BREAKPOINT_PERCENTILE_THRESHOLD ) -> None:
    if not distances:
        print("Keine Distanzen zum Plotten.")
        return

    
    y_upper_bound = .6 # Set a fixed upper bound for better visualization  
    plt.plot(distances)
    plt.ylim(0, y_upper_bound)
    plt.xlim(0, len(distances))
    breakpoint_distance_threshold = np.percentile(distances, BREAKPOINT_PERCENTILE_THRESHOLD)
    plt.axhline(y = breakpoint_distance_threshold , color="r", linestyle="-")

    num_distances_above_threshold = sum(x > breakpoint_distance_threshold for x in distances)
    plt.text(x=(len(distances) * 0.01), y=y_upper_bound / 50, s=f"{num_distances_above_threshold + 1} Chunks")

    indices_above_thresh = [i for i, x in enumerate(distances) if x > breakpoint_distance_threshold]
    colors = ["b", "g", "r", "c", "m", "y", "k"]
    for i, breakpoint_index in enumerate(indices_above_thresh):
        start_index = 0 if i == 0 else indices_above_thresh[i - 1]
        end_index = breakpoint_index
        plt.axvspan(start_index, end_index, facecolor=colors[i % len(colors)], alpha=0.25)
        plt.text(
            x=np.average([start_index, end_index]),
            y=breakpoint_distance_threshold + (y_upper_bound / 20),
            s=f"Chunk #{i}",
            horizontalalignment="center",
            rotation="vertical",
        )

    if indices_above_thresh:
        last_breakpoint = indices_above_thresh[-1]
        if last_breakpoint < len(distances):
            plt.axvspan(
                last_breakpoint,
                len(distances),
                facecolor=colors[len(indices_above_thresh) % len(colors)],
                alpha=0.25,
            )
            plt.text(
                x=np.average([last_breakpoint, len(distances)]),
                y=breakpoint_distance_threshold + (y_upper_bound / 20),
                s=f"Chunk #{len(indices_above_thresh)}",
                rotation="vertical",
            )

    plt.title("Chunks Based on Embedding Breakpoints")
    plt.xlabel("Sentence index")
    plt.ylabel("Cosine distance between sequential sentences")
    plt.show()



def plot_embedding_breakpoints(
    distances: list[float],
    breakpoint_percentile_threshold: float = 95,
    y_upper_bound: float = 0.2,
    colors: list[str] = ['b', 'g', 'r', 'c', 'm', 'y', 'k'],
) -> None:
    """
    Visualisiert Text-Chunks basierend auf Embedding-Abstandswerten (Cosine-Distances).

    Args:
        distances: Liste der Cosine-Distances zwischen aufeinanderfolgenden Sätzen.
        breakpoint_percentile_threshold: Percentil für die Bestimmung des Breakpoint-Schwellenwerts (Standard: 95).
        y_upper_bound: Obere Grenze für die y-Achse (Standard: 0.2).
        colors: Liste der Farben für die Chunks (Standard: ['b', 'g', 'r', 'c', 'm', 'y', 'k']).

    Returns:
        None (zeigt den Plot direkt an).
    """
    if not distances:
        print("Fehler: Keine Distanzen zum Plotten.")
        return

    # Breakpoint-Schwellenwert berechnen
    breakpoint_distance_threshold = np.percentile(distances, breakpoint_percentile_threshold)

    # Plot erstellen
    plt.figure(figsize=(12, 6))
    plt.plot(distances, linewidth=1.5)

    # Achsengrenzen setzen
    plt.ylim(0, y_upper_bound)
    plt.xlim(0, len(distances))

    # Breakpoint-Linie hinzufügen
    plt.axhline(
        y=breakpoint_distance_threshold,
        color='r',
        linestyle='-',
        linewidth=1.5,
        label=f'Breakpoint-Schwelle ({breakpoint_percentile_threshold}%)'
    )

    # Anzahl der Chunks berechnen und anzeigen
    num_distances_above_threshold = sum(x > breakpoint_distance_threshold for x in distances)
    plt.text(
        x=len(distances) * 0.01,
        y=y_upper_bound / 50,
        s=f"{num_distances_above_threshold + 1} Chunks",
        fontsize=12,
        bbox=dict(facecolor='white', alpha=0.8, edgecolor='none')
    )

    # Breakpoint-Indizes berechnen
    indices_above_thresh = [i for i, x in enumerate(distances) if x > breakpoint_distance_threshold]

    # Chunks visualisieren
    for i, breakpoint_index in enumerate(indices_above_thresh):
        start_index = 0 if i == 0 else indices_above_thresh[i - 1]
        end_index = breakpoint_index if i < len(indices_above_thresh) - 1 else len(distances)

        # Farbige Bereiche für die Chunks
        plt.axvspan(
            start_index,
            end_index,
            facecolor=colors[i % len(colors)],
            alpha=0.25,
            label=f'Chunk #{i + 1}' if i == 0 else None  # Nur erstes Label anzeigen
        )

        # Chunk-Nummerierung (beginnt bei 1)
        plt.text(
            x=np.average([start_index, end_index]),
            y=breakpoint_distance_threshold + (y_upper_bound / 20),
            s=f"Chunk #{i + 1}",
            horizontalalignment='center',
            rotation='vertical',
            fontsize=10,
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='none')
        )

    # Letzten Chunk behandeln
    if indices_above_thresh:
        last_breakpoint = indices_above_thresh[-1]
        if last_breakpoint < len(distances):
            plt.axvspan(
                last_breakpoint,
                len(distances),
                facecolor=colors[len(indices_above_thresh) % len(colors)],
                alpha=0.25
            )
            plt.text(
                x=np.average([last_breakpoint, len(distances)]),
                y=breakpoint_distance_threshold + (y_upper_bound / 20),
                s=f"Chunk #{len(indices_above_thresh) + 1}",
                rotation='vertical',
                fontsize=10,
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='none')
            )

    # Plot anpassen
    plt.title(
        "Text-Chunks basierend auf Embedding-Breakpoints",
        fontsize=14,
        pad=20
    )
    plt.xlabel(
        "Index der Sätze im Essay (Satzposition)",
        fontsize=12
    )
    plt.ylabel(
        "Cosine-Distance zwischen aufeinanderfolgenden Sätzen",
        fontsize=12
    )
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='upper right', fontsize=10)
    plt.tight_layout()
    plt.show()



class KamradtSemanticChunker(TextSplitter):
    """Semantic chunker using sentence embeddings and distance breakpoints.
    
    CRITICAL: This implementation extracts EXACT SUBSTRINGS from the original text
    without any transformation (no lowercase, no unicode normalization, no whitespace
    changes). This ensures perfect alignment with ground truth for evaluation.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        device: str = "cpu",
        epsilon: float = DEFAULT_EPSILON,
        embedding_batch_size: int = 500,
    ) -> None:
        super().__init__(chunk_size=256, chunk_overlap=64, length_function=len)
        self.model_name = model_name
        self.device = device
        self.epsilon = epsilon
        self.embedding_batch_size = embedding_batch_size
        self._embedding_model = SentenceTransformer(self.model_name, device=self.device)
       

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._allowed_special = set()
        self._disallowed_special = "all"



    def _split_text_regular_expresion(self, text: str) -> List[dict[str,any]]:
       # Splitting the essay on '.', '?', and '!'
        single_sentences_list = re.split(r'(?<=[.?!])\s+', text)
        print (f"{len(single_sentences_list)} senteneces were found")
        # change list of sentences in dict format to add data easy later 
        sentences_dict = [{"sentence":x, "index":i} for i , x in enumerate(single_sentences_list)  ]
        return sentences_dict

    
    def _split_into_token_units(self, text: str) -> List[Dict[str, object]]:
        """Create position-aware token units like the upstream recursive splitter."""
        token_ids = self._tokenizer.encode(
            text,
            allowed_special=self._allowed_special,
            disallowed_special=self._disallowed_special,
        )
        if not token_ids:
            return []

        encoded = bytearray()
        boundaries: List[tuple[int, int]] = []
        for token_index, token_id in enumerate(token_ids, start=1):
            encoded.extend(self._tokenizer.decode_single_token_bytes(token_id))
            try:
                char_end = len(encoded.decode("utf-8"))
            except UnicodeDecodeError:
                continue
            boundaries.append((token_index, char_end))

        units: List[Dict[str, object]] = []
        start_token = 0
        start_char = 0
        boundary_index = 0
        while start_token < len(token_ids):
            target_token = min(start_token + self.min_chunk_size, len(token_ids))
            while boundary_index < len(boundaries) and boundaries[boundary_index][0] < target_token:
                boundary_index += 1
            if boundary_index >= len(boundaries):
                end_token, end_char = len(token_ids), len(text)
            else:
                end_token, end_char = boundaries[boundary_index]

            units.append({
                "sentence": text[start_char:end_char],
                "start": start_char,
                "end": end_char,
                "index": len(units),
            })
            start_token = end_token
            start_char = end_char
            boundary_index += 1

        return units

    def _count_tokens(self, text: str) -> int:
        """Count tokens with the same tokenizer used for token-based boundaries."""
        return len(
            self._tokenizer.encode(
                text,
                allowed_special=self._allowed_special,
                disallowed_special=self._disallowed_special,
            )
        )

    def _token_end_positions(self, text: str) -> List[int]:
        """Return character offsets after each complete UTF-8 token."""
        token_ids = self._tokenizer.encode(
            text,
            allowed_special=self._allowed_special,
            disallowed_special=self._disallowed_special,
        )
        encoded = bytearray()
        positions: List[int] = []

        for token_id in token_ids:
            encoded.extend(self._tokenizer.decode_single_token_bytes(token_id))
            try:
                position = len(encoded.decode("utf-8"))
            except UnicodeDecodeError:
                # A token can end in the middle of a multibyte character.
                continue
            positions.append(position)

        if positions and positions[-1] != len(text):
            positions.append(len(text))
        elif not positions and text:
            positions.append(len(text))
        return positions

    def _split_group_by_tokens(
        self,
        text: str,
        start: int,
        end: int,
        token_budget: int,
    ) -> List[str]:
        """Split an oversized semantic group at exact token boundaries."""
        group_text = text[start:end]
        token_end_positions = self._token_end_positions(group_text)
        if not token_end_positions:
            return [group_text] if group_text.strip() else []

        chunks: List[str] = []
        chunk_start = 0
        token_start = 0
        token_count = len(token_end_positions)

        while token_start < token_count:
            token_end = min(token_start + token_budget, token_count)
            char_end = token_end_positions[token_end - 1]
            chunk = group_text[chunk_start:char_end]
            if chunk.strip():
                chunks.append(chunk)
            chunk_start = char_end
            token_start = token_end

        return chunks

    def _append_group(
        self,
        text: str,
        group: List[Dict[str, object]],
        token_budget: int,
        chunks: List[str],
    ) -> None:
        """Append a semantic group, enforcing the token budget exactly."""
        if not group:
            return

        group_start = int(group[0]["start"])
        group_end = int(group[-1]["end"])
        group_text = text[group_start:group_end]

        if self._count_tokens(group_text) <= token_budget:
            if group_text.strip():
                chunks.append(group_text)
            return

        chunks.extend(self._split_group_by_tokens(text, group_start, group_end, token_budget))

    @staticmethod
    def combine_sentences(sentences: List[Dict[str, object]], buffer_size: int = 3) -> List[Dict[str, object]]:
        """Add combined sentence context (with buffer) for embedding calculation."""
        for i in range(len(sentences)):
            combined_sentence = ""

            # Add preceding sentences (buffer)
            for j in range(i - buffer_size, i):
                if j >= 0:
                    combined_sentence += str(sentences[j]["sentence"]) + " "

            # Add center sentence
            combined_sentence += str(sentences[i]["sentence"])

            # Add following sentences (buffer)
            for j in range(i + 1, i + 1 + buffer_size):
                if j < len(sentences):
                    combined_sentence += " " + str(sentences[j]["sentence"])

            sentences[i]["combined_sentence"] = combined_sentence

        return sentences

    def _embed_texts(self, texts: List[str]) -> np.ndarray:
        """Compute embeddings for a list of texts using the model."""
        embeddings = self._embedding_model.encode(
            texts,
            batch_size=self.embedding_batch_size,
            convert_to_numpy=True,
            normalize_embeddings=False,
            show_progress_bar=False,
        )
        return np.asarray(embeddings)
    
    def calculate_sentence_embeddings(self, sentences: List[Dict[str, object]]) -> List[Dict[str, object]]:
        """Calculate embeddings for combined sentences and store them in the sentence dicts."""
        if not sentences:
            return sentences

        embeddings = self._embedding_model.encode([x['combined_sentence'] for x in sentences])
        for i, sentence in enumerate(sentences):
            sentence['combined_sentence_embedding'] = embeddings[i]

        return sentences
        

    
    def calculate_cosine_distances(self, sentences):
        distances = []
        for i in range(len(sentences) - 1):
            embedding_current = sentences[i]['combined_sentence_embedding']
            embedding_next = sentences[i + 1]['combined_sentence_embedding']
            
            # Calculate cosine similarity
            similarity = cosine_similarity([embedding_current], [embedding_next])[0][0]
            
            # Convert to cosine distance
            distance = 1 - similarity

            # Append cosine distance to the list
            distances.append(distance)

            # Store distance in the dictionary
            sentences[i]['distance_to_next'] = distance

        # Optionally handle the last sentence
        # sentences[-1]['distance_to_next'] = None  # or a default value

        return distances, sentences

    
    def _find_breakpoint_threshold(self, distances: List[float], number_of_cuts: int) -> float:
        """Binary search to find distance threshold that gives desired number of cuts."""
        lower_limit = 0.0
        upper_limit = 1.0
        distances_np = np.asarray(distances, dtype=float)
        threshold = 0.0

        while upper_limit - lower_limit > self.epsilon:
            threshold = (upper_limit + lower_limit) / 2.0
            num_points_above_threshold = np.sum(distances_np > threshold)

            if num_points_above_threshold > number_of_cuts:
                lower_limit = threshold
            else:
                upper_limit = threshold

        return threshold

    def _split_text(self, text: str) -> List[str]:
        """Split text using the upstream Kamradt semantic-chunking flow."""
        # The upstream implementation starts with recursive token-sized units.
        sentences = self._split_text_regular_expresion(text)

        if len(sentences) < 2:
            # Too few sentences to chunk semantically
            return [text] if text.strip() else []

        # Step 2: Add context for embedding calculation
        sentences = self.combine_sentences(sentences, buffer_size=3)

        sentences_embedded = self.calculate_sentence_embeddings(sentences)
        
        # Step 3: Calculate embeddings and distances
        distances, sentences = self.calculate_cosine_distances(sentences_embedded)

        if not distances:
            # No distances calculated
            return [text] if text.strip() else []

        # Initialize the start index
        start_index = 0

        # Create a list to hold the grouped sentences
        chunks = []
        breakepoint_distance_threshold = np.percentile(distances, BREAKPOINT_PERCENTILE_THRESHOLD)
        indices_above_thresh = [i for i, x in enumerate(distances) if x > breakepoint_distance_threshold]
        # Iterate through the breakpoints to slice the sentences
        for index in indices_above_thresh:
            # The end index is the current breakpoint
            end_index = index

            # Slice the sentence_dicts from the current start index to the end index
            group = sentences[start_index:end_index + 1]
            combined_text = ' '.join([d['sentence'] for d in group])
            chunks.append(combined_text)
            
            # Update the start index for the next group
            start_index = index + 1

        # The last group, if any sentences remain
        if start_index < len(sentences):
            combined_text = ' '.join([d['sentence'] for d in sentences[start_index:]])
            chunks.append(combined_text)

        return chunks
                


    def split_text(self, text: str) -> List[str]:
        """Alias for split_text for compatibility."""
        return self._split_text(text)


#KamradtModifiedChunker = SemanticChunker


if __name__ == "__main__":
    
    

    with open("data/row/Albers.txt", "r", encoding="utf-8") as f:
        text = f.read()
        semantic_chunker = KamradtSemanticChunker()
        splitted = semantic_chunker._split_text_regular_expresion(text)
        combined_sen = semantic_chunker.combine_sentences(splitted, buffer_size=3)
        embedded_sen = semantic_chunker.calculate_sentence_embeddings(combined_sen)
        distances, sentences = semantic_chunker.calculate_cosine_distances(embedded_sen)
        results_plot(distances , BREAKPOINT_PERCENTILE_THRESHOLD)

        chunks = semantic_chunker.split_text(text)
        for i, chunk in enumerate(chunks, 1):
            print(f"Chunk {i}: len={len(chunk)}")




   
   