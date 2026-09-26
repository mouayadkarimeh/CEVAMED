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
from chunking.base_chunker import TextSplitter

sys.path.append(str(Path(__file__).resolve().parents[2]))



#MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
MODEL_NAME = "deutsche-telekom/gbert-large-paraphrase-cosine"
#MODEL_NAME = "codefuse-ai/F2LLM-v2-0.6B"

DEFAULT_EPSILON = 1e-6
BREAKPOINT_PERCENTILE_THRESHOLD = 92 # Default percentile for determining breakpoints

def results_plot(distances: list[float], BREAKPOINT_PERCENTILE_THRESHOLD ) -> None:
    if not distances:
        print("Keine Distanzen zum Plotten.")
        return

    
    y_upper_bound = .6 # Set a fixed upper bound for better visualization  
    plt.plot(distances)
    plt.ylim(0, y_upper_bound)
    plt.xlim(0, len(distances))
    breakpoint_distance_threshold = np.percentile(distances, BREAKPOINT_PERCENTILE_THRESHOLD)
    plt.axhline(y = breakpoint_distance_threshold , color="r", linestyle="--")

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

    plt.title("Semantisches Chunking nach Kamradt", fontsize=15, pad=22)
    plt.suptitle("Analysierte Datei: Albers.txt", fontsize=12, x=0.5, y=0.92, color="green")
    plt.xlabel("Satz-Index")
    plt.ylabel("Kosinus-Distanz zwischen aufeinanderfolgenden Sätzen")
    plt.grid(True, alpha=0.3)
    plt.show()







class KamradtSemanticChunker(TextSplitter):
    """Semantic chunker using sentence embeddings and distance breakpoints.
    
    CRITICAL: This implementation extracts EXACT SUBSTRINGS from the original text
    without any transformation (no lowercase, no unicode normalization, no whitespace
    changes). This ensures perfect alignment with ground truth for evaluation.
    """

    def __init__(
        self,
        model_name: str = MODEL_NAME,
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

    
   

    def _count_tokens(self, text: str) -> int:
        """Count tokens with the same tokenizer used for token-based boundaries."""
        return len(
            self._tokenizer.encode(
                text,
                allowed_special=self._allowed_special,
                disallowed_special=self._disallowed_special,
            )
        )

   


   
    @staticmethod
    def combine_sentences(sentences: List[Dict[str, object]], buffer_size: int = 1) -> List[Dict[str, object]]:
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
    
    from transformers import AutoTokenizer
    
    # 1. Tokenizer laden
  
    hf_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # 2. Das fehlende statistische Perzentil für die Plot-Funktion definieren
    # (Kamradts Methode nutzt meist das 95. Perzentil für Ausreißer)
    

    with open("data/row/Albers.txt", "r", encoding="utf-8") as f:
        text = f.read()
        
        # KORREKTUR 1: Keine Argumente übergeben, damit model_name ein String bleibt
        semantic_chunker = KamradtSemanticChunker()
        
        # KORREKTUR 2: Der Klasse nachträglich beibringen, Tokens zu zählen
        # Wir überschreiben sowohl die geschützte als auch die öffentliche Variante zur Sicherheit
        token_len_fn = lambda t: len(hf_tokenizer.tokenize(t))
        semantic_chunker._length_function = token_len_fn
        semantic_chunker.length_function = token_len_fn

        # 3. Deine Teilschritte ausführen (Logik exakt beibehalten)
        splitted = semantic_chunker._split_text_regular_expresion(text)
        combined_sen = semantic_chunker.combine_sentences(splitted, buffer_size=1)
        embedded_sen = semantic_chunker.calculate_sentence_embeddings(combined_sen)
        distances, sentences = semantic_chunker.calculate_cosine_distances(embedded_sen)
        
        # Plot aufrufen (Stellt sicher, dass die Funktion im Skript existiert)
        results_plot(distances, BREAKPOINT_PERCENTILE_THRESHOLD)
        #plot_embedding_breakpoints(distances, BREAKPOINT_PERCENTILE_THRESHOLD)

        # 4. Splitten und die echten Token-Zahlen ausgeben
        chunks = semantic_chunker.split_text(text)
        print(f"\n--- Auswertung mit KamradtSemanticChunker (Statisch) ---")
        for i, chunk in enumerate(chunks, 1):
            token_count = semantic_chunker.length_function(chunk)
            print(f"Chunk {i}: len_tokens={token_count} | (len_zeichen={len(chunk)})")

        print(f"total Token count über alle Chunks: {sum(semantic_chunker.length_function(chunk) for chunk in chunks)}")

    
    




   
   