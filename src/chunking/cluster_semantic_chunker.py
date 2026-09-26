from __future__ import annotations
import os
import sys
from pathlib import Path
from typing import List, Callable, Any, Optional
import numpy as np
from matplotlib import pyplot as plt
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer
import matplotlib.patches as patches

# Interface aus der Vaterklasse laden
try:
    from chunking.base_chunker import TextSplitter
except ImportError:
    from langchain.text_splitter import TextSplitter

from chunking.recursive_token_chunker import RecursiveTokenChunker


#MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
MODEL_NAME = "deutsche-telekom/gbert-large-paraphrase-cosine"
#MODEL_NAME = "codefuse-ai/F2LLM-v2-0.6B"


def plot_semantic_matrix(matrix: np.ndarray, clusters: list[tuple[int, int]]) -> None:
    """
    Plottet die globale Ähnlichkeitsmatrix als Heatmap und zeichnet 
    die berechneten Cluster als Quadrate entlang der Diagonale ein.
    """
    if matrix.size == 0:
        print("Matrix ist leer. Plotten nicht möglich.")
        return

    plt.figure(figsize=(8, 8))
    # Heatmap der Ähnlichkeitsmatrix zeichnen (viridis oder coolwarm eignen sich gut)
    ax = plt.gca()
    im = ax.imshow(matrix, cmap="viridis", origin="upper")
    plt.colorbar(im, label="Kosinus-Ähnlichkeit (normalisiert)")

    # Farbpalette für die Rahmen der Cluster
    #colors = ["red", "orange", "cyan", "magenta", "yellow", "lime"]
    uniform_color = "white"
    # Für jedes gefundene Cluster ein Quadrat auf der Diagonale einzeichnen
    for idx, (start, end) in enumerate(clusters):
        size = end - start + 1
        # Ein Quadrat zeichnen (Startpunkt oben links im Cluster-Block)
        rect = patches.Rectangle(
            (start - 0.5, start - 0.5),  # Versatz um 0.5 für perfekte Ausrichtung auf die Pixel
            size,
            size,
            linewidth=2.5,
            #edgecolor=colors[idx % len(colors)],
            edgecolor=uniform_color,
            facecolor="none",
            label=f"Chunk #{idx}" if idx < 6 else "" # Label nur für die ersten paar anzeigen
        )
        ax.add_patch(rect)
        
        # Text-Label in die Mitte des Quadrats setzen
        ax.text(
            start + size/2 - 0.5, 
            start + size/2 - 0.5, 
            f"#{idx}", 
            color="white", 
            fontweight="bold",
            ha="center", 
            va="center"
        )

    plt.title("Globale Text-Ähnlichkeit & DP-Cluster-Segmente")
    plt.xlabel("Satz-Index")
    plt.ylabel("Satz-Index")
    plt.grid(False)
    plt.show()



def plot_cluster_sizes(chunks: list[str], max_chunk_size: int, length_fn: Callable[[str], int]) -> None:
    """
    Plottet die tatsächliche Token-Länge jedes erzeugten Cluster-Chunks.
    """
    lengths = [length_fn(chunk) for chunk in chunks]
    x = list(range(1, len(lengths) + 1))

    plt.figure(figsize=(10, 4))
    plt.bar(x, lengths, color="skyblue", edgecolor="navy", alpha=0.8, label="Tatsächliche Tokens")
    plt.axhline(y=max_chunk_size, color="r", linestyle="--", label=f"Max Limit ({max_chunk_size})")
    
    plt.title("Token-Verteilung der Cluster-Chunks")
    plt.xlabel("Chunk Nummer")
    plt.ylabel("Anzahl Tokens")
    plt.xticks(x)
    plt.legend()
    plt.grid(True, axis="y", alpha=0.3)
    plt.show()


class ClusterSemanticChunker(TextSplitter):
    def __init__(self, 
                 embedding_function: Optional[Any] = None, 
                 max_chunk_size: int = 256, 
                 min_chunk_size: int = 50,
                 length_function: Optional[Callable[[str], int]] = None, 
                 **kwargs: Any):
        
        # 1. Holt die length_function ab (Direkt oder über die Factory-Methode der Vaterklasse)
        self.length_function = length_function or kwargs.get("length_function")
        
        if self.length_function is None:
            tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
            
            
            self.length_function = lambda text: len(tokenizer.tokenize(text))

        # 2. Den internen Token-Splitter mit der echten length_function versorgen
        self.splitter = RecursiveTokenChunker(
            chunk_size=min_chunk_size,
            chunk_overlap=0,
            length_function=self.length_function,
            separators=["\n\n", "\n", ".", "?", "!", " ", ""]
        )
        
        if embedding_function is None:
            embedding_function = SentenceTransformer(MODEL_NAME)
            
        self.embedding_function = embedding_function
        self.max_cluster = max_chunk_size // min_chunk_size
        
        # 3. Parameter für die Vaterklasse vorbereiten
        kwargs["length_function"] = self.length_function
        kwargs["chunk_size"] = max_chunk_size
        super().__init__(**kwargs)
        
    def _get_similarity_matrix(self, embedding_function: Any, sentences: List[str]) -> np.ndarray:
        if not sentences:
            return np.zeros((0, 0))
            
        BATCH_SIZE = 500
        N = len(sentences)
        embedding_matrix = None

        for i in range(0, N, BATCH_SIZE):
            batch_sentences = sentences[i:i+BATCH_SIZE]
            
            # KORREKTUR: .encode() nutzen anstatt das Objekt direkt als Funktion aufzurufen
            if hasattr(embedding_function, "encode"):
                embeddings = embedding_function.encode(batch_sentences, show_progress_bar=False)
            else:
                embeddings = embedding_function(batch_sentences)

            batch_embedding_matrix = np.array(embeddings)

            if embedding_matrix is None:
                embedding_matrix = batch_embedding_matrix
            else:
                embedding_matrix = np.concatenate((embedding_matrix, batch_embedding_matrix), axis=0)

        # Normalisierung für mathematisch korrekte Kosinus-Ähnlichkeit
        norms = np.linalg.norm(embedding_matrix, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1e-9, norms)
        embedding_matrix = embedding_matrix / norms

        similarity_matrix = np.dot(embedding_matrix, embedding_matrix.T)
        return similarity_matrix

    def _calculate_reward(self, matrix: np.ndarray, start: int, end: int) -> float:
        sub_matrix = matrix[start:end+1, start:end+1]
        return float(np.sum(sub_matrix))

    def _optimal_segmentation(self, matrix: np.ndarray, max_cluster_size: int) -> List[tuple[int, int]]:
        if matrix.size == 0:
            return []
            
        # Wenn die Matrix zu klein für Triu-Operationen ist (z.B. nur 1 Satz)
        if matrix.shape[0] <= 1:
            return [(0, 0)]

        mean_value = np.mean(matrix[np.triu_indices(matrix.shape[0], k=1)])
        print(f"Mean value of the upper triangular matrix: {mean_value}")
        matrix = matrix - (mean_value)  # Matrix normalisieren
        np.fill_diagonal(matrix, 0)  

        n = matrix.shape[0]
        dp = np.zeros(n)
        segmentation = np.zeros(n, dtype=int)

        split_penalty = 1.5

        for i in range(n):
            for size in range(1, max_cluster_size + 1):
                if i - size + 1 >= 0:
                    reward = self._calculate_reward(matrix, i - size + 1, i)
                    adjusted_reward = reward
                    if i - size >= 0:
                        adjusted_reward += dp[i - size] 
                    if adjusted_reward > dp[i] or size == 1:
                        dp[i] = adjusted_reward
                        segmentation[i] = i - size + 1

        clusters = []
        i = n - 1
        while i >= 0:
            start = segmentation[i]
            clusters.append((start, i))
            i = start - 1

        clusters.reverse()
        return clusters
        
    def split_text(self, text: str) -> List[str]:
        sentences = self.splitter.split_text(text)
        if not sentences:
            return []

        similarity_matrix = self._get_similarity_matrix(self.embedding_function, sentences)
        clusters = self._optimal_segmentation(similarity_matrix, max_cluster_size=self.max_cluster)
        docs = [' '.join(sentences[start:end+1]) for start, end in clusters]
        return docs


if __name__ == "__main__":
    with open("data/row/Albers.txt" , "r" , encoding = "utf-8") as f:
        text = f.read()
        hf_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        chunker = ClusterSemanticChunker.from_huggingface_tokenizer(hf_tokenizer, max_chunk_size=400, min_chunk_size=200)
        
        # 1. Sätze holen
        splitted_sentences = chunker.splitter.split_text(text)
        total_tokens = sum(chunker.length_function(s) for s in splitted_sentences)
        print(f"Total tokens: {total_tokens}")
        # 2. Matrix berechnen & DP-Cluster direkt für das Plotten abgreifen
        sim_matrix = chunker._get_similarity_matrix(chunker.embedding_function, splitted_sentences)
        
        # Normalisierung simulieren, die intern in _optimal_segmentation passiert
        mean_val = np.mean(sim_matrix[np.triu_indices(sim_matrix.shape[0], k=1)])
        norm_matrix = sim_matrix - mean_val
        np.fill_diagonal(norm_matrix, 1.0) # Für den visuellen Kontrast der Diagonale auf 1 setzen
        
        calculated_clusters = chunker._optimal_segmentation(sim_matrix, max_cluster_size=chunker.max_cluster)
        
        # 3. HEATMAP PLOTTEN
        plot_semantic_matrix(norm_matrix, calculated_clusters)
        
        # 4. FINALE CHUNKS ERZEUGEN & GRÖSSEN PLOTTEN
        chunks = chunker.split_text(text)
        plot_cluster_sizes(chunks, chunker._chunk_size, chunker.length_function)
        

        for i, chunk in enumerate(chunks):

            print(f"Chunk {i+1} ({chunker.length_function(chunk)} Tokens):\n{chunk[:60]}...\n")
        