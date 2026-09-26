"""
this script is adapted from langchain_text_splitters.transformer_token_splitter.py
"""

from __future__ import annotations

from typing import Any, cast
from chunking.base_chunker import TextSplitter
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator

try:
    # Type ignores needed as long as sentence-transformers doesn't support Python 3.14.
    from sentence_transformers import (  # type: ignore[import-not-found, unused-ignore]
        SentenceTransformer,
    )

    _HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    _HAS_SENTENCE_TRANSFORMERS = False

#MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
#MODEL_NAME = "deutsche-telekom/gbert-large-paraphrase-cosine"
MODEL_NAME = "codefuse-ai/F2LLM-v2-0.6B"

def results_plot(chunks: list[str], chunk_size: int, length_fn: Any) -> None:
    """Plottet die exakte Länge der Chunks. 

    Der Chunk-Index auf der X-Achse startet bei 1.
    """
    if not chunks:
        print("Keine Chunks zum Plotten.")
        return

    chunk_lengths = [length_fn(text =chunk) for chunk in chunks]
    
    # --- ÄNDERUNG 1: INDEX STARTET NUN BEI 1 STATT 0 ---
    x_positions = list(range(1, len(chunk_lengths) + 1))

    plt.figure(figsize=(10, 5))
    plt.plot(x_positions, chunk_lengths, marker="o", linestyle="-", label="Tatsächliche Chunk-Größe (tokens)")
    plt.axhline(y=chunk_size, color="r", linestyle="--", label=f"Ziel-Chunk-Größe ({chunk_size} tokens)")
    
    # --- ÄNDERUNG 2: AXIS-LIMITS & URSPRUNG ERZWINGEN ---
    ax = plt.gca()
    
    # Die X-Achse beginnt beim ersten Chunk statt bei 0.
    ax.set_xlim(left=1, right=max(len(chunk_lengths), 2))
    
    # Setzt die Y-Achse exakt bei 0 an und gibt nach oben 10% Puffer über dem Ziel-Limit
    ax.set_ylim(bottom=0, top=max(max(chunk_lengths), chunk_size) * 1.1)
    
    # Deaktiviert das automatische Matplotlib-Padding für den Ursprung
    ax.use_sticky_edges = True

    # Erzwingt ganze Zahlen auf der X-Achse
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    
    plt.title("sentence transformers token text splitter" , fontsize=22 , pad=22)
    plt.suptitle("Analysierte Datei: Albers.txt", fontsize=13, x=0.5, y=0.92, color="green")
    plt.xlabel("Chunk-Index")
    plt.ylabel("Chunk-Größe (token)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()















class TransformerTokenChunker(TextSplitter):
    """Splitting text to tokens using sentence model tokenizer."""

    def __init__(
        self,
        chunk_overlap: int = 50,
        model_name: str = MODEL_NAME,
        tokens_per_chunk: int | None = None,
        model_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Create a new `TextSplitter`.

        Args:
            chunk_overlap: The number of tokens to overlap between chunks.
            model_name: The name of the sentence transformer model to use.
            tokens_per_chunk: The number of tokens per chunk.

                If `None`, uses the maximum tokens allowed by the model.
            model_kwargs: Additional parameters for model initialization.
                Parameters of sentence_transformers.SentenceTransformer can be used.

        Raises:
            ImportError: If the `sentence_transformers` package is not installed.
        """
        super().__init__(**kwargs, chunk_overlap=chunk_overlap)

        if not _HAS_SENTENCE_TRANSFORMERS:
            msg = (
                "Could not import sentence_transformers python package. "
                "This is needed in order to use SentenceTransformersTokenTextSplitter. "
                "Please install it with `pip install sentence-transformers`."
            )
            raise ImportError(msg)

        self.model_name = model_name
        self._model = SentenceTransformer(self.model_name, **(model_kwargs or {}))
        self.tokenizer = self._model.tokenizer
        self._initialize_chunk_configuration(tokens_per_chunk=tokens_per_chunk)

    def _initialize_chunk_configuration(self, *, tokens_per_chunk: int | None) -> None:
        self.maximum_tokens_per_chunk = self._model.max_seq_length

        if tokens_per_chunk is None:
            self.tokens_per_chunk = self.maximum_tokens_per_chunk
        else:
            self.tokens_per_chunk = tokens_per_chunk

        if self.tokens_per_chunk > self.maximum_tokens_per_chunk:
            msg = (
                f"The token limit of the models '{self.model_name}'"
                f" is: {self.maximum_tokens_per_chunk}."
                f" Argument tokens_per_chunk={self.tokens_per_chunk}"
                f" > maximum token limit."
            )
            raise ValueError(msg)

    def split_text(self, text: str) -> list[str]:
        """Splits the input text into smaller components by splitting text on tokens.

        This method encodes the input text using a private `_encode` method, then
        strips the start and stop token IDs from the encoded result. It returns the
        processed segments as a list of strings.

        Args:
            text: The input text to be split.

        Returns:
            A list of string components derived from the input text after encoding and
                processing.
        """

        if self.tokens_per_chunk <= self._chunk_overlap:
            msg = "tokens_per_chunk must be greater than chunk_overlap"
            raise ValueError(msg)

        input_ids = self._encode(text)[1:-1]
        chunks: list[str] = []
        start_idx = 0

        while start_idx < len(input_ids):
            end_idx = min(start_idx + self.tokens_per_chunk, len(input_ids))
            chunk = self.tokenizer.decode(input_ids[start_idx:end_idx])

            while self.count_tokens(text=chunk) > self.tokens_per_chunk:
                end_idx -= 1
                if end_idx <= start_idx:
                    msg = "Tokenizer could not produce a chunk within the token limit"
                    raise ValueError(msg)
                chunk = self.tokenizer.decode(input_ids[start_idx:end_idx])

            chunks.append(chunk)
            if end_idx == len(input_ids):
                break

            start_idx = end_idx - self._chunk_overlap

        return chunks

    def count_tokens(self, *, text: str) -> int:
        """Count content tokens without the tokenizer's boundary tokens.

        This uses the same token definition as :meth:`split_text`, so complete
        chunks do not exceed ``tokens_per_chunk`` when measured with this method.

        Args:
            text: The input text for which the token count is calculated.

        Returns:
            The number of content tokens in the encoded text.
        """
        return len(self._encode(text)[1:-1])

    _max_length_equal_32_bit_integer: int = 2**32

    def _encode(self, text: str) -> list[int]:
        token_ids_with_start_and_end_token_ids = self.tokenizer.encode(
            text,
            max_length=self._max_length_equal_32_bit_integer,
            truncation="do_not_truncate",
        )
        return cast("list[int]", token_ids_with_start_and_end_token_ids)


if __name__  == "__main__":

    with open("data/row/Albers.txt", "r", encoding="utf-8") as file:
        text = file.read()
    
        splitter = TransformerTokenChunker(
            chunk_overlap=64,
            model_name= MODEL_NAME,
            tokens_per_chunk=256,
        )
        print(f" the model name: {splitter.model_name}")
        chunks = splitter.split_text(text)
        print("Chunks:", chunks)
        print("Token count:", splitter.count_tokens(text=text))
        results_plot(
            chunks=chunks,
            chunk_size=splitter.tokens_per_chunk,
            length_fn=splitter.count_tokens,
        )
