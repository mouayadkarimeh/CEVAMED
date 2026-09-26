from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any, List, Optional , Callable
import matplotlib.pyplot as plt
from transformers import AutoTokenizer ,  PreTrainedTokenizerBase
from matplotlib.ticker import MaxNLocator
 

# Import der soeben bereitgestellten Basisklasse

try:
    from chunking.base_chunker import TextSplitter
except ImportError:  # pragma: no cover - Fallback für direkte Ausführung
    from chunking.base_chunker import TextSplitter

sys.path.append(str(Path(__file__).resolve().parents[2]))


#MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
MODEL_NAME = "deutsche-telekom/gbert-large-paraphrase-cosine"
#MODEL_NAME = "codefuse-ai/F2LLM-v2-0.6B"


def results_plot(chunks: list[str], chunk_size: int, length_fn: Any) -> None:
    """Plottet die exakte Länge der Chunks. 

    Der Index startet bei 1 und die Achsen treffen sich exakt im Ursprung.
    """
    if not chunks:
        print("Keine Chunks zum Plotten.")
        return

    chunk_lengths = [length_fn(text =chunk) for chunk in chunks]
    
    # --- ÄNDERUNG 1: INDEX STARTET NUN BEI 1 STATT 0 ---
    x_positions = list(range(1, len(chunk_lengths) + 1))

    plt.figure(figsize=(10, 5))
    plt.plot(x_positions, chunk_lengths, marker="o", linestyle="-", label="Tatsächliche Chunk-Größe (Tokens)")
    plt.axhline(y=chunk_size, color="r", linestyle="--", label=f"Ziel-Chunk-Größe ({chunk_size} Tokens)")
    
    # --- ÄNDERUNG 2: AXIS-LIMITS & URSPRUNG ERZWINGEN ---
    ax = plt.gca()
    
    # Setzt die Grenzen der X-Achse exakt von 1 bis zum letzten Chunk (ohne Abstand links/rechts)
    ax.set_xlim(left=1, right=len(chunk_lengths))
    
    # Setzt die Y-Achse exakt bei 0 an und gibt nach oben 10% Puffer über dem Ziel-Limit
    ax.set_ylim(bottom=0, top=max(max(chunk_lengths), chunk_size) * 1.1)
    
    # Deaktiviert das automatische Matplotlib-Padding für den Ursprung
    ax.use_sticky_edges = True

    # Erzwingt ganze Zahlen auf der X-Achse
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    
    plt.title("Rekursiver Token Chunker" , fontsize=22 , pad=22)
    plt.suptitle("Analysierte Datei: Albers.txt", fontsize=13, x=0.5, y=0.92, color="green")
    plt.xlabel("Chunk-Index")
    plt.ylabel("Chunk-Größe (Token)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()



def _split_text_with_regex(text: str, separator: str, keep_separator: bool) -> List[str]:
    """Splittet Text anhand eines Separators und behält diesen optional bei."""
    if separator:
        if keep_separator:
            _splits = re.split(f"({separator})", text)
            # Fehlerbehebung: Vorheriger Index-Fehler beim Zusammenfügen behoben
            splits = []
            if _splits[0]:
                splits.append(_splits[0])
            for i in range(1, len(_splits), 2):
                if i + 1 < len(_splits):
                    splits.append(_splits[i] + _splits[i + 1])
                else:
                    splits.append(_splits[i])
        else:
            splits = re.split(separator, text)
    else:
        splits = list(text)
    return [s for s in splits if s != ""]


class RecursiveTokenChunker(TextSplitter):
    """Splittet Text rekursiv auf Basis von Tokens unter Einhaltung der Basisklasse."""

    def __init__(
        self,
        chunk_size: int = 256,
        chunk_overlap: int = 64,
        separators: Optional[list[str]] = None,
        keep_separator: bool = True,
        is_separator_regex: bool = False,
        length_function: Optional[Callable[[str], int]] = None,  # Explizit typisieren
        **kwargs: Any,
    ) -> None:
        # WICHTIG: Falls keine length_function übergeben wurde, wird standardmäßig
        # der globale Tokenizer als Fallback genutzt, um mit der Basisklasse kompatibel zu sein.
        if "length_function" not in kwargs:
            tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
            


            kwargs["length_function"] = lambda text: len(tokenizer.tokenize(text))

        super().__init__(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            keep_separator=keep_separator,
            **kwargs,
        )
        
        self._separators = separators or ["\n\n", "\n", ".", "?", "!", " ", ""]

        self._is_separator_regex = is_separator_regex
   


    
    def _split_text(self, text: str, separators: List[str]) -> List[str]:
        """Interne rekursive Split-Logik."""
        final_chunks = []
        separator = separators[-1]
        new_separators = []
        
        for i, _s in enumerate(separators):
            _separator = _s if self._is_separator_regex else re.escape(_s)
            if _s == "":
                separator = _s
                break
            if re.search(_separator, text):
                separator = _s
                new_separators = separators[i + 1 :]
                break

        _separator = separator if self._is_separator_regex else re.escape(separator)
        splits = _split_text_with_regex(text, _separator, self._keep_separator)

        _good_splits = []
        _merge_separator = "" if self._keep_separator else separator
        
        for s in splits:
            # self._length_function greift nun sauber auf die Token-Länge der Basisklasse zu
            if self._length_function(s) < self._chunk_size:
                _good_splits.append(s)
            else:
                if _good_splits:
                    merged_text = self._merge_splits(_good_splits, _merge_separator)
                    final_chunks.extend(merged_text)
                    _good_splits = []
                if not new_separators:
                    final_chunks.append(s)
                else:
                    other_info = self._split_text(s, new_separators)
                    final_chunks.extend(other_info)
                    
        if _good_splits:
            merged_text = self._merge_splits(_good_splits, _merge_separator)
            final_chunks.extend(merged_text)
            
        return final_chunks

    def split_text(self, text: str) -> List[str]:
        return self._split_text(text, self._separators)


if __name__ == "__main__":

    hf_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    with open("data/row/Albers.txt" , "r" , encoding = "utf-8") as f:
        text  = f.read()
        total_tokens = len(hf_tokenizer.tokenize(text))
        splitter = RecursiveTokenChunker.from_huggingface_tokenizer(
            tokenizer=hf_tokenizer,
            chunk_size=256,
            chunk_overlap=64)
        chunks = splitter.split_text(text)
        for i, chunk in enumerate(chunks):
            print(f"Chunk {i} ({splitter._length_function(chunk)} Tokens): {chunk}")
        print( f"total tokens: {total_tokens}")    

        results_plot(chunks=chunks, chunk_size=splitter._chunk_size, length_fn=splitter._length_function)
