from __future__ import annotations

from typing import List

from .base import BaseChunker, Chunk
from src.utils.preprocessing import normalize_german_clinical_text, split_sentences


class SemanticChunker(BaseChunker):
    """Sentence/paragraph aware chunking with max-size constraints."""

    def __init__(self, max_chunk_size: int = 512, paragraph_delimiter: str = "\n\n") -> None:
        self.max_chunk_size = max_chunk_size
        self.paragraph_delimiter = paragraph_delimiter

    def chunk(self, text: str) -> List[Chunk]:
        text = normalize_german_clinical_text(text)
        paragraphs = [p.strip() for p in text.split(self.paragraph_delimiter) if p.strip()]
        output: List[Chunk] = []
        current: List[str] = []

        def flush() -> None:
            if current:
                output.append(Chunk(text=" ".join(current).strip(), index=len(output), metadata={"type": "semantic"}))
                current.clear()

        for paragraph in paragraphs or [text]:
            for sentence in split_sentences(paragraph) or [paragraph]:
                candidate = (" ".join(current + [sentence])).strip()
                if current and len(candidate) > self.max_chunk_size:
                    flush()
                current.append(sentence)
            flush()

        return [chunk for chunk in output if chunk.text]
