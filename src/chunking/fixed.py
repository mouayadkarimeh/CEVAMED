from __future__ import annotations

from typing import List

from .base import BaseChunker, Chunk


class FixedSizeChunker(BaseChunker):
    """Fixed-size chunking with configurable overlap."""

    def __init__(self, chunk_size: int = 512, overlap: int = 64) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be > 0")
        if overlap < 0 or overlap >= chunk_size:
            raise ValueError("overlap must be >= 0 and < chunk_size")
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str) -> List[Chunk]:
        stride = self.chunk_size - self.overlap
        output: List[Chunk] = []
        for i, start in enumerate(range(0, len(text), stride)):
            part = text[start : start + self.chunk_size]
            if part.strip():
                output.append(Chunk(text=part, index=i, metadata={"start": start, "end": start + len(part)}))
        return output
