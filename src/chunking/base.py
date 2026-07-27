from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class Chunk:
    """A text chunk and optional metadata."""

    text: str
    index: int
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseChunker(ABC):
    """Common interface for chunking strategies."""

    @abstractmethod
    def chunk(self, text: str) -> List[Chunk]:
        """Split input text into chunks."""


__all__ = ["Chunk", "BaseChunker"]
