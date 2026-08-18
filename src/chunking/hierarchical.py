from __future__ import annotations

from typing import List
import re

from .base import BaseChunker


class HierarchicalChunker(BaseChunker):
    """Document -> section -> paragraph -> sentence chunking."""

    def __init__(self, max_sentence_group_size: int = 400) -> None:
        self.max_sentence_group_size = max_sentence_group_size

    def _split_sections(self, text: str) -> List[tuple[str, str]]:
        lines = text.splitlines()
        sections: List[tuple[str, str]] = []
        current_title = "Dokument"
        bucket: List[str] = []
        header_re = re.compile(r"^(#+\s*)?(Anamnese|Befund|Diagnose|Therapie|Verlauf|Medikation)\b", re.IGNORECASE)

        for line in lines:
            if header_re.match(line.strip()):
                if bucket:
                    sections.append((current_title, "\n".join(bucket).strip()))
                    bucket = []
                current_title = line.strip("# ") or current_title
            else:
                bucket.append(line)
        if bucket:
            sections.append((current_title, "\n".join(bucket).strip()))
        return [(t, b) for t, b in sections if b]

    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        parts = re.split(r"(?<=[.!?])\s+", text.strip())
        return [part.strip() for part in parts if part.strip()]

    def split_text(self, text: str) -> List[str]:
        output: List[str] = []
        for _, section_body in self._split_sections(text):
            paragraphs = [p.strip() for p in section_body.split("\n\n") if p.strip()]
            for paragraph in paragraphs:
                buffer: List[str] = []
                for sentence in self._split_sentences(paragraph) or [paragraph]:
                    candidate = " ".join(buffer + [sentence]).strip()
                    if buffer and len(candidate) > self.max_sentence_group_size:
                        output.append(" ".join(buffer).strip())
                        buffer = []
                    buffer.append(sentence)
                if buffer:
                    output.append(" ".join(buffer).strip())
        return [chunk for chunk in output if chunk]
