"""Intrachunk cohesion metric for span-aligned text chunks."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from functools import lru_cache
from typing import Any, Protocol

import numpy as np


DEFAULT_ICC_MODEL = "codefuse-ai/F2LLM-v2-0.6B"
#DEFAULT_ICC_MODEL = "deutsche-telekom/gbert-large-paraphrase-cosine"
#DEFAULT_ICC_MODEL = "sentence-transformers/all-MiniLM-L6-v2"



class EmbeddingModel(Protocol):
    """Minimal interface required from an embedding model."""

    def encode(self, sentences: Sequence[str], **kwargs: Any) -> Any: ...


def load_icc_embedding_model() -> EmbeddingModel:
    """Load the configured embedding model used by ICC."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(DEFAULT_ICC_MODEL)


@lru_cache(maxsize=1)
def _text_block_tokenizer() -> Any:
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(DEFAULT_ICC_MODEL, use_fast=True)


def parse_text_blocks(
    text: str,
    *,
    tokenizer: Any | None = None,
) -> list[dict[str, str | int]]:
    """Split text at sentence punctuation using Hugging Face token offsets."""
    if not text.strip():
        return []

    active_tokenizer = tokenizer or _text_block_tokenizer()
    encoding = active_tokenizer(
        text,
        add_special_tokens=False,
        return_offsets_mapping=True,
        truncation=False,
    )
    offsets = encoding["offset_mapping"]
    blocks: list[dict[str, str | int]] = []
    block_start = 0

    def append_block(start: int, end: int) -> None:
        while start < end and text[start].isspace():
            start += 1
        while end > start and text[end - 1].isspace():
            end -= 1
        if start < end:
            blocks.append(
                {
                    "text": text[start:end],
                    "start_index": start,
                    "end_index": end,
                }
            )

    for token_start, token_end in offsets:
        token_text = text[token_start:token_end]
        followed_by_space = token_end == len(text) or text[token_end].isspace()
        if token_text in {".", "!", "?"} and followed_by_space:
            append_block(block_start, token_end)
            block_start = token_end

    append_block(block_start, len(text))
    return blocks


def _span_value(span: Mapping[str, object], key: str) -> int:
    try:
        return int(span[key])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Span requires an integer '{key}' value") from error


def _span_text(span: Mapping[str, object]) -> str:
    text = span.get("text", span.get("content"))
    if not isinstance(text, str):
        raise ValueError("Span requires a string 'text' or 'content' value")
    return text


def intrachunk_cohesion(
    chunks: Sequence[Mapping[str, object]],
    text_blocks: Sequence[Mapping[str, object]],
    *,
    embedding_model: EmbeddingModel | None = None,
) -> float:
    """Calculate mean sentence-to-chunk semantic cohesion.

    A text block belongs to every chunk whose half-open character range contains
    the block's ``start_index``. Chunks containing fewer than two blocks are
    ignored. If no valid chunks remain, the result is ``0.0``.

    Args:
        chunks: Chunk spans with ``start_index``, ``end_index`` and ``text`` or
            ``content``.
        text_blocks: Parsing-output spans with ``start_index`` and ``text`` or
            ``content``.
        embedding_model: Optional model exposing ``encode``. By default,
            ``codefuse-ai/F2LLM-v2-0.6B`` is loaded through SentenceTransformers.

    Returns:
        ICC in the range ``[0.0, 1.0]``.
    """
    valid_chunks: list[tuple[str, list[str]]] = []
    block_starts = [(_span_value(block, "start_index"), block) for block in text_blocks]

    for chunk in chunks:
        chunk_start = _span_value(chunk, "start_index")
        chunk_end = _span_value(chunk, "end_index")
        if chunk_end <= chunk_start:
            raise ValueError("Chunk 'end_index' must be greater than 'start_index'")

        block_texts = [
            _span_text(block)
            for block_start, block in block_starts
            if chunk_start <= block_start < chunk_end
        ]
        if len(block_texts) >= 2:
            valid_chunks.append((_span_text(chunk), block_texts))

    if not valid_chunks:
        return 0.0

    texts = [
        text
        for chunk_text, block_texts in valid_chunks
        for text in (chunk_text, *block_texts)
    ]
    model = embedding_model or load_icc_embedding_model()
    embeddings = np.asarray(
        model.encode(texts, normalize_embeddings=True, show_progress_bar=False),
        dtype=float,
    )
    if embeddings.ndim != 2 or embeddings.shape[0] != len(texts):
        raise ValueError("Embedding model returned an unexpected output shape")

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = np.divide(
        embeddings,
        norms,
        out=np.zeros_like(embeddings),
        where=norms != 0,
    )

    chunk_scores: list[float] = []
    embedding_index = 0
    for _, block_texts in valid_chunks:
        chunk_embedding = embeddings[embedding_index]
        embedding_index += 1
        block_embeddings = embeddings[embedding_index : embedding_index + len(block_texts)]
        embedding_index += len(block_texts)
        chunk_scores.append(float(np.mean(block_embeddings @ chunk_embedding)))

    return max(0.0, min(1.0, float(np.mean(chunk_scores))))