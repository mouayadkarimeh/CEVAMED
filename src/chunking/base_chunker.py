from __future__ import annotations
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass

from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    TypeVar,
)

from chunking.base import BaseChunker


if TYPE_CHECKING:
    from collections.abc import Callable,  Iterable
    #from collections.abc import Set as AbstractSet




try:
    from transformers.tokenization_utils_base import PreTrainedTokenizerBase

    _HAS_TRANSFORMERS = True
except ImportError:
    _HAS_TRANSFORMERS = False

logger = logging.getLogger(__name__)

TS = TypeVar("TS", bound="TextSplitter")


class TextSplitter(BaseChunker, ABC):
    """Interface for splitting text into chunks."""

    def __init__(
        self,
        chunk_size: int = 256,
        chunk_overlap: int = 64,
        length_function: Callable[[str], int] = len,
        keep_separator: bool | Literal["start", "end"] = False,  # noqa: FBT001,FBT002
        add_start_index: bool = False,  # noqa: FBT001,FBT002
        strip_whitespace: bool = True,  # noqa: FBT001,FBT002
    ) -> None:
        """Create a new `TextSplitter`.

        Args:
            chunk_size: Maximum size of chunks to return
            chunk_overlap: Overlap in characters between chunks
            length_function: Function that measures the length of given chunks
            keep_separator: Whether to keep the separator and where to place it
                in each corresponding chunk `(True='start')`
            add_start_index: If `True`, includes chunk's start index in metadata
            strip_whitespace: If `True`, strips whitespace from the start and end of
                every document

        Raises:
            ValueError: If `chunk_size` is less than or equal to 0
            ValueError: If `chunk_overlap` is less than 0
            ValueError: If `chunk_overlap` is greater than `chunk_size`
        """
        if chunk_size <= 0:
            msg = f"chunk_size must be > 0, got {chunk_size}"
            raise ValueError(msg)
        if chunk_overlap < 0:
            msg = f"chunk_overlap must be >= 0, got {chunk_overlap}"
            raise ValueError(msg)
        if chunk_overlap > chunk_size:
            msg = (
                f"Got a larger chunk overlap ({chunk_overlap}) than chunk size "
                f"({chunk_size}), should be smaller."
            )
            raise ValueError(msg)
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._length_function = length_function
        self._keep_separator = keep_separator
        self._add_start_index = add_start_index
        self._strip_whitespace = strip_whitespace

    @abstractmethod
    def split_text(self, text: str) -> list[str]:
        """Split text into multiple components.

        Args:
            text: The text to split.

        Returns:
            A list of text chunks.
        """

    def _join_docs(self, docs: list[str], separator: str) -> str | None:
        text = separator.join(docs)
        if self._strip_whitespace:
            text = text.strip()
        return text or None

    def _merge_splits(self, splits: Iterable[str], separator: str) -> list[str]:
        # We now want to combine these smaller pieces into medium size
        # chunks to send to the LLM.
        separator_len = self._length_function(separator)

        docs = []
        current_doc: list[str] = []
        total = 0
        for d in splits:
            len_ = self._length_function(d)
            if (
                total + len_ + (separator_len if len(current_doc) > 0 else 0)
                > self._chunk_size
            ):
                if total > self._chunk_size:
                    logger.warning(
                        "Created a chunk of size %d, which is longer than the "
                        "specified %d",
                        total,
                        self._chunk_size,
                    )
                if len(current_doc) > 0:
                    doc = self._join_docs(current_doc, separator)
                    if doc is not None:
                        docs.append(doc)
                    # Keep on popping if:
                    # - we have a larger chunk than in the chunk overlap
                    # - or if we still have any chunks and the length is long
                    while total > self._chunk_overlap or (
                        total + len_ + (separator_len if len(current_doc) > 0 else 0)
                        > self._chunk_size
                        and total > 0
                    ):
                        total -= self._length_function(current_doc[0]) + (
                            separator_len if len(current_doc) > 1 else 0
                        )
                        current_doc = current_doc[1:]
            current_doc.append(d)
            total += len_ + (separator_len if len(current_doc) > 1 else 0)
        doc = self._join_docs(current_doc, separator)
        if doc is not None:
            docs.append(doc)
        return docs



    @classmethod
    def from_huggingface_tokenizer(
        cls, tokenizer: PreTrainedTokenizerBase, **kwargs: Any
    ) -> TextSplitter:
        """Text splitter that uses Hugging Face tokenizer to count length.

        Args:
            tokenizer: The Hugging Face tokenizer to use.

        Returns:
            An instance of `TextSplitter` using the Hugging Face tokenizer for length
                calculation.
        """
        if not _HAS_TRANSFORMERS:
            msg = (
                "Could not import transformers python package. "
                "Please install it with `pip install transformers`."
            )
            raise ValueError(msg)

        if not isinstance(tokenizer, PreTrainedTokenizerBase):
            # unreachable: transformers absent -> PreTrainedTokenizerBase is Any
            # unused-ignore: transformers present -> branch is reachable
            msg = (  # type: ignore[unreachable, unused-ignore]
                "Tokenizer received was not an instance of PreTrainedTokenizerBase"
            )
            raise ValueError(msg)  # noqa: TRY004

        def _huggingface_tokenizer_length(text: str) -> int:
            return len(tokenizer.tokenize(text))

        return cls(length_function=_huggingface_tokenizer_length, **kwargs)
   
    
   

      

@dataclass(frozen=True)
class Tokenizer:
    """Tokenizer data class."""

    chunk_overlap: int
    """Overlap in tokens between chunks"""

    tokens_per_chunk: int
    """Maximum number of tokens per chunk"""

    decode: Callable[[list[int]], str]
    """ Function to decode a list of token IDs to a string"""

    encode: Callable[[str], list[int]]
    """ Function to encode a string to a list of token IDs"""


def split_text_on_tokens(*, text: str, tokenizer: Tokenizer) -> list[str]:
    """Split incoming text and return chunks using tokenizer.

    Args:
        text: The input text to be split.
        tokenizer: The tokenizer to use for splitting.

    Returns:
        A list of text chunks.
    """
    splits: list[str] = []
    input_ids = tokenizer.encode(text)
    start_idx = 0
    if tokenizer.tokens_per_chunk <= tokenizer.chunk_overlap:
        msg = "tokens_per_chunk must be greater than chunk_overlap"
        raise ValueError(msg)

    while start_idx < len(input_ids):
        cur_idx = min(start_idx + tokenizer.tokens_per_chunk, len(input_ids))
        chunk_ids = input_ids[start_idx:cur_idx]
        if not chunk_ids:
            break
        decoded = tokenizer.decode(chunk_ids)
        if decoded:
            splits.append(decoded)
        if cur_idx == len(input_ids):
            break
        start_idx += tokenizer.tokens_per_chunk - tokenizer.chunk_overlap
    return splits
