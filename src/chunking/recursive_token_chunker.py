
try:
    from chunking.base_chunker import TextSplitter
except ImportError:  # pragma: no cover - direct script execution fallback
    from chunking.base_chunker import TextSplitter
    
from typing import Literal, Any
import re 
import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).resolve().parents[2]))



def _split_text_with_regex(
    text: str, separator: str, *, keep_separator: bool | Literal["start", "end"]
) -> list[str]:
    # Now that we have the separator, split the text
    if separator:
        if keep_separator:
            # The parentheses in the pattern keep the delimiters in the result.
            splits_ = re.split(f"({separator})", text)
            splits = (
                ([splits_[i] + splits_[i + 1] for i in range(0, len(splits_) - 1, 2)])
                if keep_separator == "end"
                else ([splits_[i] + splits_[i + 1] for i in range(1, len(splits_), 2)])
            )
            if len(splits_) % 2 == 0:
                splits += splits_[-1:]
            splits = (
                ([*splits, splits_[-1]])
                if keep_separator == "end"
                else ([splits_[0], *splits])
            )
        else:
            splits = re.split(separator, text)
    else:
        splits = list(text)
    return [s for s in splits if s]


def results_plot(chunks: list[str], chunk_size: int) -> None:
    if not chunks:
        print("Keine Chunks zum Plotten.")
        return

    chunk_lengths = [len(chunk) for chunk in chunks]
    x_positions = list(range(len(chunk_lengths)))

    plt.plot(x_positions, chunk_lengths, marker="o")
    plt.axhline(y=chunk_size, color="r", linestyle="-", label="chunk_size")
    plt.title("Recursive Chunk Sizes")
    plt.xlabel("Chunk index")
    plt.ylabel("Chunk length")
    plt.legend()
    plt.show()


class RecursiveTokenChunker(TextSplitter):
    """Splitting text by recursively look at characters.

    Recursively tries to split by different characters to find one
    that works.
    """

    def __init__(
        self,
        separators: list[str] | None = None,
        keep_separator: bool | Literal["start", "end"] = True,  # noqa: FBT001,FBT002
        is_separator_regex: bool = False,  # noqa: FBT001,FBT002
        **kwargs: Any,
    ) -> None:
        """Create a new TextSplitter."""
        super().__init__(keep_separator=keep_separator, **kwargs)
        self._separators = separators or ["\n\n", "\n", " ", ""]
        self._is_separator_regex = is_separator_regex

    def _split_text(self, text: str, separators: list[str]) -> list[str]:
        """Split incoming text and return chunks."""
        final_chunks = []
        # Get appropriate separator to use
        separator = separators[-1]
        new_separators = []
        for i, s_ in enumerate(separators):
            separator_ = s_ if self._is_separator_regex else re.escape(s_)
            if not s_:
                separator = s_
                break
            if re.search(separator_, text):
                separator = s_
                new_separators = separators[i + 1 :]
                break

        separator_ = separator if self._is_separator_regex else re.escape(separator)
        splits = _split_text_with_regex(
            text, separator_, keep_separator=self._keep_separator
        )

        # Now go merging things, recursively splitting longer texts.
        good_splits = []
        separator_ = "" if self._keep_separator else separator
        for s in splits:
            if self._length_function(s) < self._chunk_size:
                good_splits.append(s)
            else:
                if good_splits:
                    merged_text = self._merge_splits(good_splits, separator_)
                    final_chunks.extend(merged_text)
                    good_splits = []
                if not new_separators:
                    final_chunks.append(s)
                else:
                    other_info = self._split_text(s, new_separators)
                    final_chunks.extend(other_info)
        if good_splits:
            merged_text = self._merge_splits(good_splits, separator_)
            final_chunks.extend(merged_text)
        return final_chunks

    def split_text(self, text: str) -> list[str]:
        """Split the input text into smaller chunks based on predefined separators.

        Args:
            text: The input text to be split.

        Returns:
            A list of text chunks obtained after splitting.
        """
        return self._split_text(text, self._separators)



if __name__ == "__main__":


    


    with open("data/row/Albers.txt", "r", encoding="utf-8") as f:
        text = f.read()
        recursive = RecursiveCharacterTextSplitter()
        chunks = recursive.split_text(text)
        print(chunks)
        results_plot(chunks, recursive._chunk_size)