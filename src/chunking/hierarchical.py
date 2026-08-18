from __future__ import annotations

from typing import List
import re

from .base import BaseChunker


class HierarchicalChunker(BaseChunker):
    """Document -> section -> paragraph -> sentence chunking.
    
    Returns exact substrings from the original document to preserve exact character positions.
    """

    def __init__(self, max_sentence_group_size: int = 400) -> None:
        self.max_sentence_group_size = max_sentence_group_size

    def split_text(self, text: str) -> List[str]:
        """Split text hierarchically while preserving exact substrings.
        
        Returns chunks that are exact substrings of the input text.
        """
        chunks = []
        
        # Split into paragraphs (separated by double newlines)
        paragraph_pattern = r'\n\n+'
        para_splits = [(m.start(), m.end()) for m in re.finditer(paragraph_pattern, text)]
        
        paragraph_ranges = []
        prev_end = 0
        for para_start, para_end in para_splits:
            if para_start > prev_end:
                paragraph_ranges.append((prev_end, para_start))
            prev_end = para_end
        if prev_end < len(text):
            paragraph_ranges.append((prev_end, len(text)))
        
        # Process each paragraph
        for para_start, para_end in paragraph_ranges:
            para_text = text[para_start:para_end]
            if not para_text.strip():
                continue
            
            # Split into sentences
            sent_pattern = r'(?<=[.!?])\s+'
            sent_splits = [(m.start(), m.end()) for m in re.finditer(sent_pattern, para_text)]
            
            sentence_ranges = []
            prev_end = 0
            for sent_start, sent_end in sent_splits:
                if sent_start > prev_end:
                    sentence_ranges.append((prev_end, sent_start))
                prev_end = sent_end
            if prev_end < len(para_text):
                sentence_ranges.append((prev_end, len(para_text)))
            
            # Group sentences into chunks
            buffer_start = None
            buffer_end = None
            
            for sent_start, sent_end in sentence_ranges:
                sent_text = para_text[sent_start:sent_end].strip()
                if not sent_text:
                    continue
                
                # Calculate what the chunk would be if we add this sentence
                if buffer_start is None:
                    buffer_start = sent_start
                    buffer_end = sent_end
                else:
                    candidate_text = para_text[buffer_start:sent_end]
                    if len(candidate_text) <= self.max_sentence_group_size:
                        buffer_end = sent_end
                    else:
                        # Chunk is full, save it and start new
                        chunk = text[para_start + buffer_start:para_start + buffer_end].strip()
                        if chunk:
                            chunks.append(chunk)
                        buffer_start = sent_start
                        buffer_end = sent_end
            
            # Save last chunk
            if buffer_start is not None:
                chunk = text[para_start + buffer_start:para_start + buffer_end].strip()
                if chunk:
                    chunks.append(chunk)
        
        return [c for c in chunks if c]
