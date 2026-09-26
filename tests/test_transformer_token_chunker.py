from chunking.trasnformer_token_chunker import TransformerTokenChunker


class RoundTripGrowingTokenizer:
    def encode(self, text: str, **_: object) -> list[int]:
        content_length = 10 if text == "source" else int(text) + 1
        return [101, *range(content_length), 102]

    def decode(self, token_ids: list[int]) -> str:
        return str(len(token_ids))


def test_split_text_respects_limit_after_decode_encode_round_trip() -> None:
    splitter = TransformerTokenChunker.__new__(TransformerTokenChunker)
    splitter.tokenizer = RoundTripGrowingTokenizer()
    splitter.tokens_per_chunk = 4
    splitter._chunk_overlap = 1

    chunks = splitter.split_text("source")

    assert chunks
    assert all(splitter.count_tokens(text=chunk) <= 4 for chunk in chunks)