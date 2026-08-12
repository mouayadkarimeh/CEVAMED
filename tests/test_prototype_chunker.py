from src.chunking import Chunk, chunk_text_prototype


def test_prototype_chunker_splits_long_text_into_multiple_chunks():
    text = (
        "Patient presents with fever and mild cough. "
        "Symptoms started three days ago and have increased. "
        "The patient reports fatigue, headache, and intermittent chest pain."
    )

    chunks = chunk_text_prototype(text, max_chunk_size=60)

    assert chunks
    assert len(chunks) >= 2
    assert all(isinstance(chunk, Chunk) for chunk in chunks)
    assert all(chunk.text for chunk in chunks)
    assert all(chunk.metadata.get("source") == "prototype" for chunk in chunks)
