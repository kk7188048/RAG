"""Fixed-size word chunking with overlap for MedQuAD records."""

CHUNK_SIZE_WORDS = 512
OVERLAP_WORDS = 50


def chunk_record(record: dict, source_file: str) -> list[dict]:
    """Formats one MedQuAD Q&A record as 'Q: ...\\nA: ...' and splits it
    into overlapping fixed-size word chunks."""
    text = f"Q: {record['question']}\nA: {record['answer']}"
    words = text.split()

    if len(words) <= CHUNK_SIZE_WORDS:
        spans = [words]
    else:
        spans = []
        step = CHUNK_SIZE_WORDS - OVERLAP_WORDS
        start = 0
        while start < len(words):
            spans.append(words[start : start + CHUNK_SIZE_WORDS])
            if start + CHUNK_SIZE_WORDS >= len(words):
                break
            start += step

    chunks = []
    for idx, span in enumerate(spans):
        chunks.append(
            {
                "content": " ".join(span),
                "source": source_file,
                "doc_title": record["focus_area"],
                "chunk_index": idx,
            }
        )
    return chunks


def chunk_all(records: list[dict], source_file: str) -> list[dict]:
    chunks = []
    for record in records:
        chunks.extend(chunk_record(record, source_file))
    return chunks
