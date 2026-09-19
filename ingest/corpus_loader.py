"""Loads and chunks the corpus with the same deterministic IDs used at
Pinecone upsert time. Shared by ingest.py (upsert) and the BM25 index
(local keyword search), so hybrid search's RRF merge can match vector and
keyword results by identical id.
"""

import json
from pathlib import Path

from ingest.chunker import chunk_all

CORPUS_DIR = Path(__file__).resolve().parent / "corpus"


def load_chunks_with_ids() -> list[dict]:
    all_chunks = []
    for json_file in sorted(CORPUS_DIR.glob("*.json")):
        records = json.loads(json_file.read_text())
        all_chunks.extend(chunk_all(records, source_file=json_file.stem))

    for i, chunk in enumerate(all_chunks):
        chunk["id"] = f"{chunk['source']}-{chunk['doc_title']}-{chunk['chunk_index']}-{i}"

    return all_chunks
