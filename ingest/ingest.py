"""Standalone ingest entrypoint: corpus JSON -> chunks -> embeddings -> Pinecone.

Run once before starting the API. Re-run if the corpus changes
(upserts are idempotent per deterministic vector id).

Usage: python ingest/ingest.py
"""

import sys
import time
from pathlib import Path

from pinecone import Pinecone, ServerlessSpec

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from api.config import settings
from ingest.corpus_loader import load_chunks_with_ids
from ingest.embedder import Embedder

UPSERT_BATCH_SIZE = 100


def ensure_index(pc: Pinecone, index_name: str):
    existing = {idx["name"] for idx in pc.list_indexes()}
    if index_name not in existing:
        print(f"Creating Pinecone index '{index_name}' ...")
        pc.create_index(
            name=index_name,
            dimension=settings.embedding_dim,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        while not pc.describe_index(index_name).status["ready"]:
            time.sleep(1)
    return pc.Index(index_name)


def main():
    start = time.time()

    print("Loading and chunking corpus ...")
    chunks = load_chunks_with_ids()
    print(f"  {len(chunks)} chunks from corpus files")

    pc = Pinecone(api_key=settings.pinecone_api_key)
    index = ensure_index(pc, settings.pinecone_index_name)

    embedder = Embedder(model=settings.embedding_model)

    total_docs = len({c["source"] + str(c.get("doc_title")) for c in chunks})

    for batch_start in range(0, len(chunks), UPSERT_BATCH_SIZE):
        batch = chunks[batch_start : batch_start + UPSERT_BATCH_SIZE]
        texts = [c["content"] for c in batch]
        vectors = embedder.embed_batch(texts)

        upsert_payload = []
        for chunk, vector in zip(batch, vectors):
            upsert_payload.append(
                {
                    "id": chunk["id"],
                    "values": vector,
                    "metadata": {
                        "content": chunk["content"],
                        "source": chunk["source"],
                        "doc_title": chunk["doc_title"],
                        "chunk_index": chunk["chunk_index"],
                    },
                }
            )

        index.upsert(vectors=upsert_payload)
        print(f"  upserted {min(batch_start + UPSERT_BATCH_SIZE, len(chunks))}/{len(chunks)}")

    elapsed = time.time() - start
    print(f"\nDone in {elapsed:.1f}s")
    print(f"Docs processed (approx): {total_docs}")
    print(f"Chunks embedded + upserted: {len(chunks)}")


if __name__ == "__main__":
    main()
