"""In-memory BM25 keyword index over the same chunks stored in Pinecone.

Built once at API startup from the corpus on disk (not from Pinecone) --
much cheaper than paginating the whole vector index, and produces the exact
same chunk ids since both use ingest.corpus_loader.
"""

from rank_bm25 import BM25Okapi

from ingest.corpus_loader import load_chunks_with_ids


class BM25Index:
    def __init__(self):
        self.chunks = load_chunks_with_ids()
        tokenized = [c["content"].lower().split() for c in self.chunks]
        self.bm25 = BM25Okapi(tokenized)

    def search(self, query: str, top_n: int = 20) -> list[dict]:
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_n]

        results = []
        for idx in ranked_indices:
            chunk = self.chunks[idx]
            results.append(
                {
                    "id": chunk["id"],
                    "content": chunk["content"],
                    "source": chunk["source"],
                    "doc_title": chunk["doc_title"],
                    "score": float(scores[idx]),
                }
            )
        return results
