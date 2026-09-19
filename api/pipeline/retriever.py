"""Retrieval strategies: Phase 1 vector-only, Phase 2 hybrid (BM25 + vector + RRF)."""

from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

from api.pipeline.bm25_index import BM25Index

TOP_N_FETCH = 20
RRF_K = 60


class Retriever:
    def __init__(self, pinecone_api_key: str, index_name: str, embedding_model: str):
        self.index = Pinecone(api_key=pinecone_api_key).Index(index_name)
        self.embedding_model = SentenceTransformer(embedding_model)

    def _embed(self, text: str) -> list[float]:
        return self.embedding_model.encode([text])[0].tolist()

    def vector_search(self, query_text: str, top_n: int = TOP_N_FETCH) -> list[dict]:
        """query_text is what gets embedded -- the raw question normally,
        or a HyDE hypothetical answer when the caller wants that instead."""
        query_vector = self._embed(query_text)
        result = self.index.query(vector=query_vector, top_k=top_n, include_metadata=True)
        return [
            {
                "id": match["id"],
                "content": match["metadata"]["content"],
                "source": match["metadata"]["source"],
                "doc_title": match["metadata"]["doc_title"],
                "score": match["score"],
            }
            for match in result.get("matches", [])
        ]

    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        return self.vector_search(query, top_n=TOP_N_FETCH)[:top_k]


class HybridRetriever:
    """Vector search + BM25 keyword search, merged with Reciprocal Rank Fusion."""

    def __init__(self, retriever: Retriever, bm25_index: BM25Index):
        self.retriever = retriever
        self.bm25_index = bm25_index

    def retrieve(self, query: str, top_k: int = 5, query_text_for_embedding: str | None = None) -> list[dict]:
        vector_results = self.retriever.vector_search(query_text_for_embedding or query, top_n=TOP_N_FETCH)
        bm25_results = self.bm25_index.search(query, top_n=TOP_N_FETCH)

        rrf_scores: dict[str, float] = {}
        chunk_by_id: dict[str, dict] = {}
        for ranked_list in (vector_results, bm25_results):
            for rank, chunk in enumerate(ranked_list):
                rrf_scores[chunk["id"]] = rrf_scores.get(chunk["id"], 0.0) + 1.0 / (RRF_K + rank)
                chunk_by_id[chunk["id"]] = chunk

        merged_ids = sorted(rrf_scores, key=lambda cid: rrf_scores[cid], reverse=True)[:TOP_N_FETCH]

        return [
            {
                "id": cid,
                "content": chunk_by_id[cid]["content"],
                "source": chunk_by_id[cid]["source"],
                "doc_title": chunk_by_id[cid]["doc_title"],
                "score": rrf_scores[cid],
            }
            for cid in merged_ids
        ][:top_k]
