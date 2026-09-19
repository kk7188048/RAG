"""Cross-encoder reranking: scores each (query, doc) pair jointly, which is
more accurate than comparing independent embeddings but too slow to run
over the whole corpus -- so it only re-scores the top candidates a cheaper
retriever already narrowed down."""

from sentence_transformers import CrossEncoder


class Reranker:
    def __init__(self, model: str = "BAAI/bge-reranker-v2-m3"):
        self.model = CrossEncoder(model)

    def rerank(self, query: str, docs: list[dict], top_k: int = 5) -> list[dict]:
        if not docs:
            return docs

        pairs = [(query, doc["content"]) for doc in docs]
        scores = self.model.predict(pairs)

        reranked = sorted(zip(docs, scores), key=lambda pair: pair[1], reverse=True)[:top_k]

        return [{**doc, "rerank_score": float(score)} for doc, score in reranked]
