"""Local sentence-transformers embedding wrapper (free, runs on CPU/MPS).

Swaps in for OpenAI embeddings when no OpenAI key is available. Default
model BAAI/bge-small-en-v1.5 outputs 384-dim vectors.
"""

from sentence_transformers import SentenceTransformer

BATCH_SIZE = 64


class Embedder:
    def __init__(self, api_key: str | None = None, model: str = "BAAI/bge-small-en-v1.5"):
        # api_key kept for interface parity with the OpenAI-backed version; unused here.
        self.model = SentenceTransformer(model)

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(texts, batch_size=BATCH_SIZE, show_progress_bar=False).tolist()

    def embed_all(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(texts, batch_size=BATCH_SIZE, show_progress_bar=True).tolist()
