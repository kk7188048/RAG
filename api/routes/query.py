import time

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from api.dependencies import get_crag, get_generator, get_hybrid_retriever, get_query_rewriter, get_reranker, get_retriever
from api.monitoring.metrics import (
    crag_web_fallbacks_total,
    rag_errors_total,
    rag_latency_seconds,
    rag_requests_total,
    rag_tokens_used,
    rerank_top1_score,
)
from api.pipeline.crag import CRAG
from api.pipeline.generator import Generator
from api.pipeline.query_rewriter import QueryRewriter
from api.pipeline.reranker import Reranker
from api.pipeline.retriever import HybridRetriever, Retriever

router = APIRouter()

STRATEGIES = {"vector", "hybrid", "hybrid_rerank", "crag"}


class QueryRequest(BaseModel):
    question: str
    top_k: int = 5
    retrieval_strategy: str = "vector"
    prompt_version: str = "v1_basic"
    use_hyde: bool = False


class SourceDoc(BaseModel):
    content: str
    source: str
    doc_title: str
    score: float
    rerank_score: float | None = None


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceDoc]
    latency_ms: float
    tokens_used: int
    retrieval_strategy: str
    prompt_version: str
    answer_source: str = "corpus"


@router.post("/query", response_model=QueryResponse)
def query(
    request: QueryRequest,
    retriever: Retriever = Depends(get_retriever),
    hybrid_retriever: HybridRetriever = Depends(get_hybrid_retriever),
    reranker: Reranker = Depends(get_reranker),
    generator: Generator = Depends(get_generator),
    query_rewriter: QueryRewriter = Depends(get_query_rewriter),
    crag: CRAG = Depends(get_crag),
):
    start = time.perf_counter()
    strategy = request.retrieval_strategy if request.retrieval_strategy in STRATEGIES else "vector"
    answer_source = "corpus"

    try:
        embed_text = query_rewriter.hyde_expand(request.question) if request.use_hyde else request.question

        if strategy == "vector":
            docs = retriever.vector_search(embed_text, top_n=20)[: request.top_k]
        elif strategy == "hybrid":
            docs = hybrid_retriever.retrieve(request.question, top_k=request.top_k, query_text_for_embedding=embed_text)
        elif strategy == "hybrid_rerank":
            candidates = hybrid_retriever.retrieve(request.question, top_k=20, query_text_for_embedding=embed_text)
            docs = reranker.rerank(request.question, candidates, top_k=request.top_k)
            if docs:
                rerank_top1_score.observe(docs[0]["rerank_score"])
        else:  # crag
            def base_retrieve(q: str) -> list[dict]:
                candidates = hybrid_retriever.retrieve(q, top_k=20)
                return reranker.rerank(q, candidates, top_k=10)

            docs, answer_source = crag.retrieve_with_correction(request.question, base_retrieve, top_k=request.top_k)
            if answer_source == "web_fallback":
                crag_web_fallbacks_total.inc()

        result = generator.generate(request.question, docs, prompt_version=request.prompt_version)
    except Exception as e:
        rag_errors_total.labels(error_type=type(e).__name__).inc()
        raise

    latency_ms = (time.perf_counter() - start) * 1000

    rag_requests_total.labels(strategy=strategy).inc()
    rag_latency_seconds.labels(strategy=strategy).observe(latency_ms / 1000)
    rag_tokens_used.observe(result["tokens_used"])

    return QueryResponse(
        answer=result["answer"],
        sources=[SourceDoc(**doc) for doc in docs],
        latency_ms=round(latency_ms, 1),
        tokens_used=result["tokens_used"],
        retrieval_strategy=strategy,
        prompt_version=result["prompt_version"],
        answer_source=answer_source,
    )
