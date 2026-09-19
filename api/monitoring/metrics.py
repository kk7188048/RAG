"""Prometheus metric definitions, scraped from GET /metrics."""

from prometheus_client import Counter, Histogram

rag_requests_total = Counter(
    "rag_requests_total", "Total RAG requests", ["strategy"]
)

crag_web_fallbacks_total = Counter(
    "crag_web_fallbacks_total", "Times CRAG fell back to web search instead of the corpus"
)

rag_errors_total = Counter(
    "rag_errors_total", "Total errors during request handling", ["error_type"]
)

rag_latency_seconds = Histogram(
    "rag_latency_seconds", "End-to-end request latency", ["strategy"]
)

rag_tokens_used = Histogram(
    "rag_tokens_used", "LLM tokens used per request", buckets=(100, 250, 500, 1000, 2000, 4000, 8000)
)

rerank_top1_score = Histogram(
    "rerank_top1_score", "Cross-encoder score of the top reranked document", buckets=(-5, -2, 0, 2, 5, 8, 10)
)
