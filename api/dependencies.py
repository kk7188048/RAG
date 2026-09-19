"""Shared, process-wide singletons built once at startup."""

from api.config import settings
from api.pipeline.bm25_index import BM25Index
from api.pipeline.crag import CRAG
from api.pipeline.generator import Generator
from api.pipeline.query_rewriter import QueryRewriter
from api.pipeline.reranker import Reranker
from api.pipeline.retriever import HybridRetriever, Retriever

retriever = Retriever(
    pinecone_api_key=settings.pinecone_api_key,
    index_name=settings.pinecone_index_name,
    embedding_model=settings.embedding_model,
)

bm25_index = BM25Index()
hybrid_retriever = HybridRetriever(retriever, bm25_index)

reranker = Reranker(model=settings.reranker_model)

generator = Generator(api_key=settings.nvidia_api_key, model=settings.llm_model, base_url=settings.nvidia_base_url)

query_rewriter = QueryRewriter(api_key=settings.nvidia_api_key, model=settings.llm_model, base_url=settings.nvidia_base_url)

crag = CRAG(
    api_key=settings.nvidia_api_key,
    model=settings.llm_model,
    base_url=settings.nvidia_base_url,
    query_rewriter=query_rewriter,
    tavily_api_key=settings.tavily_api_key,
)


def get_retriever() -> Retriever:
    return retriever


def get_hybrid_retriever() -> HybridRetriever:
    return hybrid_retriever


def get_reranker() -> Reranker:
    return reranker


def get_generator() -> Generator:
    return generator


def get_query_rewriter() -> QueryRewriter:
    return query_rewriter


def get_crag() -> CRAG:
    return crag
