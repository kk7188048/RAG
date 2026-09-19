"""Corrective RAG: grade retrieved docs for relevance before generating.
Falls back to web search if the corpus doesn't cover the question.

Decision logic:
  - any doc graded RELEVANT      -> keep only the RELEVANT ones, generate
  - none RELEVANT, some AMBIGUOUS -> HyDE-rewrite the query, re-retrieve, re-grade
  - still nothing RELEVANT        -> fall back to Tavily web search (if configured)
"""

from concurrent.futures import ThreadPoolExecutor
from typing import Callable

from openai import OpenAI

GRADE_PROMPT = (
    "Question: {question}\n\nDocument:\n{content}\n\n"
    "Is this document relevant to answering the question? "
    "Reply with exactly one word: RELEVANT, IRRELEVANT, or AMBIGUOUS."
)
MAX_TOKENS = 16


class CRAG:
    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str,
        query_rewriter,
        tavily_api_key: str | None = None,
    ):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.query_rewriter = query_rewriter
        self.tavily_api_key = tavily_api_key

    def _grade_one(self, question: str, doc: dict) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": GRADE_PROMPT.format(question=question, content=doc["content"])}],
                temperature=0.0,
                max_tokens=MAX_TOKENS,
            )
            text = (response.choices[0].message.content or "").strip().upper()
        except Exception:
            return "AMBIGUOUS"

        if "IRRELEVANT" in text:
            return "IRRELEVANT"
        if "RELEVANT" in text:
            return "RELEVANT"
        return "AMBIGUOUS"

    def _grade_all(self, question: str, docs: list[dict]) -> list[str]:
        if not docs:
            return []
        with ThreadPoolExecutor(max_workers=len(docs)) as pool:
            return list(pool.map(lambda d: self._grade_one(question, d), docs))

    def _web_search(self, question: str) -> list[dict]:
        if not self.tavily_api_key:
            return []
        try:
            from tavily import TavilyClient

            client = TavilyClient(api_key=self.tavily_api_key)
            results = client.search(question, max_results=5)
            return [
                {
                    "id": f"web-{i}",
                    "content": r.get("content", ""),
                    "source": "web_fallback",
                    "doc_title": r.get("title", "web result"),
                    "score": r.get("score", 0.0),
                }
                for i, r in enumerate(results.get("results", []))
            ]
        except Exception:
            return []

    def retrieve_with_correction(
        self, question: str, retrieve_fn: Callable[[str], list[dict]], top_k: int = 5
    ) -> tuple[list[dict], str]:
        """retrieve_fn(query_text) -> docs, using whatever base retrieval
        strategy (vector / hybrid / hybrid+rerank) the caller configured.
        Returns (docs, source_tag) where source_tag is 'corpus' or 'web_fallback'.
        """
        docs = retrieve_fn(question)
        grades = self._grade_all(question, docs)

        relevant = [d for d, g in zip(docs, grades) if g == "RELEVANT"]
        if relevant:
            return relevant[:top_k], "corpus"

        if "AMBIGUOUS" in grades:
            hyde_query = self.query_rewriter.hyde_expand(question)
            docs = retrieve_fn(hyde_query)
            grades = self._grade_all(question, docs)
            relevant = [d for d, g in zip(docs, grades) if g == "RELEVANT"]
            if relevant:
                return relevant[:top_k], "corpus"

        web_docs = self._web_search(question)
        if web_docs:
            return web_docs[:top_k], "web_fallback"

        # No relevant corpus docs and no web fallback available -- return
        # best-effort original results rather than nothing.
        return docs[:top_k], "corpus"
