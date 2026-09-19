"""Runs the golden set through the RAG pipeline, scores it with ragas,
logs everything to MLflow, and exits non-zero if any metric misses
the threshold in eval_config.yaml (used by CI to block merges).

Usage: python eval/run_eval.py [--strategy vector|hybrid|hybrid_rerank|crag] [--prompt-version v1_basic] [--top-k 5]
"""

import argparse
import json
import sys
import time
from pathlib import Path

import mlflow
import numpy as np
import yaml
from datasets import Dataset
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from ragas import evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.run_config import RunConfig
from ragas.metrics import answer_relevancy, context_recall, faithfulness

# Keep judge-LLM calls to a single generation per sample (cheaper, and
# avoids relying on n>1 support from whatever model is configured).
answer_relevancy.strictness = 1

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from api.config import settings
from api.dependencies import crag, generator, hybrid_retriever, query_rewriter, reranker, retriever

EVAL_DIR = Path(__file__).resolve().parent
GOLDEN_SET_PATH = EVAL_DIR / "golden_set.json"
CONFIG_PATH = EVAL_DIR / "eval_config.yaml"
CACHE_DIR = EVAL_DIR / ".cache"

STRATEGIES = {"vector", "hybrid", "hybrid_rerank", "crag"}


def load_golden_set() -> list[dict]:
    return json.loads(GOLDEN_SET_PATH.read_text())


def load_thresholds() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text())["thresholds"]


def build_retrieve_fn(strategy: str, top_k: int, use_hyde: bool):
    """Returns question -> list[docs], matching how api/routes/query.py
    dispatches strategies, so eval numbers reflect what the API actually does."""

    def embed_text(question: str) -> str:
        return query_rewriter.hyde_expand(question) if use_hyde else question

    if strategy == "vector":
        return lambda q: retriever.vector_search(embed_text(q), top_n=20)[:top_k]
    if strategy == "hybrid":
        return lambda q: hybrid_retriever.retrieve(q, top_k=top_k, query_text_for_embedding=embed_text(q))
    if strategy == "hybrid_rerank":
        def hybrid_rerank_fn(q):
            candidates = hybrid_retriever.retrieve(q, top_k=20, query_text_for_embedding=embed_text(q))
            return reranker.rerank(q, candidates, top_k=top_k)

        return hybrid_rerank_fn

    def crag_fn(q):
        def base_retrieve(query_text: str) -> list[dict]:
            candidates = hybrid_retriever.retrieve(query_text, top_k=20)
            return reranker.rerank(query_text, candidates, top_k=10)

        docs, _source = crag.retrieve_with_correction(q, base_retrieve, top_k=top_k)
        return docs

    return crag_fn


def run_pipeline(
    golden_set: list[dict],
    retrieve_fn,
    top_k: int,
    prompt_version: str,
    strategy: str,
    model_name: str,
    use_hyde: bool,
    use_cache: bool,
) -> tuple[Dataset, list[float]]:
    # Generation burns rate-limited LLM budget just like judging does, so a
    # transient failure during ragas scoring shouldn't force re-generating
    # every answer from scratch on the next attempt. Cache key includes the
    # model name so switching models can't silently reuse stale answers.
    safe_model = model_name.replace("/", "_")
    hyde_tag = "_hyde" if use_hyde else ""
    cache_path = CACHE_DIR / f"dataset_{strategy}{hyde_tag}_{prompt_version}_top{top_k}_n{len(golden_set)}_{safe_model}.json"

    CACHE_DIR.mkdir(exist_ok=True)

    records: list[dict] = []
    if use_cache and cache_path.exists():
        records = json.loads(cache_path.read_text())
        print(f"Resuming from {len(records)}/{len(golden_set)} cached answers in {cache_path.name}")

    for i in range(len(records), len(golden_set)):
        item = golden_set[i]
        t0 = time.perf_counter()
        docs = retrieve_fn(item["question"])
        result = generator.generate(item["question"], docs, prompt_version=prompt_version)
        latency_ms = (time.perf_counter() - t0) * 1000

        records.append(
            {
                "question": item["question"],
                "answer": result["answer"],
                "contexts": [d["content"] for d in docs],
                "ground_truth": item["ground_truth"],
                "latency_ms": latency_ms,
            }
        )
        cache_path.write_text(json.dumps(records))
        print(f"  generated {i + 1}/{len(golden_set)}  ({latency_ms/1000:.1f}s)")

    data = {
        "question": [r["question"] for r in records],
        "answer": [r["answer"] for r in records],
        "contexts": [r["contexts"] for r in records],
        "ground_truth": [r["ground_truth"] for r in records],
    }
    latencies = [r["latency_ms"] for r in records]
    return Dataset.from_dict(data), latencies


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", default="vector", choices=sorted(STRATEGIES))
    parser.add_argument("--prompt-version", default="v1_basic")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--use-hyde", action="store_true")
    parser.add_argument("--fail-threshold", action="store_true", help="exit 1 if any metric misses threshold (for CI)")
    parser.add_argument("--max-samples", type=int, default=None, help="limit golden set size (useful to conserve rate-limited LLM budget)")
    parser.add_argument("--regenerate", action="store_true", help="ignore cached generated answers and re-run generation")
    args = parser.parse_args()

    golden_set = load_golden_set()
    if args.max_samples:
        golden_set = golden_set[: args.max_samples]
    thresholds = load_thresholds()

    print(f"Running eval: strategy={args.strategy}, use_hyde={args.use_hyde}, prompt_version={args.prompt_version}, top_k={args.top_k}")
    print(f"Golden set size: {len(golden_set)}")

    retrieve_fn = build_retrieve_fn(args.strategy, args.top_k, args.use_hyde)

    dataset, latencies = run_pipeline(
        golden_set,
        retrieve_fn,
        args.top_k,
        args.prompt_version,
        args.strategy,
        settings.llm_model,
        args.use_hyde,
        use_cache=not args.regenerate,
    )

    print(f"Scoring with ragas (judge model: {settings.llm_model} via NVIDIA NIM, local HF embeddings) ...")
    judge_llm = LangchainLLMWrapper(
        ChatOpenAI(
            api_key=settings.nvidia_api_key,
            base_url=settings.nvidia_base_url,
            model=settings.llm_model,
            max_tokens=2048,
            timeout=90,
            max_retries=0,  # ragas' own RunConfig retries on top of this; stacking both caused multi-hour hangs
        )
    )
    judge_embeddings = LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(model_name=settings.embedding_model))
    # NVIDIA NIM enforces a single ~40 RPM cap shared across all calls on
    # this key, so keep judge-LLM concurrency modest. Bounded retries: a
    # stuck NVIDIA connection combined with ragas' default max_retries=10 and
    # exponential backoff (up to max_wait=60s each) let a handful of jobs
    # hang for OVER 7 HOURS in a prior run. 90s timeout x 2 retries bounds
    # the worst case per job to a few minutes instead.
    result = evaluate(
        dataset,
        metrics=[faithfulness, context_recall, answer_relevancy],
        llm=judge_llm,
        embeddings=judge_embeddings,
        run_config=RunConfig(max_workers=5, timeout=200, max_retries=2, max_wait=15),
    )
    # result[metric] is a per-sample list; average with nanmean since ragas
    # can emit NaN for individual samples (e.g. empty context).
    scores = {
        "faithfulness": float(np.nanmean(result["faithfulness"])),
        "context_recall": float(np.nanmean(result["context_recall"])),
        "answer_relevancy": float(np.nanmean(result["answer_relevancy"])),
    }
    p95_latency_ms = float(np.percentile(latencies, 95)) if latencies else 0.0
    p50_latency_ms = float(np.percentile(latencies, 50)) if latencies else 0.0

    mlflow.set_experiment("healthcare-rag")
    with mlflow.start_run():
        mlflow.log_params(
            {
                "retrieval_strategy": args.strategy,
                "use_hyde": args.use_hyde,
                "prompt_version": args.prompt_version,
                "chunk_size": 512,
                "overlap": 50,
                "top_k": args.top_k,
                "embedding_model": settings.embedding_model,
                "llm_model": settings.llm_model,
                "n_eval_samples": len(golden_set),
            }
        )
        mlflow.log_metrics({**scores, "p50_latency_ms": p50_latency_ms, "p95_latency_ms": p95_latency_ms})

    print("\nResults:")
    all_passed = True
    for metric, value in scores.items():
        threshold = thresholds[metric]
        passed = value >= threshold
        all_passed &= passed
        status = "PASS" if passed else "FAIL"
        print(f"  {metric:<18} {value:.3f}  (threshold {threshold})  [{status}]")
    print(f"  {'p50 latency':<18} {p50_latency_ms:.0f}ms")
    print(f"  {'p95 latency':<18} {p95_latency_ms:.0f}ms")

    if args.fail_threshold and not all_passed:
        print("\nEval failed: one or more metrics below threshold.")
        sys.exit(1)

    print("\nDone.")


if __name__ == "__main__":
    main()
