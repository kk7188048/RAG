# Healthcare RAG

A medical Q&A system: user question in → relevant MedQuAD context retrieved from Pinecone → grounded LLM answer out, with sources, latency, and token usage. Built in layers, each measured against the last with `ragas` + MLflow. Prometheus/Grafana instrumented for live monitoring.

**Status: Phases 1-3 complete** (naive RAG, advanced retrieval + CRAG, observability). CI/CD and full polish (Phases 5-6) not started.

## Eval comparison

Golden set sampled from the real MedQuAD corpus, `v1_basic` prompt, `top_k=5`. Sample sizes vary (54 for the locked baseline, 15 for later strategies — kept smaller since the judge LLM is slow and this project is for learning/portfolio purposes, not chasing statistically tight numbers).

| Strategy | n | Faithfulness | Context Recall | Answer Relevancy | p50 Latency | p95 Latency |
|---|---|---|---|---|---|---|
| Vector (baseline) | 54 | 0.750 | 1.000 | 0.804 | — | — |
| Hybrid (BM25 + Vector + RRF) | 15 | 1.000 | 1.000 | 0.873 | 26.1s | 53.5s |
| Hybrid + Rerank | 15 | 1.000 | 1.000 | 0.905 | 103.5s | 124.2s |
| CRAG | — | not formally scored | | | ~118s (single live test) | |
| HyDE | — | not formally scored — confirmed live that it retrieves on-topic docs for vague questions | | | | |

Logged in MLflow (`mlflow ui --backend-store-uri sqlite:///mlflow.db`), experiment `healthcare-rag`.

**Read these numbers as directional, not tight.** The judge LLM (`deepseek-ai/deepseek-v4-flash-0731`, a reasoning model via NVIDIA NIM's free tier) is slow and occasionally times out mid-score even with retries bounded — a meaningful fraction of judge calls fail and get excluded from the average. CRAG and HyDE were verified to work correctly via live manual testing (see below) but weren't run through the formal ragas scoring loop, since each full pass takes 15-90+ min on this model and diminishing returns set in fast for a learning project.

**Reranking made latency much worse, not better.** The plan expected ~200-400ms overhead from the cross-encoder; instead p50 latency roughly 4x'd (26s → 103s). That's because `BAAI/bge-reranker-v2-m3` runs on CPU here (no GPU), not because reranking is inherently slow — worth noting as a real cost/quality tradeoff if this were a production system.

## Manually verified (not in the eval table)

- **CRAG**: graded 10 candidate docs in parallel, found them relevant, generated a correct answer (`answer_source: "corpus"`), ~118s end-to-end.
- **HyDE**: for the vague query "chest pain after eating," retrieved genuinely on-topic docs (Angina, Coronary Heart Disease, Pericarditis) that plain vector search on the raw question would likely miss.
- **Prometheus**: `/metrics` endpoint confirmed live, all custom metrics (`rag_requests_total`, `rag_latency_seconds`, `rag_tokens_used`, `rerank_top1_score`, `crag_web_fallbacks_total`) populate correctly across strategies.

## Stack (as actually deployed, differs from the original plan in a few places)

| What | Plan | Actually used | Why |
|---|---|---|---|
| Embeddings | OpenAI `text-embedding-3-small` | Local `BAAI/bge-small-en-v1.5` (384-dim) | No OpenAI key; this is free and runs on CPU |
| LLM | Groq `llama-3.3-70b-versatile` | NVIDIA NIM `deepseek-ai/deepseek-v4-flash-0731` | Model retired on Groq; Groq's free-tier daily token quota (200k/model) was exhausted mid-eval by retries; NVIDIA NIM's free tier has no daily token cap (just a shared ~40 RPM rate limit). Most other NIM models on this key returned 404/410 or were also slow reasoning models — this was the most reliable option found |
| Reranker | `BAAI/bge-reranker-v2-m3` | same | as planned, runs on CPU (no GPU available) |
| Vector DB | Pinecone | Pinecone | as planned |
| Web fallback (CRAG) | Tavily | Not configured (no key) | `TAVILY_API_KEY` optional in `.env`; CRAG gracefully returns best-effort corpus results instead of failing when unset |

## Repo layout

- `ingest/` — corpus download (`scripts/seed_corpus.py`), chunking, embedding, Pinecone upsert (`ingest.py`), shared chunk/id loader (`corpus_loader.py`)
- `api/pipeline/` — retriever (vector + hybrid/RRF), reranker, query_rewriter (HyDE), crag, generator, bm25_index
- `api/monitoring/` — Prometheus metric definitions
- `api/routes/` — `/query` (dispatches by `retrieval_strategy`), `/health`, `/metrics`
- `eval/` — golden set, `run_eval.py` (ragas + MLflow, all 4 strategies), thresholds
- `prompts/` — prompt templates
- `docker/` — Dockerfile + compose (api + mlflow + prometheus)
- `monitoring/` — `prometheus.yml` scrape config, `grafana/dashboard.json` (5 panels: p50/p95 latency by strategy, requests/min, token cost/day, CRAG fallback rate, reranker score distribution)

## Setup

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in PINECONE_API_KEY, NVIDIA_API_KEY (TAVILY_API_KEY optional)
```

## Run

```bash
# one-time: pull MedQuAD, chunk, embed, upsert to Pinecone (~18k chunks, ~7 min)
python scripts/seed_corpus.py
python ingest/ingest.py

# start the API (loads embedding model + reranker + BM25 index at startup)
uvicorn api.main:app --reload

# smoke test
python scripts/test_query.py "What are the symptoms of diabetes?"

# try other strategies directly
curl -X POST localhost:8000/query -H "Content-Type: application/json" \
  -d '{"question":"What are the symptoms of diabetes?","retrieval_strategy":"hybrid_rerank"}'
# retrieval_strategy: vector | hybrid | hybrid_rerank | crag
# use_hyde: true/false (works with any strategy)

# metrics
curl localhost:8000/metrics

# eval (slow — this LLM is slow; full 54-question run took 1-7.5hrs depending on judge timeout behavior)
python eval/run_eval.py --strategy hybrid_rerank --max-samples 15
python eval/run_eval.py --strategy vector --fail-threshold   # full 54-question CI-style run
```

Generated eval answers are cached per-question in `eval/.cache/` (resumable — a failed/interrupted run picks up where it left off instead of re-generating from scratch).

## Docker

```bash
docker compose -f docker/docker-compose.yml up
```

Runs api + mlflow + prometheus. Grafana isn't included in compose (bring your own / Grafana Cloud) — point it at the prometheus service and import `monitoring/grafana/dashboard.json`.

## Next

Phase 5 (CI/CD eval gates) and Phase 6 (polish: architecture diagram, live demo link, screenshots) not started.
