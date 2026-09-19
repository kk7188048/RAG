"""Quick CLI smoke test against a running API instance.

Usage: python scripts/test_query.py "What are the symptoms of diabetes?"
"""

import sys

import httpx

API_URL = "http://localhost:8000/query"


def main():
    question = " ".join(sys.argv[1:]) or "What are the symptoms of diabetes?"
    response = httpx.post(API_URL, json={"question": question}, timeout=150)
    response.raise_for_status()
    data = response.json()

    print(f"Q: {question}\n")
    print(f"A: {data['answer']}\n")
    print(f"Latency: {data['latency_ms']}ms | Tokens: {data['tokens_used']}")
    print(f"\nSources ({len(data['sources'])}):")
    for src in data["sources"]:
        print(f"  - [{src['score']:.3f}] {src['doc_title']} ({src['source']})")


if __name__ == "__main__":
    main()
