"""Assembles the prompt from retrieved context and calls the LLM (NVIDIA NIM,
OpenAI-compatible API)."""

from pathlib import Path

from openai import OpenAI

PROMPTS_DIR = Path(__file__).resolve().parent.parent.parent / "prompts"
MAX_TOKENS = 1024  # generous headroom: reasoning models spend tokens on a hidden chain-of-thought before the final answer


def _load_prompt(prompt_version: str) -> str:
    path = PROMPTS_DIR / f"{prompt_version}.txt"
    return path.read_text()


def _format_context(docs: list[dict]) -> str:
    blocks = []
    for i, doc in enumerate(docs, start=1):
        blocks.append(f"[Source {i}: {doc['doc_title']}]\n{doc['content']}")
    return "\n\n".join(blocks)


class Generator:
    def __init__(self, api_key: str, model: str, base_url: str = "https://integrate.api.nvidia.com/v1"):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model

    def generate(self, question: str, docs: list[dict], prompt_version: str = "v1_basic") -> dict:
        template = _load_prompt(prompt_version)
        context = _format_context(docs)
        prompt = template.replace("{context}", context).replace("{question}", question)

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=MAX_TOKENS,
        )

        message = response.choices[0].message
        answer = message.content or "I don't have enough information to answer that."

        return {
            "answer": answer,
            "tokens_used": response.usage.total_tokens,
            "prompt_version": prompt_version,
        }
