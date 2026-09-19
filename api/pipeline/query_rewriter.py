"""HyDE (Hypothetical Document Embeddings): a short question embeds poorly
against long reference-style answers, so generate a hypothetical answer
first and embed that instead -- it "looks like" the kind of document we
want to retrieve. The hypothetical text is only ever used as a search
proxy, never shown to the user or fed to the final generator."""

from openai import OpenAI

HYDE_PROMPT = (
    "You are a medical professional. Write a 2-3 sentence factual answer to "
    "this question as if writing for a medical reference. Question: {question}"
)
MAX_TOKENS = 512


class QueryRewriter:
    def __init__(self, api_key: str, model: str, base_url: str):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model

    def hyde_expand(self, question: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": HYDE_PROMPT.format(question=question)}],
            temperature=0.3,
            max_tokens=MAX_TOKENS,
        )
        return response.choices[0].message.content or question
