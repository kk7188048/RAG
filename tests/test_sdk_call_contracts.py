"""Guard the SDK calls that must survive the Pinecone 10 / OpenAI 3 updates.

Pinecone 10 changed only gRPC ``upsert_from_dataframe`` (partial failures are
now returned instead of raised); the REST ``Index.upsert`` / ``Index.query``
calls this repo uses keep the same arguments and response shape. OpenAI removed
the ``spend_control`` request property and some beta endpoints; this repo only
calls ``chat.completions.create``.

The SDKs are replaced with fakes so this test runs offline and without
credentials. Inputs are synthetic.
"""

import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class _Encoded(list):
    def tolist(self):
        return list(self)


class _FakeSentenceTransformer:
    def __init__(self, model):
        self.model = model

    def encode(self, texts, **kwargs):
        return [_Encoded([0.25, 0.5]) for _ in texts]


class _FakeQueryResponse:
    def __init__(self, matches):
        self.matches = matches

    def get(self, key, default=None):
        return self.matches if key == "matches" else default


class _FakeIndex:
    def __init__(self):
        self.query_calls = []
        self.upsert_calls = []

    def query(self, **kwargs):
        self.query_calls.append(kwargs)
        return _FakeQueryResponse(
            [
                {
                    "id": "chunk-1",
                    "score": 0.75,
                    "metadata": {
                        "content": "synthetic content",
                        "source": "synthetic",
                        "doc_title": "synthetic doc",
                    },
                }
            ]
        )

    def upsert(self, **kwargs):
        self.upsert_calls.append(kwargs)


class _FakePinecone:
    def __init__(self, api_key):
        self.api_key = api_key
        self.index = _FakeIndex()

    def Index(self, name):  # noqa: N802 - mirrors the SDK shim name
        self.index_name = name
        return self.index


class _FakeCompletions:
    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        message = types.SimpleNamespace(content="synthetic answer")
        choice = types.SimpleNamespace(message=message)
        usage = types.SimpleNamespace(total_tokens=7)
        return types.SimpleNamespace(choices=[choice], usage=usage)


class _FakeOpenAI:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.completions = _FakeCompletions()
        self.chat = types.SimpleNamespace(completions=self.completions)


def _module(name, **attrs):
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    return module


class SdkCallContractTest(unittest.TestCase):
    def test_pipeline_uses_only_surviving_sdk_surface(self):
        fakes = {
            "openai": _module("openai", OpenAI=_FakeOpenAI),
            "pinecone": _module("pinecone", Pinecone=_FakePinecone, ServerlessSpec=object),
            "sentence_transformers": _module(
                "sentence_transformers", SentenceTransformer=_FakeSentenceTransformer
            ),
            "rank_bm25": _module("rank_bm25", BM25Okapi=object),
        }

        with patch.dict(sys.modules, fakes):
            from api.pipeline.generator import Generator
            from api.pipeline.retriever import Retriever

            generator = Generator(api_key="synthetic-key", model="synthetic-model")
            result = generator.generate(
                "synthetic question",
                [{"doc_title": "synthetic doc", "content": "synthetic content"}],
            )

            create_kwargs = generator.client.completions.calls[-1]
            # The removed request property must not be reintroduced.
            self.assertNotIn("spend_control", create_kwargs)
            self.assertIn("messages", create_kwargs)
            self.assertIn("max_tokens", create_kwargs)
            self.assertEqual(result["tokens_used"], 7)

            retriever = Retriever(
                pinecone_api_key="synthetic-key",
                index_name="synthetic-index",
                embedding_model="synthetic-model",
            )
            docs = retriever.vector_search("synthetic question", top_n=3)

            query_kwargs = retriever.index.query_calls[-1]
            self.assertEqual(query_kwargs["top_k"], 3)
            self.assertTrue(query_kwargs["include_metadata"])
            self.assertEqual(docs[0]["id"], "chunk-1")
            self.assertEqual(docs[0]["content"], "synthetic content")
            self.assertEqual(docs[0]["score"], 0.75)


if __name__ == "__main__":
    unittest.main()
