"""Guard the OpenAI surface used by the RAG pipeline.

The vendor spec removed the Assistants endpoints (``/assistants``,
``/assistants/{assistant_id}``), ``/conversations``, ``/evals`` and
``/audio/translations``, along with the Assistants-only ``auto`` enum values
on ``response_format`` and ``chunking_strategy``.  Every OpenAI call in this
repository is a plain chat completion, which the spec keeps.

The fake client below exposes only ``chat.completions.create``; reaching for a
removed namespace (``client.beta``, ``client.audio``, ``client.assistants``,
...) raises ``AttributeError`` so this test fails loudly if such a call is ever
introduced.  Inputs are synthetic and the test runs offline.
"""

import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class _FakeCompletions:
    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        message = types.SimpleNamespace(content="RELEVANT synthetic answer")
        choice = types.SimpleNamespace(message=message)
        usage = types.SimpleNamespace(total_tokens=5)
        return types.SimpleNamespace(choices=[choice], usage=usage)


class _FakeOpenAI:
    """Only the surviving chat-completions surface exists on this client."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.completions = _FakeCompletions()
        self.chat = types.SimpleNamespace(completions=self.completions)


def _module(name, **attrs):
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    return module


class OpenAICompletionsOnlyTest(unittest.TestCase):
    def test_pipeline_only_touches_chat_completions(self):
        fakes = {"openai": _module("openai", OpenAI=_FakeOpenAI)}

        with patch.dict(sys.modules, fakes):
            from api.pipeline.crag import CRAG
            from api.pipeline.generator import Generator
            from api.pipeline.query_rewriter import QueryRewriter

            generator = Generator(api_key="synthetic-key", model="synthetic-model")
            generated = generator.generate(
                "synthetic question",
                [{"doc_title": "synthetic doc", "content": "synthetic content"}],
            )
            self.assertEqual(generated["answer"], "RELEVANT synthetic answer")

            rewriter = QueryRewriter(
                api_key="synthetic-key",
                model="synthetic-model",
                base_url="https://synthetic.invalid/v1",
            )
            self.assertEqual(rewriter.hyde_expand("synthetic question"), "RELEVANT synthetic answer")

            crag = CRAG(
                api_key="synthetic-key",
                model="synthetic-model",
                base_url="https://synthetic.invalid/v1",
                query_rewriter=rewriter,
            )
            grade = crag._grade_one("synthetic question", {"content": "synthetic content"})
            self.assertEqual(grade, "RELEVANT")

            # All three call sites must have gone through chat.completions.create.
            self.assertEqual(len(generator.client.completions.calls), 1)
            self.assertEqual(len(rewriter.client.completions.calls), 1)
            self.assertEqual(len(crag.client.completions.calls), 1)

            for client in (generator.client, rewriter.client, crag.client):
                # Removed namespaces must not be reachable/used.
                self.assertFalse(hasattr(client, "beta"))
                self.assertFalse(hasattr(client, "audio"))
                self.assertFalse(hasattr(client, "assistants"))
                self.assertFalse(hasattr(client, "conversations"))
                self.assertFalse(hasattr(client, "evals"))

                for call in client.completions.calls:
                    self.assertIn("messages", call)
                    self.assertIn("max_tokens", call)
                    # Assistants-only request properties/enums must not appear.
                    self.assertNotIn("response_format", call)
                    self.assertNotIn("tool_resources", call)
                    self.assertNotIn("chunking_strategy", call)


if __name__ == "__main__":
    unittest.main()
