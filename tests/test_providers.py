from types import SimpleNamespace
from unittest.mock import patch

import openai

from jongbench.providers import OpenAIProvider, parse_spec


class _FakeBadRequest(Exception):
    pass


class _FakeCompletions:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def create(self, **params: object) -> SimpleNamespace:
        self.calls.append(dict(params))
        if "temperature" in params:
            raise _FakeBadRequest("Unsupported parameter: temperature")
        if "max_completion_tokens" in params:
            raise _FakeBadRequest("Unsupported parameter: max_completion_tokens")
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content='{"choice":0}'))],
            usage=SimpleNamespace(prompt_tokens=12, completion_tokens=3),
        )


def test_openai_compatibility_retries_multiple_unsupported_parameters() -> None:
    completions = _FakeCompletions()
    provider = OpenAIProvider("compat-model", base_url="http://localhost/v1")
    provider._client = SimpleNamespace(
        chat=SimpleNamespace(completions=completions)
    )

    with patch.object(openai, "BadRequestError", _FakeBadRequest):
        text, usage = provider.complete("system", "user", max_tokens=40)

    assert text == '{"choice":0}'
    assert usage == {"input_tokens": 12, "output_tokens": 3}
    assert len(completions.calls) == 3
    assert "temperature" not in completions.calls[-1]
    assert "max_completion_tokens" not in completions.calls[-1]
    assert completions.calls[-1]["max_tokens"] == 40


def test_compat_spec_preserves_url_ports() -> None:
    spec = parse_spec("compat:http://127.0.0.1:8080/v1:model-name")
    assert spec.base_url == "http://127.0.0.1:8080/v1"
    assert spec.model == "model-name"
