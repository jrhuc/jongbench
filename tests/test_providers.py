from types import SimpleNamespace
from unittest.mock import patch

import openai

from jongbench.providers import (
    AnthropicProvider,
    CompatProvider,
    GoogleProvider,
    OpenAIProvider,
    parse_spec,
    reasoning_levels,
)


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

    with patch.object(openai, "BadRequestError", _FakeBadRequest):
        provider.complete("system", "user", max_tokens=50)

    assert len(completions.calls) == 4
    assert "temperature" not in completions.calls[-1]
    assert "max_completion_tokens" not in completions.calls[-1]
    assert completions.calls[-1]["max_tokens"] == 50


def test_reasoning_levels_match_provider_model_capabilities() -> None:
    cases = {
        ("openai", "gpt-5.1"): ["off", "low", "medium", "high"],
        ("openai", "gpt-5.2"): ["off", "low", "medium", "high", "xhigh"],
        ("openai", "gpt-5.2-pro"): ["medium", "high", "xhigh"],
        ("google", "gemini-3-pro-preview"): ["low", "high"],
        ("google", "gemini-3.1-pro"): ["low", "medium", "high"],
        ("google", "gemini-3-flash-preview"): [
            "minimal",
            "low",
            "medium",
            "high",
        ],
        ("xai", "grok-3-mini"): ["low", "high"],
        ("xai", "grok-4.3-fast"): ["off", "low", "medium", "high"],
        ("deepseek", "deepseek-reasoner"): ["off", "high", "max"],
    }
    for (provider, model), expected in cases.items():
        assert reasoning_levels(provider, model) == expected

    assert reasoning_levels(
        "google", "gemini-3-flash-preview", {"thinking": False}
    ) == []


def test_anthropic_reasoning_levels_use_dynamic_capabilities() -> None:
    metadata = {
        "capabilities": {
            "thinking": {"types": {"disabled": {"supported": True}}},
            "effort": {
                "low": {"supported": True},
                "medium": {"supported": False},
                "high": {"supported": True},
                "max": {"supported": True},
            },
        }
    }
    assert reasoning_levels("anthropic", "future-model", metadata) == [
        "off",
        "low",
        "high",
        "max",
    ]


class _RecordingCompletions:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def create(self, **params: object) -> SimpleNamespace:
        self.calls.append(dict(params))
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="OK"))],
            usage=SimpleNamespace(prompt_tokens=2, completion_tokens=1),
        )


def _openai_client(completions: _RecordingCompletions) -> SimpleNamespace:
    return SimpleNamespace(chat=SimpleNamespace(completions=completions))


def test_openai_off_is_sent_without_downgrade() -> None:
    completions = _RecordingCompletions()
    provider = OpenAIProvider("gpt-5.2", reasoning="off")
    provider._client = _openai_client(completions)

    provider.complete("system", "user")

    assert completions.calls == [
        {
            "model": "gpt-5.2",
            "messages": [
                {"role": "system", "content": "system"},
                {"role": "user", "content": "user"},
            ],
            "max_completion_tokens": 1200,
            "temperature": 0.6,
            "reasoning_effort": "none",
        }
    ]


def test_deepseek_off_uses_native_thinking_body() -> None:
    completions = _RecordingCompletions()
    provider = CompatProvider(
        "deepseek-chat",
        "https://api.deepseek.com",
        reasoning="off",
        reasoning_style="deepseek",
    )
    provider._client = _openai_client(completions)

    provider.complete("system", "user")

    assert completions.calls[0]["extra_body"] == {"thinking": {"type": "disabled"}}
    assert "reasoning_effort" not in completions.calls[0]


def test_anthropic_adaptive_reasoning_sends_effort() -> None:
    calls: list[dict[str, object]] = []

    def create(**params: object) -> SimpleNamespace:
        calls.append(dict(params))
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text="OK")],
            usage=SimpleNamespace(input_tokens=2, output_tokens=1),
        )

    provider = AnthropicProvider("claude-opus-4-6", reasoning="xhigh")
    provider._client = SimpleNamespace(messages=SimpleNamespace(create=create))

    provider.complete("system", "user")

    assert calls[0]["thinking"] == {"type": "adaptive"}
    assert calls[0]["output_config"] == {"effort": "xhigh"}
    assert "temperature" not in calls[0]


def test_google_reasoning_uses_levels_and_legacy_budgets() -> None:
    calls: list[dict[str, object]] = []

    def generate_content(**params: object) -> SimpleNamespace:
        calls.append(dict(params))
        return SimpleNamespace(text="OK", usage_metadata=None)

    client = SimpleNamespace(models=SimpleNamespace(generate_content=generate_content))
    medium = GoogleProvider("gemini-3-flash-preview", reasoning="medium")
    medium._client = client
    high = GoogleProvider("gemini-2.5-pro", reasoning="high")
    high._client = client

    medium.complete("system", "user")
    high.complete("system", "user")

    medium_config = calls[0]["config"]
    high_config = calls[1]["config"]
    assert str(medium_config.thinking_config.thinking_level).lower().endswith("medium")
    assert high_config.thinking_config.thinking_budget == 16_000
    assert high_config.max_output_tokens == 17_200


def test_compat_spec_preserves_url_ports() -> None:
    spec = parse_spec("compat:http://127.0.0.1:8080/v1:model-name")
    assert spec.base_url == "http://127.0.0.1:8080/v1"
    assert spec.model == "model-name"
