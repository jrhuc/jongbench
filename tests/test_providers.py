from types import SimpleNamespace

import pytest

from jongbench.providers import (
    OPENROUTER_BASE_URL,
    REASONING_LEVELS,
    Provider,
    cacheable,
    list_models,
    make_provider,
    parse_spec,
    reasoning_levels,
)


def _chunk(content: str = "", reasoning: str = "", usage=None, provider=None):
    delta = SimpleNamespace(content=content or None, reasoning=reasoning or None)
    return SimpleNamespace(
        choices=[SimpleNamespace(delta=delta)],
        usage=usage,
        provider=provider,
    )


class _FakeCompletions:
    def __init__(self, chunks) -> None:
        self.chunks = chunks
        self.calls: list[dict[str, object]] = []

    def create(self, **params: object):
        self.calls.append(dict(params))
        return iter(self.chunks)


def _provider(chunks, **kwargs) -> tuple[Provider, _FakeCompletions]:
    provider = Provider("anthropic/claude-opus-5", OPENROUTER_BASE_URL, **kwargs)
    completions = _FakeCompletions(chunks)
    provider._client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    return provider, completions


def test_spec_forms_all_resolve_to_openrouter() -> None:
    for spec, model in [
        ("anthropic/claude-opus-5", "anthropic/claude-opus-5"),
        ("openrouter:anthropic/claude-opus-5", "anthropic/claude-opus-5"),
        ("anthropic:claude-opus-5", "anthropic/claude-opus-5"),
        ("openai:gpt-5.2", "openai/gpt-5.2"),
        ("google:gemini-3-pro", "google/gemini-3-pro"),
        ("xai:grok-4.1", "x-ai/grok-4.1"),
        ("meta:llama-4", "meta-llama/llama-4"),
        ("kimi:kimi-k2", "moonshotai/kimi-k2"),
        ("zai:glm-5", "z-ai/glm-5"),
    ]:
        parsed = parse_spec(spec)
        assert parsed.provider == "openrouter", spec
        assert parsed.model == model, spec
        assert parsed.base_url == OPENROUTER_BASE_URL
        assert parsed.pin == ()


def test_legacy_inference_provider_prefix_pins_routing() -> None:
    parsed = parse_spec("cerebras:openai/gpt-oss-120b")
    assert parsed.model == "openai/gpt-oss-120b"
    assert parsed.pin == ("cerebras",)

    with pytest.raises(ValueError, match="full OpenRouter id"):
        parse_spec("cerebras:gpt-oss-120b")


def test_compat_spec_preserves_url_ports() -> None:
    parsed = parse_spec("compat:http://127.0.0.1:8080/v1:model-name")
    assert parsed.provider == "compat"
    assert parsed.base_url == "http://127.0.0.1:8080/v1"
    assert parsed.model == "model-name"


def test_random_and_human_seats_are_not_providers() -> None:
    for seat in ("random", "human"):
        assert parse_spec(seat).provider == seat
        with pytest.raises(ValueError, match="handled separately"):
            make_provider(parse_spec(seat))


def test_rejects_unqualified_model_ids() -> None:
    for spec in ("claude-opus-5", "openrouter:claude-opus-5", ""):
        with pytest.raises(ValueError):
            parse_spec(spec)


def test_streams_text_and_reasoning_with_usage() -> None:
    usage = SimpleNamespace(
        prompt_tokens=2500,
        completion_tokens=40,
        prompt_tokens_details=SimpleNamespace(cached_tokens=2372),
        completion_tokens_details=SimpleNamespace(reasoning_tokens=30),
    )
    provider, completions = _provider(
        [
            _chunk(reasoning="weigh ", provider="anthropic"),
            _chunk(reasoning="the discard"),
            _chunk(content='{"choice"'),
            _chunk(content=": 3}"),
            _chunk(usage=usage),
        ]
    )

    result = provider.complete(
        [{"role": "user", "content": "go"}], max_tokens=40, temperature=0.6
    )

    assert result.text == '{"choice": 3}'
    assert result.reasoning == "weigh the discard"
    assert result.served_by == "anthropic"
    assert result.usage == {
        "input_tokens": 2500,
        "output_tokens": 40,
        "cached_input_tokens": 2372,
        "reasoning_tokens": 30,
    }
    assert completions.calls[0]["stream"] is True
    assert completions.calls[0]["stream_options"] == {"include_usage": True}
    assert completions.calls[0]["temperature"] == 0.6


def test_usage_absent_details_defaults_to_zero() -> None:
    usage = SimpleNamespace(prompt_tokens=10, completion_tokens=1)
    provider, _ = _provider([_chunk(content="ok"), _chunk(usage=usage)])

    result = provider.complete([{"role": "user", "content": "go"}])

    assert result.usage["cached_input_tokens"] == 0
    assert result.usage["reasoning_tokens"] == 0
    assert result.reasoning == ""


def test_reasoning_effort_and_provider_pin_ride_extra_body() -> None:
    provider, completions = _provider([_chunk(content="x")], reasoning="high", pin=("cerebras",))
    provider.complete([{"role": "user", "content": "go"}])

    extra = completions.calls[0]["extra_body"]
    assert extra["reasoning"] == {"effort": "high"}
    assert extra["provider"] == {"order": ["cerebras"], "allow_fallbacks": False}


def test_reasoning_none_disables_rather_than_setting_effort() -> None:
    provider, completions = _provider([_chunk(content="x")], reasoning="none")
    provider.complete([{"role": "user", "content": "go"}])

    assert completions.calls[0]["extra_body"]["reasoning"] == {"enabled": False}


def test_no_extra_body_when_unconfigured() -> None:
    provider, completions = _provider([_chunk(content="x")])
    provider.complete([{"role": "user", "content": "go"}])

    assert "extra_body" not in completions.calls[0]
    assert "temperature" not in completions.calls[0]


def test_reasoning_levels_follow_advertised_support() -> None:
    reasons = {"supported_parameters": ["reasoning", "reasoning_effort", "temperature"]}
    assert reasoning_levels("anthropic/claude-opus-5", reasons) == list(REASONING_LEVELS)

    # Reasoning without the effort ladder still gets the full list: OpenRouter
    # translates an effort it cannot pass through.
    partial = {"supported_parameters": ["reasoning", "temperature"]}
    assert reasoning_levels("anthropic/claude-haiku-4.5", partial) == list(REASONING_LEVELS)

    assert reasoning_levels("openai/gpt-4o-mini", {"supported_parameters": ["temperature"]}) == []
    assert reasoning_levels("anything", None) == []


def test_list_models_projects_catalogue(monkeypatch) -> None:
    payload = {
        "data": [
            {
                "id": "openai/gpt-4o-mini",
                "created": 100,
                "supported_parameters": ["temperature"],
            },
            {
                "id": "anthropic/claude-opus-5",
                "name": "Claude Opus 5",
                "created": 200,
                "context_length": 1000000,
                "supported_parameters": ["reasoning", "reasoning_effort", "temperature"],
            },
            {"created": 300},
        ]
    }

    class _Response:
        def read(self):
            import json

            return json.dumps(payload).encode()

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(
        "jongbench.providers.urllib_request.urlopen", lambda *a, **k: _Response()
    )

    models = list_models()

    assert [m["id"] for m in models] == ["anthropic/claude-opus-5", "openai/gpt-4o-mini"]
    assert models[0]["reasoning"] == list(REASONING_LEVELS)
    assert models[0]["supports_temperature"] is True
    assert models[1]["reasoning"] == []


def test_cacheable_marks_an_ephemeral_breakpoint() -> None:
    assert cacheable("rules") == [
        {"type": "text", "text": "rules", "cache_control": {"type": "ephemeral"}}
    ]
