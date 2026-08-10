from types import SimpleNamespace

import pytest

from jongbench.providers import (
    OPENROUTER_BASE_URL,
    Provider,
    cacheable,
    make_provider,
    parse_spec,
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


def test_bare_openrouter_id_resolves() -> None:
    parsed = parse_spec("anthropic/claude-opus-5")
    assert parsed.provider == "openrouter"
    assert parsed.model == "anthropic/claude-opus-5"
    assert parsed.base_url == OPENROUTER_BASE_URL
    assert parsed.pin == ()
    assert parsed.reasoning is None
    assert parsed.display_name == "claude-opus-5"


def test_variant_ids_keep_their_colon_suffix() -> None:
    assert parse_spec("meta-llama/llama-3.1-8b-instruct:free").model == (
        "meta-llama/llama-3.1-8b-instruct:free"
    )


def test_at_suffix_pins_inference_routing() -> None:
    parsed = parse_spec("openai/gpt-oss-120b@cerebras")
    assert parsed.model == "openai/gpt-oss-120b"
    assert parsed.pin == ("cerebras",)

    ordered = parse_spec("openai/gpt-oss-120b@cerebras,groq")
    assert ordered.pin == ("cerebras", "groq")


def test_hash_suffix_sets_reasoning_effort() -> None:
    parsed = parse_spec("openai/gpt-5.2#high")
    assert parsed.model == "openai/gpt-5.2"
    assert parsed.reasoning == "high"
    assert parsed.display_name == "gpt-5.2-high"

    combined = parse_spec("openai/gpt-oss-120b@cerebras#low")
    assert combined.pin == ("cerebras",)
    assert combined.reasoning == "low"

    with pytest.raises(ValueError):
        parse_spec("openai/gpt-5.2#hard")


def test_compat_spec_preserves_url_ports() -> None:
    parsed = parse_spec("compat:http://127.0.0.1:8080/v1:model-name")
    assert parsed.provider == "compat"
    assert parsed.base_url == "http://127.0.0.1:8080/v1"
    assert parsed.model == "model-name"


def test_special_seats_are_not_providers() -> None:
    for seat in ("mortal", "random", "human"):
        assert parse_spec(seat).provider == seat
        with pytest.raises(ValueError, match="handled separately"):
            make_provider(parse_spec(seat))


def test_rejects_unqualified_model_ids() -> None:
    for spec in ("claude-opus-5", "anthropic:claude-opus-5", "anthropic/", ""):
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


def test_spec_reasoning_reaches_the_provider() -> None:
    provider = make_provider(parse_spec("openai/gpt-5.2#high"))
    assert provider.reasoning == "high"
    assert make_provider(parse_spec("openai/gpt-5.2#high"), reasoning="low").reasoning == "low"


def test_cacheable_marks_an_ephemeral_breakpoint() -> None:
    assert cacheable("rules") == [
        {"type": "text", "text": "rules", "cache_control": {"type": "ephemeral"}}
    ]
