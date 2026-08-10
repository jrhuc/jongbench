from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any


OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_ENV_KEY = "OPENROUTER_API_KEY"

USAGE = (
    "Usage: <vendor>/<model> — an OpenRouter id, e.g. anthropic/claude-opus-5 — "
    "optionally suffixed with @<provider>[,<provider>...] to pin inference routing "
    "and #<effort> to set reasoning (off, minimal, low, medium, high, xhigh, max). "
    "Also compat:<base_url>:<model>, mortal, random, and human."
)

# jongbench's own vocabulary, kept stable so saved run configs keep resolving.
# "off" maps to OpenRouter's disable switch rather than an effort level.
REASONING_LEVELS = ("off", "minimal", "low", "medium", "high", "xhigh", "max")
_REASONING_OFF = {"off", "none"}


@dataclass(frozen=True)
class ProviderSpec:
    provider: str  # openrouter | compat | mortal | random | human
    model: str
    base_url: str | None = None
    pin: tuple[str, ...] = ()
    reasoning: str | None = None

    @property
    def display_name(self) -> str:
        if self.provider in {"mortal", "random", "human"}:
            return self.provider
        base = self.model.rpartition("/")[2] or self.model
        return f"{base}-{self.reasoning}" if self.reasoning else base


@dataclass
class Completion:
    text: str
    reasoning: str = ""
    usage: dict[str, int] = field(default_factory=dict)
    served_by: str | None = None


def parse_spec(s: str) -> ProviderSpec:
    if s in {"mortal", "random", "human"}:
        return ProviderSpec(provider=s, model=s)

    if s.startswith("compat:"):
        rest = s[len("compat:") :]
        base_url, sep, model = rest.rpartition(":")
        if not sep or not base_url or not model or "://" not in base_url:
            raise ValueError(USAGE)
        return ProviderSpec(provider="compat", model=model, base_url=base_url)

    # `#effort` before `@pin`: neither character appears in OpenRouter model ids,
    # and `:` cannot separate suffixes because ids use it for variants (`:free`).
    s, _, effort = s.partition("#")
    if effort and effort not in REASONING_LEVELS:
        raise ValueError(USAGE)
    s, _, pinned = s.partition("@")
    pin = tuple(p for p in pinned.split(",") if p) if pinned else ()

    if "/" not in s or s.startswith("/") or not s.rpartition("/")[2]:
        raise ValueError(USAGE)
    return ProviderSpec(
        provider="openrouter",
        model=s,
        base_url=OPENROUTER_BASE_URL,
        pin=pin,
        reasoning=effort or None,
    )


class Provider:
    def __init__(
        self,
        model: str,
        base_url: str,
        *,
        api_key: str | None = None,
        env_key: str = OPENROUTER_ENV_KEY,
        reasoning: str | None = None,
        pin: tuple[str, ...] = (),
    ) -> None:
        self.model = model
        self.base_url = base_url
        self.reasoning = reasoning
        self.pin = pin
        self._api_key = api_key
        self._env_key = env_key
        self._client: Any | None = None

    def __repr__(self) -> str:
        return f"{type(self).__name__}(model={self.model!r})"

    def _get_client(self) -> Any:
        if self._client is None:
            api_key = self._api_key or os.environ.get(self._env_key)
            if not api_key:
                raise RuntimeError(f"Missing {self._env_key}")
            import openai

            self._client = openai.OpenAI(
                api_key=api_key,
                base_url=self.base_url,
                max_retries=4,
                timeout=600.0,
                default_headers={
                    "HTTP-Referer": "https://github.com/jrhuc/jongbench",
                    "X-Title": "jongbench",
                },
            )
        return self._client

    def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        max_tokens: int = 1200,
        temperature: float | None = None,
    ) -> Completion:
        params: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if temperature is not None:
            params["temperature"] = temperature

        extra: dict[str, Any] = {}
        if self.reasoning is not None:
            extra["reasoning"] = (
                {"enabled": False}
                if self.reasoning in _REASONING_OFF
                else {"effort": self.reasoning}
            )
        if self.pin:
            extra["provider"] = {"order": list(self.pin), "allow_fallbacks": False}
        if extra:
            params["extra_body"] = extra

        text: list[str] = []
        reasoning: list[str] = []
        usage: dict[str, int] = {}
        served_by: str | None = None

        for chunk in self._get_client().chat.completions.create(**params):
            served_by = served_by or _extra(chunk, "provider")
            chunk_usage = getattr(chunk, "usage", None)
            if chunk_usage is not None:
                usage = _normalize_usage(chunk_usage)
            for choice in getattr(chunk, "choices", None) or []:
                delta = getattr(choice, "delta", None)
                if delta is None:
                    continue
                if getattr(delta, "content", None):
                    text.append(delta.content)
                piece = _extra(delta, "reasoning")
                if piece:
                    reasoning.append(str(piece))

        return Completion(
            text="".join(text),
            reasoning="".join(reasoning),
            usage=usage,
            served_by=served_by,
        )


def make_provider(
    spec: ProviderSpec, api_key: str | None = None, reasoning: str | None = None
) -> Provider:
    if spec.provider in {"mortal", "random", "human"}:
        raise ValueError(f"{spec.provider} seats are handled separately")
    if spec.base_url is None:
        raise ValueError(f"{spec.provider} provider requires base_url")
    env_key = OPENROUTER_ENV_KEY if spec.provider == "openrouter" else "COMPAT_API_KEY"
    return Provider(
        spec.model,
        spec.base_url,
        api_key=api_key,
        env_key=env_key,
        reasoning=reasoning if reasoning is not None else spec.reasoning,
        pin=spec.pin,
    )


def _extra(obj: Any, key: str) -> Any:
    value = getattr(obj, key, None)
    if value is not None:
        return value
    model_extra = getattr(obj, "model_extra", None)
    if isinstance(model_extra, dict):
        return model_extra.get(key)
    if isinstance(obj, dict):
        return obj.get(key)
    return None


def _normalize_usage(usage: Any) -> dict[str, int]:
    details = getattr(usage, "prompt_tokens_details", None)
    cached = _extra(details, "cached_tokens") if details is not None else None
    return {
        "input_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
        "output_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
        "cached_input_tokens": int(cached or 0),
        "reasoning_tokens": int(_reasoning_tokens(usage) or 0),
    }


def _reasoning_tokens(usage: Any) -> int | None:
    details = getattr(usage, "completion_tokens_details", None)
    if details is None:
        return None
    return _extra(details, "reasoning_tokens")


def cacheable(text: str) -> list[dict[str, Any]]:
    """A content block marked for provider-side prompt caching. OpenRouter forwards
    `cache_control` to Anthropic; providers that cache automatically ignore it."""
    return [{"type": "text", "text": text, "cache_control": {"type": "ephemeral"}}]
