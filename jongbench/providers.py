from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


USAGE = (
    "Usage: anthropic:<model>, openai:<model>, google:<model>, "
    "xai:<model>, deepseek:<model>, "
    "compat:<base_url>:<model>, random, or human"
)


@dataclass(frozen=True)
class ProviderSpec:
    provider: str
    model: str
    base_url: str | None = None

    @property
    def display_name(self) -> str:
        if self.provider == "random":
            return "random"
        return self.model


def parse_spec(s: str) -> ProviderSpec:
    if s == "random":
        return ProviderSpec(provider="random", model="random")
    if s == "human":
        return ProviderSpec(provider="human", model="human")

    for provider in ("anthropic", "openai", "google"):
        prefix = f"{provider}:"
        if s.startswith(prefix) and s[len(prefix) :]:
            return ProviderSpec(provider=provider, model=s[len(prefix) :])

    presets = {
        "xai": "https://api.x.ai/v1",
        "deepseek": "https://api.deepseek.com",
    }
    for provider, base_url in presets.items():
        prefix = f"{provider}:"
        if s.startswith(prefix) and s[len(prefix) :]:
            return ProviderSpec(
                provider=provider,
                model=s[len(prefix) :],
                base_url=base_url,
            )

    prefix = "compat:"
    if s.startswith(prefix):
        rest = s[len(prefix) :]
        if "://" not in rest:
            raise ValueError(USAGE)
        base_url, sep, model = rest.rpartition(":")
        if not sep or not base_url or not model or "://" not in base_url:
            raise ValueError(USAGE)
        return ProviderSpec(provider="compat", model=model, base_url=base_url)

    raise ValueError(USAGE)


class Provider(ABC):
    @abstractmethod
    def complete(
        self,
        system: str,
        user: str,
        *,
        max_tokens: int = 1200,
        temperature: float = 0.6,
    ) -> tuple[str, dict[str, int]]:
        raise NotImplementedError


class AnthropicProvider(Provider):
    def __init__(self, model: str, api_key: str | None = None) -> None:
        self.model = model
        self._api_key = api_key
        self._client: Any | None = None

    def __repr__(self) -> str:
        return f"{type(self).__name__}(model={self.model!r})"

    def _get_client(self) -> Any:
        if self._client is None:
            api_key = self._api_key or os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                raise RuntimeError("Missing ANTHROPIC_API_KEY")
            import anthropic

            self._client = anthropic.Anthropic(
                api_key=api_key,
                max_retries=4,
                timeout=120.0,
            )
        return self._client

    def complete(
        self,
        system: str,
        user: str,
        *,
        max_tokens: int = 1200,
        temperature: float = 0.6,
    ) -> tuple[str, dict[str, int]]:
        response = self._get_client().messages.create(
            model=self.model,
            system=system,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": user}],
        )
        text = "".join(
            getattr(block, "text", "")
            for block in getattr(response, "content", [])
            if getattr(block, "type", None) == "text" or hasattr(block, "text")
        )
        usage = getattr(response, "usage", None)
        return text, {
            "input_tokens": int(getattr(usage, "input_tokens", 0) or 0),
            "output_tokens": int(getattr(usage, "output_tokens", 0) or 0),
        }


class OpenAIProvider(Provider):
    def __init__(
        self,
        model: str,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        env_key: str | None = None,
    ) -> None:
        self.model = model
        self.base_url = base_url
        self._api_key = api_key
        self._env_key = env_key
        self._client: Any | None = None

    def __repr__(self) -> str:
        args = [f"model={self.model!r}"]
        if self.base_url is not None:
            args.append(f"base_url={self.base_url!r}")
        return f"{type(self).__name__}({', '.join(args)})"

    def _get_client(self) -> Any:
        if self._client is None:
            import openai

            if self.base_url is None:
                env_key = self._env_key or "OPENAI_API_KEY"
                api_key = self._api_key or os.environ.get(env_key)
                if not api_key:
                    raise RuntimeError(f"Missing {env_key}")
                self._client = openai.OpenAI(
                    api_key=api_key,
                    max_retries=4,
                    timeout=120.0,
                )
            else:
                env_key = self._env_key or "OPENAI_COMPAT_API_KEY"
                if self._api_key is not None:
                    api_key = self._api_key
                elif env_key == "OPENAI_COMPAT_API_KEY":
                    api_key = os.environ.get(env_key, "none")
                else:
                    api_key = os.environ.get(env_key)
                    if not api_key:
                        raise RuntimeError(f"Missing {env_key}")
                self._client = openai.OpenAI(
                    base_url=self.base_url,
                    api_key=api_key,
                    max_retries=4,
                    timeout=120.0,
                )
        return self._client

    def complete(
        self,
        system: str,
        user: str,
        *,
        max_tokens: int = 1200,
        temperature: float = 0.6,
    ) -> tuple[str, dict[str, int]]:
        import openai

        params: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "max_completion_tokens": max_tokens,
            "temperature": temperature,
        }
        try:
            response = self._get_client().chat.completions.create(**params)
        except openai.BadRequestError as exc:
            message = str(exc)
            message_lower = message.lower()
            retry_params = dict(params)
            should_retry = False
            if "temperature" in message_lower:
                retry_params.pop("temperature", None)
                should_retry = True
            if "max_completion_tokens" in message_lower:
                retry_params.pop("max_completion_tokens", None)
                retry_params["max_tokens"] = max_tokens
                should_retry = True
            if not should_retry:
                raise
            response = self._get_client().chat.completions.create(**retry_params)

        message = response.choices[0].message if response.choices else None
        content = getattr(message, "content", "") if message is not None else ""
        usage = getattr(response, "usage", None)
        return content or "", {
            "input_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
            "output_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
        }


class GoogleProvider(Provider):
    def __init__(self, model: str, api_key: str | None = None) -> None:
        self.model = model
        self._api_key = api_key
        self._client: Any | None = None

    def __repr__(self) -> str:
        return f"{type(self).__name__}(model={self.model!r})"

    def _get_client(self) -> Any:
        if self._client is None:
            api_key = self._api_key or os.environ.get("GEMINI_API_KEY")
            if not api_key:
                raise RuntimeError("Missing GEMINI_API_KEY")
            from google import genai

            self._client = genai.Client(api_key=api_key)
        return self._client

    def complete(
        self,
        system: str,
        user: str,
        *,
        max_tokens: int = 1200,
        temperature: float = 0.6,
    ) -> tuple[str, dict[str, int]]:
        from google.genai import types

        response = self._get_client().models.generate_content(
            model=self.model,
            contents=user,
            config=types.GenerateContentConfig(
                system_instruction=system,
                temperature=temperature,
                max_output_tokens=max_tokens,
            ),
        )
        usage = getattr(response, "usage_metadata", None)
        return getattr(response, "text", None) or "", {
            "input_tokens": int(getattr(usage, "prompt_token_count", 0) or 0),
            "output_tokens": int(getattr(usage, "candidates_token_count", 0) or 0),
        }


class CompatProvider(OpenAIProvider):
    def __init__(
        self,
        model: str,
        base_url: str,
        api_key: str | None = None,
        env_key: str | None = None,
    ) -> None:
        super().__init__(
            model,
            base_url=base_url,
            api_key=api_key,
            env_key=env_key,
        )


def make_provider(spec: ProviderSpec, api_key: str | None = None) -> Provider:
    if spec.provider == "anthropic":
        return AnthropicProvider(spec.model, api_key=api_key)
    if spec.provider == "openai":
        return OpenAIProvider(spec.model, api_key=api_key)
    if spec.provider == "google":
        return GoogleProvider(spec.model, api_key=api_key)
    if spec.provider == "compat":
        if spec.base_url is None:
            raise ValueError("compat provider requires base_url")
        return CompatProvider(spec.model, spec.base_url, api_key=api_key)
    if spec.provider == "xai":
        if spec.base_url is None:
            raise ValueError("xai provider requires base_url")
        return CompatProvider(
            spec.model,
            spec.base_url,
            api_key=api_key,
            env_key="XAI_API_KEY",
        )
    if spec.provider == "deepseek":
        if spec.base_url is None:
            raise ValueError("deepseek provider requires base_url")
        return CompatProvider(
            spec.model,
            spec.base_url,
            api_key=api_key,
            env_key="DEEPSEEK_API_KEY",
        )
    if spec.provider == "random":
        raise ValueError("random provider is handled separately")
    raise ValueError(USAGE)
