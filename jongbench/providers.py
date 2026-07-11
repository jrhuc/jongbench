from __future__ import annotations

import json
import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Any
from urllib import error as urllib_error
from urllib import parse as urllib_parse
from urllib import request as urllib_request


USAGE = (
    "Usage: anthropic:<model>, openai:<model>, google:<model>, "
    "xai:<model>, deepseek:<model>, "
    "compat:<base_url>:<model>, random, or human"
)

REASONING_LEVELS = (
    "off",
    "minimal",
    "low",
    "medium",
    "high",
    "xhigh",
    "max",
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
    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        reasoning: str | None = None,
    ) -> None:
        self.model = model
        self._api_key = api_key
        self.reasoning = reasoning
        self._client: Any | None = None
        self._supports_temperature = True

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
        import anthropic

        level = self.reasoning
        params: dict[str, Any] = {
            "model": self.model,
            "system": system,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": user}],
        }
        if level is None:
            if self._supports_temperature:
                params["temperature"] = temperature
        elif level == "off":
            if self._supports_temperature:
                params["temperature"] = temperature
            params["thinking"] = {"type": "disabled"}
        else:
            params["thinking"] = {"type": "adaptive"}
            params["output_config"] = {"effort": level}
        while True:
            try:
                response = self._get_client().messages.create(**params)
                break
            except anthropic.BadRequestError as exc:
                message = str(exc).lower()
                if "temperature" in message and "temperature" in params:
                    params.pop("temperature")
                    self._supports_temperature = False
                else:
                    raise

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
        reasoning: str | None = None,
        reasoning_style: str = "openai",
    ) -> None:
        self.model = model
        self.base_url = base_url
        self._api_key = api_key
        self._env_key = env_key
        self.reasoning = reasoning
        self.reasoning_style = reasoning_style
        self._client: Any | None = None
        self._supports_temperature = True
        self._uses_max_completion_tokens = True

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
        }
        if self._uses_max_completion_tokens:
            params["max_completion_tokens"] = max_tokens
        else:
            params["max_tokens"] = max_tokens
        if self._supports_temperature:
            params["temperature"] = temperature
        if self.reasoning is not None:
            if self.reasoning_style == "deepseek" and self.reasoning == "off":
                params["extra_body"] = {"thinking": {"type": "disabled"}}
            else:
                params["reasoning_effort"] = (
                    "none" if self.reasoning == "off" else self.reasoning
                )
        while True:
            try:
                response = self._get_client().chat.completions.create(**params)
                break
            except openai.BadRequestError as exc:
                message = str(exc).lower()
                changed = False
                if "temperature" in message and "temperature" in params:
                    params.pop("temperature")
                    self._supports_temperature = False
                    changed = True
                if (
                    "max_completion_tokens" in message
                    and "max_completion_tokens" in params
                ):
                    params.pop("max_completion_tokens")
                    params["max_tokens"] = max_tokens
                    self._uses_max_completion_tokens = False
                    changed = True
                if not changed:
                    raise

        message = response.choices[0].message if response.choices else None
        content = getattr(message, "content", "") if message is not None else ""
        usage = getattr(response, "usage", None)
        return content or "", {
            "input_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
            "output_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
        }


class GoogleProvider(Provider):
    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        reasoning: str | None = None,
    ) -> None:
        self.model = model
        self._api_key = api_key
        self.reasoning = reasoning
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

        config_kwargs: dict[str, Any] = {
            "system_instruction": system,
            "temperature": temperature,
            "max_output_tokens": max_tokens,
        }
        level = self.reasoning
        model_id = self.model.lower()
        if level == "off":
            config_kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=0)
        elif level is not None and "2.5" in model_id:
            budget = (
                16_000
                if level == "high"
                else _google_thinking_budget_max(model_id)
            )
            config_kwargs["max_output_tokens"] = max(max_tokens, budget + 1200)
            config_kwargs["thinking_config"] = types.ThinkingConfig(
                thinking_budget=budget
            )
        elif level is not None:
            config_kwargs["thinking_config"] = types.ThinkingConfig(
                thinking_level=level
            )
        response = self._get_client().models.generate_content(
            model=self.model,
            contents=user,
            config=types.GenerateContentConfig(**config_kwargs),
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
        reasoning: str | None = None,
        reasoning_style: str = "openai",
    ) -> None:
        super().__init__(
            model,
            base_url=base_url,
            api_key=api_key,
            env_key=env_key,
            reasoning=reasoning,
            reasoning_style=reasoning_style,
        )


def make_provider(
    spec: ProviderSpec, api_key: str | None = None, reasoning: str | None = None
) -> Provider:
    if spec.provider == "anthropic":
        return AnthropicProvider(spec.model, api_key=api_key, reasoning=reasoning)
    if spec.provider == "openai":
        return OpenAIProvider(spec.model, api_key=api_key, reasoning=reasoning)
    if spec.provider == "google":
        return GoogleProvider(spec.model, api_key=api_key, reasoning=reasoning)
    if spec.provider == "compat":
        if spec.base_url is None:
            raise ValueError("compat provider requires base_url")
        return CompatProvider(
            spec.model, spec.base_url, api_key=api_key, reasoning=reasoning
        )
    if spec.provider == "xai":
        if spec.base_url is None:
            raise ValueError("xai provider requires base_url")
        return CompatProvider(
            spec.model,
            spec.base_url,
            api_key=api_key,
            env_key="XAI_API_KEY",
            reasoning=reasoning,
        )
    if spec.provider == "deepseek":
        if spec.base_url is None:
            raise ValueError("deepseek provider requires base_url")
        return CompatProvider(
            spec.model,
            spec.base_url,
            api_key=api_key,
            env_key="DEEPSEEK_API_KEY",
            reasoning=reasoning,
            reasoning_style="deepseek",
        )
    if spec.provider == "random":
        raise ValueError("random provider is handled separately")
    raise ValueError(USAGE)


def reasoning_levels(
    provider: str,
    model: str,
    metadata: dict[str, Any] | None = None,
) -> list[str]:
    model_id = model.lower()
    if provider == "anthropic":
        dynamic = _anthropic_reasoning_levels(metadata)
        if dynamic is not None:
            return dynamic
        if _anthropic_adaptive_max(model_id):
            return ["low", "medium", "high", "xhigh", "max"]
        legacy_adaptive = ("opus-4-6", "opus-4.6", "sonnet-4-6", "sonnet-4.6")
        if any(token in model_id for token in legacy_adaptive):
            return ["low", "medium", "high", "max"]
        return []
    if provider == "openai":
        return _openai_reasoning_levels(model_id)
    if provider == "google":
        if metadata is not None and metadata.get("thinking") is False:
            return []
        if "gemini-3" in model_id:
            if "flash-image" in model_id:
                return ["minimal", "high"]
            if "pro-image" in model_id:
                return ["high"]
            if "flash" in model_id:
                return ["minimal", "low", "medium", "high"]
            minor = re.search(r"gemini-3\.(\d+)", model_id)
            return (
                ["low", "medium", "high"]
                if minor and int(minor.group(1)) >= 1
                else ["low", "high"]
            )
        if "gemini-2.5" in model_id:
            return ["high", "max"]
        return []
    if provider == "xai":
        if "grok-3-mini" in model_id:
            return ["low", "high"]
        if "grok-4.3" in model_id:
            return ["off", "low", "medium", "high"]
        if "grok-4" in model_id:
            return ["low", "medium", "high"]
        return []
    if provider == "deepseek" and "deepseek" in model_id:
        return ["off", "high", "max"]
    return []


def _anthropic_reasoning_levels(metadata: dict[str, Any] | None) -> list[str] | None:
    if metadata is None:
        return None
    capabilities = metadata.get("capabilities")
    effort = capabilities.get("effort") if isinstance(capabilities, dict) else None
    if not isinstance(effort, dict):
        return None
    levels = []
    thinking = capabilities.get("thinking")
    types = thinking.get("types") if isinstance(thinking, dict) else None
    disabled = types.get("disabled") if isinstance(types, dict) else None
    if isinstance(disabled, dict) and disabled.get("supported") is True:
        levels.append("off")
    for level in ("low", "medium", "high", "xhigh", "max"):
        support = effort.get(level)
        if isinstance(support, dict) and support.get("supported") is True:
            levels.append(level)
    return levels


def _anthropic_adaptive_max(model_id: str) -> bool:
    opus = re.search(r"opus-(\d+)[.-](\d+)", model_id)
    if opus and (int(opus.group(1)), int(opus.group(2))) >= (4, 7):
        return True
    sonnet = re.search(r"sonnet-(\d+)", model_id)
    return bool(sonnet and int(sonnet.group(1)) >= 5) or "fable-5" in model_id


def _openai_reasoning_levels(model_id: str) -> list[str]:
    if "deep-research" in model_id:
        return ["medium"]
    if not re.search(r"(?:^|/)gpt-5(?:[.-]|$)", model_id):
        if re.search(r"(?:^|/)(?:o1|o3|o4)(?:[.-]|$)", model_id):
            return ["low", "medium", "high"]
        return []
    version_match = re.search(r"(?:^|/)gpt-5[.-](\d+)(?:[.-]|$)", model_id)
    version = int(version_match.group(1)) if version_match else None
    if "-chat" in model_id:
        return ["medium"] if version is not None else []
    if re.search(r"(?:^|/)gpt-5[.-]?pro(?:[.-]|$)", model_id):
        return ["high"]
    if re.search(r"(?:^|/)gpt-5[.-]\d+[.-]pro(?:[.-]|$)", model_id):
        return ["medium", "high", "xhigh"]
    if "codex" in model_id:
        if version is not None and version >= 3:
            return ["off", "low", "medium", "high", "xhigh"]
        if "codex-max" in model_id or (version is not None and version >= 2):
            return ["low", "medium", "high", "xhigh"]
        return ["low", "medium", "high"]
    if version == 1:
        return ["off", "low", "medium", "high"]
    if version is not None and version >= 6:
        return ["low", "medium", "high", "xhigh", "max"]
    if version is not None and version >= 2:
        return ["off", "low", "medium", "high", "xhigh"]
    return ["minimal", "low", "medium", "high"]


def _google_thinking_budget_max(model_id: str) -> int:
    if "2.5" in model_id and "pro" in model_id and "flash" not in model_id:
        return 32_768
    return 24_576


def list_models(provider: str, api_key: str) -> list[dict[str, Any]]:
    openai_exclude = (
        "whisper",
        "tts",
        "embed",
        "dall-e",
        "moderation",
        "image",
        "audio",
        "realtime",
        "transcribe",
        "davinci",
        "babbage",
        "instruct",
        "codex",
        "search",
        "computer-use",
    )

    def parse_iso_created(value: Any) -> int | None:
        if not isinstance(value, str) or not value:
            return None
        text = value[:-1] + "+00:00" if value.endswith("Z") else value
        try:
            return int(datetime.fromisoformat(text).timestamp())
        except ValueError:
            return None

    def fetch_json(url: str, headers: dict[str, str], name: str) -> Any:
        request = urllib_request.Request(url, headers=headers, method="GET")
        try:
            with urllib_request.urlopen(request, timeout=15) as response:
                raw = response.read()
        except urllib_error.HTTPError as exc:
            if exc.code in (401, 403):
                raise ValueError(f"API key was rejected by {name}") from exc
            raise ValueError(f"{name} returned HTTP {exc.code}") from exc
        except (urllib_error.URLError, TimeoutError) as exc:
            raise ValueError(f"could not reach {name}") from exc
        try:
            return json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"{name} returned malformed JSON") from exc

    def list_openai_shaped(url: str, name: str) -> list[dict[str, Any]]:
        payload = fetch_json(
            url, {"Authorization": f"Bearer {api_key}"}, name
        )
        data = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(data, list):
            raise ValueError(f"{name} returned malformed JSON")
        entries: list[dict[str, Any]] = []
        for item in data:
            if not isinstance(item, dict):
                continue
            model_id = item.get("id")
            if not isinstance(model_id, str) or not model_id:
                continue
            created = item.get("created")
            if isinstance(created, bool) or not isinstance(created, int):
                created = None
            entries.append(
                {
                    "id": model_id,
                    "created": created,
                    "reasoning": reasoning_levels(name, model_id),
                }
            )
        return entries

    if provider == "anthropic":
        payload = fetch_json(
            "https://api.anthropic.com/v1/models?limit=1000",
            {
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
            },
            "anthropic",
        )
        data = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(data, list):
            raise ValueError("anthropic returned malformed JSON")
        models: list[dict[str, Any]] = []
        for item in data:
            if not isinstance(item, dict):
                continue
            model_id = item.get("id")
            if not isinstance(model_id, str) or not model_id:
                continue
            models.append(
                {
                    "id": model_id,
                    "created": parse_iso_created(item.get("created_at")),
                    "reasoning": reasoning_levels("anthropic", model_id, item),
                }
            )
    elif provider == "openai":
        models = [
            entry
            for entry in list_openai_shaped(
                "https://api.openai.com/v1/models", "openai"
            )
            if not any(token in entry["id"].lower() for token in openai_exclude)
        ]
    elif provider == "google":
        models = []
        page_token = None
        for _ in range(4):
            query = {"pageSize": "1000"}
            if page_token:
                query["pageToken"] = page_token
            url = (
                "https://generativelanguage.googleapis.com/v1beta/models?"
                + urllib_parse.urlencode(query)
            )
            payload = fetch_json(
                url, {"x-goog-api-key": api_key}, "google"
            )
            if not isinstance(payload, dict):
                raise ValueError("google returned malformed JSON")
            items = payload.get("models")
            if not isinstance(items, list):
                raise ValueError("google returned malformed JSON")
            for item in items:
                if not isinstance(item, dict):
                    continue
                methods = item.get("supportedGenerationMethods")
                if not isinstance(methods, list) or "generateContent" not in methods:
                    continue
                name = item.get("name")
                if not isinstance(name, str) or not name:
                    continue
                model_id = name.removeprefix("models/")
                if not model_id:
                    continue
                models.append(
                    {
                        "id": model_id,
                        "created": None,
                        "reasoning": reasoning_levels("google", model_id, item),
                    }
                )
            next_token = payload.get("nextPageToken")
            if not isinstance(next_token, str) or not next_token:
                break
            page_token = next_token
    elif provider == "xai":
        models = list_openai_shaped("https://api.x.ai/v1/models", "xai")
    elif provider == "deepseek":
        models = list_openai_shaped("https://api.deepseek.com/models", "deepseek")
    else:
        raise ValueError(f"unknown provider: {provider}")

    if any(entry["created"] is not None for entry in models):
        models.sort(key=lambda entry: entry["created"] or 0, reverse=True)
    return models
