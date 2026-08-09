"""Drive a jongbench seat from something other than a provider API call.

The arena is a synchronous Rust loop that calls each engine at its decision points, while
a verifiers `Env` is async and drives its agents turn by turn. `CallbackProvider` is the
seam: it is provider-shaped, so `LLMEngine` keeps building prompts, tracking the per-kyoku
delta and handling retries exactly as it does in a normal run, but the actual "call" is a
callable the caller supplies. The web UI already proves the shape works - `WebHumanIO`
blocks a seat on a `threading.Event` while the HTTP server answers it.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from . import engines, providers


def newest_user_text(messages: list[dict[str, Any]]) -> str:
    """The text of the newest user turn. Content may be a plain string or a block list
    when it carries a cache breakpoint."""
    content = next(m["content"] for m in reversed(messages) if m["role"] == "user")
    if isinstance(content, str):
        return content
    return "".join(block.get("text", "") for block in content)


class CallbackProvider:
    """Provider-shaped adapter that hands the newest user turn to `ask`.

    Only the newest turn is passed on: the consumer owns the transcript, so resending the
    history jongbench also tracks would duplicate it.
    """

    def __init__(self, ask: Callable[[str], str], name: str = "callback") -> None:
        self.ask = ask
        self.model = name
        self.reasoning = None

    def __repr__(self) -> str:
        return f"{type(self).__name__}(model={self.model!r})"

    def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        max_tokens: int = 1200,
        temperature: float | None = None,
    ) -> providers.Completion:
        del max_tokens, temperature
        return providers.Completion(text=self.ask(newest_user_text(messages)) or "")


def make_bridged_engine(
    name: str,
    ask: Callable[[str], str],
    *,
    decision_log: engines.DecisionSink | None = None,
    state_hints: bool = True,
    spectator: Any | None = None,
) -> engines.LLMEngine:
    """An `LLMEngine` whose calls are answered by `ask`.

    Concurrency is pinned to 1: a bridged seat holds one conversation, and a caller that
    drives several games through the same engine would interleave their turns into it.
    """
    engine = engines.LLMEngine(
        name,
        "bridge/callback",
        decision_log=decision_log,
        concurrency=1,
        spectator=spectator,
        state_hints=state_hints,
        conversational=True,
    )
    engine.provider = CallbackProvider(ask, name=name)
    return engine
