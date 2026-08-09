"""Join what a model said it was thinking to what Mortal thought of the result.

`evaluate.py` grades a decision without ever seeing the model's reasoning, and the
decision log captures reasoning without knowing whether the move was good. Joined, they
answer the question a rating alone cannot: what was it thinking when it blundered.

Both sides label a decision with the same board coordinates - hand, honba, own turn
number, tiles left - so the join is exact rather than positional. That matters because
the two logs are not the same length: a decision the furo toggle passed on never reached
the model, and a decision with one legal action was never graded.
"""

from __future__ import annotations

import statistics
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any

Coords = tuple[int, int, int, int]


def _coords(record: dict[str, Any]) -> Coords | None:
    try:
        return (
            int(record["kyoku"]),
            int(record["honba"]),
            int(record["junme"]),
            int(record["tiles_left"]),
        )
    except (KeyError, TypeError, ValueError):
        return None


def prob_loss(entry: dict[str, Any]) -> float:
    """How much probability mass Mortal put on a better move than the one played."""
    details = entry.get("details") or []
    index = entry.get("actual_index")
    if not details or index is None or index >= len(details):
        return 0.0
    return float(details[0]["prob"]) - float(details[index]["prob"])


@dataclass
class ReasonedDecision:
    kyoku: int
    honba: int
    junme: int
    tiles_left: int
    player_id: int
    choice_label: str
    reasoning: str
    prob_loss: float
    is_equal: bool
    expected: dict[str, Any]
    actual: dict[str, Any]
    shanten: int
    reasoning_chars: int = 0
    reasoning_tokens: int = 0

    def __post_init__(self) -> None:
        self.reasoning_chars = len(self.reasoning)


@dataclass
class ReasoningReview:
    decisions: list[ReasonedDecision] = field(default_factory=list)
    graded: int = 0
    logged: int = 0
    unjoined_logged: int = 0

    @property
    def coverage(self) -> float:
        """Share of the model's own decisions that carry both reasoning and a grade."""
        return len(self.decisions) / self.logged if self.logged else 0.0

    def worst(self, limit: int = 10) -> list[ReasonedDecision]:
        return sorted(self.decisions, key=lambda d: -d.prob_loss)[:limit]

    def summary(self) -> dict[str, Any]:
        with_reasoning = [d for d in self.decisions if d.reasoning_chars > 0]
        matched = [d for d in self.decisions if d.is_equal]
        missed = [d for d in self.decisions if not d.is_equal]
        return {
            "joined": len(self.decisions),
            "graded": self.graded,
            "logged": self.logged,
            "coverage": self.coverage,
            "with_reasoning": len(with_reasoning),
            "mean_reasoning_chars": (
                statistics.mean(d.reasoning_chars for d in with_reasoning)
                if with_reasoning
                else 0.0
            ),
            # Whether thinking longer went with playing better is the whole point of the
            # join; reported as two means rather than a correlation so a handful of
            # decisions cannot manufacture a trend.
            "mean_reasoning_chars_when_matching_mortal": (
                statistics.mean(d.reasoning_chars for d in matched) if matched else 0.0
            ),
            "mean_reasoning_chars_when_not": (
                statistics.mean(d.reasoning_chars for d in missed) if missed else 0.0
            ),
            "mean_prob_loss": (
                statistics.mean(d.prob_loss for d in self.decisions)
                if self.decisions
                else 0.0
            ),
        }


def join(
    decisions: list[dict[str, Any]],
    review: dict[str, Any],
    player_id: int | None = None,
) -> ReasoningReview:
    """Pair each logged decision with Mortal's verdict on the same board position."""
    # Coordinates are not unique on their own: riichi is a two-step action, so declaring
    # and the discard that follows share a hand, turn and wall count. Both logs are
    # chronological per seat, so the nth decision at a coordinate matches the nth entry.
    entries: dict[Coords, deque[dict[str, Any]]] = defaultdict(deque)
    graded = 0
    for entry in review.get("entries", []):
        key = _coords(entry)
        if key is not None:
            entries[key].append(entry)
            graded += 1

    out = ReasoningReview(graded=graded)
    for record in decisions:
        if player_id is not None and record.get("player_id") != player_id:
            continue
        out.logged += 1
        key = _coords(record)
        queue = entries.get(key) if key is not None else None
        if not queue:
            out.unjoined_logged += 1
            continue
        entry = queue.popleft()
        out.decisions.append(
            ReasonedDecision(
                kyoku=key[0],
                honba=key[1],
                junme=key[2],
                tiles_left=key[3],
                player_id=int(record.get("player_id", -1)),
                choice_label=str(record.get("choice_label", "")),
                reasoning=str(record.get("raw_reasoning") or ""),
                prob_loss=prob_loss(entry),
                is_equal=bool(entry.get("is_equal")),
                expected=entry.get("expected", {}),
                actual=entry.get("actual", {}),
                shanten=int(entry.get("shanten", 0)),
                reasoning_tokens=int(
                    (record.get("usage") or {}).get("reasoning_tokens", 0) or 0
                ),
            )
        )
    return out
