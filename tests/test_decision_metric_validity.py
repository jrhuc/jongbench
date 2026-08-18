from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
for extra in (ROOT, ROOT / "environments" / "riichi_decision_v1"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

pytest.importorskip("verifiers")

from riichi_decision_v1.taskset import (  # noqa: E402
    RiichiDecisionConfig,
    RiichiDecisionData,
    RiichiDecisionTask,
    RiichiDecisionTaskset,
)


class _Trace(SimpleNamespace):
    def __init__(self, reply: str):
        super().__init__(last_reply=reply, num_turns=1)
        self.metrics: dict[str, float] = {}

    def record_metric(self, name: str, value: float) -> None:
        self.metrics[name] = value


def _task() -> RiichiDecisionTask:
    config = RiichiDecisionConfig()
    return RiichiDecisionTask(
        RiichiDecisionData(
            idx=0,
            name="metric-validity",
            prompt="choose",
            system_prompt="",
            position_id="sha256:" + "1" * 64,
            board_id="sha256:" + "2" * 64,
            prompt_id="sha256:" + "3" * 64,
            game_id="game",
            menu=["a", "b"],
            rewards=[1.0, 0.0],
            q_values=[3.0, 1.0],
            best_index=0,
            tags=["pushfold"],
            reviewer_confidence=0.01,
            info={
                "seat": 0,
                "kyoku": 0,
                "honba": 0,
                "junme": 0,
                "tiles_left": 69,
                "shanten": 1,
                "at_furiten": False,
            },
            state_hints=True,
        ),
        config.task,
    )


def test_malformed_answers_remain_in_tag_slices() -> None:
    trace = _Trace("not a choice")
    assert asyncio.run(_task().q_advantage(trace)) == 0.0
    assert trace.metrics["tag_pushfold"] == 0.0


def test_raw_and_normalised_loss_expose_position_stakes() -> None:
    task = _task()
    trace = _Trace('{"choice": 1}')
    assert asyncio.run(task.q_loss(trace)) == 2.0
    assert asyncio.run(task.normalised_q_loss(trace)) == 1.0
    assert asyncio.run(task.q_span(trace)) == 2.0


def test_policy_imitation_confidence_cannot_weight_mortal_scores() -> None:
    with pytest.raises(ValueError, match="policy imitation correctness"):
        list(RiichiDecisionTaskset(RiichiDecisionConfig(confidence_weight=True)).load())
    with pytest.raises(ValueError, match="policy imitation correctness"):
        list(RiichiDecisionTaskset(RiichiDecisionConfig(min_confidence=0.5)).load())
