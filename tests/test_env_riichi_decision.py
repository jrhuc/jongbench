from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
for extra in (ROOT, ROOT / "environments" / "riichi_decision_v1"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

pytest.importorskip("verifiers")

from jongbench import positions  # noqa: E402
from riichi_decision_v1.taskset import (  # noqa: E402
    RiichiDecisionConfig,
    RiichiDecisionTaskset,
)


def _position(**overrides) -> dict:
    data = {
        "player_id": 2,
        "events": [
            {
                "type": "start_kyoku",
                "bakaze": "E",
                "kyoku": 1,
                "honba": 0,
                "kyotaku": 0,
                "oya": 0,
                "dora_marker": "3p",
                "scores": [25000, 25000, 25000, 25000],
                "tehais": [
                    ["?"] * 13,
                    ["?"] * 13,
                    ["1m", "2m", "3m", "4p", "5p", "6p", "7s", "8s", "9s", "E", "E", "P", "P"],
                    ["?"] * 13,
                ],
            },
            {"type": "tsumo", "actor": 2, "pai": "9m"},
        ],
        "menu": ["discard 1m", "discard 9m (drawn)", "discard P"],
        "rewards": [0.25, 1.0, 0.0],
        "best_index": 1,
        "kyoku": 0,
        "honba": 0,
        "junme": 1,
        "tiles_left": 69,
        "shanten": 2,
        "at_furiten": False,
        "metadata": {},
    }
    data.update(overrides)
    return data


@pytest.fixture
def taskset(tmp_path) -> RiichiDecisionTaskset:
    bank = tmp_path / "bank.jsonl"
    bank.write_text(json.dumps(_position()) + "\n" + json.dumps(_position(player_id=2)) + "\n")
    return RiichiDecisionTaskset(RiichiDecisionConfig(bank=str(bank)))


def _trace(reply: str) -> SimpleNamespace:
    return SimpleNamespace(last_reply=reply, num_turns=1)


def test_taskset_renders_a_prompt_and_carries_the_grading(taskset) -> None:
    tasks = list(taskset.load())
    assert len(tasks) == 2
    task = tasks[0]
    assert "Choose your action:" in (task.data.prompt or "")
    assert task.data.menu == ["discard 1m", "discard 9m (drawn)", "discard P"]
    assert task.data.rewards == [0.25, 1.0, 0.0]
    assert task.data.info["seat"] == 2


def test_reward_is_mortals_opinion_of_the_chosen_action(taskset) -> None:
    task = next(iter(taskset.load()))
    for reply, expected in (
        ('{"choice": 1}', 1.0),
        ('{"choice": 0}', 0.25),
        ('{"choice": 2}', 0.0),
    ):
        assert asyncio.run(task.q_advantage(_trace(reply))) == pytest.approx(expected)


def test_an_unusable_reply_scores_zero_rather_than_erroring(taskset) -> None:
    task = next(iter(taskset.load()))
    for reply in ("I would discard the 9m", '{"choice": 7}', "", '{"choice": "1"}'):
        assert asyncio.run(task.q_advantage(_trace(reply))) == 0.0
        assert asyncio.run(task.answered(_trace(reply))) == 0.0


def test_match_metric_tracks_mortals_own_choice(taskset) -> None:
    task = next(iter(taskset.load()))
    assert asyncio.run(task.matched_mortal(_trace('{"choice": 1}'))) == 1.0
    assert asyncio.run(task.matched_mortal(_trace('{"choice": 0}'))) == 0.0


def test_missing_bank_names_the_command_that_builds_one(tmp_path) -> None:
    taskset = RiichiDecisionTaskset(RiichiDecisionConfig(bank=str(tmp_path / "nope.jsonl")))
    with pytest.raises(FileNotFoundError, match="jongbench positions"):
        list(taskset.load())


def test_shipped_sample_bank_runs_out_of_the_box() -> None:
    """The default config needs no bank-building step: 128 committed positions."""
    from riichi_decision_v1.taskset import SAMPLE_BANK

    taskset = RiichiDecisionTaskset(RiichiDecisionConfig())
    assert Path(taskset.config.bank) == SAMPLE_BANK
    tasks = list(taskset.load())
    assert len(tasks) == 128
    for task in tasks:
        assert "Choose your action:" in (task.data.prompt or "")
        assert len(task.data.menu) == len(task.data.rewards) >= 2
        assert all(0.0 <= reward <= 1.0 for reward in task.data.rewards)
        assert task.data.rewards[task.data.best_index] == max(task.data.rewards)


def test_package_bundles_the_chat_harness_as_its_default() -> None:
    """`eval riichi_decision_v1` must run the plain chat loop, not a bash agent:
    verifiers takes a taskset package's exported Harness subclass as the default."""
    from verifiers.v1.harnesses.null import NullHarness
    from verifiers.v1.utils.loaders import default_harness_id, harness_class

    assert default_harness_id("riichi_decision_v1") == "riichi_decision_v1"
    assert harness_class("riichi_decision_v1") is NullHarness


def test_a_real_bank_round_trips_through_the_taskset(tmp_path) -> None:
    """The bank the CLI writes must be loadable verbatim: Position -> JSON -> task."""
    position = positions.Position.from_dict(_position())
    bank = tmp_path / "bank.jsonl"
    bank.write_text(json.dumps(position.to_dict(), separators=(",", ":")) + "\n")
    task = next(iter(RiichiDecisionTaskset(RiichiDecisionConfig(bank=str(bank))).load()))
    assert task.data.prompt == position.prompt()
