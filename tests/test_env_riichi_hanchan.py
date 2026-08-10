from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
for extra in (ROOT, ROOT / "environments" / "riichi_hanchan_v1"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

pytest.importorskip("verifiers")

import verifiers.v1 as vf  # noqa: E402
from riichi_hanchan_v1.env import (  # noqa: E402
    SEATS,
    RiichiHanchanConfig,
    RiichiHanchanEnv,
    RiichiHanchanEnvConfig,
    RiichiHanchanTaskset,
)


class FakeTrace:
    def __init__(self) -> None:
        self.rewards: dict[str, float] = {}
        self.metrics: dict[str, float] = {}
        self.info: dict = {}

    def record_reward(self, name: str, value: float, weight: float = 1.0) -> None:
        self.rewards[name] = value

    def record_metric(self, name: str, value: float) -> None:
        self.metrics[name] = value


class FakeInteraction:
    """Answers every turn with the first legal option, recording what it was asked."""

    def __init__(self, task) -> None:
        self.task = task
        self.trace = FakeTrace()
        self.prompts: list[str] = []
        self.thread_ids: set[int] = set()
        self.closed = False

    async def turn(self, message: str) -> SimpleNamespace:
        import threading

        assert not self.closed, "turn() on a closed interaction"
        self.prompts.append(message)
        self.thread_ids.add(threading.get_ident())
        return SimpleNamespace(last_reply='{"choice": 0}', terminated=False)


class FakeAgent:
    """Hands out a fresh interaction per `interaction(task)` call, one per kyoku."""

    def __init__(self, interaction_cls=FakeInteraction) -> None:
        self.interaction_cls = interaction_cls
        self.interactions: list[FakeInteraction] = []

    def interaction(self, task):
        agent = self

        class _Ctx:
            async def __aenter__(self):
                inter = agent.interaction_cls(task)
                agent.interactions.append(inter)
                return inter

            async def __aexit__(self, *exc):
                agent.interactions[-1].closed = True
                return False

        return _Ctx()

    @property
    def prompts(self) -> list[str]:
        return [p for inter in self.interactions for p in inter.prompts]

    @property
    def thread_ids(self) -> set[int]:
        return {t for inter in self.interactions for t in inter.thread_ids}


class FakeAgents:
    def __init__(self, interaction_cls=FakeInteraction) -> None:
        self.seats = [FakeAgent(interaction_cls) for _ in SEATS]
        for name, agent in zip(SEATS, self.seats, strict=True):
            setattr(self, name, agent)


@pytest.fixture(scope="module")
def episode_dir(tmp_path_factory) -> Path:
    return tmp_path_factory.mktemp("hanchan-episodes")


@pytest.fixture(scope="module")
def played(episode_dir):
    env = RiichiHanchanEnv.__new__(RiichiHanchanEnv)
    env.config = RiichiHanchanEnvConfig(log_dir=str(episode_dir))
    agents = FakeAgents()
    task = next(iter(RiichiHanchanTaskset(RiichiHanchanConfig()).load()))
    asyncio.run(env.run(task, agents))
    return agents


def test_a_full_hanchan_is_played_through_the_interactions(played) -> None:
    for seat in played.seats:
        assert len(seat.prompts) > 50, len(seat.prompts)
    # Turns are marshalled back onto the event loop, not run on the arena's thread.
    assert all(len(seat.thread_ids) == 1 for seat in played.seats)
    assert len({tuple(seat.thread_ids) for seat in played.seats}) == 1


def test_each_kyoku_gets_a_fresh_interaction(played) -> None:
    """A kyoku opens with a full board and never drags earlier kyoku along: one
    interaction per kyoku, opening turn first, delta turns after."""
    for seat in played.seats:
        assert len(seat.interactions) > 1, "expected one interaction per kyoku"
        for inter in seat.interactions:
            assert inter.prompts, "an interaction was opened without a turn"
            assert inter.prompts[0].startswith("Round:")
            for later in inter.prompts[1:]:
                assert later.startswith("Since your last action:")
            assert inter.closed


def test_placement_rewards_are_zero_sum_and_ranked(played) -> None:
    def seat_traces(seat):
        return [inter.trace for inter in seat.interactions]

    # Every kyoku trace of a seat carries the seat's final placement reward.
    for seat in played.seats:
        assert len({t.rewards["placement"] for t in seat_traces(seat)}) == 1

    rewards = [seat_traces(seat)[-1].rewards["placement"] for seat in played.seats]
    assert sorted(rewards) == pytest.approx([0.0, 1 / 3, 2 / 3, 1.0])
    assert sum(rewards) == pytest.approx(2.0)

    infos = [seat_traces(seat)[-1].info["hanchan"] for seat in played.seats]
    placements = [info["placement"] for info in infos]
    assert sorted(placements) == [1, 2, 3, 4]
    scores = [info["score"] for info in infos]
    assert sum(scores) == 100000

    # Equal scores are broken by seat, so only the strict ordering is guaranteed.
    for i in range(4):
        for j in range(4):
            if scores[i] > scores[j]:
                assert placements[i] < placements[j], (scores, placements)

    best = max(range(4), key=lambda s: scores[s])
    assert rewards[best] == pytest.approx(1.0)


def test_each_seat_records_its_own_play(played) -> None:
    for name, seat in zip(SEATS, played.seats, strict=True):
        last = seat.interactions[-1].trace
        assert last.info["hanchan"]["seat"] == name
        assert last.info["hanchan"]["kyoku_count"] == len(seat.interactions)
        assert last.metrics["decisions"] == len(seat.prompts)
        assert last.metrics["fallbacks"] == 0.0
        kyoku = [inter.trace.info["hanchan"]["kyoku"] for inter in seat.interactions]
        assert kyoku == list(range(len(seat.interactions)))


def test_episode_persists_as_a_jongbench_run_dir(played, episode_dir) -> None:
    import json

    run_dir = episode_dir / "hanchan-00000"
    logs = list((run_dir / "logs").glob("*.json.gz"))
    assert len(logs) == 1, "expected the arena to write the mjai log"

    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    assert config["names"] == list(SEATS)
    assert sorted(config["final"]["placements"].values()) == [1, 2, 3, 4]

    for name, seat in zip(SEATS, played.seats, strict=True):
        lines = [
            json.loads(line)
            for line in (run_dir / "decisions" / f"{name}.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        assert len(lines) == len(seat.prompts)
        assert all("choice" in record for record in lines)


class ToolUsingInteraction(FakeInteraction):
    """Records the SeatState served for each turn and saves one note per turn, the way
    a note() tool push would."""

    def __init__(self, task) -> None:
        super().__init__(task)
        self.states: list = []

    async def turn(self, message: str) -> SimpleNamespace:
        state = self.trace.state
        self.states.append(state)
        state.notes.append(f"note-{len(state.notes)}")
        return await super().turn(message)


@pytest.fixture(scope="module")
def played_tools():
    env = RiichiHanchanEnv.__new__(RiichiHanchanEnv)
    env.config = RiichiHanchanEnvConfig(tools=True)
    agents = FakeAgents(ToolUsingInteraction)
    task = next(iter(RiichiHanchanTaskset(RiichiHanchanConfig()).load()))
    asyncio.run(env.run(task, agents))
    return agents


def test_tools_mode_serves_the_decision_snapshot(played_tools) -> None:
    from riichi_hanchan_v1.env import SeatToolsTask
    from riichi_hanchan_v1.tools import SeatState, SeatToolset

    toolsets = SeatToolsTask.toolsets(None)
    assert len(toolsets) == 1
    assert isinstance(toolsets[0], SeatToolset)
    assert toolsets[0].config.colocated

    for seat in played_tools.seats:
        for inter in seat.interactions:
            assert isinstance(inter.task, SeatToolsTask)
            assert "TOOLS" in inter.task.data.system_prompt
            for state in inter.states:
                assert isinstance(state, SeatState)
                assert state.board.startswith("Round:")
                assert set(state.discards) == {"P0", "P1", "P2", "P3"}
                assert state.waits
        assert any(state.simulate for inter in seat.interactions for state in inter.states)
        assert not any("Engine-derived state hints" in p for p in seat.prompts)


def test_notes_survive_kyoku_resets(played_tools) -> None:
    for seat in played_tools.seats:
        assert len(seat.interactions) > 1
        seeded = [len(inter.states[0].notes) - 1 for inter in seat.interactions]
        assert seeded[0] == 0
        turns_before = 0
        for inter, seen in zip(seat.interactions, seeded, strict=True):
            assert seen == turns_before
            turns_before += len(inter.states)
        assert seat.interactions[-1].trace.metrics["notes_saved"] == turns_before


def test_seat_toolset_answers_from_its_state() -> None:
    from riichi_hanchan_v1.tools import SeatState, SeatToolset, normalize_tile

    toolset = SeatToolset(vf.ToolsetConfig(colocated=True))
    toolset._inert_state = SeatState(
        board="Round: East 1 (honba 0), kyotaku 0, dealer P0",
        discards={f"P{i}": "riichi no; discards -" for i in range(4)},
        waits="current structure: tenpai (0-shanten); waits: 3p; furiten: no",
        simulate={"1m": "1-shanten", "5p(red)": "tenpai (0-shanten); waits 3p; furiten no"},
    )

    assert toolset.board().startswith("Round: East 1")
    assert toolset.discards(2) == "P2: riichi no; discards -"
    assert toolset.discards(7) == "player must be 0, 1, 2 or 3"
    assert toolset.waits().startswith("current structure")
    assert toolset.simulate("0p") == "after discarding 5p(red): tenpai (0-shanten); waits 3p; furiten no"
    assert toolset.simulate("5pr").startswith("after discarding 5p(red)")
    assert "not a legal discard now" in toolset.simulate("9s")
    assert "1m 5p(red)" in toolset.simulate("9s")

    assert toolset.notes() == "no notes saved"
    assert toolset.note("P2 folds early") == "saved (1 notes)"
    assert toolset.note("  ") == "empty note ignored"
    assert toolset.notes() == "0: P2 folds early"

    assert normalize_tile("e") == "E"
    assert normalize_tile("0m") == "5m(red)"
    assert normalize_tile("5Sr") == "5s(red)"
    assert normalize_tile(" 3p ") == "3p"


def test_taskset_is_an_infinite_seeded_generator() -> None:
    taskset = RiichiHanchanTaskset(RiichiHanchanConfig())
    assert taskset.INFINITE
    tasks = [t for t, _ in zip(taskset.load(), range(3), strict=False)]
    seeds = [t.data.info["seed"] for t in tasks]
    assert len(set(seeds)) == 3
    assert all(t.data.prompt is None for t in tasks)
    assert all(isinstance(t, vf.Task) for t in tasks)
