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

    def __init__(self) -> None:
        self.trace = FakeTrace()
        self.prompts: list[str] = []
        self.thread_ids: set[int] = set()

    async def turn(self, message: str) -> SimpleNamespace:
        import threading

        self.prompts.append(message)
        self.thread_ids.add(threading.get_ident())
        return SimpleNamespace(last_reply='{"choice": 0}', terminated=False)


class FakeAgent:
    def __init__(self, interaction: FakeInteraction) -> None:
        self._interaction = interaction

    def interaction(self, task):
        agent = self

        class _Ctx:
            async def __aenter__(self):
                return agent._interaction

            async def __aexit__(self, *exc):
                return False

        return _Ctx()


class FakeAgents:
    def __init__(self) -> None:
        self.interactions = [FakeInteraction() for _ in SEATS]
        for name, interaction in zip(SEATS, self.interactions, strict=True):
            setattr(self, name, FakeAgent(interaction))


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
    for interaction in played.interactions:
        assert len(interaction.prompts) > 50, len(interaction.prompts)
    # Turns are marshalled back onto the event loop, not run on the arena's thread.
    assert all(len(i.thread_ids) == 1 for i in played.interactions)
    assert len({tuple(i.thread_ids) for i in played.interactions}) == 1


def test_seats_get_an_opening_board_then_deltas(played) -> None:
    for interaction in played.interactions:
        openings = [p for p in interaction.prompts if p.startswith("Round:")]
        deltas = [p for p in interaction.prompts if p.startswith("Since your last action:")]
        assert openings, "expected a full board at the start of each kyoku"
        assert deltas, "expected delta turns after the opening"
        assert len(openings) + len(deltas) == len(interaction.prompts)
        # One opening per kyoku this seat acted in, not one per decision.
        assert len(openings) < len(deltas)


def test_placement_rewards_are_zero_sum_and_ranked(played) -> None:
    rewards = [i.trace.rewards["placement"] for i in played.interactions]
    assert sorted(rewards) == pytest.approx([0.0, 1 / 3, 2 / 3, 1.0])
    assert sum(rewards) == pytest.approx(2.0)

    placements = [i.trace.info["hanchan"]["placement"] for i in played.interactions]
    assert sorted(placements) == [1, 2, 3, 4]
    scores = [i.trace.info["hanchan"]["score"] for i in played.interactions]
    assert sum(scores) == 100000

    # Equal scores are broken by seat, so only the strict ordering is guaranteed.
    for i in range(4):
        for j in range(4):
            if scores[i] > scores[j]:
                assert placements[i] < placements[j], (scores, placements)

    best = max(range(4), key=lambda s: scores[s])
    assert played.interactions[best].trace.rewards["placement"] == pytest.approx(1.0)


def test_each_seat_records_its_own_play(played) -> None:
    for name, interaction in zip(SEATS, played.interactions, strict=True):
        assert interaction.trace.info["hanchan"]["seat"] == name
        assert interaction.trace.metrics["decisions"] == len(interaction.prompts)
        assert interaction.trace.metrics["fallbacks"] == 0.0


def test_episode_persists_as_a_jongbench_run_dir(played, episode_dir) -> None:
    import json

    run_dir = episode_dir / "hanchan-00000"
    logs = list((run_dir / "logs").glob("*.json.gz"))
    assert len(logs) == 1, "expected the arena to write the mjai log"

    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    assert config["names"] == list(SEATS)
    assert sorted(config["final"]["placements"].values()) == [1, 2, 3, 4]

    for name, interaction in zip(SEATS, played.interactions, strict=True):
        lines = [
            json.loads(line)
            for line in (run_dir / "decisions" / f"{name}.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        assert len(lines) == len(interaction.prompts)
        assert all("choice" in record for record in lines)


def test_taskset_is_an_infinite_seeded_generator() -> None:
    taskset = RiichiHanchanTaskset(RiichiHanchanConfig())
    assert taskset.INFINITE
    tasks = [t for t, _ in zip(taskset.load(), range(3), strict=False)]
    seeds = [t.data.info["seed"] for t in tasks]
    assert len(set(seeds)) == 3
    assert all(t.data.prompt is None for t in tasks)
    assert all(isinstance(t, vf.Task) for t in tasks)
