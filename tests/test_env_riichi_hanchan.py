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


def _fake_call(index: int) -> SimpleNamespace:
    """A `ModelCall`-shaped record: the env reads `usage` off the trace's calls."""
    usage = SimpleNamespace(
        prompt_tokens=100 + index,
        completion_tokens=10,
        cached_input_tokens=5,
        reasoning_tokens=7,
    )
    return SimpleNamespace(usage=usage)


class FakeTrace:
    def __init__(self) -> None:
        self.rewards: dict[str, float] = {}
        self.metrics: dict[str, float] = {}
        self.info: dict = {}
        self.calls: list = []

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
        # A real turn appends to the trace's running call list and answers with the
        # messages it just added.
        self.trace.calls.append(_fake_call(len(self.prompts)))
        return SimpleNamespace(
            last_reply='{"choice": 0}',
            terminated=False,
            messages=[
                SimpleNamespace(reasoning_content=f"thought {len(self.prompts)}")
            ],
        )


class FakeAgent:
    """Hands out a fresh interaction per `interaction(task)` call, one per kyoku."""

    def __init__(self, interaction_cls=FakeInteraction, model="fake/model") -> None:
        self.interaction_cls = interaction_cls
        # A real vf.Agent always carries a resolved model; the env reads it back.
        self.config = SimpleNamespace(model=model)
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
    def __init__(self, interaction_cls=FakeInteraction, config=None) -> None:
        config = config or RiichiHanchanEnvConfig()
        models = [getattr(config, name).model or "fake/model" for name in SEATS]
        self.seats = [FakeAgent(interaction_cls, model) for model in models]
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
    assert config["max_tool_calls"] == RiichiHanchanEnvConfig().max_tool_calls
    assert config["weights"] == RiichiHanchanEnvConfig().weights
    header = json.loads(
        (run_dir / "journal.jsonl").read_text(encoding="utf-8").splitlines()[0]
    )
    assert header["max_tool_calls"] == config["max_tool_calls"]
    assert header["weights"] == config["weights"]

    for name, seat in zip(SEATS, played.seats, strict=True):
        lines = [
            json.loads(line)
            for line in (run_dir / "decisions" / f"{name}.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        assert len(lines) == len(seat.prompts)
        assert all("choice" in record for record in lines)


def test_decisions_carry_what_the_turn_cost(played, episode_dir) -> None:
    """A bridged seat is billed like a direct provider call: each decision records the
    tokens and reasoning of the turn that produced it, and only that turn's."""
    import json

    run_dir = episode_dir / "hanchan-00000"
    for name in SEATS:
        lines = (
            (run_dir / "decisions" / f"{name}.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        )
        records = [json.loads(line) for line in lines]
        assert records
        for record in records:
            # The fake numbers its turns within a kyoku; a decision must carry its own.
            turn = int(record["raw_reasoning"].removeprefix("thought "))
            # jongbench folds cache reads into input_tokens; verifiers splits them out.
            assert record["usage"] == {
                "input_tokens": 100 + turn + 5,
                "output_tokens": 10,
                "cached_input_tokens": 5,
                "reasoning_tokens": 7,
            }


def test_metered_turns_carry_the_provider_s_price() -> None:
    """A provider meters some calls and not others; the seam sums what is reported and
    stays absent otherwise, because a 0.0 would read as free rather than unknown."""
    from riichi_hanchan_v1.env import _usage_of

    def metered(cost: float) -> SimpleNamespace:
        call = _fake_call(0)
        call.usage.cost = cost
        return call

    assert _usage_of([metered(0.0004), metered(0.0002)])["cost"] == pytest.approx(
        0.0006
    )
    assert _usage_of([metered(0.0004), _fake_call(1)])["cost"] == pytest.approx(0.0004)
    assert "cost" not in _usage_of([_fake_call(1)])


class ToolUsingInteraction(FakeInteraction):
    """Records the SeatState served for each turn and saves one note per turn, the way
    a note() tool push would."""

    def __init__(self, task) -> None:
        super().__init__(task)
        self.states: list = []
        self.served_budgets: list = []

    async def turn(self, message: str) -> SimpleNamespace:
        state = self.trace.state
        self.states.append(state)
        self.served_budgets.append(state.budget)
        state.notes.append(f"note-{len(state.notes)}")
        segment = await super().turn(message)
        # Every other turn looks the board up before answering, the way a tool-using
        # seat does: the harness leaves the tool-calling message in the segment.
        if len(self.prompts) % 2 == 0:
            segment.messages.insert(
                0,
                SimpleNamespace(
                    tool_calls=[
                        SimpleNamespace(name="board"),
                        SimpleNamespace(name="waits"),
                    ]
                ),
            )
            state.budget = 0  # and it queried until the decision's budget ran out
        return segment


class RetryingToolInteraction(ToolUsingInteraction):
    async def turn(self, message: str) -> SimpleNamespace:
        import threading

        state = self.trace.state
        self.states.append(state)
        self.served_budgets.append(state.budget)
        self.prompts.append(message)
        self.thread_ids.add(threading.get_ident())
        self.trace.calls.append(_fake_call(len(self.prompts)))
        retry = message.startswith("Your previous reply was invalid:")
        if not retry and state.budget is not None:
            state.budget -= 1
        return SimpleNamespace(
            last_reply='{"choice": 0}' if retry else "invalid",
            terminated=False,
            messages=[SimpleNamespace(reasoning_content="")],
        )


class ForcedFallbackInteraction(FakeInteraction):
    async def turn(self, message: str) -> SimpleNamespace:
        segment = await super().turn(message)
        self.trace.stop_condition = "tool_budget_exhausted"
        segment.terminated = True
        return segment


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
        assert any(
            state.simulate for inter in seat.interactions for state in inter.states
        )
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


def test_tool_calls_are_counted_per_seat(played_tools) -> None:
    for seat in played_tools.seats:
        querying = sum(len(inter.states) // 2 for inter in seat.interactions)
        assert querying
        metrics = seat.interactions[-1].trace.metrics
        assert metrics["tool_turns"] == querying
        assert metrics["tool_calls"] == 2 * querying
        assert seat.interactions[-1].trace.info["hanchan"]["tools"] == {
            "board": querying,
            "waits": querying,
        }


def test_each_decision_gets_its_budget_back(played_tools) -> None:
    default = RiichiHanchanEnvConfig().max_tool_calls
    for seat in played_tools.seats:
        # The fake drains the budget on every querying turn; the next decision must
        # still open with a full one or a seat would go mute for the rest of the kyoku.
        assert all(
            budget == default
            for inter in seat.interactions
            for budget in inter.served_budgets
        )
        querying = sum(len(inter.states) // 2 for inter in seat.interactions)
        assert seat.interactions[-1].trace.metrics["budget_spent"] == querying


def test_engine_retry_keeps_the_same_decision_budget() -> None:
    env = RiichiHanchanEnv.__new__(RiichiHanchanEnv)
    env.config = RiichiHanchanEnvConfig(tools=True, max_tool_calls=4)
    agents = FakeAgents(RetryingToolInteraction, env.config)
    task = next(iter(RiichiHanchanTaskset(RiichiHanchanConfig()).load()))
    asyncio.run(env.run(task, agents))

    for seat in agents.seats:
        retries = [
            index
            for interaction in seat.interactions
            for index, prompt in enumerate(interaction.prompts)
            if prompt.startswith("Your previous reply was invalid:")
        ]
        assert retries
        for interaction in seat.interactions:
            for index, prompt in enumerate(interaction.prompts):
                if prompt.startswith("Your previous reply was invalid:"):
                    assert interaction.served_budgets[index - 1 : index + 1] == [4, 3]


def test_repeated_over_budget_tools_force_fallback_without_aborting() -> None:
    env = RiichiHanchanEnv.__new__(RiichiHanchanEnv)
    env.config = RiichiHanchanEnvConfig(tools=True, max_tool_calls=1)
    agents = FakeAgents(ForcedFallbackInteraction, env.config)
    task = next(iter(RiichiHanchanTaskset(RiichiHanchanConfig()).load()))
    asyncio.run(env.run(task, agents))

    for seat in agents.seats:
        kyokus = [
            interaction.trace.info["hanchan"]["kyoku"]
            for interaction in seat.interactions
        ]
        assert kyokus == sorted(kyokus)
        assert len(set(kyokus)) < len(kyokus)
        assert {
            interaction.trace.info["hanchan"]["kyoku_count"]
            for interaction in seat.interactions
        } == {len(set(kyokus))}
        assert all(
            interaction.prompts and interaction.prompts[0].startswith("Round:")
            for interaction in seat.interactions
        )
        assert seat.interactions[-1].trace.metrics["fallbacks"] > 0


def test_seat_toolset_answers_from_its_state() -> None:
    from riichi_hanchan_v1.tools import SeatState, SeatToolset, normalize_tile

    toolset = SeatToolset(vf.ToolsetConfig(colocated=True))
    toolset._inert_state = SeatState(
        board="Round: East 1 (honba 0), kyotaku 0, dealer P0",
        discards={f"P{i}": "riichi no; discards -" for i in range(4)},
        waits="current structure: tenpai (0-shanten); waits: 3p; furiten: no",
        simulate={
            "1m": "1-shanten",
            "5p(red)": "tenpai (0-shanten); waits 3p; furiten no",
        },
    )

    assert toolset.board().startswith("Round: East 1")
    assert toolset.discards(2) == "P2: riichi no; discards -"
    assert toolset.discards(7) == "player must be 0, 1, 2 or 3"
    assert toolset.waits().startswith("current structure")
    assert (
        toolset.simulate("0p")
        == "after discarding 5p(red): tenpai (0-shanten); waits 3p; furiten no"
    )
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


def test_a_budgeted_decision_stops_answering_when_it_runs_out() -> None:
    from riichi_hanchan_v1.tools import BUDGET_SPENT, SeatState, SeatToolset

    toolset = SeatToolset(vf.ToolsetConfig(colocated=True))
    toolset._inert_state = SeatState(board="Round: East 1", waits="tenpai", budget=2)

    assert toolset.board().startswith("Round:")
    assert toolset.waits() == "tenpai"
    assert toolset.board() == BUDGET_SPENT
    assert toolset.note("read me later") == BUDGET_SPENT
    assert toolset.state.notes == []  # a refused call must not have a side effect
    assert toolset.state.refused_tool_calls == 2
    from riichi_hanchan_v1.env import SeatToolsTask

    trace = SimpleNamespace(state=toolset.state)
    assert asyncio.run(SeatToolsTask.tool_budget_exhausted(None, trace))


def test_journal_read_trims_and_guards(tmp_path) -> None:
    from riichi_hanchan_v1.env import _Journal, _journal_header, _read_journal

    models = ["fake/model"] * len(SEATS)
    header = _journal_header(7, RiichiHanchanEnvConfig(), models, 0)
    path = tmp_path / "journal.jsonl"
    journal = _Journal(path, header)
    for seat, kyoku, honba in [
        ("seat0", 1, 0),
        ("seat1", 1, 0),
        ("seat0", 1, 1),
        ("seat0", 2, 0),
    ]:
        journal.append(
            seat, {"menu": ["x"], "choice": 0, "kyoku": kyoku, "honba": honba}
        )
    journal.close()

    rows = _read_journal(path, header)
    assert [(r["kyoku"], r["honba"]) for r in rows] == [(1, 0), (1, 0), (1, 1)]

    compacted = tmp_path / "compacted.jsonl"
    journal = _Journal(compacted, header, rows)
    journal.close()
    assert _read_journal(compacted, header) == rows
    journal = _Journal(compacted, header, rows)
    journal.append("seat0", {"menu": ["x"], "choice": 0, "kyoku": 2, "honba": 0})
    journal.close()
    assert _read_journal(compacted, header) == rows

    original = path.read_bytes()
    stale = _journal_header(8, RiichiHanchanEnvConfig(), models, 0)
    with pytest.raises(ValueError, match="different hanchan configuration"):
        _read_journal(path, stale)
    rotated = _journal_header(7, RiichiHanchanEnvConfig(), models, 1)
    with pytest.raises(ValueError, match="different hanchan configuration"):
        _read_journal(path, rotated)
    assert path.read_bytes() == original
    assert _read_journal(tmp_path / "missing.jsonl", header) == []

    with path.open("a", encoding="utf-8") as handle:
        handle.write("{cut mid-write")
    assert len(_read_journal(path, header)) == 3

    finished = tmp_path / "finished.jsonl"
    journal = _Journal(finished, header)
    journal.append("seat0", {"menu": ["x"], "choice": 0, "kyoku": 1, "honba": 0})
    journal.append("seat0", {"menu": ["x"], "choice": 0, "kyoku": 2, "honba": 0})
    journal.finish()
    journal.close()
    assert len(_read_journal(finished, header)) == 2


@pytest.fixture(scope="module")
def resumed(played, episode_dir, tmp_path_factory):
    import json

    source = episode_dir / "hanchan-00000" / "journal.jsonl"
    lines = source.read_text(encoding="utf-8").splitlines()
    assert json.loads(lines[-1]) == {"end": True}
    cut = 1 + int((len(lines) - 2) * 0.7)
    rows = [json.loads(line) for line in lines[1:cut]]
    hands_recorded = {(row["kyoku"], row["honba"]) for row in rows}

    resume_root = tmp_path_factory.mktemp("hanchan-resume")
    run_dir = resume_root / "hanchan-00000"
    run_dir.mkdir()
    (run_dir / "journal.jsonl").write_text(
        "\n".join(lines[:cut]) + "\n", encoding="utf-8"
    )

    env = RiichiHanchanEnv.__new__(RiichiHanchanEnv)
    env.config = RiichiHanchanEnvConfig(log_dir=str(resume_root))
    agents = FakeAgents()
    task = next(iter(RiichiHanchanTaskset(RiichiHanchanConfig()).load()))
    asyncio.run(env.run(task, agents))
    return agents, resume_root, len(hands_recorded) - 1


def test_resume_replays_the_journal_and_goes_live(played, resumed) -> None:
    agents, resume_root, hands_replayed = resumed
    live = sum(len(seat.prompts) for seat in agents.seats)
    full = sum(len(seat.prompts) for seat in played.seats)
    assert 0 < live < full

    # Replay reproduces the same game, so the standings match the original run.
    for orig, seat in zip(played.seats, agents.seats, strict=True):
        before = orig.interactions[-1].trace.info["hanchan"]
        after = seat.interactions[-1].trace.info["hanchan"]
        assert (before["placement"], before["score"]) == (
            after["placement"],
            after["score"],
        )
        assert seat.interactions[0].trace.info["hanchan"]["kyoku"] == hands_replayed

    journal = (resume_root / "hanchan-00000" / "journal.jsonl").read_text(
        encoding="utf-8"
    )
    assert journal.splitlines()[-1] == '{"end":true}'


def test_resume_artifacts_cover_the_whole_hanchan(played, resumed) -> None:
    import json

    agents, resume_root, _ = resumed
    for name, orig in zip(SEATS, played.seats, strict=True):
        lines = (
            (resume_root / "hanchan-00000" / "decisions" / f"{name}.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        )
        assert len(lines) == len(orig.prompts)
        assert all("choice" in json.loads(line) for line in lines)


def test_finished_journal_replays_the_episode_for_free(
    played, episode_dir, tmp_path_factory
) -> None:
    import json

    replay_root = tmp_path_factory.mktemp("hanchan-replay")
    run_dir = replay_root / "hanchan-00000"
    run_dir.mkdir()
    source = episode_dir / "hanchan-00000"
    (run_dir / "journal.jsonl").write_text(
        (source / "journal.jsonl").read_text(encoding="utf-8"), encoding="utf-8"
    )

    env = RiichiHanchanEnv.__new__(RiichiHanchanEnv)
    env.config = RiichiHanchanEnvConfig(log_dir=str(replay_root))
    agents = FakeAgents()
    task = next(iter(RiichiHanchanTaskset(RiichiHanchanConfig()).load()))
    asyncio.run(env.run(task, agents))

    assert all(seat.interactions == [] for seat in agents.seats)
    original = json.loads((source / "config.json").read_text(encoding="utf-8"))
    replayed = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    assert replayed["final"] == original["final"]


WEIGHTS = ROOT / "weights" / "mortal.pth"
mortal_weights = pytest.mark.skipif(
    not WEIGHTS.exists(), reason="needs weights/mortal.pth"
)


def _mortal_config(log_dir: Path) -> RiichiHanchanEnvConfig:
    return RiichiHanchanEnvConfig(
        seat3=vf.AgentConfig(harness={"id": "null"}, model="mortal"),
        log_dir=str(log_dir),
        weights=str(WEIGHTS),
    )


@pytest.fixture(scope="module")
def played_mortal(tmp_path_factory):
    root = tmp_path_factory.mktemp("hanchan-mortal")
    env = RiichiHanchanEnv.__new__(RiichiHanchanEnv)
    env.config = _mortal_config(root)
    agents = FakeAgents(config=env.config)
    task = next(iter(RiichiHanchanTaskset(RiichiHanchanConfig()).load()))
    asyncio.run(env.run(task, agents))
    return agents, root / "hanchan-00000"


@mortal_weights
def test_mortal_control_seat_opens_no_interactions(played_mortal) -> None:
    agents, _ = played_mortal
    assert agents.seat3.interactions == []
    steps = (0.0, 1 / 3, 2 / 3, 1.0)
    rewards = []
    for seat in agents.seats[:3]:
        assert len(seat.prompts) > 50
        per_kyoku = {inter.trace.rewards["placement"] for inter in seat.interactions}
        assert len(per_kyoku) == 1
        rewards.append(per_kyoku.pop())
    assert len(set(rewards)) == 3
    assert all(any(r == pytest.approx(s) for s in steps) for r in rewards)


@mortal_weights
def test_mortal_seat_leaves_no_decisions_or_journal_rows(played_mortal) -> None:
    import json

    _, run_dir = played_mortal
    assert (run_dir / "decisions" / "seat3.jsonl").read_text(encoding="utf-8") == ""
    lines = (run_dir / "journal.jsonl").read_text(encoding="utf-8").splitlines()
    header = json.loads(lines[0])
    assert header["models"][3] == "mortal"
    rows = [json.loads(line) for line in lines[1:]]
    assert rows[-1] == {"end": True}
    assert rows[:-1] and all(row["seat"] != "seat3" for row in rows[:-1])
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    assert sorted(config["final"]["placements"].values()) == [1, 2, 3, 4]


@mortal_weights
def test_mortal_seat_replays_deterministically(played_mortal, tmp_path_factory) -> None:
    """The journal holds only the bridged seats' choices; a replay recomputes the
    mortal seat live, which reproduces the game only if its engine is deterministic."""
    import json

    _, run_dir = played_mortal
    replay_root = tmp_path_factory.mktemp("hanchan-mortal-replay")
    new_run = replay_root / "hanchan-00000"
    new_run.mkdir()
    (new_run / "journal.jsonl").write_text(
        (run_dir / "journal.jsonl").read_text(encoding="utf-8"), encoding="utf-8"
    )

    env = RiichiHanchanEnv.__new__(RiichiHanchanEnv)
    env.config = _mortal_config(replay_root)
    agents = FakeAgents(config=env.config)
    task = next(iter(RiichiHanchanTaskset(RiichiHanchanConfig()).load()))
    asyncio.run(env.run(task, agents))

    assert all(seat.interactions == [] for seat in agents.seats)
    original = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    replayed = json.loads((new_run / "config.json").read_text(encoding="utf-8"))
    assert replayed["final"] == original["final"]


def test_seat_rotation_moves_the_table_without_moving_attribution(
    tmp_path_factory,
) -> None:
    import json

    root = tmp_path_factory.mktemp("hanchan-rotation")
    env = RiichiHanchanEnv.__new__(RiichiHanchanEnv)
    env.config = RiichiHanchanEnvConfig(log_dir=str(root), seat_rotation=True)
    agents = FakeAgents()
    loaded = RiichiHanchanTaskset(RiichiHanchanConfig()).load()
    tasks = [t for t, _ in zip(loaded, range(2), strict=False)]
    asyncio.run(env.run(tasks[1], agents))

    run_dir = root / "hanchan-00001"
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    assert config["rotation"] == 1
    assert config["table"] == ["seat3", "seat0", "seat1", "seat2"]
    assert sorted(config["final"]["placements"]) == list(SEATS)

    for index, seat in enumerate(agents.seats):
        info = seat.interactions[-1].trace.info["hanchan"]
        assert info["seat"] == SEATS[index]
        assert info["table_position"] == (index + 1) % len(SEATS)
        assert info["seat_order"] == config["table"]
        assert info["placement"] == config["final"]["placements"][SEATS[index]]
        lines = (run_dir / "decisions" / f"{SEATS[index]}.jsonl").read_text(
            encoding="utf-8"
        )
        assert len(lines.splitlines()) == len(seat.prompts)


def test_taskset_is_an_infinite_seeded_generator() -> None:
    taskset = RiichiHanchanTaskset(RiichiHanchanConfig())
    assert taskset.INFINITE
    tasks = [t for t, _ in zip(taskset.load(), range(3), strict=False)]
    seeds = [t.data.info["seed"] for t in tasks]
    assert len(set(seeds)) == 3
    assert all(t.data.prompt is None for t in tasks)
    assert all(isinstance(t, vf.Task) for t in tasks)
