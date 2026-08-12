"""riichi-hanchan-v1: four models play a full hanchan against each other.

One episode is one Tenhou-rules hanchan refereed by the vendored `libriichi` arena. Each
seat is a live interaction, so its trace is a real rollout, and the terminal reward is its
placement - the thing riichi is actually played for, and a genuinely zero-sum signal since
the four placements are a permutation of 1-4 whatever the models do.

The arena is a synchronous Rust loop that calls a seat at its decision points, while an
Env is async and drives its agents turn by turn. `jongbench.bridge` reconciles the two:
the arena runs in a worker thread, and a seat's decision is marshalled back onto the event
loop as one `interaction.turn`. That keeps the whole normal engine path - prompt building,
per-kyoku delta turns, the furo toggle, invalid-reply retries - rather than reimplementing
it against the verifiers API.

This is the expensive way to measure a model: a hanchan is ~1,000 model calls and every
seat sees a board the other three steered. For a cheap, separable, byte-identical
comparison use riichi-decision-v1, which grades single positions against Mortal.
"""

import asyncio
import json
import threading
from collections import Counter
from collections.abc import Iterator
from datetime import datetime, timezone
from itertools import count
from pathlib import Path

from jongbench import arena, bridge, engines, prompts, providers
from jongbench.artifacts import decision_filename
from riichi_hanchan_v1.tools import SeatState, SeatToolset

import verifiers.v1 as vf

SEATS = ("seat0", "seat1", "seat2", "seat3")


def _segment_reasoning(segment) -> str:
    return "\n".join(
        message.reasoning_content
        for message in segment.messages
        if getattr(message, "reasoning_content", None)
    )


def _tools_called(segment) -> Counter:
    return Counter(
        call.name
        for message in segment.messages
        for call in (getattr(message, "tool_calls", None) or ())
    )


def _usage_of(calls) -> dict[str, int | float]:
    """verifiers' accounting in the shape jongbench's decision records use: its
    `prompt_tokens` has cache reads split out, jongbench's `input_tokens` has them in."""
    total: dict[str, int | float] = dict.fromkeys(
        ("input_tokens", "output_tokens", "cached_input_tokens", "reasoning_tokens"), 0
    )
    cost = None
    for call in calls:
        if call.usage is None:
            continue
        cached = call.usage.cached_input_tokens or 0
        total["input_tokens"] += call.usage.prompt_tokens + cached
        total["output_tokens"] += call.usage.completion_tokens
        total["cached_input_tokens"] += cached
        total["reasoning_tokens"] += call.usage.reasoning_tokens or 0
        call_cost = getattr(call.usage, "cost", None)
        if call_cost is not None:
            cost = (cost or 0.0) + call_cost
    # Absent means the provider does not meter, which is not the same as free.
    if cost is not None:
        total["cost"] = cost
    return total


class RiichiHanchanData(vf.TaskData):
    info: dict
    """The hanchan's arena seed; the deal is reproducible from it."""


class RiichiHanchanTask(vf.Task[RiichiHanchanData]):
    pass


class SeatToolsTask(vf.Task):
    """A seat's per-kyoku task in tools mode: same rollout, plus the board-query
    toolset, colocated so it shares the host and reads this rollout's state channel."""

    @classmethod
    def toolsets(cls, config) -> list[vf.Toolset]:
        return [SeatToolset(vf.ToolsetConfig(colocated=True))]


class RiichiHanchanConfig(vf.TasksetConfig):
    pass


class RiichiHanchanEnvConfig(vf.EnvConfig):
    seat0: vf.AgentConfig = vf.AgentConfig(harness={"id": "null"})
    seat1: vf.AgentConfig = vf.AgentConfig(harness={"id": "null"})
    seat2: vf.AgentConfig = vf.AgentConfig(harness={"id": "null"})
    seat3: vf.AgentConfig = vf.AgentConfig(harness={"id": "null"})
    state_hints: bool = True
    """Give seats rule-derived shanten, waits and furiten, as the CLI does by default."""
    auto_pass_reactions: bool = False
    """Pass pure chi/pon/open-kan reactions without a model call (~15% of decisions).
    A cost mode: the seats never call on others' discards, which changes what is
    measured. Wins and own-turn decisions always reach the model."""
    seat_rotation: bool = False
    """Rotate the seating by the episode index: agent i sits at table position
    (i + idx) % 4. Over four episodes each agent plays every wind, which cancels the
    dealer advantage a fixed seating bakes into a short batch. Rewards are keyed by
    engine name, so a seat's placement follows it around the table."""
    tools: bool = False
    """Give each seat board-query tools instead of inline state hints: board(),
    discards(), waits(), simulate(), plus a note()/notes() scratchpad that survives
    kyoku resets. Information-seeking and memory management become part of what is
    measured — a different benchmark, not a cheaper rendering of this one."""
    max_tool_calls: int = 32
    """Tool calls one decision may spend before every tool answers "budget spent" and
    the seat must commit (0 lifts the cap). Each tool turn resends the whole decision
    conversation, so an unbounded loop costs quadratically: one observed seat burned
    2.5M input tokens on a single discard before the run had to be killed. Sized to
    clear an exhaustive sweep - simulate() on all 14 discards plus board, waits and
    three discard rows is ~20 - so the cap only fires on a loop: at 4 it cut short 25%
    of a seat's decisions, at 12 about 2%."""
    log_dir: str | None = None
    """Persist each episode as a jongbench run dir - mjai log, per-seat decision logs,
    config.json - so `jongbench review` and `jongbench reasoning` grade the rollout
    afterwards with the Mortal checkpoint."""
    weights: str = "weights/mortal.pth"
    """Checkpoint for any seat whose model is `mortal` - the Mortal NN playing as a
    control seat. Such a seat runs on CPU, produces no traces, and earns no reward;
    the LLM seats' placements are measured against it."""


class _Journal:
    """Crash journal for one episode: a header line naming what the recording is of,
    then every decision as it happens. The arena is deterministic given seed + actions,
    so a rerun replays the journal's complete hands without a single model call and
    goes live from the first unrecorded hand."""

    def __init__(self, path: Path, header: dict) -> None:
        self._lock = threading.Lock()
        self._handle = path.open("w", encoding="utf-8")
        self._write(header)

    def append(self, seat: str, record: dict) -> None:
        self._write({"seat": seat, **record})

    def finish(self) -> None:
        self._write({"end": True})

    def close(self) -> None:
        with self._lock:
            self._handle.close()

    def _write(self, row: dict) -> None:
        line = json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
        with self._lock:
            self._handle.write(line)
            self._handle.flush()


def _journal_header(
    seed: int, config: RiichiHanchanEnvConfig, models: list[str], rotation: int
) -> dict:
    return {
        "journal": 1,
        "seed": seed,
        "models": list(models),
        "rotation": rotation,
        "state_hints": bool(config.state_hints),
        "auto_pass_reactions": bool(config.auto_pass_reactions),
        "tools": bool(config.tools),
    }


def _read_journal(path: Path, header: dict) -> list[dict]:
    """The journal's replayable records: nothing on a missing file or a header that
    names a different game, and - unless the episode finished - nothing from the last
    recorded hand, which may have been cut mid-write and is re-run live instead."""
    if not path.exists():
        return []
    finished = False
    records: list[dict] = []
    with path.open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            try:
                row = json.loads(line)
            except ValueError:
                break
            if index == 0:
                if row != header:
                    return []
                continue
            if row.get("end") is True:
                finished = True
                break
            if not isinstance(row, dict) or not {
                "seat",
                "menu",
                "choice",
                "kyoku",
                "honba",
            } <= row.keys():
                break
            records.append(row)
    if not records:
        return []
    if not finished:
        last = max((row["kyoku"], row["honba"]) for row in records)
        records = [row for row in records if (row["kyoku"], row["honba"]) < last]
    return records


class RiichiHanchanEnv(vf.Env[RiichiHanchanEnvConfig]):
    async def run(self, task, agents) -> None:
        loop = asyncio.get_running_loop()
        seed = int(task.data.info["seed"])
        episode_dir: Path | None = None
        if self.config.log_dir:
            episode_dir = Path(self.config.log_dir) / f"hanchan-{task.data.idx:05d}"
            (episode_dir / "logs").mkdir(parents=True, exist_ok=True)
            (episode_dir / "decisions").mkdir(parents=True, exist_ok=True)
            (episode_dir / "review").mkdir(parents=True, exist_ok=True)

        # One interaction per seat PER KYOKU: the engine opens each kyoku with a full
        # board render, so earlier kyoku add nothing but cost and context pressure —
        # measured on a real hanchan, dragging them along triples the input tokens.
        # Rewards are only known at the end, so kyoku traces are collected as they
        # close and placement is recorded on all of them afterwards.
        seat_agents = [agents.seat0, agents.seat1, agents.seat2, agents.seat3]
        # The run's own model fills in an unpinned seat, so the resolved config is the
        # only place the real spec of every seat is knowable.
        models = [str(agent.config.model) for agent in seat_agents]
        rotation = task.data.idx % len(SEATS) if self.config.seat_rotation else 0
        open_cms: list = [None, None, None, None]
        current: list = [None, None, None, None]
        seat_traces: list[list] = [[] for _ in SEATS]
        seat_calls = [0, 0, 0, 0]
        tools = bool(self.config.tools)
        seat_notes: list[list[str]] = [[] for _ in SEATS]
        seat_tools: list[Counter] = [Counter() for _ in SEATS]
        tool_turns = [0, 0, 0, 0]
        tool_budget = self.config.max_tool_calls or None
        budget_spent = [0, 0, 0, 0]
        system_prompt = prompts.SYSTEM
        if tools:
            system_prompt += prompts.TOOLS_APPENDIX
            if tool_budget:
                system_prompt += prompts.BUDGET_LINE.format(budget=tool_budget)

        def _seat_task(index: int) -> vf.Task:
            task_cls = SeatToolsTask if tools else vf.Task
            return task_cls(
                vf.TaskData(
                    idx=task.data.idx,
                    name=f"{SEATS[index]}-k{hands_replayed + len(seat_traces[index])}",
                    prompt=None,  # each kyoku converses through its interaction
                    system_prompt=system_prompt,
                )
            )

        async def _close_current(index: int) -> None:
            if open_cms[index] is None:
                return
            seat_traces[index].append(current[index].trace)
            cm, open_cms[index], current[index] = open_cms[index], None, None
            await cm.__aexit__(None, None, None)

        async def _seat_turn(index: int, prompt: str, fresh: bool):
            if fresh or current[index] is None:
                await _close_current(index)
                cm = seat_agents[index].interaction(_seat_task(index))
                current[index] = await cm.__aenter__()
                open_cms[index] = cm
                seat_calls[index] = 0
            if tools:
                snapshot = seats[index].decision_snapshot(0) or {}
                current[index].trace.state = SeatState(
                    **snapshot, notes=list(seat_notes[index]), budget=tool_budget
                )
            segment = await current[index].turn(prompt)
            if tools:
                state = current[index].trace.state
                notes = getattr(state, "notes", None)
                if notes is not None:
                    seat_notes[index] = list(notes)
                budget_spent[index] += getattr(state, "budget", None) == 0
                called = _tools_called(segment)
                seat_tools[index] += called
                tool_turns[index] += bool(called)
            # The trace's calls accumulate over the whole kyoku, so a decision costs
            # only the ones this turn added.
            trace_calls = current[index].trace.calls
            calls, seat_calls[index] = trace_calls[seat_calls[index] :], len(trace_calls)
            return segment, calls

        def ask_from(index: int):
            def ask(prompt: str, fresh: bool) -> providers.Completion:
                # Called on the arena's worker thread; hand the turn back to the loop
                # and block this seat until the model answers, as the arena expects.
                future = asyncio.run_coroutine_threadsafe(
                    _seat_turn(index, prompt, fresh), loop
                )
                segment, calls = future.result()
                if segment.terminated:
                    raise engines.GameAborted(f"{SEATS[index]} ended its rollout")
                return providers.Completion(
                    text=segment.last_reply,
                    reasoning=_segment_reasoning(segment),
                    usage=_usage_of(calls),
                )

            return ask

        logs: list[list[dict]] = [[] for _ in SEATS]
        journal: _Journal | None = None
        replayed: dict[str, list[dict]] = {name: [] for name in SEATS}
        hands_replayed = 0
        if episode_dir is not None:
            header = _journal_header(seed, self.config, models, rotation)
            past = _read_journal(episode_dir / "journal.jsonl", header)
            hands_replayed = len({(row["kyoku"], row["honba"]) for row in past})
            for row in past:
                replayed[row.pop("seat")].append(row)
            journal = _Journal(episode_dir / "journal.jsonl", header)

        def _sink_for(index: int):
            def sink(record: dict) -> None:
                logs[index].append(record)
                if journal is not None:
                    journal.append(SEATS[index], record)

            return sink

        # A `mortal` seat is the Mortal NN itself playing as a control: no bridge, no
        # interactions, no traces. Its engine is deterministic (argmax over Q), so on
        # resume it recomputes the same choices live and needs no journal of its own.
        seats = [
            engines.make_engine(name, "mortal", weights=self.config.weights)
            if models[index] == "mortal"
            else bridge.make_bridged_engine(
                name,
                ask_from(index),
                decision_log=_sink_for(index),
                state_hints=self.config.state_hints and not tools,
                auto_pass_reactions=self.config.auto_pass_reactions,
                snapshot_decisions=tools,
                replay=replayed[name],
            )
            for index, name in enumerate(SEATS)
        ]

        # Table position is list position, and it decides the winds. Rotating the list
        # while every engine keeps its agent name means the seat that answers, the
        # decision log it writes and the placement it is paid all still line up.
        table = [seats[(position - rotation) % len(SEATS)] for position in range(len(SEATS))]

        try:
            # One hanchan per episode: a single engine driving several games at once
            # would interleave their turns into one conversation.
            summaries = await asyncio.to_thread(
                arena.run_games,
                table,
                1,
                (seed, 1),
                str(episode_dir / "logs") if episode_dir else None,
            )
        except BaseException:
            if journal is not None:
                journal.close()
            raise
        finally:
            for index in range(len(SEATS)):
                await _close_current(index)

        if journal is not None:
            journal.finish()
            journal.close()
        summary = summaries[0]
        if episode_dir is not None:
            _write_episode_artifacts(
                episode_dir, seed, summary, logs, self.config, models, rotation
            )
        scores = dict(zip(summary.names, summary.scores, strict=True))
        for index, name in enumerate(SEATS):
            placement = summary.placements[name]
            for kyoku, trace in enumerate(seat_traces[index]):
                # 1st -> 1.0, 4th -> 0.0. Every kyoku trace carries the seat's final
                # placement, so a seat's mean reward IS its placement reward.
                trace.record_reward("placement", (4 - placement) / 3)
                trace.record_metric("final_score", float(scores[name]))
                trace.record_metric("decisions", float(len(logs[index])))
                trace.record_metric(
                    "fallbacks", float(seats[index].totals["fallbacks"])
                )
                trace.record_metric(
                    "calls_declined", float(seats[index].totals["calls_declined"])
                )
                if tools:
                    trace.record_metric("notes_saved", float(len(seat_notes[index])))
                    trace.record_metric(
                        "tool_calls", float(sum(seat_tools[index].values()))
                    )
                    trace.record_metric("tool_turns", float(tool_turns[index]))
                    trace.record_metric("budget_spent", float(budget_spent[index]))
                trace.info["hanchan"] = {
                    "seat": name,
                    "model": models[index],
                    "placement": placement,
                    "score": scores[name],
                    "seed": seed,
                    "seat_order": list(summary.names),
                    "table_position": (index + rotation) % len(SEATS),
                    "kyoku": hands_replayed + kyoku,
                    "kyoku_count": hands_replayed + len(seat_traces[index]),
                }
                if tools:
                    trace.info["hanchan"]["tools"] = dict(seat_tools[index])


def _write_episode_artifacts(
    episode_dir: Path,
    seed: int,
    summary: arena.GameSummary,
    logs: list[list[dict]],
    config: RiichiHanchanEnvConfig,
    models: list[str],
    rotation: int,
) -> None:
    for name, decisions in zip(SEATS, logs, strict=True):
        path = episode_dir / "decisions" / decision_filename(name)
        with path.open("w", encoding="utf-8") as handle:
            for record in decisions:
                handle.write(
                    json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
                )
    (episode_dir / "config.json").write_text(
        json.dumps(
            {
                "label": episode_dir.name,
                "created": datetime.now(timezone.utc).isoformat(),
                # `names` are the engine names the mjai log and the reviews are keyed
                # by; `models` is the spec each one played, paired by position.
                "models": list(models),
                "names": list(SEATS),
                "games": 1,
                "seed_start": [seed, 1],
                "rotation": rotation,
                "table": [SEATS[(position - rotation) % len(SEATS)] for position in range(len(SEATS))],
                # the effective value: tools mode replaces inline hints, so a run dir
                # would otherwise claim hints the seats never saw.
                "state_hints": bool(config.state_hints) and not config.tools,
                "tools": bool(config.tools),
                "final": {
                    "names": list(summary.names),
                    "scores": list(summary.scores),
                    "placements": dict(summary.placements),
                },
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


class RiichiHanchanTaskset(vf.Taskset[RiichiHanchanTask, RiichiHanchanConfig]):
    INFINITE = True

    def load(self) -> Iterator[RiichiHanchanTask]:
        for i in count():
            yield RiichiHanchanTask(
                RiichiHanchanData(
                    idx=i,
                    name=f"hanchan#{i}",
                    prompt=None,
                    info={"seed": 20260000 + i},
                ),
                self.config.task,
            )
