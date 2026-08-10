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
from collections.abc import Iterator
from datetime import datetime, timezone
from itertools import count
from pathlib import Path

from jongbench import arena, bridge, engines, prompts
from jongbench.artifacts import decision_filename
from riichi_hanchan_v1.tools import SeatState, SeatToolset

import verifiers.v1 as vf

SEATS = ("seat0", "seat1", "seat2", "seat3")


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
    tools: bool = False
    """Give each seat board-query tools instead of inline state hints: board(),
    discards(), waits(), simulate(), plus a note()/notes() scratchpad that survives
    kyoku resets. Information-seeking and memory management become part of what is
    measured — a different benchmark, not a cheaper rendering of this one."""
    log_dir: str | None = None
    """Persist each episode as a jongbench run dir - mjai log, per-seat decision logs,
    config.json - so `jongbench review` and `jongbench reasoning` grade the rollout
    afterwards with the Mortal checkpoint."""


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
        open_cms: list = [None, None, None, None]
        current: list = [None, None, None, None]
        seat_traces: list[list] = [[] for _ in SEATS]
        tools = bool(self.config.tools)
        seat_notes: list[list[str]] = [[] for _ in SEATS]
        system_prompt = prompts.SYSTEM + (prompts.TOOLS_APPENDIX if tools else "")

        def _seat_task(index: int) -> vf.Task:
            task_cls = SeatToolsTask if tools else vf.Task
            return task_cls(
                vf.TaskData(
                    idx=task.data.idx,
                    name=f"{SEATS[index]}-k{len(seat_traces[index])}",
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
            if tools:
                snapshot = seats[index].decision_snapshot(0) or {}
                current[index].trace.state = SeatState(
                    **snapshot, notes=list(seat_notes[index])
                )
            segment = await current[index].turn(prompt)
            if tools:
                notes = getattr(current[index].trace.state, "notes", None)
                if notes is not None:
                    seat_notes[index] = list(notes)
            return segment

        def ask_from(index: int):
            def ask(prompt: str, fresh: bool) -> str:
                # Called on the arena's worker thread; hand the turn back to the loop
                # and block this seat until the model answers, as the arena expects.
                future = asyncio.run_coroutine_threadsafe(
                    _seat_turn(index, prompt, fresh), loop
                )
                segment = future.result()
                if segment.terminated:
                    raise engines.GameAborted(f"{SEATS[index]} ended its rollout")
                return segment.last_reply

            return ask

        logs: list[list[dict]] = [[] for _ in SEATS]
        seats = [
            bridge.make_bridged_engine(
                name,
                ask_from(index),
                decision_log=logs[index],
                state_hints=self.config.state_hints and not tools,
                auto_pass_reactions=self.config.auto_pass_reactions,
                snapshot_decisions=tools,
            )
            for index, name in enumerate(SEATS)
        ]

        try:
            # One hanchan per episode: a single engine driving several games at once
            # would interleave their turns into one conversation.
            summaries = await asyncio.to_thread(
                arena.run_games,
                seats,
                1,
                (seed, 1),
                str(episode_dir / "logs") if episode_dir else None,
            )
        finally:
            for index in range(len(SEATS)):
                await _close_current(index)

        summary = summaries[0]
        if episode_dir is not None:
            _write_episode_artifacts(episode_dir, seed, summary, logs, self.config)
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
                trace.info["hanchan"] = {
                    "seat": name,
                    "placement": placement,
                    "score": scores[name],
                    "seed": seed,
                    "seat_order": list(summary.names),
                    "kyoku": kyoku,
                    "kyoku_count": len(seat_traces[index]),
                }


def _write_episode_artifacts(
    episode_dir: Path,
    seed: int,
    summary: arena.GameSummary,
    logs: list[list[dict]],
    config: RiichiHanchanEnvConfig,
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
                "models": list(SEATS),
                "names": list(SEATS),
                "games": 1,
                "seed_start": [seed, 1],
                "state_hints": bool(config.state_hints),
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
