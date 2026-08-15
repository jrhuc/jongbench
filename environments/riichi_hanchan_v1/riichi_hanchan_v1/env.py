"""riichi-hanchan-v1: evaluate one model through a full hanchan.

One episode is one Tenhou-rules hanchan refereed by the vendored `libriichi` arena. Each
seat is a live interaction and keeps bounded per-kyoku traces with its terminal placement
return. Standard evaluation explicitly selects one agent, however, and reports exactly
one placement reward for it; otherwise four-seat constant-sum aggregation is invariant 0.5.

The arena is a synchronous Rust loop that calls a seat at its decision points, while an
Env is async and drives its agents turn by turn. `jongbench.bridge` reconciles the two:
the arena runs in a worker thread, and a seat's decision is marshalled back onto the event
loop as one `interaction.turn`. That keeps the whole normal engine path - prompt building,
per-kyoku delta turns, the furo toggle, invalid-reply retries - rather than reimplementing
it against the verifiers API.

A standard episode spends model calls only on the evaluated role; an all-LLM tournament
can approach 1,000 calls. Every seat sees a board the other three steered. For a cheap,
separable, byte-identical comparison use riichi-decision-v1.
"""

import asyncio
import hashlib
import json
import os
import threading
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from itertools import count
from pathlib import Path
from typing import Literal

import verifiers.v1 as vf

from jongbench import (
    arena,
    bridge,
    engines,
    prompts,
    providers,
)
from jongbench import (
    weights as weights_module,
)
from jongbench.artifacts import decision_filename
from riichi_hanchan_v1.tools import SeatState, SeatToolset

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

    @vf.stop
    async def tool_budget_exhausted(self, trace: vf.Trace) -> bool:
        return getattr(trace.state, "refused_tool_calls", 0) >= 2


class RiichiHanchanConfig(vf.TasksetConfig):
    pass


class RiichiHanchanEnvConfig(vf.EnvConfig):
    # The standard run has one policy role and fixed-strength controls. Leaving all
    # four unpinned evaluates the model against itself, whose expected placement is
    # necessarily 2.5 whatever its strength.
    seat0: vf.AgentConfig = vf.AgentConfig(harness={"id": "null"})
    seat1: vf.AgentConfig = vf.AgentConfig(harness={"id": "null"}, model="mortal")
    seat2: vf.AgentConfig = vf.AgentConfig(harness={"id": "null"}, model="mortal")
    seat3: vf.AgentConfig = vf.AgentConfig(harness={"id": "null"}, model="mortal")
    state_hints: bool = True
    """Give seats rule-derived shanten, waits and furiten, as the CLI does by default."""
    auto_pass_reactions: bool = False
    """Pass pure chi/pon/open-kan reactions without a model call (~15% of decisions).
    A cost mode: the seats never call on others' discards, which changes what is
    measured. Wins and own-turn decisions always reach the model."""
    seat_rotation: bool = True
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
    weights: str = "auto"
    """Checkpoint for any seat whose model is `mortal` - the Mortal NN playing as a
    control seat. Such a seat runs on CPU, produces no traces, and earns no reward;
    the LLM seats' placements are measured against it."""
    control_use_policy: bool = False
    """Use the Mortal checkpoint's stochastic policy head for control seats. The
    default is deterministic argmax-Q play; Phoenix policy play must be explicit."""
    evaluated_agent: Literal["seat0", "seat1", "seat2", "seat3"] | None = "seat0"
    """The one role whose placement is the standard Verifiers/Hub eval score.

    Other roles are opponents, not copies of the policy for score aggregation, even
    when their model is unpinned and therefore resolves to the run model. Set this to
    ``None`` for symmetric self-play/tournament runs: every unpinned role is then a
    policy role and pinned roles remain fixed. Tournament results stay available per
    role, but their deliberately constant-sum pooled headline is not a single-policy eval.
    """


class _Journal:
    """Crash journal for one episode: a header line naming what the recording is of,
    then every decision as it happens. The arena is deterministic given seed + actions,
    so a rerun replays the journal's complete hands without a single model call and
    goes live from the first unrecorded hand."""

    def __init__(
        self, path: Path, header: dict, records: list[dict] | None = None
    ) -> None:
        self._lock = threading.Lock()
        temporary = path.with_name(f".{path.name}.tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            for row in [header, *(records or [])]:
                handle.write(
                    json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
                )
            if records:
                handle.write('{"checkpoint":true}\n')
        temporary.replace(path)
        self._handle = path.open("a", encoding="utf-8")

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


def _resolve_control_checkpoint(
    config: RiichiHanchanEnvConfig, models: list[str]
) -> tuple[object, dict[str, object] | None]:
    """Resolve a Mortal control once, retaining compatibility with core v0.1.0."""
    if "mortal" not in models:
        return str(config.weights), None

    resolver = getattr(weights_module, "resolve_mortal_checkpoint", None)
    if resolver is not None:
        checkpoint = resolver(
            config.weights, use_policy=bool(config.control_use_policy)
        )
        return checkpoint, checkpoint.as_dict()

    # The first published environment core predates ResolvedCheckpoint. Keep its
    # immutable auto digest during the transition; new cores always take the branch
    # above and record path, source, digest, and policy from the canonical resolver.
    path = weights_module.resolve_mortal_weights(config.weights)
    if config.weights == "auto":
        digest = weights_module.auto_weights_sha256()
        source = os.environ.get("JONGBENCH_WEIGHTS_URL") or getattr(
            weights_module, "MORTAL_WEIGHTS_URL", "auto"
        )
    else:
        with path.open("rb") as handle:
            digest = hashlib.file_digest(handle, "sha256").hexdigest()
        source = str(config.weights)
    identity: dict[str, object] = {
        "path": str(path),
        "sha256": digest,
        "source": source,
        "use_policy": bool(config.control_use_policy),
    }
    return str(path), identity


def _make_control_engine(
    name: str,
    checkpoint: object,
    *,
    use_policy: bool,
):
    """Build through the canonical core API, with a v0.1.0 compatibility path."""
    if hasattr(weights_module, "ResolvedCheckpoint"):
        return engines.make_engine(
            name,
            "mortal",
            weights=checkpoint,
            use_policy=use_policy,
        )

    # Core v0.1.0 eagerly evaluated its ambient policy default even when an explicit
    # value was supplied to make_engine. Bypass that retired factory implementation
    # so the packaged environment's explicit config remains authoritative.
    from jongbench import evaluate, positions

    reviewer = evaluate.load_engine(str(checkpoint), use_policy=use_policy)
    if use_policy and not reviewer.use_policy:
        raise ValueError("configured checkpoint has no reviewer policy head")
    return positions.MortalArenaEngine(name, reviewer)


def _journal_header(
    seed: int,
    config: RiichiHanchanEnvConfig,
    models: list[str],
    rotation: int,
    checkpoint: dict[str, object] | None = None,
) -> dict:
    return {
        "journal": 2,
        "seed": seed,
        "models": list(models),
        "rotation": rotation,
        "evaluated_agent": config.evaluated_agent,
        "state_hints": bool(config.state_hints),
        "auto_pass_reactions": bool(config.auto_pass_reactions),
        "tools": bool(config.tools),
        "max_tool_calls": int(config.max_tool_calls),
        "control_use_policy": bool(config.control_use_policy),
        "reviewer_checkpoint": checkpoint,
    }


def _read_journal(path: Path, header: dict) -> list[dict]:
    """Return replayable records while preserving the last known-complete prefix."""
    if not path.exists():
        return []
    finished = False
    committed = 0
    records: list[dict] = []
    with path.open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            try:
                row = json.loads(line)
            except ValueError:
                break
            if index == 0:
                if row != header:
                    raise ValueError(
                        f"{path} belongs to a different hanchan configuration"
                    )
                continue
            if not isinstance(row, dict):
                break
            if row.get("end") is True:
                finished = True
                break
            if row.get("checkpoint") is True:
                committed = len(records)
                continue
            if not {"seat", "menu", "choice", "kyoku", "honba"} <= row.keys():
                break
            records.append(row)
    if finished or not records:
        return records
    tail = records[committed:]
    if not tail:
        return records
    last = max((row["kyoku"], row["honba"]) for row in tail)
    return records[:committed] + [
        row for row in tail if (row["kyoku"], row["honba"]) < last
    ]


@dataclass
class SeatRuntime:
    """All mutable state for one named seat during one hanchan.

    A seat owns a sequence of bounded kyoku interactions, its bridge accounting, its
    tools scratch state, and its persistence stream.  Keeping those values together
    makes it impossible to update one seat through an index into the wrong parallel
    list when seating is rotated or a journal resumes mid-hand.
    """

    index: int
    name: str
    agent: object
    model: str
    task_idx: int
    system_prompt: str
    tools_enabled: bool
    tool_budget: int | None
    control_use_policy: bool

    interaction_cm: object | None = None
    interaction: object | None = None
    traces: list = field(default_factory=list)
    trace_kyokus: list[int] = field(default_factory=list)
    current_kyoku: int = 0
    next_kyoku: int = 0
    current_call_count: int = 0

    notes: list[str] = field(default_factory=list)
    tool_counts: Counter = field(default_factory=Counter)
    tool_turns: int = 0
    budget_spent: int = 0
    decision_snapshot: dict | None = None
    active_kyoku_id: object | None = None
    remaining_budget: int | None = field(init=False)
    remaining_refusals: int = 0
    budget_exhausted: bool = False

    decisions: list[dict] = field(default_factory=list)
    replay: list[dict] = field(default_factory=list)
    replay_remaining: int = 0
    journal: _Journal | None = None
    engine: object | None = None

    def __post_init__(self) -> None:
        self.remaining_budget = self.tool_budget

    def start_after_replay(self, hands_replayed: int, journal: _Journal | None) -> None:
        self.current_kyoku = hands_replayed
        self.next_kyoku = hands_replayed
        self.journal = journal

    def queue_replay(self, record: dict) -> None:
        self.replay.append(
            {key: value for key, value in record.items() if key != "seat"}
        )
        self.replay_remaining += 1

    def sink(self, record: dict) -> None:
        """Keep the complete decision log while journaling only newly live play."""
        self.decisions.append(record)
        if self.replay_remaining:
            self.replay_remaining -= 1
        elif self.journal is not None:
            self.journal.append(self.name, record)

    def _task(self) -> vf.Task:
        task_cls = SeatToolsTask if self.tools_enabled else vf.Task
        return task_cls(
            vf.TaskData(
                idx=self.task_idx,
                name=f"{self.name}-k{self.current_kyoku}",
                prompt=None,  # each kyoku converses through its interaction
                system_prompt=self.system_prompt,
            )
        )

    async def close_interaction(self) -> None:
        if self.interaction_cm is None:
            return
        self.traces.append(self.interaction.trace)
        self.trace_kyokus.append(self.current_kyoku)
        cm = self.interaction_cm
        self.interaction_cm = None
        self.interaction = None
        await cm.__aexit__(None, None, None)

    async def turn(self, prompt: str, fresh: bool):
        snapshot = self.engine.decision_snapshot(0) or {} if self.tools_enabled else {}
        if fresh:
            await self.close_interaction()
            kyoku_id = snapshot.get("kyoku_id")
            if not self.tools_enabled or kyoku_id != self.active_kyoku_id:
                self.current_kyoku = self.next_kyoku
                self.next_kyoku += 1
            self.active_kyoku_id = kyoku_id

        if self.interaction is None:
            cm = self.agent.interaction(self._task())
            self.interaction = await cm.__aenter__()
            self.interaction_cm = cm
            self.current_call_count = 0

        if self.tools_enabled:
            if snapshot is not self.decision_snapshot:
                self.decision_snapshot = snapshot
                self.remaining_budget = self.tool_budget
                self.remaining_refusals = 0
                self.budget_exhausted = False
            self.interaction.trace.state = SeatState(
                **snapshot,
                notes=list(self.notes),
                budget=self.remaining_budget,
                refused_tool_calls=self.remaining_refusals,
            )

        segment = await self.interaction.turn(prompt)
        if self.tools_enabled:
            state = self.interaction.trace.state
            notes = getattr(state, "notes", None)
            if notes is not None:
                self.notes = list(notes)
            self.remaining_budget = getattr(state, "budget", None)
            self.remaining_refusals = getattr(state, "refused_tool_calls", 0)
            if self.remaining_budget == 0 and not self.budget_exhausted:
                self.budget_spent += 1
                self.budget_exhausted = True
            called = _tools_called(segment)
            self.tool_counts += called
            self.tool_turns += bool(called)

        # The trace's calls accumulate over the whole kyoku, so a decision costs
        # only the ones this turn added.
        trace = self.interaction.trace
        trace_calls = trace.calls
        calls = trace_calls[self.current_call_count :]
        self.current_call_count = len(trace_calls)
        forced_choice = (
            getattr(trace.state, "fallback_choice", 0)
            if getattr(trace, "stop_condition", None) == "tool_budget_exhausted"
            else None
        )
        if forced_choice is not None:
            await self.close_interaction()
        return segment, calls, forced_choice

    def ask(self, loop: asyncio.AbstractEventLoop):
        def ask(prompt: str, fresh: bool) -> providers.Completion:
            # Called on the arena's worker thread; hand the turn back to the loop
            # and block this seat until the model answers, as the arena expects.
            future = asyncio.run_coroutine_threadsafe(self.turn(prompt, fresh), loop)
            segment, calls, forced_choice = future.result()
            if segment.terminated and forced_choice is None:
                raise engines.GameAborted(f"{self.name} ended its rollout")
            text = (
                json.dumps({"choice": forced_choice})
                if forced_choice is not None
                else segment.last_reply
            )
            return providers.Completion(
                text=text,
                reasoning=_segment_reasoning(segment),
                usage=_usage_of(calls),
                fallback_reason=(
                    "tool_budget_exhausted" if forced_choice is not None else None
                ),
                reset_conversation=forced_choice is not None,
            )

        return ask

    def record_outcome(
        self,
        summary: arena.GameSummary,
        score: int,
        seed: int,
        rotation: int,
        hands_replayed: int,
        evaluated_agent: str | None,
        policy_role: bool,
    ) -> None:
        placement = summary.placements[self.name]
        placement_return = (4 - placement) / 3
        client = getattr(self.agent.config, "client", None)
        training_return = policy_role and getattr(client, "type", "eval") == "train"
        # Evaluation and training need different aggregation units. Training gets
        # every bounded kyoku trace with the hanchan return. Evaluation gets one
        # placement-bearing carrier, or the Hub would average a variable number of
        # kyoku rows (and symmetric four-seat play would always headline 0.5).
        # Keep the raw return on eval traces at weight zero so per-kyoku inspection
        # and tournament role breakdowns still see it without changing Trace.reward.
        carrier = (
            self.traces[-1]
            if self.traces and policy_role and not training_return
            else None
        )
        kyoku_count = hands_replayed + len(set(self.trace_kyokus))
        for kyoku, trace in zip(self.trace_kyokus, self.traces, strict=True):
            trace.record_reward(
                "placement_return",
                placement_return,
                weight=1.0 if training_return else 0.0,
            )
            trace.agent.trainable = training_return
            if trace is carrier:
                trace.record_reward("placement", placement_return)
                trace.agent.trainable = True
            trace.record_metric("final_score", float(score))
            trace.record_metric("decisions", float(len(self.decisions)))
            trace.record_metric("fallbacks", float(self.engine.totals["fallbacks"]))
            trace.record_metric(
                "calls_declined", float(self.engine.totals["calls_declined"])
            )
            if self.tools_enabled:
                trace.record_metric("notes_saved", float(len(self.notes)))
                trace.record_metric("tool_calls", float(sum(self.tool_counts.values())))
                trace.record_metric("tool_turns", float(self.tool_turns))
                trace.record_metric("budget_spent", float(self.budget_spent))
            trace.info["hanchan"] = {
                "seat": self.name,
                "model": self.model,
                "placement": placement,
                "score": score,
                "seed": seed,
                "seat_order": list(summary.names),
                "table_position": (self.index + rotation) % len(SEATS),
                "kyoku": kyoku,
                "kyoku_count": kyoku_count,
                "evaluated_agent": evaluated_agent,
                "policy_role": policy_role,
                "eval_carrier": trace is carrier,
            }
            if self.tools_enabled:
                trace.info["hanchan"]["tools"] = dict(self.tool_counts)


class RiichiHanchanEnv(vf.Env[RiichiHanchanEnvConfig]):
    async def setup(self, agents) -> None:
        """Declare which generated tokens belong to the policy under evaluation.

        ``AgentConfig.model`` pins an identity but does not make a role trainable:
        standing is an episode property in Verifiers 0.3.  With an explicit evaluated
        agent, every other seat is a fixed opponent.  Tournament mode keeps only
        unpinned seats trainable, because those are the roles filled by the run model.
        """
        evaluated = self.config.evaluated_agent
        for name in SEATS:
            configured = getattr(self.config, name)
            agent = getattr(agents, name)
            agent.trainable = (
                name == evaluated if evaluated is not None else configured.model is None
            )

        if evaluated is not None:
            model = str(getattr(agents, evaluated).config.model)
            if model == "mortal":
                raise ValueError(
                    f"evaluated_agent={evaluated!r} resolves to mortal, which produces "
                    "no trace; choose a live agent or use evaluated_agent=None for a "
                    "tournament"
                )

    async def run(self, task, agents) -> None:
        loop = asyncio.get_running_loop()
        seed = int(task.data.info["seed"])
        episode_dir: Path | None = None
        if self.config.log_dir:
            episode_dir = Path(self.config.log_dir) / f"hanchan-{task.data.idx:05d}"
            (episode_dir / "logs").mkdir(parents=True, exist_ok=True)
            (episode_dir / "decisions").mkdir(parents=True, exist_ok=True)
            (episode_dir / "review").mkdir(parents=True, exist_ok=True)

        tools = bool(self.config.tools)
        tool_budget = self.config.max_tool_calls or None
        system_prompt = prompts.SYSTEM
        if tools:
            system_prompt += prompts.TOOLS_APPENDIX
            if tool_budget:
                system_prompt += prompts.BUDGET_LINE.format(budget=tool_budget)

        # One interaction per seat PER KYOKU: the engine opens each kyoku with a full
        # board render, so earlier kyoku add nothing but cost and context pressure —
        # measured on a real hanchan, dragging them along triples the input tokens.
        # Returns are only known at the end, so each runtime retains its bounded traces.
        seats: list[SeatRuntime] = []
        for index, name in enumerate(SEATS):
            agent = getattr(agents, name)
            seats.append(
                SeatRuntime(
                    index=index,
                    name=name,
                    agent=agent,
                    model=str(agent.config.model),
                    task_idx=task.data.idx,
                    system_prompt=system_prompt,
                    tools_enabled=tools,
                    tool_budget=tool_budget,
                    control_use_policy=bool(self.config.control_use_policy),
                )
            )

        # The run's own model fills in an unpinned seat, so the resolved config is the
        # only place the real spec of every seat is knowable.
        models = [seat.model for seat in seats]
        control_weights, checkpoint_identity = _resolve_control_checkpoint(
            self.config, models
        )
        rotation = task.data.idx % len(SEATS) if self.config.seat_rotation else 0

        journal: _Journal | None = None
        hands_replayed = 0
        if episode_dir is not None:
            header = _journal_header(
                seed, self.config, models, rotation, checkpoint_identity
            )
            journal_path = episode_dir / "journal.jsonl"
            past = _read_journal(journal_path, header)
            hands_replayed = len({(row["kyoku"], row["honba"]) for row in past})
            journal = _Journal(journal_path, header, past)
            for row in past:
                seats[SEATS.index(str(row["seat"]))].queue_replay(row)
        for seat in seats:
            seat.start_after_replay(hands_replayed, journal)

        # A `mortal` seat is the Mortal NN itself playing as a control: no bridge, no
        # interactions, no traces. Its engine deterministically takes the argmax from
        # Q by default, or from the explicitly selected policy head, so resume can
        # recompute its choices live without journaling them.
        for seat in seats:
            if seat.model == "mortal":
                seat.engine = _make_control_engine(
                    seat.name,
                    control_weights,
                    use_policy=seat.control_use_policy,
                )
            else:
                seat.engine = bridge.make_bridged_engine(
                    seat.name,
                    seat.ask(loop),
                    decision_log=seat.sink,
                    state_hints=self.config.state_hints and not tools,
                    auto_pass_reactions=self.config.auto_pass_reactions,
                    snapshot_decisions=tools,
                    replay=seat.replay,
                )

        # Table position decides the winds. Rotating runtimes while every engine keeps
        # its agent name preserves decision logging and placement attribution.
        table = [
            seats[(position - rotation) % len(SEATS)].engine
            for position in range(len(SEATS))
        ]
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
            for seat in seats:
                await seat.close_interaction()

        if journal is not None:
            journal.finish()
            journal.close()
        summary = summaries[0]
        if episode_dir is not None:
            _write_episode_artifacts(
                episode_dir,
                seed,
                summary,
                seats,
                self.config,
                rotation,
                checkpoint_identity,
            )

        scores = dict(zip(summary.names, summary.scores, strict=True))
        evaluated = self.config.evaluated_agent
        policy_roles = (
            {evaluated}
            if evaluated is not None
            else {name for name in SEATS if getattr(self.config, name).model is None}
        )
        for seat in seats:
            seat.record_outcome(
                summary=summary,
                score=scores[seat.name],
                seed=seed,
                rotation=rotation,
                hands_replayed=hands_replayed,
                evaluated_agent=evaluated,
                policy_role=seat.name in policy_roles,
            )


def _write_episode_artifacts(
    episode_dir: Path,
    seed: int,
    summary: arena.GameSummary,
    seats: list[SeatRuntime],
    config: RiichiHanchanEnvConfig,
    rotation: int,
    checkpoint: dict[str, object] | None = None,
) -> None:
    for seat in seats:
        path = episode_dir / "decisions" / decision_filename(seat.name)
        with path.open("w", encoding="utf-8") as handle:
            for record in seat.decisions:
                handle.write(
                    json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
                )
    (episode_dir / "config.json").write_text(
        json.dumps(
            {
                "label": episode_dir.name,
                "created": datetime.now(UTC).isoformat(),
                # `names` are the engine names the mjai log and the reviews are keyed
                # by; `models` is the spec each one played, paired by position.
                "models": [seat.model for seat in seats],
                "names": list(SEATS),
                "games": 1,
                "seed_start": [seed, 1],
                "rotation": rotation,
                "evaluated_agent": config.evaluated_agent,
                "table": [
                    SEATS[(position - rotation) % len(SEATS)]
                    for position in range(len(SEATS))
                ],
                # the effective value: tools mode replaces inline hints, so a run dir
                # would otherwise claim hints the seats never saw.
                "state_hints": bool(config.state_hints) and not config.tools,
                "tools": bool(config.tools),
                "max_tool_calls": int(config.max_tool_calls),
                "control_use_policy": bool(config.control_use_policy),
                "reviewer_checkpoint": checkpoint,
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
