"""riichi-decision-v1: one riichi mahjong decision, graded by Mortal.

Each task is a real board position from a played hanchan, rendered from one seat's point
of view with its legal actions numbered. The reward is Mortal's normalised Q-advantage
for the option chosen: 1.0 for Mortal's own choice, 0.0 for its worst, linear between.

That is the per-decision term of the rating jongbench reports for a full game, so this
taskset and a played hanchan measure the same quantity. What it buys is separability:
every model sees byte-identical prompts on identical boards, which live play cannot give
you because each model steers the board it is then judged on. A decision task yields a
dense score for each model call; a hanchan outcome batch is a smoke test until its
duplicate-block standard error ships beside it.

Build a bank with `jongbench positions --out bank.jsonl`; the tasks are pure trace
scoring, so no runtime and no Mortal checkpoint are needed to evaluate against it.
"""

from __future__ import annotations

import json
import random
import re
from collections.abc import Iterator
from pathlib import Path

import verifiers.v1 as vf

from riichi_decision_v1.bank_schema import (
    BANK_FORMAT,
    BANK_SCHEMA_VERSION,
    COMPETENCE_TAGS,
    REWARD_NAME,
    REWARD_NORMALIZATION,
    BankManifest,
    BankRow,
    DecisionInfo,
    clustered_mean_ci,
    load_bank,
    permute_row,
)

SAMPLE_BANK = Path(__file__).with_name("sample_bank.jsonl.gz")
"""Shipped sample from Mortal self-play, so the taskset runs out of the box."""


def _extract_choice(text: str, n_options: int) -> int:
    if n_options <= 0:
        raise ValueError("no options available")
    choices: list[int] = []
    for match in re.finditer(r"\{[^{}]*\}", text, flags=re.DOTALL):
        try:
            obj = json.loads(match.group(0))
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and "choice" in obj:
            choice = obj["choice"]
            if not isinstance(choice, int) or isinstance(choice, bool):
                raise ValueError("choice must be an integer")
            if not 0 <= choice < n_options:
                raise ValueError("choice out of range")
            choices.append(choice)

    if choices:
        if len(set(choices)) != 1:
            raise ValueError("conflicting choice values")
        return choices[-1]

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if lines and re.fullmatch(r"\d+", lines[-1]):
        choice = int(lines[-1])
        if not 0 <= choice < n_options:
            raise ValueError("choice out of range")
        return choice

    raise ValueError("no choice found")


class RiichiDecisionData(vf.TaskData):
    position_id: str
    """Stable SHA-256 identity of the board, rendering, and grading together."""
    board_id: str
    """Identity of the board a seat faced, independent of prompt wording."""
    prompt_id: str
    """Identity of the rendered prompts and menu."""
    game_id: str
    """Cluster key: positions from one hand/game are not independent samples."""
    menu: list[str]
    """The legal actions, in the order the prompt numbers them."""
    rewards: list[float]
    """Mortal's normalised Q-advantage per menu index."""
    q_values: list[float]
    """Raw reviewer Q per menu index, in the reviewer's own return units."""
    best_index: int
    """The option Mortal would have taken."""
    tags: list[str]
    """Competence tags computed at bank-build time."""
    reviewer_confidence: float | None
    """Optional policy-imitation confidence metadata; never a grading weight."""
    info: DecisionInfo
    """Board context: seat, kyoku, honba, junme, tiles_left, shanten, at_furiten."""
    state_hints: bool
    """Which frozen prompt variant this task used."""


class RiichiDecisionTask(vf.Task[RiichiDecisionData]):
    @vf.stop
    async def single_turn(self, trace: vf.Trace) -> bool:
        return trace.num_turns >= 1

    @vf.reward(weight=1.0)
    async def q_advantage(self, trace: vf.Trace) -> float:
        """Mortal's opinion of the chosen action.

        An unparseable or out-of-range reply scores zero. Tag metrics include those
        failures; otherwise malformed replies disappear from every competence slice
        and bias the profile upward.
        """
        choice = self._choice(trace)
        reward = 0.0 if choice is None else float(self.data.rewards[choice])
        for tag in self.data.tags:
            trace.record_metric(f"tag_{tag}", reward)
        return reward

    @vf.metric
    async def matched_mortal(self, trace: vf.Trace) -> float:
        choice = self._choice(trace)
        return float(choice == self.data.best_index)

    @vf.metric
    async def answered(self, trace: vf.Trace) -> float:
        return float(self._choice(trace) is not None)

    @vf.metric
    async def q_loss(self, trace: vf.Trace) -> float:
        return self._q_loss_value(trace)

    @vf.metric
    async def normalised_q_loss(self, trace: vf.Trace) -> float:
        q_values = [float(value) for value in self.data.q_values]
        span = max(q_values) - min(q_values)
        if span <= 0.0:
            return 0.0
        return self._q_loss_value(trace) / span

    @vf.metric
    async def q_span(self, trace: vf.Trace) -> float:
        del trace
        q_values = [float(value) for value in self.data.q_values]
        return max(q_values) - min(q_values)

    @vf.metric
    async def choice_index(self, trace: vf.Trace) -> float:
        choice = self._choice(trace)
        return float(choice) if choice is not None else -1.0

    def _q_loss_value(self, trace: vf.Trace) -> float:
        choice = self._choice(trace)
        q_values = [float(value) for value in self.data.q_values]
        if choice is None:
            return max(q_values) - min(q_values)
        return max(q_values) - q_values[choice]

    def _choice(self, trace: vf.Trace) -> int | None:
        try:
            return _extract_choice(trace.last_reply or "", len(self.data.menu))
        except ValueError:
            return None


class RiichiDecisionConfig(vf.TasksetConfig):
    bank: str = str(SAMPLE_BANK)
    """Path to a rendered task bank (`.jsonl` or `.jsonl.gz`). Defaults to the
    shipped sample; build a bigger one with `jongbench positions`."""
    state_hints: bool = True
    """Include rule-derived shanten, waits and furiten, as the CLI does by default."""
    tags: str = ""
    """Comma-separated competence tags. A position is kept if it has any of them."""
    min_confidence: float | None = None
    """Retained only to reject the old, invalid confidence-filtering experiment."""
    confidence_weight: bool = False
    """Retained only to reject the old, invalid confidence-weighting experiment."""
    permute_seed: int | None = None
    """Shuffle option numbering with this seed (experimental robustness arm)."""
    both_prompt_variants: bool = False
    """Emit both hint arms (experimental; reduce as paired variants, not one mean)."""
    probes: bool = False
    """Append experimental board-comprehension questions."""


class RiichiDecisionTaskset(vf.Taskset[RiichiDecisionTask, RiichiDecisionConfig]):
    def load(self) -> Iterator[RiichiDecisionTask]:
        if self.config.min_confidence is not None or self.config.confidence_weight:
            raise ValueError(
                "reviewer confidence filtering/weighting is unsupported: the available "
                "checkpoint head predicts policy imitation correctness, not uncertainty "
                "in Mortal's Q grading"
            )
        path = Path(self.config.bank)
        if not path.exists():
            raise FileNotFoundError(
                f"no position bank at {path}. Build one with "
                f"`jongbench positions --out {path}`, or use the shipped sample: "
                f"{SAMPLE_BANK}"
            )
        _, rows = load_bank(path)
        wanted = {
            tag.strip()
            for tag in str(self.config.tags).split(",")
            if tag.strip()
        }
        invalid = [
            tag
            for tag in wanted
            if re.fullmatch(r"^[a-z][a-z0-9_]*$", tag) is None
        ]
        if invalid:
            raise ValueError(
                f"invalid competence tag(s): {', '.join(sorted(invalid))}"
            )
        filtered = [
            row for row in rows if not wanted or not wanted.isdisjoint(row["tags"])
        ]
        if not filtered:
            raise ValueError("tag filter removed every position")

        idx = 0
        variants = (
            (True, False)
            if self.config.both_prompt_variants
            else (self.config.state_hints,)
        )
        permute_rng = (
            random.Random(self.config.permute_seed)
            if self.config.permute_seed is not None
            else None
        )
        for row in filtered:
            working = row
            if permute_rng is not None:
                order = list(range(len(row["menu"])))
                permute_rng.shuffle(order)
                working = permute_row(row, order)
            for hints in variants:
                yield self._task(idx, working, state_hints=bool(hints))
                idx += 1
            if self.config.probes:
                for probe in self._probe_tasks(idx, working):
                    yield probe
                    idx += 1

    def _task(
        self, idx: int, row: BankRow, *, state_hints: bool
    ) -> RiichiDecisionTask:
        prompt_key = "prompt" if state_hints else "prompt_without_state_hints"
        return RiichiDecisionTask(
            RiichiDecisionData(
                idx=idx,
                name=row["name"],
                prompt=row[prompt_key],
                system_prompt=row["system_prompt"],
                position_id=row["id"],
                board_id=row["board_id"],
                prompt_id=row["prompt_id"],
                game_id=row["game_id"],
                menu=row["menu"],
                rewards=row["rewards"],
                q_values=row["q_values"],
                best_index=row["best_index"],
                tags=list(row["tags"]),
                reviewer_confidence=row.get("reviewer_confidence"),
                info=row["info"],
                state_hints=state_hints,
            ),
            self.config.task,
        )

    def _probe_tasks(self, idx: int, row: BankRow) -> list[RiichiDecisionTask]:
        """Rule-checkable questions about the board just shown."""
        info = row["info"]
        board = row["prompt_without_state_hints"]
        furiten_yes = bool(info["at_furiten"])
        furiten_menu = ["no", "yes"]
        tiles_left = int(info["tiles_left"])
        tile_options = sorted(
            {
                max(0, tiles_left - 2),
                max(0, tiles_left - 1),
                tiles_left,
                min(70, tiles_left + 1),
            }
        )
        probes = [
            (
                "furiten",
                (
                    f"{board}\n\nIs this seat currently in furiten?\n"
                    "Choose your action:\n0: no\n1: yes\n\n"
                    'Reply with exactly: {"choice": N}'
                ),
                furiten_menu,
                [0.0 if furiten_yes else 1.0, 1.0 if furiten_yes else 0.0],
                1 if furiten_yes else 0,
            ),
            (
                "tiles_left",
                (
                    f"{board}\n\nHow many tiles remain in the wall?\n"
                    "Choose your action:\n"
                    + "\n".join(f"{i}: {n}" for i, n in enumerate(tile_options))
                    + '\n\nReply with exactly: {"choice": N}'
                ),
                [str(n) for n in tile_options],
                [1.0 if n == tiles_left else 0.0 for n in tile_options],
                tile_options.index(tiles_left),
            ),
        ]
        tasks: list[RiichiDecisionTask] = []
        for offset, (kind, prompt, menu, rewards, best) in enumerate(probes):
            q_values = [float(value) for value in rewards]
            tasks.append(
                RiichiDecisionTask(
                    RiichiDecisionData(
                        idx=idx + offset,
                        name=f"{row['name']}-probe-{kind}",
                        prompt=prompt,
                        system_prompt=row["system_prompt"],
                        position_id=f"{row['id']}:probe:{kind}",
                        board_id=row["board_id"],
                        prompt_id=f"{row['prompt_id']}:probe:{kind}",
                        game_id=row["game_id"],
                        menu=menu,
                        rewards=rewards,
                        q_values=q_values,
                        best_index=best,
                        tags=["comprehension", kind],
                        reviewer_confidence=None,
                        info=info,
                        state_hints=False,
                    ),
                    self.config.task,
                )
            )
        return tasks


# Re-exported so tests and `from riichi_decision_v1.taskset import load_bank` keep working.
__all__ = [
    "BANK_FORMAT",
    "BANK_SCHEMA_VERSION",
    "REWARD_NAME",
    "REWARD_NORMALIZATION",
    "BankManifest",
    "BankRow",
    "COMPETENCE_TAGS",
    "DecisionInfo",
    "RiichiDecisionConfig",
    "RiichiDecisionData",
    "RiichiDecisionTask",
    "RiichiDecisionTaskset",
    "SAMPLE_BANK",
    "clustered_mean_ci",
    "load_bank",
    "_extract_choice",
]
