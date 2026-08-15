"""riichi-decision-v1: one riichi mahjong decision, graded by Mortal.

Each task is a real board position from a played hanchan, rendered from one seat's point
of view with its legal actions numbered. The reward is Mortal's normalised Q-advantage
for the option chosen: 1.0 for Mortal's own choice, 0.0 for its worst, linear between.

That is the per-decision term of the rating jongbench reports for a full game, so this
taskset and a played hanchan measure the same quantity. What it buys is separability:
every model sees byte-identical prompts on identical boards, which live play cannot give
you because each model steers the board it is then judged on. It is also ~1,000x cheaper
per graded decision, since a hanchan costs ~1,000 model calls to reach roughly as many
decisions that only that game's players ever face.

Build a bank with `jongbench positions --out bank.jsonl`; the tasks are pure trace
scoring, so no runtime and no Mortal checkpoint are needed to evaluate against it.
"""

import gzip
import json
import re
from collections.abc import Iterator
from pathlib import Path

import verifiers.v1 as vf

SAMPLE_BANK = Path(__file__).with_name("sample_bank.jsonl.gz")
"""128 positions from Mortal self-play, shipped so the taskset runs out of the box.
Uniform-random guessing scores 0.367 reward / 18.9% match on it; Mortal scores 1.0."""


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


def _validate_row(row: dict, idx: int) -> None:
    if row.get("schema_version") != 1:
        raise ValueError(f"bank row {idx} has unsupported schema_version")
    menu = row.get("menu")
    rewards = row.get("rewards")
    if not isinstance(menu, list) or not isinstance(rewards, list):
        raise TypeError(f"bank row {idx} must contain menu and rewards lists")
    if len(menu) < 2 or len(menu) != len(rewards):
        raise ValueError(f"bank row {idx} has mismatched menu and rewards")
    best_index = row.get("best_index")
    if (
        not isinstance(best_index, int)
        or isinstance(best_index, bool)
        or not 0 <= best_index < len(menu)
    ):
        raise ValueError(f"bank row {idx} has invalid best_index")


class RiichiDecisionData(vf.TaskData):
    menu: list[str]
    """The legal actions, in the order the prompt numbers them."""
    rewards: list[float]
    """Mortal's normalised Q-advantage per menu index."""
    best_index: int
    """The option Mortal would have taken."""
    info: dict
    """Board context: seat, kyoku, honba, junme, tiles_left, shanten, at_furiten."""


class RiichiDecisionTask(vf.Task[RiichiDecisionData]):
    @vf.stop
    async def single_turn(self, trace: vf.Trace) -> bool:
        return trace.num_turns >= 1

    @vf.reward(weight=1.0)
    async def q_advantage(self, trace: vf.Trace) -> float:
        """Mortal's opinion of the chosen action. An unparseable or out-of-range reply
        scores 0.0 rather than erroring: failing to answer in the required form is a
        real failure at this task, not a broken sample."""
        choice = self._choice(trace)
        if choice is None:
            return 0.0
        return float(self.data.rewards[choice])

    @vf.metric
    async def matched_mortal(self, trace: vf.Trace) -> float:
        choice = self._choice(trace)
        return float(choice == self.data.best_index)

    @vf.metric
    async def answered(self, trace: vf.Trace) -> float:
        return float(self._choice(trace) is not None)

    def _choice(self, trace: vf.Trace) -> int | None:
        try:
            return _extract_choice(trace.last_reply or "", len(self.data.menu))
        except ValueError:
            return None


class RiichiDecisionConfig(vf.TasksetConfig):
    bank: str = str(SAMPLE_BANK)
    """Path to a rendered task bank (`.jsonl` or `.jsonl.gz`). Defaults to the
    shipped 128-position sample; build a bigger one with `jongbench positions`."""
    state_hints: bool = True
    """Include rule-derived shanten, waits and furiten, as the CLI does by default."""


class RiichiDecisionTaskset(vf.Taskset[RiichiDecisionTask, RiichiDecisionConfig]):
    def load(self) -> Iterator[RiichiDecisionTask]:
        path = Path(self.config.bank)
        if not path.exists():
            raise FileNotFoundError(
                f"no position bank at {path}. Build one with "
                f"`jongbench positions --out {path}`, or use the shipped sample: "
                f"{SAMPLE_BANK}"
            )
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rt", encoding="utf-8") as handle:
            for idx, line in enumerate(handle):
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                _validate_row(row, idx)
                prompt_key = (
                    "prompt"
                    if self.config.state_hints
                    else "prompt_without_state_hints"
                )
                yield RiichiDecisionTask(
                    RiichiDecisionData(
                        idx=idx,
                        name=row["name"],
                        prompt=row[prompt_key],
                        system_prompt=row["system_prompt"],
                        menu=row["menu"],
                        rewards=row["rewards"],
                        best_index=row["best_index"],
                        info=row["info"],
                    ),
                    self.config.task,
                )
