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

import json
from collections.abc import Iterator
from pathlib import Path

import verifiers.v1 as vf
from jongbench import positions, prompts


class RiichiDecisionData(vf.TaskData):
    menu: list[str]
    """The legal actions, in the order the prompt numbers them."""
    rewards: list[float]
    """Mortal's normalised Q-advantage per menu index."""
    best_index: int
    """The option Mortal would have taken."""
    info: dict = {}
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
            return prompts.extract_choice(trace.last_reply or "", len(self.data.menu))
        except ValueError:
            return None


class RiichiDecisionConfig(vf.TasksetConfig):
    bank: str = "bank.jsonl"
    """Path to a position bank, one `jongbench.positions.Position` JSON per line."""
    state_hints: bool = True
    """Include rule-derived shanten, waits and furiten, as the CLI does by default."""


class RiichiDecisionTaskset(vf.Taskset[RiichiDecisionTask, RiichiDecisionConfig]):
    def load(self) -> Iterator[RiichiDecisionTask]:
        path = Path(self.config.bank)
        if not path.exists():
            raise FileNotFoundError(
                f"no position bank at {path}. Build one with "
                f"`jongbench positions --out {path}`"
            )
        with path.open() as handle:
            for idx, line in enumerate(handle):
                line = line.strip()
                if not line:
                    continue
                position = positions.Position.from_dict(json.loads(line))
                yield RiichiDecisionTask(
                    RiichiDecisionData(
                        idx=idx,
                        name=f"kyoku{position.kyoku}-junme{position.junme}-seat{position.player_id}",
                        prompt=position.prompt(state_hints=self.config.state_hints),
                        system_prompt=prompts.SYSTEM,
                        menu=position.menu,
                        rewards=position.rewards,
                        best_index=position.best_index,
                        info={
                            "seat": position.player_id,
                            "kyoku": position.kyoku,
                            "honba": position.honba,
                            "junme": position.junme,
                            "tiles_left": position.tiles_left,
                            "shanten": position.shanten,
                            "at_furiten": position.at_furiten,
                        },
                    ),
                    self.config.task,
                )
