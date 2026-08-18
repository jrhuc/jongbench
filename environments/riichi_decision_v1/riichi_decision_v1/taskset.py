"""riichi-decision-v1: one riichi mahjong decision, graded by Mortal.

Each task is a real board position from a played hanchan, rendered from one seat's point
of view with its legal actions numbered. The reward is Mortal's normalised Q-advantage
for the option chosen: 1.0 for Mortal's own choice, 0.0 for its worst, linear between.

That is the per-decision term of the rating jongbench reports for a full game, so this
taskset and a played hanchan measure the same quantity. What it buys is separability:
every model sees byte-identical prompts on identical boards, which live play cannot give
you because each model steers the board it is then judged on. A decision task yields a
dense score for each model call; the standard four-hanchan outcome batch spends roughly
1,000 calls to produce only four noisy placement samples.

Build a bank with `jongbench positions --out bank.jsonl`; the tasks are pure trace
scoring, so no runtime and no Mortal checkpoint are needed to evaluate against it.
"""

import gzip
import hashlib
import json
import math
import re
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any, Literal, Required, TypedDict, cast

import verifiers.v1 as vf

BANK_SCHEMA_VERSION = 2
BANK_FORMAT = "jongbench.riichi-decision-bank"
REWARD_NAME = "mortal_q_advantage"
REWARD_NORMALIZATION = "per_position_min_max"

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


class DecisionInfo(TypedDict):
    seat: int
    kyoku: int
    honba: int
    junme: int
    tiles_left: int
    shanten: int
    at_furiten: bool


class BankRow(TypedDict):
    record_type: Literal["position"]
    schema_version: Literal[2]
    id: str
    name: str
    prompt: str
    prompt_without_state_hints: str
    system_prompt: str
    menu: list[str]
    rewards: list[float]
    best_index: int
    info: DecisionInfo


class GeneratorProvenance(TypedDict):
    name: str
    version: str


class RewardContract(TypedDict):
    name: str
    normalization: str
    range: list[float]


class ReviewerProvenance(TypedDict):
    name: str
    checkpoint: str
    checkpoint_sha256: str
    temperature: float


class SourceArtifact(TypedDict):
    name: str
    sha256: str


class SourceProvenance(TypedDict, total=False):
    kind: Required[str]
    description: str
    seed: int
    games: int
    artifacts: list[SourceArtifact]


class BankManifest(TypedDict):
    record_type: Literal["manifest"]
    schema_version: Literal[2]
    bank_format: str
    generator: GeneratorProvenance
    reward: RewardContract
    reviewer: ReviewerProvenance
    source: SourceProvenance


_ROW_FIELDS = frozenset(BankRow.__required_keys__)
_INFO_FIELDS = frozenset(DecisionInfo.__required_keys__)
_MANIFEST_FIELDS = frozenset(BankManifest.__required_keys__)
_ID_FIELDS = (
    "prompt",
    "prompt_without_state_hints",
    "system_prompt",
    "menu",
    "rewards",
    "best_index",
    "info",
)
_SHA256_RE = re.compile(r"[0-9a-f]{64}")


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _position_id(row: Mapping[str, object]) -> str:
    identity = {key: row[key] for key in _ID_FIELDS}
    return "sha256:" + hashlib.sha256(_canonical_json(identity).encode()).hexdigest()


def _object(value: object, where: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{where} must be a JSON object")
    return value


def _exact_fields(
    value: Mapping[str, object], fields: frozenset[str], where: str
) -> None:
    missing = fields - value.keys()
    if missing:
        raise ValueError(
            f"{where} is missing required field(s): {', '.join(sorted(missing))}"
        )
    extra = value.keys() - fields
    if extra:
        raise ValueError(f"{where} has unknown field(s): {', '.join(sorted(extra))}")


def _nonempty_string(value: object, where: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TypeError(f"{where} must be a non-empty string")
    return value


def _integer(value: object, where: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{where} must be an integer")
    return value


def _validate_row(value: object, idx: int) -> BankRow:
    where = f"bank row {idx}"
    row = _object(value, where)
    _exact_fields(row, _ROW_FIELDS, where)
    _nonempty_string(row["record_type"], f"{where}.record_type")
    if row["record_type"] != "position":
        raise ValueError(f"{where} has invalid record_type")
    if (
        type(row["schema_version"]) is not int
        or row["schema_version"] != BANK_SCHEMA_VERSION
    ):
        raise ValueError(f"{where} has unsupported schema_version")
    for key in ("id", "name", "prompt", "prompt_without_state_hints", "system_prompt"):
        _nonempty_string(row[key], f"{where}.{key}")

    menu = row["menu"]
    if not isinstance(menu, list):
        raise TypeError(f"{where}.menu must be a list")
    if len(menu) < 2:
        raise ValueError(f"{where}.menu must contain at least two options")
    for index, label in enumerate(menu):
        _nonempty_string(label, f"{where}.menu[{index}]")
    if len(set(menu)) != len(menu):
        raise ValueError(f"{where}.menu must not contain duplicate options")

    rewards = row["rewards"]
    if not isinstance(rewards, list):
        raise TypeError(f"{where}.rewards must be a list")
    if len(rewards) != len(menu):
        raise ValueError(f"{where} has mismatched menu and rewards lengths")
    for index, reward in enumerate(rewards):
        if not isinstance(reward, (int, float)) or isinstance(reward, bool):
            raise TypeError(f"{where}.rewards[{index}] must be a number")
        if not math.isfinite(float(reward)):
            raise ValueError(f"{where}.rewards[{index}] must be finite")
        if not 0.0 <= float(reward) <= 1.0:
            raise ValueError(f"{where}.rewards[{index}] must be in [0, 1]")

    best_index = _integer(row["best_index"], f"{where}.best_index")
    if not 0 <= best_index < len(menu):
        raise ValueError(f"{where}.best_index is out of range")
    if float(rewards[best_index]) != max(map(float, rewards)):
        raise ValueError(f"{where}.best_index must select an argmax reward")

    info = _object(row["info"], f"{where}.info")
    _exact_fields(info, _INFO_FIELDS, f"{where}.info")
    seat = _integer(info["seat"], f"{where}.info.seat")
    if not 0 <= seat <= 3:
        raise ValueError(f"{where}.info.seat must be in [0, 3]")
    for key in ("kyoku", "honba", "junme"):
        if _integer(info[key], f"{where}.info.{key}") < 0:
            raise ValueError(f"{where}.info.{key} must be non-negative")
    tiles_left = _integer(info["tiles_left"], f"{where}.info.tiles_left")
    if not 0 <= tiles_left <= 70:
        raise ValueError(f"{where}.info.tiles_left must be in [0, 70]")
    shanten = _integer(info["shanten"], f"{where}.info.shanten")
    if not -1 <= shanten <= 13:
        raise ValueError(f"{where}.info.shanten must be in [-1, 13]")
    if not isinstance(info["at_furiten"], bool):
        raise TypeError(f"{where}.info.at_furiten must be a boolean")

    if row["id"] != _position_id(row):
        raise ValueError(f"{where}.id does not match its canonical content hash")
    return cast(BankRow, row)


def _validate_manifest(value: object) -> BankManifest:
    where = "bank manifest"
    manifest = _object(value, where)
    _exact_fields(manifest, _MANIFEST_FIELDS, where)
    _nonempty_string(manifest["record_type"], f"{where}.record_type")
    if manifest["record_type"] != "manifest":
        raise ValueError(f"{where} has invalid record_type")
    if (
        type(manifest["schema_version"]) is not int
        or manifest["schema_version"] != BANK_SCHEMA_VERSION
    ):
        raise ValueError(f"{where} has unsupported schema_version")
    _nonempty_string(manifest["bank_format"], f"{where}.bank_format")
    if manifest["bank_format"] != BANK_FORMAT:
        raise ValueError(f"{where} has unsupported bank_format")

    generator = _object(manifest["generator"], f"{where}.generator")
    _exact_fields(
        generator,
        frozenset(GeneratorProvenance.__required_keys__),
        f"{where}.generator",
    )
    _nonempty_string(generator["name"], f"{where}.generator.name")
    _nonempty_string(generator["version"], f"{where}.generator.version")

    reward = _object(manifest["reward"], f"{where}.reward")
    _exact_fields(
        reward, frozenset(RewardContract.__required_keys__), f"{where}.reward"
    )
    _nonempty_string(reward["name"], f"{where}.reward.name")
    _nonempty_string(reward["normalization"], f"{where}.reward.normalization")
    if reward["name"] != REWARD_NAME or reward["normalization"] != REWARD_NORMALIZATION:
        raise ValueError(f"{where} has unsupported reward semantics")
    reward_range = reward["range"]
    if not isinstance(reward_range, list) or len(reward_range) != 2:
        raise TypeError(f"{where}.reward.range must be a two-number list")
    if any(
        not isinstance(item, (int, float)) or isinstance(item, bool)
        for item in reward_range
    ):
        raise TypeError(f"{where}.reward.range must contain numbers")
    if not all(math.isfinite(float(item)) for item in reward_range):
        raise ValueError(f"{where}.reward.range must be finite")
    if list(map(float, reward_range)) != [0.0, 1.0]:
        raise ValueError(f"{where}.reward.range must be [0.0, 1.0]")

    reviewer = _object(manifest["reviewer"], f"{where}.reviewer")
    _exact_fields(
        reviewer, frozenset(ReviewerProvenance.__required_keys__), f"{where}.reviewer"
    )
    _nonempty_string(reviewer["name"], f"{where}.reviewer.name")
    _nonempty_string(reviewer["checkpoint"], f"{where}.reviewer.checkpoint")
    checkpoint_sha256 = _nonempty_string(
        reviewer["checkpoint_sha256"], f"{where}.reviewer.checkpoint_sha256"
    )
    if _SHA256_RE.fullmatch(checkpoint_sha256) is None:
        raise ValueError(f"{where}.reviewer.checkpoint_sha256 must be a SHA-256 digest")
    temperature = reviewer["temperature"]
    if not isinstance(temperature, (int, float)) or isinstance(temperature, bool):
        raise TypeError(f"{where}.reviewer.temperature must be a number")
    if not math.isfinite(float(temperature)) or float(temperature) < 0:
        raise ValueError(
            f"{where}.reviewer.temperature must be finite and non-negative"
        )

    source = _object(manifest["source"], f"{where}.source")
    _nonempty_string(source.get("kind"), f"{where}.source.kind")
    if "description" in source:
        _nonempty_string(source["description"], f"{where}.source.description")
    for key in ("seed", "games"):
        if key in source and _integer(source[key], f"{where}.source.{key}") < 0:
            raise ValueError(f"{where}.source.{key} must be non-negative")
    if "artifacts" in source:
        artifacts = source["artifacts"]
        if not isinstance(artifacts, list):
            raise TypeError(f"{where}.source.artifacts must be a list")
        for index, artifact_value in enumerate(artifacts):
            artifact_where = f"{where}.source.artifacts[{index}]"
            artifact = _object(artifact_value, artifact_where)
            _exact_fields(
                artifact, frozenset(SourceArtifact.__required_keys__), artifact_where
            )
            _nonempty_string(artifact["name"], f"{artifact_where}.name")
            digest = _nonempty_string(artifact["sha256"], f"{artifact_where}.sha256")
            if _SHA256_RE.fullmatch(digest) is None:
                raise ValueError(f"{artifact_where}.sha256 must be a SHA-256 digest")
    _canonical_json(source)
    return cast(BankManifest, manifest)


def load_bank(path: str | Path) -> tuple[BankManifest, list[BankRow]]:
    """Load and fully validate a standalone schema-v2 JSONL decision bank."""
    bank_path = Path(path)
    opener = gzip.open if bank_path.suffix == ".gz" else open
    with opener(bank_path, "rt", encoding="utf-8") as handle:
        first_line = handle.readline()
        if not first_line:
            raise ValueError("bank is empty; first line must be a manifest")
        if not first_line.strip():
            raise ValueError("bank first line must be a manifest JSON object")
        try:
            manifest_value = json.loads(first_line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON in bank manifest: {exc.msg}") from exc
        manifest = _validate_manifest(manifest_value)

        rows: list[BankRow] = []
        seen_ids: set[str] = set()
        for line_number, line in enumerate(handle, start=2):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"invalid JSON in bank row {len(rows)} "
                    f"(line {line_number}): {exc.msg}"
                ) from exc
            row = _validate_row(value, len(rows))
            if row["id"] in seen_ids:
                raise ValueError(f"bank row {len(rows)}.id duplicates an earlier row")
            seen_ids.add(row["id"])
            rows.append(row)
    if not rows:
        raise ValueError("bank must contain at least one position row")
    return manifest, rows


class RiichiDecisionData(vf.TaskData):
    position_id: str
    """Stable SHA-256 identity of the rendered and graded position."""
    menu: list[str]
    """The legal actions, in the order the prompt numbers them."""
    rewards: list[float]
    """Mortal's normalised Q-advantage per menu index."""
    best_index: int
    """The option Mortal would have taken."""
    info: DecisionInfo
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
        _, rows = load_bank(path)
        prompt_key = (
            "prompt" if self.config.state_hints else "prompt_without_state_hints"
        )
        for idx, row in enumerate(rows):
            yield RiichiDecisionTask(
                RiichiDecisionData(
                    idx=idx,
                    name=row["name"],
                    prompt=row[prompt_key],
                    system_prompt=row["system_prompt"],
                    position_id=row["id"],
                    menu=row["menu"],
                    rewards=row["rewards"],
                    best_index=row["best_index"],
                    info=row["info"],
                ),
                self.config.task,
            )
