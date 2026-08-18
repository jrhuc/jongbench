"""Schema v3 for a frozen riichi-decision bank.

Copied byte-for-byte to
``environments/riichi_decision_v1/riichi_decision_v1/bank_schema.py``.
The standalone loader must not import jongbench; a conformance test keeps the
two files identical. This module is stdlib-only.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import random
import re
from collections.abc import Iterable, Mapping, Sequence
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Literal, Required, TextIO, TypedDict, cast

BANK_SCHEMA_VERSION = 3
BANK_FORMAT = "jongbench.riichi-decision-bank"
REWARD_NAME = "mortal_q_advantage"
REWARD_NORMALIZATION = "per_position_min_max"

COMPETENCE_TAGS = (
    "pushfold",
    "defense_only",
    "riichi_choice",
    "call_choice",
    "efficiency",
    "oorasu_placement",
    "kan_choice",
    "furiten",
    "last_turns",
)
_TAG_RE = re.compile(r"^[a-z][a-z0-9_]*$")
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_CONTENT_ID_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_OPTION_LINE_RE = re.compile(r"^(\d+): (.*)$")


class DecisionInfo(TypedDict, total=False):
    seat: Required[int]
    kyoku: Required[int]
    honba: Required[int]
    junme: Required[int]
    tiles_left: Required[int]
    shanten: Required[int]
    at_furiten: Required[bool]


class BankRow(TypedDict, total=False):
    record_type: Required[Literal["position"]]
    schema_version: Required[Literal[3]]
    id: Required[str]
    board_id: Required[str]
    prompt_id: Required[str]
    name: Required[str]
    prompt: Required[str]
    prompt_without_state_hints: Required[str]
    system_prompt: Required[str]
    menu: Required[list[str]]
    rewards: Required[list[float]]
    q_values: Required[list[float]]
    best_index: Required[int]
    game_id: Required[str]
    tags: Required[list[str]]
    info: Required[DecisionInfo]
    source_log: str
    reviewer_confidence: float


class GeneratorProvenance(TypedDict):
    name: str
    version: str


class RewardContract(TypedDict):
    name: str
    normalization: str
    range: list[float]


class ReviewerProvenance(TypedDict, total=False):
    name: Required[str]
    checkpoint: Required[str]
    checkpoint_sha256: Required[str]
    temperature: Required[float]


class SourceArtifact(TypedDict):
    name: str
    sha256: str


class SourceProvenance(TypedDict, total=False):
    kind: Required[str]
    description: str
    seed: int
    games: int
    artifacts: list[SourceArtifact]


class BankManifest(TypedDict, total=False):
    record_type: Required[Literal["manifest"]]
    schema_version: Required[Literal[3]]
    bank_format: Required[str]
    generator: Required[GeneratorProvenance]
    reward: Required[RewardContract]
    reviewer: Required[ReviewerProvenance]
    source: Required[SourceProvenance]


_ROW_REQUIRED = frozenset(
    {
        "record_type",
        "schema_version",
        "id",
        "board_id",
        "prompt_id",
        "name",
        "prompt",
        "prompt_without_state_hints",
        "system_prompt",
        "menu",
        "rewards",
        "q_values",
        "best_index",
        "game_id",
        "tags",
        "info",
    }
)
_INFO_REQUIRED = frozenset(
    {
        "seat",
        "kyoku",
        "honba",
        "junme",
        "tiles_left",
        "shanten",
        "at_furiten",
    }
)
_MANIFEST_REQUIRED = frozenset(
    {
        "record_type",
        "schema_version",
        "bank_format",
        "generator",
        "reward",
        "reviewer",
        "source",
    }
)
_PROMPT_ID_FIELDS = (
    "prompt",
    "prompt_without_state_hints",
    "system_prompt",
    "menu",
)
_CONTENT_ID_FIELDS = (
    "board_id",
    "prompt_id",
    "rewards",
    "best_index",
    "q_values",
)


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _digest(value: object) -> str:
    return "sha256:" + hashlib.sha256(_canonical_json(value).encode()).hexdigest()


def board_id(*, seat: int, events: Sequence[Mapping[str, object]]) -> str:
    """Identity of the board a seat faced, independent of prompt wording."""
    return _digest({"seat": seat, "events": list(events)})


def prompt_id(row: Mapping[str, object]) -> str:
    """Identity of the rendered prompts and menu, independent of grading."""
    return _digest({key: row[key] for key in _PROMPT_ID_FIELDS})


def position_id(row: Mapping[str, object]) -> str:
    """Identity of the board, the rendering, and the grading together."""
    return _digest({key: row[key] for key in _CONTENT_ID_FIELDS})


def _object(value: object, where: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{where} must be a JSON object")
    return value


def _require_fields(
    value: Mapping[str, object], fields: frozenset[str], where: str
) -> None:
    missing = fields - value.keys()
    if missing:
        raise ValueError(
            f"{where} is missing required field(s): {', '.join(sorted(missing))}"
        )


def _exact_fields(
    value: Mapping[str, object], fields: frozenset[str], where: str
) -> None:
    _require_fields(value, fields, where)
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


def _number(value: object, where: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError(f"{where} must be a number")
    if not math.isfinite(float(value)):
        raise ValueError(f"{where} must be finite")
    return float(value)


def _content_id(value: object, where: str) -> str:
    text = _nonempty_string(value, where)
    if _CONTENT_ID_RE.fullmatch(text) is None:
        raise ValueError(f"{where} must be a sha256: digest")
    return text


def _string_list(value: object, where: str) -> list[str]:
    if not isinstance(value, list):
        raise TypeError(f"{where} must be a list")
    items: list[str] = []
    for index, item in enumerate(value):
        items.append(_nonempty_string(item, f"{where}[{index}]"))
    return items


def validate_bank_row(value: object, row_index: int | None = None) -> BankRow:
    where = "bank row" if row_index is None else f"bank row {row_index}"
    row = _object(value, where)
    _require_fields(row, _ROW_REQUIRED, where)
    _canonical_json(row)
    _nonempty_string(row["record_type"], f"{where}.record_type")
    if row["record_type"] != "position":
        raise ValueError(f"{where} has invalid record_type")
    if (
        type(row["schema_version"]) is not int
        or row["schema_version"] != BANK_SCHEMA_VERSION
    ):
        raise ValueError(f"{where} has unsupported schema_version")
    for key in (
        "name",
        "prompt",
        "prompt_without_state_hints",
        "system_prompt",
        "game_id",
    ):
        _nonempty_string(row[key], f"{where}.{key}")
    _content_id(row["id"], f"{where}.id")
    _content_id(row["board_id"], f"{where}.board_id")
    _content_id(row["prompt_id"], f"{where}.prompt_id")

    menu = _string_list(row["menu"], f"{where}.menu")
    if len(menu) < 2:
        raise ValueError(f"{where}.menu must contain at least two options")
    if len(set(menu)) != len(menu):
        raise ValueError(f"{where}.menu must not contain duplicate options")

    rewards = row["rewards"]
    if not isinstance(rewards, list):
        raise TypeError(f"{where}.rewards must be a list")
    if len(rewards) != len(menu):
        raise ValueError(f"{where} has mismatched menu and rewards lengths")
    parsed_rewards: list[float] = []
    for index, reward in enumerate(rewards):
        number = _number(reward, f"{where}.rewards[{index}]")
        if not 0.0 <= number <= 1.0:
            raise ValueError(f"{where}.rewards[{index}] must be in [0, 1]")
        parsed_rewards.append(number)

    q_values = row["q_values"]
    if not isinstance(q_values, list):
        raise TypeError(f"{where}.q_values must be a list")
    if len(q_values) != len(menu):
        raise ValueError(f"{where} has mismatched menu and q_values lengths")
    parsed_q: list[float] = []
    for index, q_value in enumerate(q_values):
        parsed_q.append(_number(q_value, f"{where}.q_values[{index}]"))

    best_index = _integer(row["best_index"], f"{where}.best_index")
    if not 0 <= best_index < len(menu):
        raise ValueError(f"{where}.best_index is out of range")
    if parsed_rewards[best_index] != max(parsed_rewards):
        raise ValueError(f"{where}.best_index must select an argmax reward")
    if parsed_q[best_index] != max(parsed_q):
        raise ValueError(f"{where}.best_index must select an argmax q_value")

    tags = _string_list(row["tags"], f"{where}.tags")
    if len(set(tags)) != len(tags):
        raise ValueError(f"{where}.tags must not contain duplicates")
    for index, tag in enumerate(tags):
        if _TAG_RE.fullmatch(tag) is None:
            raise ValueError(f"{where}.tags[{index}] is not a valid tag")

    if "source_log" in row:
        _nonempty_string(row["source_log"], f"{where}.source_log")
    if "reviewer_confidence" in row:
        confidence = _number(
            row["reviewer_confidence"], f"{where}.reviewer_confidence"
        )
        if not 0.0 <= confidence <= 1.0:
            raise ValueError(f"{where}.reviewer_confidence must be in [0, 1]")

    info = _object(row["info"], f"{where}.info")
    _require_fields(info, _INFO_REQUIRED, where=f"{where}.info")
    _canonical_json(info)
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

    if row["prompt_id"] != prompt_id(row):
        raise ValueError(f"{where}.prompt_id does not match its canonical content hash")
    if row["id"] != position_id(row):
        raise ValueError(f"{where}.id does not match its canonical content hash")
    return cast(BankRow, row)


def validate_bank_manifest(value: object) -> BankManifest:
    where = "bank manifest"
    manifest = _object(value, where)
    _require_fields(manifest, _MANIFEST_REQUIRED, where)
    _canonical_json(manifest)
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
    _require_fields(
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


def bank_manifest(
    *,
    reviewer_checkpoint: str,
    reviewer_checkpoint_sha256: str,
    source: SourceProvenance,
    reviewer_temperature: float = 0.1,
    generator_version: str | None = None,
) -> BankManifest:
    """Build the required first JSONL record for a schema-v3 bank."""
    if generator_version is None:
        try:
            generator_version = version("jongbench")
        except PackageNotFoundError:
            generator_version = "0+unknown"
    source_copy = json.loads(_canonical_json(source))
    manifest: BankManifest = {
        "record_type": "manifest",
        "schema_version": BANK_SCHEMA_VERSION,
        "bank_format": BANK_FORMAT,
        "generator": {"name": "jongbench", "version": generator_version},
        "reward": {
            "name": REWARD_NAME,
            "normalization": REWARD_NORMALIZATION,
            "range": [0.0, 1.0],
        },
        "reviewer": {
            "name": "Mortal",
            "checkpoint": reviewer_checkpoint,
            "checkpoint_sha256": reviewer_checkpoint_sha256,
            "temperature": reviewer_temperature,
        },
        "source": cast(SourceProvenance, source_copy),
    }
    return validate_bank_manifest(manifest)


def dump_bank(handle: TextIO, manifest: BankManifest, rows: Iterable[BankRow]) -> int:
    """Write one validated manifest followed by validated, uniquely identified rows."""
    checked_manifest = validate_bank_manifest(manifest)
    checked_rows: list[BankRow] = []
    seen_ids: set[str] = set()
    for index, value in enumerate(rows):
        row = validate_bank_row(value, index)
        if row["id"] in seen_ids:
            raise ValueError(f"bank row {index}.id duplicates an earlier row")
        seen_ids.add(row["id"])
        checked_rows.append(row)
    if not checked_rows:
        raise ValueError("bank must contain at least one position row")
    handle.write(_canonical_json(checked_manifest) + "\n")
    for row in checked_rows:
        handle.write(_canonical_json(row) + "\n")
    return len(checked_rows)


def load_bank(path: str | Path) -> tuple[BankManifest, list[BankRow]]:
    """Load and fully validate a standalone schema-v3 JSONL decision bank."""
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
        manifest = validate_bank_manifest(manifest_value)

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
            row = validate_bank_row(value, len(rows))
            if row["id"] in seen_ids:
                raise ValueError(f"bank row {len(rows)}.id duplicates an earlier row")
            seen_ids.add(row["id"])
            rows.append(row)
    if not rows:
        raise ValueError("bank must contain at least one position row")
    return manifest, rows


def sample_rows(
    rows: Sequence[BankRow],
    limit: int,
    *,
    seed: int,
    max_per_game: int | None = None,
) -> list[BankRow]:
    """Draw a reproducible subset spread across ``game_id`` clusters."""
    if limit <= 0:
        raise ValueError("limit must be positive")
    if limit >= len(rows):
        return list(rows)
    rng = random.Random(seed)
    by_game: dict[str, list[BankRow]] = {}
    for row in rows:
        by_game.setdefault(str(row["game_id"]), []).append(row)
    for group in by_game.values():
        rng.shuffle(group)
    if max_per_game is None:
        max_per_game = max(1, math.ceil(limit / max(len(by_game), 1)))
    capped: list[BankRow] = []
    leftover: list[BankRow] = []
    for group in by_game.values():
        capped.extend(group[:max_per_game])
        leftover.extend(group[max_per_game:])
    rng.shuffle(capped)
    rng.shuffle(leftover)
    selected = (capped + leftover)[:limit]
    selected.sort(key=lambda row: (str(row["game_id"]), str(row["name"])))
    return selected


def permute_row(row: BankRow, order: Sequence[int]) -> BankRow:
    """Renumber the menu so position-bias ablations keep the same boards."""
    n = len(row["menu"])
    if sorted(order) != list(range(n)):
        raise ValueError("order must be a permutation of menu indices")
    out = dict(row)
    out["menu"] = [row["menu"][index] for index in order]
    out["rewards"] = [row["rewards"][index] for index in order]
    out["q_values"] = [row["q_values"][index] for index in order]
    out["best_index"] = list(order).index(int(row["best_index"]))
    out["prompt"] = _permute_prompt_menu(str(row["prompt"]), order)
    out["prompt_without_state_hints"] = _permute_prompt_menu(
        str(row["prompt_without_state_hints"]), order
    )
    out["prompt_id"] = prompt_id(out)
    out["id"] = position_id(out)
    return validate_bank_row(out)


def _permute_prompt_menu(prompt: str, order: Sequence[int]) -> str:
    lines = prompt.splitlines()
    try:
        start = lines.index("Choose your action:") + 1
    except ValueError as exc:
        raise ValueError("prompt is missing a numbered action menu") from exc
    end = start
    options: list[str] = []
    while end < len(lines):
        match = _OPTION_LINE_RE.fullmatch(lines[end])
        if match is None:
            break
        options.append(match.group(2))
        end += 1
    if len(options) != len(order):
        raise ValueError("prompt menu length does not match the stored menu")
    rewritten = [f"{index}: {options[source]}" for index, source in enumerate(order)]
    return "\n".join([*lines[:start], *rewritten, *lines[end:]])


def clustered_mean_ci(
    values_by_cluster: Mapping[str, Sequence[float]],
    *,
    n_boot: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
) -> dict[str, float] | None:
    """Bootstrap CI of the mean, resampling whole ``game_id`` clusters."""
    clusters = [list(map(float, group)) for group in values_by_cluster.values() if group]
    if not clusters:
        return None
    all_values = [value for group in clusters for value in group]
    mean = sum(all_values) / len(all_values)
    if len(clusters) == 1:
        return {
            "mean": mean,
            "low": mean,
            "high": mean,
            "n": float(len(all_values)),
            "clusters": 1.0,
        }
    rng = random.Random(seed)
    means: list[float] = []
    count = len(clusters)
    for _ in range(n_boot):
        sample = [clusters[rng.randrange(count)] for _ in range(count)]
        flat = [value for group in sample for value in group]
        means.append(sum(flat) / len(flat))
    means.sort()
    lower = int(alpha / 2 * n_boot)
    upper = min(n_boot - 1, int((1 - alpha / 2) * n_boot))
    return {
        "mean": mean,
        "low": means[lower],
        "high": means[upper],
        "n": float(len(all_values)),
        "clusters": float(count),
    }
