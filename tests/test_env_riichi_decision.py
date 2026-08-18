from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import statistics
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
for extra in (ROOT, ROOT / "environments" / "riichi_decision_v1"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

pytest.importorskip("verifiers")

from riichi_decision_v1.taskset import (  # noqa: E402
    RiichiDecisionConfig,
    RiichiDecisionTaskset,
    _extract_choice,
    load_bank,
)

from jongbench import positions, prompts  # noqa: E402


def _position(**overrides) -> dict:
    data = {
        "player_id": 2,
        "events": [
            {
                "type": "start_kyoku",
                "bakaze": "E",
                "kyoku": 1,
                "honba": 0,
                "kyotaku": 0,
                "oya": 0,
                "dora_marker": "3p",
                "scores": [25000, 25000, 25000, 25000],
                "tehais": [
                    ["?"] * 13,
                    ["?"] * 13,
                    [
                        "1m",
                        "2m",
                        "3m",
                        "4p",
                        "5p",
                        "6p",
                        "7s",
                        "8s",
                        "9s",
                        "E",
                        "E",
                        "P",
                        "P",
                    ],
                    ["?"] * 13,
                ],
            },
            {"type": "tsumo", "actor": 2, "pai": "9m"},
        ],
        "menu": [
            "riichi",
            "discard 1m",
            "discard 2m",
            "discard 3m",
            "discard 9m (drawn)",
            "discard 4p",
            "discard 5p",
            "discard 6p",
            "discard 7s",
            "discard 8s",
            "discard 9s",
            "discard E",
            "discard P",
        ],
        "rewards": [
            0.1,
            0.25,
            0.2,
            0.15,
            1.0,
            0.3,
            0.35,
            0.4,
            0.45,
            0.5,
            0.55,
            0.05,
            0.0,
        ],
        "best_index": 4,
        "kyoku": 0,
        "honba": 0,
        "junme": 1,
        "tiles_left": 69,
        "shanten": 2,
        "at_furiten": False,
        "metadata": {},
    }
    data.update(overrides)
    return data


def _task_row(**overrides) -> dict:
    row = positions.Position(**_position()).to_task_dict()
    row.update(overrides)
    if "id" not in overrides:
        row["id"] = positions.position_id(row)
    return row


def _manifest(**overrides) -> dict:
    manifest = positions.bank_manifest(
        reviewer_checkpoint="test-mortal.pth",
        reviewer_checkpoint_sha256="a" * 64,
        reviewer_temperature=0.1,
        source={"kind": "test_fixture", "description": "unit-test position"},
        generator_version="test",
    )
    manifest.update(overrides)
    return manifest


def _write_raw_bank(
    path: Path, rows: list[object], manifest: object | None = None
) -> None:
    records = [_manifest() if manifest is None else manifest, *rows]
    path.write_text("".join(json.dumps(record) + "\n" for record in records))


def _write_bank(path: Path, rows: list[dict]) -> None:
    with path.open("w") as handle:
        positions.dump_bank(handle, _manifest(), rows)


@pytest.fixture
def taskset(tmp_path) -> RiichiDecisionTaskset:
    bank = tmp_path / "bank.jsonl"
    _write_bank(bank, [_task_row(), _task_row(prompt="second-position prompt")])
    return RiichiDecisionTaskset(RiichiDecisionConfig(bank=str(bank)))


def _trace(reply: str) -> SimpleNamespace:
    return SimpleNamespace(last_reply=reply, num_turns=1)


def test_taskset_renders_a_prompt_and_carries_the_grading(taskset) -> None:
    tasks = list(taskset.load())
    assert len(tasks) == 2
    task = tasks[0]
    assert "Choose your action:" in (task.data.prompt or "")
    assert task.data.menu == _task_row()["menu"]
    assert task.data.rewards == _task_row()["rewards"]
    assert task.data.info["seat"] == 2


def test_reward_is_mortals_opinion_of_the_chosen_action(taskset) -> None:
    task = next(iter(taskset.load()))
    for reply, expected in (
        ('{"choice": 4}', 1.0),
        ('{"choice": 1}', 0.25),
        ('{"choice": 12}', 0.0),
    ):
        assert asyncio.run(task.q_advantage(_trace(reply))) == pytest.approx(expected)


def test_an_unusable_reply_scores_zero_rather_than_erroring(taskset) -> None:
    task = next(iter(taskset.load()))
    for reply in ("I would discard the 9m", '{"choice": 17}', "", '{"choice": "1"}'):
        assert asyncio.run(task.q_advantage(_trace(reply))) == 0.0
        assert asyncio.run(task.answered(_trace(reply))) == 0.0


def test_match_metric_tracks_mortals_own_choice(taskset) -> None:
    task = next(iter(taskset.load()))
    assert asyncio.run(task.matched_mortal(_trace('{"choice": 4}'))) == 1.0
    assert asyncio.run(task.matched_mortal(_trace('{"choice": 1}'))) == 0.0


def test_missing_bank_names_the_command_that_builds_one(tmp_path) -> None:
    taskset = RiichiDecisionTaskset(
        RiichiDecisionConfig(bank=str(tmp_path / "nope.jsonl"))
    )
    with pytest.raises(FileNotFoundError, match="jongbench positions"):
        list(taskset.load())


def test_shipped_sample_bank_runs_out_of_the_box_and_locks_baselines() -> None:
    """The committed sample is a reproducible reference set, not just 128 valid rows."""
    from riichi_decision_v1.taskset import SAMPLE_BANK

    manifest, rows = load_bank(SAMPLE_BANK)
    assert manifest["generator"] == {"name": "jongbench", "version": "0.1.1"}
    assert manifest["reviewer"]["checkpoint_sha256"] == (
        "bfb3a6c072aa0bfd4171a9cdc77cb6c02ae42cde920843f9e5784394f23447d8"
    )
    assert manifest["source"]["kind"] == "mortal_self_play"
    assert len(rows) == 128
    assert len({row["id"] for row in rows}) == 128
    assert len({row["name"] for row in rows}) == 128
    identity_digest = hashlib.sha256(
        "\n".join(row["id"] for row in rows).encode()
    ).hexdigest()
    assert (
        identity_digest
        == "386658a2b0f288a9d7d1cf9fba57679a6285e05e2f0d573ee1c9ece435f51263"
    )
    assert statistics.mean(len(row["menu"]) for row in rows) == pytest.approx(8.7734375)
    assert statistics.mean(row["rewards"][0] for row in rows) == pytest.approx(
        0.3412620549357386
    )
    assert statistics.mean(
        sum(row["rewards"]) / len(row["rewards"]) for row in rows
    ) == pytest.approx(0.3673449002786986)
    assert statistics.mean(1 / len(row["menu"]) for row in rows) == pytest.approx(
        0.18948991546647798
    )

    taskset = RiichiDecisionTaskset(RiichiDecisionConfig())
    assert Path(taskset.config.bank) == SAMPLE_BANK
    tasks = list(taskset.load())
    assert len(tasks) == 128
    assert [task.data.position_id for task in tasks] == [row["id"] for row in rows]


def test_package_bundles_the_chat_harness_as_its_default() -> None:
    """`eval riichi_decision_v1` must run the plain chat loop, not a bash agent:
    verifiers takes a taskset package's exported Harness subclass as the default."""
    from verifiers.v1.harnesses.null import NullHarness
    from verifiers.v1.utils.loaders import default_harness_id, harness_class

    assert default_harness_id("riichi_decision_v1") == "riichi_decision_v1"
    assert harness_class("riichi_decision_v1") is NullHarness


def test_a_real_bank_round_trips_through_the_taskset(tmp_path) -> None:
    """Evaluation consumes only the frozen v2 records, not raw Position objects."""
    row = positions.Position(**_position()).to_task_dict()
    bank = tmp_path / "bank.jsonl"
    _write_bank(bank, [row])
    manifest, loaded_rows = load_bank(bank)
    task = next(
        iter(RiichiDecisionTaskset(RiichiDecisionConfig(bank=str(bank))).load())
    )
    assert manifest["record_type"] == "manifest"
    assert loaded_rows == [row]
    assert task.data.position_id == row["id"]
    assert task.data.prompt == row["prompt"]
    assert task.data.system_prompt == row["system_prompt"]


def test_state_hints_select_the_frozen_prompt_variant(tmp_path) -> None:
    row = _task_row(prompt="with hints", prompt_without_state_hints="without hints")
    bank = tmp_path / "bank.jsonl"
    _write_bank(bank, [row])
    taskset = RiichiDecisionTaskset(
        RiichiDecisionConfig(bank=str(bank), state_hints=False)
    )
    assert next(iter(taskset.load())).data.prompt == "without hints"


def test_choice_parser_matches_the_game_parser_contract() -> None:
    """The standalone copy may not import jongbench, so this corpus catches drift."""
    cases: tuple[tuple[str, int, int | None], ...] = (
        ('{"choice": 2}', 3, 2),
        ('reasoning\n```json\n{"choice": 0}\n```', 3, 0),
        ('not json {oops} then {"choice": 1}', 3, 1),
        ('[{"choice": 2}]', 3, 2),
        ('{"wrapper":{"choice":1}}', 3, 1),
        ('{"choice": 1}\n{"choice": 1}', 3, 1),
        ("work\n  2  ", 3, 2),
        ("work\n2\nthen prose", 3, None),
        ('{"choice": 1}\n{"choice": 2}', 3, None),
        ('{"choice": true}', 3, None),
        ('{"choice": 1.0}', 3, None),
        ('{"choice": "1"}', 3, None),
        ('{"choice": -1}', 3, None),
        ('{"choice": 3}', 3, None),
        ("-1", 3, None),
        ("", 3, None),
        ('{"choice": 0}', 0, None),
    )
    for text, n_options, expected in cases:
        for parser in (prompts.extract_choice, _extract_choice):
            if expected is None:
                with pytest.raises(ValueError):
                    parser(text, n_options)
            else:
                assert parser(text, n_options) == expected


def test_bank_requires_a_manifest_as_the_literal_first_line(tmp_path) -> None:
    bank = tmp_path / "bank.jsonl"
    for contents, exception, message in (
        ("", ValueError, "bank is empty"),
        ("\n" + json.dumps(_manifest()) + "\n", ValueError, "first line"),
        (json.dumps(_task_row()) + "\n", ValueError, "bank manifest"),
        (json.dumps([]) + "\n", TypeError, "JSON object"),
    ):
        bank.write_text(contents)
        with pytest.raises(exception, match=message):
            load_bank(bank)


def test_manifest_provenance_and_reward_contract_are_validated(tmp_path) -> None:
    bank = tmp_path / "bank.jsonl"
    cases = []
    manifest = _manifest()
    manifest.pop("source")
    cases.append((manifest, ValueError, "missing required field.*source"))
    manifest = _manifest(schema_version=1)
    cases.append((manifest, ValueError, "unsupported schema_version"))
    manifest = _manifest()
    manifest["reward"] = {**manifest["reward"], "name": "other"}
    cases.append((manifest, ValueError, "unsupported reward semantics"))
    manifest = _manifest()
    manifest["reviewer"] = {**manifest["reviewer"], "checkpoint_sha256": "unknown"}
    cases.append((manifest, ValueError, "SHA-256 digest"))
    manifest = _manifest()
    manifest["source"] = {"kind": ""}
    cases.append((manifest, TypeError, "source.kind"))

    for invalid, exception, message in cases:
        _write_raw_bank(bank, [_task_row()], invalid)
        with pytest.raises(exception, match=message):
            load_bank(bank)


def test_bank_rows_distinguish_invalid_structure_types_and_grading(tmp_path) -> None:
    bank = tmp_path / "bank.jsonl"
    cases: list[tuple[str, object, type[Exception], str]] = []

    def add(label: str, mutate, exception: type[Exception], message: str) -> None:
        row = _task_row()
        value = mutate(row)
        cases.append((label, row if value is None else value, exception, message))

    add("not object", lambda row: [], TypeError, "JSON object")
    add(
        "missing prompt",
        lambda row: row.pop("prompt") and None,
        ValueError,
        "missing required field.*prompt",
    )
    add(
        "unknown",
        lambda row: row.update(extra=True),
        ValueError,
        "unknown field.*extra",
    )
    add(
        "old schema",
        lambda row: row.update(schema_version=1),
        ValueError,
        "unsupported schema_version",
    )
    add(
        "prompt type",
        lambda row: row.update(prompt=3),
        TypeError,
        r"prompt must be a non-empty string",
    )
    add(
        "menu type",
        lambda row: row.update(menu="discard"),
        TypeError,
        r"menu must be a list",
    )
    add("short menu", lambda row: row.update(menu=["only"]), ValueError, "at least two")
    add("menu item", lambda row: row.update(menu=["one", 2]), TypeError, r"menu\[1\]")
    add(
        "duplicate menu",
        lambda row: row.update(menu=["same", "same"]),
        ValueError,
        "duplicate options",
    )
    add(
        "length",
        lambda row: row.update(rewards=[1.0]),
        ValueError,
        "mismatched menu and rewards",
    )
    add(
        "reward bool",
        lambda row: row["rewards"].__setitem__(0, True),
        TypeError,
        r"rewards\[0\].*number",
    )
    add(
        "reward nan",
        lambda row: row["rewards"].__setitem__(0, float("nan")),
        ValueError,
        "finite",
    )
    add(
        "reward low",
        lambda row: row["rewards"].__setitem__(0, -0.1),
        ValueError,
        r"in \[0, 1\]",
    )
    add(
        "best bool",
        lambda row: row.update(best_index=True),
        TypeError,
        "best_index.*integer",
    )
    add("best range", lambda row: row.update(best_index=99), ValueError, "out of range")
    add("best not argmax", lambda row: row.update(best_index=0), ValueError, "argmax")
    add(
        "info type",
        lambda row: row.update(info=[]),
        TypeError,
        "info must be a JSON object",
    )
    add(
        "info missing",
        lambda row: row["info"].pop("seat") and None,
        ValueError,
        "missing required field.*seat",
    )
    add(
        "seat bool",
        lambda row: row["info"].update(seat=True),
        TypeError,
        "seat must be an integer",
    )
    add(
        "seat range",
        lambda row: row["info"].update(seat=4),
        ValueError,
        r"seat must be in \[0, 3\]",
    )
    add(
        "furiten",
        lambda row: row["info"].update(at_furiten=0),
        TypeError,
        "at_furiten must be a boolean",
    )
    add(
        "identity",
        lambda row: row.update(id="sha256:" + "0" * 64),
        ValueError,
        "canonical content hash",
    )

    for label, invalid, exception, message in cases:
        _write_raw_bank(bank, [invalid])
        try:
            with pytest.raises(exception, match=message):
                load_bank(bank)
        except AssertionError as exc:
            raise AssertionError(f"invalid bank case failed: {label}") from exc


def test_duplicate_stable_position_ids_are_rejected(tmp_path) -> None:
    bank = tmp_path / "bank.jsonl"
    row = _task_row()
    _write_raw_bank(bank, [row, copy.deepcopy(row)])
    with pytest.raises(ValueError, match="duplicates an earlier row"):
        load_bank(bank)
