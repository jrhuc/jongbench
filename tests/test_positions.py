from __future__ import annotations

import gzip
import json
import sys
import tempfile
from io import StringIO
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jongbench import actions, arena, engines, evaluate, positions
from jongbench.mortal_model import DQN, Brain


def _first_kyoku() -> list[dict]:
    seats = [engines.RandomEngine(f"r{i}", seed=i) for i in range(4)]
    with tempfile.TemporaryDirectory() as tempdir:
        arena.run_games(seats, 1, seed_start=(2024, 1), log_dir=tempdir)
        log = sorted(Path(tempdir).glob("*.json.gz"))[0]
        events = [json.loads(line) for line in gzip.open(log, "rt")]
    end = next(i for i, e in enumerate(events) if e.get("type") == "end_kyoku")
    return events[: end + 1]


@pytest.fixture(scope="module")
def extracted(tmp_path_factory) -> list[positions.Position]:
    torch.manual_seed(7)
    checkpoint = tmp_path_factory.mktemp("positions") / "reviewer.pth"
    brain = Brain(version=4, num_blocks=0, conv_channels=4)
    dqn = DQN(version=4)
    torch.save(
        {
            "config": {
                "control": {"version": 4},
                "resnet": {"num_blocks": 0, "conv_channels": 4},
            },
            "mortal": brain.state_dict(),
            "current_dqn": dqn.state_dict(),
        },
        checkpoint,
    )
    engine = evaluate.load_engine(checkpoint, use_policy=False)
    return positions.extract_positions(_first_kyoku(), engine, seats=(0,))


def test_positions_are_graded_on_a_normalised_scale(extracted) -> None:
    assert extracted
    for position in extracted:
        assert len(position.rewards) == len(position.menu) >= 2
        assert min(position.rewards) == pytest.approx(0.0)
        assert max(position.rewards) == pytest.approx(1.0)
        assert all(0.0 <= r <= 1.0 for r in position.rewards)
        assert position.rewards[position.best_index] == pytest.approx(1.0)


def test_positions_are_self_contained(extracted) -> None:
    """A stored position must rebuild to the same choice set, or a taskset scored
    against `rewards` would be grading a different board than it renders."""
    for position in extracted:
        rebuilt = [str(item["label"]) for item in actions.build_menu(position.state())]
        assert rebuilt == position.menu


def test_positions_keep_the_seat_point_of_view(extracted) -> None:
    for position in extracted:
        for event in position.events:
            if event.get("type") == "start_kyoku":
                for seat, hand in enumerate(event["tehais"]):
                    if seat != position.player_id:
                        assert hand == ["?"] * 13
            if (
                event.get("type") == "tsumo"
                and event.get("actor") != position.player_id
            ):
                assert event.get("pai") == "?"


def test_positions_render_as_typed_versioned_bank_rows(extracted) -> None:
    position = extracted[len(extracted) // 2]
    text = position.prompt()
    assert "Choose your action:" in text
    assert text.count("\n") > 5

    task = position.to_task_dict()
    assert task["record_type"] == "position"
    assert task["schema_version"] == positions.BANK_SCHEMA_VERSION == 3
    assert len(task["q_values"]) == len(task["menu"])
    assert task["board_id"].startswith("sha256:")
    assert task["prompt_id"].startswith("sha256:")
    assert isinstance(task["tags"], list)
    assert task["id"] == positions.position_id(task)
    assert task["id"].startswith("sha256:")
    assert task["name"].endswith(task["id"].removeprefix("sha256:")[:12])
    assert task["prompt"] == text
    assert task["prompt_without_state_hints"] == position.prompt(state_hints=False)
    assert task["menu"] == position.menu
    assert task["rewards"] == position.rewards
    assert positions.validate_bank_row(task) is task


def test_position_identity_includes_best_index_when_argmax_is_tied(extracted) -> None:
    first = extracted[0].to_task_dict()
    first["rewards"] = [1.0, 1.0, *([0.0] * (len(first["menu"]) - 2))]
    first["q_values"] = [10.0, 10.0, *([0.0] * (len(first["menu"]) - 2))]
    first["best_index"] = 0
    first["id"] = positions.position_id(first)
    positions.validate_bank_row(first)

    second = json.loads(json.dumps(first))
    second["best_index"] = 1
    second["id"] = positions.position_id(second)
    positions.validate_bank_row(second)
    assert first["id"] != second["id"]


def test_raw_position_serialization_is_not_a_second_bank_contract() -> None:
    assert not hasattr(positions.Position, "to_dict")
    assert not hasattr(positions.Position, "from_dict")


def test_bank_dump_puts_a_provenance_manifest_first_and_rejects_duplicates(
    extracted,
) -> None:
    rows = [position.to_task_dict() for position in extracted[:2]]
    manifest = positions.bank_manifest(
        reviewer_checkpoint="Mortal test checkpoint",
        reviewer_checkpoint_sha256="a" * 64,
        reviewer_temperature=0.1,
        source={
            "kind": "test_log",
            "description": "deterministic test fixture",
            "artifacts": [{"name": "game.json", "sha256": "b" * 64}],
        },
        generator_version="test",
    )
    output = StringIO()
    assert positions.dump_bank(output, manifest, rows) == 2
    records = [json.loads(line) for line in output.getvalue().splitlines()]
    assert records[0] == manifest
    assert [record["id"] for record in records[1:]] == [row["id"] for row in rows]

    with pytest.raises(ValueError, match="duplicates an earlier row"):
        positions.dump_bank(StringIO(), manifest, [rows[0], rows[0]])
