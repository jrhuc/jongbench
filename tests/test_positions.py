from __future__ import annotations

import gzip
import json
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jongbench import actions, arena, engines, evaluate, positions

WEIGHTS = ROOT / "weights" / "mortal.pth"
pytestmark = pytest.mark.skipif(
    not WEIGHTS.exists(), reason="needs weights/mortal.pth"
)


def _first_kyoku() -> list[dict]:
    seats = [engines.RandomEngine(f"r{i}", seed=i) for i in range(4)]
    with tempfile.TemporaryDirectory() as tempdir:
        arena.run_games(seats, 1, seed_start=(2024, 1), log_dir=tempdir)
        log = sorted(Path(tempdir).glob("*.json.gz"))[0]
        events = [json.loads(line) for line in gzip.open(log, "rt")]
    end = next(i for i, e in enumerate(events) if e.get("type") == "end_kyoku")
    return events[: end + 1]


@pytest.fixture(scope="module")
def extracted() -> list[positions.Position]:
    engine = evaluate.load_engine(str(WEIGHTS))
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
            if event.get("type") == "tsumo" and event.get("actor") != position.player_id:
                assert event.get("pai") == "?"


def test_positions_render_and_round_trip(extracted) -> None:
    position = extracted[len(extracted) // 2]
    text = position.prompt()
    assert "Choose your action:" in text
    assert text.count("\n") > 5

    restored = positions.Position.from_dict(json.loads(json.dumps(position.to_dict())))
    assert restored.menu == position.menu
    assert restored.rewards == position.rewards
    assert restored.prompt() == text
