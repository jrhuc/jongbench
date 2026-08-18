from __future__ import annotations

from pathlib import Path

from jongbench import bank_schema


ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "jongbench" / "bank_schema.py"
ENV = (
    ROOT
    / "environments"
    / "riichi_decision_v1"
    / "riichi_decision_v1"
    / "bank_schema.py"
)


def test_decision_env_ships_a_byte_identical_schema_copy() -> None:
    assert CORE.read_bytes() == ENV.read_bytes()


def test_identity_splits_board_prompt_and_grading() -> None:
    events = [{"type": "start_kyoku", "tehais": [["1m"] * 13] * 4}]
    board = bank_schema.board_id(seat=0, events=events)
    other_board = bank_schema.board_id(seat=1, events=events)
    assert board != other_board
    row = {
        "prompt": "Choose your action:\n0: a\n1: b",
        "prompt_without_state_hints": "Choose your action:\n0: a\n1: b",
        "system_prompt": "sys",
        "menu": ["a", "b"],
        "rewards": [1.0, 0.0],
        "q_values": [2.0, 1.0],
        "best_index": 0,
        "board_id": board,
    }
    row["prompt_id"] = bank_schema.prompt_id(row)
    row["id"] = bank_schema.position_id(row)
    swapped = dict(row)
    swapped["prompt"] = "Choose your action:\n0: b\n1: a"
    swapped["prompt_without_state_hints"] = swapped["prompt"]
    swapped["menu"] = ["b", "a"]
    swapped["rewards"] = [0.0, 1.0]
    swapped["q_values"] = [1.0, 2.0]
    swapped["best_index"] = 1
    swapped["prompt_id"] = bank_schema.prompt_id(swapped)
    swapped["id"] = bank_schema.position_id(swapped)
    assert swapped["board_id"] == row["board_id"]
    assert swapped["prompt_id"] != row["prompt_id"]
    assert swapped["id"] != row["id"]


def _row() -> dict:
    row = {
        "record_type": "position",
        "schema_version": 3,
        "prompt": "board\n\nChoose your action:\n0: discard 1m\n1: discard 2m\n",
        "prompt_without_state_hints": (
            "board\n\nChoose your action:\n0: discard 1m\n1: discard 2m\n"
        ),
        "system_prompt": "sys",
        "menu": ["discard 1m", "discard 2m"],
        "rewards": [0.0, 1.0],
        "q_values": [0.0, 1.0],
        "best_index": 1,
        "game_id": "game",
        "tags": [],
        "info": {
            "seat": 0,
            "kyoku": 0,
            "honba": 0,
            "junme": 1,
            "tiles_left": 69,
            "shanten": 2,
            "at_furiten": False,
        },
        "board_id": "sha256:" + "a" * 64,
        "name": "test",
    }
    row["prompt_id"] = bank_schema.prompt_id(row)
    row["id"] = bank_schema.position_id(row)
    return bank_schema.validate_bank_row(row)


def test_permute_row_rewrites_menu_and_identity() -> None:
    row = _row()
    permuted = bank_schema.permute_row(row, [1, 0])
    assert permuted["menu"] == [row["menu"][1], row["menu"][0]]
    assert permuted["best_index"] == 0
    assert permuted["id"] != row["id"]
    assert permuted["board_id"] == row["board_id"]


def test_sample_rows_caps_per_game_then_fills() -> None:
    rows = []
    for game in ("g0", "g1"):
        for index in range(3):
            rows.append(
                {
                    "game_id": game,
                    "name": f"{game}-{index}",
                    "id": f"sha256:{index:064d}",
                }
            )
    sampled = bank_schema.sample_rows(rows, 3, seed=1, max_per_game=1)
    assert len(sampled) == 3
    assert {row["game_id"] for row in sampled} == {"g0", "g1"}


def test_clustered_mean_ci_resamples_whole_games() -> None:
    ci = bank_schema.clustered_mean_ci(
        {"g0": [1.0, 1.0], "g1": [0.0, 0.0]},
        n_boot=200,
        seed=0,
    )
    assert ci is not None
    assert ci["mean"] == 0.5
    assert ci["clusters"] == 2.0
    assert ci["low"] <= ci["mean"] <= ci["high"]
