from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from typing import Any

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import jongbench  # noqa: F401
import libriichi
from jongbench.cli import _build_parser
from jongbench.dataset import discover_logs, iter_gameplay_samples
from jongbench.evaluate import (
    aggregates,
    load_checkpoint,
    load_engine,
    load_mjai_log,
    networks_from_checkpoint,
    review_player,
)
from jongbench.mortal_model import ACTION_SPACE, DQN, ConfidenceHead, PolicyHead
from jongbench.train import PolicyRLConfig, TrainConfig, train, train_policy_rl

WEIGHTS = ROOT / "weights" / "mortal.pth"


class TsumogiriEngine:
    engine_type = "mjai-log"

    def __init__(self, name: str) -> None:
        self.name = name
        self.player_ids: list[int] | None = None

    def set_player_ids(self, player_ids: list[int]) -> None:
        self.player_ids = player_ids

    def react_batch(self, game_states: list[Any]) -> list[str]:
        assert self.player_ids is not None
        reactions = []
        for game_state in game_states:
            player_id = self.player_ids[game_state.game_index]
            state = game_state.state
            if state.last_cans.can_discard:
                tile = state.last_self_tsumo()
                reactions.append(
                    json.dumps(
                        {
                            "type": "dahai",
                            "actor": player_id,
                            "pai": tile,
                            "tsumogiri": True,
                        },
                        separators=(",", ":"),
                    )
                )
            else:
                reactions.append('{"type":"none"}')
        return reactions

    def start_game(self, game_idx: int) -> None:
        pass

    def end_kyoku(self, game_idx: int) -> None:
        pass

    def end_game(self, game_idx: int, scores: list[int]) -> None:
        pass


def _write_tsumogiri_log(directory: Path, seed: int = 9100) -> Path:
    arena = libriichi.arena.FourEngines(
        disable_progress_bar=True, log_dir=str(directory)
    )
    engines = [TsumogiriEngine(f"tsumogiri-{idx}") for idx in range(4)]
    arena.py_4p(engines, (seed, 5), 1)
    logs = sorted(directory.glob("*.json.gz"))
    assert logs
    return logs[0]


def test_policy_head_masks_illegal_actions() -> None:
    head = PolicyHead()
    phi = torch.randn(3, 1024)
    mask = torch.zeros(3, ACTION_SPACE, dtype=torch.bool)
    mask[0, 0] = True
    mask[1, 5] = True
    mask[1, 7] = True
    mask[2, 45] = True
    logits = head(phi, mask)
    assert torch.isneginf(logits[~mask]).all()
    assert torch.isfinite(logits[mask]).all()
    chosen = logits.argmax(-1)
    assert chosen[0].item() == 0
    assert chosen[2].item() == 45
    assert chosen[1].item() in {5, 7}


def test_confidence_head_range() -> None:
    head = ConfidenceHead()
    out = head(torch.randn(4, 1024))
    assert out.shape == (4,)
    assert torch.all(out > 0) and torch.all(out < 1)


def test_policy_head_can_exactly_clone_v4_q_distribution() -> None:
    dqn = DQN(version=4)
    policy = PolicyHead.from_dqn(dqn, temperature=0.2)
    phi = torch.randn(4, 1024)
    mask = torch.rand(4, ACTION_SPACE) > 0.5
    mask[:, 45] = True
    expected = torch.softmax(dqn(phi, mask) / 0.2, dim=-1)
    actual = torch.softmax(policy(phi, mask), dim=-1)
    torch.testing.assert_close(actual, expected)


def test_cli_new_commands_parse() -> None:
    parser = _build_parser()
    selfplay = parser.parse_args(["selfplay", "--games", "2", "--out", "training/x"])
    assert selfplay.games == 2
    train_args = parser.parse_args(["train", "--steps", "3", "--unfreeze-encoder"])
    assert train_args.steps == 3
    assert train_args.unfreeze_encoder
    policy_rl = parser.parse_args(
        [
            "policy-rl",
            "--logs",
            "training/selfplay",
            "--init",
            "weights/reviewer.pth",
            "--out",
            "weights/candidate.pth",
        ]
    )
    assert policy_rl.clip_ratio == 0.2
    improve = parser.parse_args(
        [
            "improve",
            "--init",
            "weights/reviewer.pth",
            "--out",
            "training/league",
            "--no-control",
        ]
    )
    assert improve.rounds == 4
    assert improve.control is None
    duel_args = parser.parse_args(
        ["duel", "--challenger", "weights/reviewer.pth", "--games", "8"]
    )
    assert duel_args.games == 8


def test_gameplay_dataset_from_tsumogiri_log() -> None:
    with tempfile.TemporaryDirectory() as tempdir:
        log = _write_tsumogiri_log(Path(tempdir))
        files = discover_logs(tempdir)
        assert files == [str(log)]
        samples = list(
            iter_gameplay_samples(files, shuffle_files=False, shuffle_buffer=False)
        )
        assert len(samples) > 50
        obs, action, mask, steps, reward, rank = samples[0]
        assert obs.shape[0] == 1012 and obs.shape[1] == 34
        assert mask.shape == (46,)
        assert mask[action]
        assert 0 <= action < 46
        assert steps >= 0
        assert rank in {0, 1, 2, 3}
        assert reward in {0.0, 2.0, 4.0, 6.0}
        duplicate = Path(tempdir) / "9100_1_a.json.gz"
        duplicate.write_bytes(log.read_bytes())
        challenger_samples = list(
            iter_gameplay_samples(
                [str(duplicate)],
                shuffle_files=False,
                shuffle_buffer=False,
                duplicate_challenger_only=True,
            )
        )
        assert 0 < len(challenger_samples) < len(samples)


def test_discover_logs_includes_plain_json(tmp_path: Path) -> None:
    log = tmp_path / "game.json"
    log.write_text("[]", encoding="utf-8")
    assert discover_logs(tmp_path) == [str(log)]


@pytest.mark.skipif(not WEIGHTS.exists(), reason="mortal.pth is not present")
def test_load_engine_accepts_device_and_missing_policy() -> None:
    engine = load_engine(str(WEIGHTS), device="cpu", use_policy=True)
    assert engine.policy is None
    assert engine.use_policy is False
    assert engine.device.type == "cpu"


@pytest.mark.skipif(not WEIGHTS.exists(), reason="mortal.pth is not present")
def test_train_two_steps_on_tsumogiri_log(tmp_path: Path) -> None:
    log = _write_tsumogiri_log(tmp_path)
    out = tmp_path / "reviewer.pth"
    stats = train(
        TrainConfig(
            logs=str(tmp_path),
            init=str(WEIGHTS),
            out=str(out),
            steps=2,
            batch_size=8,
            device="cpu",
            log_every=1,
            save_every=2,
            file_batch_size=1,
        )
    )
    assert out.exists()
    assert "policy_ce" in stats
    assert all(
        torch.isfinite(torch.tensor(value))
        for value in stats.values()
        if isinstance(value, float)
    )
    ckpt = torch.load(out, weights_only=True, map_location="cpu")
    assert "policy" in ckpt
    init_ckpt = load_checkpoint(str(WEIGHTS))
    _, initial_dqn, _, _, _, _ = networks_from_checkpoint(init_ckpt)
    initial_policy = PolicyHead.from_dqn(initial_dqn, temperature=0.1)
    assert any(
        not torch.equal(ckpt["policy"][key], value)
        for key, value in initial_policy.state_dict().items()
    )
    assert "mortal" in ckpt
    engine = load_engine(str(out), device="cpu", use_policy=True)
    assert engine.policy is not None
    assert engine.use_policy is True
    engine.use_policy = False
    review = review_player(load_mjai_log(str(log)), 0, engine)
    assert review["entries"]
    entry = review["entries"][0]
    assert sum(
        candidate["policy_prob"] for candidate in entry["details"]
    ) == pytest.approx(1.0, abs=1e-5)
    assert sum(entry["next_rank_probs"]) == pytest.approx(1.0, abs=1e-5)
    assert 0.0 <= entry["policy_confidence"] <= 1.0
    stats = aggregates(review)
    assert stats["policy_count"] == review["total_reviewed"]
    assert stats["policy_match_rate"] is not None
    rl_out = tmp_path / "reviewer-rl.pth"
    rl_stats = train_policy_rl(
        PolicyRLConfig(
            logs=str(tmp_path),
            init=str(out),
            anchor=str(out),
            out=str(rl_out),
            steps=2,
            batch_size=8,
            device="cpu",
            target_kl=None,
            log_every=1,
            file_batch_size=1,
        )
    )
    assert rl_stats["updates"] == 2
    rl_ckpt = torch.load(rl_out, weights_only=True, map_location="cpu")
    for key, value in ckpt["mortal"].items():
        torch.testing.assert_close(rl_ckpt["mortal"][key], value)
    for key, value in ckpt["current_dqn"].items():
        torch.testing.assert_close(rl_ckpt["current_dqn"][key], value)
    assert rl_ckpt["config"]["reviewer"]["policy_rl"]["total_updates"] == 2
