from __future__ import annotations

import json
from pathlib import Path

from jongbench.improve import ImproveConfig, improve_policy
from jongbench.selfplay import DuelResult


def _result(avg_pt: float, standard_error: float) -> DuelResult:
    return DuelResult(
        rankings=[2, 2, 2, 2],
        rank_sequence=[0, 1, 2, 3, 0, 1, 2, 3],
        seed_avg_pts=[avg_pt - standard_error, avg_pt + standard_error],
        games=8,
        avg_rank=2.5,
        avg_pt=avg_pt,
        standard_error=standard_error,
    )


def test_improve_policy_promotes_only_candidate_above_bound(
    tmp_path: Path, monkeypatch
) -> None:
    initial = tmp_path / "initial.pth"
    control = tmp_path / "control.pth"
    initial.write_bytes(b"initial")
    control.write_bytes(b"control")
    duel_results = iter(
        [
            _result(0.0, 1.0),
            _result(2.0, 1.0),
            _result(0.0, 1.0),
            _result(0.5, 1.0),
            _result(1.5, 0.5),
            _result(3.0, 0.5),
        ]
    )
    duel_calls = []

    def fake_train(config):
        Path(config.out).write_bytes(Path(config.init).read_bytes() + b"+candidate")
        assert config.duplicate_challenger_only
        return {"updates": float(config.steps)}

    def fake_duel(**kwargs):
        duel_calls.append(kwargs)
        return next(duel_results)

    monkeypatch.setattr("jongbench.improve.train_policy_rl", fake_train)
    monkeypatch.setattr("jongbench.improve.duel", fake_duel)
    out = tmp_path / "league"
    manifest = improve_policy(
        ImproveConfig(
            init=str(initial),
            control=str(control),
            out_dir=str(out),
            rounds=2,
            rollout_games=4,
            updates=2,
            batch_size=8,
            duel_games=8,
            device="cpu",
            promotion_z=1.0,
        )
    )

    assert manifest["promotions"] == 1
    assert [item["promoted"] for item in manifest["rounds"]] == [True, False]
    assert duel_calls[0]["challenger_boltzmann_epsilon"] == 1.0
    assert "challenger_boltzmann_epsilon" not in duel_calls[1]
    assert (out / "champion.pth").read_bytes() == b"initial+candidate"
    persisted = json.loads((out / "league.json").read_text(encoding="utf-8"))
    assert persisted["final_evaluations"].keys() == {
        "initial_policy",
        "control_q",
    }
