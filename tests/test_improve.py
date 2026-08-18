from __future__ import annotations

import json
from pathlib import Path

from jongbench.improve import ImproveConfig, improve_policy
from jongbench.selfplay import DuelResult
from jongbench.weights import ResolvedCheckpoint


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
        Path(config.out).write_bytes(config.init.path.read_bytes() + b"+candidate")
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
    assert persisted["config"]["init"]["path"] == str(initial)
    assert persisted["config"]["init"]["use_policy"] is True
    assert persisted["config"]["control"]["path"] == str(control)
    assert persisted["config"]["control"]["use_policy"] is False
    assert persisted["initial"] == persisted["config"]["init"]
    assert persisted["champion"]["path"] == str(out / "champion.pth")
    assert persisted["rounds"][0]["source"]["path"] == str(initial)
    assert persisted["rounds"][0]["candidate"]["path"].endswith("candidate.pth")
    assert persisted["final_evaluations"].keys() == {
        "initial_policy",
        "control_q",
    }


def test_improve_policy_resolves_auto_control(tmp_path: Path, monkeypatch) -> None:
    initial = tmp_path / "initial.pth"
    control = tmp_path / "control.pth"
    initial.write_bytes(b"initial")
    control.write_bytes(b"control")
    duel_calls = []

    def fake_train(config):
        Path(config.out).write_bytes(b"candidate")
        return {"updates": float(config.steps)}

    def fake_duel(**kwargs):
        duel_calls.append(kwargs)
        return _result(0.0, 1.0)

    monkeypatch.setattr("jongbench.improve.train_policy_rl", fake_train)
    monkeypatch.setattr("jongbench.improve.duel", fake_duel)
    from jongbench import improve as improve_module

    resolve = improve_module.resolve_mortal_checkpoint

    def fake_resolve(value, *, use_policy=None):
        if value == "auto":
            return ResolvedCheckpoint(
                path=control,
                sha256="0" * 64,
                source="https://example.test/control.pth",
                use_policy=bool(use_policy),
            )
        return resolve(value, use_policy=use_policy)

    monkeypatch.setattr(improve_module, "resolve_mortal_checkpoint", fake_resolve)
    out = tmp_path / "league"
    manifest = improve_policy(
        ImproveConfig(
            init=str(initial),
            out_dir=str(out),
            rounds=1,
            rollout_games=4,
            updates=1,
            batch_size=8,
            duel_games=8,
            device="cpu",
        )
    )

    assert manifest["promotions"] == 0
    assert duel_calls[-1]["champion_weights"] == str(control)
    assert manifest["config"]["control"] == {
        "path": str(control),
        "sha256": "0" * 64,
        "source": "https://example.test/control.pth",
        "use_policy": False,
    }
    assert manifest["config"]["control"] != "auto"
