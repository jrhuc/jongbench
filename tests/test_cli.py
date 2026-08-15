from __future__ import annotations

import gzip
import hashlib
import json
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jongbench import cli
from jongbench.artifacts import decision_filename
from jongbench.mortal_model import DQN, Brain
from jongbench.run_artifacts import (
    record_gameplay_checkpoints,
    reviews_missing,
    write_run_config,
)


def _write_test_checkpoint(path: Path) -> Path:
    import torch

    torch.manual_seed(7)
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
        path,
    )
    return path


def test_cli_workflow() -> None:
    with tempfile.TemporaryDirectory(prefix="jongbench-cli-") as tempdir:
        runs_root = Path(tempdir)
        checkpoint = _write_test_checkpoint(runs_root / "reviewer.pth")

        code = cli.main(
            [
                "selfcheck",
                "--runs-root",
                str(runs_root),
                "--weights",
                str(checkpoint),
            ]
        )
        assert code == 0

        run_dirs = sorted(path for path in runs_root.iterdir() if path.is_dir())
        assert len(run_dirs) == 1
        run_dir = run_dirs[0]

        assert (run_dir / "config.json").exists()
        config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
        assert config["state_hints"] is False
        assert config["reviewer_checkpoint"]["path"] == str(checkpoint)
        assert len(config["reviewer_checkpoint"]["sha256"]) == 64
        assert config["reviewer_checkpoint"]["use_policy"] is False
        logs = sorted((run_dir / "logs").glob("*.json.gz"))
        assert len(logs) == 1
        review_files = sorted((run_dir / "review").glob("*.json"))
        assert len(review_files) == 1
        assert (run_dir / "report.html").exists()
        assert (run_dir / "summary.json").exists()

        original_review = json.loads(review_files[0].read_text(encoding="utf-8"))
        assert original_review["reviewer_checkpoint"] == config["reviewer_checkpoint"]
        original_scores = original_review["scores"]
        assert len(original_review["players"]) == 4
        for seat in range(4):
            player = original_review["players"][str(seat)]
            assert "review" in player
            assert "aggregates" in player

        old_mtime = review_files[0].stat().st_mtime_ns
        time.sleep(0.05)
        code = cli.main(
            ["review", str(run_dir), "--force", "--weights", str(checkpoint)]
        )
        assert code == 0
        regenerated = sorted((run_dir / "review").glob("*.json"))
        assert len(regenerated) == 1
        assert regenerated[0].stat().st_mtime_ns > old_mtime
        regenerated_review = json.loads(regenerated[0].read_text(encoding="utf-8"))
        assert regenerated_review["scores"] == original_scores

        bundle_path = run_dir / "replay.json"
        with patch.object(
            cli,
            "_mortal_evaluate",
            side_effect=AssertionError("replay must not import Mortal/Torch"),
        ):
            code = cli.main(["replay", str(run_dir), "--out", str(bundle_path)])
        assert code == 0
        bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
        assert bundle["scores"] == original_scores
        assert bundle["review"]["scores"] == original_scores
        assert len(bundle["frames"]) > 50
        first = bundle["frames"][0]
        assert first["seq"] == 1
        assert first["event"]["type"] == "start_game"
        assert "seats" in first["snapshot"]

        bank_path = runs_root / "bank.jsonl.gz"
        code = cli.main(
            [
                "positions",
                "--from-log",
                str(logs[0]),
                "--weights",
                str(checkpoint),
                "--out",
                str(bank_path),
            ]
        )
        assert code == 0
        with gzip.open(bank_path, "rt", encoding="utf-8") as handle:
            bank_records = [json.loads(line) for line in handle]
        assert bank_records[0]["record_type"] == "manifest"
        assert (
            bank_records[0]["reviewer"]["checkpoint_sha256"]
            == config["reviewer_checkpoint"]["sha256"]
        )
        assert bank_records[0]["source"]["artifacts"][0]["sha256"] == (
            hashlib.sha256(logs[0].read_bytes()).hexdigest()
        )
        assert bank_records[1]["record_type"] == "position"
        assert len({row["id"] for row in bank_records[1:]}) == len(bank_records) - 1

        code = cli.main(["leaderboard", str(runs_root)])
        assert code == 0
        board = json.loads((runs_root / "leaderboard.json").read_text(encoding="utf-8"))
        assert board["episode_count"] == 1
        assert board["reviewed_count"] == 1
        # Four `random` seats are one spec: the pooled row covers the whole table.
        assert [engine["spec"] for engine in board["leaderboard"]] == ["random"]
        assert board["leaderboard"][0]["episodes"] == 1
        assert board["leaderboard"][0]["placement_counts"] == [1, 1, 1, 1]
        # A run directory is its own single-episode batch.
        assert cli.main(["leaderboard", str(run_dir)]) == 0
        assert (run_dir / "leaderboard.json").exists()

    print("OK")


def test_web_watch_success_exit_code() -> None:
    with patch(
        "jongbench.webui.run_watch_server",
        return_value="/tmp/jongbench-run",
    ) as run_watch:
        code = cli.main(
            [
                "watch",
                "--ui",
                "web",
                "--models",
                "random",
                "random",
                "random",
                "random",
            ]
        )
    assert code == 0
    assert run_watch.call_args.kwargs["state_hints"] is True

    parsed = cli._build_parser().parse_args(
        [
            "run",
            "--models",
            "random",
            "random",
            "random",
            "random",
            "--no-state-hints",
        ]
    )
    assert parsed.state_hints is False


def test_reconstruct_summary_accounts_for_riichi_sticks() -> None:
    events = [
        {"type": "start_game", "names": ["A", "B", "C", "D"], "seed": [9, 1]},
        {
            "type": "start_kyoku",
            "scores": [25000, 25000, 25000, 25000],
            "kyotaku": 0,
        },
        {"type": "reach_accepted", "actor": 1},
        {"type": "ryukyoku", "deltas": [0, 0, 0, 0]},
        {"type": "end_kyoku"},
        {"type": "end_game"},
    ]
    summary = cli.reconstruct_summary(events, Path("9_1.json.gz"))
    assert summary.scores == [26000, 24000, 25000, 25000]
    assert sum(summary.scores) == 100000


def test_decision_filename_cannot_escape_directory() -> None:
    filename = decision_filename("../../outside/model")
    assert "/" not in filename
    assert "\\" not in filename
    assert filename.endswith(".jsonl")


if __name__ == "__main__":
    test_cli_workflow()


def test_plain_json_logs_are_reviewed_with_explicit_q_mode(
    tmp_path: Path, monkeypatch
) -> None:
    run_dir = tmp_path / "run"
    (run_dir / "logs").mkdir(parents=True)
    write_run_config(
        run_dir,
        label="plain",
        models=["random"] * 4,
        names=["p0", "p1", "p2", "p3"],
        games=1,
        seed_start=(1, 1),
        state_hints=False,
    )
    log = run_dir / "logs" / "1_1.json"
    log.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "type": "start_game",
                        "seed": [1, 1],
                        "names": ["p0", "p1", "p2", "p3"],
                    }
                ),
                json.dumps({"type": "end_game"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    assert reviews_missing(run_dir)

    calls = []
    mortal = SimpleNamespace(checkpoint=None)

    def load_engine(weights, *, use_policy):
        calls.append((weights, use_policy))
        return mortal

    monkeypatch.setattr(
        cli,
        "_mortal_evaluate",
        lambda: SimpleNamespace(load_engine=load_engine),
    )
    monkeypatch.setattr(
        cli,
        "review_log",
        lambda path, summary, engine, *, temperature: {"seed": [1, 1]},
    )
    cli._evaluate_run(run_dir, "reviewer.pth", progress=False)
    assert calls == [("reviewer.pth", False)]
    assert not reviews_missing(run_dir)


def test_gameplay_checkpoint_identities_are_written_to_run_config(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    write_run_config(
        run_dir,
        label="controls",
        models=["mortal"] * 4,
        names=["seat0", "seat1", "seat2", "seat3"],
        games=1,
        seed_start=(1, 1),
        state_hints=False,
    )
    identity = {
        "path": "/cache/control.pth",
        "sha256": "a" * 64,
        "source": "auto",
        "use_policy": False,
    }
    checkpoint = SimpleNamespace(as_dict=lambda: identity)
    engines = [
        SimpleNamespace(name=f"seat{i}", checkpoint=checkpoint) for i in range(4)
    ]
    assert record_gameplay_checkpoints(run_dir, engines) == {
        f"seat{i}": identity for i in range(4)
    }
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    assert config["gameplay_checkpoints"] == {f"seat{i}": identity for i in range(4)}
