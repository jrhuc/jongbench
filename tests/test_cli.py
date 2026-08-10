from __future__ import annotations

import json
import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jongbench import cli
from jongbench.artifacts import decision_filename


def test_cli_workflow() -> None:
    with tempfile.TemporaryDirectory(prefix="jongbench-cli-") as tempdir:
        runs_root = Path(tempdir)

        code = cli.main(["selfcheck", "--runs-root", str(runs_root)])
        assert code == 0

        run_dirs = sorted(path for path in runs_root.iterdir() if path.is_dir())
        assert len(run_dirs) == 1
        run_dir = run_dirs[0]

        assert (run_dir / "config.json").exists()
        config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
        assert config["state_hints"] is False
        logs = sorted((run_dir / "logs").glob("*.json.gz"))
        assert len(logs) == 1
        review_files = sorted((run_dir / "review").glob("*.json"))
        assert len(review_files) == 1
        assert (run_dir / "report.html").exists()
        assert (run_dir / "summary.json").exists()

        original_review = json.loads(review_files[0].read_text(encoding="utf-8"))
        original_scores = original_review["scores"]
        assert len(original_review["players"]) == 4
        for seat in range(4):
            player = original_review["players"][str(seat)]
            assert "review" in player
            assert "aggregates" in player

        old_mtime = review_files[0].stat().st_mtime_ns
        time.sleep(0.05)
        code = cli.main(["review", str(run_dir), "--force"])
        assert code == 0
        regenerated = sorted((run_dir / "review").glob("*.json"))
        assert len(regenerated) == 1
        assert regenerated[0].stat().st_mtime_ns > old_mtime
        regenerated_review = json.loads(regenerated[0].read_text(encoding="utf-8"))
        assert regenerated_review["scores"] == original_scores

        bundle_path = run_dir / "replay.json"
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
    summary = cli._reconstruct_summary(events, Path("9_1.json.gz"))
    assert summary.scores == [26000, 24000, 25000, 25000]
    assert sum(summary.scores) == 100000


def test_decision_filename_cannot_escape_directory() -> None:
    filename = decision_filename("../../outside/model")
    assert "/" not in filename
    assert "\\" not in filename
    assert filename.endswith(".jsonl")


if __name__ == "__main__":
    test_cli_workflow()
