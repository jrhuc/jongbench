from __future__ import annotations

import json
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jongbench import cli


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="jongbench-cli-") as tempdir:
        runs_root = Path(tempdir)

        code = cli.main(["selfcheck", "--runs-root", str(runs_root)])
        assert code == 0

        run_dirs = sorted(path for path in runs_root.iterdir() if path.is_dir())
        assert len(run_dirs) == 1
        run_dir = run_dirs[0]

        assert (run_dir / "config.json").exists()
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

        code = cli.main(["record", str(run_dir)])
        assert code == 0
        demo_path = run_dir / "demo.json"
        assert demo_path.exists()
        demo = json.loads(demo_path.read_text(encoding="utf-8"))
        assert len(demo["frames"]) > 100
        for frame in demo["frames"]:
            assert len(frame["snapshot"]["hands"]) == 4
        assert "review" in demo

        old_mtime = review_files[0].stat().st_mtime_ns
        time.sleep(0.05)
        code = cli.main(["review", str(run_dir), "--force"])
        assert code == 0
        regenerated = sorted((run_dir / "review").glob("*.json"))
        assert len(regenerated) == 1
        assert regenerated[0].stat().st_mtime_ns > old_mtime
        regenerated_review = json.loads(regenerated[0].read_text(encoding="utf-8"))
        assert regenerated_review["scores"] == original_scores

    print("OK")


if __name__ == "__main__":
    main()
