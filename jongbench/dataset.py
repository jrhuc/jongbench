from __future__ import annotations

import gzip
import random
import shutil
import tempfile
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import torch
from libriichi.dataset import GameplayLoader
from torch.utils.data import IterableDataset

import jongbench  # noqa: F401

DEFAULT_PTS = (6.0, 4.0, 2.0, 0.0)


def discover_logs(path: str | Path) -> list[str]:
    root = Path(path)
    if root.is_file():
        return [str(root)]
    if not root.is_dir():
        raise FileNotFoundError(f"log path does not exist: {root}")
    files = {
        str(p)
        for pattern in ("*.json.gz", "*.mjson.gz", "*.json", "*.mjson")
        for p in root.rglob(pattern)
    }
    return sorted(files)


def _as_numpy(value) -> np.ndarray:
    return np.asarray(value)


def _duplicate_challenger_player(path: str | Path) -> int:
    name = Path(path).name
    for suffix in (".gz", ".mjson", ".json"):
        name = name.removesuffix(suffix)
    split = name.rsplit("_", 1)[-1]
    try:
        return {"a": 0, "b": 1, "c": 2, "d": 3}[split]
    except KeyError as exc:
        raise ValueError(f"not a duplicate-arena log filename: {path}") from exc


def iter_gameplay_samples(
    files: list[str],
    *,
    version: int = 4,
    pts: tuple[float, float, float, float] = DEFAULT_PTS,
    file_batch_size: int = 4,
    shuffle_files: bool = True,
    shuffle_buffer: bool = True,
    duplicate_challenger_only: bool = False,
) -> Iterator[tuple[np.ndarray, int, np.ndarray, int, float, int]]:
    """Yield (obs, action, mask, steps_to_done, reward, next_rank) from mjai logs.

    Reward is the full-hanchan placement point ``pts[final_rank]``, not Mortal's
    GRP kyoku-delta. ``next_rank`` is the 0-3 rank after the current kyoku.
    """
    files = list(files)
    if shuffle_files:
        random.shuffle(files)
    loader = GameplayLoader(
        version,
        oracle=False,
        always_include_kan_select=True,
    )
    pts_arr = np.asarray(pts, dtype=np.float64)
    for start in range(0, len(files), file_batch_size):
        batch_files = files[start : start + file_batch_size]
        with tempfile.TemporaryDirectory(prefix="jongbench-logs-") as tempdir:
            loader_files = []
            for index, path in enumerate(batch_files):
                with open(path, "rb") as source:
                    magic = source.read(2)
                    source.seek(0)
                    if magic == b"\x1f\x8b":
                        loader_files.append(path)
                        continue
                    target = Path(tempdir) / f"{index}.json.gz"
                    with gzip.open(target, "wb") as compressed:
                        shutil.copyfileobj(source, compressed)
                    loader_files.append(str(target))
            data = loader.load_gz_log_files(loader_files)
        buffer: list[tuple[np.ndarray, int, np.ndarray, int, float, int]] = []
        for source_path, file_games in zip(batch_files, data, strict=True):
            challenger_player = (
                _duplicate_challenger_player(source_path)
                if duplicate_challenger_only
                else None
            )
            for game in file_games:
                player_id = int(game.take_player_id())
                if challenger_player is not None and player_id != challenger_player:
                    continue
                obs = game.take_obs()
                actions = game.take_actions()
                masks = game.take_masks()
                at_kyoku = game.take_at_kyoku()
                dones = game.take_dones()
                apply_gamma = game.take_apply_gamma()
                grp = game.take_grp()
                n = len(obs)
                if n == 0:
                    continue

                rank_by_player = grp.take_rank_by_player()
                final_scores = np.asarray(grp.take_final_scores(), dtype=np.float64)
                grp_feature = _as_numpy(grp.take_feature())
                game_pts = float(pts_arr[int(rank_by_player[player_id])])

                scores_seq = np.concatenate(
                    (grp_feature[:, 3:] * 1e4, final_scores[None, :]), axis=0
                )
                rank_seq = (
                    (-scores_seq).argsort(-1, kind="stable").argsort(-1, kind="stable")
                )
                player_ranks = rank_seq[:, player_id]

                steps_to_done = np.zeros(n, dtype=np.int64)
                for i in reversed(range(n)):
                    if not dones[i]:
                        nxt = steps_to_done[i + 1] if i + 1 < n else 0
                        steps_to_done[i] = nxt + int(apply_gamma[i])

                for i in range(n):
                    rank_idx = min(int(at_kyoku[i]) + 1, len(player_ranks) - 1)
                    buffer.append(
                        (
                            _as_numpy(obs[i]).astype(np.float32, copy=False),
                            int(actions[i]),
                            _as_numpy(masks[i]).astype(bool, copy=False),
                            int(steps_to_done[i]),
                            game_pts,
                            int(player_ranks[rank_idx]),
                        )
                    )
        if shuffle_buffer:
            random.shuffle(buffer)
        yield from buffer


class GameplayIterable(IterableDataset):
    def __init__(
        self,
        files: list[str],
        *,
        version: int = 4,
        pts: tuple[float, float, float, float] = DEFAULT_PTS,
        file_batch_size: int = 4,
        infinite: bool = True,
        shuffle: bool = True,
        duplicate_challenger_only: bool = False,
    ) -> None:
        super().__init__()
        self.files = list(files)
        self.version = version
        self.pts = pts
        self.file_batch_size = file_batch_size
        self.infinite = infinite
        self.shuffle = shuffle
        self.duplicate_challenger_only = duplicate_challenger_only
        if not self.files:
            raise ValueError("no gameplay logs to train on")

    def __iter__(self):
        while True:
            yielded = False
            for obs, action, mask, steps, reward, rank in iter_gameplay_samples(
                self.files,
                version=self.version,
                pts=self.pts,
                file_batch_size=self.file_batch_size,
                shuffle_files=self.shuffle,
                shuffle_buffer=self.shuffle,
                duplicate_challenger_only=self.duplicate_challenger_only,
            ):
                yielded = True
                yield (
                    torch.from_numpy(np.ascontiguousarray(obs)),
                    torch.tensor(action, dtype=torch.long),
                    torch.from_numpy(np.ascontiguousarray(mask)),
                    torch.tensor(steps, dtype=torch.long),
                    torch.tensor(reward, dtype=torch.float32),
                    torch.tensor(rank, dtype=torch.long),
                )
            if not yielded:
                raise RuntimeError("gameplay logs contain no trainable decisions")
            if not self.infinite:
                break
