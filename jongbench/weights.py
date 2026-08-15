from __future__ import annotations

import hashlib
import os
from pathlib import Path
from urllib.request import Request, urlopen

AUTO_MORTAL_WEIGHTS = "auto"
MORTAL_WEIGHTS_URL = (
    "https://huggingface.co/VoidShine/mortal-298k/resolve/main/mortal_298k.pth"
)
MORTAL_WEIGHTS_SHA256 = (
    "bfb3a6c072aa0bfd4171a9cdc77cb6c02ae42cde920843f9e5784394f23447d8"
)
MORTAL_WEIGHTS_FILENAME = f"mortal-298k-{MORTAL_WEIGHTS_SHA256[:12]}.pth"


def mortal_weights_cache_path() -> Path:
    cache = os.environ.get("JONGBENCH_CACHE_DIR") or os.environ.get("XDG_CACHE_HOME")
    root = Path(cache).expanduser() if cache else Path.home() / ".cache"
    return root / "jongbench" / MORTAL_WEIGHTS_FILENAME


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_mortal_weights(weights: str | Path = AUTO_MORTAL_WEIGHTS) -> Path:
    if str(weights) != AUTO_MORTAL_WEIGHTS:
        path = Path(weights).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"Mortal checkpoint not found: {path}")
        return path

    path = mortal_weights_cache_path()
    if path.is_file() and _sha256(path) == MORTAL_WEIGHTS_SHA256:
        return path

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    digest = hashlib.sha256()
    request = Request(MORTAL_WEIGHTS_URL, headers={"User-Agent": "jongbench/0.1"})
    try:
        with urlopen(request, timeout=60) as response, temporary.open("wb") as handle:
            while chunk := response.read(1024 * 1024):
                digest.update(chunk)
                handle.write(chunk)
        actual = digest.hexdigest()
        if actual != MORTAL_WEIGHTS_SHA256:
            raise ValueError(
                f"Mortal checkpoint checksum mismatch: {actual} != {MORTAL_WEIGHTS_SHA256}"
            )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return path
