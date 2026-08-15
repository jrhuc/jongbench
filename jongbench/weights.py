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

WEIGHTS_URL_ENV = "JONGBENCH_WEIGHTS_URL"
WEIGHTS_SHA256_ENV = "JONGBENCH_WEIGHTS_SHA256"
WEIGHTS_USE_POLICY_ENV = "JONGBENCH_WEIGHTS_USE_POLICY"


def auto_weights_use_policy() -> bool:
    value = os.environ.get(WEIGHTS_USE_POLICY_ENV, "").strip().lower()
    if value in {"", "0", "false", "no", "off"}:
        return False
    if value in {"1", "true", "yes", "on"}:
        return True
    raise ValueError(
        f"{WEIGHTS_USE_POLICY_ENV} must be one of 1, true, yes, on, 0, false, no, off"
    )


def _auto_source() -> tuple[str, str, str]:
    configured_url = os.environ.get(WEIGHTS_URL_ENV)
    configured_sha256 = os.environ.get(WEIGHTS_SHA256_ENV)
    if bool(configured_url) != bool(configured_sha256):
        raise ValueError(
            f"{WEIGHTS_URL_ENV} and {WEIGHTS_SHA256_ENV} must be set together"
        )
    if configured_url and configured_sha256:
        sha256 = configured_sha256.lower()
        if len(sha256) != 64 or any(char not in "0123456789abcdef" for char in sha256):
            raise ValueError(f"{WEIGHTS_SHA256_ENV} must be a SHA-256 hex digest")
        return configured_url, sha256, f"weights-{sha256[:12]}.pth"
    return MORTAL_WEIGHTS_URL, MORTAL_WEIGHTS_SHA256, MORTAL_WEIGHTS_FILENAME


def auto_weights_sha256() -> str:
    return _auto_source()[1]


def mortal_weights_cache_path() -> Path:
    cache = os.environ.get("JONGBENCH_CACHE_DIR") or os.environ.get("XDG_CACHE_HOME")
    root = Path(cache).expanduser() if cache else Path.home() / ".cache"
    return root / "jongbench" / _auto_source()[2]


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

    url, expected_sha256, _ = _auto_source()
    path = mortal_weights_cache_path()
    if path.is_file() and _sha256(path) == expected_sha256:
        return path

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    digest = hashlib.sha256()
    request = Request(url, headers={"User-Agent": "jongbench/0.1"})
    try:
        with urlopen(request, timeout=60) as response, temporary.open("wb") as handle:
            while chunk := response.read(1024 * 1024):
                digest.update(chunk)
                handle.write(chunk)
        actual = digest.hexdigest()
        if actual != expected_sha256:
            raise ValueError(
                f"Mortal checkpoint checksum mismatch: {actual} != {expected_sha256}"
            )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return path
