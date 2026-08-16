from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, replace
from pathlib import Path
from urllib.request import Request, urlopen

from .artifacts import file_sha256

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


@dataclass(frozen=True, slots=True)
class ResolvedCheckpoint:
    """The immutable checkpoint and inference mode selected for one operation."""

    path: Path
    sha256: str
    source: str
    use_policy: bool

    def as_dict(self) -> dict[str, str | bool]:
        return {
            "path": str(self.path),
            "sha256": self.sha256,
            "source": self.source,
            "use_policy": self.use_policy,
        }


type CheckpointSpec = str | Path
type CheckpointInput = CheckpointSpec | ResolvedCheckpoint


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


def _cache_root() -> Path:
    cache = os.environ.get("JONGBENCH_CACHE_DIR") or os.environ.get("XDG_CACHE_HOME")
    return Path(cache).expanduser() if cache else Path.home() / ".cache"


def mortal_weights_cache_path() -> Path:
    return _cache_root() / "jongbench" / _auto_source()[2]


def _resolve_auto_weights(url: str, expected_sha256: str, filename: str) -> Path:
    path = _cache_root() / "jongbench" / filename
    if path.is_file() and file_sha256(path) == expected_sha256:
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


def resolve_mortal_checkpoint(
    weights: CheckpointInput = AUTO_MORTAL_WEIGHTS,
    *,
    use_policy: bool | None = None,
) -> ResolvedCheckpoint:
    """Resolve path, provenance, digest, and policy selection exactly once.

    An explicit ``use_policy`` takes precedence over ambient configuration. Passing an
    existing identity never reads the environment or re-resolves its path.
    """

    if isinstance(weights, ResolvedCheckpoint):
        if use_policy is None or weights.use_policy == bool(use_policy):
            return weights
        return replace(weights, use_policy=bool(use_policy))

    resolved_use_policy = (
        auto_weights_use_policy() if use_policy is None else bool(use_policy)
    )
    if str(weights) != AUTO_MORTAL_WEIGHTS:
        path = Path(weights).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"Mortal checkpoint not found: {path}")
        return ResolvedCheckpoint(
            path=path,
            sha256=file_sha256(path),
            source=str(weights),
            use_policy=resolved_use_policy,
        )

    url, expected_sha256, filename = _auto_source()
    path = _resolve_auto_weights(url, expected_sha256, filename)
    return ResolvedCheckpoint(
        path=path,
        sha256=expected_sha256,
        source=url,
        use_policy=resolved_use_policy,
    )


def resolve_mortal_weights(
    weights: CheckpointInput = AUTO_MORTAL_WEIGHTS,
) -> Path:
    """Return a checkpoint path (compatibility wrapper for the original API)."""

    if isinstance(weights, ResolvedCheckpoint):
        return weights.path
    return resolve_mortal_checkpoint(weights, use_policy=False).path


GRADING_CHECKPOINT_SOURCE = "VoidShine/mortal-298k/mortal_298k.pth"


def resolve_grading_checkpoint() -> ResolvedCheckpoint:
    """Mortal 298k only. Ambient JONGBENCH_WEIGHTS_* cannot grade.

    Phoenix is a control/opponent. Using a jongbench-trained checkpoint as the
    grader is circular and kills cross-release comparability.
    """
    path = _resolve_auto_weights(
        MORTAL_WEIGHTS_URL, MORTAL_WEIGHTS_SHA256, MORTAL_WEIGHTS_FILENAME
    )
    return ResolvedCheckpoint(
        path=path,
        sha256=MORTAL_WEIGHTS_SHA256,
        source=GRADING_CHECKPOINT_SOURCE,
        use_policy=False,
    )
