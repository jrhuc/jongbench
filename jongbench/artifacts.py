from __future__ import annotations

import gzip
import hashlib
import json
import re
from pathlib import Path
from typing import Any

_SAFE_STEM = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.+-]{0,119}")


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


_WINDOWS_RESERVED = {
    "aux",
    "clock$",
    "con",
    "nul",
    "prn",
    *(f"com{index}" for index in range(1, 10)),
    *(f"lpt{index}" for index in range(1, 10)),
}


def load_mjai_log(path: str | Path) -> list[dict[str, Any]]:
    """Load a plain or gzip-compressed MJAI JSON array/JSONL log without Torch."""
    source = Path(path)
    opener = gzip.open if source.suffix == ".gz" else open
    with opener(source, "rt", encoding="utf-8") as handle:
        text = handle.read()
    stripped = text.strip()
    if not stripped:
        return []
    if stripped.startswith("["):
        loaded = json.loads(stripped)
        if not isinstance(loaded, list):
            raise ValueError("expected a JSON array log")
        return loaded
    return [json.loads(line) for line in stripped.splitlines() if line.strip()]


def decision_filename(name: str) -> str:
    reserved_stem = name.split(".", 1)[0].casefold()
    if _SAFE_STEM.fullmatch(name) and reserved_stem not in _WINDOWS_RESERVED:
        return f"{name}.jsonl"

    slug = re.sub(r"[^A-Za-z0-9_.+-]+", "_", name).strip("._-")[:80]
    digest = hashlib.sha256(name.encode("utf-8")).hexdigest()[:12]
    return f"{slug or 'engine'}-{digest}.jsonl"
