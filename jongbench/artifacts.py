from __future__ import annotations

import hashlib
import re


_SAFE_STEM = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.+-]{0,119}")
_WINDOWS_RESERVED = {
    "aux",
    "clock$",
    "con",
    "nul",
    "prn",
    *(f"com{index}" for index in range(1, 10)),
    *(f"lpt{index}" for index in range(1, 10)),
}


def decision_filename(name: str) -> str:
    reserved_stem = name.split(".", 1)[0].casefold()
    if _SAFE_STEM.fullmatch(name) and reserved_stem not in _WINDOWS_RESERVED:
        return f"{name}.jsonl"

    slug = re.sub(r"[^A-Za-z0-9_.+-]+", "_", name).strip("._-")[:80]
    digest = hashlib.sha256(name.encode("utf-8")).hexdigest()[:12]
    return f"{slug or 'engine'}-{digest}.jsonl"
