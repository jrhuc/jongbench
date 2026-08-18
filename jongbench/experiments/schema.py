"""Canonical JSON identities and strict record-field validators."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def content_id(namespace: str, value: object) -> str:
    payload = f"{namespace}\0{_canonical_json(value)}".encode()
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _integer(value: object, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{field} must be an integer")
    return value


def _string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise TypeError(f"{field} must be a non-empty string")
    return value


def _optional_string(value: object, field: str) -> str | None:
    if value is None:
        return None
    return _string(value, field)


def _sha256(value: object, field: str, *, optional: bool = False) -> str | None:
    if value is None and optional:
        return None
    digest = _string(value, field).lower()
    if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
        raise ValueError(f"{field} must be a SHA-256 hex digest")
    return digest


def _boolean(value: object, field: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{field} must be a boolean")
    return value


def _number(value: object, field: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError(f"{field} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


def _object_list(value: object, field: str) -> list[dict[str, Any]]:
    if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
        raise TypeError(f"{field} must be a list of JSON objects")
    return value


def _string_tuple(
    value: object, field: str, *, length: int | None = None
) -> tuple[str, ...]:
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item for item in value
    ):
        raise TypeError(f"{field} must be a list of non-empty strings")
    result = tuple(value)
    if length is not None and len(result) != length:
        raise ValueError(f"{field} must contain exactly {length} items")
    return result
