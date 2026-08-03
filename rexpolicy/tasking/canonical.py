"""Strict JSON parsing and canonical hashing for task artifacts."""

from __future__ import annotations

import hashlib
import json
from typing import Any

MAX_JSON_BYTES = 1_000_000


def _reject_constant(value: str) -> None:
    raise ValueError(f"Non-finite JSON constant {value!r} is forbidden")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    output = {}
    for key, value in pairs:
        if key in output:
            raise ValueError(f"Duplicate JSON key {key!r}")
        output[key] = value
    return output


def strict_json_loads(text: str, *, max_bytes: int = MAX_JSON_BYTES) -> Any:
    """Parse one bounded JSON value while rejecting duplicates and NaN/Inf."""
    if not isinstance(text, str):
        raise TypeError("JSON input must be text")
    if max_bytes < 1:
        raise ValueError("max_bytes must be positive")
    if len(text.encode("utf-8")) > max_bytes:
        raise ValueError("JSON input exceeds the size limit")
    try:
        return json.loads(
            text,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except (RecursionError, UnicodeError) as error:
        raise ValueError("JSON input is not safely parseable") from error


def canonical_json(value: Any) -> str:
    """Serialize one JSON value with deterministic keys and number checks."""
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def canonical_fingerprint(value: Any) -> str:
    """Return the SHA-256 of the canonical ASCII JSON representation."""
    return hashlib.sha256(canonical_json(value).encode("ascii")).hexdigest()
