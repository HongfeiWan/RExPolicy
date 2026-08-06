"""Dependency-free shared value types for the Stage 0 experiment path."""

from __future__ import annotations

import hashlib
import json
from enum import Enum
from typing import Any, TypeVar


class _StringValue(str, Enum):
    """A strict string enum with one consistent parser."""

    @classmethod
    def parse(cls, value: Any, name: str) -> _StringValue:
        if not isinstance(value, str):
            raise ValueError(f"{name} must be a string")
        try:
            return cls(value)
        except ValueError as error:
            choices = ", ".join(sorted(item.value for item in cls))
            raise ValueError(f"{name} must be one of: {choices}") from error


class Stage0SystemPath(_StringValue):
    """Top-level implementation paths that must never be mixed implicitly."""

    STAGE0_STATE = "stage0_state"
    FULL_GROOT = "full_groot"


class Stage0Task(_StringValue):
    """Planned Stage 0 task families.

    Reach is the only initial end-to-end acceptance task. Push and Pick are
    named here so persisted configs remain explicit while their entry points
    can still fail closed until independent oracles exist.
    """

    REACH = "reach"
    PUSH = "push"
    PICK = "pick"


class Stage0Outcome(_StringValue):
    """Authoritative terminal outcome stored by the Stage 0 corpus."""

    SUCCESS = "success"
    FAILURE = "failure"
    TIMEOUT = "timeout"


EnumType = TypeVar("EnumType", bound=_StringValue)


def parse_string_enum(
    enum_type: type[EnumType],
    value: Any,
    name: str,
) -> EnumType:
    """Parse a string value without accepting aliases or coercions."""
    return enum_type.parse(value, name)  # type: ignore[return-value]


def canonical_fingerprint(record: Any) -> str:
    """Return a stable SHA-256 digest for one JSON-compatible record."""
    try:
        payload = json.dumps(
            record,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError) as error:
        raise ValueError("fingerprint record must be finite canonical JSON") from error
    return hashlib.sha256(payload).hexdigest()


__all__ = [
    "Stage0Outcome",
    "Stage0SystemPath",
    "Stage0Task",
    "canonical_fingerprint",
    "parse_string_enum",
]
