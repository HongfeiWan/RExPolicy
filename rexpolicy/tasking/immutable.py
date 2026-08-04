"""Small pure-stdlib helpers for immutable canonical task values."""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Any


def freeze_json(value: Any) -> Any:
    """Copy one validated JSON-like value into immutable containers."""
    if isinstance(value, Mapping):
        return MappingProxyType(
            {
                key: freeze_json(child)
                for key, child in sorted(value.items())
            }
        )
    if isinstance(value, (list, tuple)):
        return tuple(freeze_json(child) for child in value)
    return value


def thaw_json(value: Any) -> Any:
    """Return ordinary JSON containers without exposing internal references."""
    if isinstance(value, Mapping):
        return {key: thaw_json(child) for key, child in value.items()}
    if isinstance(value, tuple):
        return [thaw_json(child) for child in value]
    return value
