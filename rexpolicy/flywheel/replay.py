"""Deterministic same-state validation without archiving simulator state."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Iterator

import numpy as np


@dataclass(frozen=True)
class ReplayFingerprint:
    """Comparison metrics and a metadata-only hash for world zero."""

    digest: str
    max_float_spread: float
    field_count: int


def validate_replay_fingerprint(
    tree: dict[str, Any],
    *,
    world_count: int,
    float_tolerance: float,
) -> ReplayFingerprint:
    """Require all candidate worlds to represent the same simulator state."""
    if world_count < 1:
        raise ValueError("world_count must be positive")
    if float_tolerance <= 0.0:
        raise ValueError("float_tolerance must be positive")
    digest = hashlib.sha256()
    maximum = 0.0
    fields = 0
    for name, value in _flatten(tree):
        array = _to_numpy(value)
        if array.ndim < 1 or int(array.shape[0]) != world_count:
            raise ValueError(
                f"Replay fingerprint field {name!r} must start with "
                f"world dimension {world_count}, got {array.shape}"
            )
        reference = array[0]
        fields += 1
        digest.update(name.encode())
        digest.update(str(reference.shape).encode())
        if np.issubdtype(array.dtype, np.floating):
            if not np.isfinite(array).all():
                raise FloatingPointError(
                    f"Replay fingerprint field {name!r} is non-finite"
                )
            spread = float(np.max(np.abs(array - reference)))
            maximum = max(maximum, spread)
            if spread > float_tolerance:
                raise RuntimeError(
                    f"Candidate replay field {name!r} diverged: "
                    f"{spread:.3e} > {float_tolerance:.3e}"
                )
            quantized = np.rint(
                reference.astype(np.float64) / float_tolerance
            ).astype(np.int64)
            digest.update(b"quantized-float64")
            digest.update(quantized.tobytes(order="C"))
        else:
            if not np.all(array == reference):
                raise RuntimeError(
                    f"Candidate replay field {name!r} is not exactly equal"
                )
            contiguous = np.ascontiguousarray(reference)
            digest.update(str(contiguous.dtype).encode())
            digest.update(contiguous.tobytes(order="C"))
    if fields == 0:
        raise ValueError("Replay fingerprint tree is empty")
    return ReplayFingerprint(
        digest=digest.hexdigest(),
        max_float_spread=maximum,
        field_count=fields,
    )


def fingerprint_world_zero(
    tree: dict[str, Any],
    *,
    world_count: int,
    float_tolerance: float,
) -> str:
    """Return the same world-zero digest while still validating all branches."""
    return validate_replay_fingerprint(
        tree,
        world_count=world_count,
        float_tolerance=float_tolerance,
    ).digest


def _flatten(tree: Any, prefix: str = "") -> Iterator[tuple[str, Any]]:
    if isinstance(tree, dict):
        for key in sorted(tree):
            name = f"{prefix}.{key}" if prefix else str(key)
            yield from _flatten(tree[key], name)
        return
    yield prefix, tree


def _to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    if hasattr(value, "numpy"):
        return value.numpy()
    return np.asarray(value)
