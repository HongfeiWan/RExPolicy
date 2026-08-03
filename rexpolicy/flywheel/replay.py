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


@dataclass(frozen=True)
class CandidateReplayResult:
    """Observed result of re-executing one archived candidate prefix."""

    rewards: tuple[float, ...]
    success: bool
    failure: bool
    safety_violation: bool
    terminated: bool
    truncated: bool
    executed_action: np.ndarray


class CandidateReplayMismatch(RuntimeError):
    """An archived candidate no longer reproduces under its bound contract."""


def validate_candidate_replay(
    archived: dict[str, Any],
    replayed: CandidateReplayResult,
    *,
    action_tolerance: float,
    reward_tolerance: float,
) -> None:
    """Require replayed actions, rewards, and outcomes to match the archive."""
    for name, tolerance in (
        ("action_tolerance", action_tolerance),
        ("reward_tolerance", reward_tolerance),
    ):
        if not np.isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError(f"{name} must be positive and finite")
    archived_action = np.asarray(
        archived["action"]["action_19d"],
        dtype=np.float32,
    )
    replayed_action = np.asarray(replayed.executed_action, dtype=np.float32)
    if archived_action.shape != replayed_action.shape:
        raise CandidateReplayMismatch(
            "Historical candidate action shape changed: "
            f"{archived_action.shape} != {replayed_action.shape}"
        )
    if not np.isfinite(archived_action).all():
        raise CandidateReplayMismatch(
            "Historical candidate archive contains non-finite actions"
        )
    if not np.isfinite(replayed_action).all():
        raise CandidateReplayMismatch(
            "Historical candidate replay produced non-finite actions"
        )
    action_error = (
        float(np.max(np.abs(archived_action - replayed_action)))
        if archived_action.size
        else 0.0
    )
    if action_error > action_tolerance:
        raise CandidateReplayMismatch(
            "Historical candidate effective action changed: "
            f"{action_error:.3e} > {action_tolerance:.3e}"
        )

    archived_rewards = np.asarray(archived["rewards"], dtype=np.float64)
    replayed_rewards = np.asarray(replayed.rewards, dtype=np.float64)
    if archived_rewards.shape != replayed_rewards.shape:
        raise CandidateReplayMismatch(
            "Historical candidate reward length changed: "
            f"{archived_rewards.shape} != {replayed_rewards.shape}"
        )
    if not np.isfinite(archived_rewards).all():
        raise CandidateReplayMismatch(
            "Historical candidate archive contains non-finite rewards"
        )
    if not np.isfinite(replayed_rewards).all():
        raise CandidateReplayMismatch(
            "Historical candidate replay produced non-finite rewards"
        )
    reward_error = (
        float(np.max(np.abs(archived_rewards - replayed_rewards)))
        if archived_rewards.size
        else 0.0
    )
    if reward_error > reward_tolerance:
        raise CandidateReplayMismatch(
            "Historical candidate rewards changed: "
            f"{reward_error:.3e} > {reward_tolerance:.3e}"
        )

    archived_success = bool(archived["success"])
    archived_truncated = bool(archived["truncated"])
    archived_failure = bool(
        archived.get(
            "failure",
            bool(archived["terminated"])
            and not archived_success
            and not archived_truncated,
        )
    )
    archived_safety = bool(archived.get("safety_violation", archived_failure))
    expected = {
        "success": archived_success,
        "failure": archived_failure,
        "safety_violation": archived_safety,
        "terminated": bool(archived["terminated"]),
        "truncated": archived_truncated,
    }
    actual = {
        "success": bool(replayed.success),
        "failure": bool(replayed.failure),
        "safety_violation": bool(replayed.safety_violation),
        "terminated": bool(replayed.terminated),
        "truncated": bool(replayed.truncated),
    }
    mismatches = [
        name for name, expected_value in expected.items()
        if actual[name] != expected_value
    ]
    if mismatches:
        detail = ", ".join(
            f"{name}={expected[name]}->{actual[name]}" for name in mismatches
        )
        raise CandidateReplayMismatch(
            f"Historical candidate outcome changed: {detail}"
        )


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
