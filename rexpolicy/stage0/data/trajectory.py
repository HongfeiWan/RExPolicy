"""Strict materialized trajectory schema for state-only Stage 0 data."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from rexpolicy.stage0.types import Stage0Outcome


TRAJECTORY_SCHEMA_VERSION = 1
TRAJECTORY_SCHEMA_ID = "rexpolicy/stage0-state-trajectory/v1"
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _require_exact_keys(
    record: Mapping[str, Any],
    expected: set[str],
    *,
    context: str,
) -> None:
    actual = set(record)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(
            f"{context} keys do not match schema; missing={missing}, extra={extra}"
        )


def _require_safe_id(value: str, name: str) -> None:
    if not isinstance(value, str) or _SAFE_ID.fullmatch(value) is None:
        raise ValueError(f"{name} must match {_SAFE_ID.pattern!r}; got {value!r}")


def _require_text(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be non-empty text")


def _require_sha256(value: str, name: str) -> None:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")


def _require_non_negative_integer(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")


@dataclass(frozen=True)
class Stage0TrajectoryProvenance:
    """Identity and semantic commitments for one simulator rollout."""

    trajectory_id: str
    reset_group_id: str
    task_id: str
    oracle_id: str
    simulator_sha256: str
    experiment_sha256: str
    state_schema_sha256: str
    action_schema_sha256: str
    generation: int
    rank: int
    world: int
    reset_seed: int

    def __post_init__(self) -> None:
        _require_safe_id(self.trajectory_id, "trajectory_id")
        _require_safe_id(self.reset_group_id, "reset_group_id")
        _require_text(self.task_id, "task_id")
        _require_text(self.oracle_id, "oracle_id")
        for name in (
            "simulator_sha256",
            "experiment_sha256",
            "state_schema_sha256",
            "action_schema_sha256",
        ):
            _require_sha256(getattr(self, name), name)
        for name in ("generation", "rank", "world", "reset_seed"):
            _require_non_negative_integer(getattr(self, name), name)

    def to_record(self) -> dict[str, Any]:
        return {
            "action_schema_sha256": self.action_schema_sha256,
            "experiment_sha256": self.experiment_sha256,
            "generation": self.generation,
            "oracle_id": self.oracle_id,
            "rank": self.rank,
            "reset_group_id": self.reset_group_id,
            "reset_seed": self.reset_seed,
            "simulator_sha256": self.simulator_sha256,
            "state_schema_sha256": self.state_schema_sha256,
            "task_id": self.task_id,
            "trajectory_id": self.trajectory_id,
            "world": self.world,
        }

    @classmethod
    def from_record(
        cls,
        record: Mapping[str, Any],
    ) -> Stage0TrajectoryProvenance:
        if not isinstance(record, Mapping):
            raise ValueError("trajectory provenance must be a mapping")
        expected = {
            "action_schema_sha256",
            "experiment_sha256",
            "generation",
            "oracle_id",
            "rank",
            "reset_group_id",
            "reset_seed",
            "simulator_sha256",
            "state_schema_sha256",
            "task_id",
            "trajectory_id",
            "world",
        }
        _require_exact_keys(record, expected, context="trajectory provenance")
        return cls(**{key: record[key] for key in expected})


@dataclass(frozen=True)
class Stage0Trajectory:
    """One complete state/action rollout including its terminal outcome.

    ``states`` is ``[T + 1, state_dim]`` and ``actions`` is
    ``[T, action_dim]``.  Tensors are detached and materialized as contiguous
    CPU float32 values so shard persistence cannot retain an autograd graph or
    device-dependent representation.
    """

    provenance: Stage0TrajectoryProvenance
    states: Any
    actions: Any
    outcome: Stage0Outcome
    terminated: bool
    truncated: bool
    failure_reason: str | None = None

    def __post_init__(self) -> None:
        import torch

        if not isinstance(self.provenance, Stage0TrajectoryProvenance):
            raise TypeError("provenance must be Stage0TrajectoryProvenance")
        try:
            outcome = Stage0Outcome(self.outcome)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"unsupported Stage 0 outcome: {self.outcome!r}") from exc
        object.__setattr__(self, "outcome", outcome)

        states = _materialize_tensor(self.states, name="states", torch=torch)
        actions = _materialize_tensor(self.actions, name="actions", torch=torch)
        if states.ndim != 2 or states.shape[1] < 1:
            raise ValueError("states must have shape [T + 1, state_dim > 0]")
        if actions.ndim != 2 or actions.shape[1] < 1:
            raise ValueError("actions must have shape [T, action_dim > 0]")
        if actions.shape[0] < 1:
            raise ValueError("a trajectory must contain at least one transition")
        if states.shape[0] != actions.shape[0] + 1:
            raise ValueError(
                "states/actions are not transition aligned: expected "
                f"states.shape[0] == {actions.shape[0] + 1}, got {states.shape[0]}"
            )
        object.__setattr__(self, "states", states)
        object.__setattr__(self, "actions", actions)

        if not isinstance(self.terminated, bool) or not isinstance(
            self.truncated, bool
        ):
            raise TypeError("terminated and truncated must be bool")
        reason = self.failure_reason
        if reason is not None and (not isinstance(reason, str) or not reason.strip()):
            raise ValueError("failure_reason must be None or non-empty text")
        if outcome is Stage0Outcome.SUCCESS:
            valid = self.terminated and not self.truncated and reason is None
        elif outcome is Stage0Outcome.FAILURE:
            valid = self.terminated and not self.truncated and reason is not None
        else:
            valid = not self.terminated and self.truncated and reason is None
        if not valid:
            raise ValueError(
                "outcome flags violate the Stage 0 terminal contract: "
                f"outcome={outcome.value}, terminated={self.terminated}, "
                f"truncated={self.truncated}, failure_reason={reason!r}"
            )

    @property
    def transition_count(self) -> int:
        return int(self.actions.shape[0])

    @property
    def state_dim(self) -> int:
        return int(self.states.shape[1])

    @property
    def action_dim(self) -> int:
        return int(self.actions.shape[1])

    @property
    def is_success(self) -> bool:
        return self.outcome is Stage0Outcome.SUCCESS

    def metadata_record(self) -> dict[str, Any]:
        return {
            "failure_reason": self.failure_reason,
            "outcome": self.outcome.value,
            "provenance": self.provenance.to_record(),
            "terminated": self.terminated,
            "truncated": self.truncated,
        }


def _materialize_tensor(value: Any, *, name: str, torch: Any) -> Any:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.dtype is not torch.float32:
        raise ValueError(f"{name} must use torch.float32, got {value.dtype}")
    if value.layout is not torch.strided:
        raise ValueError(f"{name} must use the strided tensor layout")
    tensor = value.detach().to(device="cpu").contiguous().clone()
    if not bool(torch.isfinite(tensor).all().item()):
        raise ValueError(f"{name} contains non-finite values")
    return tensor


def trajectory_sha256(trajectory: Stage0Trajectory) -> str:
    """Hash semantic metadata and canonical float32 tensor bytes."""
    if not isinstance(trajectory, Stage0Trajectory):
        raise TypeError("trajectory must be Stage0Trajectory")
    digest = hashlib.sha256()
    digest.update(TRAJECTORY_SCHEMA_ID.encode("utf-8"))
    digest.update(_canonical_json(trajectory.metadata_record()))
    for name, tensor in (
        ("states", trajectory.states),
        ("actions", trajectory.actions),
    ):
        digest.update(name.encode("ascii"))
        digest.update(_canonical_json(list(tensor.shape)))
        payload = tensor.numpy().tobytes(order="C")
        digest.update(len(payload).to_bytes(8, byteorder="big", signed=False))
        digest.update(payload)
    return digest.hexdigest()


def trajectory_corpus_sha256(
    trajectories: Iterable[Stage0Trajectory],
) -> str:
    """Commit to a trajectory corpus independent of iteration order."""
    identities: list[tuple[str, str]] = []
    seen: set[str] = set()
    for trajectory in trajectories:
        trajectory_id = trajectory.provenance.trajectory_id
        if trajectory_id in seen:
            raise ValueError(f"duplicate trajectory_id in corpus: {trajectory_id}")
        seen.add(trajectory_id)
        identities.append((trajectory_id, trajectory_sha256(trajectory)))
    if not identities:
        raise ValueError("a trajectory corpus cannot be empty")
    return hashlib.sha256(_canonical_json(sorted(identities))).hexdigest()
