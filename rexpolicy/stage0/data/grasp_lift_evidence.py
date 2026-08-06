"""Immutable transition evidence for reward-free Stage 0 Grasp-Lift data.

The state-only :class:`~rexpolicy.stage0.data.trajectory.Stage0Trajectory`
schema remains unchanged.  This module stores task-specific authoring,
contact, safety, and oracle facts in a separately checksummed sidecar.  Every
tensor row describes the transition produced by the action at the same row;
measured robot state is the post-transition ``state[t + 1]`` value.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import uuid
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from rexpolicy.stage0.data.trajectory import (
    Stage0Trajectory,
    _canonical_json,
    _require_exact_keys,
    trajectory_sha256,
)
from rexpolicy.stage0.envs.grasp_lift_action import (
    DEFAULT_GRASP_LIFT_ACTION_SCHEMA,
    GRASP_LIFT_ACTION_DIM,
    GRASP_LIFT_ACTION_SCHEMA_ID,
)
from rexpolicy.stage0.envs.grasp_lift_state_view import (
    DEFAULT_GRASP_LIFT_STATE_VIEW,
    GRASP_LIFT_STATE_VIEW_ID,
)
from rexpolicy.stage0.envs.state_schema import DEFAULT_STAGE0_STATE_SCHEMA


GRASP_LIFT_EVIDENCE_SCHEMA_VERSION = 2
GRASP_LIFT_EVIDENCE_SCHEMA_ID = "rexpolicy/stage0-grasp-lift-evidence/v2"
GRASP_LIFT_EVIDENCE_DESCRIPTOR_SCHEMA_VERSION = 1
GRASP_LIFT_EVIDENCE_DESCRIPTOR_SCHEMA_ID = (
    "rexpolicy/stage0-grasp-lift-evidence-shard/v1"
)
GRASP_LIFT_COMPOSITE_CORPUS_SCHEMA_ID = (
    "rexpolicy/stage0-grasp-lift-trajectory-evidence-corpus/v1"
)

_DESCRIPTOR_SUFFIX = ".grasp-lift-evidence.json"
_PAYLOAD_SUFFIX = ".grasp-lift-evidence.pt"
_SAFE_TRAJECTORY_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_MODE_ID = re.compile(r"^stage0/grasp_lift/[a-z][a-z0-9_]+/v1$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class _TensorSpec:
    dtype: str
    width: int | None = None

    def shape(self, transition_count: int) -> tuple[int, ...]:
        if self.width is None:
            return (transition_count,)
        return (transition_count, self.width)


# Names are part of the v1 semantic schema.  Scalar transition facts are [T];
# vector facts are [T, width].  Integer evidence is normalized to int64 so
# platform- or simulator-specific int32 values cannot alter serialized meaning.
GRASP_LIFT_EVIDENCE_TENSOR_SPECS: Mapping[str, _TensorSpec] = MappingProxyType(
    {
        "proposal_action": _TensorSpec("float32", GRASP_LIFT_ACTION_DIM),
        "executed_action": _TensorSpec("float32", GRASP_LIFT_ACTION_DIM),
        "measured_eef_position": _TensorSpec("float32", 3),
        "measured_hand_joint_position": _TensorSpec("float32", 10),
        "oracle_lift_height": _TensorSpec("float32"),
        "oracle_lateral_displacement": _TensorSpec("float32"),
        "oracle_bottle_tilt": _TensorSpec("float32"),
        "authoring_has_tentative_target": _TensorSpec("bool"),
        "authoring_has_frozen_target": _TensorSpec("bool"),
        "current_opposed_grasp": _TensorSpec("bool"),
        "grasp_confirmed": _TensorSpec("bool"),
        "had_hand_contact": _TensorSpec("bool"),
        "forbidden_hand_contact": _TensorSpec("bool"),
        "collision_buffer_overflow": _TensorSpec("bool"),
        "oracle_success": _TensorSpec("bool"),
        "oracle_failure": _TensorSpec("bool"),
        "forbidden_contact_violation": _TensorSpec("bool"),
        "lateral_displacement_violation": _TensorSpec("bool"),
        "bottle_tilt_violation": _TensorSpec("bool"),
        "dropped_grasp_violation": _TensorSpec("bool"),
        "collision_buffer_overflow_violation": _TensorSpec("bool"),
        "triangle_pair_buffer_available": _TensorSpec("bool"),
        "authoring_phase_before": _TensorSpec("int64"),
        "authoring_phase_after": _TensorSpec("int64"),
        "authoring_approach_control_steps": _TensorSpec("int64"),
        "authoring_close_control_steps": _TensorSpec("int64"),
        "authoring_close_elapsed_control_steps": _TensorSpec("int64"),
        "authoring_stable_grasp_control_steps": _TensorSpec("int64"),
        "authoring_confirm_hold_control_steps": _TensorSpec("int64"),
        "authoring_lift_control_steps": _TensorSpec("int64"),
        "authoring_success_hold_control_steps": _TensorSpec("int64"),
        "opposed_grasp_max_consecutive_physics_frames": _TensorSpec("int64"),
        "finger_contact_counts": _TensorSpec("int64", 5),
        "touching_finger_count": _TensorSpec("int64"),
        "forbidden_hand_contact_count": _TensorSpec("int64"),
        "oracle_success_hold_steps": _TensorSpec("int64"),
        "oracle_drop_gap_steps": _TensorSpec("int64"),
        "rigid_contact_capacity": _TensorSpec("int64"),
        "rigid_contact_frame_max": _TensorSpec("int64"),
        "rigid_contact_overflow_frame_count": _TensorSpec("int64"),
        "rigid_contact_overflow_excess_count": _TensorSpec("int64"),
        "triangle_pair_capacity": _TensorSpec("int64"),
        "triangle_pair_frame_max": _TensorSpec("int64"),
        "triangle_pair_overflow_frame_count": _TensorSpec("int64"),
        "triangle_pair_overflow_excess_count": _TensorSpec("int64"),
    }
)


class GraspLiftEvidenceError(RuntimeError):
    """Base class for Grasp-Lift evidence persistence errors."""


class GraspLiftEvidenceIntegrityError(GraspLiftEvidenceError):
    """A committed evidence sidecar failed strict integrity validation."""


def _require_text(value: Any, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be non-empty text")


def _require_sha256(value: Any, name: str) -> None:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")


def _require_non_negative_integer(value: Any, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")


@dataclass(frozen=True)
class GraspLiftEvidenceMetadata:
    """Identity commitments shared by every transition in one sidecar."""

    trajectory_id: str
    trajectory_sha256: str
    mode_id: str
    mode_sha256: str
    action_schema_id: str
    action_schema_sha256: str
    state_view_id: str
    state_view_sha256: str
    implementation_sha256: str
    experiment_sha256: str
    reset_seed: int
    world: int
    reset_lift_axis: tuple[float, float, float]
    transition_count: int

    def __post_init__(self) -> None:
        if (
            not isinstance(self.trajectory_id, str)
            or _SAFE_TRAJECTORY_ID.fullmatch(self.trajectory_id) is None
        ):
            raise ValueError("trajectory_id is not a safe shard identifier")
        if not isinstance(self.mode_id, str) or _MODE_ID.fullmatch(self.mode_id) is None:
            raise ValueError(
                "mode_id must match 'stage0/grasp_lift/<lower_snake_case>/v1'"
            )
        for name in (
            "trajectory_sha256",
            "mode_sha256",
            "action_schema_sha256",
            "state_view_sha256",
            "implementation_sha256",
            "experiment_sha256",
        ):
            _require_sha256(getattr(self, name), name)
        _require_text(self.action_schema_id, "action_schema_id")
        _require_text(self.state_view_id, "state_view_id")
        if self.action_schema_id != GRASP_LIFT_ACTION_SCHEMA_ID:
            raise ValueError("action_schema_id does not identify Grasp-Lift v1")
        if self.action_schema_sha256 != DEFAULT_GRASP_LIFT_ACTION_SCHEMA.sha256:
            raise ValueError("action_schema_sha256 changed from Grasp-Lift v1")
        if self.state_view_id != GRASP_LIFT_STATE_VIEW_ID:
            raise ValueError("state_view_id does not identify Grasp-Lift v1")
        if self.state_view_sha256 != DEFAULT_GRASP_LIFT_STATE_VIEW.sha256:
            raise ValueError("state_view_sha256 changed from Grasp-Lift v1")
        _require_non_negative_integer(self.reset_seed, "reset_seed")
        _require_non_negative_integer(self.world, "world")
        if (
            isinstance(self.transition_count, bool)
            or not isinstance(self.transition_count, int)
            or self.transition_count < 1
        ):
            raise ValueError("transition_count must be a positive integer")
        axis = self.reset_lift_axis
        if (
            not isinstance(axis, tuple)
            or len(axis) != 3
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                for value in axis
            )
        ):
            raise ValueError("reset_lift_axis must be a tuple of three finite values")
        normalized_axis = tuple(float(value) for value in axis)
        norm = math.sqrt(sum(value * value for value in normalized_axis))
        if not math.isclose(norm, 1.0, rel_tol=0.0, abs_tol=1.0e-5):
            raise ValueError("reset_lift_axis must be unit length")
        object.__setattr__(self, "reset_lift_axis", normalized_axis)

    @classmethod
    def from_trajectory(
        cls,
        trajectory: Stage0Trajectory,
        *,
        mode_id: str,
        mode_sha256: str,
        reset_lift_axis: tuple[float, float, float],
        implementation_sha256: str | None = None,
    ) -> GraspLiftEvidenceMetadata:
        """Build metadata from the immutable trajectory commitments."""
        if not isinstance(trajectory, Stage0Trajectory):
            raise TypeError("trajectory must be Stage0Trajectory")
        provenance = trajectory.provenance
        return cls(
            trajectory_id=provenance.trajectory_id,
            trajectory_sha256=trajectory_sha256(trajectory),
            mode_id=mode_id,
            mode_sha256=mode_sha256,
            action_schema_id=GRASP_LIFT_ACTION_SCHEMA_ID,
            action_schema_sha256=provenance.action_schema_sha256,
            state_view_id=GRASP_LIFT_STATE_VIEW_ID,
            state_view_sha256=DEFAULT_GRASP_LIFT_STATE_VIEW.sha256,
            implementation_sha256=(
                provenance.simulator_sha256
                if implementation_sha256 is None
                else implementation_sha256
            ),
            experiment_sha256=provenance.experiment_sha256,
            reset_seed=provenance.reset_seed,
            world=provenance.world,
            reset_lift_axis=reset_lift_axis,
            transition_count=trajectory.transition_count,
        )

    def to_record(self) -> dict[str, Any]:
        return {
            "action_schema_id": self.action_schema_id,
            "action_schema_sha256": self.action_schema_sha256,
            "experiment_sha256": self.experiment_sha256,
            "implementation_sha256": self.implementation_sha256,
            "mode_id": self.mode_id,
            "mode_sha256": self.mode_sha256,
            "reset_lift_axis": list(self.reset_lift_axis),
            "reset_seed": self.reset_seed,
            "state_view_id": self.state_view_id,
            "state_view_sha256": self.state_view_sha256,
            "trajectory_id": self.trajectory_id,
            "trajectory_sha256": self.trajectory_sha256,
            "transition_count": self.transition_count,
            "world": self.world,
        }

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> GraspLiftEvidenceMetadata:
        if not isinstance(record, Mapping):
            raise ValueError("evidence metadata must be a mapping")
        expected = {
            "action_schema_id",
            "action_schema_sha256",
            "experiment_sha256",
            "implementation_sha256",
            "mode_id",
            "mode_sha256",
            "reset_lift_axis",
            "reset_seed",
            "state_view_id",
            "state_view_sha256",
            "trajectory_id",
            "trajectory_sha256",
            "transition_count",
            "world",
        }
        _require_exact_keys(record, expected, context="Grasp-Lift evidence metadata")
        axis = record["reset_lift_axis"]
        if not isinstance(axis, list):
            raise ValueError("reset_lift_axis record must be a JSON array")
        values = {key: record[key] for key in expected if key != "reset_lift_axis"}
        return cls(reset_lift_axis=tuple(axis), **values)


def _materialize_evidence_tensors(
    tensors: Mapping[str, Any], *, transition_count: int
) -> Mapping[str, Any]:
    import torch

    if not isinstance(tensors, Mapping):
        raise TypeError("tensors must be a mapping")
    _require_exact_keys(
        tensors,
        set(GRASP_LIFT_EVIDENCE_TENSOR_SPECS),
        context="Grasp-Lift evidence tensors",
    )
    expected_dtype = {
        "float32": torch.float32,
        "bool": torch.bool,
        "int64": torch.int64,
    }
    materialized: dict[str, Any] = {}
    for name, spec in GRASP_LIFT_EVIDENCE_TENSOR_SPECS.items():
        value = tensors[name]
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"tensors.{name} must be a Torch tensor")
        if value.device.type != "cpu":
            raise ValueError(f"tensors.{name} must already be on CPU")
        if value.layout is not torch.strided:
            raise ValueError(f"tensors.{name} must use strided layout")
        if value.dtype != expected_dtype[spec.dtype]:
            raise ValueError(
                f"tensors.{name} must use {spec.dtype}, got {value.dtype}"
            )
        expected_shape = spec.shape(transition_count)
        if tuple(value.shape) != expected_shape:
            raise ValueError(
                f"tensors.{name} must have shape {expected_shape}, "
                f"got {tuple(value.shape)}"
            )
        if value.dtype == torch.float32 and not bool(torch.isfinite(value).all()):
            raise ValueError(f"tensors.{name} contains non-finite values")
        if value.dtype == torch.int64 and bool((value < 0).any()):
            raise ValueError(f"tensors.{name} contains a negative count/value")
        materialized[name] = value.detach().contiguous().clone()
    return MappingProxyType(materialized)


def _is_monotonic_latch(value: Any) -> bool:
    if value.numel() < 2:
        return True
    return not bool((value[:-1] & ~value[1:]).any())


@dataclass(frozen=True)
class GraspLiftEvidence:
    """Strict CPU tensor evidence for one complete Grasp-Lift trajectory."""

    metadata: GraspLiftEvidenceMetadata
    tensors: Mapping[str, Any]

    def __post_init__(self) -> None:
        import torch

        if not isinstance(self.metadata, GraspLiftEvidenceMetadata):
            raise TypeError("metadata must be GraspLiftEvidenceMetadata")
        tensors = _materialize_evidence_tensors(
            self.tensors,
            transition_count=self.metadata.transition_count,
        )
        object.__setattr__(self, "tensors", tensors)

        if bool(torch.logical_and(tensors["oracle_success"], tensors["oracle_failure"]).any()):
            raise ValueError("oracle_success and oracle_failure must be exclusive")
        for name in (
            "grasp_confirmed",
            "oracle_success",
            "oracle_failure",
            "forbidden_contact_violation",
            "lateral_displacement_violation",
            "bottle_tilt_violation",
            "dropped_grasp_violation",
            "collision_buffer_overflow_violation",
        ):
            if not _is_monotonic_latch(tensors[name]):
                raise ValueError(f"tensors.{name} is not latched monotonically")
        observed_overflow = torch.logical_or(
            tensors["rigid_contact_overflow_frame_count"] > 0,
            tensors["triangle_pair_overflow_frame_count"] > 0,
        )
        if not torch.equal(tensors["collision_buffer_overflow"], observed_overflow):
            raise ValueError(
                "collision_buffer_overflow differs from collision diagnostics"
            )

    @property
    def transition_count(self) -> int:
        return self.metadata.transition_count

    @property
    def executed_action(self) -> Any:
        return self.tensors["executed_action"]

    def tensor_contract_record(self) -> dict[str, dict[str, Any]]:
        return {
            name: {
                "dtype": spec.dtype,
                "shape": list(spec.shape(self.transition_count)),
            }
            for name, spec in GRASP_LIFT_EVIDENCE_TENSOR_SPECS.items()
        }


def grasp_lift_evidence_sha256(evidence: GraspLiftEvidence) -> str:
    """Hash metadata plus canonical tensor names, shapes, dtypes, and bytes."""
    if not isinstance(evidence, GraspLiftEvidence):
        raise TypeError("evidence must be GraspLiftEvidence")
    digest = hashlib.sha256()
    digest.update(GRASP_LIFT_EVIDENCE_SCHEMA_ID.encode("utf-8"))
    digest.update(_canonical_json(evidence.metadata.to_record()))
    for name in sorted(evidence.tensors):
        tensor = evidence.tensors[name]
        digest.update(name.encode("ascii"))
        digest.update(str(tensor.dtype).removeprefix("torch.").encode("ascii"))
        digest.update(_canonical_json(list(tensor.shape)))
        payload = tensor.numpy().tobytes(order="C")
        digest.update(len(payload).to_bytes(8, byteorder="big", signed=False))
        digest.update(payload)
    return digest.hexdigest()


def validate_grasp_lift_evidence_trajectory(
    evidence: GraspLiftEvidence,
    trajectory: Stage0Trajectory,
) -> None:
    """Fail closed unless evidence is exactly aligned to ``trajectory``."""
    import torch

    if not isinstance(evidence, GraspLiftEvidence):
        raise TypeError("evidence must be GraspLiftEvidence")
    if not isinstance(trajectory, Stage0Trajectory):
        raise TypeError("trajectory must be Stage0Trajectory")
    metadata = evidence.metadata
    provenance = trajectory.provenance
    commitments = {
        "trajectory_id": (metadata.trajectory_id, provenance.trajectory_id),
        "trajectory_sha256": (metadata.trajectory_sha256, trajectory_sha256(trajectory)),
        "action_schema_sha256": (
            metadata.action_schema_sha256,
            provenance.action_schema_sha256,
        ),
        "experiment_sha256": (
            metadata.experiment_sha256,
            provenance.experiment_sha256,
        ),
        "reset_seed": (metadata.reset_seed, provenance.reset_seed),
        "world": (metadata.world, provenance.world),
        "transition_count": (metadata.transition_count, trajectory.transition_count),
    }
    for name, (actual, expected) in commitments.items():
        if actual != expected:
            raise ValueError(
                f"evidence {name} differs from trajectory: "
                f"{actual!r} != {expected!r}"
            )
    if trajectory.action_dim != GRASP_LIFT_ACTION_DIM:
        raise ValueError("trajectory does not use the 19D Grasp-Lift action")
    if not torch.equal(evidence.tensors["executed_action"], trajectory.actions):
        raise ValueError("evidence executed_action differs from trajectory.actions")

    post_states = trajectory.states[1:]
    expected_eef = post_states[:, DEFAULT_STAGE0_STATE_SCHEMA.slice("eef_position")]
    expected_hand = post_states[
        :, DEFAULT_STAGE0_STATE_SCHEMA.slice("hand_joint_position")
    ]
    if not torch.equal(evidence.tensors["measured_eef_position"], expected_eef):
        raise ValueError("measured_eef_position is not state[t + 1]")
    if not torch.equal(
        evidence.tensors["measured_hand_joint_position"], expected_hand
    ):
        raise ValueError("measured_hand_joint_position is not state[t + 1]")

    initial_goal = trajectory.states[
        0, DEFAULT_STAGE0_STATE_SCHEMA.slice("object_to_goal")
    ]
    norm = torch.linalg.vector_norm(initial_goal)
    if not bool(torch.isfinite(norm)) or float(norm) <= torch.finfo(torch.float32).eps:
        raise ValueError("trajectory reset object_to_goal does not define a lift axis")
    expected_axis = initial_goal / norm
    recorded_axis = torch.tensor(metadata.reset_lift_axis, dtype=torch.float32)
    if not torch.allclose(recorded_axis, expected_axis, rtol=0.0, atol=1.0e-6):
        raise ValueError("reset_lift_axis differs from the trajectory reset state")


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _file_sha256(path: Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _temporary_path(target: Path) -> Path:
    return target.with_name(f".{target.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}")


def _publish_immutable(temporary: Path, target: Path) -> None:
    try:
        os.link(temporary, target)
    except FileExistsError as exc:
        raise FileExistsError(f"immutable Grasp-Lift evidence exists: {target}") from exc
    finally:
        temporary.unlink(missing_ok=True)
    _fsync_directory(target.parent)


def _write_torch_temporary(path: Path, payload: Mapping[str, Any]) -> None:
    import torch

    with path.open("xb") as file:
        torch.save(dict(payload), file)
        file.flush()
        os.fsync(file.fileno())


def _write_json_temporary(path: Path, record: Mapping[str, Any]) -> None:
    encoded = (
        json.dumps(
            record,
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    with path.open("xb") as file:
        file.write(encoded)
        file.flush()
        os.fsync(file.fileno())


def write_grasp_lift_evidence_sidecar(
    directory: str | Path,
    evidence: GraspLiftEvidence,
    *,
    trajectory: Stage0Trajectory | None = None,
) -> Path:
    """Atomically publish payload then descriptor without replacing either."""
    if not isinstance(evidence, GraspLiftEvidence):
        raise TypeError("evidence must be GraspLiftEvidence")
    if trajectory is not None:
        validate_grasp_lift_evidence_trajectory(evidence, trajectory)
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    trajectory_id = evidence.metadata.trajectory_id
    payload_path = root / f"{trajectory_id}{_PAYLOAD_SUFFIX}"
    descriptor_path = root / f"{trajectory_id}{_DESCRIPTOR_SUFFIX}"
    if payload_path.exists() or descriptor_path.exists():
        raise FileExistsError(
            f"immutable Grasp-Lift evidence already exists for {trajectory_id}"
        )

    semantic_sha256 = grasp_lift_evidence_sha256(evidence)
    payload = {
        "evidence_sha256": semantic_sha256,
        "metadata": evidence.metadata.to_record(),
        "schema_id": GRASP_LIFT_EVIDENCE_SCHEMA_ID,
        "schema_version": GRASP_LIFT_EVIDENCE_SCHEMA_VERSION,
        "tensors": dict(evidence.tensors),
    }
    payload_temporary = _temporary_path(payload_path)
    descriptor_temporary = _temporary_path(descriptor_path)
    payload_published = False
    try:
        _write_torch_temporary(payload_temporary, payload)
        payload_sha256 = _file_sha256(payload_temporary)
        descriptor = {
            "descriptor_schema_id": GRASP_LIFT_EVIDENCE_DESCRIPTOR_SCHEMA_ID,
            "descriptor_schema_version": (
                GRASP_LIFT_EVIDENCE_DESCRIPTOR_SCHEMA_VERSION
            ),
            "evidence_schema_id": GRASP_LIFT_EVIDENCE_SCHEMA_ID,
            "evidence_schema_version": GRASP_LIFT_EVIDENCE_SCHEMA_VERSION,
            "evidence_sha256": semantic_sha256,
            "metadata": evidence.metadata.to_record(),
            "payload_bytes": payload_temporary.stat().st_size,
            "payload_file": payload_path.name,
            "payload_sha256": payload_sha256,
            "tensors": evidence.tensor_contract_record(),
        }
        _write_json_temporary(descriptor_temporary, descriptor)
        _publish_immutable(payload_temporary, payload_path)
        payload_published = True
        _publish_immutable(descriptor_temporary, descriptor_path)
    except Exception:
        payload_temporary.unlink(missing_ok=True)
        descriptor_temporary.unlink(missing_ok=True)
        if payload_published and not descriptor_path.exists():
            payload_path.unlink(missing_ok=True)
            _fsync_directory(root)
        raise
    return descriptor_path


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _parse_descriptor(path: Path) -> dict[str, Any]:
    try:
        record = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"invalid JSON constant: {value}")
            ),
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise GraspLiftEvidenceIntegrityError(
            f"cannot parse Grasp-Lift evidence descriptor {path}: {exc}"
        ) from exc
    if not isinstance(record, dict):
        raise GraspLiftEvidenceIntegrityError("evidence descriptor must be an object")
    expected = {
        "descriptor_schema_id",
        "descriptor_schema_version",
        "evidence_schema_id",
        "evidence_schema_version",
        "evidence_sha256",
        "metadata",
        "payload_bytes",
        "payload_file",
        "payload_sha256",
        "tensors",
    }
    try:
        _require_exact_keys(record, expected, context="Grasp-Lift evidence descriptor")
    except ValueError as exc:
        raise GraspLiftEvidenceIntegrityError(str(exc)) from exc
    if (
        record["descriptor_schema_id"]
        != GRASP_LIFT_EVIDENCE_DESCRIPTOR_SCHEMA_ID
        or record["descriptor_schema_version"]
        != GRASP_LIFT_EVIDENCE_DESCRIPTOR_SCHEMA_VERSION
    ):
        raise GraspLiftEvidenceIntegrityError("unsupported evidence descriptor schema")
    if (
        record["evidence_schema_id"] != GRASP_LIFT_EVIDENCE_SCHEMA_ID
        or record["evidence_schema_version"] != GRASP_LIFT_EVIDENCE_SCHEMA_VERSION
    ):
        raise GraspLiftEvidenceIntegrityError("unsupported evidence payload schema")
    try:
        _require_sha256(record["evidence_sha256"], "evidence_sha256")
        _require_sha256(record["payload_sha256"], "payload_sha256")
    except ValueError as exc:
        raise GraspLiftEvidenceIntegrityError(str(exc)) from exc
    return record


def _validate_descriptor_tensors(
    record: Any, evidence: GraspLiftEvidence
) -> None:
    if not isinstance(record, Mapping):
        raise GraspLiftEvidenceIntegrityError("descriptor tensors must be an object")
    expected = evidence.tensor_contract_record()
    if set(record) != set(expected):
        raise GraspLiftEvidenceIntegrityError("descriptor tensor names changed")
    for name, contract in expected.items():
        actual = record[name]
        if not isinstance(actual, Mapping):
            raise GraspLiftEvidenceIntegrityError(
                f"descriptor tensors.{name} must be an object"
            )
        try:
            _require_exact_keys(actual, {"dtype", "shape"}, context=f"tensors.{name}")
        except ValueError as exc:
            raise GraspLiftEvidenceIntegrityError(str(exc)) from exc
        if dict(actual) != contract:
            raise GraspLiftEvidenceIntegrityError(
                f"descriptor tensor contract changed for {name}"
            )


def load_grasp_lift_evidence_sidecar(
    descriptor_path: str | Path,
    *,
    trajectory: Stage0Trajectory | None = None,
) -> GraspLiftEvidence:
    """Verify checksums and contracts before returning CPU-only tensors."""
    import torch

    path = Path(descriptor_path)
    if path.is_symlink():
        raise GraspLiftEvidenceIntegrityError("evidence descriptor cannot be a symlink")
    descriptor = _parse_descriptor(path)
    payload_file = descriptor["payload_file"]
    if not isinstance(payload_file, str) or Path(payload_file).name != payload_file:
        raise GraspLiftEvidenceIntegrityError("payload_file must be a local basename")
    if not path.name.endswith(_DESCRIPTOR_SUFFIX):
        raise GraspLiftEvidenceIntegrityError("evidence descriptor suffix changed")
    expected_name = path.name.removesuffix(_DESCRIPTOR_SUFFIX) + _PAYLOAD_SUFFIX
    if payload_file != expected_name:
        raise GraspLiftEvidenceIntegrityError(
            "payload_file does not match evidence descriptor basename"
        )
    payload_path = path.parent / payload_file
    if payload_path.is_symlink() or not payload_path.is_file():
        raise GraspLiftEvidenceIntegrityError("evidence payload is missing or a symlink")
    payload_bytes = descriptor["payload_bytes"]
    if (
        isinstance(payload_bytes, bool)
        or not isinstance(payload_bytes, int)
        or payload_bytes < 1
        or payload_path.stat().st_size != payload_bytes
    ):
        raise GraspLiftEvidenceIntegrityError("evidence payload size changed")
    if _file_sha256(payload_path) != descriptor["payload_sha256"]:
        raise GraspLiftEvidenceIntegrityError("evidence payload SHA-256 mismatch")

    try:
        payload = torch.load(payload_path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise GraspLiftEvidenceIntegrityError(
            f"cannot safely load evidence payload: {exc}"
        ) from exc
    if not isinstance(payload, Mapping):
        raise GraspLiftEvidenceIntegrityError("evidence payload must be a mapping")
    expected_payload = {
        "evidence_sha256",
        "metadata",
        "schema_id",
        "schema_version",
        "tensors",
    }
    try:
        _require_exact_keys(payload, expected_payload, context="Grasp-Lift evidence payload")
    except ValueError as exc:
        raise GraspLiftEvidenceIntegrityError(str(exc)) from exc
    if (
        payload["schema_id"] != descriptor["evidence_schema_id"]
        or payload["schema_version"] != descriptor["evidence_schema_version"]
    ):
        raise GraspLiftEvidenceIntegrityError("payload evidence schema changed")
    if payload["evidence_sha256"] != descriptor["evidence_sha256"]:
        raise GraspLiftEvidenceIntegrityError("payload evidence hash changed")
    if payload["metadata"] != descriptor["metadata"]:
        raise GraspLiftEvidenceIntegrityError("payload metadata differs from descriptor")
    try:
        metadata = GraspLiftEvidenceMetadata.from_record(payload["metadata"])
        evidence = GraspLiftEvidence(metadata=metadata, tensors=payload["tensors"])
    except (TypeError, ValueError) as exc:
        raise GraspLiftEvidenceIntegrityError(
            f"invalid evidence payload contract: {exc}"
        ) from exc
    _validate_descriptor_tensors(descriptor["tensors"], evidence)
    if grasp_lift_evidence_sha256(evidence) != descriptor["evidence_sha256"]:
        raise GraspLiftEvidenceIntegrityError("evidence semantic SHA-256 mismatch")
    if trajectory is not None:
        try:
            validate_grasp_lift_evidence_trajectory(evidence, trajectory)
        except (TypeError, ValueError) as exc:
            raise GraspLiftEvidenceIntegrityError(
                f"evidence/trajectory binding failed: {exc}"
            ) from exc
    return evidence


def grasp_lift_composite_corpus_sha256(
    members: Iterable[tuple[Stage0Trajectory, GraspLiftEvidence]],
) -> str:
    """Order-independent hash binding every trajectory to its evidence."""
    identities: list[dict[str, str]] = []
    seen: set[str] = set()
    for member in members:
        if not isinstance(member, tuple) or len(member) != 2:
            raise TypeError("each corpus member must be (trajectory, evidence)")
        trajectory, evidence = member
        validate_grasp_lift_evidence_trajectory(evidence, trajectory)
        trajectory_id = trajectory.provenance.trajectory_id
        if trajectory_id in seen:
            raise ValueError(f"duplicate trajectory_id in corpus: {trajectory_id}")
        seen.add(trajectory_id)
        identities.append(
            {
                "evidence_sha256": grasp_lift_evidence_sha256(evidence),
                "trajectory_id": trajectory_id,
                "trajectory_sha256": trajectory_sha256(trajectory),
            }
        )
    if not identities:
        raise ValueError("a Grasp-Lift composite corpus cannot be empty")
    record = {
        "members": sorted(identities, key=lambda value: value["trajectory_id"]),
        "schema_id": GRASP_LIFT_COMPOSITE_CORPUS_SCHEMA_ID,
    }
    return hashlib.sha256(_canonical_json(record)).hexdigest()


__all__ = [
    "GRASP_LIFT_COMPOSITE_CORPUS_SCHEMA_ID",
    "GRASP_LIFT_EVIDENCE_DESCRIPTOR_SCHEMA_ID",
    "GRASP_LIFT_EVIDENCE_DESCRIPTOR_SCHEMA_VERSION",
    "GRASP_LIFT_EVIDENCE_SCHEMA_ID",
    "GRASP_LIFT_EVIDENCE_SCHEMA_VERSION",
    "GRASP_LIFT_EVIDENCE_TENSOR_SPECS",
    "GraspLiftEvidence",
    "GraspLiftEvidenceError",
    "GraspLiftEvidenceIntegrityError",
    "GraspLiftEvidenceMetadata",
    "grasp_lift_composite_corpus_sha256",
    "grasp_lift_evidence_sha256",
    "load_grasp_lift_evidence_sidecar",
    "validate_grasp_lift_evidence_trajectory",
    "write_grasp_lift_evidence_sidecar",
]
