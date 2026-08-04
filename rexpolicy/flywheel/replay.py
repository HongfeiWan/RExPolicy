"""Deterministic replay validation without archiving full simulator state."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Iterator

import numpy as np

from rexpolicy.tasking.dynamics_contract import (
    DYNAMICS_CONTRACT_V1_ID,
    DYNAMICS_CONTRACT_V1_SHA256,
)


REPLAY_FINGERPRINT_SCHEMA_VERSION = 2
REPLAY_FINGERPRINT_POLICY_ID = "rexpolicy/replay-fingerprint/v2"
PHYSICAL_REPLAY_GATE_SCHEMA_VERSION = 2
PHYSICAL_REPLAY_GATE_POLICY_ID = (
    "groot_newton/physical-replay-gate-tolerance-witness/v2"
)
FLOAT_QUANTIZATION_POLICY_ID = "float64-rint-nearest-even-fixed-quantum/v1"
FLOAT_QUANTIZATION_STEP = 1.0e-5
SAME_STATE_COMPARISON_POLICY_ID = "world-zero-max-abs-per-field/v1"
HISTORICAL_REPLAY_COMPARISON_POLICY_ID = (
    "archived-current-sampled-linf/v1"
)
CONTINUOUS_WITNESS_SCHEMA_VERSION = 1
CONTINUOUS_WITNESS_POLICY_ID = "flattened-even-samples-linf/v1"
CONTINUOUS_WITNESS_MAX_SAMPLES_PER_FIELD = 32
OBSERVATION_RENDER_PROVENANCE_SCHEMA_VERSION = 1
OBSERVATION_RENDER_PROVENANCE_POLICY_ID = (
    "groot_newton/render-contract-no-pixel-equality/v1"
)

# These discrete values affect task progression, termination, or safety.  The
# physical/control tensors live in the separate reward-free dynamics tree.
PHYSICAL_REPLAY_GATE_TASK_FIELDS = (
    "task_phase",
    "reach_success_hold_steps",
    "grasp_contact_frames",
    "grasp_support_gap_frames",
    "contact_gap_frames",
    "settle_frames",
    "grasp_confirmed",
    "transport_started",
    "reached_lift_height",
    "release_armed",
    "released",
    "early_release",
    "success",
    "fail",
    "episode_step",
    "success_once",
    "terminated",
    "truncated",
)


@dataclass(frozen=True)
class ContinuousFieldWitness:
    """Deterministic sampled values for one continuous state field."""

    name: str
    shape: tuple[int, ...]
    dtype: str
    element_count: int
    sample_indices: tuple[int, ...]
    sample_values: tuple[float, ...]

    def to_record(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "shape": list(self.shape),
            "dtype": self.dtype,
            "element_count": self.element_count,
            "sample_indices": list(self.sample_indices),
            "sample_values": list(self.sample_values),
        }


@dataclass(frozen=True)
class ReplayFingerprint:
    """Comparison metrics and an exact commitment to quantized world zero.

    ``digest`` is an audit-only exact commitment, not an absolute-tolerance
    equivalence relation.  Historical admission uses ``continuous_witness``;
    floating values also pass an independent per-field cross-world tolerance
    check before any record is produced.
    """

    digest: str
    max_float_spread: float
    field_count: int
    schema_version: int
    policy_id: str
    float_quantization_policy_id: str
    float_quantization_step: float
    field_schema_sha256: str
    discrete_state_sha256: str
    continuous_witness: tuple[ContinuousFieldWitness, ...]


class HistoricalReplayGateMismatch(RuntimeError):
    """A historical archive is incompatible with the current replay gate."""


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
    float_quantization_step: float = FLOAT_QUANTIZATION_STEP,
    digest_policy_id: str = REPLAY_FINGERPRINT_POLICY_ID,
) -> ReplayFingerprint:
    """Validate same-state branches and commit to quantized world zero.

    ``float_tolerance`` applies only to the in-memory, per-field comparison of
    candidate worlds.  The audit-only digest uses
    ``float_quantization_step`` as a fixed binning commitment; two values
    within the tolerance can straddle a bin boundary and therefore have
    different digests.
    """
    if world_count < 1:
        raise ValueError("world_count must be positive")
    _require_positive_finite(float_tolerance, "float_tolerance")
    _require_positive_finite(
        float_quantization_step,
        "float_quantization_step",
    )
    digest = _new_digest(
        digest_policy_id=digest_policy_id,
        float_quantization_step=float_quantization_step,
    )
    discrete_digest = hashlib.sha256(b"rexpolicy-discrete-state/v1\x00")
    field_schema: list[dict[str, Any]] = []
    continuous_witness: list[ContinuousFieldWitness] = []
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
        is_float = np.issubdtype(array.dtype, np.floating)
        field_schema.append(
            {
                "name": name,
                "shape": list(reference.shape),
                "dtype": array.dtype.str,
                "value_class": "continuous" if is_float else "discrete",
            }
        )
        if is_float:
            if not np.isfinite(array).all():
                raise FloatingPointError(
                    f"Replay fingerprint field {name!r} is non-finite"
                )
            spread = (
                float(np.max(np.abs(array - reference)))
                if array.size
                else 0.0
            )
            maximum = max(maximum, spread)
            if spread > float_tolerance:
                raise RuntimeError(
                    f"Candidate replay field {name!r} diverged: "
                    f"{spread:.3e} > {float_tolerance:.3e}"
                )
            quantized = _quantize_float(reference, float_quantization_step)
            continuous_witness.append(
                _continuous_field_witness(
                    name=name,
                    value=reference,
                )
            )
            _update_digest(
                digest,
                name=name,
                shape=reference.shape,
                encoding=FLOAT_QUANTIZATION_POLICY_ID,
                payload=quantized.tobytes(order="C"),
            )
        else:
            if not np.all(array == reference):
                raise RuntimeError(
                    f"Candidate replay field {name!r} is not exactly equal"
                )
            contiguous = np.ascontiguousarray(reference)
            _update_digest(
                digest,
                name=name,
                shape=reference.shape,
                encoding=f"exact:{contiguous.dtype.str}",
                payload=contiguous.tobytes(order="C"),
            )
            _update_digest(
                discrete_digest,
                name=name,
                shape=reference.shape,
                encoding=f"exact:{contiguous.dtype.str}",
                payload=contiguous.tobytes(order="C"),
            )
    if fields == 0:
        raise ValueError("Replay fingerprint tree is empty")
    return ReplayFingerprint(
        digest=digest.hexdigest(),
        max_float_spread=maximum,
        field_count=fields,
        schema_version=REPLAY_FINGERPRINT_SCHEMA_VERSION,
        policy_id=digest_policy_id,
        float_quantization_policy_id=FLOAT_QUANTIZATION_POLICY_ID,
        float_quantization_step=float_quantization_step,
        field_schema_sha256=_canonical_sha256(
            field_schema,
            "field_schema",
        ),
        discrete_state_sha256=discrete_digest.hexdigest(),
        continuous_witness=tuple(continuous_witness),
    )


def fingerprint_world_zero(
    tree: dict[str, Any],
    *,
    world_count: int,
    float_tolerance: float,
    float_quantization_step: float = FLOAT_QUANTIZATION_STEP,
    digest_policy_id: str = REPLAY_FINGERPRINT_POLICY_ID,
) -> str:
    """Return the same world-zero digest while still validating all branches."""
    return validate_replay_fingerprint(
        tree,
        world_count=world_count,
        float_tolerance=float_tolerance,
        float_quantization_step=float_quantization_step,
        digest_policy_id=digest_policy_id,
    ).digest


def fingerprint_each_world(
    tree: dict[str, Any],
    *,
    world_count: int,
    float_tolerance: float,
    float_quantization_step: float = FLOAT_QUANTIZATION_STEP,
    digest_policy_id: str = REPLAY_FINGERPRINT_POLICY_ID,
) -> tuple[str, ...]:
    """Commit to each branch using the same fixed quantization encoding."""
    if world_count < 1:
        raise ValueError("world_count must be positive")
    _require_positive_finite(float_tolerance, "float_tolerance")
    _require_positive_finite(
        float_quantization_step,
        "float_quantization_step",
    )
    digests = [
        _new_digest(
            digest_policy_id=digest_policy_id,
            float_quantization_step=float_quantization_step,
        )
        for _ in range(world_count)
    ]
    fields = 0
    for name, value in _flatten(tree):
        array = _to_numpy(value)
        if array.ndim < 1 or int(array.shape[0]) != world_count:
            raise ValueError(
                f"Replay fingerprint field {name!r} must start with "
                f"world dimension {world_count}, got {array.shape}"
            )
        fields += 1
        if np.issubdtype(array.dtype, np.floating) and not np.isfinite(array).all():
            raise FloatingPointError(
                f"Replay fingerprint field {name!r} is non-finite"
            )
        for world, digest in enumerate(digests):
            row = array[world]
            if np.issubdtype(array.dtype, np.floating):
                quantized = _quantize_float(row, float_quantization_step)
                _update_digest(
                    digest,
                    name=name,
                    shape=row.shape,
                    encoding=FLOAT_QUANTIZATION_POLICY_ID,
                    payload=quantized.tobytes(order="C"),
                )
            else:
                contiguous = np.ascontiguousarray(row)
                _update_digest(
                    digest,
                    name=name,
                    shape=row.shape,
                    encoding=f"exact:{contiguous.dtype.str}",
                    payload=contiguous.tobytes(order="C"),
                )
    if fields == 0:
        raise ValueError("Replay fingerprint tree is empty")
    return tuple(digest.hexdigest() for digest in digests)


def physical_replay_gate_tree(
    *,
    dynamics: dict[str, Any],
    replay: dict[str, Any],
) -> dict[str, Any]:
    """Select reward-free physics plus safety/task discrete state.

    RGB tensors and reward-derived floats are intentionally outside this
    exact gate.  The complete contact group is retained so a newly surfaced
    contact/safety signal cannot be silently ignored.
    """
    if not isinstance(dynamics, dict) or not dynamics:
        raise ValueError("Physical replay gate requires a dynamics tree")
    task = replay.get("task")
    contacts = replay.get("contacts")
    if not isinstance(task, dict) or not isinstance(contacts, dict):
        raise ValueError("Physical replay gate requires task and contact state")
    missing = [
        name for name in PHYSICAL_REPLAY_GATE_TASK_FIELDS if name not in task
    ]
    if missing:
        raise ValueError(
            "Physical replay gate is missing task fields: " + ", ".join(missing)
        )
    if not contacts:
        raise ValueError("Physical replay gate contact state is empty")
    return {
        "dynamics": dynamics,
        "discrete": {
            "task": {
                name: task[name]
                for name in PHYSICAL_REPLAY_GATE_TASK_FIELDS
            },
            "contacts": contacts,
        },
    }


def validate_physical_replay_gate(
    tree: dict[str, Any],
    *,
    world_count: int,
    float_tolerance: float,
) -> ReplayFingerprint:
    """Validate the production historical replay gate contract."""
    return validate_replay_fingerprint(
        tree,
        world_count=world_count,
        float_tolerance=float_tolerance,
        float_quantization_step=FLOAT_QUANTIZATION_STEP,
        digest_policy_id=PHYSICAL_REPLAY_GATE_POLICY_ID,
    )


def build_physical_replay_gate_record(
    fingerprint: ReplayFingerprint,
    *,
    same_state_tolerance: float,
    historical_replay_state_tolerance: float,
    simulator_fingerprint_sha256: str,
    observation_contract_sha256: str,
    render_contract: dict[str, Any],
) -> dict[str, Any]:
    """Build the versioned archive record without committing to RGB bytes."""
    _require_physical_fingerprint(fingerprint)
    _require_positive_finite(same_state_tolerance, "same_state_tolerance")
    _require_positive_finite(
        historical_replay_state_tolerance,
        "historical_replay_state_tolerance",
    )
    if historical_replay_state_tolerance < same_state_tolerance:
        raise ValueError(
            "historical_replay_state_tolerance must be greater than or "
            "equal to same_state_tolerance"
        )
    _require_sha256(
        simulator_fingerprint_sha256,
        "simulator_fingerprint_sha256",
    )
    _require_sha256(
        observation_contract_sha256,
        "observation_contract_sha256",
    )
    render_sha256 = _canonical_sha256(render_contract, "render_contract")
    return {
        "schema_version": PHYSICAL_REPLAY_GATE_SCHEMA_VERSION,
        "policy_id": PHYSICAL_REPLAY_GATE_POLICY_ID,
        "audit_quantized_digest_sha256": fingerprint.digest,
        "audit_quantized_digest_acceptance_role": "none",
        "field_count": fingerprint.field_count,
        "field_schema_sha256": fingerprint.field_schema_sha256,
        "discrete_state_sha256": fingerprint.discrete_state_sha256,
        "tree_contract": {
            "dynamics_contract_id": DYNAMICS_CONTRACT_V1_ID,
            "dynamics_contract_sha256": DYNAMICS_CONTRACT_V1_SHA256,
            "discrete_task_fields": list(PHYSICAL_REPLAY_GATE_TASK_FIELDS),
            "contact_fields": "all_replay_contact_fields",
            "rgb_fields": "excluded",
            "reward_derived_fields": "excluded",
        },
        "physical_provenance": {
            "simulator_fingerprint_sha256": simulator_fingerprint_sha256,
            "dynamics_contract_id": DYNAMICS_CONTRACT_V1_ID,
            "dynamics_contract_sha256": DYNAMICS_CONTRACT_V1_SHA256,
        },
        "float_quantization": {
            "policy_id": FLOAT_QUANTIZATION_POLICY_ID,
            "step": FLOAT_QUANTIZATION_STEP,
            "tolerance_equivalence_asserted": False,
        },
        "continuous_witness": {
            "schema_version": CONTINUOUS_WITNESS_SCHEMA_VERSION,
            "policy_id": CONTINUOUS_WITNESS_POLICY_ID,
            "max_samples_per_field": (
                CONTINUOUS_WITNESS_MAX_SAMPLES_PER_FIELD
            ),
            "selection": (
                "all flattened C-order coordinates when field size <= 32; "
                "otherwise 32 evenly spaced flattened coordinates including "
                "both endpoints"
            ),
            "acceptance": (
                "for every sampled coordinate, abs(archived-current) <= "
                "historical_replay_state_tolerance + ulp(archived) + "
                "ulp(current) + ulp(historical_replay_state_tolerance)"
            ),
            "detection_guarantee": (
                "complete for sampled coordinates; unsampled coordinate-only "
                "changes are not guaranteed to be detected"
            ),
            "fields": [
                witness.to_record()
                for witness in fingerprint.continuous_witness
            ],
        },
        "same_state_validation": {
            "policy_id": SAME_STATE_COMPARISON_POLICY_ID,
            "float_tolerance": same_state_tolerance,
            "max_float_spread": fingerprint.max_float_spread,
            "discrete_equality": "exact",
        },
        "historical_replay_validation": {
            "policy_id": HISTORICAL_REPLAY_COMPARISON_POLICY_ID,
            "continuous_float_tolerance": (
                historical_replay_state_tolerance
            ),
            "discrete_equality": "exact",
            "schema_equality": "exact",
            "provenance_equality": "exact",
        },
        "observation_render_provenance": {
            "schema_version": OBSERVATION_RENDER_PROVENANCE_SCHEMA_VERSION,
            "policy_id": OBSERVATION_RENDER_PROVENANCE_POLICY_ID,
            "observation_contract_sha256": observation_contract_sha256,
            "render_contract": render_contract,
            "render_contract_sha256": render_sha256,
            "pixel_equality_asserted": False,
        },
    }


def validate_historical_physical_replay_gate(
    archived: Any,
    *,
    current_fingerprint: ReplayFingerprint,
    same_state_tolerance: float,
    historical_replay_state_tolerance: float,
    simulator_fingerprint_sha256: str,
    observation_contract_sha256: str,
    render_contract: dict[str, Any],
) -> None:
    """Fail closed unless an archive uses and matches the current v2 gate."""
    if not isinstance(archived, dict):
        raise HistoricalReplayGateMismatch(
            "Historical archive is missing physical replay gate schema v2"
        )
    if archived.get("schema_version") != PHYSICAL_REPLAY_GATE_SCHEMA_VERSION:
        raise HistoricalReplayGateMismatch(
            "Historical physical replay gate schema is not v2; explicit "
            "migration is required"
        )
    current = build_physical_replay_gate_record(
        current_fingerprint,
        same_state_tolerance=same_state_tolerance,
        historical_replay_state_tolerance=(
            historical_replay_state_tolerance
        ),
        simulator_fingerprint_sha256=simulator_fingerprint_sha256,
        observation_contract_sha256=observation_contract_sha256,
        render_contract=render_contract,
    )
    if set(archived) != set(current):
        raise HistoricalReplayGateMismatch(
            "Historical physical replay gate fields changed"
        )
    for name in (
        "policy_id",
        "tree_contract",
        "physical_provenance",
        "float_quantization",
        "observation_render_provenance",
    ):
        if archived.get(name) != current[name]:
            raise HistoricalReplayGateMismatch(
                f"Historical physical replay gate {name} changed"
            )
    try:
        _require_sha256(
            archived.get("audit_quantized_digest_sha256"),
            "audit_quantized_digest_sha256",
        )
    except ValueError as exc:
        raise HistoricalReplayGateMismatch(
            "Historical audit-only quantized digest is invalid"
        ) from exc
    if archived.get("audit_quantized_digest_acceptance_role") != "none":
        raise HistoricalReplayGateMismatch(
            "Historical quantized digest must be audit-only"
        )
    archived_validation = archived.get("same_state_validation")
    if not isinstance(archived_validation, dict):
        raise HistoricalReplayGateMismatch(
            "Historical physical replay gate validation policy is missing"
        )
    if set(archived_validation) != set(current["same_state_validation"]):
        raise HistoricalReplayGateMismatch(
            "Historical physical replay gate validation fields changed"
        )
    for name in ("policy_id", "float_tolerance", "discrete_equality"):
        if (
            archived_validation.get(name)
            != current["same_state_validation"][name]
        ):
            raise HistoricalReplayGateMismatch(
                f"Historical physical replay gate validation {name} changed"
            )
    archived_spread = archived_validation.get("max_float_spread")
    if (
        not isinstance(archived_spread, (int, float))
        or isinstance(archived_spread, bool)
        or not math.isfinite(float(archived_spread))
        or float(archived_spread) < 0.0
        or float(archived_spread) > same_state_tolerance
    ):
        raise HistoricalReplayGateMismatch(
            "Historical physical replay gate spread is invalid"
        )
    archived_historical = archived.get("historical_replay_validation")
    current_historical = current["historical_replay_validation"]
    if archived_historical != current_historical:
        raise HistoricalReplayGateMismatch(
            "Historical replay-state validation policy changed"
        )
    if archived.get("field_count") != current_fingerprint.field_count:
        raise HistoricalReplayGateMismatch(
            "Historical physical replay gate field set changed"
        )
    if (
        archived.get("field_schema_sha256")
        != current_fingerprint.field_schema_sha256
    ):
        raise HistoricalReplayGateMismatch(
            "Historical physical replay gate field schema changed"
        )
    if (
        archived.get("discrete_state_sha256")
        != current_fingerprint.discrete_state_sha256
    ):
        raise HistoricalReplayGateMismatch(
            "Historical physical replay gate discrete state changed"
        )
    _validate_continuous_witness(
        archived.get("continuous_witness"),
        current["continuous_witness"],
        float_tolerance=historical_replay_state_tolerance,
    )


def _require_positive_finite(value: float, name: str) -> None:
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be positive and finite")


def _continuous_field_witness(
    *,
    name: str,
    value: Any,
) -> ContinuousFieldWitness:
    array = np.ascontiguousarray(value)
    flat = array.reshape(-1)
    indices = _continuous_sample_indices(int(flat.size))
    return ContinuousFieldWitness(
        name=name,
        shape=tuple(int(size) for size in array.shape),
        dtype=array.dtype.str,
        element_count=int(flat.size),
        sample_indices=indices,
        sample_values=tuple(float(flat[index]) for index in indices),
    )


def _continuous_sample_indices(element_count: int) -> tuple[int, ...]:
    if element_count < 0:
        raise ValueError("Continuous witness element count cannot be negative")
    if element_count <= CONTINUOUS_WITNESS_MAX_SAMPLES_PER_FIELD:
        return tuple(range(element_count))
    slots = CONTINUOUS_WITNESS_MAX_SAMPLES_PER_FIELD
    return tuple(
        slot * (element_count - 1) // (slots - 1)
        for slot in range(slots)
    )


def _validate_continuous_witness(
    archived: Any,
    current: dict[str, Any],
    *,
    float_tolerance: float,
) -> None:
    if not isinstance(archived, dict) or set(archived) != set(current):
        raise HistoricalReplayGateMismatch(
            "Historical continuous witness schema changed"
        )
    for name in sorted(set(current).difference({"fields"})):
        if archived.get(name) != current[name]:
            raise HistoricalReplayGateMismatch(
                f"Historical continuous witness {name} changed"
            )
    archived_fields = archived.get("fields")
    current_fields = current["fields"]
    if (
        not isinstance(archived_fields, list)
        or len(archived_fields) != len(current_fields)
    ):
        raise HistoricalReplayGateMismatch(
            "Historical continuous witness field set changed"
        )
    for archived_field, current_field in zip(
        archived_fields,
        current_fields,
        strict=True,
    ):
        if (
            not isinstance(archived_field, dict)
            or set(archived_field) != set(current_field)
        ):
            raise HistoricalReplayGateMismatch(
                "Historical continuous witness field schema changed"
            )
        for name in sorted(
            set(current_field).difference({"sample_values"})
        ):
            if archived_field.get(name) != current_field[name]:
                raise HistoricalReplayGateMismatch(
                    "Historical continuous witness sampling changed for "
                    f"{current_field['name']}"
                )
        archived_values = archived_field.get("sample_values")
        current_values = current_field["sample_values"]
        if (
            not isinstance(archived_values, list)
            or len(archived_values) != len(current_values)
        ):
            raise HistoricalReplayGateMismatch(
                "Historical continuous witness values changed shape for "
                f"{current_field['name']}"
            )
        for index, (archived_value, current_value) in enumerate(
            zip(archived_values, current_values, strict=True)
        ):
            left = _finite_witness_scalar(
                archived_value,
                field=current_field["name"],
            )
            right = _finite_witness_scalar(
                current_value,
                field=current_field["name"],
            )
            roundoff_bound = (
                math.ulp(left)
                + math.ulp(right)
                + math.ulp(float_tolerance)
            )
            if abs(left - right) > float_tolerance + roundoff_bound:
                flat_index = current_field["sample_indices"][index]
                raise HistoricalReplayGateMismatch(
                    "Historical continuous witness diverged for "
                    f"{current_field['name']}[{flat_index}]: "
                    f"archived={left:.17g}, current={right:.17g}, "
                    f"delta={abs(left - right):.6e}, "
                    f"limit={float_tolerance + roundoff_bound:.6e}"
                )


def _finite_witness_scalar(value: Any, *, field: str) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
    ):
        raise HistoricalReplayGateMismatch(
            f"Historical continuous witness for {field} is non-finite"
        )
    return float(value)


def _require_sha256(value: Any, name: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")


def _canonical_sha256(value: Any, name: str) -> str:
    try:
        payload = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be canonical JSON data") from exc
    return hashlib.sha256(payload).hexdigest()


def _require_physical_fingerprint(fingerprint: ReplayFingerprint) -> None:
    if (
        fingerprint.schema_version != REPLAY_FINGERPRINT_SCHEMA_VERSION
        or fingerprint.policy_id != PHYSICAL_REPLAY_GATE_POLICY_ID
        or fingerprint.float_quantization_policy_id
        != FLOAT_QUANTIZATION_POLICY_ID
        or fingerprint.float_quantization_step != FLOAT_QUANTIZATION_STEP
    ):
        raise ValueError("Fingerprint does not use the physical replay gate policy")
    _require_sha256(fingerprint.digest, "fingerprint.digest")
    _require_sha256(
        fingerprint.field_schema_sha256,
        "fingerprint.field_schema_sha256",
    )
    _require_sha256(
        fingerprint.discrete_state_sha256,
        "fingerprint.discrete_state_sha256",
    )
    if fingerprint.field_count < 1:
        raise ValueError("Physical replay gate fingerprint is empty")
    if (
        not math.isfinite(fingerprint.max_float_spread)
        or fingerprint.max_float_spread < 0.0
    ):
        raise ValueError("Physical replay gate spread is invalid")


def _new_digest(
    *,
    digest_policy_id: str,
    float_quantization_step: float,
) -> Any:
    if not isinstance(digest_policy_id, str) or not digest_policy_id:
        raise ValueError("digest_policy_id must be a non-empty string")
    digest = hashlib.sha256()
    header = json.dumps(
        {
            "schema_version": REPLAY_FINGERPRINT_SCHEMA_VERSION,
            "policy_id": digest_policy_id,
            "float_quantization_policy_id": FLOAT_QUANTIZATION_POLICY_ID,
            "float_quantization_step": float_quantization_step,
        },
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    digest.update(b"rexpolicy-replay-fingerprint\x00")
    digest.update(len(header).to_bytes(8, "big"))
    digest.update(header)
    return digest


def _update_digest(
    digest: Any,
    *,
    name: str,
    shape: tuple[int, ...],
    encoding: str,
    payload: bytes,
) -> None:
    header = json.dumps(
        {
            "name": name,
            "shape": list(shape),
            "encoding": encoding,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    digest.update(len(header).to_bytes(8, "big"))
    digest.update(header)
    digest.update(len(payload).to_bytes(8, "big"))
    digest.update(payload)


def _quantize_float(value: Any, step: float) -> np.ndarray:
    scaled = np.asarray(value, dtype=np.float64) / step
    limit = float(np.iinfo(np.int64).max)
    if not np.isfinite(scaled).all() or np.any(np.abs(scaled) >= limit):
        raise OverflowError("Replay fingerprint quantization exceeds int64")
    return np.ascontiguousarray(np.rint(scaled).astype(np.int64))


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
