"""Pure, leakage-safe checkpoint selection for Grasp-Lift no-z training.

The selector deliberately knows nothing about Newton, trajectory shards, or
Torch.  It accepts one complete, canonically ordered validation inventory and
selects a *checkpoint step*, never an individual lucky training seed.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from rexpolicy.stage0.types import canonical_fingerprint

GRASP_LIFT_NO_Z_SELECTION_SCHEMA_ID = "rexpolicy/stage0-grasp-lift-no-z-selection/v1"
GRASP_LIFT_NO_Z_SELECTION_SCHEMA_VERSION = 1
GRASP_LIFT_NO_Z_VALIDATION_STEPS = tuple(range(500, 10_001, 500))
GRASP_LIFT_NO_Z_VALIDATION_TRAINING_SEEDS = (31_001, 31_002, 31_003)
GRASP_LIFT_NO_Z_VALIDATION_RESET_SEEDS = (7009, 7014, 7019, 7025, 7026, 7031)
GRASP_LIFT_NO_Z_POLICY_NOISE_INDICES = (0, 1, 2, 3)

_ROLLOUT_SCHEMA_ID = "rexpolicy/stage0-grasp-lift-no-z-validation-rollout/v1"
_SEED_EVIDENCE_SCHEMA_ID = (
    "rexpolicy/stage0-grasp-lift-no-z-seed-checkpoint-evidence/v1"
)
_CHECKPOINT_RESULT_SCHEMA_ID = "rexpolicy/stage0-grasp-lift-no-z-checkpoint-result/v1"
_SELECTED_MODEL_SCHEMA_ID = "rexpolicy/stage0-grasp-lift-no-z-selected-model/v1"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")

_ROLLOUTS_PER_RESET = len(GRASP_LIFT_NO_Z_POLICY_NOISE_INDICES)
_ROLLOUTS_PER_SEED = len(GRASP_LIFT_NO_Z_VALIDATION_RESET_SEEDS) * _ROLLOUTS_PER_RESET
_ROLLOUTS_PER_CHECKPOINT = (
    len(GRASP_LIFT_NO_Z_VALIDATION_TRAINING_SEEDS) * _ROLLOUTS_PER_SEED
)
_MINIMUM_SEED_SUCCESSES = 23
_MINIMUM_SEED_RESET_SUCCESSES = 3
_MINIMUM_PASSING_SEEDS = 2
_MINIMUM_CHECKPOINT_RESET_SUCCESSES = 9
_MINIMUM_STABLE_CHECKPOINTS = 3


def _strict_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    return value


def _strict_boolean(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be boolean")
    return value


def _require_sha256(value: Any, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _freeze_diagnostic_value(value: Any, name: str) -> Any:
    """Freeze one finite, numeric-only diagnostic JSON value."""
    if value is None:
        return None
    if isinstance(value, bool):
        raise TypeError(f"{name} must not contain booleans")
    if isinstance(value, (int, float)):
        result = float(value)
        if not math.isfinite(result) or result < 0.0:
            raise ValueError(f"{name} numbers must be finite and non-negative")
        return result
    if isinstance(value, Mapping):
        keys = tuple(value)
        if any(not isinstance(key, str) or not key for key in keys):
            raise ValueError(f"{name} keys must be non-empty strings")
        frozen: dict[str, Any] = {}
        for key in sorted(keys):
            frozen[key] = _freeze_diagnostic_value(value[key], f"{name}.{key}")
        if not frozen:
            raise ValueError(f"{name} must not be empty")
        return MappingProxyType(frozen)
    raise TypeError(f"{name} must contain only mappings, numbers, or null")


def _thaw_diagnostic_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw_diagnostic_value(item) for key, item in value.items()}
    return value


def _protocol_record() -> dict[str, Any]:
    return {
        "admission": {
            "checkpoint": {
                "all_rollouts_require_zero_integrity_violation": True,
                "all_rollouts_require_zero_oracle_disagreement": True,
                "all_rollouts_require_zero_safety_violation": True,
                "minimum_passing_training_seeds": _MINIMUM_PASSING_SEEDS,
                "minimum_successes_per_reset_across_seeds_and_noise": (
                    _MINIMUM_CHECKPOINT_RESET_SUCCESSES
                ),
                "rollouts_per_reset_across_seeds_and_noise": (
                    len(GRASP_LIFT_NO_Z_VALIDATION_TRAINING_SEEDS) * _ROLLOUTS_PER_RESET
                ),
            },
            "seed": {
                "minimum_successes": _MINIMUM_SEED_SUCCESSES,
                "minimum_successes_per_reset": _MINIMUM_SEED_RESET_SUCCESSES,
                "rollouts": _ROLLOUTS_PER_SEED,
            },
        },
        "candidate_unit": "one_common_checkpoint_step_with_all_three_seeds/v1",
        "checkpoint_schedule": list(GRASP_LIFT_NO_Z_VALIDATION_STEPS),
        "diagnostics": {"offline_mse": "recorded_only_never_admission_or_ranking/v1"},
        "policy_noise_indices": list(GRASP_LIFT_NO_Z_POLICY_NOISE_INDICES),
        "ranking": (
            "longest_contiguous_eligible_run_then_earliest_run_"
            "then_lower_median_step/v1"
        ),
        "minimum_contiguous_eligible_checkpoints": _MINIMUM_STABLE_CHECKPOINTS,
        "schema_id": "rexpolicy/stage0-grasp-lift-no-z-selection-protocol/v1",
        "training_seeds": list(GRASP_LIFT_NO_Z_VALIDATION_TRAINING_SEEDS),
        "validation_reset_seeds": list(GRASP_LIFT_NO_Z_VALIDATION_RESET_SEEDS),
    }


def grasp_lift_no_z_selection_protocol_record() -> dict[str, Any]:
    """Return a fresh copy of the immutable Grasp-Lift selection protocol."""

    return _protocol_record()


GRASP_LIFT_NO_Z_SELECTION_PROTOCOL_SHA256 = canonical_fingerprint(
    grasp_lift_no_z_selection_protocol_record()
)


def _inventory_record() -> dict[str, Any]:
    return {
        "checkpoint_steps": list(GRASP_LIFT_NO_Z_VALIDATION_STEPS),
        "policy_noise_indices": list(GRASP_LIFT_NO_Z_POLICY_NOISE_INDICES),
        "rollout_cells": (
            len(GRASP_LIFT_NO_Z_VALIDATION_STEPS) * _ROLLOUTS_PER_CHECKPOINT
        ),
        "rollouts_per_seed_checkpoint": _ROLLOUTS_PER_SEED,
        "seed_checkpoint_entries": (
            len(GRASP_LIFT_NO_Z_VALIDATION_STEPS)
            * len(GRASP_LIFT_NO_Z_VALIDATION_TRAINING_SEEDS)
        ),
        "training_seeds": list(GRASP_LIFT_NO_Z_VALIDATION_TRAINING_SEEDS),
        "validation_reset_seeds": list(GRASP_LIFT_NO_Z_VALIDATION_RESET_SEEDS),
    }


@dataclass(frozen=True)
class GraspLiftNoZValidationRollout:
    """One fixed-noise closed-loop rollout on one validation reset."""

    reset_seed: int
    policy_noise_index: int
    success: bool
    safety_violation: bool
    integrity_violation: bool
    oracle_disagreement: bool

    def __post_init__(self) -> None:
        reset_seed = _strict_integer(self.reset_seed, "reset_seed")
        if reset_seed not in GRASP_LIFT_NO_Z_VALIDATION_RESET_SEEDS:
            raise ValueError("reset_seed is not in the fixed validation cohort")
        noise = _strict_integer(self.policy_noise_index, "policy_noise_index")
        if noise not in GRASP_LIFT_NO_Z_POLICY_NOISE_INDICES:
            raise ValueError("policy_noise_index is not one of the four fixed draws")
        for name in (
            "success",
            "safety_violation",
            "integrity_violation",
            "oracle_disagreement",
        ):
            _strict_boolean(getattr(self, name), name)

    @property
    def inventory_key(self) -> tuple[int, int]:
        return self.reset_seed, self.policy_noise_index

    def to_record(self) -> dict[str, Any]:
        return {
            "integrity_violation": self.integrity_violation,
            "oracle_disagreement": self.oracle_disagreement,
            "policy_noise_index": self.policy_noise_index,
            "reset_seed": self.reset_seed,
            "safety_violation": self.safety_violation,
            "schema_id": _ROLLOUT_SCHEMA_ID,
            "success": self.success,
        }


@dataclass(frozen=True)
class GraspLiftNoZSeedCheckpointEvidence:
    """Complete 24-rollout evidence for one seed at one checkpoint step."""

    checkpoint_step: int
    training_seed: int
    model_sha256: str
    rollouts: tuple[GraspLiftNoZValidationRollout, ...]
    offline_mse_diagnostic: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        step = _strict_integer(self.checkpoint_step, "checkpoint_step")
        if step not in GRASP_LIFT_NO_Z_VALIDATION_STEPS:
            raise ValueError("checkpoint_step is outside the fixed 20-step schedule")
        seed = _strict_integer(self.training_seed, "training_seed")
        if seed not in GRASP_LIFT_NO_Z_VALIDATION_TRAINING_SEEDS:
            raise ValueError("training_seed is outside the fixed three-seed family")
        _require_sha256(self.model_sha256, "model_sha256")
        if not isinstance(self.rollouts, tuple):
            raise TypeError("rollouts must be a tuple in canonical inventory order")
        if any(
            not isinstance(item, GraspLiftNoZValidationRollout)
            for item in self.rollouts
        ):
            raise TypeError(
                "rollouts must contain GraspLiftNoZValidationRollout values"
            )
        expected_keys = tuple(
            (reset_seed, noise_index)
            for reset_seed in GRASP_LIFT_NO_Z_VALIDATION_RESET_SEEDS
            for noise_index in GRASP_LIFT_NO_Z_POLICY_NOISE_INDICES
        )
        observed_keys = tuple(item.inventory_key for item in self.rollouts)
        if observed_keys != expected_keys:
            raise ValueError(
                "rollouts do not exactly match the canonical 6-reset by "
                "4-noise inventory"
            )
        if self.offline_mse_diagnostic is not None:
            if not isinstance(self.offline_mse_diagnostic, Mapping):
                raise TypeError("offline_mse_diagnostic must be a mapping or null")
            frozen = _freeze_diagnostic_value(
                self.offline_mse_diagnostic,
                "offline_mse_diagnostic",
            )
            object.__setattr__(self, "offline_mse_diagnostic", frozen)

    @property
    def inventory_key(self) -> tuple[int, int]:
        return self.checkpoint_step, self.training_seed

    def to_record(self) -> dict[str, Any]:
        return {
            "checkpoint_step": self.checkpoint_step,
            "model_sha256": self.model_sha256,
            "offline_mse_diagnostic": _thaw_diagnostic_value(
                self.offline_mse_diagnostic
            ),
            "rollouts": [item.to_record() for item in self.rollouts],
            "schema_id": _SEED_EVIDENCE_SCHEMA_ID,
            "training_seed": self.training_seed,
        }


@dataclass(frozen=True)
class GraspLiftNoZSeedCheckpointResult:
    """Success admission result for one training seed."""

    checkpoint_step: int
    training_seed: int
    model_sha256: str
    success_count: int
    successes_by_reset: tuple[int, ...]
    success_gate_passed: bool
    admission_failures: tuple[str, ...]
    offline_mse_diagnostic: Mapping[str, Any] | None

    def __post_init__(self) -> None:
        step = _strict_integer(self.checkpoint_step, "checkpoint_step")
        if step not in GRASP_LIFT_NO_Z_VALIDATION_STEPS:
            raise ValueError("seed result checkpoint_step changed")
        seed = _strict_integer(self.training_seed, "training_seed")
        if seed not in GRASP_LIFT_NO_Z_VALIDATION_TRAINING_SEEDS:
            raise ValueError("seed result training_seed changed")
        _require_sha256(self.model_sha256, "model_sha256")
        count = _strict_integer(self.success_count, "success_count")
        if not 0 <= count <= _ROLLOUTS_PER_SEED:
            raise ValueError("success_count is outside the 24-rollout inventory")
        if not isinstance(self.successes_by_reset, tuple) or len(
            self.successes_by_reset
        ) != len(GRASP_LIFT_NO_Z_VALIDATION_RESET_SEEDS):
            raise ValueError("successes_by_reset must cover the six fixed resets")
        for value in self.successes_by_reset:
            item = _strict_integer(value, "successes_by_reset item")
            if not 0 <= item <= _ROLLOUTS_PER_RESET:
                raise ValueError("successes_by_reset item is outside [0, 4]")
        if sum(self.successes_by_reset) != count:
            raise ValueError("success_count disagrees with successes_by_reset")
        _strict_boolean(self.success_gate_passed, "success_gate_passed")
        if not isinstance(self.admission_failures, tuple) or any(
            not isinstance(item, str) or not item for item in self.admission_failures
        ):
            raise ValueError("admission_failures must contain non-empty strings")
        if self.success_gate_passed != (not self.admission_failures):
            raise ValueError("success_gate_passed disagrees with admission_failures")
        if self.offline_mse_diagnostic is not None:
            if not isinstance(self.offline_mse_diagnostic, Mapping):
                raise TypeError("offline_mse_diagnostic must be a mapping or null")
            frozen = _freeze_diagnostic_value(
                self.offline_mse_diagnostic,
                "offline_mse_diagnostic",
            )
            object.__setattr__(self, "offline_mse_diagnostic", frozen)

    def to_record(self) -> dict[str, Any]:
        return {
            "admission_failures": list(self.admission_failures),
            "checkpoint_step": self.checkpoint_step,
            "model_sha256": self.model_sha256,
            "offline_mse_diagnostic": _thaw_diagnostic_value(
                self.offline_mse_diagnostic
            ),
            "schema_id": ("rexpolicy/stage0-grasp-lift-no-z-seed-checkpoint-result/v1"),
            "success_count": self.success_count,
            "success_gate_passed": self.success_gate_passed,
            "successes_by_reset": {
                str(reset_seed): count
                for reset_seed, count in zip(
                    GRASP_LIFT_NO_Z_VALIDATION_RESET_SEEDS,
                    self.successes_by_reset,
                )
            },
            "training_seed": self.training_seed,
        }


@dataclass(frozen=True)
class GraspLiftNoZCheckpointResult:
    """Joint three-seed admission result for one common checkpoint step."""

    checkpoint_step: int
    seed_results: tuple[GraspLiftNoZSeedCheckpointResult, ...]
    passing_seed_count: int
    success_count: int
    successes_by_reset: tuple[int, ...]
    safety_violation_count: int
    integrity_violation_count: int
    oracle_disagreement_count: int
    eligible: bool
    admission_failures: tuple[str, ...]

    def __post_init__(self) -> None:
        step = _strict_integer(self.checkpoint_step, "checkpoint_step")
        if step not in GRASP_LIFT_NO_Z_VALIDATION_STEPS:
            raise ValueError("checkpoint result step changed")
        if not isinstance(self.seed_results, tuple) or any(
            not isinstance(item, GraspLiftNoZSeedCheckpointResult)
            for item in self.seed_results
        ):
            raise TypeError(
                "seed_results must contain GraspLiftNoZSeedCheckpointResult values"
            )
        if (
            tuple(item.training_seed for item in self.seed_results)
            != GRASP_LIFT_NO_Z_VALIDATION_TRAINING_SEEDS
        ):
            raise ValueError("seed_results do not cover the fixed seed family")
        if any(
            item.checkpoint_step != self.checkpoint_step for item in self.seed_results
        ):
            raise ValueError("seed_results use different checkpoint steps")
        passing = _strict_integer(self.passing_seed_count, "passing_seed_count")
        if passing != sum(item.success_gate_passed for item in self.seed_results):
            raise ValueError("passing_seed_count disagrees with seed_results")
        success_count = _strict_integer(self.success_count, "success_count")
        if not 0 <= success_count <= _ROLLOUTS_PER_CHECKPOINT:
            raise ValueError("success_count is outside the 72-rollout inventory")
        if not isinstance(self.successes_by_reset, tuple) or len(
            self.successes_by_reset
        ) != len(GRASP_LIFT_NO_Z_VALIDATION_RESET_SEEDS):
            raise ValueError("checkpoint successes_by_reset changed")
        for value in self.successes_by_reset:
            item = _strict_integer(value, "successes_by_reset item")
            if not 0 <= item <= 12:
                raise ValueError("checkpoint reset success count is outside [0, 12]")
        if sum(self.successes_by_reset) != success_count:
            raise ValueError("checkpoint success_count disagrees with reset counts")
        for name in (
            "safety_violation_count",
            "integrity_violation_count",
            "oracle_disagreement_count",
        ):
            count = _strict_integer(getattr(self, name), name)
            if not 0 <= count <= _ROLLOUTS_PER_CHECKPOINT:
                raise ValueError(f"{name} is outside the 72-rollout inventory")
        _strict_boolean(self.eligible, "eligible")
        if not isinstance(self.admission_failures, tuple) or any(
            not isinstance(item, str) or not item for item in self.admission_failures
        ):
            raise ValueError("admission_failures must contain non-empty strings")
        if self.eligible != (not self.admission_failures):
            raise ValueError("eligible disagrees with admission_failures")

    def to_record(self) -> dict[str, Any]:
        return {
            "admission_failures": list(self.admission_failures),
            "checkpoint_step": self.checkpoint_step,
            "eligible": self.eligible,
            "integrity_violation_count": self.integrity_violation_count,
            "oracle_disagreement_count": self.oracle_disagreement_count,
            "passing_seed_count": self.passing_seed_count,
            "safety_violation_count": self.safety_violation_count,
            "schema_id": _CHECKPOINT_RESULT_SCHEMA_ID,
            "seed_results": [item.to_record() for item in self.seed_results],
            "success_count": self.success_count,
            "successes_by_reset": {
                str(reset_seed): count
                for reset_seed, count in zip(
                    GRASP_LIFT_NO_Z_VALIDATION_RESET_SEEDS,
                    self.successes_by_reset,
                )
            },
        }


@dataclass(frozen=True)
class GraspLiftNoZSelectedModel:
    """One member of the selected same-step three-seed family."""

    checkpoint_step: int
    training_seed: int
    model_sha256: str

    def __post_init__(self) -> None:
        step = _strict_integer(self.checkpoint_step, "checkpoint_step")
        if step not in GRASP_LIFT_NO_Z_VALIDATION_STEPS:
            raise ValueError("selected checkpoint_step changed")
        seed = _strict_integer(self.training_seed, "training_seed")
        if seed not in GRASP_LIFT_NO_Z_VALIDATION_TRAINING_SEEDS:
            raise ValueError("selected training_seed changed")
        _require_sha256(self.model_sha256, "model_sha256")

    def to_record(self) -> dict[str, Any]:
        return {
            "checkpoint_step": self.checkpoint_step,
            "model_sha256": self.model_sha256,
            "schema_id": _SELECTED_MODEL_SCHEMA_ID,
            "training_seed": self.training_seed,
        }


@dataclass(frozen=True)
class GraspLiftNoZCheckpointSelection:
    """Complete, self-hashed result of the fixed validation selection."""

    evidence: tuple[GraspLiftNoZSeedCheckpointEvidence, ...]
    checkpoint_results: tuple[GraspLiftNoZCheckpointResult, ...]
    eligible_runs: tuple[tuple[int, ...], ...]
    gate_passed: bool
    selected_checkpoint_step: int | None
    selected_family: tuple[GraspLiftNoZSelectedModel, ...]
    selection_reason: str

    def __post_init__(self) -> None:
        expected_evidence_keys = tuple(
            (step, seed)
            for step in GRASP_LIFT_NO_Z_VALIDATION_STEPS
            for seed in GRASP_LIFT_NO_Z_VALIDATION_TRAINING_SEEDS
        )
        if not isinstance(self.evidence, tuple) or any(
            not isinstance(item, GraspLiftNoZSeedCheckpointEvidence)
            for item in self.evidence
        ):
            raise TypeError(
                "evidence must contain GraspLiftNoZSeedCheckpointEvidence values"
            )
        if (
            tuple(item.inventory_key for item in self.evidence)
            != expected_evidence_keys
        ):
            raise ValueError("selection evidence inventory changed")
        if not isinstance(self.checkpoint_results, tuple) or any(
            not isinstance(item, GraspLiftNoZCheckpointResult)
            for item in self.checkpoint_results
        ):
            raise TypeError(
                "checkpoint_results must contain GraspLiftNoZCheckpointResult values"
            )
        if (
            tuple(item.checkpoint_step for item in self.checkpoint_results)
            != GRASP_LIFT_NO_Z_VALIDATION_STEPS
        ):
            raise ValueError("checkpoint_results inventory changed")
        if not isinstance(self.eligible_runs, tuple):
            raise TypeError("eligible_runs must be a tuple")
        for run in self.eligible_runs:
            if not isinstance(run, tuple) or not run:
                raise ValueError("eligible runs must be non-empty tuples")
            for step in run:
                value = _strict_integer(step, "eligible run step")
                if value not in GRASP_LIFT_NO_Z_VALIDATION_STEPS:
                    raise ValueError("eligible run step changed")
            if any(right - left != 500 for left, right in zip(run, run[1:])):
                raise ValueError("eligible run is not contiguous at 500-step spacing")
        expected_runs = _eligible_runs(self.checkpoint_results)
        if self.eligible_runs != expected_runs:
            raise ValueError("eligible_runs disagree with checkpoint_results")
        stable_runs = tuple(
            run for run in expected_runs if len(run) >= _MINIMUM_STABLE_CHECKPOINTS
        )
        expected_selected_step = None
        if stable_runs:
            expected_run = min(stable_runs, key=lambda run: (-len(run), run[0]))
            expected_selected_step = expected_run[(len(expected_run) - 1) // 2]
        _strict_boolean(self.gate_passed, "gate_passed")
        if self.selected_checkpoint_step is not None:
            selected_step = _strict_integer(
                self.selected_checkpoint_step,
                "selected_checkpoint_step",
            )
            if selected_step not in GRASP_LIFT_NO_Z_VALIDATION_STEPS:
                raise ValueError("selected_checkpoint_step changed")
        if self.gate_passed != (expected_selected_step is not None):
            raise ValueError("gate_passed disagrees with the stable eligible runs")
        if self.selected_checkpoint_step != expected_selected_step:
            raise ValueError("selected_checkpoint_step disagrees with fixed ranking")
        if not isinstance(self.selected_family, tuple) or any(
            not isinstance(item, GraspLiftNoZSelectedModel)
            for item in self.selected_family
        ):
            raise TypeError(
                "selected_family must contain GraspLiftNoZSelectedModel values"
            )
        if self.gate_passed:
            if self.selected_checkpoint_step is None:
                raise ValueError("passing selection is missing its checkpoint step")
            if tuple(item.training_seed for item in self.selected_family) != (
                GRASP_LIFT_NO_Z_VALIDATION_TRAINING_SEEDS
            ):
                raise ValueError("selected family must retain all three seeds")
            if any(
                item.checkpoint_step != self.selected_checkpoint_step
                for item in self.selected_family
            ):
                raise ValueError("selected family does not use one common step")
            expected_models = tuple(
                (item.training_seed, item.model_sha256)
                for item in self.evidence
                if item.checkpoint_step == self.selected_checkpoint_step
            )
            if (
                tuple(
                    (item.training_seed, item.model_sha256)
                    for item in self.selected_family
                )
                != expected_models
            ):
                raise ValueError("selected family model identities changed")
        elif self.selected_checkpoint_step is not None or self.selected_family:
            raise ValueError("failed selection cannot retain a selected family")
        if not isinstance(self.selection_reason, str) or not self.selection_reason:
            raise ValueError("selection_reason must be non-empty text")

    def _unsealed_record(self) -> dict[str, Any]:
        return {
            "checkpoint_results": [
                item.to_record() for item in self.checkpoint_results
            ],
            "eligible_runs": [list(run) for run in self.eligible_runs],
            "evidence": [item.to_record() for item in self.evidence],
            "gate_passed": self.gate_passed,
            "inventory": _inventory_record(),
            "protocol": _protocol_record(),
            "schema_id": GRASP_LIFT_NO_Z_SELECTION_SCHEMA_ID,
            "schema_version": GRASP_LIFT_NO_Z_SELECTION_SCHEMA_VERSION,
            "selected_checkpoint_step": self.selected_checkpoint_step,
            "selected_family": [item.to_record() for item in self.selected_family],
            "selection_reason": self.selection_reason,
        }

    @property
    def sha256(self) -> str:
        return canonical_fingerprint(self._unsealed_record())

    def to_record(self) -> dict[str, Any]:
        record = self._unsealed_record()
        record["self_sha256"] = canonical_fingerprint(record)
        return record


def _seed_result(
    evidence: GraspLiftNoZSeedCheckpointEvidence,
) -> GraspLiftNoZSeedCheckpointResult:
    successes_by_reset = tuple(
        sum(
            rollout.success
            for rollout in evidence.rollouts
            if rollout.reset_seed == reset_seed
        )
        for reset_seed in GRASP_LIFT_NO_Z_VALIDATION_RESET_SEEDS
    )
    success_count = sum(successes_by_reset)
    failures: list[str] = []
    if success_count < _MINIMUM_SEED_SUCCESSES:
        failures.append("minimum_seed_successes_not_met")
    if any(count < _MINIMUM_SEED_RESET_SUCCESSES for count in successes_by_reset):
        failures.append("minimum_seed_reset_successes_not_met")
    return GraspLiftNoZSeedCheckpointResult(
        checkpoint_step=evidence.checkpoint_step,
        training_seed=evidence.training_seed,
        model_sha256=evidence.model_sha256,
        success_count=success_count,
        successes_by_reset=successes_by_reset,
        success_gate_passed=not failures,
        admission_failures=tuple(failures),
        offline_mse_diagnostic=evidence.offline_mse_diagnostic,
    )


def _checkpoint_result(
    step: int,
    evidence: tuple[GraspLiftNoZSeedCheckpointEvidence, ...],
) -> GraspLiftNoZCheckpointResult:
    seed_results = tuple(_seed_result(item) for item in evidence)
    rollouts = tuple(rollout for item in evidence for rollout in item.rollouts)
    successes_by_reset = tuple(
        sum(rollout.success for rollout in rollouts if rollout.reset_seed == reset_seed)
        for reset_seed in GRASP_LIFT_NO_Z_VALIDATION_RESET_SEEDS
    )
    safety_count = sum(item.safety_violation for item in rollouts)
    integrity_count = sum(item.integrity_violation for item in rollouts)
    disagreement_count = sum(item.oracle_disagreement for item in rollouts)
    passing_seed_count = sum(item.success_gate_passed for item in seed_results)
    failures: list[str] = []
    if safety_count:
        failures.append("safety_violation")
    if integrity_count:
        failures.append("integrity_violation")
    if disagreement_count:
        failures.append("oracle_disagreement")
    if passing_seed_count < _MINIMUM_PASSING_SEEDS:
        failures.append("minimum_passing_training_seeds_not_met")
    if any(count < _MINIMUM_CHECKPOINT_RESET_SUCCESSES for count in successes_by_reset):
        failures.append("minimum_checkpoint_reset_successes_not_met")
    return GraspLiftNoZCheckpointResult(
        checkpoint_step=step,
        seed_results=seed_results,
        passing_seed_count=passing_seed_count,
        success_count=sum(successes_by_reset),
        successes_by_reset=successes_by_reset,
        safety_violation_count=safety_count,
        integrity_violation_count=integrity_count,
        oracle_disagreement_count=disagreement_count,
        eligible=not failures,
        admission_failures=tuple(failures),
    )


def _eligible_runs(
    results: tuple[GraspLiftNoZCheckpointResult, ...],
) -> tuple[tuple[int, ...], ...]:
    runs: list[list[int]] = []
    for result in results:
        if not result.eligible:
            continue
        if not runs or result.checkpoint_step - runs[-1][-1] != 500:
            runs.append([result.checkpoint_step])
        else:
            runs[-1].append(result.checkpoint_step)
    return tuple(tuple(run) for run in runs)


def select_grasp_lift_no_z_checkpoint(
    evidence: Sequence[GraspLiftNoZSeedCheckpointEvidence],
    *,
    validation_reset_seeds: Sequence[int] = GRASP_LIFT_NO_Z_VALIDATION_RESET_SEEDS,
) -> GraspLiftNoZCheckpointSelection:
    """Select one common step from a complete, fixed validation inventory.

    Input order is part of the audit contract: checkpoint step, training seed,
    reset seed, then policy-noise index.  Missing, duplicate, or reordered
    evidence fails before any result is produced.
    """

    if isinstance(validation_reset_seeds, (str, bytes)) or not isinstance(
        validation_reset_seeds, Sequence
    ):
        raise TypeError("validation_reset_seeds must be an ordered sequence")
    reset_seeds = tuple(validation_reset_seeds)
    if reset_seeds != GRASP_LIFT_NO_Z_VALIDATION_RESET_SEEDS:
        raise ValueError(
            "validation_reset_seeds changed or are not canonically ordered"
        )
    if isinstance(evidence, (str, bytes)) or not isinstance(evidence, Sequence):
        raise TypeError("evidence must be an ordered sequence")
    values = tuple(evidence)
    if any(not isinstance(item, GraspLiftNoZSeedCheckpointEvidence) for item in values):
        raise TypeError(
            "evidence must contain GraspLiftNoZSeedCheckpointEvidence values"
        )
    expected_keys = tuple(
        (step, seed)
        for step in GRASP_LIFT_NO_Z_VALIDATION_STEPS
        for seed in GRASP_LIFT_NO_Z_VALIDATION_TRAINING_SEEDS
    )
    if tuple(item.inventory_key for item in values) != expected_keys:
        raise ValueError(
            "evidence does not exactly match the canonical 20-step by 3-seed inventory"
        )

    checkpoint_results = tuple(
        _checkpoint_result(
            step,
            tuple(item for item in values if item.checkpoint_step == step),
        )
        for step in GRASP_LIFT_NO_Z_VALIDATION_STEPS
    )
    eligible_runs = _eligible_runs(checkpoint_results)
    stable_runs = tuple(
        run for run in eligible_runs if len(run) >= _MINIMUM_STABLE_CHECKPOINTS
    )
    if stable_runs:
        selected_run = min(stable_runs, key=lambda run: (-len(run), run[0]))
        selected_step = selected_run[(len(selected_run) - 1) // 2]
        selected_evidence = tuple(
            item for item in values if item.checkpoint_step == selected_step
        )
        selected_family = tuple(
            GraspLiftNoZSelectedModel(
                checkpoint_step=selected_step,
                training_seed=item.training_seed,
                model_sha256=item.model_sha256,
            )
            for item in selected_evidence
        )
        reason = (
            f"selected common checkpoint step {selected_step}: lower median of "
            f"the earliest longest eligible run ({len(selected_run)} checkpoints)"
        )
    else:
        selected_step = None
        selected_family = ()
        reason = (
            "validation admission failed: no contiguous run of at least three "
            "eligible common checkpoint steps"
        )
    return GraspLiftNoZCheckpointSelection(
        evidence=values,
        checkpoint_results=checkpoint_results,
        eligible_runs=eligible_runs,
        gate_passed=selected_step is not None,
        selected_checkpoint_step=selected_step,
        selected_family=selected_family,
        selection_reason=reason,
    )


def verify_grasp_lift_no_z_selection_record(record: Mapping[str, Any]) -> None:
    """Verify one sealed record and rebuild every result from raw evidence.

    The self hash detects accidental corruption.  It is not treated as an
    authenticity boundary: after checking it, this verifier parses the exact
    fixed evidence inventory, reruns the pure selector, and requires the whole
    supplied record to equal that canonical reconstruction.
    """

    if not isinstance(record, Mapping):
        raise TypeError("selection record must be a mapping")
    expected_fields = {
        "checkpoint_results",
        "eligible_runs",
        "evidence",
        "gate_passed",
        "inventory",
        "protocol",
        "schema_id",
        "schema_version",
        "selected_checkpoint_step",
        "selected_family",
        "selection_reason",
        "self_sha256",
    }
    if set(record) != expected_fields:
        raise ValueError("selection record fields changed")
    if record["schema_id"] != GRASP_LIFT_NO_Z_SELECTION_SCHEMA_ID:
        raise ValueError("selection record schema_id changed")
    if record["schema_version"] != GRASP_LIFT_NO_Z_SELECTION_SCHEMA_VERSION:
        raise ValueError("selection record schema_version changed")
    if record["inventory"] != _inventory_record():
        raise ValueError("selection record inventory changed")
    if record["protocol"] != _protocol_record():
        raise ValueError("selection record protocol changed")
    expected_sha256 = _require_sha256(record["self_sha256"], "self_sha256")
    unsealed = dict(record)
    unsealed.pop("self_sha256")
    if canonical_fingerprint(unsealed) != expected_sha256:
        raise ValueError("selection record self_sha256 mismatch")

    evidence_records = record["evidence"]
    if not isinstance(evidence_records, list):
        raise TypeError("selection record evidence must be a list")
    expected_evidence_count = (
        len(GRASP_LIFT_NO_Z_VALIDATION_STEPS)
        * len(GRASP_LIFT_NO_Z_VALIDATION_TRAINING_SEEDS)
    )
    if len(evidence_records) != expected_evidence_count:
        raise ValueError("selection record evidence inventory size changed")

    evidence: list[GraspLiftNoZSeedCheckpointEvidence] = []
    evidence_fields = {
        "checkpoint_step",
        "model_sha256",
        "offline_mse_diagnostic",
        "rollouts",
        "schema_id",
        "training_seed",
    }
    rollout_fields = {
        "integrity_violation",
        "oracle_disagreement",
        "policy_noise_index",
        "reset_seed",
        "safety_violation",
        "schema_id",
        "success",
    }
    for evidence_index, evidence_record in enumerate(evidence_records):
        evidence_name = f"selection record evidence[{evidence_index}]"
        if not isinstance(evidence_record, Mapping):
            raise TypeError(f"{evidence_name} must be a mapping")
        if set(evidence_record) != evidence_fields:
            raise ValueError(f"{evidence_name} fields changed")
        if evidence_record["schema_id"] != _SEED_EVIDENCE_SCHEMA_ID:
            raise ValueError(f"{evidence_name} schema_id changed")

        rollout_records = evidence_record["rollouts"]
        if not isinstance(rollout_records, list):
            raise TypeError(f"{evidence_name}.rollouts must be a list")
        if len(rollout_records) != _ROLLOUTS_PER_SEED:
            raise ValueError(f"{evidence_name}.rollouts inventory size changed")
        rollouts: list[GraspLiftNoZValidationRollout] = []
        for rollout_index, rollout_record in enumerate(rollout_records):
            rollout_name = f"{evidence_name}.rollouts[{rollout_index}]"
            if not isinstance(rollout_record, Mapping):
                raise TypeError(f"{rollout_name} must be a mapping")
            if set(rollout_record) != rollout_fields:
                raise ValueError(f"{rollout_name} fields changed")
            if rollout_record["schema_id"] != _ROLLOUT_SCHEMA_ID:
                raise ValueError(f"{rollout_name} schema_id changed")
            rollouts.append(
                GraspLiftNoZValidationRollout(
                    reset_seed=rollout_record["reset_seed"],
                    policy_noise_index=rollout_record["policy_noise_index"],
                    success=rollout_record["success"],
                    safety_violation=rollout_record["safety_violation"],
                    integrity_violation=rollout_record["integrity_violation"],
                    oracle_disagreement=rollout_record["oracle_disagreement"],
                )
            )
        evidence.append(
            GraspLiftNoZSeedCheckpointEvidence(
                checkpoint_step=evidence_record["checkpoint_step"],
                training_seed=evidence_record["training_seed"],
                model_sha256=evidence_record["model_sha256"],
                rollouts=tuple(rollouts),
                offline_mse_diagnostic=evidence_record["offline_mse_diagnostic"],
            )
        )

    reconstructed = select_grasp_lift_no_z_checkpoint(tuple(evidence))
    reconstructed_unsealed = reconstructed._unsealed_record()
    if (
        unsealed != reconstructed_unsealed
        or canonical_fingerprint(unsealed)
        != canonical_fingerprint(reconstructed_unsealed)
    ):
        raise ValueError(
            "selection record disagrees with evidence-derived selection"
        )


__all__ = [
    "GRASP_LIFT_NO_Z_POLICY_NOISE_INDICES",
    "GRASP_LIFT_NO_Z_SELECTION_PROTOCOL_SHA256",
    "GRASP_LIFT_NO_Z_SELECTION_SCHEMA_ID",
    "GRASP_LIFT_NO_Z_SELECTION_SCHEMA_VERSION",
    "GRASP_LIFT_NO_Z_VALIDATION_RESET_SEEDS",
    "GRASP_LIFT_NO_Z_VALIDATION_STEPS",
    "GRASP_LIFT_NO_Z_VALIDATION_TRAINING_SEEDS",
    "GraspLiftNoZCheckpointResult",
    "GraspLiftNoZCheckpointSelection",
    "GraspLiftNoZSeedCheckpointEvidence",
    "GraspLiftNoZSeedCheckpointResult",
    "GraspLiftNoZSelectedModel",
    "GraspLiftNoZValidationRollout",
    "grasp_lift_no_z_selection_protocol_record",
    "select_grasp_lift_no_z_checkpoint",
    "verify_grasp_lift_no_z_selection_record",
]
