"""Pure, auditable selection of Stage 0 validation checkpoints."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

STAGE0_VALIDATION_CHECKPOINT_SELECTION_SCHEMA_ID = (
    "rexpolicy/stage0-validation-checkpoint-selection/v2"
)
STAGE0_VALIDATION_CHECKPOINT_SELECTION_SCHEMA_VERSION = 2


def _finite_unit_interval(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a finite number in [0, 1]")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be a finite number in [0, 1]")
    return result


def _finite_non_negative(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a finite non-negative number")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be a finite non-negative number")
    return result


@dataclass(frozen=True)
class Stage0ValidationCheckpointCandidate:
    """Validation evidence associated with one conditional-policy checkpoint."""

    conditional_step: int
    success_rate: float
    contact_violation: bool
    maximum_object_displacement_m: float
    path_macro_f1: float
    success_gate_passed: bool
    path_gate_passed: bool

    def __post_init__(self) -> None:
        if (
            isinstance(self.conditional_step, bool)
            or not isinstance(self.conditional_step, int)
            or self.conditional_step < 0
        ):
            raise ValueError("conditional_step must be a non-negative integer")
        if not isinstance(self.contact_violation, bool):
            raise TypeError("contact_violation must be boolean")
        if not isinstance(self.success_gate_passed, bool):
            raise TypeError("success_gate_passed must be boolean")
        if not isinstance(self.path_gate_passed, bool):
            raise TypeError("path_gate_passed must be boolean")
        object.__setattr__(
            self,
            "success_rate",
            _finite_unit_interval(self.success_rate, "success_rate"),
        )
        object.__setattr__(
            self,
            "maximum_object_displacement_m",
            _finite_non_negative(
                self.maximum_object_displacement_m,
                "maximum_object_displacement_m",
            ),
        )
        object.__setattr__(
            self,
            "path_macro_f1",
            _finite_unit_interval(self.path_macro_f1, "path_macro_f1"),
        )

    def to_record(self) -> dict[str, Any]:
        return {
            "conditional_step": self.conditional_step,
            "contact_violation": self.contact_violation,
            "maximum_object_displacement_m": self.maximum_object_displacement_m,
            "path_macro_f1": self.path_macro_f1,
            "path_gate_passed": self.path_gate_passed,
            "success_rate": self.success_rate,
            "success_gate_passed": self.success_gate_passed,
        }


@dataclass(frozen=True)
class Stage0RankedValidationCheckpoint:
    """One candidate's position and admission status in a complete ranking."""

    rank: int
    candidate: Stage0ValidationCheckpointCandidate
    hard_safety_passed: bool
    quality_gates_passed: bool
    eligible: bool
    safety_failures: tuple[str, ...]
    quality_failures: tuple[str, ...]
    selected: bool

    def __post_init__(self) -> None:
        if (
            isinstance(self.rank, bool)
            or not isinstance(self.rank, int)
            or self.rank < 1
        ):
            raise ValueError("rank must be a positive integer")
        if not isinstance(self.candidate, Stage0ValidationCheckpointCandidate):
            raise TypeError("candidate must be a Stage0 validation candidate")
        if not isinstance(self.hard_safety_passed, bool):
            raise TypeError("hard_safety_passed must be boolean")
        if not isinstance(self.quality_gates_passed, bool):
            raise TypeError("quality_gates_passed must be boolean")
        if not isinstance(self.eligible, bool):
            raise TypeError("eligible must be boolean")
        if not isinstance(self.selected, bool):
            raise TypeError("selected must be boolean")
        if not isinstance(self.safety_failures, tuple):
            raise TypeError("safety_failures must be a tuple")
        if not isinstance(self.quality_failures, tuple):
            raise TypeError("quality_failures must be a tuple")
        safety_failures = tuple(self.safety_failures)
        quality_failures = tuple(self.quality_failures)
        if any(not isinstance(value, str) or not value for value in safety_failures):
            raise ValueError("safety_failures must contain non-empty strings")
        if any(not isinstance(value, str) or not value for value in quality_failures):
            raise ValueError("quality_failures must contain non-empty strings")
        if self.hard_safety_passed != (not safety_failures):
            raise ValueError("hard_safety_passed must agree with safety_failures")
        if self.quality_gates_passed != (not quality_failures):
            raise ValueError("quality_gates_passed must agree with quality_failures")
        if self.eligible != (self.hard_safety_passed and self.quality_gates_passed):
            raise ValueError("eligible must require safety and quality admission")
        if self.selected and not self.eligible:
            raise ValueError("an ineligible checkpoint cannot be selected")
        object.__setattr__(self, "safety_failures", safety_failures)
        object.__setattr__(self, "quality_failures", quality_failures)

    def to_record(self) -> dict[str, Any]:
        return {
            **self.candidate.to_record(),
            "admission_failures": {
                "quality": list(self.quality_failures),
                "safety": list(self.safety_failures),
            },
            "eligible": self.eligible,
            "hard_safety_passed": self.hard_safety_passed,
            "quality_failures": list(self.quality_failures),
            "quality_gates_passed": self.quality_gates_passed,
            "rank": self.rank,
            "safety_failures": list(self.safety_failures),
            "selected": self.selected,
        }


@dataclass(frozen=True)
class Stage0ValidationCheckpointSelection:
    """Machine-readable outcome of strict validation admission and ranking."""

    maximum_object_displacement_m: float
    gate_passed: bool
    selected_conditional_step: int | None
    selection_reason: str
    ranking: tuple[Stage0RankedValidationCheckpoint, ...]

    def __post_init__(self) -> None:
        limit = _finite_non_negative(
            self.maximum_object_displacement_m,
            "maximum_object_displacement_m",
        )
        if not isinstance(self.gate_passed, bool):
            raise TypeError("gate_passed must be boolean")
        if self.selected_conditional_step is not None and (
            isinstance(self.selected_conditional_step, bool)
            or not isinstance(self.selected_conditional_step, int)
            or self.selected_conditional_step < 0
        ):
            raise ValueError(
                "selected_conditional_step must be null or a non-negative integer"
            )
        if not isinstance(self.selection_reason, str) or not self.selection_reason:
            raise ValueError("selection_reason must be a non-empty string")
        ranking = tuple(self.ranking)
        if any(
            not isinstance(item, Stage0RankedValidationCheckpoint) for item in ranking
        ):
            raise TypeError("ranking must contain ranked validation checkpoints")
        if tuple(item.rank for item in ranking) != tuple(range(1, len(ranking) + 1)):
            raise ValueError("ranking positions must be contiguous and one-based")
        selected = tuple(item for item in ranking if item.selected)
        eligible = tuple(item for item in ranking if item.eligible)
        if self.gate_passed:
            if len(selected) != 1:
                raise ValueError(
                    "a passing selection must select exactly one checkpoint"
                )
            if self.selected_conditional_step != selected[0].candidate.conditional_step:
                raise ValueError("selected_conditional_step disagrees with ranking")
        elif selected or self.selected_conditional_step is not None or eligible:
            raise ValueError(
                "a failed gate cannot contain an eligible or selected checkpoint"
            )
        object.__setattr__(self, "maximum_object_displacement_m", limit)
        object.__setattr__(self, "ranking", ranking)

    def to_record(self) -> dict[str, Any]:
        return {
            "gate_passed": self.gate_passed,
            "protocol": {
                "admission": {
                    "hard_safety": {
                        "contact_violation": False,
                        "maximum_object_displacement_m_lte": (
                            self.maximum_object_displacement_m
                        ),
                    },
                    "quality_gates": {
                        "path_gate_passed": True,
                        "success_gate_passed": True,
                    },
                },
                "ranking": [
                    "eligible_desc",
                    "success_rate_desc",
                    "path_macro_f1_desc",
                    "conditional_step_asc",
                ],
            },
            "ranking": [item.to_record() for item in self.ranking],
            "schema_id": STAGE0_VALIDATION_CHECKPOINT_SELECTION_SCHEMA_ID,
            "schema_version": STAGE0_VALIDATION_CHECKPOINT_SELECTION_SCHEMA_VERSION,
            "selected_conditional_step": self.selected_conditional_step,
            "selection_reason": self.selection_reason,
        }


def select_stage0_validation_checkpoint(
    candidates: Sequence[Stage0ValidationCheckpointCandidate],
    *,
    maximum_object_displacement_m: float,
) -> Stage0ValidationCheckpointSelection:
    """Rank candidates and select only from the strictly eligible subset.

    Ranking is deterministic and independent of input order. Eligibility
    requires hard safety plus upstream validation success and path gates.  When
    every candidate fails admission, the result fails closed without selecting.
    """

    limit = _finite_non_negative(
        maximum_object_displacement_m,
        "maximum_object_displacement_m",
    )
    values = tuple(candidates)
    if any(
        not isinstance(value, Stage0ValidationCheckpointCandidate) for value in values
    ):
        raise TypeError("candidates must contain Stage0 validation candidates")
    steps = tuple(value.conditional_step for value in values)
    if len(set(steps)) != len(steps):
        raise ValueError("candidate conditional_step values must be unique")

    def admission_failures(
        candidate: Stage0ValidationCheckpointCandidate,
    ) -> tuple[tuple[str, ...], tuple[str, ...]]:
        safety = []
        if candidate.contact_violation:
            safety.append("contact_violation")
        if candidate.maximum_object_displacement_m > limit:
            safety.append("object_displacement_limit_exceeded")
        quality = []
        if not candidate.success_gate_passed:
            quality.append("success_gate_failed")
        if not candidate.path_gate_passed:
            quality.append("path_gate_failed")
        return tuple(safety), tuple(quality)

    evidence = tuple(
        (candidate, *admission_failures(candidate)) for candidate in values
    )
    ordered = sorted(
        evidence,
        key=lambda item: (
            bool(item[1] or item[2]),
            -item[0].success_rate,
            -item[0].path_macro_f1,
            item[0].conditional_step,
        ),
    )
    eligible = tuple(item for item in ordered if not item[1] and not item[2])
    selected_step = eligible[0][0].conditional_step if eligible else None
    ranking = tuple(
        Stage0RankedValidationCheckpoint(
            rank=index,
            candidate=candidate,
            hard_safety_passed=not safety_failures,
            quality_gates_passed=not quality_failures,
            eligible=not safety_failures and not quality_failures,
            safety_failures=safety_failures,
            quality_failures=quality_failures,
            selected=candidate.conditional_step == selected_step,
        )
        for index, (candidate, safety_failures, quality_failures) in enumerate(
            ordered,
            start=1,
        )
    )
    if selected_step is None:
        reason = (
            "validation admission failed: no checkpoint passed hard safety, "
            "the success gate, and the path gate"
        )
    else:
        reason = (
            f"selected conditional step {selected_step}: highest-ranked eligible "
            "checkpoint by success rate, path macro-F1, then earliest step"
        )
    return Stage0ValidationCheckpointSelection(
        maximum_object_displacement_m=limit,
        gate_passed=selected_step is not None,
        selected_conditional_step=selected_step,
        selection_reason=reason,
        ranking=ranking,
    )


__all__ = [
    "STAGE0_VALIDATION_CHECKPOINT_SELECTION_SCHEMA_ID",
    "STAGE0_VALIDATION_CHECKPOINT_SELECTION_SCHEMA_VERSION",
    "Stage0RankedValidationCheckpoint",
    "Stage0ValidationCheckpointCandidate",
    "Stage0ValidationCheckpointSelection",
    "select_stage0_validation_checkpoint",
]
