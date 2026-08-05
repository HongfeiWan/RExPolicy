"""Pure, auditable selection of Stage 0 validation checkpoints."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

STAGE0_VALIDATION_CHECKPOINT_SELECTION_SCHEMA_ID = (
    "rexpolicy/stage0-validation-checkpoint-selection/v1"
)
STAGE0_VALIDATION_CHECKPOINT_SELECTION_SCHEMA_VERSION = 1


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

    def __post_init__(self) -> None:
        if (
            isinstance(self.conditional_step, bool)
            or not isinstance(self.conditional_step, int)
            or self.conditional_step < 0
        ):
            raise ValueError("conditional_step must be a non-negative integer")
        if not isinstance(self.contact_violation, bool):
            raise TypeError("contact_violation must be boolean")
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
            "success_rate": self.success_rate,
        }


@dataclass(frozen=True)
class Stage0RankedValidationCheckpoint:
    """One candidate's position and hard-safety status in a complete ranking."""

    rank: int
    candidate: Stage0ValidationCheckpointCandidate
    hard_safety_passed: bool
    safety_failures: tuple[str, ...]
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
        if not isinstance(self.selected, bool):
            raise TypeError("selected must be boolean")
        failures = tuple(self.safety_failures)
        if any(not isinstance(value, str) or not value for value in failures):
            raise ValueError("safety_failures must contain non-empty strings")
        if self.hard_safety_passed != (not failures):
            raise ValueError("hard_safety_passed must agree with safety_failures")
        if self.selected and not self.hard_safety_passed:
            raise ValueError("an unsafe checkpoint cannot be selected")
        object.__setattr__(self, "safety_failures", failures)

    def to_record(self) -> dict[str, Any]:
        return {
            **self.candidate.to_record(),
            "hard_safety_passed": self.hard_safety_passed,
            "rank": self.rank,
            "safety_failures": list(self.safety_failures),
            "selected": self.selected,
        }


@dataclass(frozen=True)
class Stage0ValidationCheckpointSelection:
    """Machine-readable outcome of a hard-safe checkpoint selection."""

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
        if self.gate_passed:
            if len(selected) != 1:
                raise ValueError(
                    "a passing selection must select exactly one checkpoint"
                )
            if self.selected_conditional_step != selected[0].candidate.conditional_step:
                raise ValueError("selected_conditional_step disagrees with ranking")
        elif selected or self.selected_conditional_step is not None:
            raise ValueError("a failed gate cannot select a checkpoint")
        object.__setattr__(self, "maximum_object_displacement_m", limit)
        object.__setattr__(self, "ranking", ranking)

    def to_record(self) -> dict[str, Any]:
        return {
            "gate_passed": self.gate_passed,
            "protocol": {
                "hard_safety": {
                    "contact_violation": False,
                    "maximum_object_displacement_m_lte": (
                        self.maximum_object_displacement_m
                    ),
                },
                "ranking": [
                    "hard_safety_passed_desc",
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
    """Rank candidates and select only from the hard-safe subset.

    Ranking is deterministic and independent of input order. Hard safety is a
    strict eligibility gate, not a soft score: when every candidate is unsafe,
    the returned gate fails and ``selected_conditional_step`` is ``None``.
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

    def failures(
        candidate: Stage0ValidationCheckpointCandidate,
    ) -> tuple[str, ...]:
        result = []
        if candidate.contact_violation:
            result.append("contact_violation")
        if candidate.maximum_object_displacement_m > limit:
            result.append("object_displacement_limit_exceeded")
        return tuple(result)

    evidence = tuple((candidate, failures(candidate)) for candidate in values)
    ordered = sorted(
        evidence,
        key=lambda item: (
            bool(item[1]),
            -item[0].success_rate,
            -item[0].path_macro_f1,
            item[0].conditional_step,
        ),
    )
    safe = tuple(item for item in ordered if not item[1])
    selected_step = safe[0][0].conditional_step if safe else None
    ranking = tuple(
        Stage0RankedValidationCheckpoint(
            rank=index,
            candidate=candidate,
            hard_safety_passed=not candidate_failures,
            safety_failures=candidate_failures,
            selected=candidate.conditional_step == selected_step,
        )
        for index, (candidate, candidate_failures) in enumerate(ordered, start=1)
    )
    if selected_step is None:
        reason = (
            "hard safety gate failed: no validation checkpoint had zero contact "
            "and object displacement within the configured limit"
        )
    else:
        reason = (
            f"selected conditional step {selected_step}: highest-ranked hard-safe "
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
