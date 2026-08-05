"""Strict outcome records for fixed-budget Stage 0 policy evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from rexpolicy.stage0.types import Stage0Outcome, canonical_fingerprint


class ConditioningMode(str, Enum):
    """Required policy controls for measuring whether conditioning helps."""

    NO_Z = "no-z"
    ORACLE_Z = "oracle-z"
    SELECTOR_Z = "selector-z"


def _mode(value: ConditioningMode | str) -> ConditioningMode:
    try:
        return ConditioningMode(value)
    except (TypeError, ValueError) as error:
        choices = ", ".join(mode.value for mode in ConditioningMode)
        raise ValueError(f"conditioning mode must be one of: {choices}") from error


def _outcome(value: Stage0Outcome | str) -> Stage0Outcome:
    try:
        return Stage0Outcome(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"unsupported Stage 0 outcome: {value!r}") from error


@dataclass(frozen=True)
class FixedEvaluationBudget:
    """The identical reset, seed, and horizon allocation used by every mode."""

    reset_group_ids: tuple[str, ...]
    rollout_seeds: tuple[int, ...]
    max_steps_per_episode: int

    def __post_init__(self) -> None:
        if not isinstance(self.reset_group_ids, tuple) or not self.reset_group_ids:
            raise ValueError("reset_group_ids must be a non-empty tuple")
        if not isinstance(self.rollout_seeds, tuple):
            raise ValueError("rollout_seeds must be a tuple")
        if len(self.reset_group_ids) != len(self.rollout_seeds):
            raise ValueError("reset_group_ids and rollout_seeds must align")
        pairs: set[tuple[str, int]] = set()
        for index, (reset_group_id, seed) in enumerate(
            zip(self.reset_group_ids, self.rollout_seeds)
        ):
            if not isinstance(reset_group_id, str) or not reset_group_id.strip():
                raise ValueError(f"reset_group_ids[{index}] must be non-empty text")
            if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
                raise ValueError(
                    f"rollout_seeds[{index}] must be a non-negative integer"
                )
            pair = (reset_group_id, seed)
            if pair in pairs:
                raise ValueError("reset-group and rollout-seed pairs must be unique")
            pairs.add(pair)
        if (
            type(self.max_steps_per_episode) is not int
            or self.max_steps_per_episode <= 0
        ):
            raise ValueError("max_steps_per_episode must be a positive integer")

    @property
    def episode_count(self) -> int:
        return len(self.reset_group_ids)

    @property
    def allocated_world_steps(self) -> int:
        return self.episode_count * self.max_steps_per_episode

    @property
    def fingerprint(self) -> str:
        return canonical_fingerprint(self.to_record())

    def to_record(self) -> dict[str, Any]:
        return {
            "allocated_world_steps": self.allocated_world_steps,
            "max_steps_per_episode": self.max_steps_per_episode,
            "reset_group_ids": list(self.reset_group_ids),
            "rollout_seeds": list(self.rollout_seeds),
        }


@dataclass(frozen=True)
class PolicyEvaluationRun:
    """Oracle outcomes for one conditioning mode under one fixed budget."""

    mode: ConditioningMode
    budget: FixedEvaluationBudget
    outcomes: tuple[Stage0Outcome, ...]
    completed_steps: tuple[int, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "mode", _mode(self.mode))
        if not isinstance(self.budget, FixedEvaluationBudget):
            raise TypeError("budget must be a FixedEvaluationBudget")
        if not isinstance(self.outcomes, tuple):
            raise ValueError("outcomes must be a tuple")
        if not isinstance(self.completed_steps, tuple):
            raise ValueError("completed_steps must be a tuple")
        if len(self.outcomes) != self.budget.episode_count:
            raise ValueError("outcomes must contain one value per budget episode")
        if len(self.completed_steps) != self.budget.episode_count:
            raise ValueError(
                "completed_steps must contain one value per budget episode"
            )
        object.__setattr__(
            self,
            "outcomes",
            tuple(_outcome(outcome) for outcome in self.outcomes),
        )
        for index, steps in enumerate(self.completed_steps):
            if (
                type(steps) is not int
                or steps <= 0
                or steps > self.budget.max_steps_per_episode
            ):
                raise ValueError(
                    f"completed_steps[{index}] must be in [1, "
                    f"{self.budget.max_steps_per_episode}]"
                )

    @property
    def executed_world_steps(self) -> int:
        return sum(self.completed_steps)

    @property
    def outcome_counts(self) -> dict[str, int]:
        return {
            outcome.value: sum(value is outcome for value in self.outcomes)
            for outcome in Stage0Outcome
        }

    @property
    def success_rate(self) -> float:
        return self.outcome_counts[Stage0Outcome.SUCCESS.value] / len(self.outcomes)

    def to_record(self) -> dict[str, Any]:
        return {
            "allocated_world_steps": self.budget.allocated_world_steps,
            "budget_sha256": self.budget.fingerprint,
            "completed_steps": list(self.completed_steps),
            "executed_world_steps": self.executed_world_steps,
            "mode": self.mode.value,
            "outcome_counts": self.outcome_counts,
            "outcomes": [outcome.value for outcome in self.outcomes],
            "success_rate": self.success_rate,
        }
