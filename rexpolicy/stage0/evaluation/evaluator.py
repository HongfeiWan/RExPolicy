"""Fair no-z/oracle-z/selector-z comparisons for Stage 0 policies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from .metrics import (
    ConditioningMode,
    FixedEvaluationBudget,
    PolicyEvaluationRun,
)


@dataclass(frozen=True)
class ConditioningModeSummary:
    """Compact oracle summary for one fixed-budget conditioning run."""

    mode: ConditioningMode
    success_count: int
    failure_count: int
    timeout_count: int
    success_rate: float
    allocated_world_steps: int
    executed_world_steps: int
    success_rate_delta_from_no_z: float

    def to_record(self) -> dict[str, Any]:
        return {
            "allocated_world_steps": self.allocated_world_steps,
            "executed_world_steps": self.executed_world_steps,
            "failure_count": self.failure_count,
            "mode": self.mode.value,
            "success_count": self.success_count,
            "success_rate": self.success_rate,
            "success_rate_delta_from_no_z": self.success_rate_delta_from_no_z,
            "timeout_count": self.timeout_count,
        }


@dataclass(frozen=True)
class ConditioningComparison:
    """The three required controls evaluated against one immutable budget."""

    budget: FixedEvaluationBudget
    summaries: tuple[ConditioningModeSummary, ...]

    def summary(self, mode: ConditioningMode | str) -> ConditioningModeSummary:
        requested = ConditioningMode(mode)
        return next(item for item in self.summaries if item.mode is requested)

    def to_record(self) -> dict[str, Any]:
        return {
            "budget": self.budget.to_record(),
            "budget_sha256": self.budget.fingerprint,
            "comparison": "no-z_vs_oracle-z_vs_selector-z",
            "summaries": [summary.to_record() for summary in self.summaries],
        }


def compare_conditioning_runs(
    runs: Iterable[PolicyEvaluationRun],
) -> ConditioningComparison:
    """Compare exactly the three controls after enforcing budget identity."""

    values = tuple(runs)
    if len(values) != len(ConditioningMode):
        raise ValueError("comparison requires exactly three conditioning runs")
    if any(not isinstance(run, PolicyEvaluationRun) for run in values):
        raise TypeError("runs must contain only PolicyEvaluationRun values")
    by_mode = {run.mode: run for run in values}
    if len(by_mode) != len(values):
        raise ValueError("conditioning modes must be unique")
    if set(by_mode) != set(ConditioningMode):
        missing = sorted(mode.value for mode in set(ConditioningMode) - set(by_mode))
        raise ValueError(f"comparison is missing conditioning modes: {missing}")

    budget = values[0].budget
    if any(run.budget.fingerprint != budget.fingerprint for run in values[1:]):
        raise ValueError(
            "all conditioning modes must use identical reset groups, seeds, "
            "and step budgets"
        )
    no_z_rate = by_mode[ConditioningMode.NO_Z].success_rate
    summaries: list[ConditioningModeSummary] = []
    for mode in ConditioningMode:
        run = by_mode[mode]
        counts = run.outcome_counts
        summaries.append(
            ConditioningModeSummary(
                mode=mode,
                success_count=counts["success"],
                failure_count=counts["failure"],
                timeout_count=counts["timeout"],
                success_rate=run.success_rate,
                allocated_world_steps=budget.allocated_world_steps,
                executed_world_steps=run.executed_world_steps,
                success_rate_delta_from_no_z=run.success_rate - no_z_rate,
            )
        )
    return ConditioningComparison(budget=budget, summaries=tuple(summaries))
