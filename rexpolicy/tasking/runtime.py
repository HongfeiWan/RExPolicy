"""Deterministic failure-precedence evaluation of compiled task contracts."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from .canonical import canonical_fingerprint
from .capabilities import CapabilityCatalog, EventSchema
from .contract import (
    DEFAULT_TASK_COMPILER_POLICY,
    CompiledRewardProfile,
    CompiledTaskContract,
    TaskCompilerPolicy,
    validate_compiled_task_contract,
)
from .expression import MetricFrame, evaluate_expression

ACTIVE = "active"
SUCCESS = "success"
FAILURE = "failure"
TRUNCATED = "truncated"
TERMINAL_STATUSES = frozenset((SUCCESS, FAILURE, TRUNCATED))


class TaskEvaluationError(ValueError):
    """Raised for invalid contracts/frames, never for ordinary task failure."""


@dataclass(frozen=True)
class TaskRuntimeState:
    contract_fingerprint: str
    reward_profile_id: str
    reward_profile_fingerprint: str
    control_step: int
    goal_streak: int
    status: str

    def to_record(self) -> dict[str, Any]:
        return {
            "contract_fingerprint": self.contract_fingerprint,
            "reward_profile_id": self.reward_profile_id,
            "reward_profile_fingerprint": self.reward_profile_fingerprint,
            "control_step": self.control_step,
            "goal_streak": self.goal_streak,
            "status": self.status,
        }

    @property
    def fingerprint(self) -> str:
        return canonical_fingerprint(self.to_record())


@dataclass(frozen=True)
class RuleResult:
    code: str
    kind: str
    fired: bool

    def to_record(self) -> dict[str, Any]:
        return {"code": self.code, "kind": self.kind, "fired": self.fired}


@dataclass(frozen=True)
class RewardTermResult:
    term_id: str
    raw_value: float
    weight: float
    contribution: float

    def to_record(self) -> dict[str, Any]:
        return {
            "term_id": self.term_id,
            "raw_value": self.raw_value,
            "weight": self.weight,
            "contribution": self.contribution,
        }


@dataclass(frozen=True)
class TaskStepResult:
    next_state: TaskRuntimeState
    status: str
    goal_predicate: bool
    rule_results: tuple[RuleResult, ...]
    failure_codes: tuple[str, ...]
    safety_codes: tuple[str, ...]
    reward: float
    shaping_unclipped: float | None
    shaping_clipped: float | None
    reward_terms: tuple[RewardTermResult, ...]

    @property
    def success(self) -> bool:
        return self.status == SUCCESS

    @property
    def failure(self) -> bool:
        return self.status == FAILURE

    @property
    def safety_violation(self) -> bool:
        return bool(self.safety_codes)

    @property
    def truncated(self) -> bool:
        return self.status == TRUNCATED

    @property
    def terminal(self) -> bool:
        return self.status in TERMINAL_STATUSES

    def to_record(self) -> dict[str, Any]:
        return {
            "next_state": self.next_state.to_record(),
            "status": self.status,
            "goal_predicate": self.goal_predicate,
            "rule_results": [item.to_record() for item in self.rule_results],
            "failure_codes": list(self.failure_codes),
            "safety_codes": list(self.safety_codes),
            "reward": self.reward,
            "shaping_unclipped": self.shaping_unclipped,
            "shaping_clipped": self.shaping_clipped,
            "reward_terms": [item.to_record() for item in self.reward_terms],
        }

    @property
    def fingerprint(self) -> str:
        return canonical_fingerprint(self.to_record())


def _profile(
    contract: CompiledTaskContract,
    reward_profile_id: str,
) -> CompiledRewardProfile:
    profiles = {
        item.reward_profile_id: item for item in contract.reward_profiles
    }
    try:
        return profiles[reward_profile_id]
    except KeyError as error:
        raise TaskEvaluationError("Unknown compiled reward profile") from error


def _schema(
    contract: CompiledTaskContract,
    catalog: CapabilityCatalog,
) -> EventSchema:
    schemas = {
        item.event_schema_id: item for item in catalog.event_schemas
    }
    try:
        return schemas[contract.event_schema_id]
    except KeyError as error:
        raise TaskEvaluationError("Compiled EventSchema is unavailable") from error


def _validate_contract(
    contract: CompiledTaskContract,
    *,
    catalog: CapabilityCatalog,
    policy: TaskCompilerPolicy,
) -> None:
    try:
        validate_compiled_task_contract(
            contract,
            catalog=catalog,
            policy=policy,
        )
    except (TypeError, ValueError) as error:
        raise TaskEvaluationError("Compiled task contract is invalid") from error


def initial_runtime_state(
    contract: CompiledTaskContract,
    *,
    reward_profile_id: str,
    catalog: CapabilityCatalog,
    policy: TaskCompilerPolicy = DEFAULT_TASK_COMPILER_POLICY,
) -> TaskRuntimeState:
    """Create a state bound to one exact contract and reward profile."""
    _validate_contract(contract, catalog=catalog, policy=policy)
    profile = _profile(contract, reward_profile_id)
    return TaskRuntimeState(
        contract_fingerprint=contract.fingerprint,
        reward_profile_id=profile.reward_profile_id,
        reward_profile_fingerprint=profile.fingerprint,
        control_step=0,
        goal_streak=0,
        status=ACTIVE,
    )


def _validate_state(
    contract: CompiledTaskContract,
    state: TaskRuntimeState,
) -> CompiledRewardProfile:
    if state.contract_fingerprint != contract.fingerprint:
        raise TaskEvaluationError("Runtime state contract fingerprint mismatch")
    profile = _profile(contract, state.reward_profile_id)
    if state.reward_profile_fingerprint != profile.fingerprint:
        raise TaskEvaluationError("Runtime state reward profile mismatch")
    if state.status != ACTIVE:
        raise TaskEvaluationError("Terminal runtime state cannot be evaluated again")
    if (
        type(state.control_step) is not int
        or state.control_step < 0
        or state.control_step >= contract.episode_control_steps
    ):
        raise TaskEvaluationError("Runtime control step is outside the horizon")
    if (
        type(state.goal_streak) is not int
        or state.goal_streak < 0
        or state.goal_streak >= contract.goal_hold_steps
    ):
        raise TaskEvaluationError("Runtime goal streak is invalid")
    return profile


def _evaluate_program(
    program: Any,
    *,
    frame: MetricFrame,
    schema: EventSchema,
) -> Any:
    try:
        return evaluate_expression(program, frame=frame, schema=schema)
    except (FloatingPointError, TypeError, ValueError) as error:
        raise TaskEvaluationError("Compiled task expression evaluation failed") from error


def evaluate_task_step(
    contract: CompiledTaskContract,
    *,
    state: TaskRuntimeState,
    frame: MetricFrame,
    catalog: CapabilityCatalog,
    policy: TaskCompilerPolicy = DEFAULT_TASK_COMPILER_POLICY,
) -> TaskStepResult:
    """Evaluate one frame with failure > success > truncation precedence."""
    _validate_contract(contract, catalog=catalog, policy=policy)
    profile = _validate_state(contract, state)
    schema = _schema(contract, catalog)

    rule_results = tuple(
        RuleResult(
            code=rule.code,
            kind=rule.kind,
            fired=bool(
                _evaluate_program(rule.program, frame=frame, schema=schema)
            ),
        )
        for rule in contract.terminal_rules
    )
    goal_predicate = bool(
        _evaluate_program(contract.goal_program, frame=frame, schema=schema)
    )
    fired = tuple(item for item in rule_results if item.fired)
    failure_codes = tuple(item.code for item in fired)
    safety_codes = tuple(item.code for item in fired if item.kind == "safety")
    next_control_step = state.control_step + 1

    if fired:
        status = FAILURE
        goal_streak = 0
        reward = profile.failure_value
        shaping_unclipped = None
        shaping_clipped = None
        reward_terms: tuple[RewardTermResult, ...] = ()
    else:
        goal_streak = state.goal_streak + 1 if goal_predicate else 0
        if goal_streak >= contract.goal_hold_steps:
            status = SUCCESS
            goal_streak = contract.goal_hold_steps
            reward = profile.success_value
            shaping_unclipped = None
            shaping_clipped = None
            reward_terms = ()
        else:
            term_results = []
            contributions = []
            for term in profile.terms:
                raw_value = float(
                    _evaluate_program(term.program, frame=frame, schema=schema)
                )
                contribution = term.weight * raw_value
                if not math.isfinite(raw_value) or not math.isfinite(contribution):
                    raise TaskEvaluationError("Reward term produced a non-finite value")
                term_results.append(
                    RewardTermResult(
                        term_id=term.term_id,
                        raw_value=raw_value,
                        weight=term.weight,
                        contribution=contribution,
                    )
                )
                contributions.append(contribution)
            shaping_unclipped = math.fsum(contributions)
            lower, upper = profile.shaping_bounds
            shaping_clipped = min(max(shaping_unclipped, lower), upper)
            if not math.isfinite(shaping_clipped):
                raise TaskEvaluationError("Shaping reward is non-finite")
            reward = shaping_clipped
            reward_terms = tuple(term_results)
            status = (
                TRUNCATED
                if next_control_step >= contract.episode_control_steps
                else ACTIVE
            )

    next_state = TaskRuntimeState(
        contract_fingerprint=state.contract_fingerprint,
        reward_profile_id=state.reward_profile_id,
        reward_profile_fingerprint=state.reward_profile_fingerprint,
        control_step=next_control_step,
        goal_streak=goal_streak,
        status=status,
    )
    result = TaskStepResult(
        next_state=next_state,
        status=status,
        goal_predicate=goal_predicate,
        rule_results=rule_results,
        failure_codes=failure_codes,
        safety_codes=safety_codes,
        reward=reward,
        shaping_unclipped=shaping_unclipped,
        shaping_clipped=shaping_clipped,
        reward_terms=reward_terms,
    )
    canonical_fingerprint(result.to_record())
    return result
