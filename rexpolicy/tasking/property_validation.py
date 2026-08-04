"""Deterministic author-facing property gates for compiled task contracts."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any

from .canonical import canonical_fingerprint
from .capabilities import CapabilityCatalog, EventSchema, validate_event_values
from .contract import (
    DEFAULT_TASK_COMPILER_POLICY,
    CompiledTaskContract,
    TaskCompilerPolicy,
    validate_compiled_task_contract,
)
from .expression import MetricFrame
from .model import TaskSpecV2, ValidationExample
from .runtime import (
    ACTIVE,
    SUCCESS,
    TaskEvaluationError,
    TaskStepResult,
    evaluate_task_step,
    initial_runtime_state,
)

_IDENTIFIER = re.compile(r"^[a-z][a-z0-9_.-]*(?:/[a-z0-9_.-]+)*$")


@dataclass(frozen=True)
class PropertyValidationPolicy:
    policy_id: str
    max_examples: int
    max_steps_per_example: int
    max_total_steps: int
    require_safe_success: bool
    require_active_example: bool
    require_goal_positive_and_negative: bool
    require_each_rule_positive_and_negative: bool
    require_failure_precedence_example: bool
    require_reward_profile_invariance: bool
    require_deterministic_replay: bool

    def __post_init__(self) -> None:
        if not isinstance(self.policy_id, str) or _IDENTIFIER.fullmatch(
            self.policy_id
        ) is None:
            raise ValueError("Property validation policy ID is invalid")
        for name in ("max_examples", "max_steps_per_example", "max_total_steps"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"Property validation {name} must be positive")
        for name in (
            "require_safe_success",
            "require_active_example",
            "require_goal_positive_and_negative",
            "require_each_rule_positive_and_negative",
            "require_failure_precedence_example",
            "require_reward_profile_invariance",
            "require_deterministic_replay",
        ):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"Property validation {name} must be Boolean")

    def to_record(self) -> dict[str, Any]:
        return {
            "policy_id": self.policy_id,
            "max_examples": self.max_examples,
            "max_steps_per_example": self.max_steps_per_example,
            "max_total_steps": self.max_total_steps,
            "require_safe_success": self.require_safe_success,
            "require_active_example": self.require_active_example,
            "require_goal_positive_and_negative": (
                self.require_goal_positive_and_negative
            ),
            "require_each_rule_positive_and_negative": (
                self.require_each_rule_positive_and_negative
            ),
            "require_failure_precedence_example": (
                self.require_failure_precedence_example
            ),
            "require_reward_profile_invariance": (
                self.require_reward_profile_invariance
            ),
            "require_deterministic_replay": self.require_deterministic_replay,
        }

    @property
    def fingerprint(self) -> str:
        return canonical_fingerprint(self.to_record())


DEFAULT_PROPERTY_VALIDATION_POLICY = PropertyValidationPolicy(
    policy_id="rexpolicy/task_properties/v1",
    max_examples=128,
    max_steps_per_example=256,
    max_total_steps=4096,
    require_safe_success=True,
    require_active_example=True,
    require_goal_positive_and_negative=True,
    require_each_rule_positive_and_negative=True,
    require_failure_precedence_example=True,
    require_reward_profile_invariance=True,
    require_deterministic_replay=True,
)


@dataclass(frozen=True)
class PropertyIssue:
    code: str
    example_id: str | None
    path: str | None
    detail: str

    def to_record(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "example_id": self.example_id,
            "path": self.path,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class TaskPropertyReport:
    report_schema_version: int
    task_id: str
    task_spec_fingerprint: str
    contract_fingerprint: str
    validation_policy_id: str
    validation_policy_fingerprint: str
    accepted: bool
    issues: tuple[PropertyIssue, ...]
    example_count: int
    evaluated_step_count: int

    def to_record(self) -> dict[str, Any]:
        return {
            "report_schema_version": self.report_schema_version,
            "task_id": self.task_id,
            "task_spec_fingerprint": self.task_spec_fingerprint,
            "contract_fingerprint": self.contract_fingerprint,
            "validation_policy_id": self.validation_policy_id,
            "validation_policy_fingerprint": (
                self.validation_policy_fingerprint
            ),
            "accepted": self.accepted,
            "issues": [issue.to_record() for issue in self.issues],
            "example_count": self.example_count,
            "evaluated_step_count": self.evaluated_step_count,
        }

    @property
    def fingerprint(self) -> str:
        return canonical_fingerprint(self.to_record())


@dataclass(frozen=True)
class _Replay:
    results: tuple[TaskStepResult, ...]
    error: str | None


def _schema(
    contract: CompiledTaskContract,
    catalog: CapabilityCatalog,
) -> EventSchema:
    schemas = {
        item.event_schema_id: item for item in catalog.event_schemas
    }
    return schemas[contract.event_schema_id]


def _next_previous(
    current: dict[str, Any],
    schema: EventSchema,
) -> dict[str, Any]:
    return {
        metric.name: current[metric.name]
        for metric in schema.metrics
        if "previous" in metric.available_at
    }


def _replay(
    example: ValidationExample,
    *,
    reward_profile_id: str,
    contract: CompiledTaskContract,
    catalog: CapabilityCatalog,
    compiler_policy: TaskCompilerPolicy,
) -> _Replay:
    schema = _schema(contract, catalog)
    previous = dict(example.initial_metrics)
    try:
        validate_event_values(previous, schema=schema, at="previous")
    except ValueError as error:
        return _Replay((), f"initial metrics: {error}")
    try:
        state = initial_runtime_state(
            contract,
            reward_profile_id=reward_profile_id,
            catalog=catalog,
            policy=compiler_policy,
        )
    except TaskEvaluationError as error:
        return _Replay((), str(error))
    results = []
    for index, raw_current in enumerate(example.steps):
        if state.status != ACTIVE:
            return _Replay(
                tuple(results),
                f"trace continues after terminal step {index - 1}",
            )
        current = dict(raw_current)
        try:
            validate_event_values(current, schema=schema, at="current")
            result = evaluate_task_step(
                contract,
                state=state,
                frame=MetricFrame(
                    event_schema_id=schema.event_schema_id,
                    event_schema_fingerprint=schema.fingerprint,
                    previous=previous,
                    current=current,
                ),
                catalog=catalog,
                policy=compiler_policy,
            )
        except (TaskEvaluationError, ValueError) as error:
            return _Replay(tuple(results), f"step {index}: {error}")
        results.append(result)
        state = result.next_state
        previous = _next_previous(current, schema)
    return _Replay(tuple(results), None)


def _issue(
    code: str,
    detail: str,
    *,
    example_id: str | None = None,
    path: str | None = None,
) -> PropertyIssue:
    return PropertyIssue(
        code=code,
        example_id=example_id,
        path=path,
        detail=detail,
    )


def _canonical_issues(issues: list[PropertyIssue]) -> tuple[PropertyIssue, ...]:
    unique = {
        (item.code, item.example_id, item.path, item.detail): item
        for item in issues
    }
    return tuple(
        unique[key]
        for key in sorted(
            unique,
            key=lambda item: tuple("" if value is None else value for value in item),
        )
    )


def _oracle_signature(replay: _Replay) -> tuple[Any, ...]:
    return tuple(
        (
            result.status,
            result.goal_predicate,
            result.failure_codes,
            result.safety_codes,
        )
        for result in replay.results
    )


def validate_task_properties(
    task: TaskSpecV2,
    contract: CompiledTaskContract,
    *,
    catalog: CapabilityCatalog,
    policy: PropertyValidationPolicy = DEFAULT_PROPERTY_VALIDATION_POLICY,
    compiler_policy: TaskCompilerPolicy = DEFAULT_TASK_COMPILER_POLICY,
) -> TaskPropertyReport:
    """Return stable author-facing issues; raise only for trusted-state drift."""
    canonical_task = TaskSpecV2.from_record(task.to_record())
    canonical_catalog = CapabilityCatalog.from_record(catalog.to_record())
    validate_compiled_task_contract(
        contract,
        catalog=canonical_catalog,
        policy=compiler_policy,
    )
    if canonical_task.fingerprint != contract.task_spec_fingerprint:
        raise ValueError("TaskSpec/compiled contract fingerprint mismatch")
    if canonical_task.task_id != contract.task_id:
        raise ValueError("TaskSpec/compiled contract identity mismatch")

    issues: list[PropertyIssue] = []
    examples = canonical_task.validation_examples
    total_declared_steps = sum(len(item.steps) for item in examples)
    if len(examples) > policy.max_examples:
        issues.append(_issue("example_limit_exceeded", "Too many examples"))
    if total_declared_steps > policy.max_total_steps:
        issues.append(_issue("example_limit_exceeded", "Too many total steps"))
    for example in examples:
        if len(example.steps) > policy.max_steps_per_example:
            issues.append(
                _issue(
                    "example_limit_exceeded",
                    "Example has too many steps",
                    example_id=example.example_id,
                    path="steps",
                )
            )

    profile_ids = {
        item.reward_profile_id for item in contract.reward_profiles
    }
    rules = {item.code: item.kind for item in contract.terminal_rules}
    rule_fired = {code: False for code in rules}
    rule_clear = {code: False for code in rules}
    goal_true = False
    goal_false = False
    safe_success = False
    active_example = False
    failure_precedence = False
    evaluated_steps = 0

    remaining_step_budget = policy.max_total_steps
    for example_index, example in enumerate(examples):
        if (
            example_index >= policy.max_examples
            or len(example.steps) > policy.max_steps_per_example
            or len(example.steps) > remaining_step_budget
        ):
            continue
        remaining_step_budget -= len(example.steps)
        if example.reward_profile_id not in profile_ids:
            issues.append(
                _issue(
                    "unknown_reward_profile",
                    "Example references an unknown reward profile",
                    example_id=example.example_id,
                    path="reward_profile_id",
                )
            )
            continue
        unknown_codes = set(example.expected_failure_codes).difference(rules)
        if unknown_codes:
            issues.append(
                _issue(
                    "unknown_terminal_code",
                    "Example references an unknown terminal code",
                    example_id=example.example_id,
                    path="expected_failure_codes",
                )
            )
        wrong_safety = {
            code
            for code in example.expected_safety_codes
            if rules.get(code) != "safety"
        }
        if wrong_safety:
            issues.append(
                _issue(
                    "terminal_kind_mismatch",
                    "Expected safety code is not a safety rule",
                    example_id=example.example_id,
                    path="expected_safety_codes",
                )
            )

        replay = _replay(
            example,
            reward_profile_id=example.reward_profile_id,
            contract=contract,
            catalog=canonical_catalog,
            compiler_policy=compiler_policy,
        )
        if replay.error is not None:
            issues.append(
                _issue(
                    "example_replay_invalid",
                    replay.error,
                    example_id=example.example_id,
                )
            )
            continue
        evaluated_steps += len(replay.results)
        if not replay.results:
            issues.append(
                _issue(
                    "example_replay_invalid",
                    "Example produced no results",
                    example_id=example.example_id,
                )
            )
            continue
        final = replay.results[-1]
        if (
            final.status != example.expected_status
            or final.failure_codes != example.expected_failure_codes
            or final.safety_codes != example.expected_safety_codes
        ):
            issues.append(
                _issue(
                    "example_outcome_mismatch",
                    "Final status or terminal codes do not match",
                    example_id=example.example_id,
                )
            )
        safe_success = safe_success or (
            final.status == SUCCESS and not final.failure_codes
        )
        active_example = active_example or final.status == ACTIVE
        for result in replay.results:
            goal_true = goal_true or result.goal_predicate
            goal_false = goal_false or not result.goal_predicate
            failure_precedence = failure_precedence or (
                result.goal_predicate and result.failure
            )
            for rule_result in result.rule_results:
                if rule_result.fired:
                    rule_fired[rule_result.code] = True
                else:
                    rule_clear[rule_result.code] = True
            if not math.isfinite(result.reward):
                issues.append(
                    _issue(
                        "reward_non_finite",
                        "Runtime reward is non-finite",
                        example_id=example.example_id,
                    )
                )
            if result.shaping_clipped is not None:
                profile = next(
                    item
                    for item in contract.reward_profiles
                    if item.reward_profile_id == example.reward_profile_id
                )
                lower, upper = profile.shaping_bounds
                if not lower <= result.shaping_clipped <= upper:
                    issues.append(
                        _issue(
                            "reward_out_of_bounds",
                            "Shaping reward is outside declared bounds",
                            example_id=example.example_id,
                        )
                    )

        if policy.require_deterministic_replay:
            repeated = _replay(
                example,
                reward_profile_id=example.reward_profile_id,
                contract=contract,
                catalog=canonical_catalog,
                compiler_policy=compiler_policy,
            )
            if repeated != replay:
                issues.append(
                    _issue(
                        "replay_nondeterministic",
                        "Repeated trace evaluation changed",
                        example_id=example.example_id,
                    )
                )
        if policy.require_reward_profile_invariance:
            expected_oracle = _oracle_signature(replay)
            for reward_profile_id in sorted(profile_ids):
                alternate = _replay(
                    example,
                    reward_profile_id=reward_profile_id,
                    contract=contract,
                    catalog=canonical_catalog,
                    compiler_policy=compiler_policy,
                )
                if alternate.error is not None or (
                    _oracle_signature(alternate) != expected_oracle
                ):
                    issues.append(
                        _issue(
                            "reward_oracle_coupling",
                            "Reward profile changed terminal semantics",
                            example_id=example.example_id,
                        )
                    )
                    break

    if policy.require_safe_success and not safe_success:
        issues.append(_issue("safe_success_missing", "No safe success trace"))
    if policy.require_active_example and not active_example:
        issues.append(_issue("active_example_missing", "No active trace"))
    if policy.require_goal_positive_and_negative:
        if not goal_true:
            issues.append(_issue("goal_positive_missing", "Goal never becomes true"))
        if not goal_false:
            issues.append(_issue("goal_negative_missing", "Goal never becomes false"))
    if policy.require_each_rule_positive_and_negative:
        for code in sorted(rules):
            if not rule_fired[code] or not rule_clear[code]:
                issues.append(
                    _issue(
                        "terminal_rule_uncovered",
                        f"Rule {code!r} needs fired and clear coverage",
                    )
                )
    if policy.require_failure_precedence_example and not failure_precedence:
        issues.append(
            _issue(
                "failure_precedence_unproven",
                "No trace overlaps goal and failure",
            )
        )

    canonical_issues = _canonical_issues(issues)
    return TaskPropertyReport(
        report_schema_version=1,
        task_id=canonical_task.task_id,
        task_spec_fingerprint=canonical_task.fingerprint,
        contract_fingerprint=contract.fingerprint,
        validation_policy_id=policy.policy_id,
        validation_policy_fingerprint=policy.fingerprint,
        accepted=not canonical_issues,
        issues=canonical_issues,
        example_count=len(examples),
        evaluated_step_count=evaluated_steps,
    )
