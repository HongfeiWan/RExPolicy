"""Reward/oracle shadow evaluation against Groot Newton raw event signals."""

from __future__ import annotations

import math
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

from rexpolicy.tasking.capabilities import (
    CapabilityCatalog,
    EventSchema,
    validate_event_values,
)
from rexpolicy.tasking.contract import CompiledTaskContract
from rexpolicy.tasking.expression import MetricFrame
from rexpolicy.tasking.runtime import (
    TaskRuntimeState,
    TaskStepResult,
    evaluate_task_step,
)

_GROOT_REACH_FIELDS = {
    "reach.distance_m": "reach_distance",
    "reach.contact_violation": "reach_contact_violation",
    "reach.displacement_violation": "reach_displacement_violation",
}


def _vector(value: Any, name: str) -> list[Any]:
    for method in ("detach", "cpu"):
        function = getattr(value, method, None)
        if callable(function):
            value = function()
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        value = tolist()
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(f"Groot event field {name!r} must be a non-empty vector")
    if any(isinstance(item, (list, tuple, dict)) for item in value):
        raise ValueError(f"Groot event field {name!r} must be one-dimensional")
    return list(value)


def extract_groot_reach_metric_rows(
    source: Mapping[str, Any],
    *,
    event_schema: EventSchema,
) -> tuple[Mapping[str, Any], ...]:
    """Extract current raw metrics without reading reward/previous aliases."""
    expected = {
        metric.name
        for metric in event_schema.metrics
        if "current" in metric.available_at
    }
    if expected != set(_GROOT_REACH_FIELDS):
        raise ValueError("EventSchema is not the certified Groot Reach v1 schema")
    columns = {}
    for metric_name, source_name in _GROOT_REACH_FIELDS.items():
        try:
            columns[metric_name] = _vector(source[source_name], source_name)
        except KeyError as error:
            raise ValueError(f"Groot event source omits {source_name!r}") from error
    lengths = {len(values) for values in columns.values()}
    if len(lengths) != 1:
        raise ValueError("Groot event fields have inconsistent world counts")
    rows = []
    for world in range(next(iter(lengths))):
        row = {
            name: values[world] for name, values in columns.items()
        }
        validate_event_values(row, schema=event_schema, at="current")
        rows.append(MappingProxyType(row))
    return tuple(rows)


def require_identical_root_rows(
    rows: tuple[Mapping[str, Any], ...],
) -> Mapping[str, Any]:
    """Require candidate worlds to begin from one canonical physical root."""
    if not rows:
        raise ValueError("Shadow root requires at least one world")
    first = dict(rows[0])
    if any(dict(row) != first for row in rows[1:]):
        raise ValueError("Candidate worlds do not share identical root metrics")
    return MappingProxyType(first)


def _previous_row(
    current: Mapping[str, Any],
    schema: EventSchema,
) -> Mapping[str, Any]:
    previous = {
        metric.name: current[metric.name]
        for metric in schema.metrics
        if "previous" in metric.available_at
    }
    validate_event_values(previous, schema=schema, at="previous")
    return MappingProxyType(previous)


@dataclass(frozen=True)
class ShadowBranchState:
    world_states: tuple[TaskRuntimeState, ...]
    previous_rows: tuple[Mapping[str, Any], ...]


@dataclass(frozen=True)
class ShadowTransitionBatch:
    next_branch: ShadowBranchState
    results: tuple[TaskStepResult | None, ...]
    current_rows: tuple[Mapping[str, Any], ...]


def begin_shadow_decision(
    *,
    root_state: TaskRuntimeState,
    root_source: Mapping[str, Any],
    event_schema: EventSchema,
) -> ShadowBranchState:
    """Clone one actual-path task state across same-state candidate worlds."""
    rows = extract_groot_reach_metric_rows(
        root_source,
        event_schema=event_schema,
    )
    require_identical_root_rows(rows)
    return ShadowBranchState(
        world_states=tuple(root_state for _ in rows),
        previous_rows=tuple(_previous_row(row, event_schema) for row in rows),
    )


def evaluate_shadow_transition(
    branch: ShadowBranchState,
    *,
    transition_source: Mapping[str, Any],
    active_before: tuple[bool, ...],
    contract: CompiledTaskContract,
    catalog: CapabilityCatalog,
) -> ShadowTransitionBatch:
    """Evaluate only worlds active before the environment transition."""
    schema = next(
        item
        for item in catalog.event_schemas
        if item.event_schema_id == contract.event_schema_id
    )
    current_rows = extract_groot_reach_metric_rows(
        transition_source,
        event_schema=schema,
    )
    world_count = len(branch.world_states)
    if (
        len(branch.previous_rows) != world_count
        or len(current_rows) != world_count
        or len(active_before) != world_count
        or any(type(item) is not bool for item in active_before)
    ):
        raise ValueError("Shadow transition world counts are inconsistent")
    next_states = []
    next_previous = []
    results = []
    for world in range(world_count):
        if not active_before[world]:
            next_states.append(branch.world_states[world])
            next_previous.append(branch.previous_rows[world])
            results.append(None)
            continue
        result = evaluate_task_step(
            contract,
            state=branch.world_states[world],
            frame=MetricFrame(
                event_schema_id=schema.event_schema_id,
                event_schema_fingerprint=schema.fingerprint,
                previous=branch.previous_rows[world],
                current=current_rows[world],
            ),
            catalog=catalog,
        )
        next_states.append(result.next_state)
        next_previous.append(_previous_row(current_rows[world], schema))
        results.append(result)
    return ShadowTransitionBatch(
        next_branch=ShadowBranchState(
            world_states=tuple(next_states),
            previous_rows=tuple(next_previous),
        ),
        results=tuple(results),
        current_rows=current_rows,
    )


@dataclass(frozen=True)
class ShadowParityResult:
    accepted: bool
    mismatches: tuple[str, ...]


def compare_groot_step_parity(
    shadow: TaskStepResult,
    *,
    environment_reward: float,
    environment_success: bool,
    environment_failure: bool,
    environment_terminated: bool,
    environment_truncated: bool,
    reward_tolerance: float = 1.0e-6,
) -> ShadowParityResult:
    """Compare legacy environment outputs with compiled v2 semantics."""
    values = (
        environment_success,
        environment_failure,
        environment_terminated,
        environment_truncated,
    )
    if any(type(value) is not bool for value in values):
        raise ValueError("Environment parity flags must be Boolean")
    if (
        not math.isfinite(environment_reward)
        or not math.isfinite(reward_tolerance)
        or reward_tolerance < 0.0
    ):
        raise ValueError("Environment parity reward values are invalid")
    mismatches = []
    if abs(float(environment_reward) - shadow.reward) > reward_tolerance:
        mismatches.append("reward")
    if environment_success != shadow.success:
        mismatches.append("success")
    if environment_failure != shadow.failure:
        mismatches.append("failure")
    if environment_terminated != (shadow.success or shadow.failure):
        mismatches.append("terminated")
    if environment_truncated != shadow.truncated:
        mismatches.append("truncated")
    return ShadowParityResult(
        accepted=not mismatches,
        mismatches=tuple(mismatches),
    )
