"""Deterministically compile raw Event Ledger facts into QD descriptors.

The compiler deliberately has no reward or model interface.  Every emitted
metric is selected by an immutable, hash-bound aggregation policy and is
computed only from the referenced candidate's numeric Event Ledger signals or
its executed ``effective_action_19d`` sequence.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from rexpolicy.tasking.canonical import (
    canonical_fingerprint,
    canonical_json,
    strict_json_loads,
)
from rexpolicy.tasking.capabilities import EventSchema
from rexpolicy.tasking.event_ledger import (
    DecisionRootEvent,
    EpisodeEventLedger,
    TransitionEvent,
)

from .experience import format_sample_id
from .quality_diversity import (
    QualityDiversityPolicy,
    SuccessBehaviorDescriptor,
)
from .success_archive import SuccessReference

_IDENTIFIER = re.compile(r"^[a-z][a-z0-9_.-]*(?:/[a-z0-9_.-]+)*$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")

_SIGNAL_OPERATORS = frozenset(
    {
        "signal.initial",
        "signal.final",
        "signal.minimum",
        "signal.maximum",
        "signal.delta",
    }
)
_ACTION_OPERATORS = frozenset(
    {
        "effective_action_19d.path_length_l2",
        "effective_action_19d.control_step_count",
    }
)
_ALLOWED_OPERATORS = _SIGNAL_OPERATORS | _ACTION_OPERATORS


def _mapping(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{path} must be an object")
    return value


def _keys(record: Mapping[str, Any], expected: set[str], path: str) -> None:
    missing = sorted(expected.difference(record))
    unknown = sorted(set(record).difference(expected))
    if missing:
        raise ValueError(f"{path} is missing fields: {', '.join(missing)}")
    if unknown:
        raise ValueError(f"{path} has unknown fields: {', '.join(unknown)}")


def _identifier(value: Any, path: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise ValueError(f"{path} must be a versioned identifier")
    return value


def _sha256(value: Any, path: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{path} must be a lowercase SHA-256")
    return value


def _finite(value: Any, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{path} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{path} must be finite")
    return number


@dataclass(frozen=True)
class BehaviorMetricAggregation:
    """One allowlisted, directly auditable raw-fact aggregation."""

    metric_id: str
    operator: str
    signal_name: str | None
    minimum: float
    maximum: float

    @classmethod
    def from_record(
        cls,
        value: Any,
        path: str,
    ) -> BehaviorMetricAggregation:
        record = _mapping(value, path)
        _keys(
            record,
            {"metric_id", "operator", "signal_name", "minimum", "maximum"},
            path,
        )
        metric = cls(
            metric_id=_identifier(record["metric_id"], f"{path}.metric_id"),
            operator=record["operator"],
            signal_name=record["signal_name"],
            minimum=_finite(record["minimum"], f"{path}.minimum"),
            maximum=_finite(record["maximum"], f"{path}.maximum"),
        )
        metric.validate()
        return metric

    def validate(self) -> None:
        _identifier(self.metric_id, "metric_id")
        if (
            not isinstance(self.operator, str)
            or self.operator not in _ALLOWED_OPERATORS
        ):
            raise ValueError(
                f"Unsupported behavior aggregation operator {self.operator!r}"
            )
        if self.operator in _SIGNAL_OPERATORS:
            _identifier(self.signal_name, "signal_name")
        elif self.signal_name is not None:
            raise ValueError("Action aggregation signal_name must be null")
        minimum = _finite(self.minimum, "minimum")
        maximum = _finite(self.maximum, "maximum")
        if minimum > maximum:
            raise ValueError("Behavior metric aggregation bounds are reversed")

    def to_record(self) -> dict[str, Any]:
        self.validate()
        return {
            "metric_id": self.metric_id,
            "operator": self.operator,
            "signal_name": self.signal_name,
            "minimum": self.minimum,
            "maximum": self.maximum,
        }


@dataclass(frozen=True)
class QualityDescriptorCompilationPolicy:
    """Hash-bound aggregation recipes for one exact QD index policy."""

    schema_version: int
    compiler_policy_id: str
    quality_diversity_policy_sha256: str
    metrics: tuple[BehaviorMetricAggregation, ...]

    @classmethod
    def from_record(cls, value: Any) -> QualityDescriptorCompilationPolicy:
        path = "$quality_descriptor_compilation_policy"
        record = _mapping(value, path)
        _keys(
            record,
            {
                "schema_version",
                "compiler_policy_id",
                "quality_diversity_policy_sha256",
                "metrics",
            },
            path,
        )
        raw_metrics = record["metrics"]
        if not isinstance(raw_metrics, list) or not raw_metrics:
            raise ValueError(f"{path}.metrics must be a non-empty array")
        policy = cls(
            schema_version=record["schema_version"],
            compiler_policy_id=_identifier(
                record["compiler_policy_id"],
                f"{path}.compiler_policy_id",
            ),
            quality_diversity_policy_sha256=_sha256(
                record["quality_diversity_policy_sha256"],
                f"{path}.quality_diversity_policy_sha256",
            ),
            metrics=tuple(
                BehaviorMetricAggregation.from_record(
                    item,
                    f"{path}.metrics[{index}]",
                )
                for index, item in enumerate(raw_metrics)
            ),
        )
        policy.validate()
        return policy

    @classmethod
    def from_json(cls, text: str) -> QualityDescriptorCompilationPolicy:
        return cls.from_record(strict_json_loads(text))

    def validate(self) -> None:
        if self.schema_version != 1:
            raise ValueError("Quality descriptor compiler schema_version must be 1")
        _identifier(self.compiler_policy_id, "compiler_policy_id")
        _sha256(
            self.quality_diversity_policy_sha256,
            "quality_diversity_policy_sha256",
        )
        for metric in self.metrics:
            metric.validate()
        metric_ids = tuple(metric.metric_id for metric in self.metrics)
        if not metric_ids or metric_ids != tuple(sorted(set(metric_ids))):
            raise ValueError("Compiler metrics must be non-empty, sorted, and unique")

    def validate_for(
        self,
        *,
        policy: QualityDiversityPolicy,
        event_schema: EventSchema,
    ) -> None:
        self.validate()
        policy.validate()
        if self.quality_diversity_policy_sha256 != policy.fingerprint:
            raise ValueError("Compiler policy belongs to another QD policy")
        required = {
            *(dimension.dimension_id for dimension in policy.dimensions),
            policy.quality_metric_id,
        }
        actual = {metric.metric_id for metric in self.metrics}
        if actual != required:
            raise ValueError("Compiler metric set does not exactly match QD policy")
        schema_by_name = event_schema.by_name
        dimension_by_id = {
            dimension.dimension_id: dimension for dimension in policy.dimensions
        }
        for metric in self.metrics:
            dimension = dimension_by_id.get(metric.metric_id)
            if dimension is not None and (
                metric.minimum != dimension.minimum
                or metric.maximum != dimension.maximum
            ):
                raise ValueError(
                    "Compiler metric range does not match QD dimension range"
                )
            if metric.operator not in _SIGNAL_OPERATORS:
                continue
            try:
                signal = schema_by_name[metric.signal_name]
            except KeyError as error:
                raise ValueError(
                    f"Compiler signal {metric.signal_name!r} is absent from EventSchema"
                ) from error
            if signal.value_type != "number":
                raise ValueError("Behavior signal aggregations require numeric signals")
            if "current" not in signal.available_at:
                raise ValueError("Behavior signal is unavailable in current frames")
            if metric.operator != "signal.delta":
                if (
                    signal.minimum is not None
                    and metric.minimum < signal.minimum
                ):
                    raise ValueError(
                        "Compiler metric range exceeds EventSchema minimum"
                    )
                if (
                    signal.maximum is not None
                    and metric.maximum > signal.maximum
                ):
                    raise ValueError(
                        "Compiler metric range exceeds EventSchema maximum"
                    )

    def to_record(self) -> dict[str, Any]:
        self.validate()
        return {
            "schema_version": self.schema_version,
            "compiler_policy_id": self.compiler_policy_id,
            "quality_diversity_policy_sha256": (
                self.quality_diversity_policy_sha256
            ),
            "metrics": [metric.to_record() for metric in self.metrics],
        }

    def to_json(self) -> str:
        return canonical_json(self.to_record())

    @property
    def fingerprint(self) -> str:
        return canonical_fingerprint(self.to_record())


@dataclass(frozen=True)
class InitialStateGroupBinding:
    """Bind an external initial-state group artifact to one exact ledger."""

    schema_version: int
    initial_state_group_id: str
    initial_state_group_sha256: str
    event_ledger_sha256: str
    reset_recipe_sha256: str

    @classmethod
    def from_record(cls, value: Any) -> InitialStateGroupBinding:
        path = "$initial_state_group_binding"
        record = _mapping(value, path)
        _keys(
            record,
            {
                "schema_version",
                "initial_state_group_id",
                "initial_state_group_sha256",
                "event_ledger_sha256",
                "reset_recipe_sha256",
            },
            path,
        )
        binding = cls(
            schema_version=record["schema_version"],
            initial_state_group_id=_identifier(
                record["initial_state_group_id"],
                f"{path}.initial_state_group_id",
            ),
            initial_state_group_sha256=_sha256(
                record["initial_state_group_sha256"],
                f"{path}.initial_state_group_sha256",
            ),
            event_ledger_sha256=_sha256(
                record["event_ledger_sha256"],
                f"{path}.event_ledger_sha256",
            ),
            reset_recipe_sha256=_sha256(
                record["reset_recipe_sha256"],
                f"{path}.reset_recipe_sha256",
            ),
        )
        binding.validate()
        return binding

    @classmethod
    def from_json(cls, text: str) -> InitialStateGroupBinding:
        return cls.from_record(strict_json_loads(text))

    @classmethod
    def for_ledger(
        cls,
        *,
        initial_state_group_id: str,
        initial_state_group_sha256: str,
        event_ledger: EpisodeEventLedger,
    ) -> InitialStateGroupBinding:
        return cls.from_record(
            {
                "schema_version": 1,
                "initial_state_group_id": initial_state_group_id,
                "initial_state_group_sha256": initial_state_group_sha256,
                "event_ledger_sha256": event_ledger.fingerprint,
                "reset_recipe_sha256": canonical_fingerprint(
                    event_ledger.reset_recipe.to_record()
                ),
            }
        )

    def validate(self) -> None:
        if self.schema_version != 1:
            raise ValueError("Initial-state group binding schema_version must be 1")
        _identifier(self.initial_state_group_id, "initial_state_group_id")
        _sha256(self.initial_state_group_sha256, "initial_state_group_sha256")
        _sha256(self.event_ledger_sha256, "event_ledger_sha256")
        _sha256(self.reset_recipe_sha256, "reset_recipe_sha256")

    def validate_for(self, event_ledger: EpisodeEventLedger) -> None:
        self.validate()
        if self.event_ledger_sha256 != event_ledger.fingerprint:
            raise ValueError("Initial-state group is bound to another Event Ledger")
        expected_reset = canonical_fingerprint(event_ledger.reset_recipe.to_record())
        if self.reset_recipe_sha256 != expected_reset:
            raise ValueError("Initial-state group reset-recipe binding mismatch")

    def to_record(self) -> dict[str, Any]:
        self.validate()
        return {
            "schema_version": self.schema_version,
            "initial_state_group_id": self.initial_state_group_id,
            "initial_state_group_sha256": self.initial_state_group_sha256,
            "event_ledger_sha256": self.event_ledger_sha256,
            "reset_recipe_sha256": self.reset_recipe_sha256,
        }

    def to_json(self) -> str:
        return canonical_json(self.to_record())

    @property
    def fingerprint(self) -> str:
        return canonical_fingerprint(self.to_record())


def _canonical_ledger(
    event_ledger: EpisodeEventLedger,
    *,
    event_schema: EventSchema,
) -> EpisodeEventLedger:
    if not isinstance(event_ledger, EpisodeEventLedger):
        raise ValueError("event_ledger must be an EpisodeEventLedger")
    event_ledger.validate(event_schema=event_schema)
    restored = EpisodeEventLedger.from_record(
        event_ledger.to_record(),
        event_schema=event_schema,
    )
    if restored != event_ledger:
        raise ValueError("Event Ledger is not canonical")
    return restored


def _canonical_reference(reference: SuccessReference) -> SuccessReference:
    if not isinstance(reference, SuccessReference):
        raise ValueError("success reference must be a SuccessReference")
    reference.validate()
    restored = SuccessReference.from_record(reference.to_record())
    if restored != reference:
        raise ValueError("Success reference is not canonical")
    # Force strict canonical JSON number validation before any metric work.
    canonical_fingerprint(restored.to_record())
    return restored


def _candidate_events(
    *,
    reference: SuccessReference,
    ledger: EpisodeEventLedger,
) -> tuple[DecisionRootEvent, tuple[TransitionEvent, ...]]:
    provenance = ledger.collection
    if (
        reference.source_generation != provenance.data_generation
        or reference.source_rank != provenance.rank
        or reference.source_episode != provenance.episode
    ):
        raise ValueError("Success reference and Event Ledger provenance mismatch")
    expected_sample_id = format_sample_id(
        reference.source_generation,
        reference.source_rank,
        reference.source_episode,
        reference.decision,
        reference.world,
    )
    if reference.sample_id != expected_sample_id:
        raise ValueError("Success reference sample_id/decision/world mismatch")
    roots = tuple(
        event
        for event in ledger.events
        if isinstance(event, DecisionRootEvent)
        and event.decision_index == reference.decision
    )
    transitions = tuple(
        event
        for event in ledger.events
        if isinstance(event, TransitionEvent)
        and event.sample_id == reference.sample_id
    )
    if len(roots) != 1 or not transitions:
        raise ValueError("Success reference has no exact Event Ledger candidate")
    if any(
        event.decision_index != reference.decision
        or event.world != reference.world
        for event in transitions
    ):
        raise ValueError("Event Ledger candidate decision/world binding mismatch")
    if tuple(event.branch_step for event in transitions) != tuple(
        range(len(transitions))
    ):
        raise ValueError("Event Ledger candidate branch steps are not canonical")
    return roots[0], transitions


def _signal_metric(
    aggregation: BehaviorMetricAggregation,
    *,
    root: DecisionRootEvent,
    transitions: tuple[TransitionEvent, ...],
) -> float:
    values = (
        _finite(dict(root.signals)[aggregation.signal_name], aggregation.metric_id),
        *(
            _finite(
                dict(event.post_signals)[aggregation.signal_name],
                aggregation.metric_id,
            )
            for event in transitions
        ),
    )
    if aggregation.operator == "signal.initial":
        return values[0]
    if aggregation.operator == "signal.final":
        return values[-1]
    if aggregation.operator == "signal.minimum":
        return min(values)
    if aggregation.operator == "signal.maximum":
        return max(values)
    if aggregation.operator == "signal.delta":
        return _finite(values[-1] - values[0], aggregation.metric_id)
    raise AssertionError(aggregation.operator)


def _action_metric(
    aggregation: BehaviorMetricAggregation,
    *,
    transitions: tuple[TransitionEvent, ...],
) -> float:
    if aggregation.operator == "effective_action_19d.control_step_count":
        return float(len(transitions))
    if aggregation.operator == "effective_action_19d.path_length_l2":
        # Treat the zero vector as the explicit path origin, then sum the L2
        # distance between consecutive actually executed 19-D actions.
        previous = (0.0,) * 19
        segments = []
        for event in transitions:
            current = event.effective_action_19d
            segment = math.hypot(
                *(current[index] - previous[index] for index in range(19))
            )
            segments.append(segment)
            previous = current
        return _finite(math.fsum(segments), aggregation.metric_id)
    raise AssertionError(aggregation.operator)


def _compile_metrics(
    *,
    policy: QualityDiversityPolicy,
    compiler_policy: QualityDescriptorCompilationPolicy,
    root: DecisionRootEvent,
    transitions: tuple[TransitionEvent, ...],
) -> tuple[tuple[str, float], ...]:
    output = []
    for aggregation in compiler_policy.metrics:
        if aggregation.operator in _SIGNAL_OPERATORS:
            value = _signal_metric(
                aggregation,
                root=root,
                transitions=transitions,
            )
        else:
            value = _action_metric(aggregation, transitions=transitions)
        value = _finite(value, aggregation.metric_id)
        if value < aggregation.minimum or value > aggregation.maximum:
            raise ValueError(
                f"Behavior metric {aggregation.metric_id!r} is outside its "
                "compiler-policy range"
            )
        output.append((aggregation.metric_id, value))
    values = dict(output)
    for dimension in policy.dimensions:
        # ``bucket`` is the canonical QD policy range check, including the
        # closed upper boundary semantics used by the eventual index.
        dimension.bucket(values[dimension.dimension_id])
    return tuple(output)


def compile_success_behavior_descriptor(
    *,
    policy: QualityDiversityPolicy,
    compiler_policy: QualityDescriptorCompilationPolicy,
    success_reference: SuccessReference,
    event_ledger: EpisodeEventLedger,
    event_schema: EventSchema,
    reward_profile_sha256: str,
    initial_state_group: InitialStateGroupBinding,
) -> SuccessBehaviorDescriptor:
    """Compile one exact success candidate without consulting reward values."""
    reference = _canonical_reference(success_reference)
    ledger = _canonical_ledger(event_ledger, event_schema=event_schema)
    compiler_policy.validate_for(policy=policy, event_schema=event_schema)
    initial_state_group.validate_for(ledger)
    reward_sha256 = _sha256(reward_profile_sha256, "reward_profile_sha256")
    if reference.task_id != policy.task_id:
        raise ValueError("Success reference task does not match QD policy")
    if ledger.bindings.task_oracle_sha256 != policy.task_oracle_sha256:
        raise ValueError("Event Ledger task oracle does not match QD policy")
    root, transitions = _candidate_events(reference=reference, ledger=ledger)
    metrics = _compile_metrics(
        policy=policy,
        compiler_policy=compiler_policy,
        root=root,
        transitions=transitions,
    )
    descriptor = SuccessBehaviorDescriptor(
        schema_version=1,
        sample_id=reference.sample_id,
        success_reference_sha256=canonical_fingerprint(reference.to_record()),
        event_ledger_sha256=ledger.fingerprint,
        compilation_policy_sha256=compiler_policy.fingerprint,
        task_id=reference.task_id,
        task_oracle_sha256=ledger.bindings.task_oracle_sha256,
        reward_profile_id=reference.reward_profile_id,
        reward_profile_sha256=reward_sha256,
        initial_state_group_id=initial_state_group.initial_state_group_id,
        initial_state_group_sha256=(
            initial_state_group.initial_state_group_sha256
        ),
        metrics=metrics,
    )
    return SuccessBehaviorDescriptor.from_record(descriptor.to_record())


def compile_success_behavior_descriptors(
    *,
    policy: QualityDiversityPolicy,
    compiler_policy: QualityDescriptorCompilationPolicy,
    success_references: Iterable[SuccessReference],
    event_ledger: EpisodeEventLedger,
    event_schema: EventSchema,
    reward_profile_sha256_by_id: Mapping[str, str],
    initial_state_group: InitialStateGroupBinding,
) -> tuple[SuccessBehaviorDescriptor, ...]:
    """Compile every referenced success in one ledger, including non-elites."""
    references = tuple(_canonical_reference(item) for item in success_references)
    if not references:
        raise ValueError("At least one success reference is required")
    sample_ids = tuple(item.sample_id for item in references)
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("Success references must have unique sample IDs")
    fingerprints = dict(reward_profile_sha256_by_id)
    reward_ids = {item.reward_profile_id for item in references}
    if set(fingerprints) != reward_ids:
        raise ValueError("Reward-profile fingerprint set must be exact")
    descriptors = tuple(
        compile_success_behavior_descriptor(
            policy=policy,
            compiler_policy=compiler_policy,
            success_reference=reference,
            event_ledger=event_ledger,
            event_schema=event_schema,
            reward_profile_sha256=fingerprints[reference.reward_profile_id],
            initial_state_group=initial_state_group,
        )
        for reference in sorted(references, key=lambda item: item.sample_id)
    )
    if len(descriptors) != len(references):
        raise AssertionError("Quality descriptor compiler dropped a success")
    return descriptors
