"""Deterministic process labels derived from reward-agnostic Event Ledgers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .canonical import canonical_fingerprint, canonical_json, strict_json_loads
from .capabilities import CapabilityCatalog, EventSchema
from .contract import (
    DEFAULT_TASK_COMPILER_POLICY,
    CompiledTaskContract,
    TaskCompilerPolicy,
)
from .event_ledger import (
    DecisionClosedEvent,
    DecisionRootEvent,
    EpisodeEventLedger,
    TransitionEvent,
)
from .expression import MetricFrame, evaluate_expression
from .process_contract import (
    DEFAULT_PROCESS_COMPILER_POLICY,
    CompiledProcessContract,
    ProcessCompilerPolicy,
    validate_compiled_process_contract,
)

ADVANCE = "advance"
HOLD = "hold"
REGRESS = "regress"
ENTERED_UNSAFE = "entered_unsafe"
REMAINED_UNSAFE = "remained_unsafe"
LEFT_UNSAFE = "left_unsafe"
TRANSITION_CODES = frozenset(
    (ADVANCE, HOLD, REGRESS, ENTERED_UNSAFE, REMAINED_UNSAFE, LEFT_UNSAFE)
)
END_REASONS = frozenset(
    ("stage_change", "terminal", "truncated", "max_length", "candidate_end")
)


@dataclass(frozen=True)
class ProcessLabelerPolicy:
    policy_id: str
    algorithm: str
    deterministic_confidence: float

    def validate(self) -> None:
        if self.policy_id != "rexpolicy/process_labeler/v1":
            raise ValueError("Unsupported process labeler policy")
        if self.algorithm != "priority_stage_transition_segments/v1":
            raise ValueError("Unsupported process labeler algorithm")
        if self.deterministic_confidence != 1.0:
            raise ValueError("Deterministic process confidence must be 1.0")

    def to_record(self) -> dict[str, Any]:
        return {
            "policy_id": self.policy_id,
            "algorithm": self.algorithm,
            "deterministic_confidence": self.deterministic_confidence,
            "transition_codes": sorted(TRANSITION_CODES),
            "end_reasons": sorted(END_REASONS),
        }

    @property
    def fingerprint(self) -> str:
        return canonical_fingerprint(self.to_record())


DEFAULT_PROCESS_LABELER_POLICY = ProcessLabelerPolicy(
    policy_id="rexpolicy/process_labeler/v1",
    algorithm="priority_stage_transition_segments/v1",
    deterministic_confidence=1.0,
)


@dataclass(frozen=True)
class TransitionProcessLabel:
    source_sequence: int
    sample_id: str
    decision_index: int
    world: int
    branch_step: int
    stage_before: str
    stage_after: str
    transition_code: str
    segment_index: int
    confidence: float

    def to_record(self) -> dict[str, Any]:
        return {
            "source_sequence": self.source_sequence,
            "sample_id": self.sample_id,
            "decision_index": self.decision_index,
            "world": self.world,
            "branch_step": self.branch_step,
            "stage_before": self.stage_before,
            "stage_after": self.stage_after,
            "transition_code": self.transition_code,
            "segment_index": self.segment_index,
            "confidence": self.confidence,
        }


@dataclass(frozen=True)
class ProcessSegment:
    sample_id: str
    decision_index: int
    world: int
    segment_index: int
    stage_id: str
    stage_kind: str
    stage_ordinal: int | None
    start_source_sequence: int
    end_source_sequence: int
    start_branch_step: int
    end_branch_step: int
    step_count: int
    entry_code: str
    end_reason: str

    def to_record(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "decision_index": self.decision_index,
            "world": self.world,
            "segment_index": self.segment_index,
            "stage_id": self.stage_id,
            "stage_kind": self.stage_kind,
            "stage_ordinal": self.stage_ordinal,
            "start_source_sequence": self.start_source_sequence,
            "end_source_sequence": self.end_source_sequence,
            "start_branch_step": self.start_branch_step,
            "end_branch_step": self.end_branch_step,
            "step_count": self.step_count,
            "entry_code": self.entry_code,
            "end_reason": self.end_reason,
        }


@dataclass(frozen=True)
class ProcessLabelView:
    artifact_type: str
    schema_version: int
    episode_id: str
    event_ledger_sha256: str
    source_bindings_sha256: str
    task_contract_id: str
    task_contract_sha256: str
    task_oracle_sha256: str
    process_contract_id: str
    process_contract_sha256: str
    process_spec_id: str
    process_spec_sha256: str
    labeler_policy_id: str
    labeler_policy_sha256: str
    transition_labels: tuple[TransitionProcessLabel, ...]
    segments: tuple[ProcessSegment, ...]

    def to_record(self) -> dict[str, Any]:
        return {
            "artifact_type": self.artifact_type,
            "schema_version": self.schema_version,
            "episode_id": self.episode_id,
            "event_ledger_sha256": self.event_ledger_sha256,
            "source_bindings_sha256": self.source_bindings_sha256,
            "task_contract_id": self.task_contract_id,
            "task_contract_sha256": self.task_contract_sha256,
            "task_oracle_sha256": self.task_oracle_sha256,
            "process_contract_id": self.process_contract_id,
            "process_contract_sha256": self.process_contract_sha256,
            "process_spec_id": self.process_spec_id,
            "process_spec_sha256": self.process_spec_sha256,
            "labeler_policy_id": self.labeler_policy_id,
            "labeler_policy_sha256": self.labeler_policy_sha256,
            "transition_labels": [
                label.to_record() for label in self.transition_labels
            ],
            "segments": [segment.to_record() for segment in self.segments],
        }

    def to_json(self) -> str:
        return canonical_json(self.to_record())

    @property
    def fingerprint(self) -> str:
        return canonical_fingerprint(self.to_record())

    @classmethod
    def from_record(
        cls,
        value: Any,
        *,
        ledger: EpisodeEventLedger,
        task_contract: CompiledTaskContract,
        process_contract: CompiledProcessContract,
        catalog: CapabilityCatalog,
        labeler_policy: ProcessLabelerPolicy = DEFAULT_PROCESS_LABELER_POLICY,
        process_policy: ProcessCompilerPolicy = DEFAULT_PROCESS_COMPILER_POLICY,
        task_policy: TaskCompilerPolicy = DEFAULT_TASK_COMPILER_POLICY,
    ) -> ProcessLabelView:
        """Accept only the exact deterministic view of the bound source ledger."""
        if not isinstance(value, dict):
            raise ValueError("ProcessLabelView must be an object")
        expected = materialize_process_label_view(
            ledger,
            task_contract=task_contract,
            process_contract=process_contract,
            catalog=catalog,
            labeler_policy=labeler_policy,
            process_policy=process_policy,
            task_policy=task_policy,
        )
        try:
            supplied = canonical_json(value)
        except (TypeError, ValueError) as error:
            raise ValueError("ProcessLabelView is not canonical JSON") from error
        if supplied != expected.to_json():
            raise ValueError(
                "ProcessLabelView does not match deterministic materialization"
            )
        return expected

    @classmethod
    def from_json(
        cls,
        text: str,
        **bindings: Any,
    ) -> ProcessLabelView:
        return cls.from_record(strict_json_loads(text), **bindings)


@dataclass(frozen=True)
class _StageState:
    stage_id: str
    kind: str
    ordinal: int | None


def _schema(
    process_contract: CompiledProcessContract,
    catalog: CapabilityCatalog,
) -> EventSchema:
    try:
        return next(
            item
            for item in catalog.event_schemas
            if item.event_schema_id == process_contract.event_schema_id
        )
    except StopIteration as error:
        raise ValueError("Process EventSchema is unavailable") from error


def _metric_frame(
    schema: EventSchema,
    *,
    previous_signals: Mapping[str, Any],
    current_signals: Mapping[str, Any],
) -> MetricFrame:
    return MetricFrame(
        event_schema_id=schema.event_schema_id,
        event_schema_fingerprint=schema.fingerprint,
        previous={
            metric.name: previous_signals[metric.name]
            for metric in schema.metrics
            if "previous" in metric.available_at
        },
        current={
            metric.name: current_signals[metric.name]
            for metric in schema.metrics
            if "current" in metric.available_at
        },
    )


def _classify_stage(
    contract: CompiledProcessContract,
    *,
    schema: EventSchema,
    previous_signals: Mapping[str, Any],
    current_signals: Mapping[str, Any],
) -> _StageState:
    frame = _metric_frame(
        schema,
        previous_signals=previous_signals,
        current_signals=current_signals,
    )
    for stage in contract.stages:
        if evaluate_expression(stage.program, frame=frame, schema=schema):
            return _StageState(stage.stage_id, stage.kind, stage.ordinal)
    return _StageState(
        contract.default_stage_id,
        "progress",
        contract.default_stage_ordinal,
    )


def _transition_code(before: _StageState, after: _StageState) -> str:
    if after.kind == "unsafe":
        return REMAINED_UNSAFE if before.kind == "unsafe" else ENTERED_UNSAFE
    if before.kind == "unsafe":
        return LEFT_UNSAFE
    assert before.ordinal is not None and after.ordinal is not None
    if after.ordinal > before.ordinal:
        return ADVANCE
    if after.ordinal < before.ordinal:
        return REGRESS
    return HOLD


def _candidate_segments(
    *,
    transitions: list[TransitionEvent],
    stages_after: list[_StageState],
    codes: list[str],
    max_segment_steps: int,
) -> tuple[list[int], list[ProcessSegment]]:
    ranges: list[tuple[int, int]] = []
    start = 0
    for index in range(1, len(transitions)):
        if (
            stages_after[index].stage_id != stages_after[index - 1].stage_id
            or index - start >= max_segment_steps
        ):
            ranges.append((start, index - 1))
            start = index
    ranges.append((start, len(transitions) - 1))

    segment_by_transition = [0] * len(transitions)
    segments = []
    for segment_index, (first, last) in enumerate(ranges):
        for transition_index in range(first, last + 1):
            segment_by_transition[transition_index] = segment_index
        final = transitions[last]
        next_index = last + 1
        if final.terminated:
            end_reason = "terminal"
        elif final.truncated:
            end_reason = "truncated"
        elif next_index < len(transitions) and (
            stages_after[next_index].stage_id != stages_after[last].stage_id
        ):
            end_reason = "stage_change"
        elif next_index < len(transitions):
            end_reason = "max_length"
        else:
            end_reason = "candidate_end"
        stage = stages_after[first]
        segments.append(
            ProcessSegment(
                sample_id=final.sample_id,
                decision_index=final.decision_index,
                world=final.world,
                segment_index=segment_index,
                stage_id=stage.stage_id,
                stage_kind=stage.kind,
                stage_ordinal=stage.ordinal,
                start_source_sequence=transitions[first].sequence,
                end_source_sequence=final.sequence,
                start_branch_step=transitions[first].branch_step,
                end_branch_step=final.branch_step,
                step_count=last - first + 1,
                entry_code=codes[first],
                end_reason=end_reason,
            )
        )
    return segment_by_transition, segments


def materialize_process_label_view(
    ledger: EpisodeEventLedger,
    *,
    task_contract: CompiledTaskContract,
    process_contract: CompiledProcessContract,
    catalog: CapabilityCatalog,
    labeler_policy: ProcessLabelerPolicy = DEFAULT_PROCESS_LABELER_POLICY,
    process_policy: ProcessCompilerPolicy = DEFAULT_PROCESS_COMPILER_POLICY,
    task_policy: TaskCompilerPolicy = DEFAULT_TASK_COMPILER_POLICY,
) -> ProcessLabelView:
    """Label every candidate transition without mutating the fact ledger."""
    labeler_policy.validate()
    canonical_catalog = CapabilityCatalog.from_record(catalog.to_record())
    schema = _schema(process_contract, canonical_catalog)
    ledger.validate(event_schema=schema)
    validate_compiled_process_contract(
        process_contract,
        task_contract=task_contract,
        catalog=canonical_catalog,
        process_policy=process_policy,
        task_policy=task_policy,
    )
    if (
        ledger.bindings.task_contract_id != task_contract.task_contract_id
        or ledger.bindings.task_contract_sha256 != task_contract.fingerprint
    ):
        raise ValueError("Process label source TaskContract binding mismatch")
    if (
        ledger.bindings.event_schema_id != process_contract.event_schema_id
        or ledger.bindings.event_schema_sha256
        != process_contract.event_schema_fingerprint
    ):
        raise ValueError("Process label source EventSchema binding mismatch")

    labels: list[TransitionProcessLabel] = []
    segments: list[ProcessSegment] = []
    cursor = 0
    while cursor < len(ledger.events):
        root = ledger.events[cursor]
        if not isinstance(root, DecisionRootEvent):
            raise ValueError("Process label decision root is missing")
        root_signals = dict(root.signals)
        root_stage = _classify_stage(
            process_contract,
            schema=schema,
            previous_signals=root_signals,
            current_signals=root_signals,
        )
        cursor += 1
        candidates: dict[str, list[TransitionEvent]] = {}
        while cursor < len(ledger.events):
            event = ledger.events[cursor]
            cursor += 1
            if isinstance(event, DecisionClosedEvent):
                break
            if not isinstance(event, TransitionEvent):
                raise ValueError("Process label decision contains an invalid event")
            candidates.setdefault(event.sample_id, []).append(event)
        else:
            raise ValueError("Process label decision is not closed")

        for sample_id in sorted(candidates):
            transitions = candidates[sample_id]
            before = root_stage
            previous_signals = root_signals
            after_states = []
            codes = []
            for transition in transitions:
                current_signals = dict(transition.post_signals)
                after = _classify_stage(
                    process_contract,
                    schema=schema,
                    previous_signals=previous_signals,
                    current_signals=current_signals,
                )
                after_states.append(after)
                codes.append(_transition_code(before, after))
                before = after
                previous_signals = current_signals
            segment_indexes, candidate_segments = _candidate_segments(
                transitions=transitions,
                stages_after=after_states,
                codes=codes,
                max_segment_steps=process_contract.max_segment_steps,
            )
            segments.extend(candidate_segments)
            before = root_stage
            for index, transition in enumerate(transitions):
                after = after_states[index]
                labels.append(
                    TransitionProcessLabel(
                        source_sequence=transition.sequence,
                        sample_id=sample_id,
                        decision_index=transition.decision_index,
                        world=transition.world,
                        branch_step=transition.branch_step,
                        stage_before=before.stage_id,
                        stage_after=after.stage_id,
                        transition_code=codes[index],
                        segment_index=segment_indexes[index],
                        confidence=labeler_policy.deterministic_confidence,
                    )
                )
                before = after

    labels.sort(key=lambda item: item.source_sequence)
    segments.sort(
        key=lambda item: (
            item.decision_index,
            item.world,
            item.segment_index,
        )
    )
    transition_count = sum(
        isinstance(event, TransitionEvent) for event in ledger.events
    )
    if len(labels) != transition_count:
        raise ValueError("Process labels do not cover every source transition")
    view = ProcessLabelView(
        artifact_type="rexpolicy_process_label_view",
        schema_version=1,
        episode_id=ledger.episode_id,
        event_ledger_sha256=ledger.fingerprint,
        source_bindings_sha256=canonical_fingerprint(
            ledger.bindings.to_record()
        ),
        task_contract_id=task_contract.task_contract_id,
        task_contract_sha256=task_contract.fingerprint,
        task_oracle_sha256=task_contract.oracle_fingerprint,
        process_contract_id=process_contract.process_contract_id,
        process_contract_sha256=process_contract.fingerprint,
        process_spec_id=process_contract.process_spec_id,
        process_spec_sha256=process_contract.process_spec_fingerprint,
        labeler_policy_id=labeler_policy.policy_id,
        labeler_policy_sha256=labeler_policy.fingerprint,
        transition_labels=tuple(labels),
        segments=tuple(segments),
    )
    canonical_fingerprint(view.to_record())
    return view
