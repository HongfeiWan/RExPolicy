"""Auditable shadow contracts for learned process-reward models.

This module deliberately stops at prediction materialization.  Learned scores are
diagnostic sidecar data: they cannot terminate an episode, assert task success or
safety, mutate an Event Ledger, or activate a runtime reward profile.
"""

from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import dataclass
from typing import Any

from .canonical import canonical_fingerprint, canonical_json, strict_json_loads
from .process_label_view import ProcessLabelView, TransitionProcessLabel

_IDENTIFIER = re.compile(r"^[a-z][a-z0-9_.-]*(?:/[a-z0-9_.-]+)*$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_MAX_TEXT_CHARS = 256
_MAX_ACCEPTED_CALIBRATION_ERROR = 0.10

TASK_SPEC_SCHEMA_ID = "rexpolicy/task_spec/v2"
TASK_SPEC_SCHEMA_FINGERPRINT = canonical_fingerprint(
    {
        "schema_id": TASK_SPEC_SCHEMA_ID,
        "schema_version": 2,
        "required_sections": [
            "environment",
            "instructions",
            "action_projection",
            "goal",
            "terminal_rules",
            "reward_profiles",
            "validation_examples",
        ],
        "canonicalization": "strict_json_canonical_sha256/v1",
    }
)

PROCESS_LABEL_SCHEMA_ID = "rexpolicy/process_label_view/v1"
PROCESS_LABEL_SCHEMA_FINGERPRINT = canonical_fingerprint(
    {
        "schema_id": PROCESS_LABEL_SCHEMA_ID,
        "artifact_type": "rexpolicy_process_label_view",
        "schema_version": 1,
        "top_level_binding_fields": [
            "event_ledger_sha256",
            "task_contract_sha256",
            "task_oracle_sha256",
            "process_contract_sha256",
            "process_spec_sha256",
            "labeler_policy_sha256",
        ],
        "transition_label_fields": [
            "source_sequence",
            "sample_id",
            "decision_index",
            "world",
            "branch_step",
            "stage_before",
            "stage_after",
            "transition_code",
            "segment_index",
            "confidence",
        ],
        "segment_fields": [
            "sample_id",
            "decision_index",
            "world",
            "segment_index",
            "stage_id",
            "stage_kind",
            "stage_ordinal",
            "start_source_sequence",
            "end_source_sequence",
            "start_branch_step",
            "end_branch_step",
            "step_count",
            "entry_code",
            "end_reason",
        ],
        "canonicalization": "strict_json_canonical_sha256/v1",
    }
)

CONFIDENCE_SEMANTICS = "calibrated_probability_correct/v1"
SHADOW_MODE = "shadow_only"
ORACLE_AUTHORITY = "none"
REWARD_ACTIVATION = "forbidden"


def _mapping(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{path} must be an object")
    return value


def _keys(record: dict[str, Any], expected: set[str], path: str) -> None:
    missing = sorted(expected.difference(record))
    unknown = sorted(set(record).difference(expected))
    if missing:
        raise ValueError(f"{path} is missing fields: {', '.join(missing)}")
    if unknown:
        raise ValueError(f"{path} has unknown fields: {', '.join(unknown)}")


def _identifier(value: Any, path: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) > _MAX_TEXT_CHARS
        or _IDENTIFIER.fullmatch(value) is None
    ):
        raise ValueError(f"{path} must be a bounded versioned identifier")
    return value


def _text(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value or len(value) > _MAX_TEXT_CHARS:
        raise ValueError(f"{path} must be bounded non-empty text")
    if value != value.strip() or unicodedata.normalize("NFC", value) != value:
        raise ValueError(f"{path} must be trimmed NFC text")
    if any(unicodedata.category(character).startswith("C") for character in value):
        raise ValueError(f"{path} cannot contain control characters")
    return value


def _sha256(value: Any, path: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{path} must be a lowercase SHA-256")
    return value


def _positive_integer(value: Any, path: str) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{path} must be a positive integer")
    return value


def _non_negative_integer(value: Any, path: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{path} must be a non-negative integer")
    return value


def _bounded_number(
    value: Any,
    path: str,
    *,
    minimum: float,
    maximum: float,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{path} must be numeric")
    result = float(value)
    if not math.isfinite(result) or not minimum <= result <= maximum:
        raise ValueError(f"{path} must be finite and between {minimum} and {maximum}")
    return result


def _canonical_input(record: dict[str, Any], restored: dict[str, Any], name: str) -> None:
    try:
        supplied = canonical_json(record)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} is not canonical JSON") from error
    if supplied != canonical_json(restored):
        raise ValueError(f"{name} does not use its canonical representation")


@dataclass(frozen=True)
class TaskModelBinding:
    task_schema_id: str
    task_schema_sha256: str
    task_id: str
    task_spec_sha256: str
    task_contract_id: str
    task_contract_sha256: str
    task_oracle_sha256: str

    @classmethod
    def from_record(cls, value: Any, path: str) -> TaskModelBinding:
        record = _mapping(value, path)
        _keys(
            record,
            {
                "task_schema_id",
                "task_schema_sha256",
                "task_id",
                "task_spec_sha256",
                "task_contract_id",
                "task_contract_sha256",
                "task_oracle_sha256",
            },
            path,
        )
        if record["task_schema_id"] != TASK_SPEC_SCHEMA_ID:
            raise ValueError(f"{path}.task_schema_id is unsupported")
        if record["task_schema_sha256"] != TASK_SPEC_SCHEMA_FINGERPRINT:
            raise ValueError(f"{path}.task_schema_sha256 mismatch")
        return cls(
            task_schema_id=TASK_SPEC_SCHEMA_ID,
            task_schema_sha256=TASK_SPEC_SCHEMA_FINGERPRINT,
            task_id=_identifier(record["task_id"], f"{path}.task_id"),
            task_spec_sha256=_sha256(
                record["task_spec_sha256"], f"{path}.task_spec_sha256"
            ),
            task_contract_id=_identifier(
                record["task_contract_id"], f"{path}.task_contract_id"
            ),
            task_contract_sha256=_sha256(
                record["task_contract_sha256"],
                f"{path}.task_contract_sha256",
            ),
            task_oracle_sha256=_sha256(
                record["task_oracle_sha256"],
                f"{path}.task_oracle_sha256",
            ),
        )

    def to_record(self) -> dict[str, Any]:
        return {
            "task_schema_id": self.task_schema_id,
            "task_schema_sha256": self.task_schema_sha256,
            "task_id": self.task_id,
            "task_spec_sha256": self.task_spec_sha256,
            "task_contract_id": self.task_contract_id,
            "task_contract_sha256": self.task_contract_sha256,
            "task_oracle_sha256": self.task_oracle_sha256,
        }


@dataclass(frozen=True)
class ProcessLabelModelBinding:
    process_label_schema_id: str
    process_label_schema_sha256: str
    process_contract_id: str
    process_contract_sha256: str
    process_spec_id: str
    process_spec_sha256: str
    labeler_policy_id: str
    labeler_policy_sha256: str

    @classmethod
    def from_record(cls, value: Any, path: str) -> ProcessLabelModelBinding:
        record = _mapping(value, path)
        _keys(
            record,
            {
                "process_label_schema_id",
                "process_label_schema_sha256",
                "process_contract_id",
                "process_contract_sha256",
                "process_spec_id",
                "process_spec_sha256",
                "labeler_policy_id",
                "labeler_policy_sha256",
            },
            path,
        )
        if record["process_label_schema_id"] != PROCESS_LABEL_SCHEMA_ID:
            raise ValueError(f"{path}.process_label_schema_id is unsupported")
        if (
            record["process_label_schema_sha256"]
            != PROCESS_LABEL_SCHEMA_FINGERPRINT
        ):
            raise ValueError(f"{path}.process_label_schema_sha256 mismatch")
        return cls(
            process_label_schema_id=PROCESS_LABEL_SCHEMA_ID,
            process_label_schema_sha256=PROCESS_LABEL_SCHEMA_FINGERPRINT,
            process_contract_id=_identifier(
                record["process_contract_id"], f"{path}.process_contract_id"
            ),
            process_contract_sha256=_sha256(
                record["process_contract_sha256"],
                f"{path}.process_contract_sha256",
            ),
            process_spec_id=_identifier(
                record["process_spec_id"], f"{path}.process_spec_id"
            ),
            process_spec_sha256=_sha256(
                record["process_spec_sha256"], f"{path}.process_spec_sha256"
            ),
            labeler_policy_id=_identifier(
                record["labeler_policy_id"], f"{path}.labeler_policy_id"
            ),
            labeler_policy_sha256=_sha256(
                record["labeler_policy_sha256"],
                f"{path}.labeler_policy_sha256",
            ),
        )

    def to_record(self) -> dict[str, Any]:
        return {
            "process_label_schema_id": self.process_label_schema_id,
            "process_label_schema_sha256": self.process_label_schema_sha256,
            "process_contract_id": self.process_contract_id,
            "process_contract_sha256": self.process_contract_sha256,
            "process_spec_id": self.process_spec_id,
            "process_spec_sha256": self.process_spec_sha256,
            "labeler_policy_id": self.labeler_policy_id,
            "labeler_policy_sha256": self.labeler_policy_sha256,
        }


@dataclass(frozen=True)
class LabelProvenance:
    provenance_id: str
    method_id: str
    annotation_policy_sha256: str
    source_labeler_policy_id: str
    source_labeler_policy_sha256: str

    @classmethod
    def from_record(cls, value: Any, path: str) -> LabelProvenance:
        record = _mapping(value, path)
        _keys(
            record,
            {
                "provenance_id",
                "method_id",
                "annotation_policy_sha256",
                "source_labeler_policy_id",
                "source_labeler_policy_sha256",
            },
            path,
        )
        return cls(
            provenance_id=_identifier(
                record["provenance_id"], f"{path}.provenance_id"
            ),
            method_id=_identifier(record["method_id"], f"{path}.method_id"),
            annotation_policy_sha256=_sha256(
                record["annotation_policy_sha256"],
                f"{path}.annotation_policy_sha256",
            ),
            source_labeler_policy_id=_identifier(
                record["source_labeler_policy_id"],
                f"{path}.source_labeler_policy_id",
            ),
            source_labeler_policy_sha256=_sha256(
                record["source_labeler_policy_sha256"],
                f"{path}.source_labeler_policy_sha256",
            ),
        )

    def to_record(self) -> dict[str, Any]:
        return {
            "provenance_id": self.provenance_id,
            "method_id": self.method_id,
            "annotation_policy_sha256": self.annotation_policy_sha256,
            "source_labeler_policy_id": self.source_labeler_policy_id,
            "source_labeler_policy_sha256": self.source_labeler_policy_sha256,
        }

    @property
    def fingerprint(self) -> str:
        return canonical_fingerprint(self.to_record())


@dataclass(frozen=True)
class DatasetCommitment:
    split: str
    dataset_id: str
    example_count: int
    dataset_sha256: str
    ordered_process_label_views_sha256: str
    label_provenance_sha256: str

    @classmethod
    def from_record(
        cls,
        value: Any,
        path: str,
        *,
        expected_split: str,
        expected_provenance_sha256: str,
    ) -> DatasetCommitment:
        record = _mapping(value, path)
        _keys(
            record,
            {
                "split",
                "dataset_id",
                "example_count",
                "dataset_sha256",
                "ordered_process_label_views_sha256",
                "label_provenance_sha256",
            },
            path,
        )
        if record["split"] != expected_split:
            raise ValueError(f"{path}.split must be {expected_split}")
        provenance_sha256 = _sha256(
            record["label_provenance_sha256"],
            f"{path}.label_provenance_sha256",
        )
        if provenance_sha256 != expected_provenance_sha256:
            raise ValueError(f"{path} label provenance fingerprint mismatch")
        return cls(
            split=expected_split,
            dataset_id=_identifier(record["dataset_id"], f"{path}.dataset_id"),
            example_count=_positive_integer(
                record["example_count"], f"{path}.example_count"
            ),
            dataset_sha256=_sha256(
                record["dataset_sha256"], f"{path}.dataset_sha256"
            ),
            ordered_process_label_views_sha256=_sha256(
                record["ordered_process_label_views_sha256"],
                f"{path}.ordered_process_label_views_sha256",
            ),
            label_provenance_sha256=provenance_sha256,
        )

    def to_record(self) -> dict[str, Any]:
        return {
            "split": self.split,
            "dataset_id": self.dataset_id,
            "example_count": self.example_count,
            "dataset_sha256": self.dataset_sha256,
            "ordered_process_label_views_sha256": (
                self.ordered_process_label_views_sha256
            ),
            "label_provenance_sha256": self.label_provenance_sha256,
        }


@dataclass(frozen=True)
class CalibrationCommitment:
    method_id: str
    audit_dataset_sha256: str
    report_sha256: str
    sample_count: int
    expected_calibration_error: float
    maximum_expected_calibration_error: float

    @classmethod
    def from_record(
        cls,
        value: Any,
        path: str,
        *,
        audit_dataset: DatasetCommitment,
    ) -> CalibrationCommitment:
        record = _mapping(value, path)
        _keys(
            record,
            {
                "method_id",
                "audit_dataset_sha256",
                "report_sha256",
                "sample_count",
                "expected_calibration_error",
                "maximum_expected_calibration_error",
            },
            path,
        )
        dataset_sha256 = _sha256(
            record["audit_dataset_sha256"], f"{path}.audit_dataset_sha256"
        )
        if dataset_sha256 != audit_dataset.dataset_sha256:
            raise ValueError(f"{path} audit dataset fingerprint mismatch")
        sample_count = _positive_integer(
            record["sample_count"], f"{path}.sample_count"
        )
        if sample_count != audit_dataset.example_count:
            raise ValueError(f"{path}.sample_count must cover the audit dataset")
        maximum = _bounded_number(
            record["maximum_expected_calibration_error"],
            f"{path}.maximum_expected_calibration_error",
            minimum=0.0,
            maximum=_MAX_ACCEPTED_CALIBRATION_ERROR,
        )
        if maximum == 0.0:
            raise ValueError(
                f"{path}.maximum_expected_calibration_error must be positive"
            )
        observed = _bounded_number(
            record["expected_calibration_error"],
            f"{path}.expected_calibration_error",
            minimum=0.0,
            maximum=maximum,
        )
        return cls(
            method_id=_identifier(record["method_id"], f"{path}.method_id"),
            audit_dataset_sha256=dataset_sha256,
            report_sha256=_sha256(
                record["report_sha256"], f"{path}.report_sha256"
            ),
            sample_count=sample_count,
            expected_calibration_error=observed,
            maximum_expected_calibration_error=maximum,
        )

    def to_record(self) -> dict[str, Any]:
        return {
            "method_id": self.method_id,
            "audit_dataset_sha256": self.audit_dataset_sha256,
            "report_sha256": self.report_sha256,
            "sample_count": self.sample_count,
            "expected_calibration_error": self.expected_calibration_error,
            "maximum_expected_calibration_error": (
                self.maximum_expected_calibration_error
            ),
        }


@dataclass(frozen=True)
class ProcessRewardModelManifest:
    artifact_type: str
    schema_version: int
    model_id: str
    model_family_id: str
    task: TaskModelBinding
    process_labels: ProcessLabelModelBinding
    label_provenance: LabelProvenance
    train_dataset: DatasetCommitment
    audit_dataset: DatasetCommitment
    score_bounds: tuple[float, float]
    calibration: CalibrationCommitment
    confidence_semantics: str
    mode: str
    oracle_authority: str
    reward_activation: str

    @classmethod
    def from_record(cls, value: Any) -> ProcessRewardModelManifest:
        record = _mapping(value, "$process_reward_model")
        _keys(
            record,
            {
                "artifact_type",
                "schema_version",
                "model_id",
                "model_family_id",
                "task",
                "process_labels",
                "label_provenance",
                "datasets",
                "output_contract",
                "calibration",
                "usage_policy",
            },
            "$process_reward_model",
        )
        if record["artifact_type"] != "rexpolicy_process_reward_model_manifest":
            raise ValueError("Unsupported process reward model artifact_type")
        if record["schema_version"] != 1:
            raise ValueError("Process reward model schema_version must be 1")

        task = TaskModelBinding.from_record(record["task"], "$.task")
        process_labels = ProcessLabelModelBinding.from_record(
            record["process_labels"], "$.process_labels"
        )
        provenance = LabelProvenance.from_record(
            record["label_provenance"], "$.label_provenance"
        )
        if (
            provenance.source_labeler_policy_id
            != process_labels.labeler_policy_id
            or provenance.source_labeler_policy_sha256
            != process_labels.labeler_policy_sha256
        ):
            raise ValueError("Model label provenance does not bind its labeler policy")

        datasets = _mapping(record["datasets"], "$.datasets")
        _keys(datasets, {"train", "audit"}, "$.datasets")
        train = DatasetCommitment.from_record(
            datasets["train"],
            "$.datasets.train",
            expected_split="train",
            expected_provenance_sha256=provenance.fingerprint,
        )
        audit = DatasetCommitment.from_record(
            datasets["audit"],
            "$.datasets.audit",
            expected_split="audit",
            expected_provenance_sha256=provenance.fingerprint,
        )
        if train.dataset_id == audit.dataset_id:
            raise ValueError("Train and audit dataset IDs must be disjoint")
        if train.dataset_sha256 == audit.dataset_sha256:
            raise ValueError("Train and audit dataset commitments must be disjoint")
        if (
            train.ordered_process_label_views_sha256
            == audit.ordered_process_label_views_sha256
        ):
            raise ValueError("Train and audit source-view commitments must be disjoint")

        output = _mapping(record["output_contract"], "$.output_contract")
        _keys(
            output,
            {"score_min", "score_max", "confidence_semantics"},
            "$.output_contract",
        )
        score_min = _bounded_number(
            output["score_min"], "$.output_contract.score_min", minimum=-1.0, maximum=1.0
        )
        score_max = _bounded_number(
            output["score_max"], "$.output_contract.score_max", minimum=-1.0, maximum=1.0
        )
        if score_min >= score_max:
            raise ValueError("Process reward score_min must be less than score_max")
        if output["confidence_semantics"] != CONFIDENCE_SEMANTICS:
            raise ValueError("Unsupported process reward confidence semantics")

        usage = _mapping(record["usage_policy"], "$.usage_policy")
        _keys(
            usage,
            {"mode", "oracle_authority", "reward_activation"},
            "$.usage_policy",
        )
        if usage != {
            "mode": SHADOW_MODE,
            "oracle_authority": ORACLE_AUTHORITY,
            "reward_activation": REWARD_ACTIVATION,
        }:
            raise ValueError(
                "Process reward models must remain shadow-only without oracle "
                "or reward authority"
            )

        result = cls(
            artifact_type="rexpolicy_process_reward_model_manifest",
            schema_version=1,
            model_id=_identifier(record["model_id"], "$.model_id"),
            model_family_id=_identifier(
                record["model_family_id"], "$.model_family_id"
            ),
            task=task,
            process_labels=process_labels,
            label_provenance=provenance,
            train_dataset=train,
            audit_dataset=audit,
            score_bounds=(score_min, score_max),
            calibration=CalibrationCommitment.from_record(
                record["calibration"], "$.calibration", audit_dataset=audit
            ),
            confidence_semantics=CONFIDENCE_SEMANTICS,
            mode=SHADOW_MODE,
            oracle_authority=ORACLE_AUTHORITY,
            reward_activation=REWARD_ACTIVATION,
        )
        _canonical_input(record, result.to_record(), "Process reward model manifest")
        return result

    @classmethod
    def from_json(cls, text: str) -> ProcessRewardModelManifest:
        return cls.from_record(strict_json_loads(text))

    def to_record(self) -> dict[str, Any]:
        return {
            "artifact_type": self.artifact_type,
            "schema_version": self.schema_version,
            "model_id": self.model_id,
            "model_family_id": self.model_family_id,
            "task": self.task.to_record(),
            "process_labels": self.process_labels.to_record(),
            "label_provenance": self.label_provenance.to_record(),
            "datasets": {
                "train": self.train_dataset.to_record(),
                "audit": self.audit_dataset.to_record(),
            },
            "output_contract": {
                "score_min": self.score_bounds[0],
                "score_max": self.score_bounds[1],
                "confidence_semantics": self.confidence_semantics,
            },
            "calibration": self.calibration.to_record(),
            "usage_policy": {
                "mode": self.mode,
                "oracle_authority": self.oracle_authority,
                "reward_activation": self.reward_activation,
            },
        }

    def to_json(self) -> str:
        return canonical_json(self.to_record())

    @property
    def fingerprint(self) -> str:
        return canonical_fingerprint(self.to_record())


@dataclass(frozen=True)
class ProcessRewardPrediction:
    source_sequence: int
    source_label_sha256: str
    sample_id: str
    decision_index: int
    world: int
    branch_step: int
    segment_index: int
    process_score: float
    calibrated_confidence: float

    @classmethod
    def from_record(
        cls,
        value: Any,
        path: str,
        *,
        source_label: TransitionProcessLabel,
        score_bounds: tuple[float, float],
    ) -> ProcessRewardPrediction:
        record = _mapping(value, path)
        _keys(
            record,
            {
                "source_sequence",
                "source_label_sha256",
                "sample_id",
                "decision_index",
                "world",
                "branch_step",
                "segment_index",
                "process_score",
                "calibrated_confidence",
            },
            path,
        )
        expected_identity = {
            "source_sequence": source_label.source_sequence,
            "source_label_sha256": canonical_fingerprint(source_label.to_record()),
            "sample_id": source_label.sample_id,
            "decision_index": source_label.decision_index,
            "world": source_label.world,
            "branch_step": source_label.branch_step,
            "segment_index": source_label.segment_index,
        }
        for key, expected in expected_identity.items():
            if record[key] != expected:
                raise ValueError(f"{path}.{key} does not match its ProcessLabelView")
        result = cls(
            source_sequence=source_label.source_sequence,
            source_label_sha256=expected_identity["source_label_sha256"],
            sample_id=_text(source_label.sample_id, f"{path}.sample_id"),
            decision_index=_non_negative_integer(
                source_label.decision_index, f"{path}.decision_index"
            ),
            world=_non_negative_integer(source_label.world, f"{path}.world"),
            branch_step=_non_negative_integer(
                source_label.branch_step, f"{path}.branch_step"
            ),
            segment_index=_non_negative_integer(
                source_label.segment_index, f"{path}.segment_index"
            ),
            process_score=_bounded_number(
                record["process_score"],
                f"{path}.process_score",
                minimum=score_bounds[0],
                maximum=score_bounds[1],
            ),
            calibrated_confidence=_bounded_number(
                record["calibrated_confidence"],
                f"{path}.calibrated_confidence",
                minimum=0.0,
                maximum=1.0,
            ),
        )
        _canonical_input(record, result.to_record(), "Process reward prediction")
        return result

    def to_record(self) -> dict[str, Any]:
        return {
            "source_sequence": self.source_sequence,
            "source_label_sha256": self.source_label_sha256,
            "sample_id": self.sample_id,
            "decision_index": self.decision_index,
            "world": self.world,
            "branch_step": self.branch_step,
            "segment_index": self.segment_index,
            "process_score": self.process_score,
            "calibrated_confidence": self.calibrated_confidence,
        }


@dataclass(frozen=True)
class ProcessRewardShadowView:
    artifact_type: str
    schema_version: int
    mode: str
    episode_id: str
    event_ledger_sha256: str
    process_label_schema_id: str
    process_label_schema_sha256: str
    process_label_view_sha256: str
    model_id: str
    model_manifest_sha256: str
    checkpoint_sha256: str
    confidence_semantics: str
    oracle_authority: str
    reward_activation: str
    predictions: tuple[ProcessRewardPrediction, ...]

    @classmethod
    def from_record(
        cls,
        value: Any,
        *,
        process_labels: ProcessLabelView,
        manifest: ProcessRewardModelManifest,
    ) -> ProcessRewardShadowView:
        record = _mapping(value, "$process_reward_shadow")
        _keys(
            record,
            {
                "artifact_type",
                "schema_version",
            "mode",
            "episode_id",
            "event_ledger_sha256",
            "process_label_schema_id",
                "process_label_schema_sha256",
                "process_label_view_sha256",
                "model_id",
                "model_manifest_sha256",
                "checkpoint_sha256",
                "confidence_semantics",
                "oracle_authority",
                "reward_activation",
                "predictions",
            },
            "$process_reward_shadow",
        )
        canonical_manifest = ProcessRewardModelManifest.from_record(
            manifest.to_record()
        )
        _validate_label_bindings(process_labels, canonical_manifest)
        expected = {
            "artifact_type": "rexpolicy_process_reward_shadow_view",
            "schema_version": 1,
            "mode": SHADOW_MODE,
            "episode_id": process_labels.episode_id,
            "event_ledger_sha256": process_labels.event_ledger_sha256,
            "process_label_schema_id": PROCESS_LABEL_SCHEMA_ID,
            "process_label_schema_sha256": PROCESS_LABEL_SCHEMA_FINGERPRINT,
            "process_label_view_sha256": process_labels.fingerprint,
            "model_id": canonical_manifest.model_id,
            "model_manifest_sha256": canonical_manifest.fingerprint,
            "confidence_semantics": CONFIDENCE_SEMANTICS,
            "oracle_authority": ORACLE_AUTHORITY,
            "reward_activation": REWARD_ACTIVATION,
        }
        for key, expected_value in expected.items():
            if record[key] != expected_value:
                raise ValueError(
                    f"Process reward shadow {key} does not match its binding"
                )
        checkpoint_sha256 = _sha256(
            record["checkpoint_sha256"], "$.checkpoint_sha256"
        )
        raw_predictions = record["predictions"]
        if not isinstance(raw_predictions, list):
            raise ValueError("$.predictions must be an array")
        if len(raw_predictions) != len(process_labels.transition_labels):
            raise ValueError(
                "Process reward predictions must cover every transition label"
            )
        predictions = tuple(
            ProcessRewardPrediction.from_record(
                raw,
                f"$.predictions[{index}]",
                source_label=label,
                score_bounds=canonical_manifest.score_bounds,
            )
            for index, (raw, label) in enumerate(
                zip(raw_predictions, process_labels.transition_labels)
            )
        )
        result = cls(
            artifact_type=expected["artifact_type"],
            schema_version=1,
            mode=SHADOW_MODE,
            episode_id=_text(process_labels.episode_id, "$.episode_id"),
            event_ledger_sha256=process_labels.event_ledger_sha256,
            process_label_schema_id=PROCESS_LABEL_SCHEMA_ID,
            process_label_schema_sha256=PROCESS_LABEL_SCHEMA_FINGERPRINT,
            process_label_view_sha256=process_labels.fingerprint,
            model_id=canonical_manifest.model_id,
            model_manifest_sha256=canonical_manifest.fingerprint,
            checkpoint_sha256=checkpoint_sha256,
            confidence_semantics=CONFIDENCE_SEMANTICS,
            oracle_authority=ORACLE_AUTHORITY,
            reward_activation=REWARD_ACTIVATION,
            predictions=predictions,
        )
        _canonical_input(record, result.to_record(), "Process reward shadow view")
        return result

    @classmethod
    def from_json(
        cls,
        text: str,
        *,
        process_labels: ProcessLabelView,
        manifest: ProcessRewardModelManifest,
    ) -> ProcessRewardShadowView:
        return cls.from_record(
            strict_json_loads(text),
            process_labels=process_labels,
            manifest=manifest,
        )

    def to_record(self) -> dict[str, Any]:
        return {
            "artifact_type": self.artifact_type,
            "schema_version": self.schema_version,
            "mode": self.mode,
            "episode_id": self.episode_id,
            "event_ledger_sha256": self.event_ledger_sha256,
            "process_label_schema_id": self.process_label_schema_id,
            "process_label_schema_sha256": self.process_label_schema_sha256,
            "process_label_view_sha256": self.process_label_view_sha256,
            "model_id": self.model_id,
            "model_manifest_sha256": self.model_manifest_sha256,
            "checkpoint_sha256": self.checkpoint_sha256,
            "confidence_semantics": self.confidence_semantics,
            "oracle_authority": self.oracle_authority,
            "reward_activation": self.reward_activation,
            "predictions": [prediction.to_record() for prediction in self.predictions],
        }

    def to_json(self) -> str:
        return canonical_json(self.to_record())

    @property
    def fingerprint(self) -> str:
        return canonical_fingerprint(self.to_record())


def _validate_label_bindings(
    process_labels: ProcessLabelView,
    manifest: ProcessRewardModelManifest,
) -> None:
    if (
        process_labels.artifact_type != "rexpolicy_process_label_view"
        or process_labels.schema_version != 1
    ):
        raise ValueError("Unsupported ProcessLabelView schema")
    task = manifest.task
    labels = manifest.process_labels
    if (
        process_labels.task_contract_id != task.task_contract_id
        or process_labels.task_contract_sha256 != task.task_contract_sha256
        or process_labels.task_oracle_sha256 != task.task_oracle_sha256
    ):
        raise ValueError("Process reward Task binding mismatch")
    if (
        process_labels.process_contract_id != labels.process_contract_id
        or process_labels.process_contract_sha256 != labels.process_contract_sha256
        or process_labels.process_spec_id != labels.process_spec_id
        or process_labels.process_spec_sha256 != labels.process_spec_sha256
    ):
        raise ValueError("Process reward ProcessSpec binding mismatch")
    if (
        process_labels.labeler_policy_id != labels.labeler_policy_id
        or process_labels.labeler_policy_sha256 != labels.labeler_policy_sha256
    ):
        raise ValueError("Process reward labeler policy binding mismatch")


def materialize_process_reward_shadow(
    process_labels: ProcessLabelView,
    *,
    manifest: ProcessRewardModelManifest,
    checkpoint_sha256: str,
    outputs: list[dict[str, Any]],
) -> ProcessRewardShadowView:
    """Bind untrusted model outputs into a non-activating shadow sidecar."""
    canonical_manifest = ProcessRewardModelManifest.from_record(manifest.to_record())
    _validate_label_bindings(process_labels, canonical_manifest)
    if not isinstance(outputs, list):
        raise ValueError("Process reward outputs must be an array")
    if len(outputs) != len(process_labels.transition_labels):
        raise ValueError("Process reward outputs must cover every transition label")
    predictions = []
    for index, (output, label) in enumerate(
        zip(outputs, process_labels.transition_labels)
    ):
        path = f"$outputs[{index}]"
        output_record = _mapping(output, path)
        _keys(
            output_record,
            {"source_sequence", "process_score", "calibrated_confidence"},
            path,
        )
        if output_record["source_sequence"] != label.source_sequence:
            raise ValueError(f"{path}.source_sequence does not match ProcessLabelView")
        predictions.append(
            {
                "source_sequence": label.source_sequence,
                "source_label_sha256": canonical_fingerprint(label.to_record()),
                "sample_id": label.sample_id,
                "decision_index": label.decision_index,
                "world": label.world,
                "branch_step": label.branch_step,
                "segment_index": label.segment_index,
                "process_score": output_record["process_score"],
                "calibrated_confidence": output_record["calibrated_confidence"],
            }
        )
    record = {
        "artifact_type": "rexpolicy_process_reward_shadow_view",
        "schema_version": 1,
        "mode": SHADOW_MODE,
        "episode_id": process_labels.episode_id,
        "event_ledger_sha256": process_labels.event_ledger_sha256,
        "process_label_schema_id": PROCESS_LABEL_SCHEMA_ID,
        "process_label_schema_sha256": PROCESS_LABEL_SCHEMA_FINGERPRINT,
        "process_label_view_sha256": process_labels.fingerprint,
        "model_id": canonical_manifest.model_id,
        "model_manifest_sha256": canonical_manifest.fingerprint,
        "checkpoint_sha256": _sha256(checkpoint_sha256, "$checkpoint_sha256"),
        "confidence_semantics": CONFIDENCE_SEMANTICS,
        "oracle_authority": ORACLE_AUTHORITY,
        "reward_activation": REWARD_ACTIVATION,
        "predictions": predictions,
    }
    return ProcessRewardShadowView.from_record(
        record,
        process_labels=process_labels,
        manifest=canonical_manifest,
    )
