"""Reusable, task-agnostic templates for strict ProcessSpec artifacts."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from .canonical import canonical_fingerprint, canonical_json, strict_json_loads
from .process import (
    DefaultProcessStage,
    ProcessSegmentation,
    ProcessSpecV1,
    ProcessStage,
)

_IDENTIFIER = re.compile(r"^[a-z][a-z0-9_.-]*(?:/[a-z0-9_.-]+)*$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_VALIDATION_PROCESS_SPEC_ID = "process_template_validation/v1"
_VALIDATION_TASK_ID = "process_template_validation/v1"


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
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise ValueError(f"{path} must be an identifier")
    return value


def _sha256(value: Any, path: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{path} must be a lowercase SHA-256 digest")
    return value


@dataclass(frozen=True)
class ProcessTemplateV1:
    """Strict task-agnostic source for one family of ProcessSpec artifacts."""

    schema_version: int
    template_id: str
    family_id: str
    event_schema_id: str
    default_stage: DefaultProcessStage
    stages: tuple[ProcessStage, ...]
    segmentation: ProcessSegmentation

    @classmethod
    def from_record(
        cls,
        value: Any,
        *,
        expected_fingerprint: str | None = None,
    ) -> ProcessTemplateV1:
        record = _mapping(value, "$process_template")
        _keys(
            record,
            {
                "schema_version",
                "template_id",
                "family_id",
                "event_schema_id",
                "default_stage",
                "stages",
                "segmentation",
            },
            "$process_template",
        )
        if (
            type(record["schema_version"]) is not int
            or record["schema_version"] != 1
        ):
            raise ValueError("ProcessTemplate schema_version must be 1")
        if expected_fingerprint is not None:
            _sha256(expected_fingerprint, "expected_fingerprint")
            if canonical_fingerprint(record) != expected_fingerprint:
                raise ValueError("ProcessTemplate fingerprint mismatch")

        template_id = _identifier(
            record["template_id"],
            "$process_template.template_id",
        )
        family_id = _identifier(
            record["family_id"],
            "$process_template.family_id",
        )
        event_schema_id = _identifier(
            record["event_schema_id"],
            "$process_template.event_schema_id",
        )

        # Route every stage and segmentation rule through ProcessSpecV1 so a
        # template cannot silently acquire weaker parsing or ordering semantics.
        parsed = ProcessSpecV1.from_record(
            {
                "schema_version": 1,
                "process_spec_id": _VALIDATION_PROCESS_SPEC_ID,
                "task_id": _VALIDATION_TASK_ID,
                "event_schema_id": event_schema_id,
                "default_stage": record["default_stage"],
                "stages": record["stages"],
                "segmentation": record["segmentation"],
            }
        )
        return cls(
            schema_version=1,
            template_id=template_id,
            family_id=family_id,
            event_schema_id=event_schema_id,
            default_stage=parsed.default_stage,
            stages=parsed.stages,
            segmentation=parsed.segmentation,
        )

    @classmethod
    def from_json(
        cls,
        text: str,
        *,
        expected_fingerprint: str | None = None,
    ) -> ProcessTemplateV1:
        return cls.from_record(
            strict_json_loads(text),
            expected_fingerprint=expected_fingerprint,
        )

    def to_record(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "template_id": self.template_id,
            "family_id": self.family_id,
            "event_schema_id": self.event_schema_id,
            "default_stage": self.default_stage.to_record(),
            "stages": [stage.to_record() for stage in self.stages],
            "segmentation": self.segmentation.to_record(),
        }

    def to_json(self) -> str:
        return canonical_json(self.to_record())

    @property
    def fingerprint(self) -> str:
        return canonical_fingerprint(self.to_record())


def instantiate_process_spec(
    template: ProcessTemplateV1,
    task_id: str,
    process_spec_id: str,
    *,
    expected_template_id: str | None = None,
    expected_family_id: str | None = None,
    expected_event_schema_id: str | None = None,
) -> ProcessSpecV1:
    """Instantiate a canonical ProcessSpec while enforcing optional bindings."""

    if not isinstance(template, ProcessTemplateV1):
        raise TypeError("template must be a ProcessTemplateV1")
    canonical_template = ProcessTemplateV1.from_record(template.to_record())

    bindings = (
        (
            "template identity",
            canonical_template.template_id,
            expected_template_id,
            "$expected_template_id",
        ),
        (
            "family",
            canonical_template.family_id,
            expected_family_id,
            "$expected_family_id",
        ),
        (
            "EventSchema",
            canonical_template.event_schema_id,
            expected_event_schema_id,
            "$expected_event_schema_id",
        ),
    )
    for label, actual, expected, path in bindings:
        if expected is None:
            continue
        expected = _identifier(expected, path)
        if actual != expected:
            raise ValueError(f"ProcessTemplate {label} binding mismatch")

    # ProcessSpecV1 validates both caller-selected target IDs and the complete
    # materialized artifact. No template metadata is copied into the output.
    return ProcessSpecV1.from_record(
        {
            "schema_version": 1,
            "process_spec_id": process_spec_id,
            "task_id": task_id,
            "event_schema_id": canonical_template.event_schema_id,
            "default_stage": canonical_template.default_stage.to_record(),
            "stages": [stage.to_record() for stage in canonical_template.stages],
            "segmentation": canonical_template.segmentation.to_record(),
        }
    )


def decode_process_template_json(text: str) -> ProcessTemplateV1:
    """Decode one strict ProcessTemplate JSON document."""

    return ProcessTemplateV1.from_json(text)
