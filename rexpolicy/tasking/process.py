"""Strict ProcessSpec value objects for deterministic task segmentation."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Any

from .canonical import canonical_fingerprint, canonical_json, strict_json_loads
from .model import ExpressionSpec

_IDENTIFIER = re.compile(r"^[a-z][a-z0-9_.-]*(?:/[a-z0-9_.-]+)*$")
_MAX_DESCRIPTION_CHARS = 256


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
        raise ValueError(f"{path} must be a versioned identifier")
    return value


def _description(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{path} must be non-empty text")
    if value != value.strip() or len(value) > _MAX_DESCRIPTION_CHARS:
        raise ValueError(f"{path} must be trimmed and at most 256 characters")
    if unicodedata.normalize("NFC", value) != value:
        raise ValueError(f"{path} must use NFC Unicode normalization")
    if any(unicodedata.category(character).startswith("C") for character in value):
        raise ValueError(f"{path} cannot contain control characters")
    return value


def _ordinal(value: Any, path: str, *, required: bool) -> int | None:
    if value is None and not required:
        return None
    if type(value) is not int or value < 0:
        suffix = "non-negative integer" if required else "non-negative integer or null"
        raise ValueError(f"{path} must be a {suffix}")
    return value


@dataclass(frozen=True)
class DefaultProcessStage:
    stage_id: str
    ordinal: int
    description: str

    @classmethod
    def from_record(cls, value: Any, path: str) -> DefaultProcessStage:
        record = _mapping(value, path)
        _keys(record, {"stage_id", "kind", "ordinal", "description"}, path)
        if record["kind"] != "progress":
            raise ValueError(f"{path}.kind must be progress")
        ordinal = _ordinal(record["ordinal"], f"{path}.ordinal", required=True)
        assert ordinal is not None
        return cls(
            stage_id=_identifier(record["stage_id"], f"{path}.stage_id"),
            ordinal=ordinal,
            description=_description(record["description"], f"{path}.description"),
        )

    def to_record(self) -> dict[str, Any]:
        return {
            "stage_id": self.stage_id,
            "kind": "progress",
            "ordinal": self.ordinal,
            "description": self.description,
        }


@dataclass(frozen=True)
class ProcessStage:
    stage_id: str
    kind: str
    ordinal: int | None
    description: str
    predicate: ExpressionSpec

    @classmethod
    def from_record(cls, value: Any, path: str) -> ProcessStage:
        record = _mapping(value, path)
        _keys(
            record,
            {"stage_id", "kind", "ordinal", "description", "predicate"},
            path,
        )
        kind = _identifier(record["kind"], f"{path}.kind")
        if kind not in {"progress", "unsafe"}:
            raise ValueError(f"{path}.kind must be progress or unsafe")
        ordinal = _ordinal(
            record["ordinal"],
            f"{path}.ordinal",
            required=kind == "progress",
        )
        if kind == "unsafe" and ordinal is not None:
            raise ValueError(f"{path}.ordinal must be null for unsafe stages")
        return cls(
            stage_id=_identifier(record["stage_id"], f"{path}.stage_id"),
            kind=kind,
            ordinal=ordinal,
            description=_description(record["description"], f"{path}.description"),
            predicate=ExpressionSpec.from_record(
                record["predicate"],
                f"{path}.predicate",
            ),
        )

    def to_record(self) -> dict[str, Any]:
        return {
            "stage_id": self.stage_id,
            "kind": self.kind,
            "ordinal": self.ordinal,
            "description": self.description,
            "predicate": self.predicate.to_record(),
        }


@dataclass(frozen=True)
class ProcessSegmentation:
    max_segment_steps: int

    @classmethod
    def from_record(cls, value: Any, path: str) -> ProcessSegmentation:
        record = _mapping(value, path)
        _keys(record, {"max_segment_steps"}, path)
        maximum = record["max_segment_steps"]
        if type(maximum) is not int or not 1 <= maximum <= 256:
            raise ValueError(f"{path}.max_segment_steps must be between 1 and 256")
        return cls(max_segment_steps=maximum)

    def to_record(self) -> dict[str, Any]:
        return {"max_segment_steps": self.max_segment_steps}


@dataclass(frozen=True)
class ProcessSpecV1:
    schema_version: int
    process_spec_id: str
    task_id: str
    event_schema_id: str
    default_stage: DefaultProcessStage
    stages: tuple[ProcessStage, ...]
    segmentation: ProcessSegmentation

    @classmethod
    def from_record(cls, value: Any) -> ProcessSpecV1:
        record = _mapping(value, "$")
        _keys(
            record,
            {
                "schema_version",
                "process_spec_id",
                "task_id",
                "event_schema_id",
                "default_stage",
                "stages",
                "segmentation",
            },
            "$",
        )
        if record["schema_version"] != 1:
            raise ValueError("ProcessSpec schema_version must be 1")
        raw_stages = record["stages"]
        if not isinstance(raw_stages, list) or not raw_stages:
            raise ValueError("$.stages must be a non-empty priority-ordered array")
        stages = tuple(
            ProcessStage.from_record(item, f"$.stages[{index}]")
            for index, item in enumerate(raw_stages)
        )
        default = DefaultProcessStage.from_record(
            record["default_stage"],
            "$.default_stage",
        )
        if default.ordinal != 0:
            raise ValueError("ProcessSpec default progress ordinal must be zero")
        progress_seen = False
        for stage in stages:
            if stage.kind == "progress":
                progress_seen = True
            elif progress_seen:
                raise ValueError(
                    "ProcessSpec unsafe stages must precede progress stages"
                )
        stage_ids = (default.stage_id, *(stage.stage_id for stage in stages))
        if len(set(stage_ids)) != len(stage_ids):
            raise ValueError("ProcessSpec stage IDs must be unique")
        ordinals = (
            default.ordinal,
            *(stage.ordinal for stage in stages if stage.kind == "progress"),
        )
        if len(set(ordinals)) != len(ordinals):
            raise ValueError("ProcessSpec progress ordinals must be unique")
        if set(ordinals) != set(range(len(ordinals))):
            raise ValueError("ProcessSpec progress ordinals must be contiguous from zero")
        return cls(
            schema_version=1,
            process_spec_id=_identifier(
                record["process_spec_id"],
                "$.process_spec_id",
            ),
            task_id=_identifier(record["task_id"], "$.task_id"),
            event_schema_id=_identifier(
                record["event_schema_id"],
                "$.event_schema_id",
            ),
            default_stage=default,
            stages=stages,
            segmentation=ProcessSegmentation.from_record(
                record["segmentation"],
                "$.segmentation",
            ),
        )

    @classmethod
    def from_json(cls, text: str) -> ProcessSpecV1:
        return cls.from_record(strict_json_loads(text))

    def to_record(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "process_spec_id": self.process_spec_id,
            "task_id": self.task_id,
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


def decode_process_spec_json(text: str) -> ProcessSpecV1:
    """Decode one strict ProcessSpec JSON document."""
    return ProcessSpecV1.from_json(text)
