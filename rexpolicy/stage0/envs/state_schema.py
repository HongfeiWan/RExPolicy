"""Versioned 79D state contract for the Newton Stage 0 adapter."""

from __future__ import annotations

import re
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from ..config import STAGE0_STATE_DIM, STAGE0_STATE_SCHEMA_ID
from ..types import canonical_fingerprint


STAGE0_STATE_SCHEMA_VERSION = 1
STAGE0_STATE_DTYPE = "float32"
RIGHT_ROBOT_BASE_FRAME = "right_robot_base"
JOINT_SPACE_FRAME = "joint_space"
FRAME_INDEPENDENT = "frame_independent"

_FIELD_NAME = re.compile(r"^[a-z][a-z0-9_]*$")
_VALID_GROUPS = frozenset({"robot", "object", "environment", "contact", "task"})
_VALID_UNITS = frozenset(
    {
        "radian",
        "radian_per_second",
        "metre",
        "metre_per_second",
        "dimensionless",
        "count",
        "boolean",
        "one_hot",
    }
)
_VALID_FRAMES = frozenset(
    {RIGHT_ROBOT_BASE_FRAME, JOINT_SPACE_FRAME, FRAME_INDEPENDENT}
)


def _strict_keys(
    value: Mapping[str, Any],
    known: set[str],
    context: str,
) -> None:
    unknown = sorted(set(value).difference(known))
    if unknown:
        raise ValueError(f"Unknown {context} fields: " + ", ".join(unknown))


@dataclass(frozen=True)
class Stage0StateField:
    """One contiguous field in the canonical batch-major state vector."""

    name: str
    width: int
    group: str
    unit: str
    frame: str

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not _FIELD_NAME.fullmatch(self.name):
            raise ValueError("state field name must be lower_snake_case")
        if type(self.width) is not int or self.width < 1:
            raise ValueError(f"{self.name}.width must be a positive integer")
        if not isinstance(self.group, str) or self.group not in _VALID_GROUPS:
            raise ValueError(f"{self.name}.group is unsupported")
        if not isinstance(self.unit, str) or self.unit not in _VALID_UNITS:
            raise ValueError(f"{self.name}.unit is unsupported")
        if not isinstance(self.frame, str) or self.frame not in _VALID_FRAMES:
            raise ValueError(f"{self.name}.frame is unsupported")

    def to_record(self, *, offset: int) -> dict[str, Any]:
        return {
            "name": self.name,
            "width": self.width,
            "offset": offset,
            "end": offset + self.width,
            "group": self.group,
            "unit": self.unit,
            "frame": self.frame,
        }

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
        *,
        expected_offset: int,
    ) -> Stage0StateField:
        if not isinstance(value, Mapping):
            raise ValueError("state field must be an object")
        known = {"name", "width", "offset", "end", "group", "unit", "frame"}
        _strict_keys(value, known, "state field")
        required = known
        missing = sorted(required.difference(value))
        if missing:
            raise ValueError("Missing state field fields: " + ", ".join(missing))
        if type(value["offset"]) is not int or value["offset"] != expected_offset:
            raise ValueError("state field offset is not contiguous")
        field = cls(
            name=value["name"],
            width=value["width"],
            group=value["group"],
            unit=value["unit"],
            frame=value["frame"],
        )
        if (
            type(value["end"]) is not int
            or value["end"] != expected_offset + field.width
        ):
            raise ValueError("state field end does not match offset + width")
        return field


@dataclass(frozen=True)
class Stage0StateSchema:
    """An ordered immutable state vector schema with stable slices and hash."""

    fields: tuple[Stage0StateField, ...]
    schema_version: int = STAGE0_STATE_SCHEMA_VERSION
    schema_id: str = STAGE0_STATE_SCHEMA_ID
    dtype: str = STAGE0_STATE_DTYPE

    def __post_init__(self) -> None:
        if not isinstance(self.fields, tuple) or not self.fields:
            raise ValueError("fields must be a non-empty tuple")
        if not all(isinstance(item, Stage0StateField) for item in self.fields):
            raise ValueError("fields must contain Stage0StateField values")
        if (
            type(self.schema_version) is not int
            or self.schema_version != STAGE0_STATE_SCHEMA_VERSION
        ):
            raise ValueError(f"schema_version must be {STAGE0_STATE_SCHEMA_VERSION}")
        if not isinstance(self.schema_id, str) or not self.schema_id:
            raise ValueError("schema_id must be non-empty text")
        if self.dtype != STAGE0_STATE_DTYPE:
            raise ValueError(f"dtype must be {STAGE0_STATE_DTYPE!r}")
        names = [item.name for item in self.fields]
        if len(names) != len(set(names)):
            raise ValueError("state field names must be unique")

    @property
    def dimension(self) -> int:
        return sum(item.width for item in self.fields)

    @property
    def offsets(self) -> Mapping[str, int]:
        offset = 0
        result: dict[str, int] = {}
        for field in self.fields:
            result[field.name] = offset
            offset += field.width
        return MappingProxyType(result)

    def field(self, name: str) -> Stage0StateField:
        for field in self.fields:
            if field.name == name:
                return field
        raise KeyError(f"Unknown Stage 0 state field {name!r}")

    def slice(self, name: str) -> slice:
        offset = self.offsets[name]
        return slice(offset, offset + self.field(name).width)

    def to_record(self) -> dict[str, Any]:
        offset = 0
        field_records = []
        for field in self.fields:
            field_records.append(field.to_record(offset=offset))
            offset += field.width
        return {
            "schema_version": self.schema_version,
            "schema_id": self.schema_id,
            "dtype": self.dtype,
            "dimension": self.dimension,
            "fields": field_records,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> Stage0StateSchema:
        if not isinstance(value, Mapping):
            raise ValueError("state schema must be an object")
        known = {"schema_version", "schema_id", "dtype", "dimension", "fields"}
        _strict_keys(value, known, "state schema")
        missing = sorted(known.difference(value))
        if missing:
            raise ValueError("Missing state schema fields: " + ", ".join(missing))
        raw_fields = value["fields"]
        if not isinstance(raw_fields, Sequence) or isinstance(raw_fields, (str, bytes)):
            raise ValueError("state schema fields must be an array")
        offset = 0
        parsed_fields = []
        for raw_field in raw_fields:
            field = Stage0StateField.from_mapping(
                raw_field,
                expected_offset=offset,
            )
            parsed_fields.append(field)
            offset += field.width
        schema = cls(
            fields=tuple(parsed_fields),
            schema_version=value["schema_version"],
            schema_id=value["schema_id"],
            dtype=value["dtype"],
        )
        if (
            type(value["dimension"]) is not int
            or value["dimension"] != schema.dimension
        ):
            raise ValueError("state schema dimension does not match its fields")
        return schema

    @property
    def fingerprint(self) -> str:
        return canonical_fingerprint(self.to_record())

    @property
    def sha256(self) -> str:
        """Compatibility spelling for manifests that label hashes by algorithm."""
        return self.fingerprint


DEFAULT_STAGE0_STATE_FIELDS = (
    Stage0StateField("arm_joint_position", 7, "robot", "radian", JOINT_SPACE_FRAME),
    Stage0StateField(
        "arm_joint_velocity",
        7,
        "robot",
        "radian_per_second",
        JOINT_SPACE_FRAME,
    ),
    Stage0StateField("hand_joint_position", 10, "robot", "radian", JOINT_SPACE_FRAME),
    Stage0StateField(
        "hand_joint_velocity",
        10,
        "robot",
        "radian_per_second",
        JOINT_SPACE_FRAME,
    ),
    Stage0StateField("eef_position", 3, "robot", "metre", RIGHT_ROBOT_BASE_FRAME),
    Stage0StateField(
        "eef_rotation_6d",
        6,
        "robot",
        "dimensionless",
        RIGHT_ROBOT_BASE_FRAME,
    ),
    Stage0StateField("object_position", 3, "object", "metre", RIGHT_ROBOT_BASE_FRAME),
    Stage0StateField(
        "object_rotation_6d",
        6,
        "object",
        "dimensionless",
        RIGHT_ROBOT_BASE_FRAME,
    ),
    Stage0StateField(
        "object_linear_velocity",
        3,
        "object",
        "metre_per_second",
        RIGHT_ROBOT_BASE_FRAME,
    ),
    Stage0StateField(
        "object_angular_velocity",
        3,
        "object",
        "radian_per_second",
        RIGHT_ROBOT_BASE_FRAME,
    ),
    Stage0StateField(
        "goal_position",
        3,
        "environment",
        "metre",
        RIGHT_ROBOT_BASE_FRAME,
    ),
    Stage0StateField(
        "eef_to_object",
        3,
        "environment",
        "metre",
        RIGHT_ROBOT_BASE_FRAME,
    ),
    Stage0StateField(
        "object_to_goal",
        3,
        "environment",
        "metre",
        RIGHT_ROBOT_BASE_FRAME,
    ),
    Stage0StateField(
        "finger_contact_counts",
        5,
        "contact",
        "count",
        FRAME_INDEPENDENT,
    ),
    Stage0StateField(
        "has_hand_contact",
        1,
        "contact",
        "boolean",
        FRAME_INDEPENDENT,
    ),
    Stage0StateField(
        "is_grasped",
        1,
        "contact",
        "boolean",
        FRAME_INDEPENDENT,
    ),
    Stage0StateField(
        "task_phase_one_hot",
        5,
        "task",
        "one_hot",
        FRAME_INDEPENDENT,
    ),
)

DEFAULT_STAGE0_STATE_SCHEMA = Stage0StateSchema(fields=DEFAULT_STAGE0_STATE_FIELDS)
if DEFAULT_STAGE0_STATE_SCHEMA.dimension != STAGE0_STATE_DIM:
    raise RuntimeError(
        "default Stage 0 state schema dimension drifted from the config contract"
    )


__all__ = [
    "DEFAULT_STAGE0_STATE_FIELDS",
    "DEFAULT_STAGE0_STATE_SCHEMA",
    "FRAME_INDEPENDENT",
    "JOINT_SPACE_FRAME",
    "RIGHT_ROBOT_BASE_FRAME",
    "STAGE0_STATE_DTYPE",
    "STAGE0_STATE_SCHEMA_VERSION",
    "Stage0StateField",
    "Stage0StateSchema",
]
