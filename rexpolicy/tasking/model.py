"""Canonical TaskSpec v2 value objects with a strict trust boundary."""

from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import dataclass
from typing import Any

from .canonical import canonical_fingerprint, canonical_json, strict_json_loads

_IDENTIFIER = re.compile(r"^[a-z][a-z0-9_.-]*(?:/[a-z0-9_.-]+)*$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_MAX_EXPRESSION_DEPTH = 64
_MAX_INSTRUCTION_CHARS = 512


def _expect_mapping(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{path} must be an object")
    return value


def _expect_keys(
    record: dict[str, Any],
    *,
    required: set[str],
    path: str,
) -> None:
    missing = sorted(required.difference(record))
    unknown = sorted(set(record).difference(required))
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


def _positive_int(value: Any, path: str) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{path} must be a positive integer")
    return value


def _finite(value: Any, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{path} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{path} must be finite")
    return result


def _json_value(value: Any, path: str, *, depth: int = 0) -> Any:
    if depth > 16:
        raise ValueError(f"{path} exceeds the JSON nesting limit")
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        _finite(value, path)
        return value
    if isinstance(value, list):
        return [
            _json_value(item, f"{path}[{index}]", depth=depth + 1)
            for index, item in enumerate(value)
        ]
    if isinstance(value, dict):
        output = {}
        for key, item in value.items():
            if not isinstance(key, str) or not key:
                raise ValueError(f"{path} keys must be non-empty strings")
            output[key] = _json_value(
                item,
                f"{path}.{key}",
                depth=depth + 1,
            )
        return output
    raise ValueError(f"{path} contains a non-JSON value")


def _instruction(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{path} must be non-empty text")
    if value != value.strip() or len(value) > _MAX_INSTRUCTION_CHARS:
        raise ValueError(f"{path} must be trimmed and at most 512 characters")
    if unicodedata.normalize("NFC", value) != value:
        raise ValueError(f"{path} must use NFC Unicode normalization")
    if any(unicodedata.category(character).startswith("C") for character in value):
        raise ValueError(f"{path} cannot contain control characters")
    return value


@dataclass(frozen=True)
class AuditCommitment:
    count: int
    sha256: str

    @classmethod
    def from_record(cls, value: Any, path: str) -> AuditCommitment:
        record = _expect_mapping(value, path)
        _expect_keys(record, required={"count", "sha256"}, path=path)
        return cls(
            count=_positive_int(record["count"], f"{path}.count"),
            sha256=_sha256(record["sha256"], f"{path}.sha256"),
        )

    def to_record(self) -> dict[str, Any]:
        return {"count": self.count, "sha256": self.sha256}


@dataclass(frozen=True)
class InstructionSplits:
    train: tuple[str, ...]
    promotion: tuple[str, ...]
    audit: AuditCommitment

    @classmethod
    def from_record(cls, value: Any, path: str) -> InstructionSplits:
        record = _expect_mapping(value, path)
        _expect_keys(
            record,
            required={"train", "promotion", "audit"},
            path=path,
        )
        splits = []
        for name in ("train", "promotion"):
            raw = record[name]
            if not isinstance(raw, list) or not raw:
                raise ValueError(f"{path}.{name} must be a non-empty array")
            splits.append(
                tuple(
                    _instruction(item, f"{path}.{name}[{index}]")
                    for index, item in enumerate(raw)
                )
            )
        train, promotion = splits
        if len(set((*train, *promotion))) != len((*train, *promotion)):
            raise ValueError("Train and promotion instructions must be unique")
        return cls(
            train=train,
            promotion=promotion,
            audit=AuditCommitment.from_record(record["audit"], f"{path}.audit"),
        )

    def to_record(self) -> dict[str, Any]:
        return {
            "train": list(self.train),
            "promotion": list(self.promotion),
            "audit": self.audit.to_record(),
        }


@dataclass(frozen=True)
class EnvironmentSpec:
    adapter_id: str
    task_mode: str
    bindings: dict[str, Any]
    parameters: dict[str, Any]
    required_assets: tuple[str, ...]

    @classmethod
    def from_record(cls, value: Any, path: str) -> EnvironmentSpec:
        record = _expect_mapping(value, path)
        _expect_keys(
            record,
            required={
                "adapter_id",
                "task_mode",
                "bindings",
                "parameters",
                "required_assets",
            },
            path=path,
        )
        assets = record["required_assets"]
        if not isinstance(assets, list) or not assets:
            raise ValueError(f"{path}.required_assets must be non-empty")
        required_assets = tuple(
            _identifier(item, f"{path}.required_assets[{index}]")
            for index, item in enumerate(assets)
        )
        if len(set(required_assets)) != len(required_assets):
            raise ValueError("Required assets must be unique")
        bindings = _expect_mapping(record["bindings"], f"{path}.bindings")
        parameters = _expect_mapping(record["parameters"], f"{path}.parameters")
        return cls(
            adapter_id=_identifier(record["adapter_id"], f"{path}.adapter_id"),
            task_mode=_identifier(record["task_mode"], f"{path}.task_mode"),
            bindings=_json_value(bindings, f"{path}.bindings"),
            parameters=_json_value(parameters, f"{path}.parameters"),
            required_assets=required_assets,
        )

    def to_record(self) -> dict[str, Any]:
        return {
            "adapter_id": self.adapter_id,
            "task_mode": self.task_mode,
            "bindings": self.bindings,
            "parameters": self.parameters,
            "required_assets": list(self.required_assets),
        }


@dataclass(frozen=True)
class ExpressionSpec:
    op: str
    payload: dict[str, Any]

    @classmethod
    def from_record(
        cls,
        value: Any,
        path: str,
        *,
        depth: int = 0,
    ) -> ExpressionSpec:
        if depth > _MAX_EXPRESSION_DEPTH:
            raise ValueError(f"{path} exceeds expression depth limit")
        record = _expect_mapping(value, path)
        if "op" not in record:
            raise ValueError(f"{path} is missing field: op")
        op = _identifier(record["op"], f"{path}.op")
        if op == "const":
            _expect_keys(record, required={"op", "value"}, path=path)
            payload = {"value": _finite(record["value"], f"{path}.value")}
        elif op == "metric":
            _expect_keys(record, required={"op", "name", "at"}, path=path)
            payload = {
                "name": _identifier(record["name"], f"{path}.name"),
                "at": _identifier(record["at"], f"{path}.at"),
            }
        elif op == "parameter":
            _expect_keys(record, required={"op", "name"}, path=path)
            payload = {"name": _identifier(record["name"], f"{path}.name")}
        else:
            _expect_keys(record, required={"op", "args"}, path=path)
            args = record["args"]
            if not isinstance(args, list):
                raise ValueError(f"{path}.args must be an array")
            payload = {
                "args": tuple(
                    cls.from_record(
                        argument,
                        f"{path}.args[{index}]",
                        depth=depth + 1,
                    )
                    for index, argument in enumerate(args)
                )
            }
        return cls(op=op, payload=payload)

    def to_record(self) -> dict[str, Any]:
        if "args" in self.payload:
            return {
                "op": self.op,
                "args": [argument.to_record() for argument in self.payload["args"]],
            }
        return {"op": self.op, **self.payload}


@dataclass(frozen=True)
class GoalSpec:
    hold_steps: int
    predicate: ExpressionSpec

    @classmethod
    def from_record(cls, value: Any, path: str) -> GoalSpec:
        record = _expect_mapping(value, path)
        _expect_keys(record, required={"hold_steps", "predicate"}, path=path)
        return cls(
            hold_steps=_positive_int(record["hold_steps"], f"{path}.hold_steps"),
            predicate=ExpressionSpec.from_record(
                record["predicate"],
                f"{path}.predicate",
            ),
        )

    def to_record(self) -> dict[str, Any]:
        return {
            "hold_steps": self.hold_steps,
            "predicate": self.predicate.to_record(),
        }


@dataclass(frozen=True)
class TerminalRule:
    code: str
    kind: str
    predicate: ExpressionSpec

    @classmethod
    def from_record(cls, value: Any, path: str) -> TerminalRule:
        record = _expect_mapping(value, path)
        _expect_keys(record, required={"code", "kind", "predicate"}, path=path)
        kind = _identifier(record["kind"], f"{path}.kind")
        if kind not in {"failure", "safety"}:
            raise ValueError(f"{path}.kind must be failure or safety")
        return cls(
            code=_identifier(record["code"], f"{path}.code"),
            kind=kind,
            predicate=ExpressionSpec.from_record(
                record["predicate"],
                f"{path}.predicate",
            ),
        )

    def to_record(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "kind": self.kind,
            "predicate": self.predicate.to_record(),
        }


@dataclass(frozen=True)
class RewardTerm:
    term_id: str
    weight: float
    expression: ExpressionSpec

    @classmethod
    def from_record(cls, value: Any, path: str) -> RewardTerm:
        record = _expect_mapping(value, path)
        _expect_keys(
            record,
            required={"term_id", "weight", "expression"},
            path=path,
        )
        return cls(
            term_id=_identifier(record["term_id"], f"{path}.term_id"),
            weight=_finite(record["weight"], f"{path}.weight"),
            expression=ExpressionSpec.from_record(
                record["expression"],
                f"{path}.expression",
            ),
        )

    def to_record(self) -> dict[str, Any]:
        return {
            "term_id": self.term_id,
            "weight": self.weight,
            "expression": self.expression.to_record(),
        }


@dataclass(frozen=True)
class RewardProfileSpec:
    reward_profile_id: str
    terms: tuple[RewardTerm, ...]
    shaping_bounds: tuple[float, float]
    terminal_values: dict[str, float]

    @classmethod
    def from_record(cls, value: Any, path: str) -> RewardProfileSpec:
        record = _expect_mapping(value, path)
        _expect_keys(
            record,
            required={
                "reward_profile_id",
                "terms",
                "shaping_bounds",
                "terminal_values",
            },
            path=path,
        )
        raw_terms = record["terms"]
        if not isinstance(raw_terms, list) or not raw_terms:
            raise ValueError(f"{path}.terms must be non-empty")
        terms = tuple(
            RewardTerm.from_record(term, f"{path}.terms[{index}]")
            for index, term in enumerate(raw_terms)
        )
        if len({term.term_id for term in terms}) != len(terms):
            raise ValueError("Reward term IDs must be unique")
        raw_bounds = record["shaping_bounds"]
        if not isinstance(raw_bounds, list) or len(raw_bounds) != 2:
            raise ValueError(f"{path}.shaping_bounds must contain two values")
        bounds = (
            _finite(raw_bounds[0], f"{path}.shaping_bounds[0]"),
            _finite(raw_bounds[1], f"{path}.shaping_bounds[1]"),
        )
        if bounds[0] > bounds[1]:
            raise ValueError("Reward shaping bounds are reversed")
        terminal = _expect_mapping(
            record["terminal_values"],
            f"{path}.terminal_values",
        )
        _expect_keys(
            terminal,
            required={"success", "failure"},
            path=f"{path}.terminal_values",
        )
        return cls(
            reward_profile_id=_identifier(
                record["reward_profile_id"],
                f"{path}.reward_profile_id",
            ),
            terms=terms,
            shaping_bounds=bounds,
            terminal_values={
                "success": _finite(
                    terminal["success"],
                    f"{path}.terminal_values.success",
                ),
                "failure": _finite(
                    terminal["failure"],
                    f"{path}.terminal_values.failure",
                ),
            },
        )

    def to_record(self) -> dict[str, Any]:
        return {
            "reward_profile_id": self.reward_profile_id,
            "terms": [term.to_record() for term in self.terms],
            "shaping_bounds": list(self.shaping_bounds),
            "terminal_values": dict(self.terminal_values),
        }


@dataclass(frozen=True)
class ValidationExample:
    example_id: str
    previous_metrics: dict[str, Any]
    current_metrics: dict[str, Any]
    expected_success: bool
    expected_failure_codes: tuple[str, ...]

    @classmethod
    def from_record(cls, value: Any, path: str) -> ValidationExample:
        record = _expect_mapping(value, path)
        _expect_keys(
            record,
            required={
                "example_id",
                "previous_metrics",
                "current_metrics",
                "expected_success",
                "expected_failure_codes",
            },
            path=path,
        )
        if type(record["expected_success"]) is not bool:
            raise ValueError(f"{path}.expected_success must be Boolean")
        raw_codes = record["expected_failure_codes"]
        if not isinstance(raw_codes, list):
            raise ValueError(f"{path}.expected_failure_codes must be an array")
        codes = tuple(
            _identifier(code, f"{path}.expected_failure_codes[{index}]")
            for index, code in enumerate(raw_codes)
        )
        if len(set(codes)) != len(codes):
            raise ValueError("Expected failure codes must be unique")
        return cls(
            example_id=_identifier(record["example_id"], f"{path}.example_id"),
            previous_metrics=_json_value(
                _expect_mapping(
                    record["previous_metrics"],
                    f"{path}.previous_metrics",
                ),
                f"{path}.previous_metrics",
            ),
            current_metrics=_json_value(
                _expect_mapping(
                    record["current_metrics"],
                    f"{path}.current_metrics",
                ),
                f"{path}.current_metrics",
            ),
            expected_success=record["expected_success"],
            expected_failure_codes=codes,
        )

    def to_record(self) -> dict[str, Any]:
        return {
            "example_id": self.example_id,
            "previous_metrics": self.previous_metrics,
            "current_metrics": self.current_metrics,
            "expected_success": self.expected_success,
            "expected_failure_codes": list(self.expected_failure_codes),
        }


@dataclass(frozen=True)
class TaskSpecV2:
    schema_version: int
    task_id: str
    version: int
    family_id: str
    event_schema_id: str
    process_spec_id: str
    curriculum_prerequisite_ids: tuple[str, ...]
    environment: EnvironmentSpec
    instructions: InstructionSplits
    action_projection: dict[str, tuple[bool, ...]]
    goal: GoalSpec
    terminal_rules: tuple[TerminalRule, ...]
    reward_profiles: tuple[RewardProfileSpec, ...]
    validation_examples: tuple[ValidationExample, ...]

    @classmethod
    def from_record(cls, value: Any) -> TaskSpecV2:
        path = "$"
        record = _expect_mapping(value, path)
        required = {
            "schema_version",
            "task_id",
            "version",
            "family_id",
            "event_schema_id",
            "process_spec_id",
            "curriculum_prerequisite_ids",
            "environment",
            "instructions",
            "action_projection",
            "goal",
            "terminal_rules",
            "reward_profiles",
            "validation_examples",
        }
        _expect_keys(record, required=required, path=path)
        if record["schema_version"] != 2:
            raise ValueError("TaskSpec schema_version must be 2")
        prerequisites_raw = record["curriculum_prerequisite_ids"]
        if not isinstance(prerequisites_raw, list):
            raise ValueError("$.curriculum_prerequisite_ids must be an array")
        prerequisites = tuple(
            _identifier(item, f"$.curriculum_prerequisite_ids[{index}]")
            for index, item in enumerate(prerequisites_raw)
        )
        if len(set(prerequisites)) != len(prerequisites):
            raise ValueError("Curriculum prerequisites must be unique")
        action_raw = _expect_mapping(
            record["action_projection"],
            "$.action_projection",
        )
        if not action_raw:
            raise ValueError("$.action_projection cannot be empty")
        action_projection = {}
        for key, mask in action_raw.items():
            action_key = _identifier(key, f"$.action_projection.{key}")
            if not isinstance(mask, list) or not mask:
                raise ValueError(f"$.action_projection.{key} must be non-empty")
            if any(type(item) is not bool for item in mask):
                raise ValueError(f"$.action_projection.{key} must be Boolean")
            action_projection[action_key] = tuple(mask)
        terminals_raw = record["terminal_rules"]
        rewards_raw = record["reward_profiles"]
        examples_raw = record["validation_examples"]
        if not isinstance(terminals_raw, list) or not terminals_raw:
            raise ValueError("$.terminal_rules must be non-empty")
        if not isinstance(rewards_raw, list) or not rewards_raw:
            raise ValueError("$.reward_profiles must be non-empty")
        if not isinstance(examples_raw, list) or not examples_raw:
            raise ValueError("$.validation_examples must be non-empty")
        terminal_rules = tuple(
            TerminalRule.from_record(item, f"$.terminal_rules[{index}]")
            for index, item in enumerate(terminals_raw)
        )
        reward_profiles = tuple(
            RewardProfileSpec.from_record(item, f"$.reward_profiles[{index}]")
            for index, item in enumerate(rewards_raw)
        )
        examples = tuple(
            ValidationExample.from_record(
                item,
                f"$.validation_examples[{index}]",
            )
            for index, item in enumerate(examples_raw)
        )
        for values, label in (
            ((rule.code for rule in terminal_rules), "terminal rule codes"),
            (
                (profile.reward_profile_id for profile in reward_profiles),
                "reward profile IDs",
            ),
            ((example.example_id for example in examples), "validation example IDs"),
        ):
            materialized = tuple(values)
            if len(set(materialized)) != len(materialized):
                raise ValueError(f"TaskSpec {label} must be unique")
        task_id = _identifier(record["task_id"], "$.task_id")
        version = _positive_int(record["version"], "$.version")
        if not task_id.endswith(f"/v{version}"):
            raise ValueError("$.task_id version suffix must match $.version")
        if task_id in prerequisites:
            raise ValueError("A task cannot list itself as a prerequisite")
        return cls(
            schema_version=2,
            task_id=task_id,
            version=version,
            family_id=_identifier(record["family_id"], "$.family_id"),
            event_schema_id=_identifier(
                record["event_schema_id"],
                "$.event_schema_id",
            ),
            process_spec_id=_identifier(
                record["process_spec_id"],
                "$.process_spec_id",
            ),
            curriculum_prerequisite_ids=prerequisites,
            environment=EnvironmentSpec.from_record(
                record["environment"],
                "$.environment",
            ),
            instructions=InstructionSplits.from_record(
                record["instructions"],
                "$.instructions",
            ),
            action_projection=action_projection,
            goal=GoalSpec.from_record(record["goal"], "$.goal"),
            terminal_rules=terminal_rules,
            reward_profiles=reward_profiles,
            validation_examples=examples,
        )

    def to_record(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "task_id": self.task_id,
            "version": self.version,
            "family_id": self.family_id,
            "event_schema_id": self.event_schema_id,
            "process_spec_id": self.process_spec_id,
            "curriculum_prerequisite_ids": list(
                self.curriculum_prerequisite_ids
            ),
            "environment": self.environment.to_record(),
            "instructions": self.instructions.to_record(),
            "action_projection": {
                key: list(mask) for key, mask in self.action_projection.items()
            },
            "goal": self.goal.to_record(),
            "terminal_rules": [rule.to_record() for rule in self.terminal_rules],
            "reward_profiles": [
                profile.to_record() for profile in self.reward_profiles
            ],
            "validation_examples": [
                example.to_record() for example in self.validation_examples
            ],
        }

    @property
    def fingerprint(self) -> str:
        return canonical_fingerprint(self.to_record())

    def to_json(self) -> str:
        return canonical_json(self.to_record())


def decode_task_spec_json(text: str) -> TaskSpecV2:
    """Cross the untrusted JSON boundary into a canonical TaskSpec v2."""
    return TaskSpecV2.from_record(strict_json_loads(text))
