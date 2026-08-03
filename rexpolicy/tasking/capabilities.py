"""Trusted EventSchema and environment-adapter capability allowlists."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from .canonical import canonical_fingerprint, strict_json_loads
from .model import TaskSpecV2


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


def _string(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{path} must be non-empty text")
    return value


def _strings(value: Any, path: str, *, nonempty: bool = True) -> tuple[str, ...]:
    if not isinstance(value, list) or (nonempty and not value):
        raise ValueError(f"{path} must be an array of strings")
    output = tuple(_string(item, f"{path}[{index}]") for index, item in enumerate(value))
    if len(set(output)) != len(output):
        raise ValueError(f"{path} must contain unique values")
    return output


def _bound(value: Any, path: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{path} must be numeric or null")
    output = float(value)
    if not math.isfinite(output):
        raise ValueError(f"{path} must be finite")
    return output


@dataclass(frozen=True)
class MetricSpec:
    name: str
    value_type: str
    minimum: float | None
    maximum: float | None
    unit: str
    available_at: tuple[str, ...]
    allowed_purposes: tuple[str, ...]

    @classmethod
    def from_record(cls, value: Any, path: str) -> MetricSpec:
        record = _mapping(value, path)
        _keys(
            record,
            {
                "name",
                "value_type",
                "minimum",
                "maximum",
                "unit",
                "available_at",
                "allowed_purposes",
            },
            path,
        )
        value_type = _string(record["value_type"], f"{path}.value_type")
        if value_type not in {"number", "boolean"}:
            raise ValueError(f"{path}.value_type is unsupported")
        minimum = _bound(record["minimum"], f"{path}.minimum")
        maximum = _bound(record["maximum"], f"{path}.maximum")
        if value_type == "boolean" and (minimum is not None or maximum is not None):
            raise ValueError(f"{path} Boolean metrics cannot declare numeric bounds")
        if minimum is not None and maximum is not None and minimum > maximum:
            raise ValueError(f"{path} metric bounds are reversed")
        available_at = _strings(record["available_at"], f"{path}.available_at")
        if not set(available_at).issubset({"previous", "current"}):
            raise ValueError(f"{path}.available_at contains an unsupported time")
        allowed_purposes = _strings(
            record["allowed_purposes"],
            f"{path}.allowed_purposes",
        )
        if not set(allowed_purposes).issubset(
            {"goal", "terminal", "reward", "process"}
        ):
            raise ValueError(f"{path}.allowed_purposes is unsupported")
        return cls(
            name=_string(record["name"], f"{path}.name"),
            value_type=value_type,
            minimum=minimum,
            maximum=maximum,
            unit=_string(record["unit"], f"{path}.unit"),
            available_at=available_at,
            allowed_purposes=allowed_purposes,
        )

    def to_record(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "value_type": self.value_type,
            "minimum": self.minimum,
            "maximum": self.maximum,
            "unit": self.unit,
            "available_at": list(self.available_at),
            "allowed_purposes": list(self.allowed_purposes),
        }


@dataclass(frozen=True)
class EventSchema:
    schema_version: int
    event_schema_id: str
    metrics: tuple[MetricSpec, ...]

    @classmethod
    def from_record(cls, value: Any, path: str) -> EventSchema:
        record = _mapping(value, path)
        _keys(record, {"schema_version", "event_schema_id", "metrics"}, path)
        if record["schema_version"] != 1:
            raise ValueError(f"{path}.schema_version must be 1")
        raw_metrics = record["metrics"]
        if not isinstance(raw_metrics, list) or not raw_metrics:
            raise ValueError(f"{path}.metrics must be non-empty")
        metrics = tuple(sorted((
            MetricSpec.from_record(item, f"{path}.metrics[{index}]")
            for index, item in enumerate(raw_metrics)
        ), key=lambda metric: metric.name))
        if len({metric.name for metric in metrics}) != len(metrics):
            raise ValueError(f"{path} metric names must be unique")
        return cls(
            schema_version=1,
            event_schema_id=_string(
                record["event_schema_id"],
                f"{path}.event_schema_id",
            ),
            metrics=metrics,
        )

    @property
    def by_name(self) -> dict[str, MetricSpec]:
        return {metric.name: metric for metric in self.metrics}

    def to_record(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "event_schema_id": self.event_schema_id,
            "metrics": [metric.to_record() for metric in self.metrics],
        }

    @property
    def fingerprint(self) -> str:
        return canonical_fingerprint(self.to_record())


@dataclass(frozen=True)
class ParameterCapability:
    name: str
    value_type: str
    minimum: float | None
    maximum: float | None
    required: bool
    vector_length: int | None

    @classmethod
    def from_record(cls, value: Any, path: str) -> ParameterCapability:
        record = _mapping(value, path)
        _keys(
            record,
            {
                "name",
                "value_type",
                "minimum",
                "maximum",
                "required",
                "vector_length",
            },
            path,
        )
        value_type = _string(record["value_type"], f"{path}.value_type")
        if value_type not in {"number", "boolean", "number_vector"}:
            raise ValueError(f"{path}.value_type is unsupported")
        if type(record["required"]) is not bool:
            raise ValueError(f"{path}.required must be Boolean")
        minimum = _bound(record["minimum"], f"{path}.minimum")
        maximum = _bound(record["maximum"], f"{path}.maximum")
        if value_type == "boolean" and (minimum is not None or maximum is not None):
            raise ValueError(f"{path} Boolean parameters cannot have bounds")
        if minimum is not None and maximum is not None and minimum > maximum:
            raise ValueError(f"{path} parameter bounds are reversed")
        vector_length = record["vector_length"]
        if value_type == "number_vector":
            if type(vector_length) is not int or vector_length < 1:
                raise ValueError(f"{path}.vector_length must be positive")
        elif vector_length is not None:
            raise ValueError(f"{path}.vector_length must be null for scalars")
        return cls(
            name=_string(record["name"], f"{path}.name"),
            value_type=value_type,
            minimum=minimum,
            maximum=maximum,
            required=record["required"],
            vector_length=vector_length,
        )

    def validate_value(self, value: Any, path: str) -> None:
        """Validate one author-controlled value against the trusted range."""
        if self.value_type == "boolean":
            if type(value) is not bool:
                raise ValueError(f"{path} must be Boolean")
            return
        values = value if self.value_type == "number_vector" else [value]
        if self.value_type == "number_vector" and (
            not isinstance(value, list) or not value
        ):
            raise ValueError(f"{path} must be a non-empty numeric vector")
        if self.value_type == "number_vector" and len(value) != self.vector_length:
            raise ValueError(f"{path} has the wrong vector length")
        for index, item in enumerate(values):
            item_path = f"{path}[{index}]" if len(values) > 1 else path
            number = _bound(item, item_path)
            if number is None:
                raise ValueError(f"{item_path} cannot be null")
            if self.minimum is not None and number < self.minimum:
                raise ValueError(f"{item_path} is below the capability minimum")
            if self.maximum is not None and number > self.maximum:
                raise ValueError(f"{item_path} exceeds the capability maximum")

    def to_record(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "value_type": self.value_type,
            "minimum": self.minimum,
            "maximum": self.maximum,
            "required": self.required,
            "vector_length": self.vector_length,
        }


@dataclass(frozen=True)
class AdapterCapability:
    adapter_id: str
    task_modes: tuple[str, ...]
    event_schema_id: str
    process_spec_ids: tuple[str, ...]
    allowed_assets: tuple[str, ...]
    required_assets: tuple[str, ...]
    bindings: dict[str, tuple[Any, ...]]
    parameters: tuple[ParameterCapability, ...]
    action_projection: dict[str, tuple[bool, ...]]

    @classmethod
    def from_record(cls, value: Any, path: str) -> AdapterCapability:
        record = _mapping(value, path)
        _keys(
            record,
            {
                "adapter_id",
                "task_modes",
                "event_schema_id",
                "process_spec_ids",
                "allowed_assets",
                "required_assets",
                "bindings",
                "parameters",
                "action_projection",
            },
            path,
        )
        raw_parameters = record["parameters"]
        if not isinstance(raw_parameters, list):
            raise ValueError(f"{path}.parameters must be an array")
        parameters = tuple(
            ParameterCapability.from_record(item, f"{path}.parameters[{index}]")
            for index, item in enumerate(raw_parameters)
        )
        if len({item.name for item in parameters}) != len(parameters):
            raise ValueError(f"{path} parameter names must be unique")
        bindings_record = _mapping(record["bindings"], f"{path}.bindings")
        bindings = {}
        for key, allowed in bindings_record.items():
            values = allowed if isinstance(allowed, list) else None
            if not values or any(
                not isinstance(item, (str, bool, int, float))
                or (
                    isinstance(item, float)
                    and not math.isfinite(item)
                )
                for item in values
            ):
                raise ValueError(f"{path}.bindings.{key} must list scalar values")
            if len({repr(item) for item in values}) != len(values):
                raise ValueError(f"{path}.bindings.{key} values must be unique")
            bindings[key] = tuple(values)
        action_record = _mapping(
            record["action_projection"],
            f"{path}.action_projection",
        )
        action_projection = {}
        for key, mask in action_record.items():
            if not isinstance(mask, list) or not mask or any(
                type(item) is not bool for item in mask
            ):
                raise ValueError(f"{path}.action_projection.{key} is invalid")
            action_projection[key] = tuple(mask)
        allowed_assets = _strings(record["allowed_assets"], f"{path}.allowed_assets")
        required_assets = _strings(
            record["required_assets"],
            f"{path}.required_assets",
            nonempty=False,
        )
        if not set(required_assets).issubset(allowed_assets):
            raise ValueError(f"{path}.required_assets are not allowed")
        return cls(
            adapter_id=_string(record["adapter_id"], f"{path}.adapter_id"),
            task_modes=_strings(record["task_modes"], f"{path}.task_modes"),
            event_schema_id=_string(
                record["event_schema_id"],
                f"{path}.event_schema_id",
            ),
            process_spec_ids=_strings(
                record["process_spec_ids"],
                f"{path}.process_spec_ids",
            ),
            allowed_assets=allowed_assets,
            required_assets=required_assets,
            bindings=bindings,
            parameters=parameters,
            action_projection=action_projection,
        )

    def to_record(self) -> dict[str, Any]:
        return {
            "adapter_id": self.adapter_id,
            "task_modes": list(self.task_modes),
            "event_schema_id": self.event_schema_id,
            "process_spec_ids": list(self.process_spec_ids),
            "allowed_assets": list(self.allowed_assets),
            "required_assets": list(self.required_assets),
            "bindings": {key: list(values) for key, values in self.bindings.items()},
            "parameters": [item.to_record() for item in self.parameters],
            "action_projection": {
                key: list(mask) for key, mask in self.action_projection.items()
            },
        }

    @property
    def fingerprint(self) -> str:
        return canonical_fingerprint(self.to_record())


@dataclass(frozen=True)
class CapabilityCatalog:
    schema_version: int
    event_schemas: tuple[EventSchema, ...]
    adapters: tuple[AdapterCapability, ...]

    @classmethod
    def from_record(cls, value: Any) -> CapabilityCatalog:
        record = _mapping(value, "$capabilities")
        _keys(record, {"schema_version", "event_schemas", "adapters"}, "$capabilities")
        if record["schema_version"] != 1:
            raise ValueError("Capability catalog schema_version must be 1")
        raw_schemas = record["event_schemas"]
        raw_adapters = record["adapters"]
        if not isinstance(raw_schemas, list) or not raw_schemas:
            raise ValueError("Capability catalog requires event schemas")
        if not isinstance(raw_adapters, list) or not raw_adapters:
            raise ValueError("Capability catalog requires adapters")
        schemas = tuple(sorted((
            EventSchema.from_record(item, f"$capabilities.event_schemas[{index}]")
            for index, item in enumerate(raw_schemas)
        ), key=lambda schema: schema.event_schema_id))
        adapters = tuple(sorted((
            AdapterCapability.from_record(item, f"$capabilities.adapters[{index}]")
            for index, item in enumerate(raw_adapters)
        ), key=lambda adapter: adapter.adapter_id))
        if len({item.event_schema_id for item in schemas}) != len(schemas):
            raise ValueError("Capability event schema IDs must be unique")
        if len({item.adapter_id for item in adapters}) != len(adapters):
            raise ValueError("Capability adapter IDs must be unique")
        schema_ids = {item.event_schema_id for item in schemas}
        if any(adapter.event_schema_id not in schema_ids for adapter in adapters):
            raise ValueError("Adapter references an unknown event schema")
        return cls(schema_version=1, event_schemas=schemas, adapters=adapters)

    @classmethod
    def from_json(cls, text: str) -> CapabilityCatalog:
        return cls.from_record(strict_json_loads(text))

    def to_record(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "event_schemas": [item.to_record() for item in self.event_schemas],
            "adapters": [item.to_record() for item in self.adapters],
        }

    @property
    def fingerprint(self) -> str:
        return canonical_fingerprint(self.to_record())


@dataclass(frozen=True)
class ResolvedTaskCapabilities:
    event_schema: EventSchema
    adapter: AdapterCapability


def validate_task_capabilities(
    task: TaskSpecV2,
    catalog: CapabilityCatalog,
) -> ResolvedTaskCapabilities:
    """Ensure author JSON can only select an exact trusted adapter contract."""
    adapters = {item.adapter_id: item for item in catalog.adapters}
    try:
        adapter = adapters[task.environment.adapter_id]
    except KeyError as error:
        raise ValueError("Task selects an unknown environment adapter") from error
    if task.environment.task_mode not in adapter.task_modes:
        raise ValueError("Task mode is not allowed by the environment adapter")
    if task.event_schema_id != adapter.event_schema_id:
        raise ValueError("Task EventSchema does not match its adapter")
    if task.process_spec_id not in adapter.process_spec_ids:
        raise ValueError("Task process spec is not allowed by its adapter")
    task_assets = set(task.environment.required_assets)
    if not set(adapter.required_assets).issubset(task_assets):
        raise ValueError("Task omits an adapter-required asset")
    if not task_assets.issubset(adapter.allowed_assets):
        raise ValueError("Task selects an asset outside the adapter allowlist")
    if task.action_projection != adapter.action_projection:
        raise ValueError("Task action projection must equal the certified adapter mask")
    if set(task.environment.bindings) != set(adapter.bindings):
        raise ValueError("Task bindings do not match the adapter contract")
    for key, value in task.environment.bindings.items():
        if not any(
            type(value) is type(allowed) and value == allowed
            for allowed in adapter.bindings[key]
        ):
            raise ValueError(f"Task binding {key!r} is outside the allowlist")
    parameter_by_name = {item.name: item for item in adapter.parameters}
    unknown_parameters = sorted(
        set(task.environment.parameters).difference(parameter_by_name)
    )
    if unknown_parameters:
        raise ValueError(
            "Task has parameters outside the adapter allowlist: "
            + ", ".join(unknown_parameters)
        )
    missing_parameters = sorted(
        item.name
        for item in adapter.parameters
        if item.required and item.name not in task.environment.parameters
    )
    if missing_parameters:
        raise ValueError(
            "Task omits required adapter parameters: "
            + ", ".join(missing_parameters)
        )
    for name, value in task.environment.parameters.items():
        parameter_by_name[name].validate_value(
            value,
            f"$.environment.parameters.{name}",
        )
    schema = next(
        item
        for item in catalog.event_schemas
        if item.event_schema_id == task.event_schema_id
    )
    return ResolvedTaskCapabilities(event_schema=schema, adapter=adapter)
