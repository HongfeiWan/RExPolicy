"""Compilation of ProcessSpec v1 into immutable stage contracts."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from .canonical import canonical_fingerprint
from .capabilities import AdapterCapability, CapabilityCatalog, ParameterCapability
from .contract import (
    DEFAULT_TASK_COMPILER_POLICY,
    CompiledTaskContract,
    TaskCompilerPolicy,
    validate_compiled_task_contract,
)
from .expression import CompileLimits, ExpressionProgram, MetricOperand, compile_expression
from .model import TaskSpecV2
from .process import ProcessSpecV1

_IDENTIFIER = re.compile(r"^[a-z][a-z0-9_.-]*(?:/[a-z0-9_.-]+)*$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_CONTRACT_FORMAT = "rexpolicy_process_contract/v1"


def _identifier(value: Any, path: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise ValueError(f"{path} must be a versioned identifier")
    return value


def _sha256(value: Any, path: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{path} must be a lowercase SHA-256")
    return value


@dataclass(frozen=True)
class ProcessCompilerPolicy:
    policy_id: str
    expression_limits: CompileLimits
    max_stages: int
    max_segment_steps: int
    require_current_state_predicates: bool

    def validate(self) -> None:
        _identifier(self.policy_id, "process_compiler_policy.policy_id")
        self.expression_limits.validate()
        for name, value in (
            ("max_stages", self.max_stages),
            ("max_segment_steps", self.max_segment_steps),
        ):
            if type(value) is not int or value < 1:
                raise ValueError(f"process_compiler_policy.{name} must be positive")
        if type(self.require_current_state_predicates) is not bool:
            raise ValueError(
                "process_compiler_policy.require_current_state_predicates "
                "must be Boolean"
            )

    def to_record(self) -> dict[str, Any]:
        return {
            "policy_id": self.policy_id,
            "expression_limits": {
                "max_nodes": self.expression_limits.max_nodes,
                "max_instructions": self.expression_limits.max_instructions,
                "max_stack": self.expression_limits.max_stack,
                "minimum_abs_divisor": self.expression_limits.minimum_abs_divisor,
            },
            "max_stages": self.max_stages,
            "max_segment_steps": self.max_segment_steps,
            "require_current_state_predicates": (
                self.require_current_state_predicates
            ),
        }

    @property
    def fingerprint(self) -> str:
        return canonical_fingerprint(self.to_record())


DEFAULT_PROCESS_COMPILER_POLICY = ProcessCompilerPolicy(
    policy_id="rexpolicy/process_compiler/v1",
    expression_limits=CompileLimits(),
    max_stages=32,
    max_segment_steps=64,
    require_current_state_predicates=True,
)


@dataclass(frozen=True)
class CompiledProcessStage:
    stage_id: str
    kind: str
    ordinal: int | None
    description: str
    program: ExpressionProgram
    program_fingerprint: str

    def to_record(self) -> dict[str, Any]:
        return {
            "stage_id": self.stage_id,
            "kind": self.kind,
            "ordinal": self.ordinal,
            "description": self.description,
            "program_fingerprint": self.program_fingerprint,
            "program": self.program.to_record(),
        }


@dataclass(frozen=True)
class CompiledProcessContract:
    contract_format: str
    process_contract_id: str
    process_spec_id: str
    process_spec_fingerprint: str
    process_compiler_policy_id: str
    process_compiler_policy_fingerprint: str
    task_contract_id: str
    task_contract_fingerprint: str
    task_spec_fingerprint: str
    task_compiler_policy_fingerprint: str
    capability_catalog_fingerprint: str
    event_schema_id: str
    event_schema_fingerprint: str
    default_stage_id: str
    default_stage_ordinal: int
    default_stage_description: str
    stages: tuple[CompiledProcessStage, ...]
    max_segment_steps: int
    required_metrics: tuple[str, ...]

    def to_record(self) -> dict[str, Any]:
        return {
            "contract_format": self.contract_format,
            "process_contract_id": self.process_contract_id,
            "process_spec_id": self.process_spec_id,
            "process_spec_fingerprint": self.process_spec_fingerprint,
            "process_compiler_policy_id": self.process_compiler_policy_id,
            "process_compiler_policy_fingerprint": (
                self.process_compiler_policy_fingerprint
            ),
            "task_contract_id": self.task_contract_id,
            "task_contract_fingerprint": self.task_contract_fingerprint,
            "task_spec_fingerprint": self.task_spec_fingerprint,
            "task_compiler_policy_fingerprint": (
                self.task_compiler_policy_fingerprint
            ),
            "capability_catalog_fingerprint": (
                self.capability_catalog_fingerprint
            ),
            "event_schema_id": self.event_schema_id,
            "event_schema_fingerprint": self.event_schema_fingerprint,
            "default_stage": {
                "stage_id": self.default_stage_id,
                "kind": "progress",
                "ordinal": self.default_stage_ordinal,
                "description": self.default_stage_description,
            },
            "stages": [stage.to_record() for stage in self.stages],
            "max_segment_steps": self.max_segment_steps,
            "required_metrics": list(self.required_metrics),
        }

    @property
    def fingerprint(self) -> str:
        return canonical_fingerprint(self.to_record())


def _parameter_capabilities(
    adapter: AdapterCapability,
) -> dict[str, ParameterCapability]:
    return {item.name: item for item in adapter.parameters}


def _has_previous_metric(program: ExpressionProgram) -> bool:
    return any(
        instruction.opcode == "load_metric"
        and isinstance(instruction.operand, MetricOperand)
        and instruction.operand.at != "current"
        for instruction in program.instructions
    )


def compile_process_contract(
    process_spec: ProcessSpecV1,
    *,
    task: TaskSpecV2,
    task_contract: CompiledTaskContract,
    catalog: CapabilityCatalog,
    process_policy: ProcessCompilerPolicy = DEFAULT_PROCESS_COMPILER_POLICY,
    task_policy: TaskCompilerPolicy = DEFAULT_TASK_COMPILER_POLICY,
) -> CompiledProcessContract:
    """Compile priority-ordered process stages against one exact task contract."""
    process_policy.validate()
    canonical_process = ProcessSpecV1.from_record(process_spec.to_record())
    canonical_task = TaskSpecV2.from_record(task.to_record())
    canonical_catalog = CapabilityCatalog.from_record(catalog.to_record())
    validate_compiled_task_contract(
        task_contract,
        catalog=canonical_catalog,
        policy=task_policy,
    )
    if (
        canonical_process.process_spec_id != canonical_task.process_spec_id
        or canonical_process.process_spec_id != task_contract.process_spec_id
    ):
        raise ValueError("ProcessSpec identity does not match the task contract")
    if (
        canonical_process.task_id != canonical_task.task_id
        or canonical_process.task_id != task_contract.task_id
    ):
        raise ValueError("ProcessSpec task identity mismatch")
    if canonical_task.fingerprint != task_contract.task_spec_fingerprint:
        raise ValueError("ProcessSpec task fingerprint mismatch")
    if (
        canonical_process.event_schema_id != canonical_task.event_schema_id
        or canonical_process.event_schema_id != task_contract.event_schema_id
    ):
        raise ValueError("ProcessSpec EventSchema identity mismatch")
    if len(canonical_process.stages) > process_policy.max_stages:
        raise ValueError("ProcessSpec exceeds the trusted stage limit")
    if canonical_process.segmentation.max_segment_steps > min(
        process_policy.max_segment_steps,
        task_contract.episode_control_steps,
    ):
        raise ValueError("ProcessSpec segment length exceeds the trusted horizon")

    schema = next(
        item
        for item in canonical_catalog.event_schemas
        if item.event_schema_id == task_contract.event_schema_id
    )
    adapter = next(
        item
        for item in canonical_catalog.adapters
        if item.adapter_id == task_contract.adapter_id
    )
    compiled_stages = []
    required_metrics: set[str] = set()
    for stage in canonical_process.stages:
        program = compile_expression(
            stage.predicate,
            schema=schema,
            parameters=canonical_task.environment.parameters,
            parameter_capabilities=_parameter_capabilities(adapter),
            purpose="process",
            task_spec_fingerprint=canonical_task.fingerprint,
            limits=process_policy.expression_limits,
        )
        if process_policy.require_current_state_predicates and _has_previous_metric(
            program
        ):
            raise ValueError("Process stage predicates must use current-state metrics")
        required_metrics.update(program.referenced_metrics)
        compiled_stages.append(
            CompiledProcessStage(
                stage_id=stage.stage_id,
                kind=stage.kind,
                ordinal=stage.ordinal,
                description=stage.description,
                program=program,
                program_fingerprint=program.fingerprint,
            )
        )
    contract = CompiledProcessContract(
        contract_format=_CONTRACT_FORMAT,
        process_contract_id=(
            f"rexpolicy_process_v1/{canonical_process.process_spec_id}"
        ),
        process_spec_id=canonical_process.process_spec_id,
        process_spec_fingerprint=canonical_process.fingerprint,
        process_compiler_policy_id=process_policy.policy_id,
        process_compiler_policy_fingerprint=process_policy.fingerprint,
        task_contract_id=task_contract.task_contract_id,
        task_contract_fingerprint=task_contract.fingerprint,
        task_spec_fingerprint=canonical_task.fingerprint,
        task_compiler_policy_fingerprint=task_policy.fingerprint,
        capability_catalog_fingerprint=canonical_catalog.fingerprint,
        event_schema_id=schema.event_schema_id,
        event_schema_fingerprint=schema.fingerprint,
        default_stage_id=canonical_process.default_stage.stage_id,
        default_stage_ordinal=canonical_process.default_stage.ordinal,
        default_stage_description=canonical_process.default_stage.description,
        stages=tuple(compiled_stages),
        max_segment_steps=canonical_process.segmentation.max_segment_steps,
        required_metrics=tuple(sorted(required_metrics)),
    )
    validate_compiled_process_contract(
        contract,
        task_contract=task_contract,
        catalog=canonical_catalog,
        process_policy=process_policy,
        task_policy=task_policy,
    )
    return contract


def validate_compiled_process_contract(
    contract: CompiledProcessContract,
    *,
    task_contract: CompiledTaskContract,
    catalog: CapabilityCatalog,
    process_policy: ProcessCompilerPolicy = DEFAULT_PROCESS_COMPILER_POLICY,
    task_policy: TaskCompilerPolicy = DEFAULT_TASK_COMPILER_POLICY,
) -> None:
    """Fail closed when any process compiler or task dependency drifts."""
    process_policy.validate()
    canonical_catalog = CapabilityCatalog.from_record(catalog.to_record())
    validate_compiled_task_contract(
        task_contract,
        catalog=canonical_catalog,
        policy=task_policy,
    )
    if contract.contract_format != _CONTRACT_FORMAT:
        raise ValueError("Compiled process contract format mismatch")
    if (
        contract.process_compiler_policy_id != process_policy.policy_id
        or contract.process_compiler_policy_fingerprint
        != process_policy.fingerprint
    ):
        raise ValueError("Compiled process compiler policy fingerprint mismatch")
    if (
        contract.task_contract_id != task_contract.task_contract_id
        or contract.task_contract_fingerprint != task_contract.fingerprint
        or contract.task_spec_fingerprint != task_contract.task_spec_fingerprint
        or contract.task_compiler_policy_fingerprint != task_policy.fingerprint
    ):
        raise ValueError("Compiled process task binding mismatch")
    if contract.process_spec_id != task_contract.process_spec_id:
        raise ValueError("Compiled process identity mismatch")
    if contract.process_contract_id != (
        f"rexpolicy_process_v1/{contract.process_spec_id}"
    ):
        raise ValueError("Compiled process contract identity mismatch")
    _sha256(contract.process_spec_fingerprint, "process_spec_fingerprint")
    if contract.capability_catalog_fingerprint != canonical_catalog.fingerprint:
        raise ValueError("Compiled process capability fingerprint mismatch")
    schemas = {
        item.event_schema_id: item for item in canonical_catalog.event_schemas
    }
    try:
        schema = schemas[contract.event_schema_id]
    except KeyError as error:
        raise ValueError("Compiled process EventSchema is unavailable") from error
    if (
        contract.event_schema_id != task_contract.event_schema_id
        or contract.event_schema_fingerprint != schema.fingerprint
    ):
        raise ValueError("Compiled process EventSchema binding mismatch")
    if not 1 <= contract.max_segment_steps <= min(
        process_policy.max_segment_steps,
        task_contract.episode_control_steps,
    ):
        raise ValueError("Compiled process segment length exceeds trusted bounds")
    if not contract.stages or len(contract.stages) > process_policy.max_stages:
        raise ValueError("Compiled process stage count is outside trusted bounds")
    if contract.default_stage_ordinal != 0:
        raise ValueError("Compiled process default ordinal must be zero")
    stage_ids = [contract.default_stage_id]
    progress_ordinals = [contract.default_stage_ordinal]
    required_metrics: set[str] = set()
    progress_seen = False
    for stage in contract.stages:
        _identifier(stage.stage_id, "compiled_process.stage_id")
        if stage.kind not in {"progress", "unsafe"}:
            raise ValueError("Compiled process stage kind is invalid")
        if stage.kind == "progress":
            progress_seen = True
            if type(stage.ordinal) is not int or stage.ordinal < 0:
                raise ValueError("Compiled process progress ordinal is invalid")
            progress_ordinals.append(stage.ordinal)
        elif stage.ordinal is not None:
            raise ValueError("Compiled unsafe process stage has an ordinal")
        elif progress_seen:
            raise ValueError(
                "Compiled unsafe process stages must precede progress stages"
            )
        if not isinstance(stage.description, str) or not stage.description:
            raise ValueError("Compiled process stage description is invalid")
        if stage.program_fingerprint != stage.program.fingerprint:
            raise ValueError("Compiled process program fingerprint mismatch")
        program = stage.program
        if (
            program.purpose != "process"
            or program.result_type != "boolean"
            or program.result_unit != "bool"
            or program.task_spec_fingerprint != task_contract.task_spec_fingerprint
            or program.event_schema_id != schema.event_schema_id
            or program.event_schema_fingerprint != schema.fingerprint
        ):
            raise ValueError("Compiled process program binding mismatch")
        if len(program.instructions) > process_policy.expression_limits.max_instructions:
            raise ValueError("Compiled process expression exceeds trusted limits")
        if program.max_stack > process_policy.expression_limits.max_stack:
            raise ValueError("Compiled process expression exceeds trusted stack")
        if process_policy.require_current_state_predicates and _has_previous_metric(
            program
        ):
            raise ValueError("Compiled process predicate reads previous state")
        if any(
            "process" not in schema.by_name[name].allowed_purposes
            for name in program.referenced_metrics
        ):
            raise ValueError("Compiled process expression escalates metric purpose")
        required_metrics.update(program.referenced_metrics)
        stage_ids.append(stage.stage_id)
    if len(set(stage_ids)) != len(stage_ids):
        raise ValueError("Compiled process stage IDs must be unique")
    if len(set(progress_ordinals)) != len(progress_ordinals) or set(
        progress_ordinals
    ) != set(range(len(progress_ordinals))):
        raise ValueError("Compiled process ordinals must be contiguous from zero")
    if contract.required_metrics != tuple(sorted(required_metrics)):
        raise ValueError("Compiled process required metrics mismatch")
