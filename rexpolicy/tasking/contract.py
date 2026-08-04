"""Compilation of TaskSpec v2 into immutable trusted runtime contracts."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any

from .canonical import canonical_fingerprint
from .capabilities import (
    AdapterCapability,
    CapabilityCatalog,
    EventSchema,
    ParameterCapability,
    validate_task_capabilities,
)
from .expression import CompileLimits, ExpressionProgram, compile_expression
from .immutable import thaw_json
from .model import TaskSpecV2

_IDENTIFIER = re.compile(r"^[a-z][a-z0-9_.-]*(?:/[a-z0-9_.-]+)*$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_CONTRACT_FORMAT = "rexpolicy_task_contract/v1"


def _identifier(value: Any, path: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise ValueError(f"{path} must be a versioned identifier")
    return value


def _sha256(value: Any, path: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{path} must be a lowercase SHA-256")
    return value


def _positive_integer(value: Any, path: str) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{path} must be a positive integer")
    return value


def _finite_positive(value: Any, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{path} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{path} must be positive and finite")
    return result


def _frozen_pairs(values: Any) -> tuple[tuple[str, Any], ...]:
    return tuple(
        (key, value)
        for key, value in sorted(values.items())
    )


@dataclass(frozen=True)
class TaskCompilerPolicy:
    """Trusted limits that untrusted author proposals cannot relax."""

    policy_id: str
    expression_limits: CompileLimits
    expression_bytecode_version: int
    max_goal_hold_steps: int
    max_terminal_rules: int
    max_reward_profiles: int
    max_terms_per_profile: int
    max_abs_weight: float
    reserved_task_ids: tuple[str, ...]
    reserved_reward_profile_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "reserved_task_ids", tuple(self.reserved_task_ids))
        object.__setattr__(
            self,
            "reserved_reward_profile_ids",
            tuple(self.reserved_reward_profile_ids),
        )
        self.validate()

    def validate(self) -> None:
        _identifier(self.policy_id, "compiler_policy.policy_id")
        self.expression_limits.validate()
        if self.expression_bytecode_version != 1:
            raise ValueError("Compiler policy bytecode version must be 1")
        for name, value in (
            ("max_goal_hold_steps", self.max_goal_hold_steps),
            ("max_terminal_rules", self.max_terminal_rules),
            ("max_reward_profiles", self.max_reward_profiles),
            ("max_terms_per_profile", self.max_terms_per_profile),
        ):
            _positive_integer(value, f"compiler_policy.{name}")
        _finite_positive(self.max_abs_weight, "compiler_policy.max_abs_weight")
        for label, values in (
            ("reserved_task_ids", self.reserved_task_ids),
            ("reserved_reward_profile_ids", self.reserved_reward_profile_ids),
        ):
            if len(set(values)) != len(values):
                raise ValueError(f"Compiler policy {label} must be unique")
            for index, value in enumerate(values):
                _identifier(value, f"compiler_policy.{label}[{index}]")

    def to_record(self) -> dict[str, Any]:
        return {
            "policy_id": self.policy_id,
            "expression_limits": {
                "max_nodes": self.expression_limits.max_nodes,
                "max_instructions": self.expression_limits.max_instructions,
                "max_stack": self.expression_limits.max_stack,
                "minimum_abs_divisor": (
                    self.expression_limits.minimum_abs_divisor
                ),
            },
            "expression_bytecode_version": self.expression_bytecode_version,
            "max_goal_hold_steps": self.max_goal_hold_steps,
            "max_terminal_rules": self.max_terminal_rules,
            "max_reward_profiles": self.max_reward_profiles,
            "max_terms_per_profile": self.max_terms_per_profile,
            "max_abs_weight": self.max_abs_weight,
            "reserved_task_ids": list(self.reserved_task_ids),
            "reserved_reward_profile_ids": list(
                self.reserved_reward_profile_ids
            ),
        }

    @property
    def fingerprint(self) -> str:
        return canonical_fingerprint(self.to_record())


DEFAULT_TASK_COMPILER_POLICY = TaskCompilerPolicy(
    policy_id="rexpolicy/task_compiler/v1",
    expression_limits=CompileLimits(),
    expression_bytecode_version=1,
    max_goal_hold_steps=256,
    max_terminal_rules=32,
    max_reward_profiles=16,
    max_terms_per_profile=32,
    max_abs_weight=100.0,
    reserved_task_ids=("reach_green_cap/v1",),
    reserved_reward_profile_ids=("reach_progress/v1",),
)


@dataclass(frozen=True)
class CompiledTerminalRule:
    code: str
    kind: str
    program: ExpressionProgram
    program_fingerprint: str

    def to_record(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "kind": self.kind,
            "program_fingerprint": self.program_fingerprint,
            "program": self.program.to_record(),
        }


@dataclass(frozen=True)
class CompiledRewardTerm:
    term_id: str
    weight: float
    program: ExpressionProgram
    program_fingerprint: str

    def to_record(self) -> dict[str, Any]:
        return {
            "term_id": self.term_id,
            "weight": self.weight,
            "program_fingerprint": self.program_fingerprint,
            "program": self.program.to_record(),
        }


@dataclass(frozen=True)
class CompiledRewardProfile:
    reward_profile_id: str
    terms: tuple[CompiledRewardTerm, ...]
    shaping_bounds: tuple[float, float]
    success_value: float
    failure_value: float

    def to_record(self) -> dict[str, Any]:
        return {
            "reward_profile_id": self.reward_profile_id,
            "terms": [term.to_record() for term in self.terms],
            "shaping_bounds": list(self.shaping_bounds),
            "success_value": self.success_value,
            "failure_value": self.failure_value,
        }

    @property
    def fingerprint(self) -> str:
        return canonical_fingerprint(self.to_record())


@dataclass(frozen=True)
class CompiledTaskContract:
    contract_format: str
    task_contract_id: str
    compiler_policy_id: str
    compiler_policy_fingerprint: str
    task_id: str
    task_version: int
    task_spec_fingerprint: str
    capability_catalog_fingerprint: str
    adapter_id: str
    adapter_fingerprint: str
    event_schema_id: str
    event_schema_fingerprint: str
    process_spec_id: str
    curriculum_prerequisite_ids: tuple[str, ...]
    environment_bindings: tuple[tuple[str, Any], ...]
    environment_parameters: tuple[tuple[str, Any], ...]
    required_assets: tuple[str, ...]
    action_projection: tuple[tuple[str, tuple[bool, ...]], ...]
    episode_control_steps: int
    goal_hold_steps: int
    goal_program: ExpressionProgram
    goal_program_fingerprint: str
    terminal_rules: tuple[CompiledTerminalRule, ...]
    reward_profiles: tuple[CompiledRewardProfile, ...]
    required_metrics: tuple[str, ...]

    def to_record(self) -> dict[str, Any]:
        return {
            "contract_format": self.contract_format,
            "task_contract_id": self.task_contract_id,
            "compiler_policy_id": self.compiler_policy_id,
            "compiler_policy_fingerprint": self.compiler_policy_fingerprint,
            "task_id": self.task_id,
            "task_version": self.task_version,
            "task_spec_fingerprint": self.task_spec_fingerprint,
            "capability_catalog_fingerprint": (
                self.capability_catalog_fingerprint
            ),
            "adapter_id": self.adapter_id,
            "adapter_fingerprint": self.adapter_fingerprint,
            "event_schema_id": self.event_schema_id,
            "event_schema_fingerprint": self.event_schema_fingerprint,
            "process_spec_id": self.process_spec_id,
            "curriculum_prerequisite_ids": list(
                self.curriculum_prerequisite_ids
            ),
            "environment_bindings": thaw_json(dict(self.environment_bindings)),
            "environment_parameters": thaw_json(
                dict(self.environment_parameters)
            ),
            "required_assets": list(self.required_assets),
            "action_projection": {
                name: list(mask) for name, mask in self.action_projection
            },
            "episode_control_steps": self.episode_control_steps,
            "goal_hold_steps": self.goal_hold_steps,
            "goal_program_fingerprint": self.goal_program_fingerprint,
            "goal_program": self.goal_program.to_record(),
            "terminal_rules": [rule.to_record() for rule in self.terminal_rules],
            "reward_profiles": [
                profile.to_record() for profile in self.reward_profiles
            ],
            "required_metrics": list(self.required_metrics),
        }

    @property
    def fingerprint(self) -> str:
        return canonical_fingerprint(self.to_record())

    def oracle_record(self) -> dict[str, Any]:
        """Return reward-free success/failure/environment semantics."""
        oracle_metrics = set(self.goal_program.referenced_metrics)
        terminal_rules = []
        for rule in self.terminal_rules:
            oracle_metrics.update(rule.program.referenced_metrics)
            terminal_rules.append(
                {
                    "code": rule.code,
                    "kind": rule.kind,
                    "program": _oracle_program_record(rule.program),
                }
            )
        return {
            "oracle_format": "rexpolicy_task_oracle/v1",
            "task_contract_id": self.task_contract_id,
            "compiler_policy_id": self.compiler_policy_id,
            "compiler_policy_fingerprint": self.compiler_policy_fingerprint,
            "task_id": self.task_id,
            "task_version": self.task_version,
            "capability_catalog_fingerprint": (
                self.capability_catalog_fingerprint
            ),
            "adapter_id": self.adapter_id,
            "adapter_fingerprint": self.adapter_fingerprint,
            "event_schema_id": self.event_schema_id,
            "event_schema_fingerprint": self.event_schema_fingerprint,
            "process_spec_id": self.process_spec_id,
            "environment_bindings": thaw_json(dict(self.environment_bindings)),
            "environment_parameters": thaw_json(
                dict(self.environment_parameters)
            ),
            "required_assets": list(self.required_assets),
            "action_projection": {
                name: list(mask) for name, mask in self.action_projection
            },
            "episode_control_steps": self.episode_control_steps,
            "goal_hold_steps": self.goal_hold_steps,
            "goal_program": _oracle_program_record(self.goal_program),
            "terminal_rules": terminal_rules,
            "required_metrics": sorted(oracle_metrics),
        }

    @property
    def oracle_fingerprint(self) -> str:
        """Identify task truth independently of reward-profile semantics."""
        return canonical_fingerprint(self.oracle_record())


def _oracle_program_record(program: ExpressionProgram) -> dict[str, Any]:
    """Strip the full TaskSpec hash while retaining executable semantics."""
    record = program.to_record()
    del record["task_spec_fingerprint"]
    return record


def _canonical_inputs(
    task: TaskSpecV2,
    catalog: CapabilityCatalog,
) -> tuple[TaskSpecV2, CapabilityCatalog]:
    """Copy even directly-constructed dataclasses through strict value models."""
    canonical_task = TaskSpecV2.from_record(task.to_record())
    canonical_catalog = CapabilityCatalog.from_record(catalog.to_record())
    return canonical_task, canonical_catalog


def _parameter_capabilities(
    adapter: AdapterCapability,
) -> dict[str, ParameterCapability]:
    return {item.name: item for item in adapter.parameters}


def _compile_program(
    expression: Any,
    *,
    task: TaskSpecV2,
    schema: EventSchema,
    adapter: AdapterCapability,
    purpose: str,
    policy: TaskCompilerPolicy,
) -> ExpressionProgram:
    return compile_expression(
        expression,
        schema=schema,
        parameters=task.environment.parameters,
        parameter_capabilities=_parameter_capabilities(adapter),
        purpose=purpose,
        task_spec_fingerprint=task.fingerprint,
        limits=policy.expression_limits,
    )


def compile_task_contract(
    task: TaskSpecV2,
    *,
    catalog: CapabilityCatalog,
    policy: TaskCompilerPolicy = DEFAULT_TASK_COMPILER_POLICY,
) -> CompiledTaskContract:
    """Compile a canonical task into a policy-bound immutable contract."""
    policy.validate()
    canonical_task, canonical_catalog = _canonical_inputs(task, catalog)
    resolved = validate_task_capabilities(canonical_task, canonical_catalog)
    adapter = resolved.adapter
    schema = resolved.event_schema

    if canonical_task.task_id in policy.reserved_task_ids:
        raise ValueError("Task ID is reserved for legacy semantics")
    if canonical_task.task_id.startswith("rexpolicy_taskspec_"):
        raise ValueError("Task ID uses a reserved contract namespace")
    if canonical_task.goal.hold_steps > min(
        canonical_task.environment.episode_control_steps,
        policy.max_goal_hold_steps,
    ):
        raise ValueError("Goal hold_steps exceeds the trusted episode horizon")
    if len(canonical_task.terminal_rules) > policy.max_terminal_rules:
        raise ValueError("Task exceeds the trusted terminal-rule limit")
    if len(canonical_task.reward_profiles) > policy.max_reward_profiles:
        raise ValueError("Task exceeds the trusted reward-profile limit")

    goal_program = _compile_program(
        canonical_task.goal.predicate,
        task=canonical_task,
        schema=schema,
        adapter=adapter,
        purpose="goal",
        policy=policy,
    )
    terminal_rules = []
    for rule in sorted(canonical_task.terminal_rules, key=lambda item: item.code):
        program = _compile_program(
            rule.predicate,
            task=canonical_task,
            schema=schema,
            adapter=adapter,
            purpose=rule.kind,
            policy=policy,
        )
        terminal_rules.append(
            CompiledTerminalRule(
                code=rule.code,
                kind=rule.kind,
                program=program,
                program_fingerprint=program.fingerprint,
            )
        )

    reward_profiles = []
    for profile in sorted(
        canonical_task.reward_profiles,
        key=lambda item: item.reward_profile_id,
    ):
        if profile.reward_profile_id in policy.reserved_reward_profile_ids:
            raise ValueError("Reward profile ID is reserved for legacy semantics")
        if len(profile.terms) > policy.max_terms_per_profile:
            raise ValueError("Reward profile exceeds the trusted term limit")
        lower, upper = profile.shaping_bounds
        success = profile.terminal_values["success"]
        failure = profile.terminal_values["failure"]
        if not failure <= lower <= upper <= success:
            raise ValueError(
                "Reward terminal values must bound the shaping interval"
            )
        compiled_terms = []
        for term in sorted(profile.terms, key=lambda item: item.term_id):
            if term.weight == 0.0 or abs(term.weight) > policy.max_abs_weight:
                raise ValueError("Reward term weight is outside trusted bounds")
            program = _compile_program(
                term.expression,
                task=canonical_task,
                schema=schema,
                adapter=adapter,
                purpose="reward",
                policy=policy,
            )
            compiled_terms.append(
                CompiledRewardTerm(
                    term_id=term.term_id,
                    weight=term.weight,
                    program=program,
                    program_fingerprint=program.fingerprint,
                )
            )
        reward_profiles.append(
            CompiledRewardProfile(
                reward_profile_id=profile.reward_profile_id,
                terms=tuple(compiled_terms),
                shaping_bounds=profile.shaping_bounds,
                success_value=success,
                failure_value=failure,
            )
        )

    required_metrics = set(goal_program.referenced_metrics)
    for rule in terminal_rules:
        required_metrics.update(rule.program.referenced_metrics)
    for profile in reward_profiles:
        for term in profile.terms:
            required_metrics.update(term.program.referenced_metrics)
    contract = CompiledTaskContract(
        contract_format=_CONTRACT_FORMAT,
        task_contract_id=f"rexpolicy_taskspec_v2/{canonical_task.task_id}",
        compiler_policy_id=policy.policy_id,
        compiler_policy_fingerprint=policy.fingerprint,
        task_id=canonical_task.task_id,
        task_version=canonical_task.version,
        task_spec_fingerprint=canonical_task.fingerprint,
        capability_catalog_fingerprint=canonical_catalog.fingerprint,
        adapter_id=adapter.adapter_id,
        adapter_fingerprint=adapter.fingerprint,
        event_schema_id=schema.event_schema_id,
        event_schema_fingerprint=schema.fingerprint,
        process_spec_id=canonical_task.process_spec_id,
        curriculum_prerequisite_ids=tuple(
            sorted(canonical_task.curriculum_prerequisite_ids)
        ),
        environment_bindings=_frozen_pairs(canonical_task.environment.bindings),
        environment_parameters=_frozen_pairs(
            canonical_task.environment.parameters
        ),
        required_assets=tuple(sorted(canonical_task.environment.required_assets)),
        action_projection=tuple(
            (name, tuple(mask))
            for name, mask in sorted(canonical_task.action_projection.items())
        ),
        episode_control_steps=canonical_task.environment.episode_control_steps,
        goal_hold_steps=canonical_task.goal.hold_steps,
        goal_program=goal_program,
        goal_program_fingerprint=goal_program.fingerprint,
        terminal_rules=tuple(terminal_rules),
        reward_profiles=tuple(reward_profiles),
        required_metrics=tuple(sorted(required_metrics)),
    )
    validate_compiled_task_contract(
        contract,
        catalog=canonical_catalog,
        policy=policy,
    )
    return contract


def _validate_program_binding(
    program: ExpressionProgram,
    stored_fingerprint: str,
    *,
    contract: CompiledTaskContract,
    schema: EventSchema,
    purpose: str,
    policy: TaskCompilerPolicy,
) -> None:
    _sha256(stored_fingerprint, "program_fingerprint")
    if stored_fingerprint != program.fingerprint:
        raise ValueError("Compiled expression program fingerprint mismatch")
    if (
        program.bytecode_version != policy.expression_bytecode_version
        or program.purpose != purpose
        or program.task_spec_fingerprint != contract.task_spec_fingerprint
        or program.event_schema_id != schema.event_schema_id
        or program.event_schema_fingerprint != schema.fingerprint
    ):
        raise ValueError("Compiled expression program binding mismatch")
    expected_type = (
        "boolean" if purpose in {"goal", "failure", "safety"} else "number"
    )
    if program.result_type != expected_type:
        raise ValueError("Compiled expression program result type mismatch")
    if purpose == "reward" and program.result_unit != "1":
        raise ValueError("Compiled reward program is not unitless")
    if len(program.instructions) > policy.expression_limits.max_instructions:
        raise ValueError("Compiled expression exceeds trusted instruction limits")
    if program.max_stack > policy.expression_limits.max_stack:
        raise ValueError("Compiled expression exceeds trusted stack limits")
    unknown_metrics = set(program.referenced_metrics).difference(schema.by_name)
    if tuple(sorted(set(program.referenced_metrics))) != program.referenced_metrics:
        raise ValueError("Compiled expression metric references are not canonical")
    if unknown_metrics:
        raise ValueError("Compiled expression references unknown metrics")
    if any(
        purpose not in schema.by_name[name].allowed_purposes
        for name in program.referenced_metrics
    ):
        raise ValueError("Compiled expression escalates metric purpose")


def validate_compiled_task_contract(
    contract: CompiledTaskContract,
    *,
    catalog: CapabilityCatalog,
    policy: TaskCompilerPolicy = DEFAULT_TASK_COMPILER_POLICY,
) -> None:
    """Fail closed if a compiled contract or trusted dependency drifted."""
    policy.validate()
    canonical_catalog = CapabilityCatalog.from_record(catalog.to_record())
    if contract.contract_format != _CONTRACT_FORMAT:
        raise ValueError("Compiled task contract format mismatch")
    if (
        contract.compiler_policy_id != policy.policy_id
        or contract.compiler_policy_fingerprint != policy.fingerprint
    ):
        raise ValueError("Compiled task compiler policy fingerprint mismatch")
    if contract.capability_catalog_fingerprint != canonical_catalog.fingerprint:
        raise ValueError("Compiled task capability catalog fingerprint mismatch")
    _sha256(contract.task_spec_fingerprint, "task_spec_fingerprint")
    if contract.task_contract_id != f"rexpolicy_taskspec_v2/{contract.task_id}":
        raise ValueError("Compiled task contract identity mismatch")
    _identifier(contract.task_id, "contract.task_id")
    _positive_integer(contract.task_version, "contract.task_version")
    if not contract.task_id.endswith(f"/v{contract.task_version}"):
        raise ValueError("Compiled task ID/version mismatch")
    if contract.task_id in policy.reserved_task_ids:
        raise ValueError("Compiled task uses a legacy identity")
    adapters = {
        item.adapter_id: item for item in canonical_catalog.adapters
    }
    schemas = {
        item.event_schema_id: item for item in canonical_catalog.event_schemas
    }
    try:
        adapter = adapters[contract.adapter_id]
        schema = schemas[contract.event_schema_id]
    except KeyError as error:
        raise ValueError("Compiled task references an unknown capability") from error
    if (
        contract.adapter_fingerprint != adapter.fingerprint
        or contract.event_schema_fingerprint != schema.fingerprint
        or adapter.event_schema_id != schema.event_schema_id
    ):
        raise ValueError("Compiled task capability fingerprint mismatch")
    if not 1 <= contract.episode_control_steps <= adapter.max_episode_control_steps:
        raise ValueError("Compiled task episode horizon mismatch")
    if contract.process_spec_id not in adapter.process_spec_ids:
        raise ValueError("Compiled task process binding mismatch")
    if tuple(sorted(contract.environment_bindings)) != contract.environment_bindings:
        raise ValueError("Compiled environment bindings are not canonical")
    bindings = dict(contract.environment_bindings)
    if len(bindings) != len(contract.environment_bindings) or set(bindings) != set(
        adapter.bindings
    ):
        raise ValueError("Compiled environment bindings mismatch")
    for name, value in bindings.items():
        if not any(
            type(value) is type(allowed) and value == allowed
            for allowed in adapter.bindings[name]
        ):
            raise ValueError("Compiled environment binding is not allowed")
    if tuple(sorted(contract.environment_parameters)) != (
        contract.environment_parameters
    ):
        raise ValueError("Compiled environment parameters are not canonical")
    parameters = dict(contract.environment_parameters)
    if len(parameters) != len(contract.environment_parameters):
        raise ValueError("Compiled environment parameters contain duplicates")
    parameter_capabilities = _parameter_capabilities(adapter)
    if set(parameters).difference(parameter_capabilities):
        raise ValueError("Compiled environment parameter is not allowed")
    if any(
        item.required and item.name not in parameters
        for item in adapter.parameters
    ):
        raise ValueError("Compiled environment parameter is missing")
    for name, value in parameters.items():
        parameter_capabilities[name].validate_value(
            value,
            f"contract.environment_parameters.{name}",
        )
    expected_projection = tuple(
        (name, tuple(mask))
        for name, mask in sorted(adapter.action_projection.items())
    )
    if contract.action_projection != expected_projection:
        raise ValueError("Compiled action projection mismatch")
    if not set(adapter.required_assets).issubset(contract.required_assets) or not set(
        contract.required_assets
    ).issubset(adapter.allowed_assets):
        raise ValueError("Compiled required assets mismatch")
    if not 1 <= contract.goal_hold_steps <= min(
        policy.max_goal_hold_steps,
        contract.episode_control_steps,
    ):
        raise ValueError("Compiled task goal hold is outside trusted bounds")
    _validate_program_binding(
        contract.goal_program,
        contract.goal_program_fingerprint,
        contract=contract,
        schema=schema,
        purpose="goal",
        policy=policy,
    )
    if len(contract.terminal_rules) > policy.max_terminal_rules:
        raise ValueError("Compiled task has too many terminal rules")
    rule_codes = [rule.code for rule in contract.terminal_rules]
    if rule_codes != sorted(rule_codes) or len(set(rule_codes)) != len(rule_codes):
        raise ValueError("Compiled terminal rules are not canonical")
    for rule in contract.terminal_rules:
        if rule.kind not in {"failure", "safety"}:
            raise ValueError("Compiled terminal rule kind is invalid")
        _validate_program_binding(
            rule.program,
            rule.program_fingerprint,
            contract=contract,
            schema=schema,
            purpose=rule.kind,
            policy=policy,
        )
    if len(contract.reward_profiles) > policy.max_reward_profiles:
        raise ValueError("Compiled task has too many reward profiles")
    profile_ids = [item.reward_profile_id for item in contract.reward_profiles]
    if profile_ids != sorted(profile_ids) or len(set(profile_ids)) != len(profile_ids):
        raise ValueError("Compiled reward profiles are not canonical")
    for profile in contract.reward_profiles:
        if profile.reward_profile_id in policy.reserved_reward_profile_ids:
            raise ValueError("Compiled reward profile uses a legacy identity")
        lower, upper = profile.shaping_bounds
        reward_values = (
            profile.failure_value,
            lower,
            upper,
            profile.success_value,
        )
        if not all(math.isfinite(value) for value in reward_values) or not (
            profile.failure_value <= lower <= upper <= profile.success_value
        ):
            raise ValueError("Compiled reward terminal ordering is invalid")
        if len(profile.terms) > policy.max_terms_per_profile:
            raise ValueError("Compiled reward profile has too many terms")
        term_ids = [item.term_id for item in profile.terms]
        if term_ids != sorted(term_ids) or len(set(term_ids)) != len(term_ids):
            raise ValueError("Compiled reward terms are not canonical")
        for term in profile.terms:
            if (
                not math.isfinite(term.weight)
                or term.weight == 0.0
                or abs(term.weight) > policy.max_abs_weight
            ):
                raise ValueError("Compiled reward weight is outside trusted bounds")
            _validate_program_binding(
                term.program,
                term.program_fingerprint,
                contract=contract,
                schema=schema,
                purpose="reward",
                policy=policy,
            )
    required_metrics = set(contract.goal_program.referenced_metrics)
    for rule in contract.terminal_rules:
        required_metrics.update(rule.program.referenced_metrics)
    for profile in contract.reward_profiles:
        for term in profile.terms:
            required_metrics.update(term.program.referenced_metrics)
    if contract.required_metrics != tuple(sorted(required_metrics)):
        raise ValueError("Compiled required metrics mismatch")
