"""Fail-closed static certification for automatically authored TaskSpecs.

The proposer output handled here is untrusted.  This module is deliberately
limited to deterministic, CPU-only checks: strict JSON decoding, intent locks,
trusted compilation, and the existing public property suite.  A successful
result is still quarantined; it carries no admission or activation authority.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .authoring import (
    DEFAULT_AUTHORING_POLICY,
    AuthoringIntent,
    AuthoringPolicy,
    ProposalAttempt,
    PublicIssue,
    normalize_public_issues,
)
from .authoring_brief import (
    CandidateBriefViolation,
    PublicAuthoringBrief,
    validate_candidate_against_brief,
)
from .canonical import canonical_fingerprint, canonical_json, strict_json_loads
from .capabilities import CapabilityCatalog
from .contract import (
    DEFAULT_TASK_COMPILER_POLICY,
    CompiledTaskContract,
    TaskCompilerPolicy,
    compile_task_contract,
    validate_compiled_task_contract,
)
from .immutable import freeze_json, thaw_json
from .model import TaskSpecV2
from .property_validation import (
    DEFAULT_PROPERTY_VALIDATION_POLICY,
    PropertyValidationPolicy,
    TaskPropertyReport,
    validate_task_properties,
)
from .process import ProcessSpecV1
from .process_contract import (
    DEFAULT_PROCESS_COMPILER_POLICY,
    CompiledProcessContract,
    ProcessCompilerPolicy,
    compile_process_contract,
)

_IDENTIFIER = re.compile(r"^[a-z][a-z0-9_.-]*(?:/[a-z0-9_.-]+)*$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_CANDIDATE_STATE = "quarantined_static"
_ADMISSION_STATUS = "not_admitted"

_BUNDLE_KEYS = {
    "schema_version",
    "candidate_state",
    "admission_status",
    "intent_fingerprint",
    "attempt_fingerprint",
    "raw_response_sha256",
    "raw_response_bytes",
    "task_spec_fingerprint",
    "task_spec",
    "task_contract_fingerprint",
    "task_contract",
    "task_oracle_sha256",
    "task_oracle",
    "property_report_fingerprint",
    "property_report",
    "process_spec_fingerprint",
    "process_spec",
    "process_contract_fingerprint",
    "process_contract",
    "capability_catalog_fingerprint",
    "compiler_policy_id",
    "compiler_policy_fingerprint",
    "process_compiler_policy_id",
    "process_compiler_policy_fingerprint",
    "property_policy_id",
    "property_policy_fingerprint",
    "authoring_policy_id",
    "authoring_policy_fingerprint",
    "authoring_brief_id",
    "authoring_brief_fingerprint",
    "preauthoring_audit_plan_fingerprint",
}

_PROCESS_CONTRACT_KEYS = {
    "contract_format",
    "process_contract_id",
    "process_spec_id",
    "process_spec_fingerprint",
    "process_compiler_policy_id",
    "process_compiler_policy_fingerprint",
    "task_contract_id",
    "task_contract_fingerprint",
    "task_spec_fingerprint",
    "task_compiler_policy_fingerprint",
    "capability_catalog_fingerprint",
    "event_schema_id",
    "event_schema_fingerprint",
    "default_stage",
    "stages",
    "max_segment_steps",
    "required_metrics",
}

_CONTRACT_KEYS = {
    "contract_format",
    "task_contract_id",
    "compiler_policy_id",
    "compiler_policy_fingerprint",
    "task_id",
    "task_version",
    "task_spec_fingerprint",
    "capability_catalog_fingerprint",
    "adapter_id",
    "adapter_fingerprint",
    "event_schema_id",
    "event_schema_fingerprint",
    "process_spec_id",
    "curriculum_prerequisite_ids",
    "environment_bindings",
    "environment_parameters",
    "required_assets",
    "action_projection",
    "episode_control_steps",
    "goal_hold_steps",
    "goal_program_fingerprint",
    "goal_program",
    "terminal_rules",
    "reward_profiles",
    "required_metrics",
}

_ORACLE_KEYS = {
    "oracle_format",
    "task_contract_id",
    "compiler_policy_id",
    "compiler_policy_fingerprint",
    "task_id",
    "task_version",
    "capability_catalog_fingerprint",
    "adapter_id",
    "adapter_fingerprint",
    "event_schema_id",
    "event_schema_fingerprint",
    "process_spec_id",
    "environment_bindings",
    "environment_parameters",
    "required_assets",
    "action_projection",
    "episode_control_steps",
    "goal_hold_steps",
    "goal_program",
    "terminal_rules",
    "required_metrics",
}

_REPORT_KEYS = {
    "report_schema_version",
    "task_id",
    "task_spec_fingerprint",
    "contract_fingerprint",
    "validation_policy_id",
    "validation_policy_fingerprint",
    "accepted",
    "issues",
    "example_count",
    "evaluated_step_count",
}


class TrustedAuthoringBindingError(ValueError):
    """A fixed-detail failure for drift in trusted authoring dependencies."""


_TRUSTED_BINDING_MESSAGE = "Trusted authoring dependency binding mismatch"


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


def _sha256(value: Any, path: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{path} must be a lowercase SHA-256")
    return value


def _identifier(value: Any, path: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise ValueError(f"{path} must be a versioned identifier")
    return value


def _nonnegative_integer(value: Any, path: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{path} must be a non-negative integer")
    return value


def _dependency_error() -> TrustedAuthoringBindingError:
    return TrustedAuthoringBindingError(_TRUSTED_BINDING_MESSAGE)


def _canonical_trusted_inputs(
    *,
    intent: AuthoringIntent,
    brief: PublicAuthoringBrief,
    parent_task: TaskSpecV2 | None,
    parent_contract: CompiledTaskContract | None,
    catalog: CapabilityCatalog,
    process_specs: Mapping[str, ProcessSpecV1],
    compiler_policy: TaskCompilerPolicy,
    process_policy: ProcessCompilerPolicy,
    property_policy: PropertyValidationPolicy,
    authoring_policy: AuthoringPolicy,
) -> tuple[
    AuthoringIntent,
    CapabilityCatalog,
    dict[str, ProcessSpecV1],
    AuthoringPolicy,
    PublicAuthoringBrief,
    TaskSpecV2 | None,
    CompiledTaskContract | None,
]:
    """Copy trusted values and verify every dependency locked by the intent."""
    try:
        canonical_authoring_policy = AuthoringPolicy.from_record(
            authoring_policy.to_record()
        )
        canonical_intent = AuthoringIntent.from_record(
            intent.to_record(),
            policy=canonical_authoring_policy,
        )
        canonical_brief = PublicAuthoringBrief.from_record(brief.to_record())
        canonical_brief.validate_against(canonical_intent)
        canonical_catalog = CapabilityCatalog.from_record(catalog.to_record())
        compiler_policy.validate()
        process_policy.validate()
        if not isinstance(process_specs, Mapping) or not process_specs:
            raise ValueError("ProcessSpec registry must be a non-empty mapping")
        canonical_process_specs: dict[str, ProcessSpecV1] = {}
        for process_spec_id, process_spec in process_specs.items():
            _identifier(process_spec_id, "process_specs key")
            canonical_process = ProcessSpecV1.from_record(process_spec.to_record())
            if canonical_process.process_spec_id != process_spec_id:
                raise ValueError("ProcessSpec registry key mismatch")
            canonical_process_specs[process_spec_id] = canonical_process
        process_fingerprints = dict(canonical_intent.process_spec_fingerprints)
        process_bindings_match = (
            canonical_intent.schema_version == 2
            and set(canonical_process_specs)
            == set(canonical_intent.allowed_process_spec_ids)
            and canonical_intent.process_compiler_policy_id
            == process_policy.policy_id
            and canonical_intent.process_compiler_policy_fingerprint
            == process_policy.fingerprint
            and all(
                process_fingerprints.get(process_spec_id)
                == canonical_process_specs[process_spec_id].fingerprint
                for process_spec_id in canonical_intent.allowed_process_spec_ids
            )
        )
        bindings_match = all(
            (
                process_bindings_match,
                canonical_intent.capability_catalog_fingerprint
                == canonical_catalog.fingerprint,
                canonical_intent.compiler_policy_id == compiler_policy.policy_id,
                canonical_intent.compiler_policy_fingerprint
                == compiler_policy.fingerprint,
                canonical_intent.property_policy_id == property_policy.policy_id,
                canonical_intent.property_policy_fingerprint
                == property_policy.fingerprint,
                canonical_intent.authoring_policy_id
                == canonical_authoring_policy.policy_id,
                canonical_intent.authoring_policy_fingerprint
                == canonical_authoring_policy.fingerprint,
            )
        )
        expects_parent = canonical_brief.parent_task_spec_fingerprint is not None
        if expects_parent != (parent_task is not None and parent_contract is not None):
            raise ValueError("Parent authoring artifacts are incomplete")
        if expects_parent:
            assert parent_task is not None and parent_contract is not None
            canonical_parent_task = TaskSpecV2.from_record(parent_task.to_record())
            validate_compiled_task_contract(
                parent_contract,
                catalog=canonical_catalog,
                policy=compiler_policy,
            )
            if (
                canonical_parent_task.fingerprint
                != canonical_brief.parent_task_spec_fingerprint
                or parent_contract.task_spec_fingerprint
                != canonical_parent_task.fingerprint
                or parent_contract.oracle_fingerprint
                != canonical_brief.parent_task_oracle_fingerprint
            ):
                raise ValueError("Parent authoring binding drifted")
            canonical_parent_contract = parent_contract
        else:
            canonical_parent_task = None
            canonical_parent_contract = None
    except Exception:
        raise _dependency_error() from None
    if not bindings_match:
        raise _dependency_error()
    return (
        canonical_intent,
        canonical_catalog,
        canonical_process_specs,
        canonical_authoring_policy,
        canonical_brief,
        canonical_parent_task,
        canonical_parent_contract,
    )


def _response_bytes(raw_stdout: str | bytes) -> bytes:
    if isinstance(raw_stdout, bytes):
        return raw_stdout
    if isinstance(raw_stdout, str):
        try:
            return raw_stdout.encode("utf-8")
        except UnicodeError:
            raise ValueError("Proposer output is not valid UTF-8") from None
    raise TypeError("Proposer output must be text or bytes")


@dataclass(frozen=True)
class StaticCandidateAssessment:
    """A safe decision plus trusted artifacts only when static checks pass."""

    intent_fingerprint: str
    raw_response_sha256: str
    raw_response_bytes: int
    capability_catalog_fingerprint: str
    compiler_policy_id: str
    compiler_policy_fingerprint: str
    process_compiler_policy_id: str
    process_compiler_policy_fingerprint: str
    property_policy_id: str
    property_policy_fingerprint: str
    authoring_policy_id: str
    authoring_policy_fingerprint: str
    authoring_brief_id: str
    authoring_brief_fingerprint: str
    preauthoring_audit_plan_fingerprint: str
    issues: tuple[PublicIssue, ...]
    task_spec: TaskSpecV2 | None
    task_contract: CompiledTaskContract | None
    process_spec: ProcessSpecV1 | None
    process_contract: CompiledProcessContract | None
    property_report: TaskPropertyReport | None

    @property
    def accepted(self) -> bool:
        artifacts_present = all(
            item is not None
            for item in (
                self.task_spec,
                self.task_contract,
                self.process_spec,
                self.process_contract,
                self.property_report,
            )
        )
        return not self.issues and artifacts_present

    def to_bundle(self, attempt: ProposalAttempt) -> StaticCandidateBundle:
        """Bind a successful assessment to its exact hash-only attempt."""
        if not self.accepted:
            raise ValueError("Rejected static assessment cannot form a bundle")
        if (
            attempt.intent_fingerprint != self.intent_fingerprint
            or attempt.raw_response_sha256 != self.raw_response_sha256
            or attempt.raw_response_bytes != self.raw_response_bytes
            or not attempt.response_complete
            or attempt.status != "candidate"
            or attempt.issues
        ):
            raise ValueError("Proposal attempt does not match static assessment")
        task = self.task_spec
        contract = self.task_contract
        process_spec = self.process_spec
        process_contract = self.process_contract
        report = self.property_report
        if (
            task is None
            or contract is None
            or process_spec is None
            or process_contract is None
            or report is None
        ):
            raise ValueError("Accepted static assessment is incomplete")
        record = {
            "schema_version": 1,
            "candidate_state": _CANDIDATE_STATE,
            "admission_status": _ADMISSION_STATUS,
            "intent_fingerprint": self.intent_fingerprint,
            "attempt_fingerprint": attempt.fingerprint,
            "raw_response_sha256": self.raw_response_sha256,
            "raw_response_bytes": self.raw_response_bytes,
            "task_spec_fingerprint": task.fingerprint,
            "task_spec": task.to_record(),
            "task_contract_fingerprint": contract.fingerprint,
            "task_contract": contract.to_record(),
            "task_oracle_sha256": contract.oracle_fingerprint,
            "task_oracle": contract.oracle_record(),
            "property_report_fingerprint": report.fingerprint,
            "property_report": report.to_record(),
            "process_spec_fingerprint": process_spec.fingerprint,
            "process_spec": process_spec.to_record(),
            "process_contract_fingerprint": process_contract.fingerprint,
            "process_contract": process_contract.to_record(),
            "capability_catalog_fingerprint": (
                self.capability_catalog_fingerprint
            ),
            "compiler_policy_id": self.compiler_policy_id,
            "compiler_policy_fingerprint": self.compiler_policy_fingerprint,
            "process_compiler_policy_id": self.process_compiler_policy_id,
            "process_compiler_policy_fingerprint": (
                self.process_compiler_policy_fingerprint
            ),
            "property_policy_id": self.property_policy_id,
            "property_policy_fingerprint": self.property_policy_fingerprint,
            "authoring_policy_id": self.authoring_policy_id,
            "authoring_policy_fingerprint": self.authoring_policy_fingerprint,
            "authoring_brief_id": self.authoring_brief_id,
            "authoring_brief_fingerprint": self.authoring_brief_fingerprint,
            "preauthoring_audit_plan_fingerprint": (
                self.preauthoring_audit_plan_fingerprint
            ),
        }
        return StaticCandidateBundle.from_record(record, attempt=attempt)


def _assessment(
    *,
    intent: AuthoringIntent,
    response_sha256: str,
    response_bytes: int,
    catalog: CapabilityCatalog,
    compiler_policy: TaskCompilerPolicy,
    process_policy: ProcessCompilerPolicy,
    property_policy: PropertyValidationPolicy,
    authoring_policy: AuthoringPolicy,
    issues: tuple[PublicIssue, ...] = (),
    task: TaskSpecV2 | None = None,
    contract: CompiledTaskContract | None = None,
    process_spec: ProcessSpecV1 | None = None,
    process_contract: CompiledProcessContract | None = None,
    report: TaskPropertyReport | None = None,
) -> StaticCandidateAssessment:
    return StaticCandidateAssessment(
        intent_fingerprint=intent.fingerprint,
        raw_response_sha256=response_sha256,
        raw_response_bytes=response_bytes,
        capability_catalog_fingerprint=catalog.fingerprint,
        compiler_policy_id=compiler_policy.policy_id,
        compiler_policy_fingerprint=compiler_policy.fingerprint,
        process_compiler_policy_id=process_policy.policy_id,
        process_compiler_policy_fingerprint=process_policy.fingerprint,
        property_policy_id=property_policy.policy_id,
        property_policy_fingerprint=property_policy.fingerprint,
        authoring_policy_id=authoring_policy.policy_id,
        authoring_policy_fingerprint=authoring_policy.fingerprint,
        authoring_brief_id=intent.authoring_brief_id or "",
        authoring_brief_fingerprint=intent.authoring_brief_fingerprint or "",
        preauthoring_audit_plan_fingerprint=(
            intent.preauthoring_audit_plan_fingerprint or ""
        ),
        issues=issues,
        task_spec=task,
        task_contract=contract,
        process_spec=process_spec,
        process_contract=process_contract,
        property_report=report,
    )


def validate_static_candidate(
    raw_stdout: str | bytes,
    *,
    intent: AuthoringIntent,
    brief: PublicAuthoringBrief,
    parent_task: TaskSpecV2 | None,
    parent_contract: CompiledTaskContract | None,
    catalog: CapabilityCatalog,
    process_specs: Mapping[str, ProcessSpecV1],
    compiler_policy: TaskCompilerPolicy = DEFAULT_TASK_COMPILER_POLICY,
    process_policy: ProcessCompilerPolicy = DEFAULT_PROCESS_COMPILER_POLICY,
    property_policy: PropertyValidationPolicy = (
        DEFAULT_PROPERTY_VALIDATION_POLICY
    ),
    authoring_policy: AuthoringPolicy = DEFAULT_AUTHORING_POLICY,
) -> StaticCandidateAssessment:
    """Apply the complete deterministic static boundary to proposer stdout.

    Candidate-caused failures return only policy-owned :class:`PublicIssue`
    values.  Drift in trusted dependencies is an operator error and therefore
    raises a fixed-detail :class:`TrustedAuthoringBindingError` instead of
    asking an untrusted proposer to repair policy state.
    """
    (
        canonical_intent,
        canonical_catalog,
        canonical_process_specs,
        canonical_authoring_policy,
        canonical_brief,
        canonical_parent_task,
        canonical_parent_contract,
    ) = (
        _canonical_trusted_inputs(
            intent=intent,
            brief=brief,
            parent_task=parent_task,
            parent_contract=parent_contract,
            catalog=catalog,
            process_specs=process_specs,
            compiler_policy=compiler_policy,
            process_policy=process_policy,
            property_policy=property_policy,
            authoring_policy=authoring_policy,
        )
    )
    try:
        payload = _response_bytes(raw_stdout)
    except (TypeError, ValueError):
        payload = b""
        issues = normalize_public_issues(
            (("invalid_json", "$"),),
            policy=canonical_authoring_policy,
        )
        return _assessment(
            intent=canonical_intent,
            response_sha256=canonical_fingerprint({"invalid_utf8": True}),
            response_bytes=0,
            catalog=canonical_catalog,
            compiler_policy=compiler_policy,
            process_policy=process_policy,
            property_policy=property_policy,
            authoring_policy=canonical_authoring_policy,
            issues=issues,
        )

    response_sha256 = hashlib.sha256(payload).hexdigest()
    byte_count = len(payload)
    if byte_count > canonical_authoring_policy.max_response_bytes:
        issues = normalize_public_issues(
            (("response_too_large", "$"),),
            policy=canonical_authoring_policy,
        )
        return _assessment(
            intent=canonical_intent,
            response_sha256=response_sha256,
            response_bytes=byte_count,
            catalog=canonical_catalog,
            compiler_policy=compiler_policy,
            process_policy=process_policy,
            property_policy=property_policy,
            authoring_policy=canonical_authoring_policy,
            issues=issues,
        )
    try:
        text = payload.decode("utf-8")
        proposal_record = strict_json_loads(
            text,
            max_bytes=canonical_authoring_policy.max_response_bytes,
        )
    except Exception:
        issues = normalize_public_issues(
            (("invalid_json", "$"),),
            policy=canonical_authoring_policy,
        )
        return _assessment(
            intent=canonical_intent,
            response_sha256=response_sha256,
            response_bytes=byte_count,
            catalog=canonical_catalog,
            compiler_policy=compiler_policy,
            process_policy=process_policy,
            property_policy=property_policy,
            authoring_policy=canonical_authoring_policy,
            issues=issues,
        )
    try:
        task = TaskSpecV2.from_record(proposal_record)
    except Exception:
        issues = normalize_public_issues(
            (("schema_invalid", "$"),),
            policy=canonical_authoring_policy,
        )
        return _assessment(
            intent=canonical_intent,
            response_sha256=response_sha256,
            response_bytes=byte_count,
            catalog=canonical_catalog,
            compiler_policy=compiler_policy,
            process_policy=process_policy,
            property_policy=property_policy,
            authoring_policy=canonical_authoring_policy,
            issues=issues,
        )

    raw_issues: list[tuple[str, str]] = []
    if task.task_id != canonical_intent.task_id:
        raw_issues.append(("identity_locked", "$.task_id"))
    if task.version != canonical_intent.task_version:
        raw_issues.append(("identity_locked", "$.version"))
    if task.family_id != canonical_intent.family_id:
        raw_issues.append(("identity_locked", "$.family_id"))
    if task.instructions.audit != canonical_intent.audit_commitment:
        raw_issues.append(("audit_commitment_locked", "$.instructions.audit"))
    if task.environment.adapter_id not in canonical_intent.allowed_adapter_ids:
        raw_issues.append(("adapter_not_allowed", "$.environment.adapter_id"))
    if task.event_schema_id not in canonical_intent.allowed_event_schema_ids:
        raw_issues.append(("event_schema_not_allowed", "$.event_schema_id"))
    if task.process_spec_id not in canonical_intent.allowed_process_spec_ids:
        raw_issues.append(("process_spec_not_allowed", "$.process_spec_id"))
    if raw_issues:
        return _assessment(
            intent=canonical_intent,
            response_sha256=response_sha256,
            response_bytes=byte_count,
            catalog=canonical_catalog,
            compiler_policy=compiler_policy,
            process_policy=process_policy,
            property_policy=property_policy,
            authoring_policy=canonical_authoring_policy,
            issues=normalize_public_issues(
                raw_issues,
                policy=canonical_authoring_policy,
            ),
        )

    try:
        validate_candidate_against_brief(
            task,
            brief=canonical_brief,
            parent_task=canonical_parent_task,
            parent_contract=canonical_parent_contract,
        )
    except CandidateBriefViolation:
        return _assessment(
            intent=canonical_intent,
            response_sha256=response_sha256,
            response_bytes=byte_count,
            catalog=canonical_catalog,
            compiler_policy=compiler_policy,
            process_policy=process_policy,
            property_policy=property_policy,
            authoring_policy=canonical_authoring_policy,
            issues=normalize_public_issues(
                (("brief_violation", "$"),),
                policy=canonical_authoring_policy,
            ),
        )

    try:
        contract = compile_task_contract(
            task,
            catalog=canonical_catalog,
            policy=compiler_policy,
        )
    except Exception:
        issues = normalize_public_issues(
            (("compile_rejected", "$.environment"),),
            policy=canonical_authoring_policy,
        )
        return _assessment(
            intent=canonical_intent,
            response_sha256=response_sha256,
            response_bytes=byte_count,
            catalog=canonical_catalog,
            compiler_policy=compiler_policy,
            process_policy=process_policy,
            property_policy=property_policy,
            authoring_policy=canonical_authoring_policy,
            issues=issues,
        )
    try:
        process_spec = canonical_process_specs[task.process_spec_id]
        process_contract = compile_process_contract(
            process_spec,
            task=task,
            task_contract=contract,
            catalog=canonical_catalog,
            process_policy=process_policy,
            task_policy=compiler_policy,
        )
    except Exception:
        issues = normalize_public_issues(
            (("compile_rejected", "$.environment"),),
            policy=canonical_authoring_policy,
        )
        return _assessment(
            intent=canonical_intent,
            response_sha256=response_sha256,
            response_bytes=byte_count,
            catalog=canonical_catalog,
            compiler_policy=compiler_policy,
            process_policy=process_policy,
            property_policy=property_policy,
            authoring_policy=canonical_authoring_policy,
            issues=issues,
        )
    try:
        report = validate_task_properties(
            task,
            contract,
            catalog=canonical_catalog,
            policy=property_policy,
            compiler_policy=compiler_policy,
        )
    except Exception:
        report = None
    if report is None or not report.accepted:
        issues = normalize_public_issues(
            (("property_rejected", "$.validation_examples"),),
            policy=canonical_authoring_policy,
        )
        return _assessment(
            intent=canonical_intent,
            response_sha256=response_sha256,
            response_bytes=byte_count,
            catalog=canonical_catalog,
            compiler_policy=compiler_policy,
            process_policy=process_policy,
            property_policy=property_policy,
            authoring_policy=canonical_authoring_policy,
            issues=issues,
        )
    return _assessment(
        intent=canonical_intent,
        response_sha256=response_sha256,
        response_bytes=byte_count,
        catalog=canonical_catalog,
        compiler_policy=compiler_policy,
        process_policy=process_policy,
        property_policy=property_policy,
        authoring_policy=canonical_authoring_policy,
        task=task,
        contract=contract,
        process_spec=process_spec,
        process_contract=process_contract,
        report=report,
    )


@dataclass(frozen=True)
class StaticCandidateBundle:
    """Immutable evidence for a statically valid but unadmitted candidate."""

    schema_version: int
    candidate_state: str
    admission_status: str
    intent_fingerprint: str
    attempt_fingerprint: str
    raw_response_sha256: str
    raw_response_bytes: int
    task_spec_fingerprint: str
    task_spec: Mapping[str, Any]
    task_contract_fingerprint: str
    task_contract: Mapping[str, Any]
    task_oracle_sha256: str
    task_oracle: Mapping[str, Any]
    property_report_fingerprint: str
    property_report: Mapping[str, Any]
    process_spec_fingerprint: str
    process_spec: Mapping[str, Any]
    process_contract_fingerprint: str
    process_contract: Mapping[str, Any]
    capability_catalog_fingerprint: str
    compiler_policy_id: str
    compiler_policy_fingerprint: str
    process_compiler_policy_id: str
    process_compiler_policy_fingerprint: str
    property_policy_id: str
    property_policy_fingerprint: str
    authoring_policy_id: str
    authoring_policy_fingerprint: str
    authoring_brief_id: str
    authoring_brief_fingerprint: str
    preauthoring_audit_plan_fingerprint: str

    @classmethod
    def from_record(
        cls,
        value: Any,
        *,
        expected_fingerprint: str | None = None,
        intent: AuthoringIntent | None = None,
        brief: PublicAuthoringBrief | None = None,
        parent_task: TaskSpecV2 | None = None,
        parent_contract: CompiledTaskContract | None = None,
        attempt: ProposalAttempt | None = None,
        catalog: CapabilityCatalog | None = None,
        process_specs: Mapping[str, ProcessSpecV1] | None = None,
        compiler_policy: TaskCompilerPolicy = DEFAULT_TASK_COMPILER_POLICY,
        process_policy: ProcessCompilerPolicy = DEFAULT_PROCESS_COMPILER_POLICY,
        property_policy: PropertyValidationPolicy = (
            DEFAULT_PROPERTY_VALIDATION_POLICY
        ),
        authoring_policy: AuthoringPolicy = DEFAULT_AUTHORING_POLICY,
    ) -> StaticCandidateBundle:
        record = _mapping(value, "$static_candidate_bundle")
        _keys(record, _BUNDLE_KEYS, "$static_candidate_bundle")
        if expected_fingerprint is not None:
            _sha256(expected_fingerprint, "expected_fingerprint")
            if canonical_fingerprint(record) != expected_fingerprint:
                raise ValueError("Static candidate bundle fingerprint mismatch")
        if record["schema_version"] != 1:
            raise ValueError("StaticCandidateBundle schema_version must be 1")
        if record["candidate_state"] != _CANDIDATE_STATE:
            raise ValueError("Static candidate must remain quarantined_static")
        if record["admission_status"] != _ADMISSION_STATUS:
            raise ValueError("Static candidate carries no admission authority")

        task_record = _mapping(record["task_spec"], "$static_candidate_bundle.task_spec")
        contract_record = _mapping(
            record["task_contract"],
            "$static_candidate_bundle.task_contract",
        )
        oracle_record = _mapping(
            record["task_oracle"],
            "$static_candidate_bundle.task_oracle",
        )
        report_record = _mapping(
            record["property_report"],
            "$static_candidate_bundle.property_report",
        )
        process_spec_record = _mapping(
            record["process_spec"],
            "$static_candidate_bundle.process_spec",
        )
        process_contract_record = _mapping(
            record["process_contract"],
            "$static_candidate_bundle.process_contract",
        )
        _keys(contract_record, _CONTRACT_KEYS, "$static_candidate_bundle.task_contract")
        _keys(oracle_record, _ORACLE_KEYS, "$static_candidate_bundle.task_oracle")
        _keys(report_record, _REPORT_KEYS, "$static_candidate_bundle.property_report")
        _keys(
            process_contract_record,
            _PROCESS_CONTRACT_KEYS,
            "$static_candidate_bundle.process_contract",
        )

        task = TaskSpecV2.from_record(task_record)
        process_spec = ProcessSpecV1.from_record(process_spec_record)
        task_fingerprint = _sha256(
            record["task_spec_fingerprint"],
            "$static_candidate_bundle.task_spec_fingerprint",
        )
        contract_fingerprint = _sha256(
            record["task_contract_fingerprint"],
            "$static_candidate_bundle.task_contract_fingerprint",
        )
        oracle_sha256 = _sha256(
            record["task_oracle_sha256"],
            "$static_candidate_bundle.task_oracle_sha256",
        )
        report_fingerprint = _sha256(
            record["property_report_fingerprint"],
            "$static_candidate_bundle.property_report_fingerprint",
        )
        process_spec_fingerprint = _sha256(
            record["process_spec_fingerprint"],
            "$static_candidate_bundle.process_spec_fingerprint",
        )
        process_contract_fingerprint = _sha256(
            record["process_contract_fingerprint"],
            "$static_candidate_bundle.process_contract_fingerprint",
        )
        if task.fingerprint != task_fingerprint:
            raise ValueError("Static candidate TaskSpec fingerprint mismatch")
        if canonical_fingerprint(contract_record) != contract_fingerprint:
            raise ValueError("Static candidate task contract fingerprint mismatch")
        if canonical_fingerprint(oracle_record) != oracle_sha256:
            raise ValueError("Static candidate task oracle fingerprint mismatch")
        if canonical_fingerprint(report_record) != report_fingerprint:
            raise ValueError("Static candidate property report fingerprint mismatch")
        if process_spec.fingerprint != process_spec_fingerprint:
            raise ValueError("Static candidate ProcessSpec fingerprint mismatch")
        if (
            canonical_fingerprint(process_contract_record)
            != process_contract_fingerprint
        ):
            raise ValueError("Static candidate process contract fingerprint mismatch")

        catalog_fingerprint = _sha256(
            record["capability_catalog_fingerprint"],
            "$static_candidate_bundle.capability_catalog_fingerprint",
        )
        compiler_policy_id = _identifier(
            record["compiler_policy_id"],
            "$static_candidate_bundle.compiler_policy_id",
        )
        compiler_policy_fingerprint = _sha256(
            record["compiler_policy_fingerprint"],
            "$static_candidate_bundle.compiler_policy_fingerprint",
        )
        process_policy_id = _identifier(
            record["process_compiler_policy_id"],
            "$static_candidate_bundle.process_compiler_policy_id",
        )
        process_policy_fingerprint = _sha256(
            record["process_compiler_policy_fingerprint"],
            "$static_candidate_bundle.process_compiler_policy_fingerprint",
        )
        property_policy_id = _identifier(
            record["property_policy_id"],
            "$static_candidate_bundle.property_policy_id",
        )
        property_policy_fingerprint = _sha256(
            record["property_policy_fingerprint"],
            "$static_candidate_bundle.property_policy_fingerprint",
        )
        authoring_policy_id = _identifier(
            record["authoring_policy_id"],
            "$static_candidate_bundle.authoring_policy_id",
        )
        authoring_policy_fingerprint = _sha256(
            record["authoring_policy_fingerprint"],
            "$static_candidate_bundle.authoring_policy_fingerprint",
        )
        authoring_brief_id = _identifier(
            record["authoring_brief_id"],
            "$static_candidate_bundle.authoring_brief_id",
        )
        authoring_brief_fingerprint = _sha256(
            record["authoring_brief_fingerprint"],
            "$static_candidate_bundle.authoring_brief_fingerprint",
        )
        audit_plan_fingerprint = _sha256(
            record["preauthoring_audit_plan_fingerprint"],
            "$static_candidate_bundle.preauthoring_audit_plan_fingerprint",
        )

        if (
            contract_record["task_id"] != task.task_id
            or contract_record["task_version"] != task.version
            or contract_record["task_spec_fingerprint"] != task_fingerprint
            or contract_record["capability_catalog_fingerprint"]
            != catalog_fingerprint
            or contract_record["compiler_policy_id"] != compiler_policy_id
            or contract_record["compiler_policy_fingerprint"]
            != compiler_policy_fingerprint
            or contract_record["adapter_id"] != task.environment.adapter_id
            or contract_record["event_schema_id"] != task.event_schema_id
            or contract_record["process_spec_id"] != task.process_spec_id
        ):
            raise ValueError("Static candidate contract binding mismatch")
        if (
            oracle_record["task_id"] != task.task_id
            or oracle_record["task_version"] != task.version
            or oracle_record["task_contract_id"]
            != contract_record["task_contract_id"]
            or oracle_record["capability_catalog_fingerprint"]
            != catalog_fingerprint
            or oracle_record["compiler_policy_id"] != compiler_policy_id
            or oracle_record["compiler_policy_fingerprint"]
            != compiler_policy_fingerprint
        ):
            raise ValueError("Static candidate oracle binding mismatch")
        if (
            report_record["report_schema_version"] != 1
            or report_record["task_id"] != task.task_id
            or report_record["task_spec_fingerprint"] != task_fingerprint
            or report_record["contract_fingerprint"] != contract_fingerprint
            or report_record["validation_policy_id"] != property_policy_id
            or report_record["validation_policy_fingerprint"]
            != property_policy_fingerprint
            or report_record["accepted"] is not True
            or report_record["issues"] != []
        ):
            raise ValueError("Static candidate property report binding mismatch")
        if (
            process_spec.process_spec_id != task.process_spec_id
            or process_spec.task_id != task.task_id
            or process_spec.event_schema_id != task.event_schema_id
            or process_contract_record["process_spec_id"]
            != process_spec.process_spec_id
            or process_contract_record["process_spec_fingerprint"]
            != process_spec_fingerprint
            or process_contract_record["process_compiler_policy_id"]
            != process_policy_id
            or process_contract_record["process_compiler_policy_fingerprint"]
            != process_policy_fingerprint
            or process_contract_record["task_contract_id"]
            != contract_record["task_contract_id"]
            or process_contract_record["task_contract_fingerprint"]
            != contract_fingerprint
            or process_contract_record["task_spec_fingerprint"]
            != task_fingerprint
            or process_contract_record["task_compiler_policy_fingerprint"]
            != compiler_policy_fingerprint
            or process_contract_record["capability_catalog_fingerprint"]
            != catalog_fingerprint
            or process_contract_record["event_schema_id"]
            != task.event_schema_id
        ):
            raise ValueError("Static candidate process contract binding mismatch")

        intent_fingerprint = _sha256(
            record["intent_fingerprint"],
            "$static_candidate_bundle.intent_fingerprint",
        )
        attempt_fingerprint = _sha256(
            record["attempt_fingerprint"],
            "$static_candidate_bundle.attempt_fingerprint",
        )
        response_sha256 = _sha256(
            record["raw_response_sha256"],
            "$static_candidate_bundle.raw_response_sha256",
        )
        response_bytes = _nonnegative_integer(
            record["raw_response_bytes"],
            "$static_candidate_bundle.raw_response_bytes",
        )

        if attempt is not None and (
            attempt.fingerprint != attempt_fingerprint
            or attempt.intent_fingerprint != intent_fingerprint
            or attempt.raw_response_sha256 != response_sha256
            or attempt.raw_response_bytes != response_bytes
            or attempt.status != "candidate"
            or attempt.issues
        ):
            raise ValueError("Static candidate attempt binding mismatch")
        if intent is not None:
            if brief is None or catalog is None or process_specs is None:
                raise ValueError(
                    "Brief, catalog, and ProcessSpecs are required to verify AuthoringIntent"
                )
            (
                canonical_intent,
                canonical_catalog,
                canonical_process_specs,
                canonical_authoring,
                canonical_brief,
                canonical_parent_task,
                canonical_parent_contract,
            ) = (
                _canonical_trusted_inputs(
                    intent=intent,
                    brief=brief,
                    parent_task=parent_task,
                    parent_contract=parent_contract,
                    catalog=catalog,
                    process_specs=process_specs,
                    compiler_policy=compiler_policy,
                    process_policy=process_policy,
                    property_policy=property_policy,
                    authoring_policy=authoring_policy,
                )
            )
            if (
                canonical_intent.fingerprint != intent_fingerprint
                or task.task_id != canonical_intent.task_id
                or task.version != canonical_intent.task_version
                or task.family_id != canonical_intent.family_id
                or task.instructions.audit != canonical_intent.audit_commitment
                or task.environment.adapter_id
                not in canonical_intent.allowed_adapter_ids
                or task.event_schema_id
                not in canonical_intent.allowed_event_schema_ids
                or task.process_spec_id
                not in canonical_intent.allowed_process_spec_ids
                or canonical_catalog.fingerprint != catalog_fingerprint
                or canonical_process_specs[task.process_spec_id].fingerprint
                != process_spec_fingerprint
                or canonical_authoring.policy_id != authoring_policy_id
                or canonical_authoring.fingerprint
                != authoring_policy_fingerprint
                or canonical_intent.authoring_brief_id != authoring_brief_id
                or canonical_intent.authoring_brief_fingerprint
                != authoring_brief_fingerprint
                or canonical_intent.preauthoring_audit_plan_fingerprint
                != audit_plan_fingerprint
            ):
                raise ValueError("Static candidate intent binding mismatch")
            try:
                validate_candidate_against_brief(
                    task,
                    brief=canonical_brief,
                    parent_task=canonical_parent_task,
                    parent_contract=canonical_parent_contract,
                )
            except CandidateBriefViolation as error:
                raise ValueError("Static candidate brief binding mismatch") from error
        elif catalog is not None and catalog.fingerprint != catalog_fingerprint:
            raise ValueError("Static candidate capability catalog mismatch")

        if catalog is not None:
            if process_specs is None:
                raise ValueError("ProcessSpecs are required for semantic revalidation")
            try:
                rebuilt_contract = compile_task_contract(
                    task,
                    catalog=catalog,
                    policy=compiler_policy,
                )
                rebuilt_report = validate_task_properties(
                    task,
                    rebuilt_contract,
                    catalog=catalog,
                    policy=property_policy,
                    compiler_policy=compiler_policy,
                )
                rebuilt_process_spec = ProcessSpecV1.from_record(
                    process_specs[task.process_spec_id].to_record()
                )
                rebuilt_process_contract = compile_process_contract(
                    rebuilt_process_spec,
                    task=task,
                    task_contract=rebuilt_contract,
                    catalog=catalog,
                    process_policy=process_policy,
                    task_policy=compiler_policy,
                )
            except Exception:
                raise ValueError("Static candidate semantic revalidation failed") from None
            if (
                rebuilt_contract.to_record() != contract_record
                or rebuilt_contract.oracle_record() != oracle_record
                or rebuilt_report.to_record() != report_record
                or rebuilt_process_spec.to_record() != process_spec_record
                or rebuilt_process_contract.to_record()
                != process_contract_record
                or not rebuilt_report.accepted
            ):
                raise ValueError("Static candidate rebuilt artifacts mismatch")

        return cls(
            schema_version=1,
            candidate_state=_CANDIDATE_STATE,
            admission_status=_ADMISSION_STATUS,
            intent_fingerprint=intent_fingerprint,
            attempt_fingerprint=attempt_fingerprint,
            raw_response_sha256=response_sha256,
            raw_response_bytes=response_bytes,
            task_spec_fingerprint=task_fingerprint,
            task_spec=freeze_json(task.to_record()),
            task_contract_fingerprint=contract_fingerprint,
            task_contract=freeze_json(contract_record),
            task_oracle_sha256=oracle_sha256,
            task_oracle=freeze_json(oracle_record),
            property_report_fingerprint=report_fingerprint,
            property_report=freeze_json(report_record),
            process_spec_fingerprint=process_spec_fingerprint,
            process_spec=freeze_json(process_spec.to_record()),
            process_contract_fingerprint=process_contract_fingerprint,
            process_contract=freeze_json(process_contract_record),
            capability_catalog_fingerprint=catalog_fingerprint,
            compiler_policy_id=compiler_policy_id,
            compiler_policy_fingerprint=compiler_policy_fingerprint,
            process_compiler_policy_id=process_policy_id,
            process_compiler_policy_fingerprint=process_policy_fingerprint,
            property_policy_id=property_policy_id,
            property_policy_fingerprint=property_policy_fingerprint,
            authoring_policy_id=authoring_policy_id,
            authoring_policy_fingerprint=authoring_policy_fingerprint,
            authoring_brief_id=authoring_brief_id,
            authoring_brief_fingerprint=authoring_brief_fingerprint,
            preauthoring_audit_plan_fingerprint=audit_plan_fingerprint,
        )

    @classmethod
    def from_json(
        cls,
        text: str,
        **kwargs: Any,
    ) -> StaticCandidateBundle:
        return cls.from_record(strict_json_loads(text), **kwargs)

    def to_record(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "candidate_state": self.candidate_state,
            "admission_status": self.admission_status,
            "intent_fingerprint": self.intent_fingerprint,
            "attempt_fingerprint": self.attempt_fingerprint,
            "raw_response_sha256": self.raw_response_sha256,
            "raw_response_bytes": self.raw_response_bytes,
            "task_spec_fingerprint": self.task_spec_fingerprint,
            "task_spec": thaw_json(self.task_spec),
            "task_contract_fingerprint": self.task_contract_fingerprint,
            "task_contract": thaw_json(self.task_contract),
            "task_oracle_sha256": self.task_oracle_sha256,
            "task_oracle": thaw_json(self.task_oracle),
            "property_report_fingerprint": self.property_report_fingerprint,
            "property_report": thaw_json(self.property_report),
            "process_spec_fingerprint": self.process_spec_fingerprint,
            "process_spec": thaw_json(self.process_spec),
            "process_contract_fingerprint": self.process_contract_fingerprint,
            "process_contract": thaw_json(self.process_contract),
            "capability_catalog_fingerprint": (
                self.capability_catalog_fingerprint
            ),
            "compiler_policy_id": self.compiler_policy_id,
            "compiler_policy_fingerprint": self.compiler_policy_fingerprint,
            "process_compiler_policy_id": self.process_compiler_policy_id,
            "process_compiler_policy_fingerprint": (
                self.process_compiler_policy_fingerprint
            ),
            "property_policy_id": self.property_policy_id,
            "property_policy_fingerprint": self.property_policy_fingerprint,
            "authoring_policy_id": self.authoring_policy_id,
            "authoring_policy_fingerprint": self.authoring_policy_fingerprint,
            "authoring_brief_id": self.authoring_brief_id,
            "authoring_brief_fingerprint": self.authoring_brief_fingerprint,
            "preauthoring_audit_plan_fingerprint": (
                self.preauthoring_audit_plan_fingerprint
            ),
        }

    def to_json(self) -> str:
        return canonical_json(self.to_record())

    @property
    def fingerprint(self) -> str:
        return canonical_fingerprint(self.to_record())
