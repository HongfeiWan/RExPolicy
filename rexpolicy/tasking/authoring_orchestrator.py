"""Deterministic orchestration for bounded automatic TaskSpec authoring.

The proposer is untrusted.  It receives one public, hash-bound authoring brief
and policy-owned repair messages.  Every response is quarantined; even a fully
valid static candidate is never admitted or activated by this module.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .authoring import (
    DEFAULT_AUTHORING_POLICY,
    AuthoringIntent,
    AuthoringPolicy,
    ProposalAttempt,
    PublicIssue,
    normalize_public_issues,
)
from .authoring_candidate import (
    StaticCandidateBundle,
    validate_static_candidate,
)
from .authoring_brief import (
    PublicAuthoringBrief,
    PublicCurriculumGap,
)
from .authoring_quarantine import (
    AuthoringQuarantineWrite,
    write_authoring_quarantine,
)
from .authoring_runner import (
    DEFAULT_PROPOSER_EXECUTION_POLICY,
    ProposerCommand,
    ProposerExecutionPolicy,
    ProposerInvocation,
    run_proposer,
)
from .canonical import canonical_fingerprint
from .capabilities import CapabilityCatalog
from .contract import (
    DEFAULT_TASK_COMPILER_POLICY,
    CompiledTaskContract,
    TaskCompilerPolicy,
)
from .model import TaskSpecV2
from .property_validation import (
    DEFAULT_PROPERTY_VALIDATION_POLICY,
    PropertyValidationPolicy,
)
from .process import ProcessSpecV1
from .process_contract import (
    DEFAULT_PROCESS_COMPILER_POLICY,
    ProcessCompilerPolicy,
)

_IDENTIFIER = re.compile(r"^[a-z][a-z0-9_.-]*(?:/[a-z0-9_.-]+)*$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_FINAL_STATES = frozenset(("exhausted", "quarantined_static"))


def _identifier(value: Any, path: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise ValueError(f"{path} must be a versioned identifier")
    return value


def _sha256(value: Any, path: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{path} must be a lowercase SHA-256")
    return value


def _public_text(value: Any, path: str, *, maximum: int) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{path} must be non-empty public text")
    if value != value.strip() or len(value) > maximum:
        raise ValueError(f"{path} must be trimmed and at most {maximum} characters")
    if unicodedata.normalize("NFC", value) != value:
        raise ValueError(f"{path} must use NFC Unicode normalization")
    if any(unicodedata.category(character).startswith("C") for character in value):
        raise ValueError(f"{path} cannot contain control characters")
    return value




@dataclass(frozen=True)
class AuthoringPromptTemplate:
    schema_version: int
    template_id: str
    instructions: tuple[str, ...]

    def to_record(self) -> dict[str, Any]:
        if self.schema_version != 1:
            raise ValueError("Authoring prompt template schema_version must be 1")
        _identifier(self.template_id, "$authoring_template.template_id")
        if not self.instructions:
            raise ValueError("Authoring prompt template instructions cannot be empty")
        for index, instruction in enumerate(self.instructions):
            _public_text(
                instruction,
                f"$authoring_template.instructions[{index}]",
                maximum=512,
            )
        return {
            "schema_version": self.schema_version,
            "template_id": self.template_id,
            "instructions": list(self.instructions),
        }

    @property
    def fingerprint(self) -> str:
        return canonical_fingerprint(self.to_record())


DEFAULT_AUTHORING_TEMPLATE = AuthoringPromptTemplate(
    schema_version=1,
    template_id="rexpolicy/task_authoring_prompt/v1",
    instructions=(
        "Return exactly one TaskSpec v2 JSON object and no surrounding text.",
        "Use only capabilities and identifiers allowed by the authoring intent.",
        "Keep the task identity and sealed audit commitment exactly unchanged.",
        "Treat goal, failure, and safety predicates as the task oracle.",
        "Treat reward profiles as non-authoritative rebuildable scoring views.",
        "Never invent, request, or reveal sealed audit content or results.",
    ),
)


def _validated_process_specs(
    *,
    intent: AuthoringIntent,
    process_specs: Mapping[str, ProcessSpecV1],
    process_policy: ProcessCompilerPolicy,
) -> tuple[ProcessSpecV1, ...]:
    if intent.schema_version != 2:
        raise ValueError("Automatic authoring requires AuthoringIntent v2")
    if (
        intent.process_compiler_policy_id != process_policy.policy_id
        or intent.process_compiler_policy_fingerprint != process_policy.fingerprint
    ):
        raise ValueError("Authoring intent process compiler policy binding drifted")
    if set(process_specs) != set(intent.allowed_process_spec_ids):
        raise ValueError("Authoring process registry does not match the intent")
    expected_fingerprints = dict(intent.process_spec_fingerprints)
    canonical_specs = []
    for process_spec_id in intent.allowed_process_spec_ids:
        spec = ProcessSpecV1.from_record(process_specs[process_spec_id].to_record())
        if spec.process_spec_id != process_spec_id:
            raise ValueError("Authoring process registry identity mismatch")
        if expected_fingerprints.get(process_spec_id) != spec.fingerprint:
            raise ValueError("Authoring process registry fingerprint drifted")
        canonical_specs.append(spec)
    return tuple(canonical_specs)


@dataclass(frozen=True)
class ProposerBinding:
    schema_version: int
    proposer_id: str
    model_id: str
    command_fingerprint: str
    execution_policy_id: str
    execution_policy_fingerprint: str
    template_id: str
    template_fingerprint: str

    def to_record(self) -> dict[str, Any]:
        if self.schema_version != 1:
            raise ValueError("ProposerBinding schema_version must be 1")
        for path, value in (
            ("proposer_id", self.proposer_id),
            ("model_id", self.model_id),
            ("execution_policy_id", self.execution_policy_id),
            ("template_id", self.template_id),
        ):
            _identifier(value, f"$proposer_binding.{path}")
        for path, value in (
            ("command_fingerprint", self.command_fingerprint),
            ("execution_policy_fingerprint", self.execution_policy_fingerprint),
            ("template_fingerprint", self.template_fingerprint),
        ):
            _sha256(value, f"$proposer_binding.{path}")
        return {
            "schema_version": self.schema_version,
            "proposer_id": self.proposer_id,
            "model_id": self.model_id,
            "command_fingerprint": self.command_fingerprint,
            "execution_policy_id": self.execution_policy_id,
            "execution_policy_fingerprint": self.execution_policy_fingerprint,
            "template_id": self.template_id,
            "template_fingerprint": self.template_fingerprint,
        }

    @property
    def fingerprint(self) -> str:
        return canonical_fingerprint(self.to_record())


@dataclass(frozen=True)
class AuthoringSessionContract:
    schema_version: int
    intent_fingerprint: str
    brief_fingerprint: str
    proposer_binding_fingerprint: str
    capability_catalog_fingerprint: str
    process_registry_fingerprint: str
    compiler_policy_fingerprint: str
    process_compiler_policy_fingerprint: str
    property_policy_fingerprint: str
    authoring_policy_fingerprint: str
    preauthoring_audit_plan_fingerprint: str

    def to_record(self) -> dict[str, Any]:
        if self.schema_version != 1:
            raise ValueError("AuthoringSessionContract schema_version must be 1")
        record = {
            "schema_version": self.schema_version,
            "intent_fingerprint": self.intent_fingerprint,
            "brief_fingerprint": self.brief_fingerprint,
            "proposer_binding_fingerprint": self.proposer_binding_fingerprint,
            "capability_catalog_fingerprint": self.capability_catalog_fingerprint,
            "process_registry_fingerprint": self.process_registry_fingerprint,
            "compiler_policy_fingerprint": self.compiler_policy_fingerprint,
            "process_compiler_policy_fingerprint": (
                self.process_compiler_policy_fingerprint
            ),
            "property_policy_fingerprint": self.property_policy_fingerprint,
            "authoring_policy_fingerprint": self.authoring_policy_fingerprint,
            "preauthoring_audit_plan_fingerprint": (
                self.preauthoring_audit_plan_fingerprint
            ),
        }
        for name, value in record.items():
            if name != "schema_version":
                _sha256(value, f"$authoring_session.{name}")
        return record

    @property
    def fingerprint(self) -> str:
        return canonical_fingerprint(self.to_record())


def build_authoring_session(
    *,
    intent: AuthoringIntent,
    brief: PublicAuthoringBrief,
    catalog: CapabilityCatalog,
    process_specs: Mapping[str, ProcessSpecV1],
    command: ProposerCommand,
    model_id: str,
    template: AuthoringPromptTemplate,
    compiler_policy: TaskCompilerPolicy,
    property_policy: PropertyValidationPolicy,
    process_policy: ProcessCompilerPolicy,
    authoring_policy: AuthoringPolicy,
    execution_policy: ProposerExecutionPolicy,
) -> tuple[ProposerBinding, AuthoringSessionContract]:
    brief.validate_against(intent)
    canonical_command = command.canonical()
    execution_policy.validate(authoring_policy=authoring_policy)
    canonical_specs = _validated_process_specs(
        intent=intent,
        process_specs=process_specs,
        process_policy=process_policy,
    )
    binding = ProposerBinding(
        schema_version=1,
        proposer_id=canonical_command.command_id,
        model_id=_identifier(model_id, "$authoring_job.model_id"),
        command_fingerprint=canonical_command.fingerprint,
        execution_policy_id=execution_policy.policy_id,
        execution_policy_fingerprint=execution_policy.fingerprint,
        template_id=template.template_id,
        template_fingerprint=template.fingerprint,
    )
    process_registry_fingerprint = canonical_fingerprint(
        {
            "process_specs": [
                {
                    "process_spec_id": process_spec.process_spec_id,
                    "process_spec_fingerprint": process_spec.fingerprint,
                }
                for process_spec in canonical_specs
            ]
        }
    )
    if intent.preauthoring_audit_plan_fingerprint is None:
        raise ValueError("Authoring intent has no pre-authoring audit plan")
    session = AuthoringSessionContract(
        schema_version=1,
        intent_fingerprint=intent.fingerprint,
        brief_fingerprint=brief.fingerprint,
        proposer_binding_fingerprint=binding.fingerprint,
        capability_catalog_fingerprint=catalog.fingerprint,
        process_registry_fingerprint=process_registry_fingerprint,
        compiler_policy_fingerprint=compiler_policy.fingerprint,
        process_compiler_policy_fingerprint=process_policy.fingerprint,
        property_policy_fingerprint=property_policy.fingerprint,
        authoring_policy_fingerprint=authoring_policy.fingerprint,
        preauthoring_audit_plan_fingerprint=(
            intent.preauthoring_audit_plan_fingerprint
        ),
    )
    canonical_fingerprint(session.to_record())
    return binding, session


def build_authoring_request(
    *,
    intent: AuthoringIntent,
    brief: PublicAuthoringBrief,
    catalog: CapabilityCatalog,
    process_specs: Mapping[str, ProcessSpecV1],
    parent_task: TaskSpecV2 | None,
    parent_contract: CompiledTaskContract | None,
    session: AuthoringSessionContract,
    ordinal: int,
    parent: ProposalAttempt | None,
    template: AuthoringPromptTemplate = DEFAULT_AUTHORING_TEMPLATE,
    compiler_policy: TaskCompilerPolicy = DEFAULT_TASK_COMPILER_POLICY,
    property_policy: PropertyValidationPolicy = DEFAULT_PROPERTY_VALIDATION_POLICY,
    process_policy: ProcessCompilerPolicy = DEFAULT_PROCESS_COMPILER_POLICY,
    authoring_policy: AuthoringPolicy = DEFAULT_AUTHORING_POLICY,
) -> dict[str, Any]:
    """Build the complete public proposer request for one attempt."""
    brief.validate_against(intent)
    if type(ordinal) is not int or not 1 <= ordinal <= authoring_policy.max_attempts:
        raise ValueError("Authoring request ordinal is outside the policy limit")
    if (ordinal == 1) != (parent is None):
        raise ValueError("First authoring request cannot have a parent")
    if parent is not None:
        if parent.intent_fingerprint != intent.fingerprint:
            raise ValueError("Authoring request parent belongs to another intent")
        if parent.ordinal + 1 != ordinal or parent.status != "repair_requested":
            raise ValueError("Authoring request parent lineage is invalid")
    if intent.capability_catalog_fingerprint != catalog.fingerprint:
        raise ValueError("Authoring intent capability catalog binding drifted")
    if (
        intent.compiler_policy_id != compiler_policy.policy_id
        or intent.compiler_policy_fingerprint != compiler_policy.fingerprint
    ):
        raise ValueError("Authoring intent compiler policy binding drifted")
    if (
        intent.property_policy_id != property_policy.policy_id
        or intent.property_policy_fingerprint != property_policy.fingerprint
    ):
        raise ValueError("Authoring intent property policy binding drifted")
    if (
        intent.authoring_policy_id != authoring_policy.policy_id
        or intent.authoring_policy_fingerprint != authoring_policy.fingerprint
    ):
        raise ValueError("Authoring intent authoring policy binding drifted")
    canonical_process_specs = _validated_process_specs(
        intent=intent,
        process_specs=process_specs,
        process_policy=process_policy,
    )
    expects_parent = brief.parent_task_spec_fingerprint is not None
    if expects_parent != (parent_task is not None and parent_contract is not None):
        raise ValueError("Authoring parent artifacts are incomplete")
    if expects_parent:
        assert parent_task is not None and parent_contract is not None
        if (
            parent_task.fingerprint != brief.parent_task_spec_fingerprint
            or parent_contract.task_spec_fingerprint != parent_task.fingerprint
            or parent_contract.oracle_fingerprint
            != brief.parent_task_oracle_fingerprint
        ):
            raise ValueError("Authoring parent artifact binding drifted")
    process_registry_fingerprint = canonical_fingerprint(
        {
            "process_specs": [
                {
                    "process_spec_id": process_spec.process_spec_id,
                    "process_spec_fingerprint": process_spec.fingerprint,
                }
                for process_spec in canonical_process_specs
            ]
        }
    )
    if (
        session.intent_fingerprint != intent.fingerprint
        or session.brief_fingerprint != brief.fingerprint
        or session.capability_catalog_fingerprint != catalog.fingerprint
        or session.process_registry_fingerprint != process_registry_fingerprint
        or session.compiler_policy_fingerprint != compiler_policy.fingerprint
        or session.process_compiler_policy_fingerprint != process_policy.fingerprint
        or session.property_policy_fingerprint != property_policy.fingerprint
        or session.authoring_policy_fingerprint != authoring_policy.fingerprint
        or session.preauthoring_audit_plan_fingerprint
        != intent.preauthoring_audit_plan_fingerprint
    ):
        raise ValueError("Authoring session dependency binding drifted")
    previous_issues = () if parent is None else parent.issues
    return {
        "schema_version": 1,
        "protocol_id": "rexpolicy/task_authoring_request/v1",
        "session": session.to_record(),
        "attempt": {
            "ordinal": ordinal,
            "parent_attempt_fingerprint": (
                None if parent is None else parent.fingerprint
            ),
            "public_issues": [issue.to_record() for issue in previous_issues],
        },
        "intent": intent.to_record(),
        "public_brief": brief.to_record(),
        "parent_task": None if parent_task is None else parent_task.to_record(),
        "template": template.to_record(),
        "output_contract": {
            "media_type": "application/json",
            "single_object": True,
            "task_spec_schema_version": 2,
            "unknown_fields": "forbidden",
            "max_response_bytes": authoring_policy.max_response_bytes,
            "candidate_state_after_static_acceptance": "quarantined_static",
            "automatic_admission": False,
            "automatic_activation": False,
        },
        "capability_catalog": catalog.to_record(),
        "allowed_process_specs": [
            process_spec.to_record() for process_spec in canonical_process_specs
        ],
        "trusted_policy_bindings": {
            "compiler_policy_id": compiler_policy.policy_id,
            "compiler_policy_fingerprint": compiler_policy.fingerprint,
            "property_policy_id": property_policy.policy_id,
            "property_policy_fingerprint": property_policy.fingerprint,
            "process_compiler_policy_id": process_policy.policy_id,
            "process_compiler_policy_fingerprint": process_policy.fingerprint,
            "authoring_policy_id": authoring_policy.policy_id,
            "authoring_policy_fingerprint": authoring_policy.fingerprint,
        },
    }


def _invocation_issues(status: str) -> tuple[PublicIssue, ...] | None:
    if status == "completed":
        return None
    if status == "timeout":
        pairs = (("proposer_timeout", "$"),)
    elif status == "stdout_limit":
        pairs = (("response_too_large", "$"),)
    else:
        pairs = (("proposer_failed", "$"),)
    return normalize_public_issues(pairs)


def validate_authoring_attempt_chain(
    *,
    intent: AuthoringIntent,
    session: AuthoringSessionContract,
    proposer_binding: ProposerBinding,
    attempts: tuple[ProposalAttempt, ...],
    invocations: tuple[ProposerInvocation, ...],
    authoring_policy: AuthoringPolicy = DEFAULT_AUTHORING_POLICY,
) -> None:
    """Validate one complete, single-child authoring retry chain."""
    if not attempts or len(attempts) != len(invocations):
        raise ValueError("Authoring attempt chain is incomplete")
    if len(attempts) > authoring_policy.max_attempts:
        raise ValueError("Authoring attempt chain exceeds the policy limit")
    if (
        session.intent_fingerprint != intent.fingerprint
        or session.proposer_binding_fingerprint != proposer_binding.fingerprint
    ):
        raise ValueError("Authoring attempt chain session binding drifted")
    fingerprints: set[str] = set()
    previous: ProposalAttempt | None = None
    for index, (attempt, invocation) in enumerate(
        zip(attempts, invocations),
        start=1,
    ):
        canonical_attempt = ProposalAttempt.from_record(
            attempt.to_record(),
            policy=authoring_policy,
        )
        if canonical_attempt.schema_version != 2:
            raise ValueError("Automatic authoring attempts must use schema v2")
        if (
            canonical_attempt.intent_fingerprint != intent.fingerprint
            or canonical_attempt.session_fingerprint != session.fingerprint
            or canonical_attempt.ordinal != index
            or canonical_attempt.parent_attempt_fingerprint
            != (None if previous is None else previous.fingerprint)
            or canonical_attempt.proposer_id != proposer_binding.proposer_id
            or canonical_attempt.model_id != proposer_binding.model_id
            or canonical_attempt.template_fingerprint
            != proposer_binding.template_fingerprint
            or canonical_attempt.invocation_fingerprint != invocation.fingerprint
            or canonical_attempt.request_fingerprint != invocation.request_sha256
            or canonical_attempt.raw_response_sha256 != invocation.stdout_sha256
            or canonical_attempt.raw_response_bytes != invocation.stdout_bytes
            or canonical_attempt.response_complete == invocation.stdout_truncated
            or invocation.command_id != proposer_binding.proposer_id
            or invocation.command_sha256 != proposer_binding.command_fingerprint
            or invocation.execution_policy_id
            != proposer_binding.execution_policy_id
            or invocation.execution_policy_sha256
            != proposer_binding.execution_policy_fingerprint
        ):
            raise ValueError("Authoring attempt chain provenance drifted")
        if previous is not None and previous.status != "repair_requested":
            raise ValueError("Only repair-requested attempts may have a child")
        if canonical_attempt.fingerprint in fingerprints:
            raise ValueError("Authoring attempt chain repeats an attempt")
        fingerprints.add(canonical_attempt.fingerprint)
        previous = canonical_attempt
    if attempts[-1].status not in {"candidate", "rejected"}:
        raise ValueError("Authoring attempt chain has no terminal decision")


@dataclass(frozen=True)
class AuthoringJobResult:
    schema_version: int
    intent_fingerprint: str
    brief_fingerprint: str
    command_fingerprint: str
    execution_policy_fingerprint: str
    session_fingerprint: str
    attempt_fingerprints: tuple[str, ...]
    quarantine_descriptor_sha256s: tuple[str, ...]
    final_state: str
    static_candidate_bundle_fingerprint: str | None

    def to_record(self) -> dict[str, Any]:
        if self.schema_version != 1:
            raise ValueError("AuthoringJobResult schema_version must be 1")
        for name, value in (
            ("intent", self.intent_fingerprint),
            ("brief", self.brief_fingerprint),
            ("command", self.command_fingerprint),
            ("execution_policy", self.execution_policy_fingerprint),
            ("session", self.session_fingerprint),
        ):
            _sha256(value, f"$authoring_job_result.{name}_fingerprint")
        if not self.attempt_fingerprints:
            raise ValueError("AuthoringJobResult must contain at least one attempt")
        for index, value in enumerate(self.attempt_fingerprints):
            _sha256(value, f"$authoring_job_result.attempt_fingerprints[{index}]")
        for index, value in enumerate(self.quarantine_descriptor_sha256s):
            _sha256(
                value,
                f"$authoring_job_result.quarantine_descriptor_sha256s[{index}]",
            )
        if len(self.attempt_fingerprints) != len(
            self.quarantine_descriptor_sha256s
        ):
            raise ValueError("Every authoring attempt must have one quarantine entry")
        if self.final_state not in _FINAL_STATES:
            raise ValueError("AuthoringJobResult final state is invalid")
        has_bundle = self.static_candidate_bundle_fingerprint is not None
        if has_bundle != (self.final_state == "quarantined_static"):
            raise ValueError("AuthoringJobResult candidate state is inconsistent")
        if has_bundle:
            _sha256(
                self.static_candidate_bundle_fingerprint,
                "$authoring_job_result.static_candidate_bundle_fingerprint",
            )
        return {
            "schema_version": self.schema_version,
            "intent_fingerprint": self.intent_fingerprint,
            "brief_fingerprint": self.brief_fingerprint,
            "command_fingerprint": self.command_fingerprint,
            "execution_policy_fingerprint": self.execution_policy_fingerprint,
            "session_fingerprint": self.session_fingerprint,
            "attempt_fingerprints": list(self.attempt_fingerprints),
            "quarantine_descriptor_sha256s": list(
                self.quarantine_descriptor_sha256s
            ),
            "final_state": self.final_state,
            "static_candidate_bundle_fingerprint": (
                self.static_candidate_bundle_fingerprint
            ),
        }

    @property
    def fingerprint(self) -> str:
        return canonical_fingerprint(self.to_record())


@dataclass(frozen=True)
class AuthoringJobRun:
    result: AuthoringJobResult
    attempts: tuple[ProposalAttempt, ...]
    invocations: tuple[ProposerInvocation, ...]
    quarantine_writes: tuple[AuthoringQuarantineWrite, ...]
    static_candidate_bundle: StaticCandidateBundle | None


def run_automatic_authoring(
    *,
    intent: AuthoringIntent,
    brief: PublicAuthoringBrief,
    catalog: CapabilityCatalog,
    process_specs: Mapping[str, ProcessSpecV1],
    parent_task: TaskSpecV2 | None,
    parent_contract: CompiledTaskContract | None,
    command: ProposerCommand,
    model_id: str,
    quarantine_dir: Path,
    environment: Mapping[str, str] | None = None,
    template: AuthoringPromptTemplate = DEFAULT_AUTHORING_TEMPLATE,
    compiler_policy: TaskCompilerPolicy = DEFAULT_TASK_COMPILER_POLICY,
    property_policy: PropertyValidationPolicy = DEFAULT_PROPERTY_VALIDATION_POLICY,
    process_policy: ProcessCompilerPolicy = DEFAULT_PROCESS_COMPILER_POLICY,
    authoring_policy: AuthoringPolicy = DEFAULT_AUTHORING_POLICY,
    execution_policy: ProposerExecutionPolicy = DEFAULT_PROPOSER_EXECUTION_POLICY,
) -> AuthoringJobRun:
    """Run bounded propose/repair attempts and stop at static quarantine."""
    proposer_binding, session = build_authoring_session(
        intent=intent,
        brief=brief,
        catalog=catalog,
        process_specs=process_specs,
        command=command,
        model_id=model_id,
        template=template,
        compiler_policy=compiler_policy,
        property_policy=property_policy,
        process_policy=process_policy,
        authoring_policy=authoring_policy,
        execution_policy=execution_policy,
    )
    attempts: list[ProposalAttempt] = []
    invocations: list[ProposerInvocation] = []
    writes: list[AuthoringQuarantineWrite] = []
    parent: ProposalAttempt | None = None

    for ordinal in range(1, authoring_policy.max_attempts + 1):
        request = build_authoring_request(
            intent=intent,
            brief=brief,
            catalog=catalog,
            process_specs=process_specs,
            parent_task=parent_task,
            parent_contract=parent_contract,
            session=session,
            ordinal=ordinal,
            parent=parent,
            template=template,
            compiler_policy=compiler_policy,
            property_policy=property_policy,
            process_policy=process_policy,
            authoring_policy=authoring_policy,
        )
        run = run_proposer(
            command=command,
            request=request,
            environment=environment,
            execution_policy=execution_policy,
            authoring_policy=authoring_policy,
        )
        request_fingerprint = canonical_fingerprint(request)
        if run.invocation.request_sha256 != request_fingerprint:
            raise ValueError("Proposer invocation request binding mismatch")

        invocation_issues = _invocation_issues(run.invocation.status)
        assessment = None
        if invocation_issues is None:
            assessment = validate_static_candidate(
                run.stdout,
                intent=intent,
                brief=brief,
                parent_task=parent_task,
                parent_contract=parent_contract,
                catalog=catalog,
                process_specs=process_specs,
                compiler_policy=compiler_policy,
                property_policy=property_policy,
                process_policy=process_policy,
                authoring_policy=authoring_policy,
            )
            public_issues = assessment.issues
        else:
            public_issues = invocation_issues

        accepted = assessment is not None and assessment.accepted
        if accepted:
            status = "candidate"
        elif ordinal < authoring_policy.max_attempts:
            status = "repair_requested"
        else:
            status = "rejected"
        attempt = ProposalAttempt.record_response(
            run.stdout,
            intent=intent,
            proposer_id=command.command_id,
            model_id=model_id,
            template_fingerprint=template.fingerprint,
            request_fingerprint=request_fingerprint,
            status=status,
            issues=((issue.code, issue.path) for issue in public_issues),
            parent=parent,
            session_fingerprint=session.fingerprint,
            invocation_fingerprint=run.invocation.fingerprint,
            response_complete=not run.invocation.stdout_truncated,
            policy=authoring_policy,
        )
        bundle = assessment.to_bundle(attempt) if accepted else None
        write = write_authoring_quarantine(
            run_dir=Path(quarantine_dir),
            raw_response=run.stdout,
            attempt=attempt,
            invocation=run.invocation,
            static_candidate_bundle=bundle,
            policy=authoring_policy,
        )
        attempts.append(attempt)
        invocations.append(run.invocation)
        writes.append(write)

        if bundle is not None:
            result = AuthoringJobResult(
                schema_version=1,
                intent_fingerprint=intent.fingerprint,
                brief_fingerprint=brief.fingerprint,
                command_fingerprint=command.fingerprint,
                execution_policy_fingerprint=execution_policy.fingerprint,
                session_fingerprint=session.fingerprint,
                attempt_fingerprints=tuple(item.fingerprint for item in attempts),
                quarantine_descriptor_sha256s=tuple(
                    item.descriptor_sha256 for item in writes
                ),
                final_state="quarantined_static",
                static_candidate_bundle_fingerprint=bundle.fingerprint,
            )
            validate_authoring_attempt_chain(
                intent=intent,
                session=session,
                proposer_binding=proposer_binding,
                attempts=tuple(attempts),
                invocations=tuple(invocations),
                authoring_policy=authoring_policy,
            )
            canonical_fingerprint(result.to_record())
            return AuthoringJobRun(
                result,
                tuple(attempts),
                tuple(invocations),
                tuple(writes),
                bundle,
            )
        if status == "rejected":
            result = AuthoringJobResult(
                schema_version=1,
                intent_fingerprint=intent.fingerprint,
                brief_fingerprint=brief.fingerprint,
                command_fingerprint=command.fingerprint,
                execution_policy_fingerprint=execution_policy.fingerprint,
                session_fingerprint=session.fingerprint,
                attempt_fingerprints=tuple(item.fingerprint for item in attempts),
                quarantine_descriptor_sha256s=tuple(
                    item.descriptor_sha256 for item in writes
                ),
                final_state="exhausted",
                static_candidate_bundle_fingerprint=None,
            )
            validate_authoring_attempt_chain(
                intent=intent,
                session=session,
                proposer_binding=proposer_binding,
                attempts=tuple(attempts),
                invocations=tuple(invocations),
                authoring_policy=authoring_policy,
            )
            canonical_fingerprint(result.to_record())
            return AuthoringJobRun(
                result,
                tuple(attempts),
                tuple(invocations),
                tuple(writes),
                None,
            )
        parent = attempt

    raise AssertionError("Authoring policy attempt loop did not terminate")
