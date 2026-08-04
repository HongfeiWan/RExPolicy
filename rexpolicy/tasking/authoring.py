"""Trusted immutable records for bounded automatic TaskSpec authoring.

This module deliberately contains no model or subprocess integration.  It
defines the policy-owned envelope around a future proposer: a fixed intent,
public repair-safe issues, and hash-only attempt provenance.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Iterable

from .canonical import canonical_fingerprint, canonical_json, strict_json_loads
from .model import AuditCommitment

_IDENTIFIER = re.compile(r"^[a-z][a-z0-9_.-]*(?:/[a-z0-9_.-]+)*$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_PUBLIC_PATH = re.compile(r"^\$(?:\.[a-z][a-z0-9_]*)*$")
_MAX_POLICY_ATTEMPTS = 8
_MAX_POLICY_RESPONSE_BYTES = 1_000_000
_MAX_PUBLIC_MESSAGE_CHARS = 160
_ATTEMPT_STATUSES = frozenset({"candidate", "repair_requested", "rejected"})


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


def _sha256(value: Any, path: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{path} must be a lowercase SHA-256")
    return value


def _positive_integer(value: Any, path: str, *, maximum: int | None = None) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{path} must be a positive integer")
    if maximum is not None and value > maximum:
        raise ValueError(f"{path} exceeds the trusted maximum {maximum}")
    return value


def _nonnegative_integer(value: Any, path: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{path} must be a non-negative integer")
    return value


def _public_path(value: Any, path: str) -> str:
    if not isinstance(value, str) or _PUBLIC_PATH.fullmatch(value) is None:
        raise ValueError(f"{path} must be a normalized public JSON path")
    return value


def _public_message(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{path} must be non-empty text")
    if value != value.strip() or len(value) > _MAX_PUBLIC_MESSAGE_CHARS:
        raise ValueError(
            f"{path} must be trimmed and at most "
            f"{_MAX_PUBLIC_MESSAGE_CHARS} characters"
        )
    if unicodedata.normalize("NFC", value) != value:
        raise ValueError(f"{path} must use NFC Unicode normalization")
    if any(unicodedata.category(character).startswith("C") for character in value):
        raise ValueError(f"{path} cannot contain control characters")
    return value


def _sorted_identifiers(value: Any, path: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{path} must be a non-empty array")
    output = tuple(
        _identifier(item, f"{path}[{index}]")
        for index, item in enumerate(value)
    )
    if output != tuple(sorted(set(output))):
        raise ValueError(f"{path} must be sorted and unique")
    return output


@dataclass(frozen=True)
class PublicIssueRule:
    """One policy-owned public repair message and its permitted targets."""

    code: str
    allowed_paths: tuple[str, ...]
    message: str

    @classmethod
    def from_record(cls, value: Any, path: str) -> PublicIssueRule:
        record = _mapping(value, path)
        _keys(record, {"code", "allowed_paths", "message"}, path)
        raw_paths = record["allowed_paths"]
        if not isinstance(raw_paths, list) or not raw_paths:
            raise ValueError(f"{path}.allowed_paths must be a non-empty array")
        allowed_paths = tuple(
            _public_path(item, f"{path}.allowed_paths[{index}]")
            for index, item in enumerate(raw_paths)
        )
        if allowed_paths != tuple(sorted(set(allowed_paths))):
            raise ValueError(f"{path}.allowed_paths must be sorted and unique")
        return cls(
            code=_identifier(record["code"], f"{path}.code"),
            allowed_paths=allowed_paths,
            message=_public_message(record["message"], f"{path}.message"),
        )

    def to_record(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "allowed_paths": list(self.allowed_paths),
            "message": self.message,
        }


@dataclass(frozen=True)
class AuthoringPolicy:
    """Trusted limits and the complete public repair-message allowlist."""

    schema_version: int
    policy_id: str
    max_attempts: int
    max_response_bytes: int
    issue_rules: tuple[PublicIssueRule, ...]

    @classmethod
    def from_record(cls, value: Any) -> AuthoringPolicy:
        record = _mapping(value, "$authoring_policy")
        _keys(
            record,
            {
                "schema_version",
                "policy_id",
                "max_attempts",
                "max_response_bytes",
                "issue_rules",
            },
            "$authoring_policy",
        )
        if record["schema_version"] != 1:
            raise ValueError("AuthoringPolicy schema_version must be 1")
        raw_rules = record["issue_rules"]
        if not isinstance(raw_rules, list) or not raw_rules:
            raise ValueError("AuthoringPolicy issue_rules must be non-empty")
        rules = tuple(
            PublicIssueRule.from_record(
                item,
                f"$authoring_policy.issue_rules[{index}]",
            )
            for index, item in enumerate(raw_rules)
        )
        codes = tuple(rule.code for rule in rules)
        if codes != tuple(sorted(set(codes))):
            raise ValueError(
                "AuthoringPolicy issue rule codes must be sorted and unique"
            )
        return cls(
            schema_version=1,
            policy_id=_identifier(
                record["policy_id"],
                "$authoring_policy.policy_id",
            ),
            max_attempts=_positive_integer(
                record["max_attempts"],
                "$authoring_policy.max_attempts",
                maximum=_MAX_POLICY_ATTEMPTS,
            ),
            max_response_bytes=_positive_integer(
                record["max_response_bytes"],
                "$authoring_policy.max_response_bytes",
                maximum=_MAX_POLICY_RESPONSE_BYTES,
            ),
            issue_rules=rules,
        )

    @classmethod
    def from_json(cls, text: str) -> AuthoringPolicy:
        return cls.from_record(strict_json_loads(text))

    @property
    def issue_rule_by_code(self) -> dict[str, PublicIssueRule]:
        return {rule.code: rule for rule in self.issue_rules}

    def to_record(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "policy_id": self.policy_id,
            "max_attempts": self.max_attempts,
            "max_response_bytes": self.max_response_bytes,
            "issue_rules": [rule.to_record() for rule in self.issue_rules],
        }

    def to_json(self) -> str:
        return canonical_json(self.to_record())

    @property
    def fingerprint(self) -> str:
        return canonical_fingerprint(self.to_record())


_DEFAULT_POLICY_RECORD = {
    "schema_version": 1,
    "policy_id": "rexpolicy/task_authoring/v1",
    "max_attempts": 3,
    "max_response_bytes": 262_144,
    "issue_rules": [
        {
            "code": "adapter_not_allowed",
            "allowed_paths": ["$.environment.adapter_id"],
            "message": "Select an adapter allowed by the authoring intent.",
        },
        {
            "code": "audit_commitment_locked",
            "allowed_paths": ["$.instructions.audit"],
            "message": "Keep the sealed audit commitment fixed.",
        },
        {
            "code": "compile_rejected",
            "allowed_paths": [
                "$.action_projection",
                "$.environment",
                "$.goal",
                "$.reward_profiles",
                "$.terminal_rules",
            ],
            "message": "Revise this field to satisfy the trusted compiler policy.",
        },
        {
            "code": "event_schema_not_allowed",
            "allowed_paths": ["$.event_schema_id"],
            "message": "Select an event schema allowed by the authoring intent.",
        },
        {
            "code": "identity_locked",
            "allowed_paths": ["$.family_id", "$.task_id", "$.version"],
            "message": "Keep the task identity fixed by the authoring intent.",
        },
        {
            "code": "invalid_json",
            "allowed_paths": ["$"],
            "message": "Return exactly one valid JSON object.",
        },
        {
            "code": "process_spec_not_allowed",
            "allowed_paths": ["$.process_spec_id"],
            "message": "Select a process spec allowed by the authoring intent.",
        },
        {
            "code": "property_rejected",
            "allowed_paths": ["$.validation_examples"],
            "message": "Revise the public examples to satisfy the property policy.",
        },
        {
            "code": "response_too_large",
            "allowed_paths": ["$"],
            "message": "Keep the proposal within the response size limit.",
        },
        {
            "code": "schema_invalid",
            "allowed_paths": ["$"],
            "message": "Match the required TaskSpec v2 JSON schema.",
        },
    ],
}

DEFAULT_AUTHORING_POLICY = AuthoringPolicy.from_record(_DEFAULT_POLICY_RECORD)


@dataclass(frozen=True)
class PublicIssue:
    """A normalized issue that cannot carry arbitrary internal diagnostics."""

    code: str
    path: str
    message: str

    @classmethod
    def from_record(
        cls,
        value: Any,
        path: str,
        *,
        policy: AuthoringPolicy = DEFAULT_AUTHORING_POLICY,
    ) -> PublicIssue:
        record = _mapping(value, path)
        _keys(record, {"code", "path", "message"}, path)
        issue = normalize_public_issue(
            code=record["code"],
            path=record["path"],
            policy=policy,
        )
        supplied_message = _public_message(record["message"], f"{path}.message")
        if supplied_message != issue.message:
            raise ValueError(f"{path}.message is not the policy-owned public message")
        return issue

    def to_record(self) -> dict[str, Any]:
        return {"code": self.code, "path": self.path, "message": self.message}


def normalize_public_issue(
    *,
    code: Any,
    path: Any,
    policy: AuthoringPolicy = DEFAULT_AUTHORING_POLICY,
) -> PublicIssue:
    """Map a code/path pair to an exact policy-owned public message."""
    normalized_code = _identifier(code, "issue.code")
    normalized_path = _public_path(path, "issue.path")
    try:
        rule = policy.issue_rule_by_code[normalized_code]
    except KeyError as error:
        raise ValueError(
            "Issue code is outside the authoring policy allowlist"
        ) from error
    if normalized_path not in rule.allowed_paths:
        raise ValueError("Issue path is outside the authoring policy allowlist")
    return PublicIssue(
        code=rule.code,
        path=normalized_path,
        message=rule.message,
    )


def normalize_public_issues(
    issues: Iterable[tuple[str, str]],
    *,
    policy: AuthoringPolicy = DEFAULT_AUTHORING_POLICY,
) -> tuple[PublicIssue, ...]:
    """Normalize, deduplicate, and canonically order public repair issues."""
    normalized = {
        (issue.code, issue.path): issue
        for issue in (
            normalize_public_issue(code=code, path=path, policy=policy)
            for code, path in issues
        )
    }
    return tuple(normalized[key] for key in sorted(normalized))


@dataclass(frozen=True)
class AuthoringIntent:
    """One immutable authoring target and all trusted dependency identities."""

    schema_version: int
    task_id: str
    task_version: int
    family_id: str
    audit_commitment: AuditCommitment
    allowed_adapter_ids: tuple[str, ...]
    allowed_event_schema_ids: tuple[str, ...]
    allowed_process_spec_ids: tuple[str, ...]
    capability_catalog_fingerprint: str
    compiler_policy_id: str
    compiler_policy_fingerprint: str
    property_policy_id: str
    property_policy_fingerprint: str
    authoring_policy_id: str
    authoring_policy_fingerprint: str

    @classmethod
    def from_record(
        cls,
        value: Any,
        *,
        policy: AuthoringPolicy = DEFAULT_AUTHORING_POLICY,
    ) -> AuthoringIntent:
        record = _mapping(value, "$authoring_intent")
        _keys(
            record,
            {
                "schema_version",
                "task_id",
                "task_version",
                "family_id",
                "audit_commitment",
                "allowed_adapter_ids",
                "allowed_event_schema_ids",
                "allowed_process_spec_ids",
                "capability_catalog_fingerprint",
                "compiler_policy_id",
                "compiler_policy_fingerprint",
                "property_policy_id",
                "property_policy_fingerprint",
                "authoring_policy_id",
                "authoring_policy_fingerprint",
            },
            "$authoring_intent",
        )
        if record["schema_version"] != 1:
            raise ValueError("AuthoringIntent schema_version must be 1")
        task_id = _identifier(record["task_id"], "$authoring_intent.task_id")
        task_version = _positive_integer(
            record["task_version"],
            "$authoring_intent.task_version",
        )
        if not task_id.endswith(f"/v{task_version}"):
            raise ValueError("AuthoringIntent task ID suffix must match task_version")
        authoring_policy_id = _identifier(
            record["authoring_policy_id"],
            "$authoring_intent.authoring_policy_id",
        )
        authoring_policy_fingerprint = _sha256(
            record["authoring_policy_fingerprint"],
            "$authoring_intent.authoring_policy_fingerprint",
        )
        if authoring_policy_id != policy.policy_id:
            raise ValueError("AuthoringIntent authoring policy ID mismatch")
        if authoring_policy_fingerprint != policy.fingerprint:
            raise ValueError("AuthoringIntent authoring policy fingerprint mismatch")
        return cls(
            schema_version=1,
            task_id=task_id,
            task_version=task_version,
            family_id=_identifier(
                record["family_id"],
                "$authoring_intent.family_id",
            ),
            audit_commitment=AuditCommitment.from_record(
                record["audit_commitment"],
                "$authoring_intent.audit_commitment",
            ),
            allowed_adapter_ids=_sorted_identifiers(
                record["allowed_adapter_ids"],
                "$authoring_intent.allowed_adapter_ids",
            ),
            allowed_event_schema_ids=_sorted_identifiers(
                record["allowed_event_schema_ids"],
                "$authoring_intent.allowed_event_schema_ids",
            ),
            allowed_process_spec_ids=_sorted_identifiers(
                record["allowed_process_spec_ids"],
                "$authoring_intent.allowed_process_spec_ids",
            ),
            capability_catalog_fingerprint=_sha256(
                record["capability_catalog_fingerprint"],
                "$authoring_intent.capability_catalog_fingerprint",
            ),
            compiler_policy_id=_identifier(
                record["compiler_policy_id"],
                "$authoring_intent.compiler_policy_id",
            ),
            compiler_policy_fingerprint=_sha256(
                record["compiler_policy_fingerprint"],
                "$authoring_intent.compiler_policy_fingerprint",
            ),
            property_policy_id=_identifier(
                record["property_policy_id"],
                "$authoring_intent.property_policy_id",
            ),
            property_policy_fingerprint=_sha256(
                record["property_policy_fingerprint"],
                "$authoring_intent.property_policy_fingerprint",
            ),
            authoring_policy_id=authoring_policy_id,
            authoring_policy_fingerprint=authoring_policy_fingerprint,
        )

    @classmethod
    def from_json(
        cls,
        text: str,
        *,
        policy: AuthoringPolicy = DEFAULT_AUTHORING_POLICY,
    ) -> AuthoringIntent:
        return cls.from_record(strict_json_loads(text), policy=policy)

    def to_record(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "task_id": self.task_id,
            "task_version": self.task_version,
            "family_id": self.family_id,
            "audit_commitment": self.audit_commitment.to_record(),
            "allowed_adapter_ids": list(self.allowed_adapter_ids),
            "allowed_event_schema_ids": list(self.allowed_event_schema_ids),
            "allowed_process_spec_ids": list(self.allowed_process_spec_ids),
            "capability_catalog_fingerprint": self.capability_catalog_fingerprint,
            "compiler_policy_id": self.compiler_policy_id,
            "compiler_policy_fingerprint": self.compiler_policy_fingerprint,
            "property_policy_id": self.property_policy_id,
            "property_policy_fingerprint": self.property_policy_fingerprint,
            "authoring_policy_id": self.authoring_policy_id,
            "authoring_policy_fingerprint": self.authoring_policy_fingerprint,
        }

    def to_json(self) -> str:
        return canonical_json(self.to_record())

    @property
    def fingerprint(self) -> str:
        return canonical_fingerprint(self.to_record())


@dataclass(frozen=True)
class ProposalAttempt:
    """Hash-only provenance for one completed proposer attempt."""

    schema_version: int
    intent_fingerprint: str
    ordinal: int
    parent_attempt_fingerprint: str | None
    proposer_id: str
    model_id: str
    template_fingerprint: str
    request_fingerprint: str
    raw_response_sha256: str
    raw_response_bytes: int
    status: str
    issues: tuple[PublicIssue, ...]

    @classmethod
    def from_record(
        cls,
        value: Any,
        *,
        policy: AuthoringPolicy = DEFAULT_AUTHORING_POLICY,
    ) -> ProposalAttempt:
        record = _mapping(value, "$proposal_attempt")
        _keys(
            record,
            {
                "schema_version",
                "intent_fingerprint",
                "ordinal",
                "parent_attempt_fingerprint",
                "proposer_id",
                "model_id",
                "template_fingerprint",
                "request_fingerprint",
                "raw_response_sha256",
                "raw_response_bytes",
                "status",
                "issues",
            },
            "$proposal_attempt",
        )
        if record["schema_version"] != 1:
            raise ValueError("ProposalAttempt schema_version must be 1")
        ordinal = _positive_integer(
            record["ordinal"],
            "$proposal_attempt.ordinal",
            maximum=policy.max_attempts,
        )
        raw_parent = record["parent_attempt_fingerprint"]
        parent = (
            None
            if raw_parent is None
            else _sha256(raw_parent, "$proposal_attempt.parent_attempt_fingerprint")
        )
        if (ordinal == 1) != (parent is None):
            raise ValueError(
                "ProposalAttempt first ordinal must have no parent and repairs must"
                " have a parent"
            )
        status = _identifier(record["status"], "$proposal_attempt.status")
        if status not in _ATTEMPT_STATUSES:
            raise ValueError("ProposalAttempt status is unsupported")
        raw_issues = record["issues"]
        if not isinstance(raw_issues, list):
            raise ValueError("$proposal_attempt.issues must be an array")
        issues = tuple(
            PublicIssue.from_record(
                item,
                f"$proposal_attempt.issues[{index}]",
                policy=policy,
            )
            for index, item in enumerate(raw_issues)
        )
        issue_keys = tuple((issue.code, issue.path) for issue in issues)
        if issue_keys != tuple(sorted(set(issue_keys))):
            raise ValueError("ProposalAttempt issues must be sorted and unique")
        if (status == "candidate") != (not issues):
            raise ValueError("Only candidate attempts may have no public issues")
        if status == "repair_requested" and ordinal >= policy.max_attempts:
            raise ValueError(
                "ProposalAttempt cannot request repair at the attempt limit"
            )
        byte_count = _nonnegative_integer(
            record["raw_response_bytes"],
            "$proposal_attempt.raw_response_bytes",
        )
        too_large = byte_count > policy.max_response_bytes
        has_size_issue = any(issue.code == "response_too_large" for issue in issues)
        if too_large != has_size_issue:
            raise ValueError(
                "ProposalAttempt response size and response_too_large issue disagree"
            )
        if too_large and status != "rejected":
            raise ValueError("Oversized proposer responses must be rejected")
        return cls(
            schema_version=1,
            intent_fingerprint=_sha256(
                record["intent_fingerprint"],
                "$proposal_attempt.intent_fingerprint",
            ),
            ordinal=ordinal,
            parent_attempt_fingerprint=parent,
            proposer_id=_identifier(
                record["proposer_id"],
                "$proposal_attempt.proposer_id",
            ),
            model_id=_identifier(record["model_id"], "$proposal_attempt.model_id"),
            template_fingerprint=_sha256(
                record["template_fingerprint"],
                "$proposal_attempt.template_fingerprint",
            ),
            request_fingerprint=_sha256(
                record["request_fingerprint"],
                "$proposal_attempt.request_fingerprint",
            ),
            raw_response_sha256=_sha256(
                record["raw_response_sha256"],
                "$proposal_attempt.raw_response_sha256",
            ),
            raw_response_bytes=byte_count,
            status=status,
            issues=issues,
        )

    @classmethod
    def record_response(
        cls,
        raw_response: str | bytes,
        *,
        intent: AuthoringIntent,
        proposer_id: str,
        model_id: str,
        template_fingerprint: str,
        request_fingerprint: str,
        status: str,
        issues: Iterable[tuple[str, str]] = (),
        parent: ProposalAttempt | None = None,
        policy: AuthoringPolicy = DEFAULT_AUTHORING_POLICY,
    ) -> ProposalAttempt:
        """Record response provenance without retaining the response itself."""
        if intent.authoring_policy_id != policy.policy_id or (
            intent.authoring_policy_fingerprint != policy.fingerprint
        ):
            raise ValueError("AuthoringIntent is not bound to this authoring policy")
        if parent is not None and parent.intent_fingerprint != intent.fingerprint:
            raise ValueError("Parent attempt belongs to a different authoring intent")
        if parent is not None and parent.status != "repair_requested":
            raise ValueError("Only a repair-requested attempt may have a child")
        ordinal = 1 if parent is None else parent.ordinal + 1
        if ordinal > policy.max_attempts:
            raise ValueError("Authoring attempt limit is exhausted")
        if not isinstance(raw_response, (str, bytes)):
            raise TypeError("Proposer response must be text or bytes")
        response_bytes = (
            raw_response.encode("utf-8")
            if isinstance(raw_response, str)
            else raw_response
        )
        normalized_issues = normalize_public_issues(issues, policy=policy)
        record = {
            "schema_version": 1,
            "intent_fingerprint": intent.fingerprint,
            "ordinal": ordinal,
            "parent_attempt_fingerprint": (
                None if parent is None else parent.fingerprint
            ),
            "proposer_id": proposer_id,
            "model_id": model_id,
            "template_fingerprint": template_fingerprint,
            "request_fingerprint": request_fingerprint,
            "raw_response_sha256": hashlib.sha256(response_bytes).hexdigest(),
            "raw_response_bytes": len(response_bytes),
            "status": status,
            "issues": [issue.to_record() for issue in normalized_issues],
        }
        return cls.from_record(record, policy=policy)

    @classmethod
    def from_json(
        cls,
        text: str,
        *,
        policy: AuthoringPolicy = DEFAULT_AUTHORING_POLICY,
    ) -> ProposalAttempt:
        return cls.from_record(strict_json_loads(text), policy=policy)

    def to_record(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "intent_fingerprint": self.intent_fingerprint,
            "ordinal": self.ordinal,
            "parent_attempt_fingerprint": self.parent_attempt_fingerprint,
            "proposer_id": self.proposer_id,
            "model_id": self.model_id,
            "template_fingerprint": self.template_fingerprint,
            "request_fingerprint": self.request_fingerprint,
            "raw_response_sha256": self.raw_response_sha256,
            "raw_response_bytes": self.raw_response_bytes,
            "status": self.status,
            "issues": [issue.to_record() for issue in self.issues],
        }

    def to_json(self) -> str:
        return canonical_json(self.to_record())

    @property
    def fingerprint(self) -> str:
        return canonical_fingerprint(self.to_record())


def decode_authoring_policy_json(text: str) -> AuthoringPolicy:
    """Decode one strict authoring-policy JSON artifact."""
    return AuthoringPolicy.from_json(text)


def decode_authoring_intent_json(
    text: str,
    *,
    policy: AuthoringPolicy = DEFAULT_AUTHORING_POLICY,
) -> AuthoringIntent:
    """Decode an intent and require its exact trusted policy binding."""
    return AuthoringIntent.from_json(text, policy=policy)


def decode_proposal_attempt_json(
    text: str,
    *,
    policy: AuthoringPolicy = DEFAULT_AUTHORING_POLICY,
) -> ProposalAttempt:
    """Decode one completed hash-only proposal-attempt artifact."""
    return ProposalAttempt.from_json(text, policy=policy)
