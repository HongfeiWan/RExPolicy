"""Immutable quarantine storage for automatic TaskSpec authoring evidence.

The archive is deliberately a commit-marker store rather than a mutable state
directory.  Payloads are published immutably and ``descriptor.json`` is linked
last.  Readers ignore entries without that descriptor, so an interrupted write
can never look like a complete authoring attempt.
"""

from __future__ import annotations

import errno
import hashlib
import os
import re
import stat
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .authoring import (
    DEFAULT_AUTHORING_POLICY,
    AuthoringPolicy,
    ProposalAttempt,
)
from .authoring_runner import ProposerInvocation
from .canonical import canonical_fingerprint, canonical_json, strict_json_loads

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_ARTIFACT_TYPE = "rexpolicy_task_authoring_quarantine"
_DESCRIPTOR_FILENAME = "descriptor.json"
_RAW_RESPONSE_FILENAME = "raw-response.bin"
_ATTEMPT_FILENAME = "proposal-attempt.json"
_INVOCATION_FILENAME = "proposer-invocation.json"
_CANDIDATE_FILENAME = "static-candidate-bundle.json"
_JSON_FILE_LIMIT = 1_000_000
_PARTIAL_FILENAME = re.compile(
    r"^\.(?:descriptor\.json|raw-response\.bin|proposal-attempt\.json|"
    r"proposer-invocation\.json|static-candidate-bundle\.json)\.partial-"
    r"[0-9a-f]{32}$"
)
_INVOCATION_STATUSES = frozenset(
    {"completed", "nonzero_exit", "timeout", "stdout_limit", "stderr_limit"}
)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _require_sha256(value: Any, path: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{path} must be a lowercase SHA-256")
    return value


def _require_nonnegative_integer(value: Any, path: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{path} must be a non-negative integer")
    return value


def _require_positive_integer(value: Any, path: str) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{path} must be a positive integer")
    return value


def _require_text(value: Any, path: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or "\x00" in value
        or any(character in "\r\n" for character in value)
    ):
        raise ValueError(f"{path} must be non-empty single-line text")
    return value


def _require_keys(record: dict[str, Any], expected: set[str], path: str) -> None:
    missing = sorted(expected.difference(record))
    unknown = sorted(set(record).difference(expected))
    if missing:
        raise ValueError(f"{path} is missing fields: {', '.join(missing)}")
    if unknown:
        raise ValueError(f"{path} has unknown fields: {', '.join(unknown)}")


def _canonical_record(value: Any, path: str) -> dict[str, Any]:
    if isinstance(value, Mapping):
        source: Any = dict(value)
    else:
        to_record = getattr(value, "to_record", None)
        if not callable(to_record):
            raise TypeError(f"{path} must be a mapping or expose to_record()")
        source = to_record()
    try:
        payload = canonical_json(source)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{path} is not canonical JSON data") from error
    record = strict_json_loads(payload, max_bytes=_JSON_FILE_LIMIT)
    if not isinstance(record, dict):
        raise ValueError(f"{path} must be a JSON object")
    supplied_fingerprint = getattr(value, "fingerprint", None)
    if supplied_fingerprint is not None and supplied_fingerprint != canonical_fingerprint(record):
        raise ValueError(f"{path} fingerprint does not match its record")
    return record


@dataclass(frozen=True)
class QuarantineFile:
    """One exact payload committed by a quarantine descriptor."""

    filename: str
    sha256: str
    bytes: int

    @classmethod
    def from_record(
        cls,
        value: Any,
        *,
        path: str,
        expected_filename: str,
    ) -> QuarantineFile:
        if not isinstance(value, dict):
            raise ValueError(f"{path} must be an object")
        _require_keys(value, {"filename", "sha256", "bytes"}, path)
        if value["filename"] != expected_filename:
            raise ValueError(f"{path}.filename is not the fixed archive filename")
        return cls(
            filename=expected_filename,
            sha256=_require_sha256(value["sha256"], f"{path}.sha256"),
            bytes=_require_nonnegative_integer(value["bytes"], f"{path}.bytes"),
        )

    def to_record(self) -> dict[str, Any]:
        return {
            "filename": self.filename,
            "sha256": self.sha256,
            "bytes": self.bytes,
        }


@dataclass(frozen=True)
class AuthoringQuarantineDescriptor:
    """Commit marker binding every byte in one quarantined attempt."""

    schema_version: int
    artifact_type: str
    intent_fingerprint: str
    attempt_ordinal: int
    attempt_fingerprint: str
    raw_response: QuarantineFile
    proposal_attempt: QuarantineFile
    proposer_invocation: QuarantineFile
    static_candidate_bundle: QuarantineFile | None

    @classmethod
    def from_record(cls, value: Any) -> AuthoringQuarantineDescriptor:
        if not isinstance(value, dict):
            raise ValueError("$quarantine_descriptor must be an object")
        _require_keys(
            value,
            {
                "schema_version",
                "artifact_type",
                "intent_fingerprint",
                "attempt_ordinal",
                "attempt_fingerprint",
                "raw_response",
                "proposal_attempt",
                "proposer_invocation",
                "static_candidate_bundle",
            },
            "$quarantine_descriptor",
        )
        if value["schema_version"] != 1:
            raise ValueError("Quarantine descriptor schema_version must be 1")
        if value["artifact_type"] != _ARTIFACT_TYPE:
            raise ValueError("Quarantine descriptor artifact_type is unsupported")
        raw_candidate = value["static_candidate_bundle"]
        candidate = (
            None
            if raw_candidate is None
            else QuarantineFile.from_record(
                raw_candidate,
                path="$quarantine_descriptor.static_candidate_bundle",
                expected_filename=_CANDIDATE_FILENAME,
            )
        )
        return cls(
            schema_version=1,
            artifact_type=_ARTIFACT_TYPE,
            intent_fingerprint=_require_sha256(
                value["intent_fingerprint"],
                "$quarantine_descriptor.intent_fingerprint",
            ),
            attempt_ordinal=_require_positive_integer(
                value["attempt_ordinal"],
                "$quarantine_descriptor.attempt_ordinal",
            ),
            attempt_fingerprint=_require_sha256(
                value["attempt_fingerprint"],
                "$quarantine_descriptor.attempt_fingerprint",
            ),
            raw_response=QuarantineFile.from_record(
                value["raw_response"],
                path="$quarantine_descriptor.raw_response",
                expected_filename=_RAW_RESPONSE_FILENAME,
            ),
            proposal_attempt=QuarantineFile.from_record(
                value["proposal_attempt"],
                path="$quarantine_descriptor.proposal_attempt",
                expected_filename=_ATTEMPT_FILENAME,
            ),
            proposer_invocation=QuarantineFile.from_record(
                value["proposer_invocation"],
                path="$quarantine_descriptor.proposer_invocation",
                expected_filename=_INVOCATION_FILENAME,
            ),
            static_candidate_bundle=candidate,
        )

    def to_record(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "artifact_type": self.artifact_type,
            "intent_fingerprint": self.intent_fingerprint,
            "attempt_ordinal": self.attempt_ordinal,
            "attempt_fingerprint": self.attempt_fingerprint,
            "raw_response": self.raw_response.to_record(),
            "proposal_attempt": self.proposal_attempt.to_record(),
            "proposer_invocation": self.proposer_invocation.to_record(),
            "static_candidate_bundle": (
                None
                if self.static_candidate_bundle is None
                else self.static_candidate_bundle.to_record()
            ),
        }

    @property
    def fingerprint(self) -> str:
        return canonical_fingerprint(self.to_record())


@dataclass(frozen=True)
class AuthoringQuarantineWrite:
    intent_fingerprint: str
    attempt_fingerprint: str
    ordinal: int
    descriptor_sha256: str
    relative_path: str
    created: bool


@dataclass(frozen=True)
class AuthoringQuarantineEntry:
    raw_response: bytes
    attempt: ProposalAttempt
    invocation: ProposerInvocation
    static_candidate_bundle: dict[str, Any] | None
    descriptor: AuthoringQuarantineDescriptor
    relative_path: str


def _file_record(filename: str, payload: bytes) -> QuarantineFile:
    return QuarantineFile(filename=filename, sha256=_sha256(payload), bytes=len(payload))


def _relative_entry_path(attempt: ProposalAttempt) -> Path:
    return (
        Path("archive")
        / "authoring"
        / "quarantine"
        / attempt.intent_fingerprint[:2]
        / attempt.intent_fingerprint
        / f"attempt-{attempt.ordinal:03d}"
        / attempt.fingerprint
    )


def _relative_lookup_path(
    *, intent_fingerprint: str, ordinal: int, attempt_fingerprint: str
) -> Path:
    _require_sha256(intent_fingerprint, "intent_fingerprint")
    _require_sha256(attempt_fingerprint, "attempt_fingerprint")
    _require_positive_integer(ordinal, "ordinal")
    return (
        Path("archive")
        / "authoring"
        / "quarantine"
        / intent_fingerprint[:2]
        / intent_fingerprint
        / f"attempt-{ordinal:03d}"
        / attempt_fingerprint
    )


def _resolved_root(run_dir: Path, *, create: bool) -> Path:
    root = Path(run_dir)
    if create:
        root.mkdir(parents=True, exist_ok=True)
    try:
        return root.resolve(strict=True)
    except OSError as error:
        raise FileNotFoundError("Quarantine run directory is unavailable") from error


def _checked_directory(path: Path) -> None:
    try:
        mode = path.lstat().st_mode
    except OSError as error:
        raise FileNotFoundError(f"Quarantine directory is unavailable: {path}") from error
    if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
        raise ValueError(f"Quarantine path component is not a real directory: {path}")


def _walk_directory(root: Path, relative: Path, *, create: bool) -> Path:
    current = root
    _checked_directory(current)
    for part in relative.parts:
        if part in {"", ".", ".."} or os.sep in part:
            raise ValueError("Unsafe quarantine path component")
        child = current / part
        if create:
            try:
                child.mkdir()
                _fsync_directory(current)
            except FileExistsError:
                pass
        _checked_directory(child)
        current = child
    return current


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _existing_regular_file(path: Path) -> bool:
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError:
        return False
    if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
        raise ValueError(f"Quarantine payload is not a real regular file: {path}")
    return True


def _publish_immutable(path: Path, payload: bytes) -> bool:
    """Hard-link one fully fsynced payload without ever replacing a target."""
    if _existing_regular_file(path):
        if path.read_bytes() != payload:
            raise FileExistsError(f"Immutable quarantine payload conflicts at {path}")
        return False
    temporary = path.with_name(f".{path.name}.partial-{uuid.uuid4().hex}")
    try:
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(temporary, flags, 0o600)
        try:
            with os.fdopen(descriptor, "wb", closefd=True) as file:
                file.write(payload)
                file.flush()
                os.fsync(file.fileno())
        except BaseException:
            # fdopen owns the descriptor after it succeeds.
            raise
        try:
            os.link(temporary, path, follow_symlinks=False)
            created = True
        except FileExistsError:
            if not _existing_regular_file(path) or path.read_bytes() != payload:
                raise FileExistsError(
                    f"Concurrent quarantine payload conflicts at {path}"
                )
            created = False
        _fsync_directory(path.parent)
        return created
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _read_payload(path: Path, expected: QuarantineFile, *, maximum: int) -> bytes:
    if not _existing_regular_file(path):
        raise ValueError(f"Quarantine payload is missing: {expected.filename}")
    size = path.lstat().st_size
    if size != expected.bytes:
        raise ValueError(f"Quarantine payload size drifted: {expected.filename}")
    if size > maximum:
        raise ValueError(f"Quarantine payload exceeds its trusted bound: {expected.filename}")
    payload = path.read_bytes()
    if _sha256(payload) != expected.sha256:
        raise ValueError(f"Quarantine payload hash drifted: {expected.filename}")
    return payload


def _canonical_json_payload(record: Mapping[str, Any]) -> bytes:
    return canonical_json(dict(record)).encode("ascii")


def _parse_canonical_record(payload: bytes, path: str) -> dict[str, Any]:
    try:
        text = payload.decode("ascii")
    except UnicodeDecodeError as error:
        raise ValueError(f"{path} must be canonical ASCII JSON") from error
    record = strict_json_loads(text, max_bytes=_JSON_FILE_LIMIT)
    if not isinstance(record, dict):
        raise ValueError(f"{path} must be a JSON object")
    if canonical_json(record).encode("ascii") != payload:
        raise ValueError(f"{path} is not canonical JSON")
    return record


def _invocation_from_record(value: Any) -> ProposerInvocation:
    if not isinstance(value, dict):
        raise ValueError("$proposer_invocation must be an object")
    fields = {
        "schema_version",
        "command_id",
        "command_sha256",
        "execution_policy_id",
        "execution_policy_sha256",
        "request_sha256",
        "request_bytes",
        "status",
        "exit_code",
        "stdout_sha256",
        "stdout_bytes",
        "stdout_truncated",
        "stderr_sha256",
        "stderr_bytes",
        "stderr_truncated",
    }
    _require_keys(value, fields, "$proposer_invocation")
    if value["schema_version"] != 1:
        raise ValueError("Proposer invocation schema_version must be 1")
    status_value = value["status"]
    if status_value not in _INVOCATION_STATUSES:
        raise ValueError("Proposer invocation status is invalid")
    exit_code = value["exit_code"]
    if status_value == "completed":
        if exit_code != 0:
            raise ValueError("Completed proposer invocation must exit with zero")
    elif status_value == "nonzero_exit":
        if type(exit_code) is not int or exit_code == 0:
            raise ValueError("Nonzero proposer invocation must retain its exit code")
    elif exit_code is not None:
        raise ValueError("Terminated proposer invocation cannot claim an exit code")
    stdout_truncated = value["stdout_truncated"]
    stderr_truncated = value["stderr_truncated"]
    if type(stdout_truncated) is not bool or type(stderr_truncated) is not bool:
        raise ValueError("Proposer truncation fields must be booleans")
    if stdout_truncated != (status_value == "stdout_limit"):
        raise ValueError("Proposer stdout truncation disagrees with status")
    if stderr_truncated != (status_value == "stderr_limit"):
        raise ValueError("Proposer stderr truncation disagrees with status")
    invocation = ProposerInvocation(
        schema_version=1,
        command_id=_require_text(value["command_id"], "$proposer_invocation.command_id"),
        command_sha256=_require_sha256(
            value["command_sha256"], "$proposer_invocation.command_sha256"
        ),
        execution_policy_id=_require_text(
            value["execution_policy_id"], "$proposer_invocation.execution_policy_id"
        ),
        execution_policy_sha256=_require_sha256(
            value["execution_policy_sha256"],
            "$proposer_invocation.execution_policy_sha256",
        ),
        request_sha256=_require_sha256(
            value["request_sha256"], "$proposer_invocation.request_sha256"
        ),
        request_bytes=_require_nonnegative_integer(
            value["request_bytes"], "$proposer_invocation.request_bytes"
        ),
        status=status_value,
        exit_code=exit_code,
        stdout_sha256=_require_sha256(
            value["stdout_sha256"], "$proposer_invocation.stdout_sha256"
        ),
        stdout_bytes=_require_nonnegative_integer(
            value["stdout_bytes"], "$proposer_invocation.stdout_bytes"
        ),
        stdout_truncated=stdout_truncated,
        stderr_sha256=_require_sha256(
            value["stderr_sha256"], "$proposer_invocation.stderr_sha256"
        ),
        stderr_bytes=_require_nonnegative_integer(
            value["stderr_bytes"], "$proposer_invocation.stderr_bytes"
        ),
        stderr_truncated=stderr_truncated,
    )
    if invocation.to_record() != value:
        raise ValueError("Proposer invocation record is not canonical")
    return invocation


def _validate_chain(
    *,
    raw_response: bytes,
    attempt: ProposalAttempt,
    invocation: ProposerInvocation,
    candidate: dict[str, Any] | None,
    policy: AuthoringPolicy,
) -> None:
    if attempt.ordinal > policy.max_attempts:
        raise ValueError("Proposal attempt exceeds the authoring attempt limit")
    if len(raw_response) > policy.max_response_bytes + 1:
        raise ValueError("Raw proposer response exceeds the bounded capture limit")
    if attempt.raw_response_sha256 != _sha256(raw_response):
        raise ValueError("Proposal attempt does not bind the raw response hash")
    if attempt.raw_response_bytes != len(raw_response):
        raise ValueError("Proposal attempt does not bind the raw response size")
    canonical_invocation = _invocation_from_record(invocation.to_record())
    if attempt.schema_version == 2:
        if attempt.invocation_fingerprint != canonical_invocation.fingerprint:
            raise ValueError("Proposal attempt invocation binding drifted")
        if attempt.response_complete == canonical_invocation.stdout_truncated:
            raise ValueError("Proposal attempt response completeness drifted")
    if canonical_invocation.stdout_sha256 != attempt.raw_response_sha256:
        raise ValueError("Proposer invocation stdout hash does not bind the attempt")
    if canonical_invocation.stdout_bytes != attempt.raw_response_bytes:
        raise ValueError("Proposer invocation stdout size does not bind the attempt")
    if canonical_invocation.request_sha256 != attempt.request_fingerprint:
        raise ValueError("Proposer invocation request does not bind the attempt")
    if candidate is None:
        return
    if not attempt.response_complete or canonical_invocation.status != "completed":
        raise ValueError("Static candidate requires one complete proposer response")
    if candidate.get("intent_fingerprint") != attempt.intent_fingerprint:
        raise ValueError("Static candidate bundle belongs to a different intent")
    bindings = [
        candidate[key]
        for key in ("attempt_fingerprint", "proposal_attempt_fingerprint")
        if key in candidate
    ]
    if not bindings or any(binding != attempt.fingerprint for binding in bindings):
        raise ValueError("Static candidate bundle does not bind the proposal attempt")
    if "raw_response_sha256" in candidate and (
        candidate["raw_response_sha256"] != attempt.raw_response_sha256
    ):
        raise ValueError("Static candidate bundle raw response binding drifted")


def _descriptor_for(
    *,
    raw_payload: bytes,
    attempt_payload: bytes,
    invocation_payload: bytes,
    candidate_payload: bytes | None,
    attempt: ProposalAttempt,
) -> AuthoringQuarantineDescriptor:
    return AuthoringQuarantineDescriptor(
        schema_version=1,
        artifact_type=_ARTIFACT_TYPE,
        intent_fingerprint=attempt.intent_fingerprint,
        attempt_ordinal=attempt.ordinal,
        attempt_fingerprint=attempt.fingerprint,
        raw_response=_file_record(_RAW_RESPONSE_FILENAME, raw_payload),
        proposal_attempt=_file_record(_ATTEMPT_FILENAME, attempt_payload),
        proposer_invocation=_file_record(_INVOCATION_FILENAME, invocation_payload),
        static_candidate_bundle=(
            None
            if candidate_payload is None
            else _file_record(_CANDIDATE_FILENAME, candidate_payload)
        ),
    )


def _assert_directory_members(path: Path, expected: set[str]) -> None:
    unknown = []
    for item in path.iterdir():
        if item.name in expected:
            continue
        # A power loss can leave the fully private hard-link source behind.
        # It is never referenced by the descriptor and must not prevent an
        # otherwise identical retry or a verified read.
        if _PARTIAL_FILENAME.fullmatch(item.name) is not None:
            if not _existing_regular_file(item):
                raise ValueError("Quarantine partial payload is not a regular file")
            continue
        unknown.append(item.name)
    unknown.sort()
    if unknown:
        raise ValueError(
            "Quarantine entry contains uncommitted or unknown files: "
            + ", ".join(unknown)
        )


def write_authoring_quarantine(
    *,
    run_dir: Path,
    raw_response: bytes | str,
    attempt: ProposalAttempt,
    invocation: ProposerInvocation,
    static_candidate_bundle: Mapping[str, Any] | Any | None = None,
    policy: AuthoringPolicy = DEFAULT_AUTHORING_POLICY,
    _fault_injector: Callable[[str], None] | None = None,
) -> AuthoringQuarantineWrite:
    """Commit one immutable authoring attempt, with its descriptor written last."""
    if not isinstance(attempt, ProposalAttempt):
        raise TypeError("attempt must be a ProposalAttempt")
    # Re-parse to enforce this exact policy, not merely the construction policy.
    attempt = ProposalAttempt.from_record(attempt.to_record(), policy=policy)
    if not isinstance(raw_response, (bytes, str)):
        raise TypeError("raw_response must be bytes or text")
    raw_payload = (
        raw_response.encode("utf-8")
        if isinstance(raw_response, str)
        else bytes(raw_response)
    )
    candidate = (
        None
        if static_candidate_bundle is None
        else _canonical_record(static_candidate_bundle, "$static_candidate_bundle")
    )
    _validate_chain(
        raw_response=raw_payload,
        attempt=attempt,
        invocation=invocation,
        candidate=candidate,
        policy=policy,
    )
    attempt_payload = _canonical_json_payload(attempt.to_record())
    invocation_payload = _canonical_json_payload(invocation.to_record())
    candidate_payload = (
        None if candidate is None else _canonical_json_payload(candidate)
    )
    descriptor = _descriptor_for(
        raw_payload=raw_payload,
        attempt_payload=attempt_payload,
        invocation_payload=invocation_payload,
        candidate_payload=candidate_payload,
        attempt=attempt,
    )
    descriptor_payload = _canonical_json_payload(descriptor.to_record())
    relative = _relative_entry_path(attempt)
    root = _resolved_root(Path(run_dir), create=True)
    entry_dir = _walk_directory(root, relative, create=True)
    descriptor_path = entry_dir / _DESCRIPTOR_FILENAME

    if _existing_regular_file(descriptor_path):
        try:
            existing = load_authoring_quarantine(
                run_dir=root,
                intent_fingerprint=attempt.intent_fingerprint,
                ordinal=attempt.ordinal,
                attempt_fingerprint=attempt.fingerprint,
                policy=policy,
            )
        except (OSError, TypeError, ValueError) as error:
            raise FileExistsError(
                "Existing quarantine entry failed immutable verification"
            ) from error
        if (
            existing.raw_response != raw_payload
            or existing.attempt != attempt
            or existing.invocation != invocation
            or existing.static_candidate_bundle != candidate
            or existing.descriptor != descriptor
        ):
            raise FileExistsError("Existing quarantine entry has different content")
        return AuthoringQuarantineWrite(
            intent_fingerprint=attempt.intent_fingerprint,
            attempt_fingerprint=attempt.fingerprint,
            ordinal=attempt.ordinal,
            descriptor_sha256=descriptor.fingerprint,
            relative_path=relative.as_posix(),
            created=False,
        )

    payloads = [
        (_RAW_RESPONSE_FILENAME, raw_payload),
        (_ATTEMPT_FILENAME, attempt_payload),
        (_INVOCATION_FILENAME, invocation_payload),
    ]
    if candidate_payload is not None:
        payloads.append((_CANDIDATE_FILENAME, candidate_payload))
    elif _existing_regular_file(entry_dir / _CANDIDATE_FILENAME):
        raise FileExistsError("Incomplete quarantine entry has a conflicting candidate")

    expected_before_descriptor = {filename for filename, _ in payloads}
    _assert_directory_members(entry_dir, expected_before_descriptor)
    for filename, payload in payloads:
        _publish_immutable(entry_dir / filename, payload)
        if _fault_injector is not None:
            _fault_injector(f"after:{filename}")
    _assert_directory_members(entry_dir, expected_before_descriptor)
    _fsync_directory(entry_dir)
    if _fault_injector is not None:
        _fault_injector("before:descriptor.json")
    # This is the commit marker.  No fallible user hook or payload write follows.
    created = _publish_immutable(descriptor_path, descriptor_payload)
    return AuthoringQuarantineWrite(
        intent_fingerprint=attempt.intent_fingerprint,
        attempt_fingerprint=attempt.fingerprint,
        ordinal=attempt.ordinal,
        descriptor_sha256=descriptor.fingerprint,
        relative_path=relative.as_posix(),
        created=created,
    )


def load_authoring_quarantine(
    *,
    run_dir: Path,
    intent_fingerprint: str,
    ordinal: int,
    attempt_fingerprint: str,
    policy: AuthoringPolicy = DEFAULT_AUTHORING_POLICY,
) -> AuthoringQuarantineEntry:
    """Load and verify every hash and semantic link in a committed entry."""
    relative = _relative_lookup_path(
        intent_fingerprint=intent_fingerprint,
        ordinal=ordinal,
        attempt_fingerprint=attempt_fingerprint,
    )
    root = _resolved_root(Path(run_dir), create=False)
    entry_dir = _walk_directory(root, relative, create=False)
    descriptor_path = entry_dir / _DESCRIPTOR_FILENAME
    if not _existing_regular_file(descriptor_path):
        raise ValueError("Quarantine entry is incomplete: descriptor is absent")
    descriptor_bytes = descriptor_path.read_bytes()
    if len(descriptor_bytes) > _JSON_FILE_LIMIT:
        raise ValueError("Quarantine descriptor exceeds its trusted bound")
    descriptor_record = _parse_canonical_record(
        descriptor_bytes, "$quarantine_descriptor"
    )
    descriptor = AuthoringQuarantineDescriptor.from_record(descriptor_record)
    if descriptor_bytes != _canonical_json_payload(descriptor.to_record()):
        raise ValueError("Quarantine descriptor record is not canonical")
    if descriptor.fingerprint != _sha256(descriptor_bytes):
        raise ValueError("Quarantine descriptor fingerprint drifted")
    if descriptor.intent_fingerprint != intent_fingerprint:
        raise ValueError("Quarantine descriptor intent does not match its path")
    if descriptor.attempt_ordinal != ordinal:
        raise ValueError("Quarantine descriptor ordinal does not match its path")
    if descriptor.attempt_fingerprint != attempt_fingerprint:
        raise ValueError("Quarantine descriptor attempt does not match its path")

    expected_filenames = {
        _DESCRIPTOR_FILENAME,
        _RAW_RESPONSE_FILENAME,
        _ATTEMPT_FILENAME,
        _INVOCATION_FILENAME,
    }
    if descriptor.static_candidate_bundle is not None:
        expected_filenames.add(_CANDIDATE_FILENAME)
    _assert_directory_members(entry_dir, expected_filenames)

    raw_payload = _read_payload(
        entry_dir / _RAW_RESPONSE_FILENAME,
        descriptor.raw_response,
        maximum=policy.max_response_bytes + 1,
    )
    attempt_payload = _read_payload(
        entry_dir / _ATTEMPT_FILENAME,
        descriptor.proposal_attempt,
        maximum=_JSON_FILE_LIMIT,
    )
    invocation_payload = _read_payload(
        entry_dir / _INVOCATION_FILENAME,
        descriptor.proposer_invocation,
        maximum=_JSON_FILE_LIMIT,
    )
    attempt_record = _parse_canonical_record(attempt_payload, "$proposal_attempt")
    attempt = ProposalAttempt.from_record(attempt_record, policy=policy)
    if attempt_payload != _canonical_json_payload(attempt.to_record()):
        raise ValueError("Proposal attempt record is not canonical")
    if attempt.fingerprint != descriptor.proposal_attempt.sha256:
        raise ValueError("Proposal attempt fingerprint drifted")
    if attempt.fingerprint != attempt_fingerprint:
        raise ValueError("Proposal attempt does not match its content-addressed path")

    invocation_record = _parse_canonical_record(
        invocation_payload, "$proposer_invocation"
    )
    invocation = _invocation_from_record(invocation_record)
    if invocation.fingerprint != descriptor.proposer_invocation.sha256:
        raise ValueError("Proposer invocation fingerprint drifted")

    candidate: dict[str, Any] | None = None
    if descriptor.static_candidate_bundle is not None:
        candidate_payload = _read_payload(
            entry_dir / _CANDIDATE_FILENAME,
            descriptor.static_candidate_bundle,
            maximum=_JSON_FILE_LIMIT,
        )
        candidate = _parse_canonical_record(
            candidate_payload, "$static_candidate_bundle"
        )
        if canonical_fingerprint(candidate) != descriptor.static_candidate_bundle.sha256:
            raise ValueError("Static candidate bundle fingerprint drifted")

    _validate_chain(
        raw_response=raw_payload,
        attempt=attempt,
        invocation=invocation,
        candidate=candidate,
        policy=policy,
    )
    return AuthoringQuarantineEntry(
        raw_response=raw_payload,
        attempt=attempt,
        invocation=invocation,
        static_candidate_bundle=candidate,
        descriptor=descriptor,
        relative_path=relative.as_posix(),
    )
