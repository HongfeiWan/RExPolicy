"""Durable, single-chain execution for automatic TaskSpec authoring.

The journal in this module makes the authoring attempt budget a property of a
run directory, rather than of one Python process.  A fixed manifest is written
before any proposer can start.  Each proposer slot then has three immutable
records: the canonical request, a launch marker, and a completion record bound
to the immutable quarantine entry.

There is deliberately no automatic recovery from a launch marker without a
completion record.  In that state the process may have crossed the external
proposer boundary, so retrying would violate the fixed attempt budget.
"""

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import os
import re
import stat
import uuid
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any, Union

from .authoring import (
    DEFAULT_AUTHORING_POLICY,
    AuthoringIntent,
    AuthoringPolicy,
    ProposalAttempt,
    normalize_public_issues,
)
from .authoring_brief import PublicAuthoringBrief
from .authoring_candidate import (
    StaticCandidateBundle,
    validate_static_candidate,
)
from .authoring_quarantine import (
    AuthoringQuarantineEntry,
    AuthoringQuarantineWrite,
    load_authoring_quarantine,
    write_authoring_quarantine,
)
from .authoring_runner import (
    DEFAULT_PROPOSER_EXECUTION_POLICY,
    BubblewrapProposerCommand,
    ProposerCommand,
    ProposerExecutionPolicy,
    ProposerInvocation,
    run_proposer,
)
from .canonical import canonical_fingerprint, canonical_json, strict_json_loads
from .capabilities import CapabilityCatalog
from .contract import (
    DEFAULT_TASK_COMPILER_POLICY,
    CompiledTaskContract,
    TaskCompilerPolicy,
)
from .curriculum import CurriculumSnapshot
from .model import TaskSpecV2
from .process import ProcessSpecV1
from .process_contract import (
    DEFAULT_PROCESS_COMPILER_POLICY,
    ProcessCompilerPolicy,
)
from .property_validation import (
    DEFAULT_PROPERTY_VALIDATION_POLICY,
    PropertyValidationPolicy,
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_MANIFEST = "manifest.json"
_RESULT = "result.json"
_REQUEST = "request.json"
_LAUNCH = "launch.json"
_COMPLETION = "completion.json"
_MAX_JOB_JSON_BYTES = 4_000_000
_ProposerRuntimeCommand = Union[ProposerCommand, BubblewrapProposerCommand]


class AmbiguousAuthoringAttemptError(RuntimeError):
    """A proposer launch may have happened, so no retry is safe."""


class AuthoringJobDriftError(ValueError):
    """Durable authoring state differs from the requested immutable job."""


def _require_sha256(value: Any, path: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise AuthoringJobDriftError(f"{path} must be a lowercase SHA-256")
    return value


def _canonical_payload(record: Mapping[str, Any]) -> bytes:
    return canonical_json(dict(record)).encode("ascii")


def _parse_canonical(path: Path, *, maximum: int = _MAX_JOB_JSON_BYTES) -> dict[str, Any]:
    try:
        payload = path.read_bytes()
    except OSError as error:
        raise AuthoringJobDriftError(f"Cannot read authoring journal record {path.name}") from error
    if len(payload) > maximum:
        raise AuthoringJobDriftError(f"Authoring journal record {path.name} is too large")
    try:
        text = payload.decode("ascii")
        record = strict_json_loads(text, max_bytes=maximum)
    except (TypeError, UnicodeError, ValueError) as error:
        raise AuthoringJobDriftError(
            f"Authoring journal record {path.name} is not strict JSON"
        ) from error
    if not isinstance(record, dict) or payload != _canonical_payload(record):
        raise AuthoringJobDriftError(
            f"Authoring journal record {path.name} is not canonical"
        )
    return record


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _publish_once(path: Path, record: Mapping[str, Any]) -> bool:
    """Publish canonical JSON without replacing an existing inode."""
    payload = _canonical_payload(record)
    if len(payload) > _MAX_JOB_JSON_BYTES:
        raise ValueError("Authoring journal record exceeds the trusted size limit")
    if path.exists():
        if not path.is_file() or path.is_symlink() or path.read_bytes() != payload:
            raise AuthoringJobDriftError(f"Immutable authoring record {path.name} drifted")
        return False
    temporary = path.parent / f".{path.name}.partial-{uuid.uuid4().hex}"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(temporary, flags, 0o600)
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("short authoring journal write")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    try:
        os.link(temporary, path, follow_symlinks=False)
        _fsync_directory(path.parent)
        return True
    except FileExistsError:
        if not path.is_file() or path.is_symlink() or path.read_bytes() != payload:
            raise AuthoringJobDriftError(f"Immutable authoring record {path.name} drifted")
        return False
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _checked_directory(path: Path, *, create: bool) -> Path:
    if create:
        path.mkdir(parents=True, exist_ok=True)
    try:
        mode = path.lstat().st_mode
    except OSError as error:
        raise AuthoringJobDriftError("Authoring job directory is unavailable") from error
    if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
        raise AuthoringJobDriftError("Authoring job path must be a real directory")
    return path.resolve(strict=True)


@contextlib.contextmanager
def _exclusive_job_lock(run_dir: Path) -> Iterator[Path]:
    root = _checked_directory(Path(run_dir), create=True)
    lock_path = root / ".authoring-job.lock"
    flags = os.O_RDWR | os.O_CREAT
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(lock_path, flags, 0o600)
    except OSError as error:
        raise AuthoringJobDriftError("Cannot open the authoring job lock") from error
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield root
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _environment_binding(environment: Mapping[str, str] | None) -> dict[str, Any]:
    supplied = dict(environment or {})
    return {
        "names": sorted(supplied),
        "value_sha256s": {
            name: hashlib.sha256(value.encode("utf-8")).hexdigest()
            for name, value in sorted(supplied.items())
        },
    }


def _validated_execution_environment(
    environment: Mapping[str, str] | None,
    *,
    execution_policy: ProposerExecutionPolicy,
    authoring_policy: AuthoringPolicy,
) -> dict[str, str]:
    """Reject deterministic launch configuration errors before journaling."""
    execution_policy.validate(authoring_policy=authoring_policy)
    supplied = dict(environment or {})
    if set(supplied) != set(execution_policy.allowed_environment_names):
        raise ValueError("Proposer environment does not match the trusted allowlist")
    if any(
        not isinstance(value, str) or "\x00" in value
        for value in supplied.values()
    ):
        raise ValueError("Proposer environment value is invalid")
    return supplied


def _validate_request_size(
    request: Mapping[str, Any],
    *,
    execution_policy: ProposerExecutionPolicy,
) -> None:
    if len(_canonical_payload(request)) > execution_policy.max_request_bytes:
        raise ValueError("Proposer request exceeds the trusted size limit")


def _manifest_record(
    *,
    intent: AuthoringIntent,
    brief: PublicAuthoringBrief,
    curriculum_snapshot: CurriculumSnapshot,
    catalog: CapabilityCatalog,
    process_specs: Mapping[str, ProcessSpecV1],
    parent_task: TaskSpecV2 | None,
    parent_contract: CompiledTaskContract | None,
    command: _ProposerRuntimeCommand,
    model_id: str,
    proposer_binding: Any,
    session: Any,
    environment: Mapping[str, str] | None,
    template: Any,
    compiler_policy: TaskCompilerPolicy,
    property_policy: PropertyValidationPolicy,
    process_policy: ProcessCompilerPolicy,
    authoring_policy: AuthoringPolicy,
    execution_policy: ProposerExecutionPolicy,
) -> dict[str, Any]:
    canonical_command = command.canonical()
    canonical_command.verify_executable()
    if not isinstance(curriculum_snapshot, CurriculumSnapshot):
        raise ValueError("curriculum_snapshot must be one exact CurriculumSnapshot")
    canonical_snapshot = CurriculumSnapshot.from_record(
        curriculum_snapshot.to_record()
    )
    source_snapshot_fingerprint = _require_sha256(
        brief.source_curriculum_snapshot_fingerprint,
        "$public_brief.source_curriculum_snapshot_fingerprint",
    )
    if canonical_snapshot.fingerprint != source_snapshot_fingerprint:
        raise AuthoringJobDriftError(
            "Authoring brief does not bind the exact curriculum snapshot"
        )
    process_records = [
        process_specs[process_id].to_record()
        for process_id in sorted(process_specs)
    ]
    record = {
        "schema_version": 1,
        "artifact_type": "rexpolicy_task_authoring_job/v1",
        "intent": intent.to_record(),
        "intent_fingerprint": intent.fingerprint,
        "public_brief": brief.to_record(),
        "brief_fingerprint": brief.fingerprint,
        "curriculum_snapshot": canonical_snapshot.to_record(),
        "source_curriculum_snapshot_fingerprint": source_snapshot_fingerprint,
        "preauthoring_audit_plan_fingerprint": (
            intent.preauthoring_audit_plan_fingerprint
        ),
        "proposer_binding": proposer_binding.to_record(),
        "proposer_binding_fingerprint": proposer_binding.fingerprint,
        "session": session.to_record(),
        "session_fingerprint": session.fingerprint,
        "command": canonical_command.to_record(),
        "command_fingerprint": canonical_command.fingerprint,
        "model_id": model_id,
        "environment_binding": _environment_binding(environment),
        "capability_catalog": catalog.to_record(),
        "capability_catalog_fingerprint": catalog.fingerprint,
        "process_specs": process_records,
        "process_registry_fingerprint": canonical_fingerprint(
            {
                "process_specs": [
                    {
                        "process_spec_id": item.process_spec_id,
                        "process_spec_fingerprint": item.fingerprint,
                    }
                    for item in (
                        ProcessSpecV1.from_record(value) for value in process_records
                    )
                ]
            }
        ),
        "parent_task": None if parent_task is None else parent_task.to_record(),
        "parent_task_fingerprint": (
            None if parent_task is None else parent_task.fingerprint
        ),
        "parent_contract": (
            None if parent_contract is None else parent_contract.to_record()
        ),
        "parent_contract_fingerprint": (
            None if parent_contract is None else parent_contract.fingerprint
        ),
        "template": template.to_record(),
        "template_fingerprint": template.fingerprint,
        "compiler_policy": compiler_policy.to_record(),
        "compiler_policy_fingerprint": compiler_policy.fingerprint,
        "property_policy": property_policy.to_record(),
        "property_policy_fingerprint": property_policy.fingerprint,
        "process_policy": process_policy.to_record(),
        "process_policy_fingerprint": process_policy.fingerprint,
        "authoring_policy": authoring_policy.to_record(),
        "authoring_policy_fingerprint": authoring_policy.fingerprint,
        "execution_policy": execution_policy.to_record(),
        "execution_policy_fingerprint": execution_policy.fingerprint,
    }
    if record["preauthoring_audit_plan_fingerprint"] is None:
        raise ValueError("Durable authoring requires a pre-authoring audit plan")
    return record


def _validate_manifest(path: Path, expected: Mapping[str, Any]) -> None:
    if not path.exists():
        _publish_once(path, expected)
        return
    actual = _parse_canonical(path)
    if actual != dict(expected):
        raise AuthoringJobDriftError("Authoring job manifest binding drifted")


def _attempt_directory(attempts_dir: Path, ordinal: int) -> Path:
    path = attempts_dir / f"attempt-{ordinal:03d}"
    return _checked_directory(path, create=True)


def _assert_attempt_layout(attempts_dir: Path, *, maximum: int) -> None:
    expected_names = {f"attempt-{ordinal:03d}" for ordinal in range(1, maximum + 1)}
    for item in attempts_dir.iterdir():
        if item.name.startswith(".") and ".partial-" in item.name:
            continue
        if item.name not in expected_names or item.is_symlink() or not item.is_dir():
            raise AuthoringJobDriftError("Authoring attempt journal layout drifted")


def _assert_job_layout(job_dir: Path) -> None:
    for item in job_dir.iterdir():
        if item.name in {_MANIFEST, _RESULT, "attempts"}:
            continue
        if item.name.startswith(".") and ".partial-" in item.name and item.is_file():
            continue
        raise AuthoringJobDriftError("Authoring job journal layout drifted")


def _launch_record(
    *,
    ordinal: int,
    request_fingerprint: str,
    session_fingerprint: str,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "ordinal": ordinal,
        "request_fingerprint": request_fingerprint,
        "session_fingerprint": session_fingerprint,
    }


def _completion_record(
    *,
    ordinal: int,
    request_fingerprint: str,
    launch_fingerprint: str,
    attempt: ProposalAttempt,
    invocation: ProposerInvocation,
    write: AuthoringQuarantineWrite,
    bundle: StaticCandidateBundle | None,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "ordinal": ordinal,
        "request_fingerprint": request_fingerprint,
        "launch_fingerprint": launch_fingerprint,
        "attempt": attempt.to_record(),
        "attempt_fingerprint": attempt.fingerprint,
        "invocation": invocation.to_record(),
        "invocation_fingerprint": invocation.fingerprint,
        "quarantine_descriptor_sha256": write.descriptor_sha256,
        "quarantine_relative_path": write.relative_path,
        "static_candidate_bundle_fingerprint": (
            None if bundle is None else bundle.fingerprint
        ),
    }


def _load_completed_attempt(
    *,
    root: Path,
    attempt_dir: Path,
    ordinal: int,
    session_fingerprint: str,
    policy: AuthoringPolicy,
) -> tuple[dict[str, Any], AuthoringQuarantineEntry]:
    request = _parse_canonical(attempt_dir / _REQUEST)
    request_fingerprint = canonical_fingerprint(request)
    launch = _parse_canonical(attempt_dir / _LAUNCH)
    expected_launch = _launch_record(
        ordinal=ordinal,
        request_fingerprint=request_fingerprint,
        session_fingerprint=session_fingerprint,
    )
    if launch != expected_launch:
        raise AuthoringJobDriftError("Authoring launch marker drifted")
    completion = _parse_canonical(attempt_dir / _COMPLETION)
    expected_keys = {
        "schema_version", "ordinal", "request_fingerprint", "launch_fingerprint",
        "attempt", "attempt_fingerprint", "invocation", "invocation_fingerprint",
        "quarantine_descriptor_sha256", "quarantine_relative_path",
        "static_candidate_bundle_fingerprint",
    }
    if set(completion) != expected_keys or completion.get("schema_version") != 1:
        raise AuthoringJobDriftError("Authoring completion record shape drifted")
    if (
        completion["ordinal"] != ordinal
        or completion["request_fingerprint"] != request_fingerprint
        or completion["launch_fingerprint"] != canonical_fingerprint(launch)
    ):
        raise AuthoringJobDriftError("Authoring completion request binding drifted")
    attempt_fingerprint = _require_sha256(
        completion["attempt_fingerprint"], "$completion.attempt_fingerprint"
    )
    attempt = ProposalAttempt.from_record(completion["attempt"], policy=policy)
    if attempt.fingerprint != attempt_fingerprint:
        raise AuthoringJobDriftError("Authoring completion attempt fingerprint drifted")
    entry = load_authoring_quarantine(
        run_dir=root,
        intent_fingerprint=attempt.intent_fingerprint,
        ordinal=ordinal,
        attempt_fingerprint=attempt.fingerprint,
        policy=policy,
    )
    if (
        completion["invocation"] != entry.invocation.to_record()
        or completion["invocation_fingerprint"] != entry.invocation.fingerprint
        or completion["attempt"] != entry.attempt.to_record()
        or completion["quarantine_descriptor_sha256"] != entry.descriptor.fingerprint
        or completion["quarantine_relative_path"] != entry.relative_path
    ):
        raise AuthoringJobDriftError("Authoring completion quarantine binding drifted")
    expected_bundle = (
        None
        if entry.static_candidate_bundle is None
        else canonical_fingerprint(entry.static_candidate_bundle)
    )
    if completion["static_candidate_bundle_fingerprint"] != expected_bundle:
        raise AuthoringJobDriftError("Authoring completion candidate binding drifted")
    if (entry.attempt.status == "candidate") != (expected_bundle is not None):
        raise AuthoringJobDriftError("Authoring terminal candidate state drifted")
    if entry.attempt.status == "rejected" and ordinal != policy.max_attempts:
        raise AuthoringJobDriftError("Authoring job rejected before exhausting its budget")
    return request, entry


def _result_record(
    *,
    intent: AuthoringIntent,
    brief: PublicAuthoringBrief,
    command: _ProposerRuntimeCommand,
    execution_policy: ProposerExecutionPolicy,
    session: Any,
    attempts: list[ProposalAttempt],
    writes: list[AuthoringQuarantineWrite],
    bundle: StaticCandidateBundle | None,
) -> dict[str, Any]:
    from .authoring_orchestrator import AuthoringJobResult

    result = AuthoringJobResult(
        schema_version=1,
        intent_fingerprint=intent.fingerprint,
        brief_fingerprint=brief.fingerprint,
        command_fingerprint=command.fingerprint,
        execution_policy_fingerprint=execution_policy.fingerprint,
        session_fingerprint=session.fingerprint,
        attempt_fingerprints=tuple(item.fingerprint for item in attempts),
        quarantine_descriptor_sha256s=tuple(item.descriptor_sha256 for item in writes),
        final_state=("quarantined_static" if bundle is not None else "exhausted"),
        static_candidate_bundle_fingerprint=(None if bundle is None else bundle.fingerprint),
    )
    return result.to_record()


def _job_result_from_record(record: Mapping[str, Any]) -> Any:
    from .authoring_orchestrator import AuthoringJobResult

    expected = {
        "schema_version", "intent_fingerprint", "brief_fingerprint",
        "command_fingerprint", "execution_policy_fingerprint", "session_fingerprint",
        "attempt_fingerprints", "quarantine_descriptor_sha256s", "final_state",
        "static_candidate_bundle_fingerprint",
    }
    if set(record) != expected:
        raise AuthoringJobDriftError("Authoring result record shape drifted")
    result = AuthoringJobResult(
        schema_version=record["schema_version"],
        intent_fingerprint=record["intent_fingerprint"],
        brief_fingerprint=record["brief_fingerprint"],
        command_fingerprint=record["command_fingerprint"],
        execution_policy_fingerprint=record["execution_policy_fingerprint"],
        session_fingerprint=record["session_fingerprint"],
        attempt_fingerprints=tuple(record["attempt_fingerprints"]),
        quarantine_descriptor_sha256s=tuple(record["quarantine_descriptor_sha256s"]),
        final_state=record["final_state"],
        static_candidate_bundle_fingerprint=record["static_candidate_bundle_fingerprint"],
    )
    if result.to_record() != dict(record):
        raise AuthoringJobDriftError("Authoring result record is not canonical")
    return result


def run_durable_automatic_authoring(
    *,
    intent: AuthoringIntent,
    brief: PublicAuthoringBrief,
    curriculum_snapshot: CurriculumSnapshot,
    catalog: CapabilityCatalog,
    process_specs: Mapping[str, ProcessSpecV1],
    parent_task: TaskSpecV2 | None,
    parent_contract: CompiledTaskContract | None,
    command: _ProposerRuntimeCommand,
    model_id: str,
    quarantine_dir: Path,
    environment: Mapping[str, str] | None = None,
    template: Any,
    compiler_policy: TaskCompilerPolicy = DEFAULT_TASK_COMPILER_POLICY,
    property_policy: PropertyValidationPolicy = DEFAULT_PROPERTY_VALIDATION_POLICY,
    process_policy: ProcessCompilerPolicy = DEFAULT_PROCESS_COMPILER_POLICY,
    authoring_policy: AuthoringPolicy = DEFAULT_AUTHORING_POLICY,
    execution_policy: ProposerExecutionPolicy = DEFAULT_PROPOSER_EXECUTION_POLICY,
) -> Any:
    """Run or resume exactly one immutable authoring chain in ``quarantine_dir``."""
    from .authoring_orchestrator import (
        AuthoringJobRun,
        build_authoring_request,
        build_authoring_session,
        validate_authoring_attempt_chain,
    )

    supplied_environment = _validated_execution_environment(
        environment,
        execution_policy=execution_policy,
        authoring_policy=authoring_policy,
    )
    proposer_binding, session = build_authoring_session(
        intent=intent, brief=brief, catalog=catalog, process_specs=process_specs,
        command=command, model_id=model_id, template=template,
        compiler_policy=compiler_policy, property_policy=property_policy,
        process_policy=process_policy, authoring_policy=authoring_policy,
        execution_policy=execution_policy,
    )
    manifest = _manifest_record(
        intent=intent, brief=brief, curriculum_snapshot=curriculum_snapshot,
        catalog=catalog, process_specs=process_specs,
        parent_task=parent_task, parent_contract=parent_contract, command=command,
        model_id=model_id, proposer_binding=proposer_binding, session=session,
        environment=supplied_environment, template=template,
        compiler_policy=compiler_policy,
        property_policy=property_policy, process_policy=process_policy,
        authoring_policy=authoring_policy, execution_policy=execution_policy,
    )

    with _exclusive_job_lock(Path(quarantine_dir)) as root:
        job_dir = _checked_directory(root / "authoring-job", create=True)
        attempts_dir = _checked_directory(job_dir / "attempts", create=True)
        _validate_manifest(job_dir / _MANIFEST, manifest)
        _assert_job_layout(job_dir)
        _assert_attempt_layout(attempts_dir, maximum=authoring_policy.max_attempts)

        attempts: list[ProposalAttempt] = []
        invocations: list[ProposerInvocation] = []
        writes: list[AuthoringQuarantineWrite] = []
        entries: list[AuthoringQuarantineEntry] = []
        parent: ProposalAttempt | None = None
        next_ordinal = 1

        for ordinal in range(1, authoring_policy.max_attempts + 1):
            attempt_dir = attempts_dir / f"attempt-{ordinal:03d}"
            if not attempt_dir.exists():
                next_ordinal = ordinal
                break
            attempt_dir = _checked_directory(attempt_dir, create=False)
            request_path = attempt_dir / _REQUEST
            launch_path = attempt_dir / _LAUNCH
            completion_path = attempt_dir / _COMPLETION
            unknown = {
                item.name for item in attempt_dir.iterdir()
                if item.name not in {_REQUEST, _LAUNCH, _COMPLETION}
                and not (item.name.startswith(".") and ".partial-" in item.name)
            }
            if unknown:
                raise AuthoringJobDriftError("Authoring attempt contains unknown records")
            if not request_path.exists():
                raise AuthoringJobDriftError("Authoring attempt directory has no request")
            if launch_path.exists() and not completion_path.exists():
                raise AmbiguousAuthoringAttemptError(
                    f"Authoring attempt {ordinal} was launched without durable completion"
                )
            if completion_path.exists() and not launch_path.exists():
                raise AuthoringJobDriftError("Authoring completion has no launch marker")
            if not launch_path.exists():
                next_ordinal = ordinal
                break
            persisted_request, entry = _load_completed_attempt(
                root=root, attempt_dir=attempt_dir, ordinal=ordinal,
                session_fingerprint=session.fingerprint, policy=authoring_policy,
            )
            expected_request = build_authoring_request(
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
            if persisted_request != expected_request:
                raise AuthoringJobDriftError("Completed authoring request drifted")
            attempt = entry.attempt
            attempts.append(attempt)
            invocations.append(entry.invocation)
            entries.append(entry)
            writes.append(
                AuthoringQuarantineWrite(
                    intent_fingerprint=attempt.intent_fingerprint,
                    attempt_fingerprint=attempt.fingerprint,
                    ordinal=ordinal,
                    descriptor_sha256=entry.descriptor.fingerprint,
                    relative_path=entry.relative_path,
                    created=False,
                )
            )
            parent = attempt
            next_ordinal = ordinal + 1
            if attempt.status in {"candidate", "rejected"}:
                if any(
                    (attempts_dir / f"attempt-{later:03d}").exists()
                    for later in range(ordinal + 1, authoring_policy.max_attempts + 1)
                ):
                    raise AuthoringJobDriftError("Authoring journal continues past terminal state")
                break

        terminal = bool(attempts and attempts[-1].status in {"candidate", "rejected"})
        if attempts:
            validate_authoring_attempt_chain(
                intent=intent, session=session, proposer_binding=proposer_binding,
                attempts=tuple(attempts), invocations=tuple(invocations),
                authoring_policy=authoring_policy,
                require_terminal=terminal,
            )

        if terminal:
            candidate_record = entries[-1].static_candidate_bundle
            bundle = None
            if candidate_record is not None:
                bundle = StaticCandidateBundle.from_record(
                    candidate_record, intent=intent, brief=brief,
                    parent_task=parent_task, parent_contract=parent_contract,
                    attempt=attempts[-1], catalog=catalog, process_specs=process_specs,
                    compiler_policy=compiler_policy, process_policy=process_policy,
                    property_policy=property_policy, authoring_policy=authoring_policy,
                )
            expected_result = _result_record(
                intent=intent, brief=brief, command=command,
                execution_policy=execution_policy, session=session,
                attempts=attempts, writes=writes, bundle=bundle,
            )
            result_path = job_dir / _RESULT
            _publish_once(result_path, expected_result)
            actual_result = _job_result_from_record(_parse_canonical(result_path))
            if actual_result.to_record() != expected_result:
                raise AuthoringJobDriftError("Authoring terminal result drifted")
            return AuthoringJobRun(
                actual_result, tuple(attempts), tuple(invocations), tuple(writes), bundle
            )

        if next_ordinal > authoring_policy.max_attempts:
            raise AuthoringJobDriftError("Authoring attempt budget ended without terminal state")

        for ordinal in range(next_ordinal, authoring_policy.max_attempts + 1):
            request = build_authoring_request(
                intent=intent, brief=brief, catalog=catalog, process_specs=process_specs,
                parent_task=parent_task, parent_contract=parent_contract,
                session=session, ordinal=ordinal, parent=parent, template=template,
                compiler_policy=compiler_policy, property_policy=property_policy,
                process_policy=process_policy, authoring_policy=authoring_policy,
            )
            _validate_request_size(request, execution_policy=execution_policy)
            request_fingerprint = canonical_fingerprint(request)
            attempt_dir = _attempt_directory(attempts_dir, ordinal)
            request_path = attempt_dir / _REQUEST
            if request_path.exists() and _parse_canonical(request_path) != request:
                raise AuthoringJobDriftError("Persisted authoring request drifted")
            _publish_once(request_path, request)
            launch = _launch_record(
                ordinal=ordinal, request_fingerprint=request_fingerprint,
                session_fingerprint=session.fingerprint,
            )
            launch_path = attempt_dir / _LAUNCH
            if launch_path.exists():
                raise AmbiguousAuthoringAttemptError(
                    f"Authoring attempt {ordinal} has an unresolved launch marker"
                )
            _publish_once(launch_path, launch)

            proposer_run = run_proposer(
                command=command, request=request, environment=supplied_environment,
                execution_policy=execution_policy, authoring_policy=authoring_policy,
            )
            if proposer_run.invocation.request_sha256 != request_fingerprint:
                raise ValueError("Proposer invocation request binding mismatch")
            status = proposer_run.invocation.status
            if status == "completed":
                assessment = validate_static_candidate(
                    proposer_run.stdout, intent=intent, brief=brief,
                    parent_task=parent_task, parent_contract=parent_contract,
                    catalog=catalog, process_specs=process_specs,
                    compiler_policy=compiler_policy, property_policy=property_policy,
                    process_policy=process_policy, authoring_policy=authoring_policy,
                )
                issues = assessment.issues
            else:
                assessment = None
                code = (
                    "proposer_timeout" if status == "timeout" else
                    "response_too_large" if status == "stdout_limit" else
                    "proposer_failed"
                )
                issues = normalize_public_issues(((code, "$"),), policy=authoring_policy)
            accepted = assessment is not None and assessment.accepted
            attempt_status = (
                "candidate" if accepted else
                "repair_requested" if ordinal < authoring_policy.max_attempts else
                "rejected"
            )
            attempt = ProposalAttempt.record_response(
                proposer_run.stdout, intent=intent, proposer_id=command.command_id,
                model_id=model_id, template_fingerprint=template.fingerprint,
                request_fingerprint=request_fingerprint, status=attempt_status,
                issues=((issue.code, issue.path) for issue in issues), parent=parent,
                session_fingerprint=session.fingerprint,
                invocation_fingerprint=proposer_run.invocation.fingerprint,
                response_complete=not proposer_run.invocation.stdout_truncated,
                policy=authoring_policy,
            )
            bundle = assessment.to_bundle(attempt) if accepted else None
            write = write_authoring_quarantine(
                run_dir=root, raw_response=proposer_run.stdout, attempt=attempt,
                invocation=proposer_run.invocation, static_candidate_bundle=bundle,
                policy=authoring_policy,
            )
            completion = _completion_record(
                ordinal=ordinal, request_fingerprint=request_fingerprint,
                launch_fingerprint=canonical_fingerprint(launch), attempt=attempt,
                invocation=proposer_run.invocation, write=write, bundle=bundle,
            )
            _publish_once(attempt_dir / _COMPLETION, completion)
            attempts.append(attempt)
            invocations.append(proposer_run.invocation)
            writes.append(write)
            parent = attempt

            if attempt_status in {"candidate", "rejected"}:
                validate_authoring_attempt_chain(
                    intent=intent, session=session, proposer_binding=proposer_binding,
                    attempts=tuple(attempts), invocations=tuple(invocations),
                    authoring_policy=authoring_policy,
                )
                expected_result = _result_record(
                    intent=intent, brief=brief, command=command,
                    execution_policy=execution_policy, session=session,
                    attempts=attempts, writes=writes, bundle=bundle,
                )
                _publish_once(job_dir / _RESULT, expected_result)
                result = _job_result_from_record(expected_result)
                return AuthoringJobRun(
                    result, tuple(attempts), tuple(invocations), tuple(writes), bundle
                )

    raise AssertionError("Durable authoring loop did not terminate")
