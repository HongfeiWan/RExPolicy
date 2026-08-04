"""Durable single-head storage for authored-candidate lifecycle events.

The admission state machine validates transitions in memory.  This module adds
the persistence boundary: an immutable manifest, a contiguous append-only
journal, compare-and-swap (CAS) head updates, detached authorization
signatures, and an independently stored monotonic anchor.

The anchor must live in a separately administered persistence domain.  No
scheme confined to one administrator-writable directory can detect that
administrator rolling the entire directory back to a previously valid copy.
Directory ownership, mount integrity, and ancestor replacement are deployment
controls; this module validates final path components but is not a dirfd-based
filesystem sandbox.
"""

from __future__ import annotations

import base64
import contextlib
import fcntl
import hashlib
import os
import re
import stat
import uuid
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from .authoring_admission import (
    ACTIVE,
    ADMITTED_DORMANT,
    QUARANTINED_STATIC,
    CandidateArtifactBinding,
    LifecycleEvent,
)
from .canonical import canonical_fingerprint, canonical_json, strict_json_loads

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_IDENTIFIER = re.compile(r"^[a-z][a-z0-9_.-]*(?:/[a-z0-9_.-]+)*$")
_MAX_RECORD_BYTES = 4_000_000
_MANIFEST_NAME = "manifest.json"
_HEAD_NAME = "head.json"
_JOURNAL_NAME = "journal"
_LOCK_NAME = ".lifecycle.lock"
_REQUIRED_EVIDENCE = frozenset(
    {
        "runtime_certification",
        "sealed_audit_certification",
        "runtime_policy",
        "sealed_audit_suite",
        "capability_claim_policy",
        "capability_claim_report",
        "episode_results",
    }
)
_SIGNATURE_PURPOSES = frozenset({"lifecycle_authority", "generation_boundary"})


class LifecycleStoreError(ValueError):
    """Persisted lifecycle state is invalid, forked, stale, or tampered."""


class LifecycleCASMismatch(LifecycleStoreError):
    """The caller did not present the store's current head."""


class LifecycleRollbackDetected(LifecycleStoreError):
    """The external anchor proves that local lifecycle state went backwards."""


class DetachedSignatureVerifier(Protocol):
    """Trust-boundary protocol; implementations must fail closed."""

    def verify(self, *, key_id: str, payload: bytes, signature: bytes) -> bool:
        """Return true only for a valid signature from the named trusted key."""


class LifecycleAnchorStore(Protocol):
    """Independently administered monotonic watermark storage."""

    def read(self) -> "LifecycleAnchor | None": ...

    def compare_and_set(
        self,
        *,
        expected: "LifecycleAnchor | None",
        replacement: "LifecycleAnchor",
    ) -> None: ...


def _mapping(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise LifecycleStoreError(f"{path} must be an object")
    return dict(value)


def _keys(record: Mapping[str, Any], expected: set[str], path: str) -> None:
    missing = sorted(expected.difference(record))
    unknown = sorted(set(record).difference(expected))
    if missing:
        raise LifecycleStoreError(f"{path} is missing fields: {', '.join(missing)}")
    if unknown:
        raise LifecycleStoreError(f"{path} has unknown fields: {', '.join(unknown)}")


def _sha256(value: Any, path: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise LifecycleStoreError(f"{path} must be a lowercase SHA-256")
    return value


def _identifier(value: Any, path: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise LifecycleStoreError(f"{path} must be an identifier")
    return value


def _canonical_payload(record: Mapping[str, Any]) -> bytes:
    return canonical_json(dict(record)).encode("ascii")


def _parse_record(path: Path) -> dict[str, Any]:
    try:
        mode = path.lstat().st_mode
    except OSError as error:
        raise LifecycleStoreError(f"Cannot inspect lifecycle record {path.name}") from error
    if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
        raise LifecycleStoreError(f"Lifecycle record {path.name} must be a real file")
    try:
        payload = path.read_bytes()
    except OSError as error:
        raise LifecycleStoreError(f"Cannot read lifecycle record {path.name}") from error
    if len(payload) > _MAX_RECORD_BYTES:
        raise LifecycleStoreError(f"Lifecycle record {path.name} is too large")
    try:
        record = strict_json_loads(payload.decode("ascii"), max_bytes=_MAX_RECORD_BYTES)
    except (TypeError, UnicodeError, ValueError) as error:
        raise LifecycleStoreError(
            f"Lifecycle record {path.name} is not strict JSON"
        ) from error
    if not isinstance(record, dict) or payload != _canonical_payload(record):
        raise LifecycleStoreError(f"Lifecycle record {path.name} is not canonical")
    return record


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_new(
    path: Path,
    record: Mapping[str, Any],
    *,
    staging_directory: Path | None = None,
) -> None:
    payload = _canonical_payload(record)
    if len(payload) > _MAX_RECORD_BYTES:
        raise LifecycleStoreError("Lifecycle record exceeds the trusted size limit")
    staging = path.parent if staging_directory is None else staging_directory
    temporary = staging / f".{path.name}.partial-{uuid.uuid4().hex}"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(temporary, flags, 0o600)
        try:
            view = memoryview(payload)
            while view:
                count = os.write(descriptor, view)
                if count <= 0:
                    raise OSError("short lifecycle record write")
                view = view[count:]
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        try:
            os.link(temporary, path, follow_symlinks=False)
        except FileExistsError as error:
            raise LifecycleStoreError(
                f"Immutable lifecycle record {path.name} exists"
            ) from error
        _fsync_directory(path.parent)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _replace(path: Path, record: Mapping[str, Any]) -> None:
    payload = _canonical_payload(record)
    temporary = path.parent / f".{path.name}.partial-{uuid.uuid4().hex}"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(temporary, flags, 0o600)
    try:
        view = memoryview(payload)
        while view:
            count = os.write(descriptor, view)
            if count <= 0:
                raise OSError("short lifecycle record write")
            view = view[count:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    try:
        os.replace(temporary, path)
        _fsync_directory(path.parent)
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
        raise LifecycleStoreError("Lifecycle directory is unavailable") from error
    if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
        raise LifecycleStoreError("Lifecycle path must be a real directory")
    return path.resolve(strict=True)


@contextlib.contextmanager
def _exclusive_lock(root: Path, name: str = _LOCK_NAME) -> Iterator[None]:
    lock_path = root / name
    flags = os.O_RDWR | os.O_CREAT
    flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(lock_path, flags, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


@dataclass(frozen=True)
class LifecycleStoreManifest:
    schema_version: int
    store_id: str
    candidate_fingerprint: str
    audit_plan_fingerprint: str
    authoring_intent_fingerprint: str
    evidence_fingerprints: tuple[tuple[str, str], ...]
    trusted_signers: tuple[tuple[str, tuple[str, ...]], ...]

    @classmethod
    def issue(
        cls,
        *,
        store_id: str,
        candidate: Any,
        audit_plan_fingerprint: str,
        authoring_intent_fingerprint: str,
        evidence_fingerprints: Mapping[str, str],
        trusted_signers: Mapping[str, Sequence[str]],
    ) -> "LifecycleStoreManifest":
        return cls.from_record(
            {
                "schema_version": 1,
                "artifact_type": "rexpolicy_authoring_lifecycle_store/v1",
                "store_id": store_id,
                "candidate_fingerprint": CandidateArtifactBinding.from_candidate(
                    candidate
                ).candidate_fingerprint,
                "audit_plan_fingerprint": audit_plan_fingerprint,
                "authoring_intent_fingerprint": authoring_intent_fingerprint,
                "evidence_fingerprints": dict(evidence_fingerprints),
                "trusted_signers": {
                    purpose: list(key_ids)
                    for purpose, key_ids in trusted_signers.items()
                },
            }
        )

    @classmethod
    def from_record(cls, value: Any) -> "LifecycleStoreManifest":
        path = "$lifecycle_store_manifest"
        record = _mapping(value, path)
        _keys(
            record,
            {
                "schema_version",
                "artifact_type",
                "store_id",
                "candidate_fingerprint",
                "audit_plan_fingerprint",
                "authoring_intent_fingerprint",
                "evidence_fingerprints",
                "trusted_signers",
            },
            path,
        )
        if record["schema_version"] != 1 or type(record["schema_version"]) is not int:
            raise LifecycleStoreError("Lifecycle manifest schema_version must be 1")
        if record["artifact_type"] != "rexpolicy_authoring_lifecycle_store/v1":
            raise LifecycleStoreError("Lifecycle manifest artifact_type is invalid")
        evidence = _mapping(record["evidence_fingerprints"], f"{path}.evidence_fingerprints")
        if set(evidence) != _REQUIRED_EVIDENCE:
            raise LifecycleStoreError("Lifecycle manifest evidence set is incomplete")
        evidence_items = tuple(
            (name, _sha256(evidence[name], f"{path}.evidence_fingerprints.{name}"))
            for name in sorted(evidence)
        )
        signers = _mapping(record["trusted_signers"], f"{path}.trusted_signers")
        if set(signers) != _SIGNATURE_PURPOSES:
            raise LifecycleStoreError("Lifecycle manifest signer purposes are incomplete")
        signer_items: list[tuple[str, tuple[str, ...]]] = []
        for purpose in sorted(signers):
            values = signers[purpose]
            if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
                raise LifecycleStoreError("Trusted signer IDs must be a sequence")
            key_ids = tuple(_identifier(item, f"{path}.trusted_signers.{purpose}") for item in values)
            if not key_ids or tuple(sorted(set(key_ids))) != key_ids:
                raise LifecycleStoreError("Trusted signer IDs must be non-empty, unique, and sorted")
            signer_items.append((purpose, key_ids))
        signer_policy = dict(signer_items)
        if set(signer_policy["lifecycle_authority"]).intersection(
            signer_policy["generation_boundary"]
        ):
            raise LifecycleStoreError(
                "Lifecycle authority and generation-boundary signers must be disjoint"
            )
        return cls(
            schema_version=1,
            store_id=_identifier(record["store_id"], f"{path}.store_id"),
            candidate_fingerprint=_sha256(record["candidate_fingerprint"], f"{path}.candidate_fingerprint"),
            audit_plan_fingerprint=_sha256(record["audit_plan_fingerprint"], f"{path}.audit_plan_fingerprint"),
            authoring_intent_fingerprint=_sha256(record["authoring_intent_fingerprint"], f"{path}.authoring_intent_fingerprint"),
            evidence_fingerprints=evidence_items,
            trusted_signers=tuple(signer_items),
        )

    @property
    def evidence(self) -> dict[str, str]:
        return dict(self.evidence_fingerprints)

    @property
    def signer_policy(self) -> dict[str, tuple[str, ...]]:
        return dict(self.trusted_signers)

    def to_record(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "artifact_type": "rexpolicy_authoring_lifecycle_store/v1",
            "store_id": self.store_id,
            "candidate_fingerprint": self.candidate_fingerprint,
            "audit_plan_fingerprint": self.audit_plan_fingerprint,
            "authoring_intent_fingerprint": self.authoring_intent_fingerprint,
            "evidence_fingerprints": dict(self.evidence_fingerprints),
            "trusted_signers": {
                purpose: list(key_ids) for purpose, key_ids in self.trusted_signers
            },
        }

    @property
    def fingerprint(self) -> str:
        return canonical_fingerprint(self.to_record())


@dataclass(frozen=True)
class DetachedSignature:
    schema_version: int
    purpose: str
    key_id: str
    payload_sha256: str
    signature_base64: str

    @classmethod
    def from_record(cls, value: Any) -> "DetachedSignature":
        path = "$detached_signature"
        record = _mapping(value, path)
        _keys(record, {"schema_version", "purpose", "key_id", "payload_sha256", "signature_base64"}, path)
        if record["schema_version"] != 1 or type(record["schema_version"]) is not int:
            raise LifecycleStoreError("Detached signature schema_version must be 1")
        purpose = record["purpose"]
        if purpose not in _SIGNATURE_PURPOSES:
            raise LifecycleStoreError("Detached signature purpose is invalid")
        encoded = record["signature_base64"]
        if not isinstance(encoded, str):
            raise LifecycleStoreError("Detached signature must be base64 text")
        try:
            base64.b64decode(encoded, validate=True)
        except (ValueError, base64.binascii.Error) as error:
            raise LifecycleStoreError("Detached signature is not canonical base64") from error
        return cls(1, purpose, _identifier(record["key_id"], f"{path}.key_id"), _sha256(record["payload_sha256"], f"{path}.payload_sha256"), encoded)

    @classmethod
    def encode(cls, *, purpose: str, key_id: str, payload: bytes, signature: bytes) -> "DetachedSignature":
        return cls.from_record(
            {
                "schema_version": 1,
                "purpose": purpose,
                "key_id": key_id,
                "payload_sha256": hashlib.sha256(payload).hexdigest(),
                "signature_base64": base64.b64encode(signature).decode("ascii"),
            }
        )

    def to_record(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "purpose": self.purpose,
            "key_id": self.key_id,
            "payload_sha256": self.payload_sha256,
            "signature_base64": self.signature_base64,
        }

    @property
    def signature(self) -> bytes:
        return base64.b64decode(self.signature_base64, validate=True)


@dataclass(frozen=True)
class LifecycleAnchor:
    schema_version: int
    manifest_fingerprint: str
    sequence: int
    head_fingerprint: str

    @classmethod
    def from_record(cls, value: Any) -> "LifecycleAnchor":
        path = "$lifecycle_anchor"
        record = _mapping(value, path)
        _keys(record, {"schema_version", "manifest_fingerprint", "sequence", "head_fingerprint"}, path)
        if record["schema_version"] != 1 or type(record["schema_version"]) is not int:
            raise LifecycleStoreError("Lifecycle anchor schema_version must be 1")
        sequence = record["sequence"]
        if type(sequence) is not int or sequence < 0:
            raise LifecycleStoreError("Lifecycle anchor sequence must be nonnegative")
        return cls(1, _sha256(record["manifest_fingerprint"], f"{path}.manifest_fingerprint"), sequence, _sha256(record["head_fingerprint"], f"{path}.head_fingerprint"))

    def to_record(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "manifest_fingerprint": self.manifest_fingerprint,
            "sequence": self.sequence,
            "head_fingerprint": self.head_fingerprint,
        }


@dataclass(frozen=True)
class LifecycleJournalEntry:
    schema_version: int
    manifest_fingerprint: str
    sequence: int
    previous_entry_sha256: str | None
    event: LifecycleEvent
    signatures: tuple[DetachedSignature, ...]

    @property
    def fingerprint(self) -> str:
        return canonical_fingerprint(self.to_record())

    def to_record(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "manifest_fingerprint": self.manifest_fingerprint,
            "sequence": self.sequence,
            "previous_entry_sha256": self.previous_entry_sha256,
            "lifecycle_event": self.event.to_record(),
            "lifecycle_event_fingerprint": self.event.fingerprint,
            "signatures": [item.to_record() for item in self.signatures],
        }


@dataclass(frozen=True)
class LifecycleStoreSnapshot:
    manifest: LifecycleStoreManifest
    entries: tuple[LifecycleJournalEntry, ...]

    @property
    def head_fingerprint(self) -> str | None:
        return None if not self.entries else self.entries[-1].fingerprint


class Ed25519DetachedVerifier:
    """Optional ``cryptography`` adapter; importing this module stays stdlib-only."""

    def __init__(self, public_keys: Mapping[str, bytes]) -> None:
        self._public_keys = dict(public_keys)

    def verify(self, *, key_id: str, payload: bytes, signature: bytes) -> bool:
        key_bytes = self._public_keys.get(key_id)
        if key_bytes is None:
            return False
        try:
            from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

            Ed25519PublicKey.from_public_bytes(key_bytes).verify(signature, payload)
            return True
        except (ImportError, ValueError, TypeError):
            return False
        except Exception as error:
            # InvalidSignature deliberately remains fail-closed without making
            # cryptography a module import dependency.
            if error.__class__.__name__ == "InvalidSignature":
                return False
            return False


class AdmissionLifecycleReplay:
    """Replay adapter that delegates every record to the admission state machine.

    All evidence is fixed when this adapter is constructed.  The store manifest
    separately commits and cross-checks the candidate, plan, intent, and all
    seven evidence fingerprints before every initialize, load, append, or
    recovery operation.
    """

    def __init__(
        self,
        *,
        candidate: Any,
        plan: Any,
        intent: Any,
        verification_context: Any,
        runtime_certification: Any,
        sealed_audit_certification: Any,
        runtime_policy_id: str,
        runtime_policy_fingerprint: str,
        suite: Any,
        policy: Any,
        report: Any,
        episode_results: Sequence[Any],
    ) -> None:
        self._candidate = candidate
        self._plan = plan
        self._intent = intent
        self._verification_context = verification_context
        self._evidence = {
            "runtime_certification": runtime_certification,
            "sealed_audit_certification": sealed_audit_certification,
            "runtime_policy_id": runtime_policy_id,
            "runtime_policy_fingerprint": runtime_policy_fingerprint,
            "suite": suite,
            "policy": policy,
            "report": report,
            "episode_results": tuple(episode_results),
        }
        self._candidate_fingerprint = CandidateArtifactBinding.from_candidate(
            candidate
        ).candidate_fingerprint
        self._audit_plan_fingerprint = self._record_fingerprint(
            plan,
            "$lifecycle_replay.audit_plan",
        )
        self._authoring_intent_fingerprint = self._record_fingerprint(
            intent,
            "$lifecycle_replay.authoring_intent",
        )
        self._evidence_fingerprints = {
            "runtime_certification": self._record_fingerprint(
                runtime_certification,
                "$lifecycle_replay.runtime_certification",
            ),
            "sealed_audit_certification": self._record_fingerprint(
                sealed_audit_certification,
                "$lifecycle_replay.sealed_audit_certification",
            ),
            "runtime_policy": _sha256(
                runtime_policy_fingerprint,
                "$lifecycle_replay.runtime_policy_fingerprint",
            ),
            "sealed_audit_suite": self._record_fingerprint(
                suite,
                "$lifecycle_replay.sealed_audit_suite",
            ),
            "capability_claim_policy": self._record_fingerprint(
                policy,
                "$lifecycle_replay.capability_claim_policy",
            ),
            "capability_claim_report": self._record_fingerprint(
                report,
                "$lifecycle_replay.capability_claim_report",
            ),
            "episode_results": canonical_fingerprint(
                [
                    self._record(item, "$lifecycle_replay.episode_results")
                    for item in self._evidence["episode_results"]
                ]
            ),
        }

    @staticmethod
    def _record(value: Any, path: str) -> dict[str, Any]:
        method = getattr(value, "to_record", None)
        if not callable(method):
            raise LifecycleStoreError(f"{path} must expose to_record()")
        return _mapping(method(), path)

    @classmethod
    def _record_fingerprint(cls, value: Any, path: str) -> str:
        record = cls._record(value, path)
        fingerprint = canonical_fingerprint(record)
        advertised = getattr(value, "fingerprint", fingerprint)
        if advertised != fingerprint:
            raise LifecycleStoreError(f"{path} advertised fingerprint is invalid")
        return fingerprint

    @property
    def candidate_fingerprint(self) -> str:
        return self._candidate_fingerprint

    @property
    def audit_plan_fingerprint(self) -> str:
        return self._audit_plan_fingerprint

    @property
    def authoring_intent_fingerprint(self) -> str:
        return self._authoring_intent_fingerprint

    @property
    def evidence_fingerprints(self) -> dict[str, str]:
        return dict(self._evidence_fingerprints)

    def __call__(
        self,
        records: Sequence[Mapping[str, Any]],
    ) -> tuple[LifecycleEvent, ...]:
        events: list[LifecycleEvent] = []
        for value in records:
            record = _mapping(value, "$lifecycle_replay.event")
            evidence = (
                {}
                if record.get("state") == QUARANTINED_STATIC
                else self._evidence
            )
            events.append(
                LifecycleEvent.from_record(
                    record,
                    candidate=self._candidate,
                    plan=self._plan,
                    intent=self._intent,
                    verification_context=self._verification_context,
                    history=tuple(events),
                    **evidence,
                )
            )
        return tuple(events)


class FileLifecycleAnchor:
    """A file-backed anchor; ``path`` must be outside the lifecycle root."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    @contextlib.contextmanager
    def _lock(self) -> Iterator[None]:
        parent = _checked_directory(self.path.parent, create=True)
        with _exclusive_lock(parent, f".{self.path.name}.lock"):
            yield

    def read(self) -> LifecycleAnchor | None:
        with self._lock():
            return self._read_unlocked()

    def _read_unlocked(self) -> LifecycleAnchor | None:
        if not self.path.exists():
            return None
        if self.path.is_symlink() or not self.path.is_file():
            raise LifecycleStoreError("Lifecycle anchor must be a real file")
        return LifecycleAnchor.from_record(_parse_record(self.path))

    def compare_and_set(self, *, expected: LifecycleAnchor | None, replacement: LifecycleAnchor) -> None:
        with self._lock():
            current = self._read_unlocked()
            if current != expected:
                raise LifecycleCASMismatch("External lifecycle anchor CAS is stale")
            if current is not None:
                if replacement.manifest_fingerprint != current.manifest_fingerprint:
                    raise LifecycleRollbackDetected("Lifecycle anchor manifest changed")
                if replacement.sequence != current.sequence + 1:
                    raise LifecycleRollbackDetected("Lifecycle anchor must advance by one")
            elif replacement.sequence != 0:
                raise LifecycleRollbackDetected(
                    "Initial lifecycle anchor must begin at sequence zero"
                )
            _replace(self.path, replacement.to_record())


class AuthoringLifecycleStore:
    """Single-writer-by-lock, single-head-by-CAS lifecycle journal.

    A file anchor is rejected when it is inside ``root``.  A custom anchor
    implementation cannot be located mechanically; deploying it in an
    independently administered persistence domain remains a trust prerequisite.
    """

    def __init__(
        self,
        root: Path,
        *,
        replay: Callable[[Sequence[Mapping[str, Any]]], Sequence[LifecycleEvent]],
        signature_verifier: DetachedSignatureVerifier,
        anchor_store: LifecycleAnchorStore,
    ) -> None:
        self.root = Path(root)
        self._replay = replay
        self._signature_verifier = signature_verifier
        self._anchor_store = anchor_store

    def initialize(self, manifest: LifecycleStoreManifest) -> LifecycleStoreSnapshot:
        self._verify_replay_context(manifest)
        root = _checked_directory(self.root, create=True)
        self._verify_anchor_location(root)
        with _exclusive_lock(root):
            journal = root / _JOURNAL_NAME
            journal.mkdir(exist_ok=True)
            _checked_directory(journal, create=False)
            path = root / _MANIFEST_NAME
            if path.exists():
                existing = LifecycleStoreManifest.from_record(_parse_record(path))
                if existing != manifest:
                    raise LifecycleStoreError("Immutable lifecycle manifest drifted")
            else:
                _write_new(path, manifest.to_record())
            return self._load_locked(root, manifest)

    def load(self, manifest: LifecycleStoreManifest) -> LifecycleStoreSnapshot:
        self._verify_replay_context(manifest)
        root = _checked_directory(self.root, create=False)
        self._verify_anchor_location(root)
        with _exclusive_lock(root):
            return self._load_locked(root, manifest)

    def signing_payload(
        self,
        *,
        manifest: LifecycleStoreManifest,
        event: LifecycleEvent,
        expected_head: str | None,
        purpose: str,
        key_id: str,
    ) -> bytes:
        if purpose not in _SIGNATURE_PURPOSES:
            raise LifecycleStoreError("Signature purpose is invalid")
        canonical_key_id = _identifier(key_id, "$signing_payload.key_id")
        record = {
            "schema_version": 1,
            "artifact_type": "rexpolicy_lifecycle_transition_authorization/v1",
            "purpose": purpose,
            "key_id": canonical_key_id,
            "store_id": manifest.store_id,
            "manifest_fingerprint": manifest.fingerprint,
            "candidate_fingerprint": manifest.candidate_fingerprint,
            "audit_plan_fingerprint": manifest.audit_plan_fingerprint,
            "authoring_intent_fingerprint": manifest.authoring_intent_fingerprint,
            "evidence_fingerprints": manifest.evidence,
            "sequence": event.sequence,
            "expected_head_sha256": expected_head,
            "lifecycle_event_fingerprint": event.fingerprint,
            "state": event.state,
        }
        return _canonical_payload(record)

    def append(
        self,
        *,
        manifest: LifecycleStoreManifest,
        event: LifecycleEvent,
        expected_head: str | None,
        signatures: Sequence[DetachedSignature] = (),
    ) -> LifecycleStoreSnapshot:
        self._verify_replay_context(manifest)
        if expected_head is not None:
            _sha256(expected_head, "expected_head")
        root = _checked_directory(self.root, create=False)
        self._verify_anchor_location(root)
        with _exclusive_lock(root):
            snapshot = self._load_locked(root, manifest)
            if snapshot.head_fingerprint != expected_head:
                raise LifecycleCASMismatch("Lifecycle head CAS is stale")
            previous = snapshot.entries[-1] if snapshot.entries else None
            if event.sequence != len(snapshot.entries):
                raise LifecycleStoreError("Lifecycle event sequence is not contiguous")
            if event.candidate_fingerprint != manifest.candidate_fingerprint or event.audit_plan_fingerprint != manifest.audit_plan_fingerprint:
                raise LifecycleStoreError("Lifecycle event crosses manifest bindings")
            supplied = tuple(signatures)
            self._verify_signatures(manifest, event, expected_head, supplied)
            entry = LifecycleJournalEntry(
                1,
                manifest.fingerprint,
                event.sequence,
                None if previous is None else previous.fingerprint,
                event,
                supplied,
            )
            # Replay the complete prospective chain before any durable write.
            prospective = snapshot.entries + (entry,)
            self._replay_and_check(manifest, prospective)
            journal_path = root / _JOURNAL_NAME / f"{event.sequence:020d}.json"
            _write_new(
                journal_path,
                entry.to_record(),
                staging_directory=root,
            )
            anchor = LifecycleAnchor(1, manifest.fingerprint, event.sequence, entry.fingerprint)
            old_anchor = None if previous is None else LifecycleAnchor(1, manifest.fingerprint, previous.sequence, previous.fingerprint)
            self._anchor_store.compare_and_set(expected=old_anchor, replacement=anchor)
            _replace(root / _HEAD_NAME, anchor.to_record())
            return LifecycleStoreSnapshot(manifest, prospective)

    def recover_pending_tail(
        self,
        *,
        manifest: LifecycleStoreManifest,
    ) -> LifecycleStoreSnapshot:
        """Anchor one fully verified pending record after an interrupted append.

        This is intentionally explicit.  Normal reads reject every unanchored
        tail.  Recovery accepts exactly one contiguous record whose predecessor
        is still the independent anchor, then performs the same anchor CAS and
        local-head publication as append.  Calling it after completion is
        idempotent.
        """
        self._verify_replay_context(manifest)
        root = _checked_directory(self.root, create=False)
        self._verify_anchor_location(root)
        with _exclusive_lock(root):
            snapshot = self._load_locked(
                root,
                manifest,
                allow_pending_tail=True,
            )
            if not snapshot.entries:
                return snapshot
            tail = snapshot.entries[-1]
            replacement = LifecycleAnchor(
                1,
                manifest.fingerprint,
                tail.sequence,
                tail.fingerprint,
            )
            current = self._anchor_store.read()
            if current == replacement:
                _replace(root / _HEAD_NAME, replacement.to_record())
                return snapshot
            if len(snapshot.entries) == 1:
                predecessor = None
            else:
                previous = snapshot.entries[-2]
                predecessor = LifecycleAnchor(
                    1,
                    manifest.fingerprint,
                    previous.sequence,
                    previous.fingerprint,
                )
            if current != predecessor:
                raise LifecycleRollbackDetected(
                    "Pending lifecycle tail does not extend the external anchor"
                )
            self._anchor_store.compare_and_set(
                expected=predecessor,
                replacement=replacement,
            )
            _replace(root / _HEAD_NAME, replacement.to_record())
            return snapshot

    def _verify_replay_context(self, manifest: LifecycleStoreManifest) -> None:
        try:
            candidate = self._replay.candidate_fingerprint
            plan = self._replay.audit_plan_fingerprint
            intent = self._replay.authoring_intent_fingerprint
            evidence = self._replay.evidence_fingerprints
        except AttributeError as error:
            raise LifecycleStoreError(
                "Lifecycle replay context does not expose exact manifest bindings"
            ) from error
        if not isinstance(evidence, Mapping):
            raise LifecycleStoreError(
                "Lifecycle replay evidence fingerprints must be an object"
            )
        canonical_evidence = {
            name: _sha256(value, f"$lifecycle_replay.evidence_fingerprints.{name}")
            for name, value in evidence.items()
        }
        if set(canonical_evidence) != _REQUIRED_EVIDENCE:
            raise LifecycleStoreError(
                "Lifecycle replay evidence fingerprint set is incomplete"
            )
        if (
            _sha256(candidate, "$lifecycle_replay.candidate_fingerprint")
            != manifest.candidate_fingerprint
            or _sha256(plan, "$lifecycle_replay.audit_plan_fingerprint")
            != manifest.audit_plan_fingerprint
            or _sha256(intent, "$lifecycle_replay.authoring_intent_fingerprint")
            != manifest.authoring_intent_fingerprint
            or canonical_evidence != manifest.evidence
        ):
            raise LifecycleStoreError(
                "Lifecycle replay context differs from the immutable manifest"
            )

    def _verify_anchor_location(self, root: Path) -> None:
        if not isinstance(self._anchor_store, FileLifecycleAnchor):
            return
        try:
            anchor_path = self._anchor_store.path.resolve(strict=False)
        except OSError as error:
            raise LifecycleStoreError(
                "Cannot resolve the lifecycle anchor location"
            ) from error
        if anchor_path == root or root in anchor_path.parents:
            raise LifecycleStoreError(
                "File lifecycle anchor must be outside the lifecycle root"
            )

    def _verify_signatures(
        self,
        manifest: LifecycleStoreManifest,
        event: LifecycleEvent,
        expected_head: str | None,
        signatures: tuple[DetachedSignature, ...],
    ) -> None:
        required: tuple[str, ...] = ()
        if event.state == ADMITTED_DORMANT:
            required = ("lifecycle_authority",)
        elif event.state == ACTIVE:
            required = ("lifecycle_authority", "generation_boundary")
        if tuple(item.purpose for item in signatures) != required:
            raise LifecycleStoreError("Lifecycle transition signature set is not exact")
        policy = manifest.signer_policy
        seen: set[tuple[str, str]] = set()
        for item in signatures:
            if (item.purpose, item.key_id) in seen:
                raise LifecycleStoreError("Duplicate lifecycle signature")
            seen.add((item.purpose, item.key_id))
            if item.key_id not in policy[item.purpose]:
                raise LifecycleStoreError("Lifecycle signature key is not trusted")
            payload = self.signing_payload(
                manifest=manifest,
                event=event,
                expected_head=expected_head,
                purpose=item.purpose,
                key_id=item.key_id,
            )
            if hashlib.sha256(payload).hexdigest() != item.payload_sha256:
                raise LifecycleStoreError("Lifecycle signature payload binding mismatch")
            verified = self._signature_verifier.verify(
                key_id=item.key_id,
                payload=payload,
                signature=item.signature,
            )
            if verified is not True:
                raise LifecycleStoreError("Lifecycle detached signature verification failed")

    def _load_locked(
        self,
        root: Path,
        expected_manifest: LifecycleStoreManifest,
        *,
        allow_pending_tail: bool = False,
    ) -> LifecycleStoreSnapshot:
        path = root / _MANIFEST_NAME
        if not path.exists():
            raise LifecycleStoreError("Lifecycle manifest is missing")
        manifest = LifecycleStoreManifest.from_record(_parse_record(path))
        if manifest != expected_manifest:
            raise LifecycleStoreError("Lifecycle manifest does not match requested bindings")
        journal = _checked_directory(root / _JOURNAL_NAME, create=False)
        files = sorted(journal.iterdir())
        expected_names = [f"{index:020d}.json" for index in range(len(files))]
        if [item.name for item in files] != expected_names or any(item.is_symlink() or not item.is_file() for item in files):
            raise LifecycleStoreError("Lifecycle journal is not a contiguous immutable sequence")
        raw_entries: list[tuple[dict[str, Any], tuple[DetachedSignature, ...]]] = []
        previous_fingerprint: str | None = None
        for index, item in enumerate(files):
            record = _parse_record(item)
            path = "$lifecycle_journal_entry"
            _keys(record, {"schema_version", "manifest_fingerprint", "sequence", "previous_entry_sha256", "lifecycle_event", "lifecycle_event_fingerprint", "signatures"}, path)
            if record["schema_version"] != 1 or type(record["schema_version"]) is not int:
                raise LifecycleStoreError("Lifecycle journal schema_version must be 1")
            if record["manifest_fingerprint"] != manifest.fingerprint:
                raise LifecycleStoreError("Lifecycle journal manifest binding mismatch")
            if record["sequence"] != index or type(record["sequence"]) is not int:
                raise LifecycleStoreError("Lifecycle journal sequence mismatch")
            if record["previous_entry_sha256"] != previous_fingerprint:
                raise LifecycleStoreError("Lifecycle journal predecessor mismatch")
            event_record = _mapping(record["lifecycle_event"], f"{path}.lifecycle_event")
            if canonical_fingerprint(event_record) != record["lifecycle_event_fingerprint"]:
                raise LifecycleStoreError("Lifecycle event fingerprint mismatch")
            signature_records = record["signatures"]
            if isinstance(signature_records, (str, bytes)) or not isinstance(signature_records, Sequence):
                raise LifecycleStoreError("Lifecycle signatures must be a sequence")
            signatures = tuple(DetachedSignature.from_record(value) for value in signature_records)
            raw_entries.append((event_record, signatures))
            previous_fingerprint = canonical_fingerprint(record)
        event_records = [record for record, _ in raw_entries]
        try:
            restored = tuple(self._replay(event_records))
        except (TypeError, ValueError) as error:
            raise LifecycleStoreError("Lifecycle journal replay failed") from error
        if len(restored) != len(raw_entries) or any(not isinstance(event, LifecycleEvent) for event in restored):
            raise LifecycleStoreError("Lifecycle replay did not return the complete event chain")
        entries: list[LifecycleJournalEntry] = []
        for index, (event, (_, signatures)) in enumerate(zip(restored, raw_entries)):
            previous = entries[-1] if entries else None
            entry = LifecycleJournalEntry(1, manifest.fingerprint, index, None if previous is None else previous.fingerprint, event, signatures)
            if entry.fingerprint != canonical_fingerprint(_parse_record(files[index])):
                raise LifecycleStoreError("Lifecycle replay changed a journal event")
            entries.append(entry)
        self._replay_and_check(manifest, tuple(entries))
        anchor = self._anchor_store.read()
        if entries:
            expected_anchor = LifecycleAnchor(
                1,
                manifest.fingerprint,
                entries[-1].sequence,
                entries[-1].fingerprint,
            )
            if anchor is None:
                if not (allow_pending_tail and len(entries) == 1):
                    raise LifecycleRollbackDetected(
                        "External lifecycle anchor is missing"
                    )
                return LifecycleStoreSnapshot(manifest, tuple(entries))
            if anchor.manifest_fingerprint != manifest.fingerprint:
                raise LifecycleRollbackDetected("External lifecycle anchor binds another manifest")
            if anchor.sequence > expected_anchor.sequence:
                raise LifecycleRollbackDetected("Local lifecycle journal was truncated or rolled back")
            if anchor.sequence == expected_anchor.sequence and anchor.head_fingerprint != expected_anchor.head_fingerprint:
                raise LifecycleRollbackDetected("Lifecycle journal fork differs from external anchor")
            if anchor.sequence < expected_anchor.sequence:
                pending_is_exact = (
                    allow_pending_tail
                    and expected_anchor.sequence == anchor.sequence + 1
                    and entries[anchor.sequence].fingerprint
                    == anchor.head_fingerprint
                )
                if pending_is_exact:
                    return LifecycleStoreSnapshot(manifest, tuple(entries))
                raise LifecycleRollbackDetected(
                    "Lifecycle journal has multiple or forked unanchored tails"
                )
            head_path = root / _HEAD_NAME
            if head_path.exists():
                head = LifecycleAnchor.from_record(_parse_record(head_path))
                if head != expected_anchor:
                    # A committed external anchor makes an interrupted local head
                    # update recoverable without weakening rollback detection.
                    _replace(head_path, expected_anchor.to_record())
            else:
                _replace(head_path, expected_anchor.to_record())
        else:
            if anchor is not None:
                raise LifecycleRollbackDetected(
                    "External anchor proves local lifecycle deletion or store substitution"
                )
            if (root / _HEAD_NAME).exists():
                raise LifecycleStoreError("Lifecycle head exists without journal events")
        return LifecycleStoreSnapshot(manifest, tuple(entries))

    def _replay_and_check(self, manifest: LifecycleStoreManifest, entries: Sequence[LifecycleJournalEntry]) -> None:
        records = [entry.event.to_record() for entry in entries]
        try:
            restored = tuple(self._replay(records))
        except (TypeError, ValueError) as error:
            raise LifecycleStoreError("Lifecycle journal replay failed") from error
        if len(restored) != len(entries) or any(not isinstance(item, LifecycleEvent) for item in restored):
            raise LifecycleStoreError("Lifecycle replay did not return the complete event chain")
        for index, (entry, event) in enumerate(zip(entries, restored)):
            if event.sequence != index or event.fingerprint != entry.event.fingerprint:
                raise LifecycleStoreError("Lifecycle replay changed an event")
            if event.candidate_fingerprint != manifest.candidate_fingerprint or event.audit_plan_fingerprint != manifest.audit_plan_fingerprint:
                raise LifecycleStoreError("Lifecycle replay crosses manifest bindings")
            if event.state == QUARANTINED_STATIC:
                if (
                    event.runtime_certification_fingerprint is not None
                    or event.sealed_audit_certification_fingerprint is not None
                ):
                    raise LifecycleStoreError(
                        "Quarantine event unexpectedly binds certification evidence"
                    )
            elif (
                event.runtime_certification_fingerprint
                != manifest.evidence["runtime_certification"]
                or event.sealed_audit_certification_fingerprint
                != manifest.evidence["sealed_audit_certification"]
            ):
                raise LifecycleStoreError(
                    "Lifecycle certification evidence differs from the manifest"
                )
            self._verify_signatures(manifest, event, entry.previous_entry_sha256, entry.signatures)
