"""Durable, one-shot persistence for Grasp-Lift validation claims.

The claim file is the consumption boundary for the formal validation cohort.
It lives in the repository's Git common directory so that a different
worktree, branch, or output directory cannot claim the same cohort again.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
import subprocess
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rexpolicy.stage0.data.grasp_lift_validation import (
    GRASP_LIFT_LOCKED_TEST_ABSENT,
    GRASP_LIFT_VALIDATION_CLAIM_SCHEMA_ID,
    GRASP_LIFT_VALIDATION_PURPOSE,
)
from rexpolicy.stage0.types import canonical_fingerprint


GRASP_LIFT_VALIDATION_CLAIM_REGISTRY = "rexpolicy-validation-claims"
GRASP_LIFT_VALIDATION_CLAIM_RECEIPT_SCHEMA_ID = (
    "rexpolicy/stage0-grasp-lift-validation-claim-receipt/v1"
)
__all__ = [
    "GRASP_LIFT_VALIDATION_CLAIM_RECEIPT_SCHEMA_ID",
    "GRASP_LIFT_VALIDATION_CLAIM_REGISTRY",
    "ClaimVerifier",
    "GraspLiftValidationClaimReceipt",
    "persist_grasp_lift_validation_claim",
    "verify_persisted_grasp_lift_validation_claim",
]
_MAX_CLAIM_BYTES = 64 * 1024
_CLAIM_KEYS = {
    "candidate_inventory_sha256",
    "claim_id",
    "cohort_identity_sha256",
    "evaluator_implementation_sha256",
    "locked_test",
    "preflight_sha256",
    "purpose",
    "schema_id",
    "selection_protocol_sha256",
    "self_sha256",
}
_SHA256_FIELDS = {
    "candidate_inventory_sha256",
    "claim_id",
    "cohort_identity_sha256",
    "evaluator_implementation_sha256",
    "preflight_sha256",
    "selection_protocol_sha256",
    "self_sha256",
}
_RECEIPT_KEYS = {
    "claim_file_sha256",
    "claim_id",
    "claim_path",
    "claim_self_sha256",
    "git_common_directory",
    "repository_root",
    "schema_id",
    "self_sha256",
}

ClaimVerifier = Callable[[Mapping[str, Any]], None]


def _require_sha256(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _canonical_claim_payload(record: Mapping[str, Any]) -> bytes:
    try:
        payload = (
            json.dumps(
                dict(record),
                allow_nan=False,
                ensure_ascii=True,
                indent=2,
                sort_keys=True,
            )
            + "\n"
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeError) as error:
        raise ValueError("validation claim is not strict JSON") from error
    if len(payload) > _MAX_CLAIM_BYTES:
        raise ValueError("validation claim exceeds the trusted size limit")
    return payload


def _strict_claim_snapshot(
    claim: Mapping[str, Any],
    *,
    verifier: ClaimVerifier,
) -> tuple[dict[str, Any], bytes]:
    """Validate and detach a claim before any filesystem state is touched."""
    if not isinstance(claim, Mapping):
        raise TypeError("claim must be a mapping")
    if not callable(verifier):
        raise TypeError("verifier must be callable")
    payload = _canonical_claim_payload(claim)
    try:
        record = json.loads(payload.decode("ascii"))
    except (UnicodeError, json.JSONDecodeError) as error:  # pragma: no cover
        raise RuntimeError("internally inconsistent claim serialization") from error
    if not isinstance(record, dict) or set(record) != _CLAIM_KEYS:
        raise ValueError("validation claim fields changed")
    for field in _SHA256_FIELDS:
        _require_sha256(record[field], f"validation claim {field}")
    unsigned = dict(record)
    self_sha256 = unsigned.pop("self_sha256")
    if canonical_fingerprint(unsigned) != self_sha256:
        raise ValueError("validation claim self_sha256 changed")
    if (
        record["schema_id"] != GRASP_LIFT_VALIDATION_CLAIM_SCHEMA_ID
        or record["purpose"] != GRASP_LIFT_VALIDATION_PURPOSE
        or record["locked_test"] != GRASP_LIFT_LOCKED_TEST_ABSENT
    ):
        raise ValueError("validation claim protocol fields changed")

    verifier_input = json.loads(payload.decode("ascii"))
    verifier(verifier_input)
    if verifier_input != record:
        raise RuntimeError("validation claim verifier mutated its input")
    return record, payload


def _one_git_path(
    repository_root: Path,
    arguments: list[str],
    *,
    context: str,
) -> Path:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "--path-format=absolute", *arguments],
            cwd=repository_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError(f"cannot resolve Git {context}") from error
    lines = completed.stdout.splitlines()
    if len(lines) != 1 or not lines[0]:
        raise RuntimeError(f"Git returned an invalid {context}")
    path = Path(lines[0])
    if not path.is_absolute():
        raise RuntimeError(f"Git {context} is not absolute")
    if path.is_symlink():
        raise RuntimeError(f"Git {context} cannot be a symlink")
    try:
        return path.resolve(strict=True)
    except OSError as error:
        raise RuntimeError(f"Git {context} does not exist") from error


def _git_common_directory(repository_root: str | Path) -> tuple[Path, Path]:
    requested = Path(repository_root).expanduser()
    if requested.is_symlink() or not requested.is_dir():
        raise ValueError("repository_root must be a real directory")
    root = requested.resolve()
    top_level = _one_git_path(root, ["--show-toplevel"], context="top level")
    if top_level != root:
        raise ValueError("repository_root must be the Git worktree root")
    common = _one_git_path(root, ["--git-common-dir"], context="common directory")
    if common.is_symlink() or not common.is_dir():
        raise RuntimeError("Git common directory must be a real directory")
    return root, common


def _directory_flags() -> int:
    return (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )


def _open_registry(common: Path, *, create: bool) -> tuple[int, int]:
    common_descriptor = os.open(common, _directory_flags())
    try:
        if create:
            try:
                os.mkdir(
                    GRASP_LIFT_VALIDATION_CLAIM_REGISTRY,
                    mode=0o700,
                    dir_fd=common_descriptor,
                )
            except FileExistsError:
                pass
            else:
                os.fsync(common_descriptor)
        registry_descriptor = os.open(
            GRASP_LIFT_VALIDATION_CLAIM_REGISTRY,
            _directory_flags(),
            dir_fd=common_descriptor,
        )
    except BaseException:
        os.close(common_descriptor)
        raise
    return common_descriptor, registry_descriptor


def _close_registry(common_descriptor: int, registry_descriptor: int) -> None:
    try:
        os.close(registry_descriptor)
    finally:
        os.close(common_descriptor)


def _write_all(descriptor: int, payload: bytes) -> None:
    view = memoryview(payload)
    while view:
        written = os.write(descriptor, view)
        if written <= 0:
            raise OSError("short validation claim write")
        view = view[written:]


def _read_all(descriptor: int) -> bytes:
    chunks: list[bytes] = []
    size = 0
    while True:
        chunk = os.read(descriptor, min(8192, _MAX_CLAIM_BYTES + 1 - size))
        if not chunk:
            return b"".join(chunks)
        chunks.append(chunk)
        size += len(chunk)
        if size > _MAX_CLAIM_BYTES:
            raise ValueError("persisted validation claim is too large")


@dataclass(frozen=True)
class GraspLiftValidationClaimReceipt:
    """A path-bound receipt returned only after the claim is durable."""

    repository_root: Path
    git_common_directory: Path
    claim_path: Path
    claim_id: str
    claim_self_sha256: str
    claim_file_sha256: str

    def __post_init__(self) -> None:
        for path, name in (
            (self.repository_root, "repository_root"),
            (self.git_common_directory, "git_common_directory"),
            (self.claim_path, "claim_path"),
        ):
            if not isinstance(path, Path) or not path.is_absolute():
                raise ValueError(f"receipt {name} must be an absolute pathlib.Path")
        for value, name in (
            (self.claim_id, "claim_id"),
            (self.claim_self_sha256, "claim_self_sha256"),
            (self.claim_file_sha256, "claim_file_sha256"),
        ):
            _require_sha256(value, f"receipt {name}")
        expected = (
            self.git_common_directory
            / GRASP_LIFT_VALIDATION_CLAIM_REGISTRY
            / f"{self.claim_id}.json"
        )
        if self.claim_path != expected:
            raise ValueError("receipt claim_path is not the fixed claim path")

    def to_record(self) -> dict[str, Any]:
        """Return a canonical, self-hashed record for runner audit output."""
        record: dict[str, Any] = {
            "claim_file_sha256": self.claim_file_sha256,
            "claim_id": self.claim_id,
            "claim_path": str(self.claim_path),
            "claim_self_sha256": self.claim_self_sha256,
            "git_common_directory": str(self.git_common_directory),
            "repository_root": str(self.repository_root),
            "schema_id": GRASP_LIFT_VALIDATION_CLAIM_RECEIPT_SCHEMA_ID,
        }
        record["self_sha256"] = canonical_fingerprint(record)
        return record

    @classmethod
    def from_record(
        cls,
        value: Mapping[str, Any],
    ) -> GraspLiftValidationClaimReceipt:
        """Strictly reload and self-verify a serialized receipt."""
        if not isinstance(value, Mapping) or set(value) != _RECEIPT_KEYS:
            raise ValueError("validation claim receipt fields changed")
        record = dict(value)
        self_sha256 = _require_sha256(
            record.pop("self_sha256"),
            "validation claim receipt self_sha256",
        )
        if canonical_fingerprint(record) != self_sha256:
            raise ValueError("validation claim receipt self_sha256 changed")
        if record["schema_id"] != GRASP_LIFT_VALIDATION_CLAIM_RECEIPT_SCHEMA_ID:
            raise ValueError("validation claim receipt schema changed")
        for field in (
            "claim_path",
            "git_common_directory",
            "repository_root",
        ):
            if not isinstance(record[field], str):
                raise TypeError(f"validation claim receipt {field} must be a string")
        receipt = cls(
            repository_root=Path(record["repository_root"]),
            git_common_directory=Path(record["git_common_directory"]),
            claim_path=Path(record["claim_path"]),
            claim_id=record["claim_id"],
            claim_self_sha256=record["claim_self_sha256"],
            claim_file_sha256=record["claim_file_sha256"],
        )
        if receipt.to_record() != dict(value):
            raise ValueError("validation claim receipt is not canonical")
        return receipt


def _receipt(
    *,
    root: Path,
    common: Path,
    record: Mapping[str, Any],
    payload: bytes,
) -> GraspLiftValidationClaimReceipt:
    claim_id = record["claim_id"]
    return GraspLiftValidationClaimReceipt(
        repository_root=root,
        git_common_directory=common,
        claim_path=(common / GRASP_LIFT_VALIDATION_CLAIM_REGISTRY / f"{claim_id}.json"),
        claim_id=claim_id,
        claim_self_sha256=record["self_sha256"],
        claim_file_sha256=hashlib.sha256(payload).hexdigest(),
    )


def persist_grasp_lift_validation_claim(
    repository_root: str | Path,
    claim: Mapping[str, Any],
    *,
    verifier: ClaimVerifier,
) -> GraspLiftValidationClaimReceipt:
    """Create and durably flush the one allowed claim, or fail closed.

    If creation succeeds but a later write or flush fails, the target is
    deliberately retained.  A retry will encounter ``O_EXCL`` and fail rather
    than risk consuming the validation cohort twice.
    """
    record, payload = _strict_claim_snapshot(claim, verifier=verifier)
    root, common = _git_common_directory(repository_root)
    receipt = _receipt(root=root, common=common, record=record, payload=payload)
    common_descriptor, registry_descriptor = _open_registry(common, create=True)
    try:
        flags = (
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        try:
            descriptor = os.open(
                receipt.claim_path.name,
                flags,
                0o600,
                dir_fd=registry_descriptor,
            )
        except FileExistsError as error:
            raise FileExistsError(
                f"formal validation claim already exists: {receipt.claim_path}"
            ) from error
        try:
            try:
                _write_all(descriptor, payload)
                os.fsync(descriptor)
            except BaseException:
                # Best effort makes even a partial sentinel durable.  It must
                # never be deleted or retried under the same cohort identity.
                try:
                    os.fsync(descriptor)
                except OSError:
                    pass
                try:
                    os.fsync(registry_descriptor)
                except OSError:
                    pass
                raise
        finally:
            os.close(descriptor)
        os.fsync(registry_descriptor)
    finally:
        _close_registry(common_descriptor, registry_descriptor)
    return receipt


def verify_persisted_grasp_lift_validation_claim(
    repository_root: str | Path,
    claim: Mapping[str, Any],
    *,
    verifier: ClaimVerifier,
    receipt: GraspLiftValidationClaimReceipt | None = None,
) -> GraspLiftValidationClaimReceipt:
    """Verify the exact canonical claim at its sole Git-common-dir path.

    This is the loader-side guard: an in-memory claim is insufficient.  The
    claim must already exist as a real regular file with byte-for-byte
    canonical content.  The file and registry are flushed again before a
    receipt is returned.
    """
    record, payload = _strict_claim_snapshot(claim, verifier=verifier)
    root, common = _git_common_directory(repository_root)
    observed_receipt = _receipt(
        root=root,
        common=common,
        record=record,
        payload=payload,
    )
    if receipt is not None:
        if not isinstance(receipt, GraspLiftValidationClaimReceipt):
            raise TypeError("receipt must be GraspLiftValidationClaimReceipt")
        if receipt != observed_receipt:
            raise ValueError("validation claim receipt changed")

    common_descriptor, registry_descriptor = _open_registry(common, create=False)
    try:
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(
            observed_receipt.claim_path.name,
            flags,
            dir_fd=registry_descriptor,
        )
        try:
            metadata = os.fstat(descriptor)
            if not stat.S_ISREG(metadata.st_mode):
                raise ValueError("persisted validation claim is not a regular file")
            observed_payload = _read_all(descriptor)
            if observed_payload != payload:
                raise ValueError(
                    "persisted validation claim is non-canonical or changed"
                )
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        os.fsync(registry_descriptor)
    finally:
        _close_registry(common_descriptor, registry_descriptor)
    return observed_receipt
