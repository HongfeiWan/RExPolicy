"""Immutable, checksummed shard persistence for Stage 0 trajectories."""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from rexpolicy.stage0.data.trajectory import (
    TRAJECTORY_SCHEMA_ID,
    Stage0Trajectory,
    Stage0TrajectoryProvenance,
    _require_exact_keys,
    trajectory_sha256,
)
from rexpolicy.stage0.types import Stage0Outcome


SHARD_DESCRIPTOR_SCHEMA_VERSION = 1
SHARD_DESCRIPTOR_SCHEMA_ID = "rexpolicy/stage0-trajectory-shard/v1"
_DESCRIPTOR_SUFFIX = ".stage0.json"
_PAYLOAD_SUFFIX = ".stage0.pt"


class Stage0ShardError(RuntimeError):
    """Base class for Stage 0 persistence failures."""


class Stage0ShardIntegrityError(Stage0ShardError):
    """Raised when a descriptor or its immutable payload is inconsistent."""


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _file_sha256(path: Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _temporary_path(target: Path) -> Path:
    return target.with_name(f".{target.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}")


def _publish_immutable(temporary: Path, target: Path) -> None:
    """Atomically link a completed file without ever replacing a target."""
    try:
        os.link(temporary, target)
    except FileExistsError as exc:
        raise FileExistsError(f"immutable Stage 0 shard exists: {target}") from exc
    finally:
        temporary.unlink(missing_ok=True)
    _fsync_directory(target.parent)


def _write_torch_temporary(path: Path, payload: Mapping[str, Any]) -> None:
    import torch

    with path.open("xb") as file:
        torch.save(dict(payload), file)
        file.flush()
        os.fsync(file.fileno())


def _write_json_temporary(path: Path, record: Mapping[str, Any]) -> None:
    encoded = (
        json.dumps(
            record,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    with path.open("xb") as file:
        file.write(encoded)
        file.flush()
        os.fsync(file.fileno())


def write_trajectory_shard(
    directory: str | Path,
    trajectory: Stage0Trajectory,
) -> Path:
    """Persist one trajectory and publish its JSON descriptor last.

    The descriptor is the commit marker.  Both final files are immutable:
    an existing payload or descriptor causes the write to fail instead of
    being replaced.
    """
    if not isinstance(trajectory, Stage0Trajectory):
        raise TypeError("trajectory must be Stage0Trajectory")
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    trajectory_id = trajectory.provenance.trajectory_id
    payload_path = root / f"{trajectory_id}{_PAYLOAD_SUFFIX}"
    descriptor_path = root / f"{trajectory_id}{_DESCRIPTOR_SUFFIX}"
    if payload_path.exists() or descriptor_path.exists():
        raise FileExistsError(
            f"immutable Stage 0 shard already exists for {trajectory_id}"
        )

    content_sha256 = trajectory_sha256(trajectory)
    payload = {
        "actions": trajectory.actions,
        "metadata": trajectory.metadata_record(),
        "schema_id": SHARD_DESCRIPTOR_SCHEMA_ID,
        "schema_version": SHARD_DESCRIPTOR_SCHEMA_VERSION,
        "states": trajectory.states,
        "trajectory_schema_id": TRAJECTORY_SCHEMA_ID,
        "trajectory_sha256": content_sha256,
    }
    payload_temporary = _temporary_path(payload_path)
    descriptor_temporary = _temporary_path(descriptor_path)
    payload_published = False
    try:
        _write_torch_temporary(payload_temporary, payload)
        payload_digest = _file_sha256(payload_temporary)
        payload_bytes = payload_temporary.stat().st_size
        descriptor = {
            "metadata": trajectory.metadata_record(),
            "payload_bytes": payload_bytes,
            "payload_file": payload_path.name,
            "payload_sha256": payload_digest,
            "schema_id": SHARD_DESCRIPTOR_SCHEMA_ID,
            "schema_version": SHARD_DESCRIPTOR_SCHEMA_VERSION,
            "tensors": {
                "actions": {
                    "dtype": "float32",
                    "shape": list(trajectory.actions.shape),
                },
                "states": {
                    "dtype": "float32",
                    "shape": list(trajectory.states.shape),
                },
            },
            "trajectory_schema_id": TRAJECTORY_SCHEMA_ID,
            "trajectory_sha256": content_sha256,
        }
        _write_json_temporary(descriptor_temporary, descriptor)
        _publish_immutable(payload_temporary, payload_path)
        payload_published = True
        _publish_immutable(descriptor_temporary, descriptor_path)
    except Exception:
        payload_temporary.unlink(missing_ok=True)
        descriptor_temporary.unlink(missing_ok=True)
        # A descriptor is the commit marker.  If publishing it failed, remove
        # only the uncommitted payload created by this call.
        if payload_published and not descriptor_path.exists():
            payload_path.unlink(missing_ok=True)
            _fsync_directory(root)
        raise
    return descriptor_path


def _parse_descriptor(path: Path) -> dict[str, Any]:
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise Stage0ShardIntegrityError(
            f"cannot parse Stage 0 shard descriptor {path}: {exc}"
        ) from exc
    if not isinstance(record, dict):
        raise Stage0ShardIntegrityError("shard descriptor must contain an object")
    expected = {
        "metadata",
        "payload_bytes",
        "payload_file",
        "payload_sha256",
        "schema_id",
        "schema_version",
        "tensors",
        "trajectory_schema_id",
        "trajectory_sha256",
    }
    try:
        _require_exact_keys(record, expected, context="shard descriptor")
    except ValueError as exc:
        raise Stage0ShardIntegrityError(str(exc)) from exc
    if record["schema_version"] != SHARD_DESCRIPTOR_SCHEMA_VERSION:
        raise Stage0ShardIntegrityError("unsupported shard descriptor version")
    if record["schema_id"] != SHARD_DESCRIPTOR_SCHEMA_ID:
        raise Stage0ShardIntegrityError("shard descriptor schema ID changed")
    if record["trajectory_schema_id"] != TRAJECTORY_SCHEMA_ID:
        raise Stage0ShardIntegrityError("trajectory schema ID changed")
    return record


def _validate_descriptor_tensor(
    record: Any,
    *,
    name: str,
    tensor: Any,
) -> None:
    if not isinstance(record, Mapping):
        raise Stage0ShardIntegrityError(f"descriptor tensors.{name} must be an object")
    try:
        _require_exact_keys(record, {"dtype", "shape"}, context=f"tensors.{name}")
    except ValueError as exc:
        raise Stage0ShardIntegrityError(str(exc)) from exc
    if record["dtype"] != "float32" or record["shape"] != list(tensor.shape):
        raise Stage0ShardIntegrityError(
            f"descriptor tensor contract changed for {name}"
        )


def load_trajectory_shard(descriptor_path: str | Path) -> Stage0Trajectory:
    """Verify and safely load one committed trajectory shard."""
    import torch

    path = Path(descriptor_path)
    descriptor = _parse_descriptor(path)
    payload_file = descriptor["payload_file"]
    if not isinstance(payload_file, str) or Path(payload_file).name != payload_file:
        raise Stage0ShardIntegrityError("payload_file must be a local basename")
    expected_name = path.name.removesuffix(_DESCRIPTOR_SUFFIX) + _PAYLOAD_SUFFIX
    if payload_file != expected_name:
        raise Stage0ShardIntegrityError(
            f"payload_file does not match descriptor basename: {payload_file!r}"
        )
    payload_path = path.parent / payload_file
    if not payload_path.is_file():
        raise Stage0ShardIntegrityError(
            f"trajectory payload is missing: {payload_path}"
        )
    if (
        isinstance(descriptor["payload_bytes"], bool)
        or not isinstance(descriptor["payload_bytes"], int)
        or descriptor["payload_bytes"] < 1
        or payload_path.stat().st_size != descriptor["payload_bytes"]
    ):
        raise Stage0ShardIntegrityError("trajectory payload size changed")
    payload_digest = _file_sha256(payload_path)
    if payload_digest != descriptor["payload_sha256"]:
        raise Stage0ShardIntegrityError("trajectory payload SHA-256 mismatch")

    try:
        payload = torch.load(payload_path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise Stage0ShardIntegrityError(
            f"cannot safely load trajectory payload: {exc}"
        ) from exc
    if not isinstance(payload, Mapping):
        raise Stage0ShardIntegrityError("trajectory payload must contain a mapping")
    expected_payload = {
        "actions",
        "metadata",
        "schema_id",
        "schema_version",
        "states",
        "trajectory_schema_id",
        "trajectory_sha256",
    }
    try:
        _require_exact_keys(payload, expected_payload, context="trajectory payload")
    except ValueError as exc:
        raise Stage0ShardIntegrityError(str(exc)) from exc
    for key in ("schema_id", "schema_version", "trajectory_schema_id"):
        if payload[key] != descriptor[key]:
            raise Stage0ShardIntegrityError(f"payload {key} changed")
    if payload["metadata"] != descriptor["metadata"]:
        raise Stage0ShardIntegrityError("payload metadata differs from descriptor")
    if payload["trajectory_sha256"] != descriptor["trajectory_sha256"]:
        raise Stage0ShardIntegrityError(
            "payload trajectory hash differs from descriptor"
        )

    metadata = payload["metadata"]
    if not isinstance(metadata, Mapping):
        raise Stage0ShardIntegrityError("trajectory metadata must be a mapping")
    try:
        _require_exact_keys(
            metadata,
            {
                "failure_reason",
                "outcome",
                "provenance",
                "terminated",
                "truncated",
            },
            context="trajectory metadata",
        )
        trajectory = Stage0Trajectory(
            provenance=Stage0TrajectoryProvenance.from_record(metadata["provenance"]),
            states=payload["states"],
            actions=payload["actions"],
            outcome=Stage0Outcome(metadata["outcome"]),
            terminated=metadata["terminated"],
            truncated=metadata["truncated"],
            failure_reason=metadata["failure_reason"],
        )
    except (TypeError, ValueError) as exc:
        raise Stage0ShardIntegrityError(
            f"invalid trajectory payload contract: {exc}"
        ) from exc
    tensors = descriptor["tensors"]
    if not isinstance(tensors, Mapping) or set(tensors) != {"states", "actions"}:
        raise Stage0ShardIntegrityError("descriptor tensors keys do not match schema")
    _validate_descriptor_tensor(
        tensors["states"],
        name="states",
        tensor=trajectory.states,
    )
    _validate_descriptor_tensor(
        tensors["actions"], name="actions", tensor=trajectory.actions
    )
    if trajectory_sha256(trajectory) != descriptor["trajectory_sha256"]:
        raise Stage0ShardIntegrityError("trajectory semantic SHA-256 mismatch")
    return trajectory
