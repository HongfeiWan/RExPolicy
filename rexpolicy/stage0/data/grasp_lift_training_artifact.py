"""Immutable, tensor-free persistence for Grasp-Lift training contracts."""

from __future__ import annotations

import json
import os
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from rexpolicy.stage0.checkpoint import file_sha256
from rexpolicy.stage0.data.grasp_lift_training import (
    GraspLiftTrainingData,
    build_grasp_lift_training_data,
    load_grasp_lift_training_corpus,
)
from rexpolicy.stage0.grasp_lift_pilot import (
    seal_grasp_lift_pilot_record,
    verify_grasp_lift_pilot_record,
)
from rexpolicy.stage0.normalization import (
    Stage0Normalization,
    write_stage0_normalization,
)


GRASP_LIFT_TRAINING_ARTIFACT_MANIFEST_SCHEMA_ID = (
    "rexpolicy/stage0-grasp-lift-training-artifact-manifest/v1"
)
GRASP_LIFT_TRAINING_ARTIFACT_COMMIT_SCHEMA_ID = (
    "rexpolicy/stage0-grasp-lift-training-artifact-commit/v1"
)
GRASP_LIFT_TRAINING_ARTIFACT_STORAGE_POLICY = {
    "held_out_tensors": "never_opened_or_stored",
    "model_tensors": "not_stored; deterministically rebuilt from train shards",
    "normalization": "train-only semantic record",
    "schema_id": "rexpolicy/stage0-grasp-lift-training-storage/v1",
    "test_data": "not_created",
}
_ARTIFACT_FILES = frozenset(
    {
        "commit.json",
        "manifest.json",
        "normalization.json",
    }
)
_MANIFEST_KEYS = {
    "audit_dataset_sha256",
    "dataset_record",
    "normalization_file",
    "normalization_file_sha256",
    "normalization_sha256",
    "schema_id",
    "self_sha256",
    "storage_policy",
    "train_content_sha256",
}
_COMMIT_KEYS = {
    "audit_dataset_sha256",
    "manifest_file",
    "manifest_file_sha256",
    "manifest_sha256",
    "normalization_file",
    "normalization_file_sha256",
    "normalization_sha256",
    "schema_id",
    "self_sha256",
    "train_content_sha256",
}


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _read_json(path: Path, context: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{context} is missing or a symlink")
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=lambda constant: (_ for _ in ()).throw(
                ValueError(f"invalid JSON constant: {constant}")
            ),
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"cannot read {context}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{context} must contain an object")
    return value


def _write_json(path: Path, record: Mapping[str, Any]) -> None:
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
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _require_exact_keys(
    record: Mapping[str, Any],
    expected: set[str],
    context: str,
) -> None:
    if set(record) != expected:
        raise ValueError(f"{context} fields changed")


def _verify_artifact_directory(
    directory: str | Path,
    *,
    allow_partial: bool,
) -> Path:
    requested = Path(directory).expanduser()
    if requested.is_symlink() or not requested.is_dir():
        raise ValueError("training artifact must be a directory, not a symlink")
    if not allow_partial and ".partial-" in requested.name:
        raise ValueError("partial training artifacts cannot be loaded")
    root = requested.resolve()
    entries = tuple(root.iterdir())
    if {entry.name for entry in entries} != _ARTIFACT_FILES:
        raise ValueError("training artifact file set changed")
    if any(entry.is_symlink() or not entry.is_file() for entry in entries):
        raise ValueError("training artifact contains a symlink or non-file")
    return root


@dataclass(frozen=True)
class GraspLiftTrainingArtifact:
    """Strictly reloaded artifact plus its rebuilt train-only tensors."""

    root: Path
    data: GraspLiftTrainingData
    normalization: Stage0Normalization
    manifest: Mapping[str, Any]
    commit: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not isinstance(self.root, Path) or not self.root.is_absolute():
            raise ValueError("artifact root must be an absolute pathlib.Path")
        if not isinstance(self.data, GraspLiftTrainingData):
            raise TypeError("data must be GraspLiftTrainingData")
        if not isinstance(self.normalization, Stage0Normalization):
            raise TypeError("normalization must be Stage0Normalization")
        if self.normalization.to_record() != self.data.normalization.to_record():
            raise ValueError(
                "artifact normalization differs from rebuilt training data"
            )
        for value, name in (
            (self.manifest, "manifest"),
            (self.commit, "commit"),
        ):
            if not isinstance(value, Mapping):
                raise TypeError(f"{name} must be a mapping")
        object.__setattr__(self, "manifest", MappingProxyType(dict(self.manifest)))
        object.__setattr__(self, "commit", MappingProxyType(dict(self.commit)))

    @property
    def manifest_sha256(self) -> str:
        return str(self.manifest["self_sha256"])

    @property
    def artifact_sha256(self) -> str:
        return str(self.commit["self_sha256"])

    @property
    def train_content_sha256(self) -> str:
        return str(self.manifest["train_content_sha256"])

    @property
    def audit_dataset_sha256(self) -> str:
        return str(self.manifest["audit_dataset_sha256"])


def _load_grasp_lift_training_artifact(
    directory: str | Path,
    *,
    pilot_directory: str | Path,
    allow_partial: bool,
) -> GraspLiftTrainingArtifact:
    root = _verify_artifact_directory(directory, allow_partial=allow_partial)
    commit = _read_json(root / "commit.json", "training artifact commit")
    _require_exact_keys(commit, _COMMIT_KEYS, "training artifact commit")
    verify_grasp_lift_pilot_record(commit, "training artifact commit")
    if commit["schema_id"] != GRASP_LIFT_TRAINING_ARTIFACT_COMMIT_SCHEMA_ID:
        raise ValueError("training artifact commit schema changed")
    if (
        commit["manifest_file"] != "manifest.json"
        or commit["normalization_file"] != "normalization.json"
    ):
        raise ValueError("training artifact commit file names changed")

    manifest_path = root / "manifest.json"
    manifest = _read_json(manifest_path, "training artifact manifest")
    _require_exact_keys(manifest, _MANIFEST_KEYS, "training artifact manifest")
    verify_grasp_lift_pilot_record(manifest, "training artifact manifest")
    if manifest["schema_id"] != GRASP_LIFT_TRAINING_ARTIFACT_MANIFEST_SCHEMA_ID:
        raise ValueError("training artifact manifest schema changed")
    if manifest["normalization_file"] != "normalization.json":
        raise ValueError("training artifact normalization file name changed")
    if manifest["storage_policy"] != GRASP_LIFT_TRAINING_ARTIFACT_STORAGE_POLICY:
        raise ValueError("training artifact storage policy changed")
    if commit["manifest_sha256"] != manifest["self_sha256"] or commit[
        "manifest_file_sha256"
    ] != file_sha256(manifest_path):
        raise ValueError("training artifact manifest commitment changed")

    normalization_path = root / "normalization.json"
    normalization_file_sha256 = file_sha256(normalization_path)
    if (
        manifest["normalization_file_sha256"] != normalization_file_sha256
        or commit["normalization_file_sha256"] != normalization_file_sha256
    ):
        raise ValueError("training artifact normalization file hash changed")
    normalization = Stage0Normalization.from_record(
        _read_json(normalization_path, "training artifact normalization")
    )
    if (
        manifest["normalization_sha256"] != normalization.sha256
        or commit["normalization_sha256"] != normalization.sha256
    ):
        raise ValueError("training artifact normalization semantic hash changed")

    corpus = load_grasp_lift_training_corpus(pilot_directory)
    data = build_grasp_lift_training_data(corpus)
    if normalization.to_record() != data.normalization.to_record():
        raise ValueError("persisted normalization is not the rebuilt train-only fit")
    if manifest["dataset_record"] != data.to_record():
        raise ValueError("training artifact dataset record changed")
    if (
        manifest["train_content_sha256"] != data.train_content_sha256
        or commit["train_content_sha256"] != data.train_content_sha256
    ):
        raise ValueError("training artifact train content changed")
    if (
        manifest["audit_dataset_sha256"] != data.dataset_sha256
        or commit["audit_dataset_sha256"] != data.dataset_sha256
    ):
        raise ValueError("training artifact audit dataset changed")
    return GraspLiftTrainingArtifact(
        root=root,
        data=data,
        normalization=normalization,
        manifest=manifest,
        commit=commit,
    )


def load_grasp_lift_training_artifact(
    directory: str | Path,
    *,
    pilot_directory: str | Path,
) -> GraspLiftTrainingArtifact:
    """Strictly reload and independently rebuild one published artifact."""
    return _load_grasp_lift_training_artifact(
        directory,
        pilot_directory=pilot_directory,
        allow_partial=False,
    )


def publish_grasp_lift_training_artifact(
    target_directory: str | Path,
    *,
    pilot_directory: str | Path,
) -> GraspLiftTrainingArtifact:
    """Build, verify, and atomically publish a three-file training artifact."""
    target = Path(target_directory).expanduser()
    if target.is_symlink() or target.exists():
        raise FileExistsError(f"training artifact target already exists: {target}")
    parent = target.parent
    parent.mkdir(parents=True, exist_ok=True)
    if parent.is_symlink() or not parent.is_dir():
        raise ValueError("training artifact parent must be a real directory")

    corpus = load_grasp_lift_training_corpus(pilot_directory)
    data = build_grasp_lift_training_data(corpus)
    partial = parent / f".{target.name}.partial-{uuid.uuid4().hex}"
    partial.mkdir()
    try:
        normalization_path = partial / "normalization.json"
        write_stage0_normalization(normalization_path, data.normalization)
        normalization_file_sha256 = file_sha256(normalization_path)
        manifest = seal_grasp_lift_pilot_record(
            {
                "audit_dataset_sha256": data.dataset_sha256,
                "dataset_record": data.to_record(),
                "normalization_file": "normalization.json",
                "normalization_file_sha256": normalization_file_sha256,
                "normalization_sha256": data.normalization.sha256,
                "schema_id": GRASP_LIFT_TRAINING_ARTIFACT_MANIFEST_SCHEMA_ID,
                "storage_policy": GRASP_LIFT_TRAINING_ARTIFACT_STORAGE_POLICY,
                "train_content_sha256": data.train_content_sha256,
            }
        )
        manifest_path = partial / "manifest.json"
        _write_json(manifest_path, manifest)
        commit = seal_grasp_lift_pilot_record(
            {
                "audit_dataset_sha256": data.dataset_sha256,
                "manifest_file": "manifest.json",
                "manifest_file_sha256": file_sha256(manifest_path),
                "manifest_sha256": manifest["self_sha256"],
                "normalization_file": "normalization.json",
                "normalization_file_sha256": normalization_file_sha256,
                "normalization_sha256": data.normalization.sha256,
                "schema_id": GRASP_LIFT_TRAINING_ARTIFACT_COMMIT_SCHEMA_ID,
                "train_content_sha256": data.train_content_sha256,
            }
        )
        _write_json(partial / "commit.json", commit)
        _fsync_directory(partial)
        _load_grasp_lift_training_artifact(
            partial,
            pilot_directory=pilot_directory,
            allow_partial=True,
        )
        if target.is_symlink() or target.exists():
            raise FileExistsError(
                f"training artifact target appeared during publication: {target}"
            )
        partial.rename(target)
        _fsync_directory(parent)
    finally:
        if partial.exists():
            shutil.rmtree(partial)
    return load_grasp_lift_training_artifact(
        target,
        pilot_directory=pilot_directory,
    )


__all__ = [
    "GRASP_LIFT_TRAINING_ARTIFACT_COMMIT_SCHEMA_ID",
    "GRASP_LIFT_TRAINING_ARTIFACT_MANIFEST_SCHEMA_ID",
    "GRASP_LIFT_TRAINING_ARTIFACT_STORAGE_POLICY",
    "GraspLiftTrainingArtifact",
    "load_grasp_lift_training_artifact",
    "publish_grasp_lift_training_artifact",
]
