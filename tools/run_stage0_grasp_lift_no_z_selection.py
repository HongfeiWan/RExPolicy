#!/usr/bin/env python3
"""Consume Grasp-Lift validation once and select one no-z checkpoint family."""

from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any


_CPU_THREADS = 16
_INTEROP_THREADS = 4
_CUBLAS_WORKSPACE_CONFIG = ":4096:8"
_TF32_OVERRIDE = "0"
_TRAINING_SEEDS = (31_001, 31_002, 31_003)
_CHECKPOINT_STEPS = tuple(range(0, 10_001, 500))
_CANDIDATE_STEPS = _CHECKPOINT_STEPS[1:]
_VALIDATION_RESET_SEEDS = (7009, 7014, 7019, 7025, 7026, 7031)
_ROLLOUT_EPISODES = 24
_MAX_EPISODE_STEPS = 72
_LOCKED_TEST_ABSENT = MappingProxyType(
    {"artifact": None, "consumed": False, "policy": "not_created"}
)
_STATUS_SCHEMA_ID = "rexpolicy/stage0-grasp-lift-no-z-selection-status/v1"
_AUDIT_SCHEMA_ID = "rexpolicy/stage0-grasp-lift-no-z-selection-audit/v1"
_EVALUATOR_SCHEMA_ID = "rexpolicy/stage0-grasp-lift-no-z-selection-evaluator/v1"
_PILOT_SIMULATOR_SCHEMA_ID = "rexpolicy/stage0-grasp-lift-newton-runtime/v1"
_ENVIRONMENT_RUNTIME_VARIANTS = frozenset({"device", "num_envs"})
_PILOT_SIMULATOR_KEYS = {
    "action_schema_sha256",
    "authoring_mode_sha256",
    "environment_config",
    "implementation_sha256",
    "oracle",
    "oracle_id",
    "schema_id",
    "task_metadata",
}
_CHECKPOINT_KEYS = {
    "checkpoint_name",
    "checkpoint_sha256",
    "checksums_file_sha256",
    "complete_file_sha256",
    "manifest_file_sha256",
    "model_payload_sha256",
    "model_schema_sha256",
    "sequence",
}


def _canonical_fingerprint(record: Any) -> str:
    import hashlib

    try:
        payload = json.dumps(
            record,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeError) as error:
        raise ValueError("record is not finite canonical JSON") from error
    return hashlib.sha256(payload).hexdigest()


def _file_sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sealed(record: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(record)
    if "self_sha256" in value:
        raise ValueError("cannot seal a record that already has self_sha256")
    value["self_sha256"] = _canonical_fingerprint(value)
    return value


def _verify_sealed(record: Mapping[str, Any], context: str) -> None:
    if not isinstance(record, Mapping):
        raise TypeError(f"{context} must be a mapping")
    value = dict(record)
    digest = value.pop("self_sha256", None)
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
        or _canonical_fingerprint(value) != digest
    ):
        raise ValueError(f"{context} self_sha256 changed")


def _require_sha256(value: Any, context: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{context} must be a lowercase SHA-256 digest")
    return value


def _environment_semantic_projection(record: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(record, Mapping) or any(
        not isinstance(key, str) for key in record
    ):
        raise TypeError("pilot environment config must be a string-keyed mapping")
    if not _ENVIRONMENT_RUNTIME_VARIANTS.issubset(record):
        raise ValueError("pilot environment config lost runtime variant fields")
    return {
        key: value
        for key, value in record.items()
        if key not in _ENVIRONMENT_RUNTIME_VARIANTS
    }


def _verify_simulator_semantics(
    accepted: Mapping[str, Any],
    *,
    evaluation_environment: Mapping[str, Any],
    evaluation_task_metadata: Mapping[str, Any],
    evaluation_oracle: Mapping[str, Any],
    action_schema_sha256: str,
    authoring_mode_sha256: str,
    oracle_id: str,
) -> None:
    if not isinstance(accepted, Mapping) or set(accepted) != _PILOT_SIMULATOR_KEYS:
        raise ValueError("accepted pilot simulator record fields changed")
    if accepted["schema_id"] != _PILOT_SIMULATOR_SCHEMA_ID:
        raise ValueError("accepted pilot simulator schema changed")
    for value, name in (
        (accepted["action_schema_sha256"], "accepted action schema"),
        (accepted["authoring_mode_sha256"], "accepted authoring mode"),
        (accepted["implementation_sha256"], "accepted implementation"),
        (action_schema_sha256, "evaluation action schema"),
        (authoring_mode_sha256, "evaluation authoring mode"),
    ):
        _require_sha256(value, name)
    if (
        accepted["action_schema_sha256"] != action_schema_sha256
        or accepted["authoring_mode_sha256"] != authoring_mode_sha256
        or accepted["oracle_id"] != oracle_id
        or accepted["oracle"] != dict(evaluation_oracle)
        or _canonical_fingerprint(accepted["task_metadata"])
        != _canonical_fingerprint(evaluation_task_metadata)
        or _environment_semantic_projection(accepted["environment_config"])
        != _environment_semantic_projection(evaluation_environment)
    ):
        raise ValueError(
            "evaluation simulator mechanics differ from the accepted pilot"
        )


def _accepted_pilot_simulator_contract(
    manifest: Mapping[str, Any],
    *,
    expected_manifest_sha256: str,
    evaluation_environment: Mapping[str, Any],
    evaluation_task_metadata: Mapping[str, Any],
    evaluation_oracle: Mapping[str, Any],
    action_schema_sha256: str,
    authoring_mode_sha256: str,
    oracle_id: str,
) -> tuple[dict[str, Any], str]:
    """Bind evaluation mechanics to the immutable accepted pilot manifest."""

    _verify_sealed(manifest, "accepted pilot manifest")
    expected_manifest_sha256 = _require_sha256(
        expected_manifest_sha256,
        "accepted pilot manifest commitment",
    )
    if manifest["self_sha256"] != expected_manifest_sha256:
        raise ValueError("pilot manifest differs from the training artifact")
    if manifest.get("locked_test") != _LOCKED_TEST_ABSENT:
        raise ValueError("accepted pilot manifest touched locked test data")
    simulator = manifest.get("simulator_record")
    if not isinstance(simulator, Mapping) or set(simulator) != _PILOT_SIMULATOR_KEYS:
        raise ValueError("accepted pilot simulator record fields changed")
    accepted = dict(simulator)
    simulator_sha256 = _require_sha256(
        manifest.get("simulator_sha256"),
        "accepted pilot simulator commitment",
    )
    if _canonical_fingerprint(accepted) != simulator_sha256:
        raise ValueError("accepted pilot simulator commitment changed")
    _verify_simulator_semantics(
        accepted,
        evaluation_environment=evaluation_environment,
        evaluation_task_metadata=evaluation_task_metadata,
        evaluation_oracle=evaluation_oracle,
        action_schema_sha256=action_schema_sha256,
        authoring_mode_sha256=authoring_mode_sha256,
        oracle_id=oracle_id,
    )
    return accepted, simulator_sha256


def _json_payload(record: Mapping[str, Any]) -> bytes:
    try:
        return (
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
        raise ValueError("output record is not strict JSON") from error


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_DIRECTORY", 0),
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_json_exclusive(path: Path, record: Mapping[str, Any]) -> None:
    if path.parent.is_symlink() or not path.parent.is_dir():
        raise ValueError("output parent must be a real directory")
    payload = _json_payload(record)
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor = os.open(path, flags, 0o600)
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("short JSON write")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    _fsync_directory(path.parent)


def _replace_json(path: Path, record: Mapping[str, Any]) -> None:
    if path.parent.is_symlink() or not path.parent.is_dir():
        raise ValueError("output parent must be a real directory")
    temporary = path.parent / f".{path.name}.tmp-{uuid.uuid4().hex}"
    try:
        _write_json_exclusive(temporary, record)
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _strict_json(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"required JSON file is missing or a symlink: {path}")

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    try:
        value = json.loads(
            path.read_text(encoding="ascii"),
            object_pairs_hook=reject_duplicates,
            parse_constant=lambda constant: (_ for _ in ()).throw(
                ValueError(f"invalid JSON constant: {constant}")
            ),
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"cannot read strict JSON {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"JSON file must contain an object: {path}")
    return value


@dataclass(frozen=True)
class SelectionArguments:
    repository_root: Path
    config: Path
    pilot_directory: Path
    training_artifact: Path
    output_directory: Path
    device: str
    run_directories: Mapping[int, Path]

    def __post_init__(self) -> None:
        for name in (
            "repository_root",
            "config",
            "pilot_directory",
            "training_artifact",
            "output_directory",
        ):
            value = getattr(self, name)
            if not isinstance(value, Path) or not value.is_absolute():
                raise ValueError(f"{name} must be an absolute pathlib.Path")
        if (
            not isinstance(self.device, str)
            or not self.device.startswith("cuda:")
            or not self.device.removeprefix("cuda:").isdigit()
        ):
            raise ValueError("device must be an explicit cuda:N device")
        if not isinstance(self.run_directories, Mapping) or set(
            self.run_directories
        ) != set(_TRAINING_SEEDS):
            raise ValueError("run_directories must map exactly the three fixed seeds")
        values = tuple(self.run_directories[seed] for seed in _TRAINING_SEEDS)
        if any(
            not isinstance(value, Path) or not value.is_absolute() for value in values
        ):
            raise ValueError("all run directories must be absolute pathlib.Paths")
        if len(set(values)) != len(values):
            raise ValueError("the three run directories must be distinct")
        object.__setattr__(
            self,
            "run_directories",
            MappingProxyType(
                {seed: self.run_directories[seed] for seed in _TRAINING_SEEDS}
            ),
        )


@dataclass(frozen=True)
class CandidateCommitment:
    checkpoint_step: int
    training_seed: int
    run_directory: Path
    checkpoint: Mapping[str, Any]

    @property
    def inventory_key(self) -> tuple[int, int]:
        return self.checkpoint_step, self.training_seed


@dataclass(frozen=True)
class PreparedSelection:
    artifact: Any
    config: Any
    normalization: Any
    implementation_record: Mapping[str, Any]
    implementation_sha256: str
    evaluator_record: Mapping[str, Any]
    evaluator_sha256: str
    selection_protocol_sha256: str
    runtime_record: Mapping[str, Any]
    newton_runtime: Any
    environment_config_record: Mapping[str, Any]
    task_metadata_record: Mapping[str, Any]
    oracle_record: Mapping[str, Any]
    accepted_simulator_record: Mapping[str, Any]
    accepted_simulator_sha256: str


@dataclass(frozen=True)
class EnvironmentHandle:
    raw_environment: Any
    environment: Any
    oracle: Any
    record: Mapping[str, Any]


@dataclass(frozen=True)
class CandidateEvaluation:
    evidence: Any
    record: Mapping[str, Any]


class SelectionOutput:
    """Fresh output with write-once evidence and replaceable status only."""

    def __init__(self, root: Path):
        if not isinstance(root, Path) or not root.is_absolute():
            raise ValueError("selection output must be an absolute pathlib.Path")
        self.root = root
        self.created = False
        self.claimed = False

    @property
    def status_path(self) -> Path:
        return self.root / "status.json"

    def create(self) -> None:
        if self.root.is_symlink() or self.root.exists():
            raise FileExistsError(f"selection output already exists: {self.root}")
        self.root.parent.mkdir(parents=True, exist_ok=True)
        if self.root.parent.is_symlink() or not self.root.parent.is_dir():
            raise ValueError("selection output parent must be a real directory")
        self.root.mkdir(mode=0o700)
        _fsync_directory(self.root.parent)
        self.created = True
        _write_json_exclusive(
            self.status_path,
            {
                "complete": False,
                "locked_test": dict(_LOCKED_TEST_ABSENT),
                "schema_id": _STATUS_SCHEMA_ID,
                "state": "initialized",
                "validation_claimed": False,
            },
        )

    def write_once(self, name: str, record: Mapping[str, Any]) -> Path:
        if not self.created or name not in {
            "audit.json",
            "claim.json",
            "preflight.json",
            "selection.json",
        }:
            raise ValueError("unsupported write-once output record")
        target = self.root / name
        _write_json_exclusive(target, record)
        return target

    def update_status(self, **fields: Any) -> None:
        if not self.created:
            raise RuntimeError("selection output has not been created")
        record = {
            "complete": False,
            "locked_test": dict(_LOCKED_TEST_ABSENT),
            "schema_id": _STATUS_SCHEMA_ID,
            "state": "running",
            "validation_claimed": self.claimed,
            **fields,
        }
        _replace_json(self.status_path, record)

    def mark_claimed(self, *, claim_id: str, receipt_sha256: str) -> None:
        self.claimed = True
        self.update_status(
            claim_id=claim_id,
            claim_receipt_sha256=receipt_sha256,
            state="claimed",
        )

    def mark_complete(self, record: Mapping[str, Any]) -> None:
        if not self.claimed:
            raise RuntimeError("validation selection cannot complete without a claim")
        _replace_json(
            self.status_path,
            {
                **dict(record),
                "complete": True,
                "locked_test": dict(_LOCKED_TEST_ABSENT),
                "schema_id": _STATUS_SCHEMA_ID,
                "state": "complete",
                "validation_claimed": True,
            },
        )

    def mark_failed(self, error: BaseException) -> None:
        if not self.created:
            return
        _replace_json(
            self.status_path,
            {
                "complete": False,
                "error": str(error)[:1000],
                "error_type": type(error).__name__,
                "locked_test": dict(_LOCKED_TEST_ABSENT),
                "retry_policy": (
                    "forbidden; validation claim is burned"
                    if self.claimed
                    else "fresh output required"
                ),
                "schema_id": _STATUS_SCHEMA_ID,
                "state": "burned" if self.claimed else "error",
                "validation_claimed": self.claimed,
            },
        )


def _candidate_commitments(
    preflight: Mapping[str, Any],
    run_directories: Mapping[int, Path],
) -> tuple[CandidateCommitment, ...]:
    runs = preflight.get("runs") if isinstance(preflight, Mapping) else None
    if not isinstance(runs, list) or len(runs) != len(_TRAINING_SEEDS):
        raise ValueError("preflight must contain exactly three fixed runs")
    by_seed: dict[int, dict[int, Mapping[str, Any]]] = {}
    for raw_run, seed in zip(runs, _TRAINING_SEEDS, strict=True):
        if not isinstance(raw_run, Mapping) or raw_run.get("training_seed") != seed:
            raise ValueError("preflight run order or training seed changed")
        expected_root = run_directories[seed]
        if raw_run.get("run_directory") != str(expected_root):
            raise ValueError("preflight run directory differs from the explicit input")
        checkpoints = raw_run.get("checkpoints")
        if not isinstance(checkpoints, list) or len(checkpoints) != len(
            _CHECKPOINT_STEPS
        ):
            raise ValueError("preflight checkpoint inventory size changed")
        indexed: dict[int, Mapping[str, Any]] = {}
        for raw_checkpoint, expected_step in zip(
            checkpoints,
            _CHECKPOINT_STEPS,
            strict=True,
        ):
            if not isinstance(raw_checkpoint, Mapping) or set(raw_checkpoint) != (
                _CHECKPOINT_KEYS
            ):
                raise ValueError("preflight checkpoint fields changed")
            checkpoint = dict(raw_checkpoint)
            if (
                checkpoint["sequence"] != expected_step
                or checkpoint["checkpoint_name"] != f"checkpoint-{expected_step:012d}"
            ):
                raise ValueError("preflight checkpoint order or identity changed")
            indexed[expected_step] = checkpoint
        if set(indexed) != set(_CHECKPOINT_STEPS):
            raise ValueError("preflight checkpoint schedule changed")
        by_seed[seed] = indexed
    result = tuple(
        CandidateCommitment(
            checkpoint_step=step,
            training_seed=seed,
            run_directory=run_directories[seed],
            checkpoint=MappingProxyType(dict(by_seed[seed][step])),
        )
        for step in _CANDIDATE_STEPS
        for seed in _TRAINING_SEEDS
    )
    expected = tuple(
        (step, seed) for step in _CANDIDATE_STEPS for seed in _TRAINING_SEEDS
    )
    if tuple(item.inventory_key for item in result) != expected or len(result) != 60:
        raise RuntimeError("candidate inventory is not the fixed 20 by 3 grid")
    return result


def _receipt_record(receipt: Any) -> dict[str, Any]:
    method = getattr(receipt, "to_record", None)
    if not callable(method):
        raise TypeError("durable validation receipt must expose to_record()")
    record = method()
    _verify_sealed(record, "validation claim receipt")
    return record


class DefaultSelectionBackend:
    """Concrete CUDA/Newton backend, imported only after preimport controls."""

    def __init__(
        self,
        *,
        torch: Any,
        affinity: tuple[int, ...],
        cuda_runtime: Mapping[str, Any],
    ):
        self.torch = torch
        self.affinity = affinity
        self.cuda_runtime = dict(cuda_runtime)

    def prepare(self, arguments: SelectionArguments) -> PreparedSelection:
        from rexpolicy.stage0.data.grasp_lift_training_artifact import (
            load_grasp_lift_training_artifact,
        )
        from rexpolicy.stage0.evaluation.grasp_lift_checkpoint_loader import (
            GRASP_LIFT_EVALUATION_CHECKPOINT_SCHEMA_ID,
        )
        from rexpolicy.stage0.evaluation.grasp_lift_no_z_evaluation import (
            GRASP_LIFT_OFFLINE_DIAGNOSTIC_FLOW_STEPS,
            GRASP_LIFT_OFFLINE_DIAGNOSTIC_NOISE_NAMESPACE,
        )
        from rexpolicy.stage0.evaluation.grasp_lift_rollout import (
            GRASP_LIFT_FORMAL_FLOW_SAMPLE_STEPS,
            GRASP_LIFT_FORMAL_NOISE_VARIANTS,
            GRASP_LIFT_POLICY_NOISE_NAMESPACE,
        )
        from rexpolicy.stage0.envs.grasp_lift_action import (
            DEFAULT_GRASP_LIFT_ACTION_SCHEMA,
        )
        from rexpolicy.stage0.envs.grasp_lift_modes import (
            DEFAULT_GRASP_LIFT_AUTHORING_MODE,
        )
        from rexpolicy.stage0.envs.grasp_lift_oracle import GRASP_LIFT_ORACLE_ID
        from rexpolicy.stage0.grasp_lift_no_z import load_grasp_lift_no_z_config
        from rexpolicy.stage0.grasp_lift_no_z_selection import (
            GRASP_LIFT_NO_Z_SELECTION_PROTOCOL_SHA256,
            grasp_lift_no_z_selection_protocol_record,
        )
        from rexpolicy.stage0.grasp_lift_pilot import (
            GraspLiftPilotRuntimeConfig,
            grasp_lift_pilot_environment_config_record,
            grasp_lift_pilot_oracle_record,
            grasp_lift_pilot_task_metadata_record,
        )
        from rexpolicy.stage0.implementation_fingerprint import (
            capture_stage0_implementation_fingerprint,
        )

        implementation = capture_stage0_implementation_fingerprint(
            arguments.repository_root
        )
        config = load_grasp_lift_no_z_config(arguments.config)
        if (
            tuple(config.training_seeds) != _TRAINING_SEEDS
            or config.cpu_threads != _CPU_THREADS
            or config.interop_threads != _INTEROP_THREADS
            or config.cublas_workspace_config != _CUBLAS_WORKSPACE_CONFIG
            or config.nvidia_tf32_override != _TF32_OVERRIDE
            or config.torch_allow_tf32_cublas_override != _TF32_OVERRIDE
            or config.allow_tf32
            or not config.deterministic_algorithms
            or not config.require_cuda
        ):
            raise RuntimeError("formal evaluation runtime contract changed")
        artifact = load_grasp_lift_training_artifact(
            arguments.training_artifact,
            pilot_directory=arguments.pilot_directory,
        )
        newton_runtime = GraspLiftPilotRuntimeConfig(
            device=arguments.device,
            cpu_threads=_CPU_THREADS,
            num_envs=_ROLLOUT_EPISODES,
            reset_groups=_ROLLOUT_EPISODES,
            seed=7000,
            max_episode_steps=_MAX_EPISODE_STEPS,
            rigid_contacts_per_env=2048,
            triangle_pairs_per_env=200_000,
        )
        environment_config = grasp_lift_pilot_environment_config_record(
            newton_runtime
        )
        task_metadata = grasp_lift_pilot_task_metadata_record()
        oracle_record = grasp_lift_pilot_oracle_record()
        accepted_manifest = _strict_json(
            arguments.pilot_directory / "manifest.json"
        )
        accepted_simulator, accepted_simulator_sha256 = (
            _accepted_pilot_simulator_contract(
                accepted_manifest,
                expected_manifest_sha256=(
                    artifact.data.corpus.source_manifest_sha256
                ),
                evaluation_environment=environment_config,
                evaluation_task_metadata=task_metadata,
                evaluation_oracle=oracle_record,
                action_schema_sha256=DEFAULT_GRASP_LIFT_ACTION_SCHEMA.sha256,
                authoring_mode_sha256=DEFAULT_GRASP_LIFT_AUTHORING_MODE.sha256,
                oracle_id=GRASP_LIFT_ORACLE_ID,
            )
        )
        selection_protocol = grasp_lift_no_z_selection_protocol_record()
        if _canonical_fingerprint(selection_protocol) != (
            GRASP_LIFT_NO_Z_SELECTION_PROTOCOL_SHA256
        ):
            raise RuntimeError("selection protocol public hash changed")
        evaluator = _sealed(
            {
                "checkpoint_loader_schema_id": (
                    GRASP_LIFT_EVALUATION_CHECKPOINT_SCHEMA_ID
                ),
                "implementation": implementation.to_record(),
                "implementation_sha256": implementation.sha256,
                "locked_test": dict(_LOCKED_TEST_ABSENT),
                "offline_diagnostic": {
                    "flow_sample_steps": GRASP_LIFT_OFFLINE_DIAGNOSTIC_FLOW_STEPS,
                    "noise_namespace": GRASP_LIFT_OFFLINE_DIAGNOSTIC_NOISE_NAMESPACE,
                    "selection_role": "descriptive_only_never_ranked_or_admitted",
                },
                "rollout": {
                    "flow_sample_steps": GRASP_LIFT_FORMAL_FLOW_SAMPLE_STEPS,
                    "noise_namespace": GRASP_LIFT_POLICY_NOISE_NAMESPACE,
                    "noise_variants_per_reset": GRASP_LIFT_FORMAL_NOISE_VARIANTS,
                },
                "simulator_binding": {
                    "accepted_simulator_sha256": accepted_simulator_sha256,
                    "environment_config": environment_config,
                    "environment_config_sha256": _canonical_fingerprint(
                        environment_config
                    ),
                    "environment_equivalence": {
                        "allowed_runtime_variants": sorted(
                            _ENVIRONMENT_RUNTIME_VARIANTS
                        ),
                        "accepted_semantic_projection_sha256": (
                            _canonical_fingerprint(
                                _environment_semantic_projection(
                                    accepted_simulator["environment_config"]
                                )
                            )
                        ),
                    },
                    "oracle": oracle_record,
                    "task_metadata": task_metadata,
                },
                "schema_id": _EVALUATOR_SCHEMA_ID,
                "selection_protocol": selection_protocol,
                "selection_protocol_sha256": (
                    GRASP_LIFT_NO_Z_SELECTION_PROTOCOL_SHA256
                ),
            }
        )
        return PreparedSelection(
            artifact=artifact,
            config=config,
            normalization=artifact.normalization,
            implementation_record=implementation.to_record(),
            implementation_sha256=implementation.sha256,
            evaluator_record=evaluator,
            evaluator_sha256=evaluator["self_sha256"],
            selection_protocol_sha256=(GRASP_LIFT_NO_Z_SELECTION_PROTOCOL_SHA256),
            runtime_record={
                **self.cuda_runtime,
                "cpu_affinity": list(self.affinity),
                "cpu_threads": _CPU_THREADS,
                "interop_threads": _INTEROP_THREADS,
                "schema_id": ("rexpolicy/stage0-grasp-lift-no-z-selection-runtime/v1"),
            },
            newton_runtime=newton_runtime,
            environment_config_record=MappingProxyType(dict(environment_config)),
            task_metadata_record=MappingProxyType(dict(task_metadata)),
            oracle_record=MappingProxyType(dict(oracle_record)),
            accepted_simulator_record=MappingProxyType(dict(accepted_simulator)),
            accepted_simulator_sha256=accepted_simulator_sha256,
        )

    def preflight(
        self,
        arguments: SelectionArguments,
        prepared: PreparedSelection,
    ) -> dict[str, Any]:
        from rexpolicy.stage0.data.grasp_lift_validation import (
            preflight_grasp_lift_validation_selection,
        )

        return preflight_grasp_lift_validation_selection(
            arguments.run_directories,
            training_artifact=prepared.artifact,
        )

    def verify_preflight(
        self,
        preflight: Mapping[str, Any],
        prepared: PreparedSelection,
    ) -> None:
        from rexpolicy.stage0.data.grasp_lift_validation import (
            verify_grasp_lift_validation_preflight_record,
        )

        verify_grasp_lift_validation_preflight_record(
            preflight,
            training_artifact=prepared.artifact,
        )

    def build_claim(
        self,
        preflight: Mapping[str, Any],
        prepared: PreparedSelection,
    ) -> dict[str, Any]:
        from rexpolicy.stage0.data.grasp_lift_validation import (
            build_grasp_lift_validation_claim_record,
        )

        return build_grasp_lift_validation_claim_record(
            preflight,
            training_artifact=prepared.artifact,
            selection_protocol_sha256=prepared.selection_protocol_sha256,
            evaluator_implementation_sha256=prepared.evaluator_sha256,
        )

    def _claim_verifier(
        self,
        preflight: Mapping[str, Any],
        prepared: PreparedSelection,
    ) -> Any:
        from rexpolicy.stage0.data.grasp_lift_validation import (
            verify_grasp_lift_validation_claim_record,
        )

        def verify(claim: Mapping[str, Any]) -> None:
            verify_grasp_lift_validation_claim_record(
                claim,
                preflight=preflight,
                training_artifact=prepared.artifact,
            )

        return verify

    def persist_claim(
        self,
        arguments: SelectionArguments,
        claim: Mapping[str, Any],
        preflight: Mapping[str, Any],
        prepared: PreparedSelection,
    ) -> Any:
        from rexpolicy.stage0.data.grasp_lift_validation_claim import (
            persist_grasp_lift_validation_claim,
        )

        return persist_grasp_lift_validation_claim(
            arguments.repository_root,
            claim,
            verifier=self._claim_verifier(preflight, prepared),
        )

    def verify_claim_receipt(
        self,
        arguments: SelectionArguments,
        claim: Mapping[str, Any],
        preflight: Mapping[str, Any],
        prepared: PreparedSelection,
        receipt: Any,
    ) -> Any:
        from rexpolicy.stage0.data.grasp_lift_validation_claim import (
            verify_persisted_grasp_lift_validation_claim,
        )

        return verify_persisted_grasp_lift_validation_claim(
            arguments.repository_root,
            claim,
            verifier=self._claim_verifier(preflight, prepared),
            receipt=receipt,
        )

    def load_validation(
        self,
        arguments: SelectionArguments,
        prepared: PreparedSelection,
        preflight: Mapping[str, Any],
        claim: Mapping[str, Any],
        receipt: Any,
    ) -> Any:
        from rexpolicy.stage0.data.grasp_lift_validation import (
            load_claimed_grasp_lift_validation,
        )

        return load_claimed_grasp_lift_validation(
            arguments.pilot_directory,
            repository_root=arguments.repository_root,
            training_artifact=prepared.artifact,
            preflight=preflight,
            claim=claim,
            claim_receipt=receipt,
        )

    def build_budget(self, validation: Any) -> Any:
        from rexpolicy.stage0.evaluation.grasp_lift_rollout import (
            GraspLiftRolloutBudget,
        )

        reset_groups: list[tuple[str, int]] = []
        for member in validation.members:
            commitment = member.commitment
            provenance = member.model_trajectory.provenance
            if provenance.reset_seed != commitment.reset_seed:
                raise ValueError("validation model-view reset seed changed")
            reset_groups.append((provenance.reset_group_id, commitment.reset_seed))
        if tuple(seed for _, seed in reset_groups) != _VALIDATION_RESET_SEEDS:
            raise ValueError("validation budget seeds changed")
        budget = GraspLiftRolloutBudget.from_reset_groups(
            reset_groups,
            max_steps_per_episode=_MAX_EPISODE_STEPS,
        )
        if budget.episode_count != _ROLLOUT_EPISODES or budget.reset_count != 6:
            raise RuntimeError("formal rollout budget is not six resets by four noises")
        return budget

    def move_validation_batch(self, validation: Any, device: str) -> Any:
        return validation.normalized_batch.to(device, dtype=self.torch.float32)

    def build_environment(
        self,
        arguments: SelectionArguments,
        prepared: PreparedSelection,
        budget: Any,
    ) -> EnvironmentHandle:
        from rexpolicy.envs.groot_newton_env import (
            GrootNewtonEnv,
            GrootNewtonEnvConfig,
        )
        from rexpolicy.stage0.envs.grasp_lift_action import (
            DEFAULT_GRASP_LIFT_ACTION_SCHEMA,
            GRASP_LIFT_EFFECTIVE_ACTION_MASK,
        )
        from rexpolicy.stage0.envs.grasp_lift_modes import (
            DEFAULT_GRASP_LIFT_AUTHORING_MODE,
        )
        from rexpolicy.stage0.envs.grasp_lift_oracle import (
            GRASP_LIFT_ORACLE_ID,
            GraspLiftSuccessOracle,
        )
        from rexpolicy.stage0.envs.state_only import Stage0StateOnlyEnv

        runtime = prepared.newton_runtime
        expected_environment = dict(prepared.environment_config_record)
        expected_task = dict(prepared.task_metadata_record)
        oracle_record = dict(prepared.oracle_record)
        if _canonical_fingerprint(prepared.accepted_simulator_record) != (
            prepared.accepted_simulator_sha256
        ):
            raise RuntimeError("accepted pilot simulator record changed in memory")
        _verify_simulator_semantics(
            prepared.accepted_simulator_record,
            evaluation_environment=expected_environment,
            evaluation_task_metadata=expected_task,
            evaluation_oracle=oracle_record,
            action_schema_sha256=DEFAULT_GRASP_LIFT_ACTION_SCHEMA.sha256,
            authoring_mode_sha256=DEFAULT_GRASP_LIFT_AUTHORING_MODE.sha256,
            oracle_id=GRASP_LIFT_ORACLE_ID,
        )
        environment_config = GrootNewtonEnvConfig(**expected_environment)
        if asdict(environment_config) != expected_environment:
            raise RuntimeError("Newton environment config drifted from pilot contract")
        raw = GrootNewtonEnv(environment_config)
        try:
            if _canonical_fingerprint(raw.task_metadata) != (
                _canonical_fingerprint(expected_task)
            ):
                raise RuntimeError("Newton task metadata drifted from pilot contract")
            expected_mask = self.torch.tensor(
                GRASP_LIFT_EFFECTIVE_ACTION_MASK,
                dtype=self.torch.bool,
                device=arguments.device,
            )
            observed_mask = raw.effective_action_mask_torch()
            if (
                not isinstance(observed_mask, self.torch.Tensor)
                or observed_mask.device != expected_mask.device
                or not self.torch.equal(observed_mask, expected_mask)
                or expected_task["effective_action_mask"]
                != expected_mask.cpu().tolist()
                or not self.torch.equal(
                    prepared.normalization.effective_action_mask,
                    expected_mask.cpu(),
                )
            ):
                raise RuntimeError("Newton effective action mask changed")
            if raw.frames_per_action != (
                DEFAULT_GRASP_LIFT_AUTHORING_MODE.full_opposed_physics_frames
            ):
                raise RuntimeError("Newton frames-per-action contract changed")
            adapter = Stage0StateOnlyEnv(raw)
            if adapter.num_envs != budget.episode_count:
                raise RuntimeError("Newton world count differs from rollout budget")
            oracle = GraspLiftSuccessOracle(
                budget.episode_count,
                lift_height_m=oracle_record["lift_height_m"],
                success_hold_control_steps=oracle_record["success_hold_control_steps"],
                lateral_displacement_limit_m=oracle_record[
                    "lateral_displacement_limit_m"
                ],
                bottle_tilt_limit_rad=oracle_record["bottle_tilt_limit_rad"],
                grasp_confirm_physics_frames=oracle_record[
                    "grasp_confirm_physics_frames"
                ],
                drop_confirm_control_steps=oracle_record["drop_confirm_control_steps"],
            )
            if oracle_record["oracle_id"] != GRASP_LIFT_ORACLE_ID:
                raise RuntimeError("formal Grasp-Lift oracle identity changed")
            record = _sealed(
                {
                    "action_schema_sha256": DEFAULT_GRASP_LIFT_ACTION_SCHEMA.sha256,
                    "accepted_simulator_sha256": (
                        prepared.accepted_simulator_sha256
                    ),
                    "environment_config": expected_environment,
                    "environment_config_sha256": _canonical_fingerprint(
                        expected_environment
                    ),
                    "oracle": oracle_record,
                    "reward_enforcement": (
                        "reward_mode_none_and_every_rollout_transition_zero_checked"
                    ),
                    "rollout_budget": budget.to_record(),
                    "rollout_budget_sha256": budget.sha256,
                    "runtime": runtime.to_record(),
                    "schema_id": (
                        "rexpolicy/stage0-grasp-lift-no-z-selection-newton/v1"
                    ),
                    "state_only": True,
                    "task_metadata": expected_task,
                }
            )
            return EnvironmentHandle(
                raw_environment=raw,
                environment=adapter,
                oracle=oracle,
                record=record,
            )
        except BaseException:
            raw.close()
            raise

    def evaluate_candidate(
        self,
        candidate: CandidateCommitment,
        *,
        prepared: PreparedSelection,
        validation: Any,
        gpu_batch: Any,
        budget: Any,
        environment: EnvironmentHandle,
    ) -> CandidateEvaluation:
        from rexpolicy.stage0.evaluation.grasp_lift_checkpoint_loader import (
            load_grasp_lift_no_z_evaluation_checkpoint,
        )
        from rexpolicy.stage0.evaluation.grasp_lift_no_z_evaluation import (
            evaluate_grasp_lift_offline_action_diagnostic,
            grasp_lift_rollout_to_selection_evidence,
        )
        from rexpolicy.stage0.evaluation.grasp_lift_rollout import (
            rollout_grasp_lift_no_z_policy,
        )

        policy, checkpoint_record = load_grasp_lift_no_z_evaluation_checkpoint(
            candidate.run_directory,
            step=candidate.checkpoint_step,
            config=prepared.config,
            device=self.cuda_runtime["device"],
            preflight_checkpoint=candidate.checkpoint,
        )
        model_sha256 = checkpoint_record["model"]["payload_sha256"]
        if (
            model_sha256 != candidate.checkpoint["model_payload_sha256"]
            or checkpoint_record["checkpoint"]["step"] != candidate.checkpoint_step
        ):
            raise RuntimeError("model-only checkpoint load escaped preflight binding")
        diagnostic = evaluate_grasp_lift_offline_action_diagnostic(
            policy,
            gpu_batch,
            validation.normalization,
            validation_data_sha256=validation.data_sha256,
        )
        rollout = rollout_grasp_lift_no_z_policy(
            environment.environment,
            policy,
            validation.normalization,
            budget,
            oracle=environment.oracle,
        )
        evidence = grasp_lift_rollout_to_selection_evidence(
            rollout,
            checkpoint_step=candidate.checkpoint_step,
            training_seed=candidate.training_seed,
            model_sha256=model_sha256,
            offline_mse_diagnostic=diagnostic,
        )
        if evidence.inventory_key != candidate.inventory_key:
            raise RuntimeError("candidate evidence inventory key changed")
        record = _sealed(
            {
                "checkpoint_load": checkpoint_record,
                "checkpoint_step": candidate.checkpoint_step,
                "evidence": evidence.to_record(),
                "model_sha256": model_sha256,
                "offline_diagnostic": diagnostic,
                "rollout": rollout.to_record(),
                "rollout_sha256": rollout.sha256,
                "schema_id": (
                    "rexpolicy/stage0-grasp-lift-no-z-candidate-evaluation/v1"
                ),
                "training_seed": candidate.training_seed,
            }
        )
        return CandidateEvaluation(evidence=evidence, record=record)

    def select(self, evidence: Sequence[Any]) -> Any:
        from rexpolicy.stage0.grasp_lift_no_z_selection import (
            select_grasp_lift_no_z_checkpoint,
        )

        return select_grasp_lift_no_z_checkpoint(evidence)

    def verify_selection(self, record: Mapping[str, Any]) -> None:
        from rexpolicy.stage0.grasp_lift_no_z_selection import (
            verify_grasp_lift_no_z_selection_record,
        )

        verify_grasp_lift_no_z_selection_record(record)

    def build_audit(
        self,
        *,
        arguments: SelectionArguments,
        prepared: PreparedSelection,
        preflight: Mapping[str, Any],
        claim: Mapping[str, Any],
        receipt_record: Mapping[str, Any],
        validation: Any,
        budget: Any,
        environment: EnvironmentHandle,
        evaluations: Sequence[CandidateEvaluation],
        selection_record: Mapping[str, Any],
        output_file_hashes: Mapping[str, str],
    ) -> dict[str, Any]:
        return _sealed(
            {
                "candidate_evaluations": [dict(item.record) for item in evaluations],
                "candidate_inventory_sha256": preflight["candidate_inventory_sha256"],
                "claim": dict(claim),
                "claim_receipt": dict(receipt_record),
                "counts": {
                    "candidate_models": len(evaluations),
                    "checkpoint_steps": len(_CANDIDATE_STEPS),
                    "rollout_cells_per_model": budget.episode_count,
                    "training_seeds": len(_TRAINING_SEEDS),
                    "validation_resets": len(_VALIDATION_RESET_SEEDS),
                },
                "environment": dict(environment.record),
                "evaluator": dict(prepared.evaluator_record),
                "implementation": dict(prepared.implementation_record),
                "implementation_sha256": prepared.implementation_sha256,
                "locked_test": dict(_LOCKED_TEST_ABSENT),
                "output_files": dict(output_file_hashes),
                "preflight_sha256": preflight["self_sha256"],
                "purpose": "grasp_lift_no_z_model_selection/v1",
                "repository_root": str(arguments.repository_root),
                "runtime": dict(prepared.runtime_record),
                "schema_id": _AUDIT_SCHEMA_ID,
                "selection_sha256": selection_record["self_sha256"],
                "validation_data": dict(validation.record),
                "validation_read_scope": {
                    "engineering_excluded_opened": False,
                    "formal_validation_members_opened": 6,
                    "locked_test_opened": False,
                    "normalized_batch_device_copies": 1,
                },
            }
        )

    def close_environment(self, environment: EnvironmentHandle) -> None:
        environment.raw_environment.close()


def _run_selection(
    arguments: SelectionArguments,
    backend: Any,
) -> dict[str, Any]:
    output = SelectionOutput(arguments.output_directory)
    environment: EnvironmentHandle | None = None
    try:
        if arguments.output_directory.is_symlink() or (
            arguments.output_directory.exists()
        ):
            raise FileExistsError(
                f"selection output already exists: {arguments.output_directory}"
            )
        prepared = backend.prepare(arguments)
        output.create()
        preflight = backend.preflight(arguments, prepared)
        backend.verify_preflight(preflight, prepared)
        candidates = _candidate_commitments(preflight, arguments.run_directories)
        preflight_path = output.write_once("preflight.json", preflight)
        output.update_status(
            candidate_count=len(candidates),
            preflight_sha256=preflight["self_sha256"],
            state="preflight_complete",
        )

        claim = backend.build_claim(preflight, prepared)
        # Persist may have created a partial, deliberately non-reusable
        # sentinel even when it raises.  From this call boundary onward the
        # conservative state is always burned.
        output.claimed = True
        receipt = backend.persist_claim(
            arguments,
            claim,
            preflight,
            prepared,
        )
        verified_receipt = backend.verify_claim_receipt(
            arguments,
            claim,
            preflight,
            prepared,
            receipt,
        )
        if verified_receipt != receipt:
            raise RuntimeError("durable validation claim receipt changed")
        receipt_record = _receipt_record(receipt)
        claim_path = output.write_once("claim.json", claim)
        output.mark_claimed(
            claim_id=claim["claim_id"],
            receipt_sha256=receipt_record["self_sha256"],
        )

        # This is the first allowed held-out operation.  The loader itself
        # re-verifies the persisted receipt before opening exactly six shards.
        validation = backend.load_validation(
            arguments,
            prepared,
            preflight,
            claim,
            receipt,
        )
        budget = backend.build_budget(validation)
        if (
            budget.episode_count != _ROLLOUT_EPISODES
            or tuple(budget.reset_seeds[::4]) != _VALIDATION_RESET_SEEDS
        ):
            raise RuntimeError("formal rollout budget changed")
        environment = backend.build_environment(
            arguments,
            prepared,
            budget,
        )
        gpu_batch = backend.move_validation_batch(validation, arguments.device)

        evaluations: list[CandidateEvaluation] = []
        for index, candidate in enumerate(candidates, start=1):
            evaluation = backend.evaluate_candidate(
                candidate,
                prepared=prepared,
                validation=validation,
                gpu_batch=gpu_batch,
                budget=budget,
                environment=environment,
            )
            if not isinstance(evaluation, CandidateEvaluation):
                raise TypeError("candidate evaluator returned an unsupported value")
            evaluations.append(evaluation)
            output.update_status(
                candidate_count=len(candidates),
                completed_candidates=index,
                current_checkpoint_step=candidate.checkpoint_step,
                current_training_seed=candidate.training_seed,
                state="evaluating",
            )
        if tuple(item.evidence.inventory_key for item in evaluations) != tuple(
            item.inventory_key for item in candidates
        ):
            raise RuntimeError("completed evidence inventory changed")

        selection = backend.select(tuple(item.evidence for item in evaluations))
        selection_record = selection.to_record()
        backend.verify_selection(selection_record)
        _verify_sealed(selection_record, "selection record")
        selection_path = output.write_once("selection.json", selection_record)

        persisted_preflight = _strict_json(preflight_path)
        persisted_claim = _strict_json(claim_path)
        persisted_selection = _strict_json(selection_path)
        if (
            persisted_preflight != dict(preflight)
            or persisted_claim != dict(claim)
            or persisted_selection != dict(selection_record)
        ):
            raise RuntimeError("write-once selection inputs changed on disk")
        backend.verify_preflight(persisted_preflight, prepared)
        backend.verify_selection(persisted_selection)
        final_receipt = backend.verify_claim_receipt(
            arguments,
            persisted_claim,
            persisted_preflight,
            prepared,
            receipt,
        )
        if final_receipt != receipt:
            raise RuntimeError("durable validation claim changed during evaluation")

        output_hashes = {
            "claim.json": _file_sha256(claim_path),
            "preflight.json": _file_sha256(preflight_path),
            "selection.json": _file_sha256(selection_path),
        }
        audit = backend.build_audit(
            arguments=arguments,
            prepared=prepared,
            preflight=preflight,
            claim=claim,
            receipt_record=receipt_record,
            validation=validation,
            budget=budget,
            environment=environment,
            evaluations=tuple(evaluations),
            selection_record=selection_record,
            output_file_hashes=output_hashes,
        )
        _verify_sealed(audit, "selection audit")
        audit_path = output.write_once("audit.json", audit)
        persisted_audit = _strict_json(audit_path)
        if persisted_audit != dict(audit):
            raise RuntimeError("write-once selection audit changed on disk")
        _verify_sealed(persisted_audit, "persisted selection audit")
        audit_file_sha256 = _file_sha256(audit_path)
        backend.close_environment(environment)
        environment = None
        complete = {
            "audit_file_sha256": audit_file_sha256,
            "audit_sha256": audit["self_sha256"],
            "claim_id": claim["claim_id"],
            "evaluated_candidates": len(evaluations),
            "gate_passed": selection.gate_passed,
            "selected_checkpoint_step": selection.selected_checkpoint_step,
            "selection_file_sha256": output_hashes["selection.json"],
            "selection_sha256": selection_record["self_sha256"],
        }
        output.mark_complete(complete)
        return _strict_json(output.status_path)
    except BaseException as error:
        if environment is not None:
            try:
                backend.close_environment(environment)
            except Exception as close_error:  # noqa: BLE001 - preserve root failure
                print(
                    "failed to close selection environment: "
                    f"{type(close_error).__name__}: {close_error}",
                    file=sys.stderr,
                )
            environment = None
        try:
            output.mark_failed(error)
        except Exception as status_error:  # noqa: BLE001 - preserve root failure
            print(
                "failed to persist selection error status: "
                f"{type(status_error).__name__}: {status_error}",
                file=sys.stderr,
            )
        raise


def _restrict_preimport_runtime() -> tuple[int, ...]:
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = _CUBLAS_WORKSPACE_CONFIG
    os.environ["NVIDIA_TF32_OVERRIDE"] = _TF32_OVERRIDE
    os.environ["TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"] = _TF32_OVERRIDE
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[name] = str(_CPU_THREADS)
    if not hasattr(os, "sched_getaffinity") or not hasattr(os, "sched_setaffinity"):
        raise RuntimeError("formal selection requires Linux CPU affinity")
    available = tuple(sorted(os.sched_getaffinity(0)))
    if len(available) < _CPU_THREADS:
        raise RuntimeError(
            f"formal selection requires {_CPU_THREADS} CPUs; found {len(available)}"
        )
    selected = available[:_CPU_THREADS]
    os.sched_setaffinity(0, selected)
    observed = tuple(sorted(os.sched_getaffinity(0)))
    if observed != selected:
        raise RuntimeError("formal selection CPU affinity restriction failed")
    return observed


def _configure_torch(torch: Any, device_name: str) -> dict[str, Any]:
    if os.environ.get("WORLD_SIZE", "1") != "1" or os.environ.get("RANK", "0") != "0":
        raise RuntimeError("formal validation selection is single-process only")
    if not torch.cuda.is_available():
        raise RuntimeError("formal validation selection requires CUDA")
    device = torch.device(device_name)
    if device.type != "cuda" or device.index is None:
        raise ValueError("formal validation device must be explicit cuda:N")
    if device.index >= torch.cuda.device_count():
        raise ValueError("requested CUDA device is not visible")
    torch.cuda.set_device(device)
    torch.set_num_threads(_CPU_THREADS)
    torch.set_num_interop_threads(_INTEROP_THREADS)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    properties = torch.cuda.get_device_properties(device)
    return {
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "compute_capability": [properties.major, properties.minor],
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "device": str(device),
        "gpu_name": properties.name,
        "gpu_total_memory_bytes": properties.total_memory,
        "nvidia_tf32_override": os.environ.get("NVIDIA_TF32_OVERRIDE"),
        "torch_allow_tf32_cublas_override": os.environ.get(
            "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"
        ),
        "torch_compiled_cuda_version": torch.version.cuda,
        "visible_cuda_devices": torch.cuda.device_count(),
    }


def _absolute(value: str) -> Path:
    return Path(os.path.abspath(Path(value).expanduser()))


def _parse_args() -> SelectionArguments:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/stage0/grasp_lift_no_z_bc.json",
    )
    parser.add_argument("--pilot-directory", required=True)
    parser.add_argument("--training-artifact", required=True)
    parser.add_argument("--seed-31001-run", required=True)
    parser.add_argument("--seed-31002-run", required=True)
    parser.add_argument("--seed-31003-run", required=True)
    parser.add_argument("--output-directory", required=True)
    parser.add_argument("--device", required=True)
    parsed = parser.parse_args()
    repository_root = Path(__file__).resolve().parents[1]
    return SelectionArguments(
        repository_root=repository_root,
        config=_absolute(parsed.config),
        pilot_directory=_absolute(parsed.pilot_directory),
        training_artifact=_absolute(parsed.training_artifact),
        output_directory=_absolute(parsed.output_directory),
        device=parsed.device,
        run_directories={
            31_001: _absolute(parsed.seed_31001_run),
            31_002: _absolute(parsed.seed_31002_run),
            31_003: _absolute(parsed.seed_31003_run),
        },
    )


def main() -> None:
    arguments = _parse_args()
    affinity = _restrict_preimport_runtime()

    import torch

    cuda_runtime = _configure_torch(torch, arguments.device)
    backend = DefaultSelectionBackend(
        torch=torch,
        affinity=affinity,
        cuda_runtime=cuda_runtime,
    )
    result = _run_selection(arguments, backend)
    print(json.dumps(result, allow_nan=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
