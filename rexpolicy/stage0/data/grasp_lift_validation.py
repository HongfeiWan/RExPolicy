"""One-shot, leakage-safe validation data for Grasp-Lift no-z policies.

The preflight in this module is deliberately tensor-free with respect to the
held-out cohort.  It verifies all three equal-budget training runs before a
caller creates a durable claim.  Only :func:`load_claimed_grasp_lift_validation`
is allowed to open the six formal-validation shards, and it has no test-data
path or generic split API.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

import torch

from rexpolicy.stage0.checkpoint import (
    Stage0CheckpointHashes,
    Stage0CheckpointManager,
    file_sha256,
)
from rexpolicy.stage0.data.grasp_lift_evidence import (
    GraspLiftEvidence,
    grasp_lift_evidence_sha256,
    load_grasp_lift_evidence_sidecar,
    validate_grasp_lift_evidence_trajectory,
)
from rexpolicy.stage0.data.grasp_lift_training import (
    GRASP_LIFT_ENGINEERING_EXCLUDED_SEEDS,
    GRASP_LIFT_EXCLUDED_ROLE,
    GRASP_LIFT_FORMAL_VALIDATION_SEEDS,
    GRASP_LIFT_MODEL_ACTION_VIEW_SHA256,
    GRASP_LIFT_TRAINING_WINDOW_POLICY,
    GRASP_LIFT_VALIDATION_ROLE,
    GraspLiftRoleCommitment,
    _load_accepted_pilot_registry,
    grasp_lift_training_role,
    materialize_grasp_lift_model_trajectory,
)
from rexpolicy.stage0.data.grasp_lift_training_artifact import (
    GRASP_LIFT_TRAINING_ARTIFACT_STORAGE_POLICY,
    GraspLiftTrainingArtifact,
)
from rexpolicy.stage0.data.store import load_trajectory_shard
from rexpolicy.stage0.data.trajectory import (
    Stage0Trajectory,
    trajectory_corpus_sha256,
    trajectory_sha256,
)
from rexpolicy.stage0.data.windows import Stage0SuccessWindow, extract_success_windows
from rexpolicy.stage0.envs.grasp_lift_state_view import (
    DEFAULT_GRASP_LIFT_STATE_VIEW,
)
from rexpolicy.stage0.grasp_lift_no_z import (
    GRASP_LIFT_NO_Z_TRAINING_SEEDS,
    GraspLiftNoZConfig,
    GraspLiftNoZSampler,
    build_grasp_lift_no_z_policy,
    normalize_grasp_lift_no_z_batch_once,
)
from rexpolicy.stage0.grasp_lift_pilot import (
    seal_grasp_lift_pilot_record,
    verify_grasp_lift_pilot_record,
)
from rexpolicy.stage0.normalization import Stage0Normalization
from rexpolicy.stage0.trainers.batch import Stage0WindowBatch, collate_success_windows
from rexpolicy.stage0.types import canonical_fingerprint

if TYPE_CHECKING:
    from rexpolicy.stage0.data.grasp_lift_validation_claim import (
        GraspLiftValidationClaimReceipt,
    )

GRASP_LIFT_VALIDATION_PREFLIGHT_SCHEMA_ID = (
    "rexpolicy/stage0-grasp-lift-no-z-validation-preflight/v1"
)
GRASP_LIFT_VALIDATION_COHORT_SCHEMA_ID = (
    "rexpolicy/stage0-grasp-lift-formal-validation-cohort/v1"
)
GRASP_LIFT_VALIDATION_CLAIM_ID_SCHEMA_ID = (
    "rexpolicy/stage0-grasp-lift-validation-claim-identity/v1"
)
GRASP_LIFT_VALIDATION_CLAIM_SCHEMA_ID = (
    "rexpolicy/stage0-grasp-lift-validation-claim/v1"
)
GRASP_LIFT_VALIDATION_DATA_SCHEMA_ID = (
    "rexpolicy/stage0-grasp-lift-claimed-validation-data/v1"
)
GRASP_LIFT_VALIDATION_PURPOSE = "grasp_lift_no_z_model_selection/v1"
GRASP_LIFT_VALIDATION_CHECKPOINT_STEPS = tuple(range(0, 10_001, 500))
GRASP_LIFT_VALIDATION_CANDIDATE_STEPS = GRASP_LIFT_VALIDATION_CHECKPOINT_STEPS[1:]
GRASP_LIFT_LOCKED_TEST_ABSENT: Mapping[str, Any] = MappingProxyType(
    {
        "artifact": None,
        "consumed": False,
        "policy": "not_created",
    }
)

_RUN_CONTRACT_SCHEMA_ID = "rexpolicy/stage0-grasp-lift-no-z-run-contract/v1"
_RUN_STATUS_SCHEMA_ID = "rexpolicy/stage0-grasp-lift-no-z-status/v1"
_CHECKPOINT_EXTRA_SCHEMA_ID = "rexpolicy/stage0-grasp-lift-no-z-checkpoint-extra/v1"
_RUN_CONTRACT_KEYS = {
    "artifact_sha256",
    "audit_dataset_sha256",
    "checkpoint_hashes",
    "config",
    "config_sha256",
    "cuda_runtime",
    "freeze_record",
    "implementation",
    "implementation_sha256",
    "locked_test",
    "normalization_sha256",
    "protocol_sha256",
    "run_config_sha256",
    "schema_id",
    "self_sha256",
    "train_content_sha256",
    "training_seed",
    "validation_policy",
    "world_size",
}
_RUN_STATUS_KEYS = {
    "artifact_sha256",
    "attempt_index",
    "complete",
    "final_checkpoint",
    "global_step",
    "schema_id",
    "state",
    "train_content_sha256",
    "training_seed",
    "validation_consumed",
}
_CHECKPOINT_EXTRA_KEYS = {
    "audit_dataset_sha256",
    "implementation_sha256",
    "locked_test",
    "normalization_sha256",
    "run_contract_sha256",
    "schema_id",
    "train_content_sha256",
    "training_only",
    "validation_consumed",
}
_PILOT_MEMBER_KEYS = {
    "evidence_descriptor",
    "evidence_sha256",
    "final_authoring_phase",
    "formal_gate_eligible",
    "outcome",
    "reset_group_id",
    "reset_seed",
    "safety_violation",
    "split",
    "terminal_transition_from_phase",
    "trajectory_descriptor",
    "trajectory_id",
    "trajectory_sha256",
}
_PREFLIGHT_KEYS = {
    "candidate_inventory_sha256",
    "candidate_steps",
    "checkpoint_steps",
    "cohort_identity",
    "cohort_identity_sha256",
    "locked_test",
    "normalization_sha256",
    "runs",
    "schema_id",
    "self_sha256",
    "source_composite_corpus_sha256",
    "source_manifest_sha256",
    "train_content_sha256",
    "training_artifact_sha256",
}
_RUN_PREFLIGHT_KEYS = {
    "checkpoints",
    "implementation_sha256",
    "run_contract_file_sha256",
    "run_contract_sha256",
    "run_directory",
    "status_file_sha256",
    "training_seed",
    "world_size",
}
_CHECKPOINT_PREFLIGHT_KEYS = {
    "checkpoint_name",
    "checkpoint_sha256",
    "checksums_file_sha256",
    "complete_file_sha256",
    "manifest_file_sha256",
    "model_payload_sha256",
    "model_schema_sha256",
    "sequence",
}
_COHORT_KEYS = {
    "excluded_commitments",
    "formal_validation_commitments",
    "locked_test",
    "purpose",
    "role_registry_sha256",
    "schema_id",
    "self_sha256",
    "source_composite_corpus_sha256",
    "source_manifest_sha256",
}
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
_VALIDATION_DATA_KEYS = {
    "claim_sha256",
    "cohort_identity_sha256",
    "counts",
    "data_sha256",
    "locked_test",
    "members",
    "normalization_policy",
    "normalization_sha256",
    "normalized_batch_sha256",
    "schema_id",
    "self_sha256",
    "viewed_corpus_sha256",
    "window_content_sha256",
    "window_policy_sha256",
}


def _require_sha256(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _locked_test_absent_record() -> dict[str, Any]:
    return dict(GRASP_LIFT_LOCKED_TEST_ABSENT)


def _exact_mapping(value: Any, keys: set[str], context: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != keys:
        raise ValueError(f"{context} fields changed")
    return dict(value)


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
        raise TypeError(f"{context} must contain an object")
    return value


def _require_real_directory(path: str | Path, context: str) -> Path:
    requested = Path(path).expanduser()
    if requested.is_symlink() or not requested.is_dir():
        raise ValueError(f"{context} must be a directory and cannot be a symlink")
    return requested.resolve()


def _require_training_artifact(
    artifact: GraspLiftTrainingArtifact,
) -> GraspLiftTrainingArtifact:
    if not isinstance(artifact, GraspLiftTrainingArtifact):
        raise TypeError("training_artifact must be GraspLiftTrainingArtifact")
    if artifact.normalization.to_record() != artifact.data.normalization.to_record():
        raise ValueError("training artifact normalization changed")
    if artifact.train_content_sha256 != artifact.data.train_content_sha256:
        raise ValueError("training artifact train-content commitment changed")
    if artifact.audit_dataset_sha256 != artifact.data.dataset_sha256:
        raise ValueError("training artifact audit-data commitment changed")
    if (
        artifact.manifest.get("storage_policy")
        != GRASP_LIFT_TRAINING_ARTIFACT_STORAGE_POLICY
    ):
        raise ValueError("training artifact storage policy changed")
    if artifact.data.window_splits.validation or artifact.data.window_splits.test:
        raise ValueError("training artifact materialized held-out windows")
    for value, name in (
        (artifact.artifact_sha256, "training artifact SHA-256"),
        (artifact.audit_dataset_sha256, "training audit dataset SHA-256"),
        (artifact.train_content_sha256, "training content SHA-256"),
        (artifact.normalization.sha256, "training normalization SHA-256"),
    ):
        _require_sha256(value, name)
    return artifact


def _model_schema_sha256(model: torch.nn.Module) -> str:
    entries = [
        {"dtype": str(value.dtype), "name": name, "shape": list(value.shape)}
        for name, value in model.state_dict().items()
    ]
    encoded = json.dumps(
        entries,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _strict_checkpoint_inventory(root: Path) -> tuple[Path, ...]:
    checkpoints = root / "checkpoints"
    if checkpoints.is_symlink() or not checkpoints.is_dir():
        raise ValueError("run checkpoints directory is missing or a symlink")
    entries = tuple(checkpoints.iterdir())
    expected_names = {
        Stage0CheckpointManager.checkpoint_name(step)
        for step in GRASP_LIFT_VALIDATION_CHECKPOINT_STEPS
    }
    if {entry.name for entry in entries} != expected_names:
        raise ValueError(
            "run checkpoint inventory is not exactly step 0 plus 20 candidates"
        )
    if any(entry.is_symlink() or not entry.is_dir() for entry in entries):
        raise ValueError("run checkpoint inventory contains a symlink or non-directory")
    return tuple(sorted(entries, key=lambda item: item.name))


def _strict_checkpoint_tree(path: Path) -> None:
    entries = tuple(path.rglob("*"))
    if any(entry.is_symlink() for entry in entries):
        raise ValueError("checkpoint tree contains a symlink")
    directories = {str(entry.relative_to(path)) for entry in entries if entry.is_dir()}
    if directories != {"models", "optimizers", "ranks"}:
        raise ValueError("checkpoint directory inventory changed")
    if any(not entry.is_dir() and not entry.is_file() for entry in entries):
        raise ValueError("checkpoint tree contains an unsupported filesystem entry")


def _validate_run_contract(
    contract: Mapping[str, Any],
    *,
    training_seed: int,
    artifact: GraspLiftTrainingArtifact,
    expected_model_schema_sha256: str,
) -> tuple[GraspLiftNoZConfig, Stage0CheckpointHashes]:
    record = _exact_mapping(contract, _RUN_CONTRACT_KEYS, "no-z run contract")
    verify_grasp_lift_pilot_record(record, "no-z run contract")
    if record["schema_id"] != _RUN_CONTRACT_SCHEMA_ID:
        raise ValueError("no-z run contract schema changed")
    if record["training_seed"] != training_seed:
        raise ValueError("no-z run directory is bound to the wrong training seed")
    if record["locked_test"] != "not_created" or record["validation_policy"] != (
        "not opened during fixed-budget training"
    ):
        raise ValueError("no-z training touched held-out or locked-test data")
    world_size = record["world_size"]
    if (
        isinstance(world_size, bool)
        or not isinstance(world_size, int)
        or world_size < 1
    ):
        raise ValueError("no-z run world_size is invalid")

    config = GraspLiftNoZConfig.from_record(record["config"])
    config.require_training_seed(training_seed)
    if config.sha256 != record["config_sha256"]:
        raise ValueError("no-z run config hash changed")
    if (
        len(artifact.data.train_trajectories) != config.expected_train_trajectories
        or len(artifact.data.window_splits.train) != config.expected_train_windows
    ):
        raise ValueError("training artifact no longer matches the no-z data budget")
    if record["run_config_sha256"] != config.run_sha256(
        training_seed,
        world_size=world_size,
    ):
        raise ValueError("no-z run-specific config hash changed")
    if config.global_batch_size % world_size:
        raise ValueError("no-z world_size no longer divides the global batch")

    implementation = record["implementation"]
    if (
        not isinstance(implementation, Mapping)
        or canonical_fingerprint(implementation) != record["implementation_sha256"]
    ):
        raise ValueError("no-z implementation record hash changed")
    _require_sha256(record["implementation_sha256"], "implementation_sha256")
    if record["artifact_sha256"] != artifact.artifact_sha256:
        raise ValueError("no-z run training artifact changed")
    if record["audit_dataset_sha256"] != artifact.audit_dataset_sha256:
        raise ValueError("no-z run audit dataset changed")
    if record["train_content_sha256"] != artifact.train_content_sha256:
        raise ValueError("no-z run train content changed")
    if record["normalization_sha256"] != artifact.normalization.sha256:
        raise ValueError("no-z run normalization changed")

    sampler = GraspLiftNoZSampler(
        training_seed=training_seed,
        dataset_size=config.expected_train_windows,
        global_batch_size=config.global_batch_size,
    )
    expected_protocol_sha256 = canonical_fingerprint(
        {
            "action_feature_mask": artifact.normalization.effective_action_mask.tolist(),
            "action_model_view_sha256": GRASP_LIFT_MODEL_ACTION_VIEW_SHA256,
            "freeze_record": record["freeze_record"],
            "implementation_sha256": record["implementation_sha256"],
            "normalization_sha256": artifact.normalization.sha256,
            "run_config_sha256": record["run_config_sha256"],
            "sampler": sampler.to_record(),
            "schema_id": "rexpolicy/stage0-grasp-lift-no-z-protocol/v1",
            "training_membership_sha256": (
                artifact.data.corpus.training_membership_sha256
            ),
        }
    )
    if record["protocol_sha256"] != expected_protocol_sha256:
        raise ValueError("no-z training protocol hash changed")
    expected_experiment_sha256 = canonical_fingerprint(
        {
            "implementation_sha256": record["implementation_sha256"],
            "protocol_sha256": expected_protocol_sha256,
            "run_config_sha256": record["run_config_sha256"],
            "schema_id": "rexpolicy/stage0-grasp-lift-no-z-experiment/v1",
            "train_content_sha256": artifact.train_content_sha256,
        }
    )
    hashes = Stage0CheckpointHashes.from_record(record["checkpoint_hashes"])
    expected_hashes = Stage0CheckpointHashes(
        config_sha256=expected_experiment_sha256,
        state_schema_sha256=DEFAULT_GRASP_LIFT_STATE_VIEW.sha256,
        dataset_sha256=artifact.train_content_sha256,
        split_sha256=expected_protocol_sha256,
    )
    if hashes != expected_hashes:
        raise ValueError("no-z checkpoint semantic hashes changed")

    expected_policy, expected_freeze = build_grasp_lift_no_z_policy(
        config,
        device="cpu",
    )
    if record["freeze_record"] != expected_freeze:
        raise ValueError("no-z frozen latent branch changed")
    if _model_schema_sha256(expected_policy) != expected_model_schema_sha256:
        raise RuntimeError("internally inconsistent expected no-z model schema")
    return config, hashes


def _validate_complete_status(
    status: Mapping[str, Any],
    *,
    training_seed: int,
    artifact: GraspLiftTrainingArtifact,
) -> None:
    record = _exact_mapping(status, _RUN_STATUS_KEYS, "no-z run status")
    expected_final = Stage0CheckpointManager.checkpoint_name(
        GRASP_LIFT_VALIDATION_CHECKPOINT_STEPS[-1]
    )
    if (
        record["schema_id"] != _RUN_STATUS_SCHEMA_ID
        or record["complete"] is not True
        or record["state"] != "complete"
        or record["global_step"] != GRASP_LIFT_VALIDATION_CHECKPOINT_STEPS[-1]
        or record["final_checkpoint"] != expected_final
        or record["training_seed"] != training_seed
        or record["validation_consumed"] is not False
        or record["artifact_sha256"] != artifact.artifact_sha256
        or record["train_content_sha256"] != artifact.train_content_sha256
    ):
        raise ValueError("no-z run is not an exact complete, unconsumed training run")
    if (
        isinstance(record["attempt_index"], bool)
        or not isinstance(record["attempt_index"], int)
        or record["attempt_index"] < 0
    ):
        raise ValueError("no-z run attempt_index is invalid")


def _validate_checkpoint_manifest(
    manifest: Mapping[str, Any],
    *,
    path: Path,
    sequence: int,
    contract: Mapping[str, Any],
    hashes: Stage0CheckpointHashes,
    artifact: GraspLiftTrainingArtifact,
    expected_model_schema_sha256: str,
) -> dict[str, Any]:
    if manifest["sequence"] != sequence or manifest["checkpoint_name"] != path.name:
        raise ValueError("checkpoint sequence identity changed")
    expected_phase = (
        "complete"
        if sequence == GRASP_LIFT_VALIDATION_CHECKPOINT_STEPS[-1]
        else "no_z_policy"
    )
    if manifest["phase"] != expected_phase or manifest["steps"] != {
        "global": sequence,
        "no_z_policy": sequence,
    }:
        raise ValueError("checkpoint phase or step record changed")
    if manifest["world_size"] != contract["world_size"]:
        raise ValueError("checkpoint world_size changed")
    if Stage0CheckpointHashes.from_record(manifest["hashes"]) != hashes:
        raise ValueError("checkpoint semantic hashes differ from the run contract")
    if set(manifest["models"]) != {"no_z_policy"} or set(manifest["optimizers"]) != {
        "no_z_policy"
    }:
        raise ValueError("checkpoint component inventory changed")
    model = manifest["models"]["no_z_policy"]
    if (
        not isinstance(model, Mapping)
        or model.get("file") != "models/no_z_policy.pt"
        or model.get("type") != "rexpolicy.stage0.runtime.Stage0FlowPolicy"
        or model.get("schema_sha256") != expected_model_schema_sha256
    ):
        raise ValueError("checkpoint no-z model schema changed")
    optimizer = manifest["optimizers"]["no_z_policy"]
    if (
        not isinstance(optimizer, Mapping)
        or optimizer.get("file") != "optimizers/no_z_policy.pt"
        or optimizer.get("type") != "torch.optim.adamw.AdamW"
    ):
        raise ValueError("checkpoint no-z optimizer schema changed")
    extra = _exact_mapping(
        manifest["extra"],
        _CHECKPOINT_EXTRA_KEYS,
        "no-z checkpoint extra",
    )
    expected_extra = {
        "audit_dataset_sha256": artifact.audit_dataset_sha256,
        "implementation_sha256": contract["implementation_sha256"],
        "locked_test": "not_created",
        "normalization_sha256": artifact.normalization.sha256,
        "run_contract_sha256": contract["self_sha256"],
        "schema_id": _CHECKPOINT_EXTRA_SCHEMA_ID,
        "train_content_sha256": artifact.train_content_sha256,
        "training_only": True,
        "validation_consumed": False,
    }
    if extra != expected_extra:
        raise ValueError("checkpoint training-only commitment changed")
    return dict(model)


def grasp_lift_validation_cohort_identity(
    training_artifact: GraspLiftTrainingArtifact,
) -> dict[str, Any]:
    """Build the global identity of the one consumable six-member cohort."""
    artifact = _require_training_artifact(training_artifact)
    corpus = artifact.data.corpus
    if tuple(item.reset_seed for item in corpus.validation_commitments) != (
        GRASP_LIFT_FORMAL_VALIDATION_SEEDS
    ):
        raise ValueError("formal validation commitments changed")
    if tuple(item.reset_seed for item in corpus.excluded_commitments) != (
        GRASP_LIFT_ENGINEERING_EXCLUDED_SEEDS
    ):
        raise ValueError("engineering-excluded commitments changed")
    return seal_grasp_lift_pilot_record(
        {
            "excluded_commitments": [
                item.to_record() for item in corpus.excluded_commitments
            ],
            "formal_validation_commitments": [
                item.to_record() for item in corpus.validation_commitments
            ],
            "locked_test": _locked_test_absent_record(),
            "purpose": GRASP_LIFT_VALIDATION_PURPOSE,
            "role_registry_sha256": corpus.role_registry_sha256,
            "schema_id": GRASP_LIFT_VALIDATION_COHORT_SCHEMA_ID,
            "source_composite_corpus_sha256": corpus.source_composite_corpus_sha256,
            "source_manifest_sha256": corpus.source_manifest_sha256,
        }
    )


def grasp_lift_validation_claim_id(cohort_identity: Mapping[str, Any]) -> str:
    """Return a cohort-global key that deliberately excludes models/protocols."""
    record = _exact_mapping(cohort_identity, _COHORT_KEYS, "validation cohort")
    verify_grasp_lift_pilot_record(record, "validation cohort")
    if (
        record["schema_id"] != GRASP_LIFT_VALIDATION_COHORT_SCHEMA_ID
        or record["purpose"] != GRASP_LIFT_VALIDATION_PURPOSE
        or record["locked_test"] != GRASP_LIFT_LOCKED_TEST_ABSENT
    ):
        raise ValueError("validation cohort identity changed")
    return canonical_fingerprint(
        {
            "cohort_identity_sha256": record["self_sha256"],
            "purpose": GRASP_LIFT_VALIDATION_PURPOSE,
            "schema_id": GRASP_LIFT_VALIDATION_CLAIM_ID_SCHEMA_ID,
        }
    )


def _selection_cohort_identity(
    artifact: GraspLiftTrainingArtifact,
    external_cohort_identity: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if external_cohort_identity is None:
        return grasp_lift_validation_cohort_identity(artifact)
    record = _exact_mapping(
        external_cohort_identity,
        _COHORT_KEYS,
        "external validation cohort",
    )
    # Computing the claim ID performs the complete schema, purpose, locked-test,
    # and self-hash verification without opening any tensor shard.
    grasp_lift_validation_claim_id(record)
    if record["excluded_commitments"] != []:
        raise ValueError("external validation cohort cannot contain exclusions")
    raw_commitments = record["formal_validation_commitments"]
    if not isinstance(raw_commitments, list) or len(raw_commitments) != 6:
        raise ValueError("external validation cohort must commit exactly six members")
    commitments: list[GraspLiftRoleCommitment] = []
    for raw_commitment in raw_commitments:
        commitment_record = _exact_mapping(
            raw_commitment,
            {
                "evidence_sha256",
                "raw_trajectory_sha256",
                "reset_seed",
                "role",
                "trajectory_id",
            },
            "external validation commitment",
        )
        commitment = GraspLiftRoleCommitment(**commitment_record)
        if commitment.role != GRASP_LIFT_VALIDATION_ROLE:
            raise ValueError("external cohort member role changed")
        commitments.append(commitment)
    seeds = tuple(item.reset_seed for item in commitments)
    trajectory_ids = tuple(item.trajectory_id for item in commitments)
    if (
        len(set(seeds)) != len(seeds)
        or tuple(sorted(seeds)) != seeds
        or set(seeds).intersection(GRASP_LIFT_FORMAL_VALIDATION_SEEDS)
        or len(set(trajectory_ids)) != len(trajectory_ids)
    ):
        raise ValueError("external validation cohort identity or seed order changed")
    expected_role_registry_sha256 = canonical_fingerprint(
        {
            "roles": [item.to_record() for item in commitments],
            "schema_id": "rexpolicy/stage0-grasp-lift-frozen-roles/v1",
        }
    )
    if record["role_registry_sha256"] != expected_role_registry_sha256:
        raise ValueError("external validation role registry changed")
    for name in ("source_composite_corpus_sha256", "source_manifest_sha256"):
        _require_sha256(record[name], f"external validation {name}")
    if (
        record["source_manifest_sha256"]
        == artifact.data.corpus.source_manifest_sha256
        or record["source_composite_corpus_sha256"]
        == artifact.data.corpus.source_composite_corpus_sha256
    ):
        raise ValueError("external validation cohort must be independently authored")
    return record


def preflight_grasp_lift_validation_selection(
    run_directories: Mapping[int, str | Path],
    *,
    training_artifact: GraspLiftTrainingArtifact,
    external_cohort_identity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Verify three complete runs without opening any held-out tensor shard."""
    artifact = _require_training_artifact(training_artifact)
    if not isinstance(run_directories, Mapping) or set(run_directories) != set(
        GRASP_LIFT_NO_Z_TRAINING_SEEDS
    ):
        raise ValueError("run_directories must explicitly map the three fixed seeds")

    expected_config = GraspLiftNoZConfig.from_record(
        _read_json(
            _require_real_directory(
                run_directories[GRASP_LIFT_NO_Z_TRAINING_SEEDS[0]],
                "no-z run",
            )
            / "run-contract.json",
            "no-z run contract",
        )["config"]
    )
    expected_policy, _ = build_grasp_lift_no_z_policy(expected_config, device="cpu")
    expected_model_schema_sha256 = _model_schema_sha256(expected_policy)
    run_records: list[dict[str, Any]] = []
    common: dict[str, Any] | None = None

    for training_seed in GRASP_LIFT_NO_Z_TRAINING_SEEDS:
        root = _require_real_directory(run_directories[training_seed], "no-z run")
        contract_path = root / "run-contract.json"
        status_path = root / "status.json"
        contract = _read_json(contract_path, "no-z run contract")
        config, hashes = _validate_run_contract(
            contract,
            training_seed=training_seed,
            artifact=artifact,
            expected_model_schema_sha256=expected_model_schema_sha256,
        )
        if config.to_record() != expected_config.to_record():
            raise ValueError("the three no-z runs do not share one fixed config")
        status = _read_json(status_path, "no-z run status")
        _validate_complete_status(
            status,
            training_seed=training_seed,
            artifact=artifact,
        )
        shared = {
            "artifact_sha256": contract["artifact_sha256"],
            "audit_dataset_sha256": contract["audit_dataset_sha256"],
            "config_sha256": contract["config_sha256"],
            "freeze_record": contract["freeze_record"],
            "implementation": contract["implementation"],
            "implementation_sha256": contract["implementation_sha256"],
            "normalization_sha256": contract["normalization_sha256"],
            "train_content_sha256": contract["train_content_sha256"],
            "world_size": contract["world_size"],
        }
        if common is None:
            common = shared
        elif shared != common:
            raise ValueError("the three no-z run commitments are inconsistent")

        manager = Stage0CheckpointManager(root)
        checkpoint_records: list[dict[str, Any]] = []
        for path, sequence in zip(
            _strict_checkpoint_inventory(root),
            GRASP_LIFT_VALIDATION_CHECKPOINT_STEPS,
        ):
            _strict_checkpoint_tree(path)
            manifest = manager.verify(path)
            strict_manifest = _read_json(path / "manifest.json", "checkpoint manifest")
            if strict_manifest != manifest:
                raise ValueError("checkpoint manifest parsing is not canonical")
            model = _validate_checkpoint_manifest(
                manifest,
                path=path,
                sequence=sequence,
                contract=contract,
                hashes=hashes,
                artifact=artifact,
                expected_model_schema_sha256=expected_model_schema_sha256,
            )
            checksums = _read_json(path / "checksums.json", "checkpoint checksums")
            complete = _read_json(path / "COMPLETE", "checkpoint COMPLETE")
            model_payload_sha256 = checksums.get("files", {}).get(model["file"])
            _require_sha256(model_payload_sha256, "checkpoint model payload SHA-256")
            checkpoint_commit = {
                "checksums_file_sha256": file_sha256(path / "checksums.json"),
                "complete_file_sha256": file_sha256(path / "COMPLETE"),
                "manifest_file_sha256": file_sha256(path / "manifest.json"),
                "model_payload_sha256": model_payload_sha256,
                "sequence": sequence,
            }
            if complete.get("sequence") != sequence:
                raise ValueError("checkpoint COMPLETE sequence changed")
            checkpoint_records.append(
                {
                    "checkpoint_name": path.name,
                    "checkpoint_sha256": canonical_fingerprint(checkpoint_commit),
                    **checkpoint_commit,
                    "model_schema_sha256": model["schema_sha256"],
                }
            )
        run_records.append(
            {
                "checkpoints": checkpoint_records,
                "implementation_sha256": contract["implementation_sha256"],
                "run_contract_file_sha256": file_sha256(contract_path),
                "run_contract_sha256": contract["self_sha256"],
                "run_directory": str(root),
                "status_file_sha256": file_sha256(status_path),
                "training_seed": training_seed,
                "world_size": contract["world_size"],
            }
        )

    cohort = _selection_cohort_identity(artifact, external_cohort_identity)
    candidate_inventory = [
        {
            "checkpoint_sha256": checkpoint["checkpoint_sha256"],
            "sequence": checkpoint["sequence"],
            "training_seed": run["training_seed"],
        }
        for run in run_records
        for checkpoint in run["checkpoints"]
        if checkpoint["sequence"] in GRASP_LIFT_VALIDATION_CANDIDATE_STEPS
    ]
    return seal_grasp_lift_pilot_record(
        {
            "candidate_inventory_sha256": canonical_fingerprint(candidate_inventory),
            "candidate_steps": list(GRASP_LIFT_VALIDATION_CANDIDATE_STEPS),
            "checkpoint_steps": list(GRASP_LIFT_VALIDATION_CHECKPOINT_STEPS),
            "cohort_identity": cohort,
            "cohort_identity_sha256": cohort["self_sha256"],
            "locked_test": _locked_test_absent_record(),
            "normalization_sha256": artifact.normalization.sha256,
            "runs": run_records,
            "schema_id": GRASP_LIFT_VALIDATION_PREFLIGHT_SCHEMA_ID,
            "source_composite_corpus_sha256": cohort[
                "source_composite_corpus_sha256"
            ],
            "source_manifest_sha256": cohort["source_manifest_sha256"],
            "train_content_sha256": artifact.train_content_sha256,
            "training_artifact_sha256": artifact.artifact_sha256,
        }
    )


def verify_grasp_lift_validation_preflight_record(
    preflight: Mapping[str, Any],
    *,
    training_artifact: GraspLiftTrainingArtifact,
    external_cohort_identity: Mapping[str, Any] | None = None,
) -> None:
    artifact = _require_training_artifact(training_artifact)
    record = _exact_mapping(preflight, _PREFLIGHT_KEYS, "validation preflight")
    verify_grasp_lift_pilot_record(record, "validation preflight")
    if (
        record["schema_id"] != GRASP_LIFT_VALIDATION_PREFLIGHT_SCHEMA_ID
        or record["locked_test"] != GRASP_LIFT_LOCKED_TEST_ABSENT
        or record["checkpoint_steps"] != list(GRASP_LIFT_VALIDATION_CHECKPOINT_STEPS)
        or record["candidate_steps"] != list(GRASP_LIFT_VALIDATION_CANDIDATE_STEPS)
        or record["training_artifact_sha256"] != artifact.artifact_sha256
        or record["train_content_sha256"] != artifact.train_content_sha256
        or record["normalization_sha256"] != artifact.normalization.sha256
    ):
        raise ValueError("validation preflight binding changed")
    expected_cohort = _selection_cohort_identity(
        artifact,
        external_cohort_identity,
    )
    if (
        record["cohort_identity"] != expected_cohort
        or record["cohort_identity_sha256"] != expected_cohort["self_sha256"]
        or record["source_manifest_sha256"]
        != expected_cohort["source_manifest_sha256"]
        or record["source_composite_corpus_sha256"]
        != expected_cohort["source_composite_corpus_sha256"]
    ):
        raise ValueError("validation preflight cohort changed")
    runs = record["runs"]
    if not isinstance(runs, list) or len(runs) != 3:
        raise ValueError("validation preflight must contain three runs")
    inventory: list[dict[str, Any]] = []
    common: dict[str, Any] | None = None
    run_directories: set[str] = set()
    for raw_run, training_seed in zip(runs, GRASP_LIFT_NO_Z_TRAINING_SEEDS):
        run = _exact_mapping(raw_run, _RUN_PREFLIGHT_KEYS, "preflight run")
        if run["training_seed"] != training_seed:
            raise ValueError("preflight run order or seed changed")
        _require_sha256(run["implementation_sha256"], "implementation_sha256")
        _require_sha256(run["run_contract_file_sha256"], "run contract file hash")
        _require_sha256(run["run_contract_sha256"], "run contract hash")
        _require_sha256(run["status_file_sha256"], "status file hash")
        if (
            isinstance(run["world_size"], bool)
            or not isinstance(run["world_size"], int)
            or run["world_size"] < 1
        ):
            raise ValueError("preflight world_size is invalid")
        if (
            not isinstance(run["run_directory"], str)
            or not Path(run["run_directory"]).is_absolute()
        ):
            raise ValueError("preflight run directory must be absolute")
        if run["run_directory"] in run_directories:
            raise ValueError("preflight run directories must be distinct")
        run_directories.add(run["run_directory"])
        checkpoints = run["checkpoints"]
        if not isinstance(checkpoints, list) or len(checkpoints) != len(
            GRASP_LIFT_VALIDATION_CHECKPOINT_STEPS
        ):
            raise ValueError("preflight checkpoint count changed")
        for raw_checkpoint, sequence in zip(
            checkpoints,
            GRASP_LIFT_VALIDATION_CHECKPOINT_STEPS,
        ):
            checkpoint = _exact_mapping(
                raw_checkpoint,
                _CHECKPOINT_PREFLIGHT_KEYS,
                "preflight checkpoint",
            )
            if checkpoint["sequence"] != sequence or checkpoint[
                "checkpoint_name"
            ] != Stage0CheckpointManager.checkpoint_name(sequence):
                raise ValueError("preflight checkpoint schedule changed")
            for name in (
                "checkpoint_sha256",
                "checksums_file_sha256",
                "complete_file_sha256",
                "manifest_file_sha256",
                "model_payload_sha256",
                "model_schema_sha256",
            ):
                _require_sha256(checkpoint[name], f"preflight {name}")
            shared = {
                "implementation_sha256": run["implementation_sha256"],
                "model_schema_sha256": checkpoint["model_schema_sha256"],
                "world_size": run["world_size"],
            }
            if common is None:
                common = shared
            elif shared != common:
                raise ValueError("preflight run commitments are inconsistent")
            expected_checkpoint_sha256 = canonical_fingerprint(
                {
                    "checksums_file_sha256": checkpoint["checksums_file_sha256"],
                    "complete_file_sha256": checkpoint["complete_file_sha256"],
                    "manifest_file_sha256": checkpoint["manifest_file_sha256"],
                    "model_payload_sha256": checkpoint["model_payload_sha256"],
                    "sequence": sequence,
                }
            )
            if checkpoint["checkpoint_sha256"] != expected_checkpoint_sha256:
                raise ValueError("preflight checkpoint self-commitment changed")
            if sequence in GRASP_LIFT_VALIDATION_CANDIDATE_STEPS:
                inventory.append(
                    {
                        "checkpoint_sha256": checkpoint["checkpoint_sha256"],
                        "sequence": sequence,
                        "training_seed": training_seed,
                    }
                )
    if record["candidate_inventory_sha256"] != canonical_fingerprint(inventory):
        raise ValueError("validation candidate inventory hash changed")


def build_grasp_lift_validation_claim_record(
    preflight: Mapping[str, Any],
    *,
    training_artifact: GraspLiftTrainingArtifact,
    selection_protocol_sha256: str,
    evaluator_implementation_sha256: str,
    external_cohort_identity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build, but do not persist, the record used by an O_EXCL caller."""
    verify_grasp_lift_validation_preflight_record(
        preflight,
        training_artifact=training_artifact,
        external_cohort_identity=external_cohort_identity,
    )
    _require_sha256(selection_protocol_sha256, "selection_protocol_sha256")
    _require_sha256(
        evaluator_implementation_sha256,
        "evaluator_implementation_sha256",
    )
    cohort = preflight["cohort_identity"]
    return seal_grasp_lift_pilot_record(
        {
            "candidate_inventory_sha256": preflight["candidate_inventory_sha256"],
            "claim_id": grasp_lift_validation_claim_id(cohort),
            "cohort_identity_sha256": cohort["self_sha256"],
            "evaluator_implementation_sha256": evaluator_implementation_sha256,
            "locked_test": _locked_test_absent_record(),
            "preflight_sha256": preflight["self_sha256"],
            "purpose": GRASP_LIFT_VALIDATION_PURPOSE,
            "schema_id": GRASP_LIFT_VALIDATION_CLAIM_SCHEMA_ID,
            "selection_protocol_sha256": selection_protocol_sha256,
        }
    )


def verify_grasp_lift_validation_claim_record(
    claim: Mapping[str, Any],
    *,
    preflight: Mapping[str, Any],
    training_artifact: GraspLiftTrainingArtifact,
    external_cohort_identity: Mapping[str, Any] | None = None,
) -> None:
    verify_grasp_lift_validation_preflight_record(
        preflight,
        training_artifact=training_artifact,
        external_cohort_identity=external_cohort_identity,
    )
    record = _exact_mapping(claim, _CLAIM_KEYS, "validation claim")
    verify_grasp_lift_pilot_record(record, "validation claim")
    for name in (
        "candidate_inventory_sha256",
        "claim_id",
        "cohort_identity_sha256",
        "evaluator_implementation_sha256",
        "preflight_sha256",
        "selection_protocol_sha256",
    ):
        _require_sha256(record[name], f"validation claim {name}")
    cohort = preflight["cohort_identity"]
    if (
        record["schema_id"] != GRASP_LIFT_VALIDATION_CLAIM_SCHEMA_ID
        or record["purpose"] != GRASP_LIFT_VALIDATION_PURPOSE
        or record["locked_test"] != GRASP_LIFT_LOCKED_TEST_ABSENT
        or record["claim_id"] != grasp_lift_validation_claim_id(cohort)
        or record["cohort_identity_sha256"] != cohort["self_sha256"]
        or record["preflight_sha256"] != preflight["self_sha256"]
        or record["candidate_inventory_sha256"]
        != preflight["candidate_inventory_sha256"]
    ):
        raise ValueError("validation claim binding changed")


def _strict_validation_shard_path(
    root: Path,
    relative_value: Any,
    *,
    trajectory_id: str,
    evidence: bool,
) -> Path:
    suffix = ".grasp-lift-evidence" if evidence else ".stage0"
    expected = Path("shards") / f"{trajectory_id}{suffix}.json"
    if not isinstance(relative_value, str) or Path(relative_value) != expected:
        raise ValueError("formal validation shard path changed")
    shards = root / "shards"
    descriptor = root / expected
    payload = descriptor.with_suffix(".pt")
    if shards.is_symlink() or not shards.is_dir():
        raise ValueError("pilot shards directory is missing or a symlink")
    for path in (descriptor, payload):
        if path.is_symlink() or not path.is_file():
            raise ValueError("formal validation shard is missing or a symlink")
    return descriptor


@dataclass(frozen=True)
class GraspLiftValidationMember:
    """One claimed raw member and its deterministic fixed model view."""

    commitment: GraspLiftRoleCommitment
    trajectory: Stage0Trajectory
    evidence: GraspLiftEvidence
    model_trajectory: Stage0Trajectory

    def __post_init__(self) -> None:
        if not isinstance(self.commitment, GraspLiftRoleCommitment) or (
            self.commitment.role != GRASP_LIFT_VALIDATION_ROLE
        ):
            raise ValueError("validation member commitment changed")
        if not isinstance(self.trajectory, Stage0Trajectory) or not isinstance(
            self.evidence,
            GraspLiftEvidence,
        ):
            raise TypeError("validation member tensors use unsupported types")
        if (
            self.trajectory.provenance.trajectory_id != self.commitment.trajectory_id
            or self.trajectory.provenance.reset_seed != self.commitment.reset_seed
            or trajectory_sha256(self.trajectory)
            != self.commitment.raw_trajectory_sha256
        ):
            raise ValueError("validation trajectory differs from its commitment")
        if not self.trajectory.is_success:
            raise ValueError(
                "formal validation member is not successful authoring data"
            )
        validate_grasp_lift_evidence_trajectory(self.evidence, self.trajectory)
        if grasp_lift_evidence_sha256(self.evidence) != self.commitment.evidence_sha256:
            raise ValueError("validation evidence differs from its commitment")
        expected_model = materialize_grasp_lift_model_trajectory(self.trajectory)
        if trajectory_sha256(self.model_trajectory) != trajectory_sha256(
            expected_model
        ):
            raise ValueError("validation fixed model view changed")


def _tensor_record(tensor: torch.Tensor) -> dict[str, Any]:
    if not isinstance(tensor, torch.Tensor) or tensor.device.type != "cpu":
        raise ValueError("validation content hashing requires CPU tensors")
    value = tensor.detach().contiguous()
    return {
        "dtype": str(value.dtype).removeprefix("torch."),
        "payload_sha256": hashlib.sha256(value.numpy().tobytes(order="C")).hexdigest(),
        "shape": list(value.shape),
    }


def _window_content_sha256(windows: tuple[Stage0SuccessWindow, ...]) -> str:
    records = []
    for window in windows:
        records.append(
            {
                "corpus_sha256": window.corpus_sha256,
                "reset_group_id": window.reset_group_id,
                "start": window.start,
                "tensors": {
                    name: _tensor_record(getattr(window, name))
                    for name in (
                        "action_chunk",
                        "action_mask",
                        "current_state",
                        "future_actions",
                        "future_mask",
                        "future_states",
                    )
                },
                "trajectory_id": window.trajectory_id,
                "trajectory_sha256": window.trajectory_sha256,
                "window_policy_sha256": window.window_policy_sha256,
            }
        )
    return canonical_fingerprint(
        {
            "schema_id": "rexpolicy/stage0-grasp-lift-validation-windows/v1",
            "windows": records,
        }
    )


def _batch_content_sha256(batch: Stage0WindowBatch) -> str:
    return canonical_fingerprint(
        {
            "corpus_sha256": batch.corpus_sha256,
            "reset_group_ids": list(batch.reset_group_ids),
            "schema_id": "rexpolicy/stage0-grasp-lift-validation-batch/v1",
            "starts": list(batch.starts),
            "tensors": {
                name: _tensor_record(getattr(batch, name))
                for name in (
                    "action_chunks",
                    "action_mask",
                    "current_states",
                    "future_actions",
                    "future_mask",
                    "future_states",
                )
            },
            "trajectory_ids": list(batch.trajectory_ids),
            "window_policy_sha256": batch.window_policy_sha256,
        }
    )


@dataclass(frozen=True)
class GraspLiftClaimedValidationData:
    """Exactly six claimed members plus raw and train-normalized model windows."""

    members: tuple[GraspLiftValidationMember, ...]
    windows: tuple[Stage0SuccessWindow, ...]
    normalized_batch: Stage0WindowBatch
    normalization: Stage0Normalization
    record: Mapping[str, Any]

    def __post_init__(self) -> None:
        if any(
            not isinstance(item, GraspLiftValidationMember) for item in self.members
        ):
            raise TypeError("claimed validation members changed type")
        if tuple(item.commitment.reset_seed for item in self.members) != (
            GRASP_LIFT_FORMAL_VALIDATION_SEEDS
        ):
            raise ValueError("claimed validation seed order changed")
        if (
            not self.windows
            or any(not isinstance(item, Stage0SuccessWindow) for item in self.windows)
            or not isinstance(self.normalized_batch, Stage0WindowBatch)
            or not isinstance(self.normalization, Stage0Normalization)
        ):
            raise ValueError("claimed validation windows are empty or invalid")
        record = _exact_mapping(
            self.record,
            _VALIDATION_DATA_KEYS,
            "claimed validation data",
        )
        verify_grasp_lift_pilot_record(record, "claimed validation data")
        model_trajectories = tuple(item.model_trajectory for item in self.members)
        viewed_corpus_sha256 = trajectory_corpus_sha256(model_trajectories)
        expected_windows = tuple(
            window
            for trajectory in model_trajectories
            for window in extract_success_windows(
                trajectory,
                GRASP_LIFT_TRAINING_WINDOW_POLICY,
                corpus_sha256=viewed_corpus_sha256,
            )
        )
        window_content_sha256 = _window_content_sha256(self.windows)
        if len(self.windows) != len(
            expected_windows
        ) or window_content_sha256 != _window_content_sha256(expected_windows):
            raise ValueError("claimed validation window content changed")
        expected_normalized = normalize_grasp_lift_no_z_batch_once(
            collate_success_windows(self.windows),
            self.normalization,
            source_corpus_sha256=viewed_corpus_sha256,
        )
        normalized_batch_sha256 = _batch_content_sha256(self.normalized_batch)
        if normalized_batch_sha256 != _batch_content_sha256(expected_normalized):
            raise ValueError("claimed validation normalized batch changed")
        member_records = [
            {
                **member.commitment.to_record(),
                "model_trajectory_sha256": trajectory_sha256(member.model_trajectory),
            }
            for member in self.members
        ]
        data_sha256 = canonical_fingerprint(
            {
                "members": member_records,
                "normalization_sha256": self.normalization.sha256,
                "normalized_batch_sha256": normalized_batch_sha256,
                "schema_id": "rexpolicy/stage0-grasp-lift-validation-content/v1",
                "viewed_corpus_sha256": viewed_corpus_sha256,
                "window_content_sha256": window_content_sha256,
                "window_policy_sha256": GRASP_LIFT_TRAINING_WINDOW_POLICY.sha256,
            }
        )
        for name in ("claim_sha256", "cohort_identity_sha256"):
            _require_sha256(record[name], f"claimed validation {name}")
        if (
            record["schema_id"] != GRASP_LIFT_VALIDATION_DATA_SCHEMA_ID
            or record["locked_test"] != GRASP_LIFT_LOCKED_TEST_ABSENT
            or record["counts"]
            != {
                "formal_validation_trajectories": len(self.members),
                "formal_validation_windows": len(self.windows),
            }
            or record["members"] != member_records
            or record["normalization_policy"]
            != "reuse_exact_train_fit; never fit validation"
            or record["normalization_sha256"] != self.normalization.sha256
            or record["viewed_corpus_sha256"] != viewed_corpus_sha256
            or record["window_policy_sha256"]
            != GRASP_LIFT_TRAINING_WINDOW_POLICY.sha256
            or record["window_content_sha256"] != window_content_sha256
            or record["normalized_batch_sha256"] != normalized_batch_sha256
            or record["data_sha256"] != data_sha256
        ):
            raise ValueError("claimed validation data binding changed")
        object.__setattr__(self, "record", MappingProxyType(record))

    @property
    def self_sha256(self) -> str:
        return str(self.record["self_sha256"])

    @property
    def data_sha256(self) -> str:
        return str(self.record["data_sha256"])


def load_claimed_grasp_lift_validation(
    pilot_directory: str | Path,
    *,
    repository_root: str | Path,
    training_artifact: GraspLiftTrainingArtifact,
    preflight: Mapping[str, Any],
    claim: Mapping[str, Any],
    claim_receipt: GraspLiftValidationClaimReceipt,
) -> GraspLiftClaimedValidationData:
    """Open exactly six validation shards after a verified durable claim."""
    artifact = _require_training_artifact(training_artifact)
    verify_grasp_lift_validation_claim_record(
        claim,
        preflight=preflight,
        training_artifact=artifact,
    )
    from rexpolicy.stage0.data.grasp_lift_validation_claim import (
        GraspLiftValidationClaimReceipt,
        verify_persisted_grasp_lift_validation_claim,
    )

    if not isinstance(claim_receipt, GraspLiftValidationClaimReceipt):
        raise TypeError(
            "claim_receipt must prove one durable Git-common validation claim"
        )
    verify_persisted_grasp_lift_validation_claim(
        repository_root,
        claim,
        verifier=lambda value: verify_grasp_lift_validation_claim_record(
            value,
            preflight=preflight,
            training_artifact=artifact,
        ),
        receipt=claim_receipt,
    )
    root, manifest = _load_accepted_pilot_registry(pilot_directory)
    if manifest.get("acceptance", {}).get("passed") is not True:
        raise ValueError("Grasp-Lift pilot did not pass acceptance")
    if manifest.get("locked_test") != GRASP_LIFT_LOCKED_TEST_ABSENT:
        raise ValueError("accepted pilot touched locked test")
    if (
        manifest.get("self_sha256") != artifact.data.corpus.source_manifest_sha256
        or manifest.get("composite_corpus_sha256")
        != artifact.data.corpus.source_composite_corpus_sha256
    ):
        raise ValueError("claimed pilot source changed")

    members = manifest.get("members")
    if not isinstance(members, list) or len(members) != 32:
        raise ValueError("accepted pilot member registry changed")
    by_role: dict[str, list[tuple[GraspLiftRoleCommitment, dict[str, Any]]]] = {
        GRASP_LIFT_VALIDATION_ROLE: [],
        GRASP_LIFT_EXCLUDED_ROLE: [],
    }
    all_commitments: list[GraspLiftRoleCommitment] = []
    for raw_member in members:
        member = _exact_mapping(raw_member, _PILOT_MEMBER_KEYS, "pilot member")
        role = grasp_lift_training_role(member)
        commitment = GraspLiftRoleCommitment(
            trajectory_id=member["trajectory_id"],
            reset_seed=member["reset_seed"],
            role=role,
            raw_trajectory_sha256=member["trajectory_sha256"],
            evidence_sha256=member["evidence_sha256"],
        )
        all_commitments.append(commitment)
        if role in by_role:
            by_role[role].append((commitment, member))
    for values in by_role.values():
        values.sort(key=lambda item: item[0].reset_seed)
    corpus = artifact.data.corpus
    expected_commitments = (
        tuple(item.commitment for item in corpus.train)
        + corpus.validation_commitments
        + corpus.excluded_commitments
    )
    observed_by_id = {item.trajectory_id: item.to_record() for item in all_commitments}
    expected_by_id = {
        item.trajectory_id: item.to_record() for item in expected_commitments
    }
    if len(observed_by_id) != 32 or observed_by_id != expected_by_id:
        raise ValueError("pilot role commitments differ from training artifact")
    if tuple(item[0] for item in by_role[GRASP_LIFT_VALIDATION_ROLE]) != (
        corpus.validation_commitments
    ) or tuple(item[0] for item in by_role[GRASP_LIFT_EXCLUDED_ROLE]) != (
        corpus.excluded_commitments
    ):
        raise ValueError("pilot held-out commitments differ from training artifact")
    if preflight["cohort_identity"] != grasp_lift_validation_cohort_identity(artifact):
        raise ValueError("claimed validation cohort changed before shard open")

    loaded: list[GraspLiftValidationMember] = []
    for commitment, member in by_role[GRASP_LIFT_VALIDATION_ROLE]:
        if commitment.reset_seed in GRASP_LIFT_ENGINEERING_EXCLUDED_SEEDS:
            raise RuntimeError(
                "engineering smoke seed reached formal validation loader"
            )
        if (
            member["outcome"] != "success"
            or member["formal_gate_eligible"] is not True
            or member["safety_violation"] is not False
        ):
            raise ValueError("formal validation member acceptance changed")
        trajectory_path = _strict_validation_shard_path(
            root,
            member["trajectory_descriptor"],
            trajectory_id=commitment.trajectory_id,
            evidence=False,
        )
        evidence_path = _strict_validation_shard_path(
            root,
            member["evidence_descriptor"],
            trajectory_id=commitment.trajectory_id,
            evidence=True,
        )
        trajectory = load_trajectory_shard(trajectory_path)
        evidence = load_grasp_lift_evidence_sidecar(
            evidence_path,
            trajectory=trajectory,
        )
        loaded.append(
            GraspLiftValidationMember(
                commitment=commitment,
                trajectory=trajectory,
                evidence=evidence,
                model_trajectory=materialize_grasp_lift_model_trajectory(trajectory),
            )
        )

    validation_members = tuple(loaded)
    model_trajectories = tuple(item.model_trajectory for item in validation_members)
    viewed_corpus_sha256 = trajectory_corpus_sha256(model_trajectories)
    windows = tuple(
        window
        for trajectory in model_trajectories
        for window in extract_success_windows(
            trajectory,
            GRASP_LIFT_TRAINING_WINDOW_POLICY,
            corpus_sha256=viewed_corpus_sha256,
        )
    )
    if not windows:
        raise ValueError("formal validation produced no success windows")
    normalized_batch = normalize_grasp_lift_no_z_batch_once(
        collate_success_windows(windows),
        artifact.normalization,
        source_corpus_sha256=viewed_corpus_sha256,
    )
    window_content_sha256 = _window_content_sha256(windows)
    normalized_batch_sha256 = _batch_content_sha256(normalized_batch)
    member_records = [
        {
            **member.commitment.to_record(),
            "model_trajectory_sha256": trajectory_sha256(member.model_trajectory),
        }
        for member in validation_members
    ]
    data_sha256 = canonical_fingerprint(
        {
            "members": member_records,
            "normalization_sha256": artifact.normalization.sha256,
            "normalized_batch_sha256": normalized_batch_sha256,
            "schema_id": "rexpolicy/stage0-grasp-lift-validation-content/v1",
            "viewed_corpus_sha256": viewed_corpus_sha256,
            "window_content_sha256": window_content_sha256,
            "window_policy_sha256": GRASP_LIFT_TRAINING_WINDOW_POLICY.sha256,
        }
    )
    record = seal_grasp_lift_pilot_record(
        {
            "claim_sha256": claim["self_sha256"],
            "cohort_identity_sha256": preflight["cohort_identity_sha256"],
            "counts": {
                "formal_validation_trajectories": len(validation_members),
                "formal_validation_windows": len(windows),
            },
            "data_sha256": data_sha256,
            "locked_test": _locked_test_absent_record(),
            "members": member_records,
            "normalization_policy": "reuse_exact_train_fit; never fit validation",
            "normalization_sha256": artifact.normalization.sha256,
            "normalized_batch_sha256": normalized_batch_sha256,
            "schema_id": GRASP_LIFT_VALIDATION_DATA_SCHEMA_ID,
            "viewed_corpus_sha256": viewed_corpus_sha256,
            "window_content_sha256": window_content_sha256,
            "window_policy_sha256": GRASP_LIFT_TRAINING_WINDOW_POLICY.sha256,
        }
    )
    return GraspLiftClaimedValidationData(
        members=validation_members,
        windows=windows,
        normalized_batch=normalized_batch,
        normalization=artifact.normalization,
        record=record,
    )


__all__ = [
    "GRASP_LIFT_LOCKED_TEST_ABSENT",
    "GRASP_LIFT_VALIDATION_CANDIDATE_STEPS",
    "GRASP_LIFT_VALIDATION_CHECKPOINT_STEPS",
    "GRASP_LIFT_VALIDATION_CLAIM_ID_SCHEMA_ID",
    "GRASP_LIFT_VALIDATION_CLAIM_SCHEMA_ID",
    "GRASP_LIFT_VALIDATION_COHORT_SCHEMA_ID",
    "GRASP_LIFT_VALIDATION_DATA_SCHEMA_ID",
    "GRASP_LIFT_VALIDATION_PREFLIGHT_SCHEMA_ID",
    "GRASP_LIFT_VALIDATION_PURPOSE",
    "GraspLiftClaimedValidationData",
    "GraspLiftValidationMember",
    "build_grasp_lift_validation_claim_record",
    "grasp_lift_validation_claim_id",
    "grasp_lift_validation_cohort_identity",
    "load_claimed_grasp_lift_validation",
    "preflight_grasp_lift_validation_selection",
    "verify_grasp_lift_validation_claim_record",
    "verify_grasp_lift_validation_preflight_record",
]
