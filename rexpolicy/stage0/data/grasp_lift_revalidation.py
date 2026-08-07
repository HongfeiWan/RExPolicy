"""Claim-bound loader for an independently authored Grasp-Lift cohort.

Authoring schemas and metadata verification live only in
``grasp_lift_validation_cohort``.  This module adds the model-independent
cohort identity, the existing durable claim envelope, and the sole path that
may open the six new tensor shards after that claim is durable.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

from rexpolicy.stage0.data.grasp_lift_evidence import (
    grasp_lift_composite_corpus_sha256,
    load_grasp_lift_evidence_sidecar,
)
from rexpolicy.stage0.data.grasp_lift_training import (
    GRASP_LIFT_TRAINING_WINDOW_POLICY,
    GRASP_LIFT_VALIDATION_ROLE,
    GraspLiftRoleCommitment,
    materialize_grasp_lift_model_trajectory,
)
from rexpolicy.stage0.data.grasp_lift_training_artifact import (
    GraspLiftTrainingArtifact,
)
from rexpolicy.stage0.data.grasp_lift_validation import (
    GRASP_LIFT_VALIDATION_CLAIM_SCHEMA_ID,
    GRASP_LIFT_VALIDATION_COHORT_SCHEMA_ID,
    GRASP_LIFT_VALIDATION_PURPOSE,
    GraspLiftValidationMember,
    _batch_content_sha256,
    _require_training_artifact,
    _strict_validation_shard_path,
    _window_content_sha256,
    grasp_lift_validation_claim_id,
)
from rexpolicy.stage0.data.grasp_lift_validation_cohort import (
    GRASP_LIFT_VALIDATION_LOCKED_TEST,
    GRASP_LIFT_VALIDATION_PURPOSE as GRASP_LIFT_REVALIDATION_PURPOSE,
    GRASP_LIFT_VALIDATION_RESET_SEEDS as GRASP_LIFT_REVALIDATION_RESET_SEEDS,
    GRASP_LIFT_VALIDATION_SEED_NAMESPACE as GRASP_LIFT_REVALIDATION_SEED_NAMESPACE,
    grasp_lift_validation_cohort_slot_id,
    load_grasp_lift_validation_authoring_config,
    load_grasp_lift_validation_authoring_metadata,
)
from rexpolicy.stage0.data.store import load_trajectory_shard
from rexpolicy.stage0.data.trajectory import (
    trajectory_corpus_sha256,
    trajectory_sha256,
)
from rexpolicy.stage0.data.windows import Stage0SuccessWindow, extract_success_windows
from rexpolicy.stage0.grasp_lift_no_z import normalize_grasp_lift_no_z_batch_once
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


GRASP_LIFT_REVALIDATION_COHORT_IDENTITY_SCHEMA_ID = (
    GRASP_LIFT_VALIDATION_COHORT_SCHEMA_ID
)
GRASP_LIFT_REVALIDATION_DATA_SCHEMA_ID = (
    "rexpolicy/stage0-grasp-lift-claimed-revalidation-data/v1"
)
_LOCKED_TEST_ABSENT = dict(GRASP_LIFT_VALIDATION_LOCKED_TEST)
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


def _exact_mapping(value: Any, keys: set[str], context: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != keys:
        raise ValueError(f"{context} fields changed")
    return dict(value)


def _require_sha256(value: Any, context: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{context} must be a lowercase SHA-256 digest")
    return value


def _commitments(manifest: Mapping[str, Any]) -> tuple[GraspLiftRoleCommitment, ...]:
    members = manifest["members"]
    if not isinstance(members, list) or len(members) != 6:
        raise ValueError("revalidation manifest must contain exactly six members")
    result = tuple(
        GraspLiftRoleCommitment(
            trajectory_id=member["trajectory_id"],
            reset_seed=member["reset_seed"],
            role=GRASP_LIFT_VALIDATION_ROLE,
            raw_trajectory_sha256=member["trajectory_sha256"],
            evidence_sha256=member["evidence_sha256"],
        )
        for member in members
    )
    if tuple(item.reset_seed for item in result) != GRASP_LIFT_REVALIDATION_RESET_SEEDS:
        raise ValueError("revalidation commitment seed order changed")
    if len({item.trajectory_id for item in result}) != 6:
        raise ValueError("revalidation trajectory identity is duplicated")
    return result


def _cohort_identity(
    manifest: Mapping[str, Any],
    commitments: tuple[GraspLiftRoleCommitment, ...],
) -> dict[str, Any]:
    role_registry_sha256 = canonical_fingerprint(
        {
            "roles": [item.to_record() for item in commitments],
            "schema_id": "rexpolicy/stage0-grasp-lift-frozen-roles/v1",
        }
    )
    return seal_grasp_lift_pilot_record(
        {
            "excluded_commitments": [],
            "formal_validation_commitments": [item.to_record() for item in commitments],
            "locked_test": dict(_LOCKED_TEST_ABSENT),
            "purpose": GRASP_LIFT_VALIDATION_PURPOSE,
            "role_registry_sha256": role_registry_sha256,
            "schema_id": GRASP_LIFT_VALIDATION_COHORT_SCHEMA_ID,
            "source_composite_corpus_sha256": manifest["composite_corpus_sha256"],
            "source_manifest_sha256": manifest["self_sha256"],
        }
    )


@dataclass(frozen=True)
class GraspLiftRevalidationMetadata:
    """Strict authoring metadata and a v1-compatible cohort identity."""

    root: Path
    config: Mapping[str, Any]
    manifest: Mapping[str, Any]
    status: Mapping[str, Any]
    commit: Mapping[str, Any]
    commitments: tuple[GraspLiftRoleCommitment, ...]
    cohort_identity: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not isinstance(self.root, Path) or not self.root.is_absolute():
            raise ValueError("revalidation root must be absolute")
        if tuple(item.reset_seed for item in self.commitments) != (
            GRASP_LIFT_REVALIDATION_RESET_SEEDS
        ):
            raise ValueError("revalidation metadata seed order changed")
        identity = _exact_mapping(
            self.cohort_identity, _COHORT_KEYS, "revalidation cohort identity"
        )
        verify_grasp_lift_pilot_record(identity, "revalidation cohort identity")
        if (
            identity["schema_id"] != GRASP_LIFT_VALIDATION_COHORT_SCHEMA_ID
            or identity["purpose"] != GRASP_LIFT_VALIDATION_PURPOSE
            or identity["locked_test"] != _LOCKED_TEST_ABSENT
            or identity["source_manifest_sha256"] != self.manifest["self_sha256"]
            or identity["source_composite_corpus_sha256"]
            != self.manifest["composite_corpus_sha256"]
        ):
            raise ValueError("revalidation cohort identity changed")
        for name in ("config", "manifest", "status", "commit", "cohort_identity"):
            object.__setattr__(self, name, MappingProxyType(dict(getattr(self, name))))

    @property
    def cohort_identity_sha256(self) -> str:
        return str(self.cohort_identity["self_sha256"])


def load_grasp_lift_revalidation_metadata(
    cohort_directory: str | Path,
) -> GraspLiftRevalidationMetadata:
    """Verify four metadata JSON files without listing or opening ``shards``."""

    root = Path(cohort_directory).expanduser()
    if root.is_symlink() or not root.is_dir():
        raise ValueError("revalidation cohort must be a real directory")
    root = root.resolve()
    metadata = load_grasp_lift_validation_authoring_metadata(root)
    if set(metadata) != {"commit", "config", "config_sha256", "manifest", "status"}:
        raise ValueError("validation authoring metadata fields changed")
    config = load_grasp_lift_validation_authoring_config(root / "config.json")
    if config.to_record() != metadata["config"]:
        raise RuntimeError("validation authoring config metadata changed")
    if (
        grasp_lift_validation_cohort_slot_id(config)
        != metadata["manifest"]["cohort_slot_id"]
    ):
        raise ValueError("revalidation cohort slot changed")
    commitments = _commitments(metadata["manifest"])
    identity = _cohort_identity(metadata["manifest"], commitments)
    return GraspLiftRevalidationMetadata(
        root=root,
        config=metadata["config"],
        manifest=metadata["manifest"],
        status=metadata["status"],
        commit=metadata["commit"],
        commitments=commitments,
        cohort_identity=identity,
    )


def grasp_lift_revalidation_claim_id(
    metadata: GraspLiftRevalidationMetadata,
) -> str:
    if not isinstance(metadata, GraspLiftRevalidationMetadata):
        raise TypeError("metadata must be GraspLiftRevalidationMetadata")
    return grasp_lift_validation_claim_id(metadata.cohort_identity)


def build_grasp_lift_revalidation_claim_record(
    metadata: GraspLiftRevalidationMetadata,
    *,
    candidate_inventory_sha256: str,
    preflight_sha256: str,
    selection_protocol_sha256: str,
    evaluator_implementation_sha256: str,
) -> dict[str, Any]:
    if not isinstance(metadata, GraspLiftRevalidationMetadata):
        raise TypeError("metadata must be GraspLiftRevalidationMetadata")
    for value, name in (
        (candidate_inventory_sha256, "candidate inventory"),
        (preflight_sha256, "preflight"),
        (selection_protocol_sha256, "selection protocol"),
        (evaluator_implementation_sha256, "evaluator implementation"),
    ):
        _require_sha256(value, f"revalidation {name} SHA-256")
    return seal_grasp_lift_pilot_record(
        {
            "candidate_inventory_sha256": candidate_inventory_sha256,
            "claim_id": grasp_lift_revalidation_claim_id(metadata),
            "cohort_identity_sha256": metadata.cohort_identity_sha256,
            "evaluator_implementation_sha256": evaluator_implementation_sha256,
            "locked_test": dict(_LOCKED_TEST_ABSENT),
            "preflight_sha256": preflight_sha256,
            "purpose": GRASP_LIFT_VALIDATION_PURPOSE,
            "schema_id": GRASP_LIFT_VALIDATION_CLAIM_SCHEMA_ID,
            "selection_protocol_sha256": selection_protocol_sha256,
        }
    )


def verify_grasp_lift_revalidation_claim_record(
    claim: Mapping[str, Any],
    *,
    metadata: GraspLiftRevalidationMetadata,
    candidate_inventory_sha256: str,
    preflight_sha256: str,
    selection_protocol_sha256: str,
    evaluator_implementation_sha256: str,
) -> None:
    expected = build_grasp_lift_revalidation_claim_record(
        metadata,
        candidate_inventory_sha256=candidate_inventory_sha256,
        preflight_sha256=preflight_sha256,
        selection_protocol_sha256=selection_protocol_sha256,
        evaluator_implementation_sha256=evaluator_implementation_sha256,
    )
    record = _exact_mapping(claim, _CLAIM_KEYS, "revalidation claim")
    verify_grasp_lift_pilot_record(record, "revalidation claim")
    if record != expected:
        raise ValueError("revalidation claim binding changed")


@dataclass(frozen=True)
class GraspLiftClaimedRevalidationData:
    members: tuple[GraspLiftValidationMember, ...]
    windows: tuple[Stage0SuccessWindow, ...]
    normalized_batch: Stage0WindowBatch
    normalization: Stage0Normalization
    record: Mapping[str, Any]

    def __post_init__(self) -> None:
        if tuple(item.commitment.reset_seed for item in self.members) != (
            GRASP_LIFT_REVALIDATION_RESET_SEEDS
        ):
            raise ValueError("claimed revalidation member order changed")
        if not self.windows or not isinstance(self.normalized_batch, Stage0WindowBatch):
            raise ValueError("claimed revalidation windows are empty or invalid")
        if not isinstance(self.normalization, Stage0Normalization):
            raise TypeError("claimed revalidation normalization changed type")
        record = dict(self.record)
        verify_grasp_lift_pilot_record(record, "claimed revalidation data")
        if (
            record.get("schema_id") != GRASP_LIFT_REVALIDATION_DATA_SCHEMA_ID
            or record.get("locked_test") != _LOCKED_TEST_ABSENT
        ):
            raise ValueError("claimed revalidation data binding changed")
        object.__setattr__(self, "record", MappingProxyType(record))

    @property
    def data_sha256(self) -> str:
        return str(self.record["data_sha256"])


def load_claimed_grasp_lift_revalidation(
    cohort_directory: str | Path,
    *,
    repository_root: str | Path,
    training_artifact: GraspLiftTrainingArtifact,
    claim: Mapping[str, Any],
    claim_receipt: GraspLiftValidationClaimReceipt,
    candidate_inventory_sha256: str,
    preflight_sha256: str,
    selection_protocol_sha256: str,
    evaluator_implementation_sha256: str,
) -> GraspLiftClaimedRevalidationData:
    """Open exactly six sidecar members after the durable claim is verified."""

    artifact = _require_training_artifact(training_artifact)
    metadata = load_grasp_lift_revalidation_metadata(cohort_directory)
    verify_grasp_lift_revalidation_claim_record(
        claim,
        metadata=metadata,
        candidate_inventory_sha256=candidate_inventory_sha256,
        preflight_sha256=preflight_sha256,
        selection_protocol_sha256=selection_protocol_sha256,
        evaluator_implementation_sha256=evaluator_implementation_sha256,
    )
    from rexpolicy.stage0.data.grasp_lift_validation_claim import (
        GraspLiftValidationClaimReceipt,
        verify_persisted_grasp_lift_validation_claim,
    )

    if not isinstance(claim_receipt, GraspLiftValidationClaimReceipt):
        raise TypeError("claim_receipt must prove one durable Git-common claim")

    def verifier(value: Mapping[str, Any]) -> None:
        verify_grasp_lift_revalidation_claim_record(
            value,
            metadata=metadata,
            candidate_inventory_sha256=candidate_inventory_sha256,
            preflight_sha256=preflight_sha256,
            selection_protocol_sha256=selection_protocol_sha256,
            evaluator_implementation_sha256=evaluator_implementation_sha256,
        )

    verify_persisted_grasp_lift_validation_claim(
        repository_root,
        claim,
        verifier=verifier,
        receipt=claim_receipt,
    )

    loaded: list[GraspLiftValidationMember] = []
    authored_corpus: list[tuple[Any, Any]] = []
    for commitment, raw_member in zip(
        metadata.commitments, metadata.manifest["members"], strict=True
    ):
        member = dict(raw_member)
        trajectory_path = _strict_validation_shard_path(
            metadata.root,
            member["trajectory_descriptor"],
            trajectory_id=commitment.trajectory_id,
            evidence=False,
        )
        evidence_path = _strict_validation_shard_path(
            metadata.root,
            member["evidence_descriptor"],
            trajectory_id=commitment.trajectory_id,
            evidence=True,
        )
        trajectory = load_trajectory_shard(trajectory_path)
        evidence = load_grasp_lift_evidence_sidecar(
            evidence_path, trajectory=trajectory
        )
        authored_corpus.append((trajectory, evidence))
        if trajectory.provenance.reset_group_id != member["reset_group_id"]:
            raise ValueError("revalidation reset group changed after claim")
        loaded.append(
            GraspLiftValidationMember(
                commitment=commitment,
                trajectory=trajectory,
                evidence=evidence,
                model_trajectory=materialize_grasp_lift_model_trajectory(trajectory),
            )
        )

    if (
        grasp_lift_composite_corpus_sha256(authored_corpus)
        != metadata.manifest["composite_corpus_sha256"]
    ):
        raise ValueError("revalidation composite corpus changed after claim")

    members = tuple(loaded)
    model_trajectories = tuple(item.model_trajectory for item in members)
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
        raise ValueError("revalidation produced no success windows")
    normalized_batch = normalize_grasp_lift_no_z_batch_once(
        collate_success_windows(windows),
        artifact.normalization,
        source_corpus_sha256=viewed_corpus_sha256,
    )
    window_content_sha256 = _window_content_sha256(windows)
    normalized_batch_sha256 = _batch_content_sha256(normalized_batch)
    member_records = [
        {
            **item.commitment.to_record(),
            "model_trajectory_sha256": trajectory_sha256(item.model_trajectory),
            "reset_group_id": item.model_trajectory.provenance.reset_group_id,
        }
        for item in members
    ]
    data_sha256 = canonical_fingerprint(
        {
            "cohort_identity_sha256": metadata.cohort_identity_sha256,
            "members": member_records,
            "normalization_sha256": artifact.normalization.sha256,
            "normalized_batch_sha256": normalized_batch_sha256,
            "schema_id": "rexpolicy/stage0-grasp-lift-revalidation-content/v1",
            "training_artifact_sha256": artifact.artifact_sha256,
            "viewed_corpus_sha256": viewed_corpus_sha256,
            "window_content_sha256": window_content_sha256,
            "window_policy_sha256": GRASP_LIFT_TRAINING_WINDOW_POLICY.sha256,
        }
    )
    record = seal_grasp_lift_pilot_record(
        {
            "claim_sha256": claim["self_sha256"],
            "cohort_identity_sha256": metadata.cohort_identity_sha256,
            "counts": {
                "formal_validation_trajectories": 6,
                "formal_validation_windows": len(windows),
            },
            "data_sha256": data_sha256,
            "locked_test": dict(_LOCKED_TEST_ABSENT),
            "members": member_records,
            "normalization_policy": "reuse_exact_train_fit; never fit validation",
            "normalization_sha256": artifact.normalization.sha256,
            "normalized_batch_sha256": normalized_batch_sha256,
            "schema_id": GRASP_LIFT_REVALIDATION_DATA_SCHEMA_ID,
            "train_content_sha256": artifact.train_content_sha256,
            "training_artifact_sha256": artifact.artifact_sha256,
            "viewed_corpus_sha256": viewed_corpus_sha256,
            "window_content_sha256": window_content_sha256,
            "window_policy_sha256": GRASP_LIFT_TRAINING_WINDOW_POLICY.sha256,
        }
    )
    return GraspLiftClaimedRevalidationData(
        members=members,
        windows=windows,
        normalized_batch=normalized_batch,
        normalization=artifact.normalization,
        record=record,
    )


__all__ = [
    "GRASP_LIFT_REVALIDATION_COHORT_IDENTITY_SCHEMA_ID",
    "GRASP_LIFT_REVALIDATION_DATA_SCHEMA_ID",
    "GRASP_LIFT_REVALIDATION_PURPOSE",
    "GRASP_LIFT_REVALIDATION_RESET_SEEDS",
    "GRASP_LIFT_REVALIDATION_SEED_NAMESPACE",
    "GraspLiftClaimedRevalidationData",
    "GraspLiftRevalidationMetadata",
    "build_grasp_lift_revalidation_claim_record",
    "grasp_lift_revalidation_claim_id",
    "load_claimed_grasp_lift_revalidation",
    "load_grasp_lift_revalidation_metadata",
    "verify_grasp_lift_revalidation_claim_record",
]
