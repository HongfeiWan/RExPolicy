"""Leakage-safe model views over the accepted Grasp-Lift pilot corpus."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import torch

from rexpolicy.stage0.data.grasp_lift_evidence import (
    GRASP_LIFT_COMPOSITE_CORPUS_SCHEMA_ID,
    GraspLiftEvidence,
    grasp_lift_evidence_sha256,
    load_grasp_lift_evidence_sidecar,
    validate_grasp_lift_evidence_trajectory,
)
from rexpolicy.stage0.data.store import load_trajectory_shard
from rexpolicy.stage0.data.trajectory import (
    Stage0Trajectory,
    trajectory_corpus_sha256,
    trajectory_sha256,
)
from rexpolicy.stage0.data.windows import (
    Stage0WindowPolicy,
    extract_success_windows,
)
from rexpolicy.stage0.data.splits import Stage0WindowSplits
from rexpolicy.stage0.envs.grasp_lift_action import (
    DEFAULT_GRASP_LIFT_ACTION_SCHEMA,
    GRASP_LIFT_EFFECTIVE_ACTION_MASK,
    GRASP_LIFT_MODEL_ACTION_REPRESENTATION_ID,
    grasp_lift_physical_to_model_action,
)
from rexpolicy.stage0.envs.grasp_lift_state_view import (
    DEFAULT_GRASP_LIFT_STATE_VIEW,
    GRASP_LIFT_FIXED_PHASE_ONE_HOT,
    apply_grasp_lift_state_view,
)
from rexpolicy.stage0.envs.state_schema import DEFAULT_STAGE0_STATE_SCHEMA
from rexpolicy.stage0.grasp_lift_pilot import (
    verify_grasp_lift_pilot_record,
)
from rexpolicy.stage0.normalization import (
    Stage0Normalization,
    fit_stage0_normalization,
)
from rexpolicy.stage0.types import canonical_fingerprint


GRASP_LIFT_TRAINING_DATA_SCHEMA_ID = "rexpolicy/stage0-grasp-lift-training-data/v1"
GRASP_LIFT_MODEL_ACTION_VIEW_SCHEMA_ID = (
    "rexpolicy/stage0-grasp-lift-model-action-view/v1"
)
GRASP_LIFT_MODEL_ACTION_VIEW_SHA256 = canonical_fingerprint(
    {
        "model_representation_id": GRASP_LIFT_MODEL_ACTION_REPRESENTATION_ID,
        "operation": "physical executed action to model action exactly once",
        "schema_id": GRASP_LIFT_MODEL_ACTION_VIEW_SCHEMA_ID,
        "source_action_schema_sha256": DEFAULT_GRASP_LIFT_ACTION_SCHEMA.sha256,
    }
)
GRASP_LIFT_ACCEPTED_PILOT_MANIFEST_SHA256 = (
    "7afd7bc74dc223b6101d3add4b4aac1b47959b547b6a983c906cebe09b136cf3"
)
GRASP_LIFT_ACCEPTED_PILOT_CORPUS_SHA256 = (
    "16d438d79e2fb37505e2f2b070842095d025b4ce338c1f472765d10872701620"
)
GRASP_LIFT_TRAINING_WINDOW_POLICY = Stage0WindowPolicy(
    action_horizon=8,
    future_horizon=48,
    stride=1,
)
GRASP_LIFT_ENGINEERING_EXCLUDED_SEEDS = (7001, 7005)
GRASP_LIFT_FORMAL_VALIDATION_SEEDS = (7009, 7014, 7019, 7025, 7026, 7031)
GRASP_LIFT_FORMAL_TRAIN_SEEDS = (
    7000,
    7002,
    7003,
    7004,
    7006,
    7007,
    7008,
    7010,
    7011,
    7012,
    7013,
    7015,
    7016,
    7017,
    7018,
    7020,
    7021,
    7022,
    7023,
    7024,
    7027,
    7028,
    7029,
    7030,
)
GRASP_LIFT_TRAINING_ROLE = "train"
GRASP_LIFT_VALIDATION_ROLE = "formal_validation"
GRASP_LIFT_EXCLUDED_ROLE = "engineering_smoke_excluded"
_TRAINING_ROLES = {
    GRASP_LIFT_TRAINING_ROLE,
    GRASP_LIFT_VALIDATION_ROLE,
    GRASP_LIFT_EXCLUDED_ROLE,
}


def _require_sha256(value: Any, name: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _load_accepted_pilot_registry(
    pilot_directory: str | Path,
) -> tuple[Path, dict[str, Any]]:
    """Read only the pinned manifest; never open held-out tensor shards."""
    requested = Path(pilot_directory).expanduser()
    if requested.is_symlink() or not requested.is_dir():
        raise ValueError("pilot output must be a directory and cannot be a symlink")
    root = requested.resolve()
    manifest_path = root / "manifest.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ValueError("pilot manifest is missing or a symlink")
    try:
        manifest = json.loads(
            manifest_path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=lambda constant: (_ for _ in ()).throw(
                ValueError(f"invalid JSON constant: {constant}")
            ),
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"cannot read pinned Grasp-Lift manifest: {error}") from error
    if not isinstance(manifest, dict):
        raise ValueError("pilot manifest must contain an object")
    verify_grasp_lift_pilot_record(manifest, "pilot manifest")
    if manifest["self_sha256"] != GRASP_LIFT_ACCEPTED_PILOT_MANIFEST_SHA256:
        raise ValueError("pilot is not the accepted Grasp-Lift training source")
    if manifest["composite_corpus_sha256"] != GRASP_LIFT_ACCEPTED_PILOT_CORPUS_SHA256:
        raise ValueError("pilot composite corpus is not accepted for training")
    members = manifest.get("members")
    if not isinstance(members, list) or len(members) != 32:
        raise ValueError("accepted Grasp-Lift member registry changed")
    return root, manifest


def _strict_train_shard_path(
    root: Path,
    relative_value: Any,
    *,
    trajectory_id: str,
    evidence: bool,
) -> Path:
    suffix = ".grasp-lift-evidence" if evidence else ".stage0"
    relative = Path(relative_value) if isinstance(relative_value, str) else None
    expected = Path("shards") / f"{trajectory_id}{suffix}.json"
    if relative != expected:
        raise ValueError("accepted Grasp-Lift train shard path changed")
    shards = root / "shards"
    descriptor = root / expected
    payload = descriptor.with_suffix(".pt")
    if shards.is_symlink() or not shards.is_dir():
        raise ValueError("pilot shards directory is missing or a symlink")
    for path in (descriptor, payload):
        if path.is_symlink() or not path.is_file():
            raise ValueError("pilot train shard is missing or a symlink")
    return descriptor


def _training_window_content_sha256(windows: tuple[Any, ...]) -> str:
    """Bind ordered window metadata plus every tensor's exact CPU bytes."""
    tensor_fields = (
        "current_state",
        "action_chunk",
        "action_mask",
        "future_actions",
        "future_states",
        "future_mask",
    )
    records: list[dict[str, Any]] = []
    for window in windows:
        tensors: dict[str, Any] = {}
        for name in tensor_fields:
            tensor = getattr(window, name)
            payload = tensor.numpy().tobytes(order="C")
            tensors[name] = {
                "dtype": str(tensor.dtype).removeprefix("torch."),
                "payload_sha256": hashlib.sha256(payload).hexdigest(),
                "shape": list(tensor.shape),
            }
        records.append(
            {
                "corpus_sha256": window.corpus_sha256,
                "reset_group_id": window.reset_group_id,
                "start": window.start,
                "tensors": tensors,
                "trajectory_id": window.trajectory_id,
                "trajectory_sha256": window.trajectory_sha256,
                "window_policy_sha256": window.window_policy_sha256,
            }
        )
    return canonical_fingerprint(
        {
            "schema_id": "rexpolicy/stage0-grasp-lift-train-windows/v1",
            "windows": records,
        }
    )


def grasp_lift_training_role(member: Mapping[str, Any]) -> str:
    """Map one frozen pilot member to its immutable downstream role."""
    split = member.get("split")
    seed = member.get("reset_seed")
    eligible = member.get("formal_gate_eligible")
    if split not in {"train", "validation"}:
        raise ValueError("Grasp-Lift pilot member split changed")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("Grasp-Lift pilot member seed changed")
    if not isinstance(eligible, bool):
        raise ValueError("Grasp-Lift pilot member eligibility changed")
    if not eligible:
        if split != "validation" or seed not in GRASP_LIFT_ENGINEERING_EXCLUDED_SEEDS:
            raise ValueError("unexpected ineligible Grasp-Lift pilot member")
        return GRASP_LIFT_EXCLUDED_ROLE
    if split == "train":
        if seed not in GRASP_LIFT_FORMAL_TRAIN_SEEDS:
            raise ValueError("formal Grasp-Lift train cohort changed")
        return GRASP_LIFT_TRAINING_ROLE
    if seed not in GRASP_LIFT_FORMAL_VALIDATION_SEEDS:
        raise ValueError("formal Grasp-Lift validation cohort changed")
    return GRASP_LIFT_VALIDATION_ROLE


@dataclass(frozen=True)
class GraspLiftRoleCommitment:
    """Tensor-free membership commitment retained for a frozen pilot role."""

    trajectory_id: str
    reset_seed: int
    role: str
    raw_trajectory_sha256: str
    evidence_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.trajectory_id, str) or not self.trajectory_id:
            raise ValueError("trajectory_id must be non-empty text")
        if (
            isinstance(self.reset_seed, bool)
            or not isinstance(self.reset_seed, int)
            or self.reset_seed < 0
        ):
            raise ValueError("reset_seed must be a non-negative integer")
        if self.role not in _TRAINING_ROLES:
            raise ValueError("unsupported Grasp-Lift role commitment")
        _require_sha256(self.raw_trajectory_sha256, "raw_trajectory_sha256")
        _require_sha256(self.evidence_sha256, "evidence_sha256")

    def to_record(self) -> dict[str, Any]:
        return {
            "evidence_sha256": self.evidence_sha256,
            "raw_trajectory_sha256": self.raw_trajectory_sha256,
            "reset_seed": self.reset_seed,
            "role": self.role,
            "trajectory_id": self.trajectory_id,
        }


@dataclass(frozen=True)
class GraspLiftTrainingMember:
    """One strictly verified raw trajectory and its non-model sidecar."""

    trajectory: Stage0Trajectory
    evidence: GraspLiftEvidence
    role: str
    raw_trajectory_sha256: str
    evidence_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.trajectory, Stage0Trajectory):
            raise TypeError("trajectory must be Stage0Trajectory")
        if not isinstance(self.evidence, GraspLiftEvidence):
            raise TypeError("evidence must be GraspLiftEvidence")
        if self.role not in _TRAINING_ROLES:
            raise ValueError("unsupported Grasp-Lift training role")
        if trajectory_sha256(self.trajectory) != self.raw_trajectory_sha256:
            raise ValueError("raw Grasp-Lift trajectory hash changed")
        validate_grasp_lift_evidence_trajectory(self.evidence, self.trajectory)
        if grasp_lift_evidence_sha256(self.evidence) != self.evidence_sha256:
            raise ValueError("raw Grasp-Lift evidence hash changed")
        if self.trajectory.outcome.value != "success":
            raise ValueError("accepted Grasp-Lift training members must succeed")

    @property
    def trajectory_id(self) -> str:
        return self.trajectory.provenance.trajectory_id

    @property
    def reset_seed(self) -> int:
        return self.trajectory.provenance.reset_seed

    @property
    def commitment(self) -> GraspLiftRoleCommitment:
        return GraspLiftRoleCommitment(
            trajectory_id=self.trajectory_id,
            reset_seed=self.reset_seed,
            role=self.role,
            raw_trajectory_sha256=self.raw_trajectory_sha256,
            evidence_sha256=self.evidence_sha256,
        )


@dataclass(frozen=True)
class GraspLiftTrainingCorpus:
    """Frozen roles from the pilot; no generic re-splitting is permitted."""

    train: tuple[GraspLiftTrainingMember, ...]
    validation_commitments: tuple[GraspLiftRoleCommitment, ...]
    excluded_commitments: tuple[GraspLiftRoleCommitment, ...]
    source_manifest_sha256: str
    source_composite_corpus_sha256: str

    def __post_init__(self) -> None:
        if self.source_manifest_sha256 != GRASP_LIFT_ACCEPTED_PILOT_MANIFEST_SHA256:
            raise ValueError("Grasp-Lift training source manifest changed")
        if (
            self.source_composite_corpus_sha256
            != GRASP_LIFT_ACCEPTED_PILOT_CORPUS_SHA256
        ):
            raise ValueError("Grasp-Lift training source corpus changed")
        expected = (
            (
                self.train,
                GraspLiftTrainingMember,
                GRASP_LIFT_TRAINING_ROLE,
                24,
            ),
            (
                self.validation_commitments,
                GraspLiftRoleCommitment,
                GRASP_LIFT_VALIDATION_ROLE,
                6,
            ),
            (
                self.excluded_commitments,
                GraspLiftRoleCommitment,
                GRASP_LIFT_EXCLUDED_ROLE,
                2,
            ),
        )
        all_ids: set[str] = set()
        for members, member_type, role, count in expected:
            if not isinstance(members, tuple) or len(members) != count:
                raise ValueError(f"Grasp-Lift {role} member count changed")
            if any(not isinstance(item, member_type) for item in members):
                raise TypeError(f"Grasp-Lift {role} member type changed")
            if tuple(item.trajectory_id for item in members) != tuple(
                sorted(item.trajectory_id for item in members)
            ):
                raise ValueError(f"Grasp-Lift {role} members are not ordered")
            if any(item.role != role for item in members):
                raise ValueError(f"Grasp-Lift {role} role changed")
            ids = {item.trajectory_id for item in members}
            if all_ids & ids:
                raise ValueError("Grasp-Lift training roles overlap")
            all_ids.update(ids)
        if tuple(item.reset_seed for item in self.validation_commitments) != (
            GRASP_LIFT_FORMAL_VALIDATION_SEEDS
        ):
            raise ValueError("Grasp-Lift formal validation seeds changed")
        if tuple(item.reset_seed for item in self.train) != (
            GRASP_LIFT_FORMAL_TRAIN_SEEDS
        ):
            raise ValueError("Grasp-Lift formal train seeds changed")
        if tuple(item.reset_seed for item in self.excluded_commitments) != (
            GRASP_LIFT_ENGINEERING_EXCLUDED_SEEDS
        ):
            raise ValueError("Grasp-Lift excluded seeds changed")
        commitments = (
            tuple(item.commitment for item in self.train)
            + self.validation_commitments
            + self.excluded_commitments
        )
        reconstructed_composite = canonical_fingerprint(
            {
                "members": sorted(
                    (
                        {
                            "evidence_sha256": item.evidence_sha256,
                            "trajectory_id": item.trajectory_id,
                            "trajectory_sha256": item.raw_trajectory_sha256,
                        }
                        for item in commitments
                    ),
                    key=lambda value: value["trajectory_id"],
                ),
                "schema_id": GRASP_LIFT_COMPOSITE_CORPUS_SCHEMA_ID,
            }
        )
        if reconstructed_composite != self.source_composite_corpus_sha256:
            raise ValueError(
                "Grasp-Lift role commitments do not form the source corpus"
            )

    @property
    def training_membership_sha256(self) -> str:
        return canonical_fingerprint(
            {
                "members": [item.commitment.to_record() for item in self.train],
                "schema_id": "rexpolicy/stage0-grasp-lift-train-membership/v1",
            }
        )

    @property
    def role_registry_sha256(self) -> str:
        commitments = (
            tuple(item.commitment for item in self.train)
            + self.validation_commitments
            + self.excluded_commitments
        )
        return canonical_fingerprint(
            {
                "roles": [item.to_record() for item in commitments],
                "schema_id": "rexpolicy/stage0-grasp-lift-frozen-roles/v1",
            }
        )


def load_grasp_lift_training_corpus(
    pilot_directory: str | Path,
) -> GraspLiftTrainingCorpus:
    """Load only the 24 train shards from the pinned tensor-free registry."""
    root, manifest = _load_accepted_pilot_registry(pilot_directory)
    if manifest["acceptance"]["passed"] is not True:
        raise ValueError("Grasp-Lift pilot did not pass authoring acceptance")
    if manifest["locked_test"] != {
        "artifact": None,
        "consumed": False,
        "policy": "not_created",
    }:
        raise ValueError("Grasp-Lift training source touched locked test data")

    train: list[GraspLiftTrainingMember] = []
    held_out: dict[str, list[GraspLiftRoleCommitment]] = {
        GRASP_LIFT_VALIDATION_ROLE: [],
        GRASP_LIFT_EXCLUDED_ROLE: [],
    }
    for record in manifest["members"]:
        role = grasp_lift_training_role(record)
        commitment = GraspLiftRoleCommitment(
            trajectory_id=record["trajectory_id"],
            reset_seed=record["reset_seed"],
            role=role,
            raw_trajectory_sha256=record["trajectory_sha256"],
            evidence_sha256=record["evidence_sha256"],
        )
        if role != GRASP_LIFT_TRAINING_ROLE:
            held_out[role].append(commitment)
            continue
        trajectory_descriptor = _strict_train_shard_path(
            root,
            record["trajectory_descriptor"],
            trajectory_id=record["trajectory_id"],
            evidence=False,
        )
        evidence_descriptor = _strict_train_shard_path(
            root,
            record["evidence_descriptor"],
            trajectory_id=record["trajectory_id"],
            evidence=True,
        )
        trajectory = load_trajectory_shard(trajectory_descriptor)
        evidence = load_grasp_lift_evidence_sidecar(
            evidence_descriptor,
            trajectory=trajectory,
        )
        train.append(
            GraspLiftTrainingMember(
                trajectory=trajectory,
                evidence=evidence,
                role=role,
                raw_trajectory_sha256=record["trajectory_sha256"],
                evidence_sha256=record["evidence_sha256"],
            )
        )
    train.sort(key=lambda item: item.trajectory_id)
    for commitments in held_out.values():
        commitments.sort(key=lambda item: item.trajectory_id)
    return GraspLiftTrainingCorpus(
        train=tuple(train),
        validation_commitments=tuple(held_out[GRASP_LIFT_VALIDATION_ROLE]),
        excluded_commitments=tuple(held_out[GRASP_LIFT_EXCLUDED_ROLE]),
        source_manifest_sha256=manifest["self_sha256"],
        source_composite_corpus_sha256=manifest["composite_corpus_sha256"],
    )


def materialize_grasp_lift_model_trajectory(
    trajectory: Stage0Trajectory,
) -> Stage0Trajectory:
    """Apply the fixed state view and hybrid 13-DOF model action view."""
    if not isinstance(trajectory, Stage0Trajectory):
        raise TypeError("trajectory must be Stage0Trajectory")
    if (
        trajectory.provenance.action_schema_sha256
        != DEFAULT_GRASP_LIFT_ACTION_SCHEMA.sha256
    ):
        raise ValueError("trajectory is not a raw Grasp-Lift physical-action shard")
    if trajectory.provenance.state_schema_sha256 != DEFAULT_STAGE0_STATE_SCHEMA.sha256:
        raise ValueError("trajectory state schema changed")
    reset_rotation = trajectory.states[
        0, DEFAULT_STAGE0_STATE_SCHEMA.slice("eef_rotation_6d")
    ]
    physical_rotation = trajectory.actions[0, 3:9]
    if not torch.allclose(
        physical_rotation,
        reset_rotation,
        rtol=0.0,
        atol=1.0e-6,
    ):
        raise ValueError("physical action does not hold the reset EEF rotation")
    viewed_states = apply_grasp_lift_state_view(trajectory.states)
    reference = trajectory.actions[:1].expand_as(trajectory.actions).clone()
    model_actions = grasp_lift_physical_to_model_action(
        trajectory.states[:-1],
        trajectory.actions,
        reference,
    )
    return Stage0Trajectory(
        provenance=replace(
            trajectory.provenance,
            action_schema_sha256=GRASP_LIFT_MODEL_ACTION_VIEW_SHA256,
        ),
        states=viewed_states,
        actions=model_actions,
        outcome=trajectory.outcome,
        terminated=trajectory.terminated,
        truncated=trajectory.truncated,
        failure_reason=trajectory.failure_reason,
    )


def fit_grasp_lift_train_normalization(
    corpus: GraspLiftTrainingCorpus,
    train_trajectories: tuple[Stage0Trajectory, ...],
) -> Stage0Normalization:
    """Fit once from unique transformed train samples before window copying."""
    if not isinstance(corpus, GraspLiftTrainingCorpus):
        raise TypeError("corpus must be the strictly verified training corpus")
    if not isinstance(train_trajectories, tuple) or len(train_trajectories) != 24:
        raise ValueError("normalization requires the frozen 24-trajectory train set")
    trajectory_ids = tuple(
        trajectory.provenance.trajectory_id for trajectory in train_trajectories
    )
    if len(set(trajectory_ids)) != len(trajectory_ids):
        raise ValueError("normalization received duplicate training trajectories")
    if trajectory_ids != tuple(item.trajectory_id for item in corpus.train):
        raise ValueError("normalization training membership changed")
    expected_view_hashes = tuple(
        trajectory_sha256(materialize_grasp_lift_model_trajectory(item.trajectory))
        for item in corpus.train
    )
    if tuple(trajectory_sha256(item) for item in train_trajectories) != (
        expected_view_hashes
    ):
        raise ValueError("normalization training trajectory content changed")
    phase = torch.tensor(GRASP_LIFT_FIXED_PHASE_ONE_HOT, dtype=torch.float32)
    phase_slice = DEFAULT_GRASP_LIFT_STATE_VIEW.to_record()["phase_slice"]
    for trajectory in train_trajectories:
        if not isinstance(trajectory, Stage0Trajectory) or not trajectory.is_success:
            raise ValueError("normalization accepts successful trajectories only")
        observed_phase = trajectory.states[:, phase_slice[0] : phase_slice[1]]
        if not torch.equal(observed_phase, phase.expand_as(observed_phase)):
            raise ValueError("normalization received a raw legacy task phase")
        if bool(trajectory.actions[:, 3:9].count_nonzero()):
            raise ValueError("normalization received physical rotation targets")
    states = torch.cat(tuple(item.states for item in train_trajectories), dim=0)
    actions = torch.cat(tuple(item.actions for item in train_trajectories), dim=0)
    effective = torch.tensor(GRASP_LIFT_EFFECTIVE_ACTION_MASK, dtype=torch.bool)
    normalization = fit_stage0_normalization(
        states,
        actions,
        effective_action_mask=effective,
        epsilon=1.0e-4,
    )
    if not torch.equal(normalization.effective_action_mask, effective):
        raise RuntimeError("Grasp-Lift normalization action mask changed")
    return normalization


@dataclass(frozen=True)
class GraspLiftTrainingData:
    """Train-only model views; validation remains sealed until selection."""

    corpus: GraspLiftTrainingCorpus
    train_trajectories: tuple[Stage0Trajectory, ...]
    window_splits: Stage0WindowSplits
    normalization: Stage0Normalization
    window_phase_by_key: Mapping[tuple[str, int], int]

    def __post_init__(self) -> None:
        if not isinstance(self.corpus, GraspLiftTrainingCorpus):
            raise TypeError("corpus must be GraspLiftTrainingCorpus")
        if (
            not isinstance(self.train_trajectories, tuple)
            or len(self.train_trajectories) != 24
        ):
            raise ValueError("viewed train trajectory count changed")
        expected_ids = tuple(item.trajectory_id for item in self.corpus.train)
        observed_ids = tuple(
            item.provenance.trajectory_id for item in self.train_trajectories
        )
        if observed_ids != expected_ids:
            raise ValueError("viewed train trajectory membership changed")
        expected_trajectories = tuple(
            materialize_grasp_lift_model_trajectory(item.trajectory)
            for item in self.corpus.train
        )
        if tuple(trajectory_sha256(item) for item in self.train_trajectories) != tuple(
            trajectory_sha256(item) for item in expected_trajectories
        ):
            raise ValueError("viewed train trajectory content changed")
        if self.window_splits.validation or self.window_splits.test:
            raise ValueError("training must not materialize validation or test windows")
        expected_corpus_sha256 = trajectory_corpus_sha256(self.train_trajectories)
        if self.window_splits.corpus_sha256 != expected_corpus_sha256:
            raise ValueError("train-only viewed corpus hash changed")
        if (
            self.window_splits.split_policy_sha256
            != self.corpus.training_membership_sha256
        ):
            raise ValueError("train membership hash changed")
        if (
            self.window_splits.window_policy_sha256
            != GRASP_LIFT_TRAINING_WINDOW_POLICY.sha256
        ):
            raise ValueError("Grasp-Lift training window policy changed")
        expected_windows = tuple(
            window
            for trajectory in self.train_trajectories
            for window in extract_success_windows(
                trajectory,
                GRASP_LIFT_TRAINING_WINDOW_POLICY,
                corpus_sha256=expected_corpus_sha256,
            )
        )
        if len(self.window_splits.train) != len(expected_windows):
            raise ValueError("Grasp-Lift train window count changed")
        tensor_fields = (
            "current_state",
            "action_chunk",
            "action_mask",
            "future_actions",
            "future_states",
            "future_mask",
        )
        scalar_fields = (
            "trajectory_id",
            "reset_group_id",
            "start",
            "trajectory_sha256",
            "corpus_sha256",
            "window_policy_sha256",
        )
        for observed, expected in zip(self.window_splits.train, expected_windows):
            if any(
                getattr(observed, name) != getattr(expected, name)
                for name in scalar_fields
            ) or any(
                not torch.equal(getattr(observed, name), getattr(expected, name))
                for name in tensor_fields
            ):
                raise ValueError("Grasp-Lift train window content changed")
        expected_phase = {
            (window.trajectory_id, window.start): int(
                next(
                    item.evidence
                    for item in self.corpus.train
                    if item.trajectory_id == window.trajectory_id
                ).tensors["authoring_phase_before"][window.start]
            )
            for window in expected_windows
        }
        observed_phase = dict(self.window_phase_by_key)
        if observed_phase != expected_phase:
            raise ValueError("Grasp-Lift window phase index changed")
        if any(phase_id not in set(range(9)) for phase_id in observed_phase.values()):
            raise ValueError("Grasp-Lift window phase is outside the authoring schema")
        object.__setattr__(
            self,
            "window_phase_by_key",
            MappingProxyType(observed_phase),
        )
        expected_normalization = fit_grasp_lift_train_normalization(
            self.corpus, self.train_trajectories
        )
        if self.normalization.sha256 != expected_normalization.sha256:
            raise ValueError("Grasp-Lift train-only normalization changed")

    @property
    def dataset_sha256(self) -> str:
        return canonical_fingerprint(self.to_record())

    @property
    def window_phase_sha256(self) -> str:
        entries = [
            {
                "authoring_phase_before": self.window_phase_by_key[
                    (window.trajectory_id, window.start)
                ],
                "start": window.start,
                "trajectory_id": window.trajectory_id,
            }
            for window in self.window_splits.train
        ]
        return canonical_fingerprint(
            {
                "entries": entries,
                "schema_id": "rexpolicy/stage0-grasp-lift-window-phase/v1",
            }
        )

    @property
    def train_window_content_sha256(self) -> str:
        return _training_window_content_sha256(self.window_splits.train)

    @property
    def train_content_sha256(self) -> str:
        """Hash model-affecting train content without any held-out commitment."""
        return canonical_fingerprint(
            {
                "model_action_view_sha256": GRASP_LIFT_MODEL_ACTION_VIEW_SHA256,
                "normalization_sha256": self.normalization.sha256,
                "schema_id": "rexpolicy/stage0-grasp-lift-train-content/v1",
                "state_view_sha256": DEFAULT_GRASP_LIFT_STATE_VIEW.sha256,
                "train_viewed_corpus_sha256": self.window_splits.corpus_sha256,
                "training_membership_sha256": (self.corpus.training_membership_sha256),
                "viewed_trajectories": [
                    {
                        "trajectory_id": item.provenance.trajectory_id,
                        "viewed_trajectory_sha256": trajectory_sha256(item),
                    }
                    for item in self.train_trajectories
                ],
                "window_phase_sha256": self.window_phase_sha256,
                "window_policy_sha256": self.window_splits.window_policy_sha256,
                "window_count": len(self.window_splits.train),
                "window_content_sha256": self.train_window_content_sha256,
            }
        )

    def to_record(self) -> dict[str, Any]:
        phase_counts: dict[str, int] = {}
        for window in self.window_splits.train:
            phase = self.window_phase_by_key[(window.trajectory_id, window.start)]
            phase_counts[str(phase)] = phase_counts.get(str(phase), 0) + 1
        return {
            "counts": {
                "excluded_commitments": len(self.corpus.excluded_commitments),
                "formal_validation_commitments": len(
                    self.corpus.validation_commitments
                ),
                "train_trajectories": len(self.train_trajectories),
                "train_windows": len(self.window_splits.train),
                "validation_windows": 0,
                "test_windows": 0,
            },
            "effective_action_mask": list(GRASP_LIFT_EFFECTIVE_ACTION_MASK),
            "model_action_view_schema_id": GRASP_LIFT_MODEL_ACTION_VIEW_SCHEMA_ID,
            "model_action_view_sha256": GRASP_LIFT_MODEL_ACTION_VIEW_SHA256,
            "normalization_sha256": self.normalization.sha256,
            "phase_counts": {
                GRASP_LIFT_TRAINING_ROLE: dict(sorted(phase_counts.items()))
            },
            "phase_usage": "sampler-metadata-only; never a model input",
            "role_registry_sha256": self.corpus.role_registry_sha256,
            "schema_id": GRASP_LIFT_TRAINING_DATA_SCHEMA_ID,
            "selection_policy": (
                "formal validation deferred; consume once after training"
            ),
            "source_composite_corpus_sha256": (
                self.corpus.source_composite_corpus_sha256
            ),
            "source_action_schema_sha256": DEFAULT_GRASP_LIFT_ACTION_SCHEMA.sha256,
            "source_manifest_sha256": self.corpus.source_manifest_sha256,
            "state_view_sha256": DEFAULT_GRASP_LIFT_STATE_VIEW.sha256,
            "test_policy": "not_created",
            "training_membership_sha256": self.corpus.training_membership_sha256,
            "train_content_sha256": self.train_content_sha256,
            "train_window_content_sha256": self.train_window_content_sha256,
            "viewed_corpus_sha256": self.window_splits.corpus_sha256,
            "viewed_trajectories": [
                {
                    "trajectory_id": item.provenance.trajectory_id,
                    "viewed_trajectory_sha256": trajectory_sha256(item),
                }
                for item in self.train_trajectories
            ],
            "window_phase_sha256": self.window_phase_sha256,
            "window_policy": GRASP_LIFT_TRAINING_WINDOW_POLICY.to_record(),
            "window_policy_sha256": self.window_splits.window_policy_sha256,
        }


def build_grasp_lift_training_data(
    corpus: GraspLiftTrainingCorpus,
) -> GraspLiftTrainingData:
    """Build deterministic train-only views without materializing held-out data."""
    if not isinstance(corpus, GraspLiftTrainingCorpus):
        raise TypeError("corpus must be GraspLiftTrainingCorpus")
    train = tuple(
        materialize_grasp_lift_model_trajectory(item.trajectory)
        for item in corpus.train
    )
    viewed_corpus_sha256 = trajectory_corpus_sha256(train)

    def windows(trajectories: tuple[Stage0Trajectory, ...]) -> tuple[Any, ...]:
        return tuple(
            window
            for trajectory in trajectories
            for window in extract_success_windows(
                trajectory,
                GRASP_LIFT_TRAINING_WINDOW_POLICY,
                corpus_sha256=viewed_corpus_sha256,
            )
        )

    train_windows = windows(train)
    splits = Stage0WindowSplits(
        train=train_windows,
        validation=(),
        test=(),
        corpus_sha256=viewed_corpus_sha256,
        split_policy_sha256=corpus.training_membership_sha256,
        window_policy_sha256=GRASP_LIFT_TRAINING_WINDOW_POLICY.sha256,
    )
    evidence_by_id = {item.trajectory_id: item.evidence for item in corpus.train}
    phase_by_key = {
        (window.trajectory_id, window.start): int(
            evidence_by_id[window.trajectory_id].tensors["authoring_phase_before"][
                window.start
            ]
        )
        for window in train_windows
    }
    return GraspLiftTrainingData(
        corpus=corpus,
        train_trajectories=train,
        window_splits=splits,
        normalization=fit_grasp_lift_train_normalization(corpus, train),
        window_phase_by_key=MappingProxyType(phase_by_key),
    )


__all__ = [
    "GRASP_LIFT_ACCEPTED_PILOT_CORPUS_SHA256",
    "GRASP_LIFT_ACCEPTED_PILOT_MANIFEST_SHA256",
    "GRASP_LIFT_ENGINEERING_EXCLUDED_SEEDS",
    "GRASP_LIFT_EXCLUDED_ROLE",
    "GRASP_LIFT_FORMAL_TRAIN_SEEDS",
    "GRASP_LIFT_FORMAL_VALIDATION_SEEDS",
    "GRASP_LIFT_MODEL_ACTION_VIEW_SCHEMA_ID",
    "GRASP_LIFT_MODEL_ACTION_VIEW_SHA256",
    "GRASP_LIFT_TRAINING_DATA_SCHEMA_ID",
    "GRASP_LIFT_TRAINING_ROLE",
    "GRASP_LIFT_TRAINING_WINDOW_POLICY",
    "GRASP_LIFT_VALIDATION_ROLE",
    "GraspLiftRoleCommitment",
    "GraspLiftTrainingCorpus",
    "GraspLiftTrainingData",
    "GraspLiftTrainingMember",
    "build_grasp_lift_training_data",
    "fit_grasp_lift_train_normalization",
    "grasp_lift_training_role",
    "load_grasp_lift_training_corpus",
    "materialize_grasp_lift_model_trajectory",
]
