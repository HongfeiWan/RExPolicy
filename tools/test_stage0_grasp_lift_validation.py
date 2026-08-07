"""CPU-only tests for one-shot Grasp-Lift validation consumption."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import torch

from rexpolicy.stage0.checkpoint import (
    STAGE0_CHECKPOINT_SCHEMA_ID,
    STAGE0_CHECKPOINT_SCHEMA_VERSION,
    Stage0CheckpointHashes,
    Stage0CheckpointManager,
)
from rexpolicy.stage0.data.grasp_lift_evidence import (
    grasp_lift_evidence_sha256,
)
from rexpolicy.stage0.data.grasp_lift_training import (
    GRASP_LIFT_ENGINEERING_EXCLUDED_SEEDS,
    GRASP_LIFT_EXCLUDED_ROLE,
    GRASP_LIFT_FORMAL_TRAIN_SEEDS,
    GRASP_LIFT_FORMAL_VALIDATION_SEEDS,
    GRASP_LIFT_MODEL_ACTION_VIEW_SHA256,
    GRASP_LIFT_TRAINING_ROLE,
    GRASP_LIFT_VALIDATION_ROLE,
    GraspLiftRoleCommitment,
)
from rexpolicy.stage0.data.grasp_lift_training_artifact import (
    GRASP_LIFT_TRAINING_ARTIFACT_STORAGE_POLICY,
    GraspLiftTrainingArtifact,
)
from rexpolicy.stage0.data.grasp_lift_validation import (
    GRASP_LIFT_LOCKED_TEST_ABSENT,
    GRASP_LIFT_VALIDATION_CHECKPOINT_STEPS,
    build_grasp_lift_validation_claim_record,
    grasp_lift_validation_claim_id,
    load_claimed_grasp_lift_validation,
    preflight_grasp_lift_validation_selection,
    verify_grasp_lift_validation_claim_record,
    verify_grasp_lift_validation_preflight_record,
)
from rexpolicy.stage0.data.trajectory import trajectory_sha256
from rexpolicy.stage0.envs.grasp_lift_action import (
    GRASP_LIFT_EFFECTIVE_ACTION_MASK,
)
from rexpolicy.stage0.envs.grasp_lift_state_view import (
    DEFAULT_GRASP_LIFT_STATE_VIEW,
)
from rexpolicy.stage0.grasp_lift_no_z import (
    GRASP_LIFT_NO_Z_TRAINING_SEEDS,
    GraspLiftNoZSampler,
    build_grasp_lift_no_z_policy,
    load_grasp_lift_no_z_config,
)
from rexpolicy.stage0.grasp_lift_pilot import seal_grasp_lift_pilot_record
from rexpolicy.stage0.normalization import Stage0Normalization
from rexpolicy.stage0.types import canonical_fingerprint
from tools.test_stage0_grasp_lift_training import _evidence, _physical_trajectory

_MODULE = "rexpolicy.stage0.data.grasp_lift_validation"


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _normalization() -> Stage0Normalization:
    return Stage0Normalization(
        state_mean=torch.zeros(79, dtype=torch.float32),
        state_std=torch.ones(79, dtype=torch.float32),
        action_mean=torch.zeros(19, dtype=torch.float32),
        action_std=torch.ones(19, dtype=torch.float32),
        effective_action_mask=torch.tensor(
            GRASP_LIFT_EFFECTIVE_ACTION_MASK,
            dtype=torch.bool,
        ),
        state_sample_count=100,
        action_sample_count=100,
    )


def _commitment(
    trajectory_id: str,
    reset_seed: int,
    role: str,
    *,
    trajectory: Any | None = None,
    evidence: Any | None = None,
) -> GraspLiftRoleCommitment:
    return GraspLiftRoleCommitment(
        trajectory_id=trajectory_id,
        reset_seed=reset_seed,
        role=role,
        raw_trajectory_sha256=(
            trajectory_sha256(trajectory)
            if trajectory is not None
            else f"{reset_seed + 100:064x}"
        ),
        evidence_sha256=(
            grasp_lift_evidence_sha256(evidence)
            if evidence is not None
            else f"{reset_seed + 200:064x}"
        ),
    )


def _fake_artifact() -> tuple[
    GraspLiftTrainingArtifact,
    dict[int, Any],
    dict[int, Any],
]:
    validation_trajectories = {
        seed: _physical_trajectory(
            f"validation-{index:03d}",
            reset_seed=seed,
            value_offset=0.01 * index,
        )
        for index, seed in enumerate(GRASP_LIFT_FORMAL_VALIDATION_SEEDS)
    }
    validation_evidence = {
        seed: _evidence(trajectory)
        for seed, trajectory in validation_trajectories.items()
    }
    excluded_trajectories = {
        seed: _physical_trajectory(
            f"excluded-{index:03d}",
            reset_seed=seed,
            value_offset=0.2 + 0.01 * index,
        )
        for index, seed in enumerate(GRASP_LIFT_ENGINEERING_EXCLUDED_SEEDS)
    }
    excluded_evidence = {
        seed: _evidence(trajectory)
        for seed, trajectory in excluded_trajectories.items()
    }
    train_commitments = tuple(
        _commitment(f"train-{index:03d}", seed, GRASP_LIFT_TRAINING_ROLE)
        for index, seed in enumerate(GRASP_LIFT_FORMAL_TRAIN_SEEDS)
    )
    validation_commitments = tuple(
        _commitment(
            f"validation-{index:03d}",
            seed,
            GRASP_LIFT_VALIDATION_ROLE,
            trajectory=validation_trajectories[seed],
            evidence=validation_evidence[seed],
        )
        for index, seed in enumerate(GRASP_LIFT_FORMAL_VALIDATION_SEEDS)
    )
    excluded_commitments = tuple(
        _commitment(
            f"excluded-{index:03d}",
            seed,
            GRASP_LIFT_EXCLUDED_ROLE,
            trajectory=excluded_trajectories[seed],
            evidence=excluded_evidence[seed],
        )
        for index, seed in enumerate(GRASP_LIFT_ENGINEERING_EXCLUDED_SEEDS)
    )
    all_commitments = train_commitments + validation_commitments + excluded_commitments
    training_membership_sha256 = canonical_fingerprint(
        {
            "members": [item.to_record() for item in train_commitments],
            "schema_id": "rexpolicy/stage0-grasp-lift-train-membership/v1",
        }
    )
    role_registry_sha256 = canonical_fingerprint(
        {
            "roles": [item.to_record() for item in all_commitments],
            "schema_id": "rexpolicy/stage0-grasp-lift-frozen-roles/v1",
        }
    )
    corpus = SimpleNamespace(
        train=tuple(SimpleNamespace(commitment=item) for item in train_commitments),
        validation_commitments=validation_commitments,
        excluded_commitments=excluded_commitments,
        training_membership_sha256=training_membership_sha256,
        role_registry_sha256=role_registry_sha256,
        source_manifest_sha256="a" * 64,
        source_composite_corpus_sha256="b" * 64,
    )
    normalization = _normalization()
    train_content_sha256 = "c" * 64
    audit_dataset_sha256 = "d" * 64
    data = SimpleNamespace(
        normalization=normalization,
        train_content_sha256=train_content_sha256,
        dataset_sha256=audit_dataset_sha256,
        train_trajectories=(None,) * 24,
        window_splits=SimpleNamespace(
            train=(None,) * 1030,
            validation=(),
            test=(),
        ),
        corpus=corpus,
    )
    artifact = object.__new__(GraspLiftTrainingArtifact)
    object.__setattr__(artifact, "root", Path("/synthetic/training-artifact"))
    object.__setattr__(artifact, "data", data)
    object.__setattr__(artifact, "normalization", normalization)
    object.__setattr__(
        artifact,
        "manifest",
        {
            "audit_dataset_sha256": audit_dataset_sha256,
            "self_sha256": "e" * 64,
            "storage_policy": GRASP_LIFT_TRAINING_ARTIFACT_STORAGE_POLICY,
            "train_content_sha256": train_content_sha256,
        },
    )
    object.__setattr__(artifact, "commit", {"self_sha256": "f" * 64})
    trajectories = {**validation_trajectories, **excluded_trajectories}
    evidence = {**validation_evidence, **excluded_evidence}
    return artifact, trajectories, evidence


def _contract(
    artifact: GraspLiftTrainingArtifact,
    *,
    training_seed: int,
) -> tuple[dict[str, Any], str]:
    config = load_grasp_lift_no_z_config(
        Path(__file__).resolve().parents[1] / "configs/stage0/grasp_lift_no_z_bc.json"
    )
    policy, freeze_record = build_grasp_lift_no_z_policy(config, device="cpu")
    entries = [
        {"dtype": str(value.dtype), "name": name, "shape": list(value.shape)}
        for name, value in policy.state_dict().items()
    ]
    model_schema_sha256 = (
        __import__("hashlib")
        .sha256(
            json.dumps(
                entries,
                allow_nan=False,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("ascii")
        )
        .hexdigest()
    )
    implementation = {
        "schema_id": "rexpolicy/test-implementation/v1",
        "version": "fixed",
    }
    implementation_sha256 = canonical_fingerprint(implementation)
    world_size = 1
    run_config_sha256 = config.run_sha256(training_seed, world_size=world_size)
    sampler = GraspLiftNoZSampler(
        training_seed=training_seed,
        dataset_size=1030,
        global_batch_size=2048,
    )
    protocol_sha256 = canonical_fingerprint(
        {
            "action_feature_mask": artifact.normalization.effective_action_mask.tolist(),
            "action_model_view_sha256": GRASP_LIFT_MODEL_ACTION_VIEW_SHA256,
            "freeze_record": freeze_record,
            "implementation_sha256": implementation_sha256,
            "normalization_sha256": artifact.normalization.sha256,
            "run_config_sha256": run_config_sha256,
            "sampler": sampler.to_record(),
            "schema_id": "rexpolicy/stage0-grasp-lift-no-z-protocol/v1",
            "training_membership_sha256": (
                artifact.data.corpus.training_membership_sha256
            ),
        }
    )
    experiment_sha256 = canonical_fingerprint(
        {
            "implementation_sha256": implementation_sha256,
            "protocol_sha256": protocol_sha256,
            "run_config_sha256": run_config_sha256,
            "schema_id": "rexpolicy/stage0-grasp-lift-no-z-experiment/v1",
            "train_content_sha256": artifact.train_content_sha256,
        }
    )
    hashes = Stage0CheckpointHashes(
        config_sha256=experiment_sha256,
        state_schema_sha256=DEFAULT_GRASP_LIFT_STATE_VIEW.sha256,
        dataset_sha256=artifact.train_content_sha256,
        split_sha256=protocol_sha256,
    )
    contract = seal_grasp_lift_pilot_record(
        {
            "artifact_sha256": artifact.artifact_sha256,
            "audit_dataset_sha256": artifact.audit_dataset_sha256,
            "checkpoint_hashes": hashes.to_record(),
            "config": config.to_record(),
            "config_sha256": config.sha256,
            "cuda_runtime": {"device": "synthetic"},
            "freeze_record": freeze_record,
            "implementation": implementation,
            "implementation_sha256": implementation_sha256,
            "locked_test": "not_created",
            "normalization_sha256": artifact.normalization.sha256,
            "protocol_sha256": protocol_sha256,
            "run_config_sha256": run_config_sha256,
            "schema_id": "rexpolicy/stage0-grasp-lift-no-z-run-contract/v1",
            "train_content_sha256": artifact.train_content_sha256,
            "training_seed": training_seed,
            "validation_policy": "not opened during fixed-budget training",
            "world_size": world_size,
        }
    )
    return contract, model_schema_sha256


def _make_runs(
    root: Path,
    artifact: GraspLiftTrainingArtifact,
) -> dict[int, Path]:
    result: dict[int, Path] = {}
    for training_seed in GRASP_LIFT_NO_Z_TRAINING_SEEDS:
        run = root / f"seed-{training_seed}"
        checkpoints = run / "checkpoints"
        checkpoints.mkdir(parents=True)
        contract, model_schema_sha256 = _contract(
            artifact,
            training_seed=training_seed,
        )
        _write_json(run / "run-contract.json", contract)
        _write_json(
            run / "status.json",
            {
                "artifact_sha256": artifact.artifact_sha256,
                "attempt_index": 0,
                "complete": True,
                "final_checkpoint": Stage0CheckpointManager.checkpoint_name(10_000),
                "global_step": 10_000,
                "schema_id": "rexpolicy/stage0-grasp-lift-no-z-status/v1",
                "state": "complete",
                "train_content_sha256": artifact.train_content_sha256,
                "training_seed": training_seed,
                "validation_consumed": False,
            },
        )
        for sequence in GRASP_LIFT_VALIDATION_CHECKPOINT_STEPS:
            checkpoint = checkpoints / Stage0CheckpointManager.checkpoint_name(sequence)
            for name in ("models", "optimizers", "ranks"):
                (checkpoint / name).mkdir(parents=True, exist_ok=True)
            (checkpoint / "models/no_z_policy.pt").write_bytes(b"model")
            (checkpoint / "optimizers/no_z_policy.pt").write_bytes(b"optimizer")
            (checkpoint / "ranks/rank-00000.pt").write_bytes(b"rank")
            manifest = {
                "checkpoint_name": checkpoint.name,
                "extra": {
                    "audit_dataset_sha256": artifact.audit_dataset_sha256,
                    "implementation_sha256": contract["implementation_sha256"],
                    "locked_test": "not_created",
                    "normalization_sha256": artifact.normalization.sha256,
                    "run_contract_sha256": contract["self_sha256"],
                    "schema_id": (
                        "rexpolicy/stage0-grasp-lift-no-z-checkpoint-extra/v1"
                    ),
                    "train_content_sha256": artifact.train_content_sha256,
                    "training_only": True,
                    "validation_consumed": False,
                },
                "hashes": contract["checkpoint_hashes"],
                "models": {
                    "no_z_policy": {
                        "file": "models/no_z_policy.pt",
                        "schema_sha256": model_schema_sha256,
                        "type": "rexpolicy.stage0.runtime.Stage0FlowPolicy",
                    }
                },
                "optimizers": {
                    "no_z_policy": {
                        "file": "optimizers/no_z_policy.pt",
                        "type": "torch.optim.adamw.AdamW",
                    }
                },
                "phase": "complete" if sequence == 10_000 else "no_z_policy",
                "ranks": {"0": {"file": "ranks/rank-00000.pt"}},
                "schema_id": STAGE0_CHECKPOINT_SCHEMA_ID,
                "schema_version": STAGE0_CHECKPOINT_SCHEMA_VERSION,
                "sequence": sequence,
                "steps": {"global": sequence, "no_z_policy": sequence},
                "world_size": 1,
            }
            _write_json(checkpoint / "manifest.json", manifest)
            _write_json(
                checkpoint / "checksums.json",
                {
                    "files": {
                        "models/no_z_policy.pt": "1" * 64,
                        "optimizers/no_z_policy.pt": "2" * 64,
                        "ranks/rank-00000.pt": "3" * 64,
                    },
                    "schema_id": STAGE0_CHECKPOINT_SCHEMA_ID,
                    "schema_version": STAGE0_CHECKPOINT_SCHEMA_VERSION,
                },
            )
            _write_json(
                checkpoint / "COMPLETE",
                {
                    "checkpoint_name": checkpoint.name,
                    "checksums_sha256": "4" * 64,
                    "manifest_sha256": "5" * 64,
                    "schema_id": STAGE0_CHECKPOINT_SCHEMA_ID,
                    "schema_version": STAGE0_CHECKPOINT_SCHEMA_VERSION,
                    "sequence": sequence,
                },
            )
        result[training_seed] = run
    return result


def _verified_manifest(_manager: Any, checkpoint: str | Path) -> dict[str, Any]:
    return json.loads((Path(checkpoint) / "manifest.json").read_text(encoding="utf-8"))


def _pilot_member(commitment: GraspLiftRoleCommitment) -> dict[str, Any]:
    split = "train" if commitment.role == GRASP_LIFT_TRAINING_ROLE else "validation"
    eligible = commitment.role != GRASP_LIFT_EXCLUDED_ROLE
    return {
        "evidence_descriptor": (
            f"shards/{commitment.trajectory_id}.grasp-lift-evidence.json"
        ),
        "evidence_sha256": commitment.evidence_sha256,
        "final_authoring_phase": 7,
        "formal_gate_eligible": eligible,
        "outcome": "success",
        "reset_group_id": f"reset-{commitment.reset_seed}",
        "reset_seed": commitment.reset_seed,
        "safety_violation": False,
        "split": split,
        "terminal_transition_from_phase": 6,
        "trajectory_descriptor": (f"shards/{commitment.trajectory_id}.stage0.json"),
        "trajectory_id": commitment.trajectory_id,
        "trajectory_sha256": commitment.raw_trajectory_sha256,
    }


class GraspLiftValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.addCleanup(self.temporary.cleanup)
        self.artifact, self.trajectories, self.evidence = _fake_artifact()
        self.runs = _make_runs(self.root / "runs", self.artifact)
        self.verifier = patch(
            f"{_MODULE}.Stage0CheckpointManager.verify",
            autospec=True,
            side_effect=_verified_manifest,
        )
        self.verifier.start()
        self.addCleanup(self.verifier.stop)

    def _preflight(self) -> dict[str, Any]:
        return preflight_grasp_lift_validation_selection(
            self.runs,
            training_artifact=self.artifact,
        )

    def _claim(self, preflight: dict[str, Any]) -> dict[str, Any]:
        return build_grasp_lift_validation_claim_record(
            preflight,
            training_artifact=self.artifact,
            selection_protocol_sha256="6" * 64,
            evaluator_implementation_sha256="7" * 64,
        )

    def test_preflight_is_tensor_free_and_binds_exact_inventory(self) -> None:
        with (
            patch(
                f"{_MODULE}.load_trajectory_shard",
                side_effect=AssertionError("preflight opened validation trajectory"),
            ) as trajectory_loader,
            patch(
                f"{_MODULE}.load_grasp_lift_evidence_sidecar",
                side_effect=AssertionError("preflight opened validation evidence"),
            ) as evidence_loader,
        ):
            preflight = self._preflight()

        trajectory_loader.assert_not_called()
        evidence_loader.assert_not_called()
        self.assertEqual(len(preflight["runs"]), 3)
        self.assertTrue(all(len(run["checkpoints"]) == 21 for run in preflight["runs"]))
        self.assertEqual(preflight["locked_test"], GRASP_LIFT_LOCKED_TEST_ABSENT)
        verify_grasp_lift_validation_preflight_record(
            preflight,
            training_artifact=self.artifact,
        )

    def test_preflight_rejects_extra_partial_and_symlink_checkpoints(self) -> None:
        checkpoints = self.runs[31001] / "checkpoints"
        extra = checkpoints / ".partial-checkpoint-000000000500-deadbeef"
        extra.mkdir()
        with self.assertRaisesRegex(ValueError, "inventory"):
            self._preflight()
        extra.rmdir()

        expected = checkpoints / Stage0CheckpointManager.checkpoint_name(500)
        moved = self.root / "moved-checkpoint"
        expected.rename(moved)
        expected.symlink_to(moved, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "symlink"):
            self._preflight()

    def test_incomplete_or_validation_consumed_status_fails_closed(self) -> None:
        path = self.runs[31002] / "status.json"
        status = json.loads(path.read_text(encoding="utf-8"))
        status["validation_consumed"] = True
        _write_json(path, status)
        with self.assertRaisesRegex(ValueError, "unconsumed"):
            self._preflight()

    def test_claim_id_is_cohort_global_but_claim_binds_protocol(self) -> None:
        preflight = self._preflight()
        first = self._claim(preflight)
        second = build_grasp_lift_validation_claim_record(
            preflight,
            training_artifact=self.artifact,
            selection_protocol_sha256="8" * 64,
            evaluator_implementation_sha256="9" * 64,
        )
        self.assertEqual(first["claim_id"], second["claim_id"])
        self.assertEqual(
            first["claim_id"],
            grasp_lift_validation_claim_id(preflight["cohort_identity"]),
        )
        self.assertNotEqual(first["self_sha256"], second["self_sha256"])
        verify_grasp_lift_validation_claim_record(
            first,
            preflight=preflight,
            training_artifact=self.artifact,
        )
        tampered = dict(first)
        tampered["selection_protocol_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "self_sha256"):
            verify_grasp_lift_validation_claim_record(
                tampered,
                preflight=preflight,
                training_artifact=self.artifact,
            )

    def test_claimed_loader_opens_exactly_six_and_reuses_train_fit(self) -> None:
        preflight = self._preflight()
        claim = self._claim(preflight)
        pilot = self.root / "pilot"
        shards = pilot / "shards"
        shards.mkdir(parents=True)
        commitments = (
            tuple(item.commitment for item in self.artifact.data.corpus.train)
            + self.artifact.data.corpus.validation_commitments
            + self.artifact.data.corpus.excluded_commitments
        )
        manifest = {
            "acceptance": {"passed": True},
            "composite_corpus_sha256": (
                self.artifact.data.corpus.source_composite_corpus_sha256
            ),
            "locked_test": dict(GRASP_LIFT_LOCKED_TEST_ABSENT),
            "members": [_pilot_member(item) for item in commitments],
            "self_sha256": self.artifact.data.corpus.source_manifest_sha256,
        }
        for commitment in (
            self.artifact.data.corpus.validation_commitments
            + self.artifact.data.corpus.excluded_commitments
        ):
            for suffix in (
                ".stage0.json",
                ".stage0.pt",
                ".grasp-lift-evidence.json",
                ".grasp-lift-evidence.pt",
            ):
                (shards / f"{commitment.trajectory_id}{suffix}").write_bytes(b"x")

        trajectory_paths: list[Path] = []
        evidence_paths: list[Path] = []

        def load_trajectory(path: str | Path) -> Any:
            selected = Path(path)
            trajectory_paths.append(selected)
            seed = next(
                item.reset_seed
                for item in self.artifact.data.corpus.validation_commitments
                if selected.name.startswith(item.trajectory_id + ".")
            )
            return self.trajectories[seed]

        def load_evidence(
            path: str | Path,
            *,
            trajectory: Any,
        ) -> Any:
            selected = Path(path)
            evidence_paths.append(selected)
            return self.evidence[trajectory.provenance.reset_seed]

        with (
            patch(
                f"{_MODULE}._load_accepted_pilot_registry",
                return_value=(pilot, manifest),
            ),
            patch(
                f"{_MODULE}.load_trajectory_shard",
                side_effect=load_trajectory,
            ),
            patch(
                f"{_MODULE}.load_grasp_lift_evidence_sidecar",
                side_effect=load_evidence,
            ),
        ):
            data = load_claimed_grasp_lift_validation(
                pilot,
                training_artifact=self.artifact,
                preflight=preflight,
                claim=claim,
            )

        self.assertEqual(len(trajectory_paths), 6)
        self.assertEqual(len(evidence_paths), 6)
        opened = {path.name for path in trajectory_paths + evidence_paths}
        for commitment in self.artifact.data.corpus.excluded_commitments:
            self.assertFalse(
                any(name.startswith(commitment.trajectory_id) for name in opened)
            )
        self.assertEqual(
            tuple(item.commitment.reset_seed for item in data.members),
            GRASP_LIFT_FORMAL_VALIDATION_SEEDS,
        )
        self.assertEqual(
            data.record["normalization_sha256"],
            self.artifact.normalization.sha256,
        )
        self.assertEqual(
            data.record["normalization_policy"],
            "reuse_exact_train_fit; never fit validation",
        )
        self.assertEqual(data.record["locked_test"], GRASP_LIFT_LOCKED_TEST_ABSENT)
        unsealed = dict(data.record)
        observed_self_sha256 = unsealed.pop("self_sha256")
        self.assertEqual(canonical_fingerprint(unsealed), observed_self_sha256)
        self.assertEqual(len(data.record["data_sha256"]), 64)

    def test_invalid_claim_opens_no_shard(self) -> None:
        preflight = self._preflight()
        claim = self._claim(preflight)
        claim["preflight_sha256"] = "0" * 64
        with (
            patch(f"{_MODULE}.load_trajectory_shard") as trajectory_loader,
            patch(f"{_MODULE}.load_grasp_lift_evidence_sidecar") as evidence_loader,
            self.assertRaisesRegex(ValueError, "self_sha256"),
        ):
            load_claimed_grasp_lift_validation(
                self.root / "does-not-matter",
                training_artifact=self.artifact,
                preflight=preflight,
                claim=claim,
            )
        trajectory_loader.assert_not_called()
        evidence_loader.assert_not_called()


if __name__ == "__main__":
    unittest.main()
