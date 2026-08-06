"""End-to-end integrity tests for a committed Grasp-Lift pilot artifact."""

from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from typing import Any, Mapping

import torch

from rexpolicy.stage0.data.grasp_lift_evidence import (
    GRASP_LIFT_EVIDENCE_TENSOR_SPECS,
    GraspLiftEvidence,
    GraspLiftEvidenceMetadata,
    grasp_lift_composite_corpus_sha256,
    grasp_lift_evidence_sha256,
    write_grasp_lift_evidence_sidecar,
)
from rexpolicy.stage0.data.store import write_trajectory_shard
from rexpolicy.stage0.data.trajectory import (
    Stage0Trajectory,
    Stage0TrajectoryProvenance,
    trajectory_sha256,
)
from rexpolicy.stage0.envs.grasp_lift_action import (
    DEFAULT_GRASP_LIFT_ACTION_SCHEMA,
)
from rexpolicy.stage0.envs.grasp_lift_modes import (
    DEFAULT_GRASP_LIFT_AUTHORING_MODE,
)
from rexpolicy.stage0.envs.grasp_lift_oracle import GRASP_LIFT_ORACLE_ID
from rexpolicy.stage0.envs.grasp_lift_state_view import (
    DEFAULT_GRASP_LIFT_STATE_VIEW,
)
from rexpolicy.stage0.envs.state_schema import DEFAULT_STAGE0_STATE_SCHEMA
from rexpolicy.stage0.grasp_lift_pilot import (
    GRASP_LIFT_PILOT_COMMIT_SCHEMA_ID,
    GRASP_LIFT_PILOT_FORMAL_CONFIG_SHA256,
    GRASP_LIFT_PILOT_OUTPUT_SCHEMA_ID,
    GRASP_LIFT_PILOT_STATUS_SCHEMA_ID,
    GraspLiftPilotConfig,
    assign_grasp_lift_pilot_splits,
    evaluate_grasp_lift_close7_authoring_gate,
    grasp_lift_pilot_authoring_lineage_record,
    grasp_lift_pilot_environment_config_record,
    grasp_lift_pilot_formal_gate_eligible,
    grasp_lift_pilot_oracle_record,
    grasp_lift_pilot_split_policy,
    grasp_lift_pilot_task_metadata_record,
    grasp_lift_pilot_validation_claim_record,
    load_grasp_lift_pilot_manifest,
    seal_grasp_lift_pilot_record,
)
from rexpolicy.stage0.implementation_fingerprint import (
    STAGE0_IMPLEMENTATION_DISTRIBUTIONS,
    Stage0DistributionVersion,
    Stage0ImplementationFingerprint,
)
from rexpolicy.stage0.types import Stage0Outcome, canonical_fingerprint
from tools.run_stage0_grasp_lift_pilot import _claim_validation_once


def _write_json(path: Path, record: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(record, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _config() -> GraspLiftPilotConfig:
    return GraspLiftPilotConfig.from_mapping(
        {
            "acceptance": {
                "locked_test_policy": "not_created",
                "minimum_success_rate": 0.75,
                "require_zero_oracle_failures": True,
                "require_zero_safety_violations": True,
                "validation_fraction": 0.25,
            },
            "runtime": {
                "cpu_threads": 16,
                "device": "cuda:0",
                "max_episode_steps": 72,
                "num_envs": 8,
                "reset_groups": 32,
                "rigid_contacts_per_env": 2048,
                "seed": 7000,
                "triangle_pairs_per_env": 200000,
            },
            "schema_version": 1,
        }
    )


def _evidence_tensors(trajectory: Stage0Trajectory) -> dict[str, torch.Tensor]:
    dtypes = {
        "bool": torch.bool,
        "float32": torch.float32,
        "int64": torch.int64,
    }
    tensors = {
        name: torch.zeros(
            spec.shape(trajectory.transition_count),
            dtype=dtypes[spec.dtype],
        )
        for name, spec in GRASP_LIFT_EVIDENCE_TENSOR_SPECS.items()
    }
    tensors["proposal_action"] = trajectory.actions.clone()
    tensors["executed_action"] = trajectory.actions.clone()
    tensors["measured_eef_position"] = trajectory.states[
        1:, DEFAULT_STAGE0_STATE_SCHEMA.slice("eef_position")
    ].clone()
    tensors["measured_hand_joint_position"] = trajectory.states[
        1:, DEFAULT_STAGE0_STATE_SCHEMA.slice("hand_joint_position")
    ].clone()
    tensors["triangle_pair_buffer_available"].fill_(True)
    return tensors


def _write_committed_artifact(
    root: Path,
    *,
    all_eligible_success: bool = False,
    excluded_failure: bool = False,
) -> dict[str, Any]:
    config = _config()
    shards = root / "shards"
    shards.mkdir()
    implementation_fingerprint = Stage0ImplementationFingerprint(
        git_head="1" * 40,
        python_implementation="CPython",
        python_version="3.11.0",
        distributions=tuple(
            Stage0DistributionVersion(
                component=component,
                distribution=distribution,
                version=None,
            )
            for component, distribution in STAGE0_IMPLEMENTATION_DISTRIBUTIONS
        ),
    )
    implementation = implementation_fingerprint.to_record()
    implementation_sha256 = implementation_fingerprint.sha256
    oracle_record = grasp_lift_pilot_oracle_record()
    simulator_record = {
        "action_schema_sha256": DEFAULT_GRASP_LIFT_ACTION_SCHEMA.sha256,
        "authoring_mode_sha256": DEFAULT_GRASP_LIFT_AUTHORING_MODE.sha256,
        "environment_config": grasp_lift_pilot_environment_config_record(
            config.runtime
        ),
        "implementation_sha256": implementation_sha256,
        "oracle": oracle_record,
        "oracle_id": GRASP_LIFT_ORACLE_ID,
        "schema_id": "rexpolicy/stage0-grasp-lift-newton-runtime/v1",
        "task_metadata": grasp_lift_pilot_task_metadata_record(),
    }
    simulator_sha256 = canonical_fingerprint(simulator_record)
    corpus = []
    members = []
    for group_index in range(config.runtime.reset_groups):
        reset_seed = config.runtime.seed + group_index
        formal_gate_eligible = grasp_lift_pilot_formal_gate_eligible(
            reset_seed
        )
        if all_eligible_success and formal_gate_eligible:
            outcome = Stage0Outcome.SUCCESS
            terminated = True
            truncated = False
            failure_reason = None
            final_phase = 5
            terminal_from_phase = 4
        elif excluded_failure and reset_seed == 7001:
            outcome = Stage0Outcome.FAILURE
            terminated = True
            truncated = False
            failure_reason = "forbidden_contact"
            final_phase = 6
            terminal_from_phase = 3
        else:
            outcome = Stage0Outcome.TIMEOUT
            terminated = False
            truncated = True
            failure_reason = None
            final_phase = 0
            terminal_from_phase = 0
        reset_group_id = (
            f"grasp-reset-g{group_index:06d}-s{reset_seed:010d}"
        )
        trajectory_id = (
            f"grasp-g{group_index:06d}-thumb-index-side-close7-"
            f"h{DEFAULT_GRASP_LIFT_AUTHORING_MODE.sha256[:12]}-"
            f"s{reset_seed:010d}"
        )
        states = torch.zeros(
            (2, DEFAULT_STAGE0_STATE_SCHEMA.dimension),
            dtype=torch.float32,
        )
        states[:, DEFAULT_STAGE0_STATE_SCHEMA.slice("object_to_goal")] = (
            torch.tensor((0.0, 0.0, 1.0), dtype=torch.float32)
        )
        trajectory = Stage0Trajectory(
            provenance=Stage0TrajectoryProvenance(
                trajectory_id=trajectory_id,
                reset_group_id=reset_group_id,
                task_id="grasp_lift_green_bottle/v1",
                oracle_id=GRASP_LIFT_ORACLE_ID,
                simulator_sha256=simulator_sha256,
                experiment_sha256=config.sha256,
                state_schema_sha256=DEFAULT_STAGE0_STATE_SCHEMA.sha256,
                action_schema_sha256=DEFAULT_GRASP_LIFT_ACTION_SCHEMA.sha256,
                generation=0,
                rank=0,
                world=group_index % config.runtime.num_envs,
                reset_seed=reset_seed,
            ),
            states=states,
            actions=torch.zeros((1, 19), dtype=torch.float32),
            outcome=outcome,
            terminated=terminated,
            truncated=truncated,
            failure_reason=failure_reason,
        )
        evidence_tensors = _evidence_tensors(trajectory)
        if outcome is Stage0Outcome.SUCCESS:
            evidence_tensors["authoring_phase_before"].fill_(4)
            evidence_tensors["authoring_phase_after"].fill_(5)
            evidence_tensors["oracle_success"].fill_(True)
        elif outcome is Stage0Outcome.FAILURE:
            evidence_tensors["authoring_phase_before"].fill_(3)
            evidence_tensors["authoring_phase_after"].fill_(6)
            evidence_tensors["oracle_failure"].fill_(True)
            evidence_tensors["forbidden_contact_violation"].fill_(True)
        evidence = GraspLiftEvidence(
            metadata=GraspLiftEvidenceMetadata.from_trajectory(
                trajectory,
                mode_id=DEFAULT_GRASP_LIFT_AUTHORING_MODE.mode_id,
                mode_sha256=DEFAULT_GRASP_LIFT_AUTHORING_MODE.sha256,
                reset_lift_axis=(0.0, 0.0, 1.0),
                implementation_sha256=implementation_sha256,
            ),
            tensors=evidence_tensors,
        )
        trajectory_descriptor = write_trajectory_shard(shards, trajectory)
        evidence_descriptor = write_grasp_lift_evidence_sidecar(
            shards,
            evidence,
            trajectory=trajectory,
        )
        corpus.append((trajectory, evidence))
        members.append(
            {
                "evidence_descriptor": str(
                    evidence_descriptor.relative_to(root)
                ),
                "evidence_sha256": grasp_lift_evidence_sha256(evidence),
                "final_authoring_phase": final_phase,
                "formal_gate_eligible": formal_gate_eligible,
                "outcome": outcome.value,
                "reset_group_id": reset_group_id,
                "reset_seed": reset_seed,
                "safety_violation": outcome is Stage0Outcome.FAILURE,
                "trajectory_descriptor": str(
                    trajectory_descriptor.relative_to(root)
                ),
                "trajectory_id": trajectory_id,
                "trajectory_sha256": trajectory_sha256(trajectory),
                "terminal_transition_from_phase": terminal_from_phase,
            }
        )
    split_by_id = assign_grasp_lift_pilot_splits(
        {
            member["trajectory_id"]: member["reset_group_id"]
            for member in members
        },
        validation_fraction=0.25,
    )
    for member in members:
        member["split"] = split_by_id[member["trajectory_id"]]
    outcomes = {
        outcome: sum(member["outcome"] == outcome for member in members)
        for outcome in ("failure", "success", "timeout")
    }
    split_counts = {
        split: {
            outcome: sum(
                member["split"] == split and member["outcome"] == outcome
                for member in members
            )
            for outcome in outcomes
        }
        for split in ("train", "validation")
    }
    split_policy = grasp_lift_pilot_split_policy()
    authoring_lineage = grasp_lift_pilot_authoring_lineage_record()
    authoring_gate = evaluate_grasp_lift_close7_authoring_gate(members)
    eligible_member_count = sum(
        member["formal_gate_eligible"] for member in members
    )
    eligible_success_count = sum(
        member["formal_gate_eligible"] and member["outcome"] == "success"
        for member in members
    )
    eligible_failure_count = sum(
        member["formal_gate_eligible"] and member["outcome"] == "failure"
        for member in members
    )
    eligible_safety_violation_count = sum(
        member["formal_gate_eligible"] and member["safety_violation"]
        for member in members
    )
    safety_violation_count = sum(
        member["safety_violation"] for member in members
    )
    success_rate = eligible_success_count / float(eligible_member_count)
    acceptance_passed = bool(
        success_rate >= config.acceptance.minimum_success_rate
        and eligible_failure_count == 0
        and eligible_safety_violation_count == 0
        and authoring_gate["passed"]
    )
    validation_claim = seal_grasp_lift_pilot_record(
        grasp_lift_pilot_validation_claim_record(
            experiment_sha256=config.sha256,
            implementation_sha256=implementation_sha256,
        )
    )
    manifest = seal_grasp_lift_pilot_record(
        {
            "acceptance": {
                "authoring_gate": authoring_gate,
                "eligible_member_count": eligible_member_count,
                "eligible_failure_count": eligible_failure_count,
                "eligible_safety_violation_count": (
                    eligible_safety_violation_count
                ),
                "eligible_success_count": eligible_success_count,
                "engineering_smoke_excluded_member_count": 2,
                "passed": acceptance_passed,
                "policy": config.acceptance.to_record(),
                "safety_violation_count": safety_violation_count,
                "success_rate": success_rate,
            },
            "action_schema": DEFAULT_GRASP_LIFT_ACTION_SCHEMA.to_record(),
            "action_schema_sha256": DEFAULT_GRASP_LIFT_ACTION_SCHEMA.sha256,
            "authoring_mode": DEFAULT_GRASP_LIFT_AUTHORING_MODE.to_record(),
            "authoring_mode_sha256": DEFAULT_GRASP_LIFT_AUTHORING_MODE.sha256,
            "authoring_lineage": authoring_lineage,
            "authoring_lineage_sha256": canonical_fingerprint(
                authoring_lineage
            ),
            "composite_corpus_sha256": grasp_lift_composite_corpus_sha256(
                corpus
            ),
            "cpu_affinity": list(range(16)),
            "elapsed_seconds": 1.0,
            "experiment_sha256": config.sha256,
            "implementation": implementation,
            "implementation_sha256": implementation_sha256,
            "locked_test": {
                "artifact": None,
                "consumed": False,
                "policy": "not_created",
            },
            "members": sorted(members, key=lambda member: member["trajectory_id"]),
            "outcome_counts": outcomes,
            "schema_id": GRASP_LIFT_PILOT_OUTPUT_SCHEMA_ID,
            "simulator_record": simulator_record,
            "simulator_sha256": simulator_sha256,
            "split_counts": split_counts,
            "split_policy": split_policy,
            "split_policy_sha256": canonical_fingerprint(split_policy),
            "state_view": DEFAULT_GRASP_LIFT_STATE_VIEW.to_record(),
            "state_view_sha256": DEFAULT_GRASP_LIFT_STATE_VIEW.sha256,
            "validation_claim": validation_claim,
            "validation_claim_sha256": validation_claim["self_sha256"],
            "visuals": {
                "camera_textures": False,
                "scene_asset": None,
                "scene_visuals": False,
            },
            "world_steps": 32,
        }
    )
    status = seal_grasp_lift_pilot_record(
        {
            "acceptance_passed": acceptance_passed,
            "complete": True,
            "manifest_sha256": manifest["self_sha256"],
            "outcome_counts": outcomes,
            "schema_id": GRASP_LIFT_PILOT_STATUS_SCHEMA_ID,
        }
    )
    commit = seal_grasp_lift_pilot_record(
        {
            "config_file": "config.json",
            "config_sha256": config.sha256,
            "manifest_file": "manifest.json",
            "manifest_sha256": manifest["self_sha256"],
            "schema_id": GRASP_LIFT_PILOT_COMMIT_SCHEMA_ID,
            "status_file": "status.json",
            "status_sha256": status["self_sha256"],
        }
    )
    _write_json(root / "config.json", config.to_record())
    _write_json(root / "manifest.json", manifest)
    _write_json(root / "status.json", status)
    _write_json(root / "commit.json", commit)
    return manifest


class GraspLiftPilotArtifactTest(unittest.TestCase):
    def test_validation_claim_is_exclusive_across_worktree_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repository = root / "repository"
            sibling = root / "sibling"
            subprocess.run(
                ["git", "init", "--quiet", str(repository)], check=True
            )
            (repository / "marker.txt").write_text("claim test\n", encoding="utf-8")
            subprocess.run(
                ["git", "-C", str(repository), "add", "marker.txt"],
                check=True,
            )
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(repository),
                    "-c",
                    "user.name=RExPolicy Test",
                    "-c",
                    "user.email=test@example.invalid",
                    "commit",
                    "--quiet",
                    "-m",
                    "claim test",
                ],
                check=True,
            )
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(repository),
                    "worktree",
                    "add",
                    "--quiet",
                    "--detach",
                    str(sibling),
                ],
                check=True,
            )
            with self.assertRaisesRegex(ValueError, "formal pilot config"):
                _claim_validation_once(
                    repository_root=repository,
                    experiment_sha256="a" * 64,
                    implementation_sha256="b" * 64,
                )
            self.assertFalse(
                (repository / ".git" / "rexpolicy-validation-claims").exists()
            )
            claim = _claim_validation_once(
                repository_root=repository,
                experiment_sha256=GRASP_LIFT_PILOT_FORMAL_CONFIG_SHA256,
                implementation_sha256="b" * 64,
            )
            self.assertEqual(len(claim["self_sha256"]), 64)
            with self.assertRaisesRegex(RuntimeError, "already claimed"):
                _claim_validation_once(
                    repository_root=sibling,
                    experiment_sha256=GRASP_LIFT_PILOT_FORMAL_CONFIG_SHA256,
                    implementation_sha256="d" * 64,
                )

    def test_excluded_failure_is_diagnostic_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = _write_committed_artifact(
                root,
                all_eligible_success=True,
                excluded_failure=True,
            )
            loaded = load_grasp_lift_pilot_manifest(root)
            acceptance = loaded["acceptance"]
            self.assertTrue(acceptance["passed"])
            self.assertEqual(acceptance["eligible_failure_count"], 0)
            self.assertEqual(acceptance["eligible_safety_violation_count"], 0)
            self.assertEqual(acceptance["safety_violation_count"], 1)
            self.assertEqual(manifest["outcome_counts"]["failure"], 1)

    def test_complete_artifact_round_trip_recomputes_the_corpus(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = _write_committed_artifact(root)
            loaded = load_grasp_lift_pilot_manifest(root)
            self.assertEqual(loaded["self_sha256"], manifest["self_sha256"])

    def test_resealed_false_member_hash_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = _write_committed_artifact(root)
            manifest.pop("self_sha256")
            manifest["members"][0]["trajectory_sha256"] = "f" * 64
            manifest = seal_grasp_lift_pilot_record(manifest)
            status = _read_unsealed(root / "status.json")
            status["manifest_sha256"] = manifest["self_sha256"]
            status = seal_grasp_lift_pilot_record(status)
            commit = _read_unsealed(root / "commit.json")
            commit["manifest_sha256"] = manifest["self_sha256"]
            commit["status_sha256"] = status["self_sha256"]
            commit = seal_grasp_lift_pilot_record(commit)
            _write_json(root / "manifest.json", manifest)
            _write_json(root / "status.json", status)
            _write_json(root / "commit.json", commit)
            with self.assertRaisesRegex(ValueError, "trajectory hash changed"):
                load_grasp_lift_pilot_manifest(root)

    def test_missing_payload_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = _write_committed_artifact(root)
            trajectory_id = manifest["members"][0]["trajectory_id"]
            (root / "shards" / f"{trajectory_id}.stage0.pt").unlink()
            with self.assertRaisesRegex(ValueError, "payload.*missing"):
                load_grasp_lift_pilot_manifest(root)


def _read_unsealed(path: Path) -> dict[str, Any]:
    record = json.loads(path.read_text(encoding="utf-8"))
    record.pop("self_sha256")
    return record


if __name__ == "__main__":
    unittest.main()
