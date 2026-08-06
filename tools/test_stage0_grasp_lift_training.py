"""Tests for leakage-safe Grasp-Lift training views and frozen roles."""

from __future__ import annotations

import unittest
from dataclasses import fields
from pathlib import Path
from unittest.mock import patch

import torch

from rexpolicy.stage0.data.grasp_lift_evidence import (
    GRASP_LIFT_COMPOSITE_CORPUS_SCHEMA_ID,
    GRASP_LIFT_EVIDENCE_TENSOR_SPECS,
    GraspLiftEvidence,
    GraspLiftEvidenceMetadata,
    grasp_lift_evidence_sha256,
)
from rexpolicy.stage0.data.grasp_lift_training import (
    GRASP_LIFT_ACCEPTED_PILOT_CORPUS_SHA256,
    GRASP_LIFT_ACCEPTED_PILOT_MANIFEST_SHA256,
    GRASP_LIFT_ENGINEERING_EXCLUDED_SEEDS,
    GRASP_LIFT_EXCLUDED_ROLE,
    GRASP_LIFT_FORMAL_TRAIN_SEEDS,
    GRASP_LIFT_FORMAL_VALIDATION_SEEDS,
    GRASP_LIFT_MODEL_ACTION_VIEW_SHA256,
    GRASP_LIFT_TRAINING_ROLE,
    GRASP_LIFT_TRAINING_WINDOW_POLICY,
    GRASP_LIFT_VALIDATION_ROLE,
    GraspLiftRoleCommitment,
    GraspLiftTrainingCorpus,
    GraspLiftTrainingData,
    GraspLiftTrainingMember,
    build_grasp_lift_training_data,
    fit_grasp_lift_train_normalization,
    grasp_lift_training_role,
    load_grasp_lift_training_corpus,
    materialize_grasp_lift_model_trajectory,
)
from rexpolicy.stage0.data.trajectory import (
    Stage0Trajectory,
    Stage0TrajectoryProvenance,
    trajectory_sha256,
)
from rexpolicy.stage0.data.windows import extract_success_windows
from rexpolicy.stage0.envs.grasp_lift_action import (
    DEFAULT_GRASP_LIFT_ACTION_SCHEMA,
    GRASP_LIFT_EFFECTIVE_ACTION_MASK,
)
from rexpolicy.stage0.envs.grasp_lift_modes import (
    DEFAULT_GRASP_LIFT_AUTHORING_MODE,
)
from rexpolicy.stage0.envs.grasp_lift_state_view import (
    GRASP_LIFT_FIXED_PHASE_ONE_HOT,
)
from rexpolicy.stage0.envs.state_schema import DEFAULT_STAGE0_STATE_SCHEMA
from rexpolicy.stage0.types import Stage0Outcome, canonical_fingerprint


def _physical_trajectory(
    trajectory_id: str,
    *,
    reset_seed: int,
    transition_count: int = 3,
    value_offset: float = 0.0,
) -> Stage0Trajectory:
    state_dim = DEFAULT_STAGE0_STATE_SCHEMA.dimension
    states = torch.zeros((transition_count + 1, state_dim), dtype=torch.float32)
    step = torch.arange(transition_count + 1, dtype=torch.float32)
    states[:, 0] = value_offset + step * 0.25

    hand_slice = DEFAULT_STAGE0_STATE_SCHEMA.slice("hand_joint_position")
    hand_base = torch.linspace(-0.3, 0.6, 10, dtype=torch.float32)
    states[:, hand_slice] = hand_base + step[:, None] * 0.01 + value_offset

    eef_slice = DEFAULT_STAGE0_STATE_SCHEMA.slice("eef_position")
    states[:, eef_slice] = torch.stack(
        (
            value_offset + 0.01 * step,
            -0.2 + 0.02 * step,
            0.5 + 0.03 * step,
        ),
        dim=1,
    )
    reset_rotation = torch.tensor((1.0, 0.0, 0.0, 0.0, 1.0, 0.0), dtype=torch.float32)
    states[:, DEFAULT_STAGE0_STATE_SCHEMA.slice("eef_rotation_6d")] = reset_rotation
    states[:, DEFAULT_STAGE0_STATE_SCHEMA.slice("object_to_goal")] = torch.tensor(
        (0.0, 0.0, 1.0), dtype=torch.float32
    )
    phase_slice = DEFAULT_STAGE0_STATE_SCHEMA.slice("task_phase_one_hot")
    states[:, phase_slice] = torch.tensor(
        (0.0, 1.0, 0.0, 0.0, 0.0), dtype=torch.float32
    )

    actions = torch.zeros((transition_count, 19), dtype=torch.float32)
    action_step = torch.arange(transition_count, dtype=torch.float32)
    delta = torch.stack(
        (
            0.01 + action_step * 0.001,
            -0.02 - action_step * 0.002,
            0.03 + action_step * 0.003,
        ),
        dim=1,
    )
    actions[:, :3] = states[:-1, eef_slice] + delta
    actions[:, 3:9] = reset_rotation
    actions[:, 9:19] = (
        torch.linspace(0.1, 1.0, 10, dtype=torch.float32)
        + action_step[:, None] * 0.02
        + value_offset * 0.1
    )

    provenance = Stage0TrajectoryProvenance(
        trajectory_id=trajectory_id,
        reset_group_id=f"reset-{reset_seed}",
        task_id="grasp_lift_green_bottle/v1",
        oracle_id="grasp-lift-oracle/v1",
        simulator_sha256="1" * 64,
        experiment_sha256="2" * 64,
        state_schema_sha256=DEFAULT_STAGE0_STATE_SCHEMA.sha256,
        action_schema_sha256=DEFAULT_GRASP_LIFT_ACTION_SCHEMA.sha256,
        generation=1,
        rank=0,
        world=0,
        reset_seed=reset_seed,
    )
    return Stage0Trajectory(
        provenance=provenance,
        states=states,
        actions=actions,
        outcome=Stage0Outcome.SUCCESS,
        terminated=True,
        truncated=False,
    )


def _evidence(trajectory: Stage0Trajectory) -> GraspLiftEvidence:
    count = trajectory.transition_count
    dtype_by_name = {
        "float32": torch.float32,
        "bool": torch.bool,
        "int64": torch.int64,
    }
    tensors = {
        name: torch.zeros(spec.shape(count), dtype=dtype_by_name[spec.dtype])
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
    tensors["authoring_phase_before"] = torch.arange(count, dtype=torch.int64) % 5
    tensors["authoring_phase_after"] = torch.arange(count, dtype=torch.int64) % 5
    metadata = GraspLiftEvidenceMetadata.from_trajectory(
        trajectory,
        mode_id=DEFAULT_GRASP_LIFT_AUTHORING_MODE.mode_id,
        mode_sha256=DEFAULT_GRASP_LIFT_AUTHORING_MODE.sha256,
        reset_lift_axis=(0.0, 0.0, 1.0),
    )
    return GraspLiftEvidence(metadata=metadata, tensors=tensors)


def _member(
    trajectory_id: str,
    *,
    reset_seed: int,
    role: str,
    value_offset: float,
) -> GraspLiftTrainingMember:
    trajectory = _physical_trajectory(
        trajectory_id,
        reset_seed=reset_seed,
        value_offset=value_offset,
    )
    evidence = _evidence(trajectory)
    return GraspLiftTrainingMember(
        trajectory=trajectory,
        evidence=evidence,
        role=role,
        raw_trajectory_sha256=trajectory_sha256(trajectory),
        evidence_sha256=grasp_lift_evidence_sha256(evidence),
    )


def _fake_corpus_members() -> tuple[
    tuple[GraspLiftTrainingMember, ...],
    tuple[GraspLiftRoleCommitment, ...],
    tuple[GraspLiftRoleCommitment, ...],
]:
    train = tuple(
        _member(
            f"train-{index:03d}",
            reset_seed=GRASP_LIFT_FORMAL_TRAIN_SEEDS[index],
            role=GRASP_LIFT_TRAINING_ROLE,
            value_offset=float(index) / 10.0,
        )
        for index in range(24)
    )
    validation_commitments = tuple(
        GraspLiftRoleCommitment(
            trajectory_id=f"validation-{index:03d}",
            reset_seed=seed,
            role=GRASP_LIFT_VALIDATION_ROLE,
            raw_trajectory_sha256=f"{seed:064x}",
            evidence_sha256=f"{seed + 1:064x}",
        )
        for index, seed in enumerate(GRASP_LIFT_FORMAL_VALIDATION_SEEDS)
    )
    excluded_commitments = tuple(
        GraspLiftRoleCommitment(
            trajectory_id=f"excluded-{index:03d}",
            reset_seed=seed,
            role=GRASP_LIFT_EXCLUDED_ROLE,
            raw_trajectory_sha256=f"{seed:064x}",
            evidence_sha256=f"{seed + 1:064x}",
        )
        for index, seed in enumerate(GRASP_LIFT_ENGINEERING_EXCLUDED_SEEDS)
    )
    return train, validation_commitments, excluded_commitments


def _consistent_synthetic_corpus() -> GraspLiftTrainingCorpus:
    train, validation, excluded = _fake_corpus_members()
    commitments = tuple(member.commitment for member in train) + validation + excluded
    synthetic_composite_sha256 = canonical_fingerprint(
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
                key=lambda item: item["trajectory_id"],
            ),
            "schema_id": GRASP_LIFT_COMPOSITE_CORPUS_SCHEMA_ID,
        }
    )
    constant = (
        "rexpolicy.stage0.data.grasp_lift_training."
        "GRASP_LIFT_ACCEPTED_PILOT_CORPUS_SHA256"
    )
    with patch(constant, synthetic_composite_sha256):
        return GraspLiftTrainingCorpus(
            train=train,
            validation_commitments=validation,
            excluded_commitments=excluded,
            source_manifest_sha256=GRASP_LIFT_ACCEPTED_PILOT_MANIFEST_SHA256,
            source_composite_corpus_sha256=synthetic_composite_sha256,
        )


class GraspLiftTrainingTests(unittest.TestCase):
    def test_member_roles_are_frozen_without_generic_resplitting(self) -> None:
        self.assertEqual(
            grasp_lift_training_role(
                {
                    "split": "train",
                    "reset_seed": GRASP_LIFT_FORMAL_TRAIN_SEEDS[0],
                    "formal_gate_eligible": True,
                }
            ),
            GRASP_LIFT_TRAINING_ROLE,
        )
        with self.assertRaisesRegex(ValueError, "train cohort changed"):
            grasp_lift_training_role(
                {
                    "split": "train",
                    "reset_seed": 6001,
                    "formal_gate_eligible": True,
                }
            )
        for seed in GRASP_LIFT_FORMAL_VALIDATION_SEEDS:
            self.assertEqual(
                grasp_lift_training_role(
                    {
                        "split": "validation",
                        "reset_seed": seed,
                        "formal_gate_eligible": True,
                    }
                ),
                GRASP_LIFT_VALIDATION_ROLE,
            )
        for seed in GRASP_LIFT_ENGINEERING_EXCLUDED_SEEDS:
            self.assertEqual(
                grasp_lift_training_role(
                    {
                        "split": "validation",
                        "reset_seed": seed,
                        "formal_gate_eligible": False,
                    }
                ),
                GRASP_LIFT_EXCLUDED_ROLE,
            )
        with self.assertRaisesRegex(ValueError, "validation cohort changed"):
            grasp_lift_training_role(
                {
                    "split": "validation",
                    "reset_seed": 7999,
                    "formal_gate_eligible": True,
                }
            )
        with self.assertRaisesRegex(ValueError, "unexpected ineligible"):
            grasp_lift_training_role(
                {
                    "split": "train",
                    "reset_seed": 7001,
                    "formal_gate_eligible": False,
                }
            )

    def test_model_view_transforms_actions_without_mutating_raw_data(self) -> None:
        raw = _physical_trajectory("transform-000", reset_seed=42, transition_count=5)
        raw_states = raw.states.clone()
        raw_actions = raw.actions.clone()
        raw_sha256 = trajectory_sha256(raw)

        viewed = materialize_grasp_lift_model_trajectory(raw)

        self.assertEqual(trajectory_sha256(raw), raw_sha256)
        self.assertTrue(torch.equal(raw.states, raw_states))
        self.assertTrue(torch.equal(raw.actions, raw_actions))
        expected_states = raw_states.clone()
        phase_slice = DEFAULT_STAGE0_STATE_SCHEMA.slice("task_phase_one_hot")
        expected_states[:, phase_slice] = torch.tensor(
            GRASP_LIFT_FIXED_PHASE_ONE_HOT, dtype=torch.float32
        )
        self.assertTrue(torch.equal(viewed.states, expected_states))

        expected_actions = raw_actions.clone()
        expected_actions[:, :3] -= raw_states[
            :-1, DEFAULT_STAGE0_STATE_SCHEMA.slice("eef_position")
        ]
        expected_actions[:, 3:9] = 0.0
        self.assertTrue(torch.equal(viewed.actions, expected_actions))
        self.assertTrue(torch.equal(viewed.actions[:, 9:19], raw_actions[:, 9:19]))
        self.assertEqual(
            viewed.provenance.action_schema_sha256,
            GRASP_LIFT_MODEL_ACTION_VIEW_SHA256,
        )
        raw_provenance = raw.provenance.to_record()
        viewed_provenance = viewed.provenance.to_record()
        raw_provenance.pop("action_schema_sha256")
        viewed_provenance.pop("action_schema_sha256")
        self.assertEqual(viewed_provenance, raw_provenance)
        self.assertEqual(viewed.outcome, raw.outcome)
        self.assertNotEqual(viewed.states.data_ptr(), raw.states.data_ptr())
        self.assertNotEqual(viewed.actions.data_ptr(), raw.actions.data_ptr())
        with self.assertRaisesRegex(ValueError, "not a raw Grasp-Lift"):
            materialize_grasp_lift_model_trajectory(viewed)

    def test_normalization_uses_exactly_24_train_trajectories(self) -> None:
        corpus = _consistent_synthetic_corpus()
        train = tuple(
            materialize_grasp_lift_model_trajectory(member.trajectory)
            for member in corpus.train
        )
        normalization = fit_grasp_lift_train_normalization(corpus, train)
        expected_mask = torch.tensor(GRASP_LIFT_EFFECTIVE_ACTION_MASK, dtype=torch.bool)

        self.assertTrue(torch.equal(normalization.effective_action_mask, expected_mask))
        self.assertEqual(int(expected_mask.sum()), 13)
        self.assertEqual(normalization.state_sample_count, 24 * 4)
        self.assertEqual(normalization.action_sample_count, 24 * 3)
        self.assertTrue(torch.equal(normalization.action_mean[3:9], torch.zeros(6)))
        self.assertTrue(torch.equal(normalization.action_std[3:9], torch.ones(6)))
        expected_xyz_mean = (
            torch.cat(tuple(item.actions[:, :3] for item in train), dim=0)
            .to(torch.float64)
            .mean(dim=0)
            .to(torch.float32)
        )
        self.assertTrue(torch.equal(normalization.action_mean[:3], expected_xyz_mean))

        validation = materialize_grasp_lift_model_trajectory(
            _physical_trajectory(
                "normalization-validation",
                reset_seed=9000,
                transition_count=4,
                value_offset=10000.0,
            )
        )
        with self.assertRaisesRegex(ValueError, "frozen 24-trajectory train set"):
            fit_grasp_lift_train_normalization(corpus, train + (validation,))
        duplicate_train = train[:-1] + (train[0],)
        with self.assertRaisesRegex(ValueError, "duplicate training trajectories"):
            fit_grasp_lift_train_normalization(corpus, duplicate_train)
        replaced_train = train[:-1] + (validation,)
        with self.assertRaisesRegex(ValueError, "training membership changed"):
            fit_grasp_lift_train_normalization(corpus, replaced_train)

    def test_held_out_roles_are_tensor_free_and_fake_source_is_rejected(self) -> None:
        train, validation, excluded = _fake_corpus_members()
        held_out = validation + excluded
        self.assertEqual(len(held_out), 8)
        for commitment in held_out:
            self.assertFalse(hasattr(commitment, "trajectory"))
            self.assertFalse(hasattr(commitment, "evidence"))
        self.assertNotIn(
            "validation_trajectories",
            {field.name for field in fields(GraspLiftTrainingData)},
        )
        with self.assertRaisesRegex(
            ValueError, "role commitments do not form the source corpus"
        ):
            GraspLiftTrainingCorpus(
                train=train,
                validation_commitments=validation,
                excluded_commitments=excluded,
                source_manifest_sha256=(GRASP_LIFT_ACCEPTED_PILOT_MANIFEST_SHA256),
                source_composite_corpus_sha256=(
                    GRASP_LIFT_ACCEPTED_PILOT_CORPUS_SHA256
                ),
            )

    def test_window_policy_covers_the_complete_42_step_future(self) -> None:
        policy = GRASP_LIFT_TRAINING_WINDOW_POLICY
        self.assertEqual(
            (policy.action_horizon, policy.future_horizon, policy.stride),
            (8, 48, 1),
        )
        trajectory = materialize_grasp_lift_model_trajectory(
            _physical_trajectory("window-000", reset_seed=1234, transition_count=42)
        )
        windows = extract_success_windows(trajectory, policy)
        self.assertEqual(len(windows), 42)
        self.assertEqual(tuple(window.start for window in windows), tuple(range(42)))
        self.assertEqual(tuple(windows[0].action_chunk.shape), (8, 19))
        self.assertEqual(tuple(windows[0].future_actions.shape), (48, 19))
        self.assertEqual(int(windows[0].future_mask.sum()), 42)
        self.assertFalse(bool(windows[0].future_mask[42:].any()))
        self.assertFalse(bool(windows[0].future_actions[42:].count_nonzero()))
        self.assertEqual(int(windows[-1].future_mask.sum()), 1)
        self.assertEqual(int(windows[-1].action_mask.sum()), 1)

    def test_train_hashes_bind_exact_window_content(self) -> None:
        data = build_grasp_lift_training_data(_consistent_synthetic_corpus())
        self.assertEqual(len(data.window_splits.train), 72)
        self.assertFalse(data.window_splits.validation)
        self.assertFalse(data.window_splits.test)
        window_sha256 = data.train_window_content_sha256
        train_sha256 = data.train_content_sha256
        dataset_sha256 = data.dataset_sha256

        padded_value = data.window_splits.train[0].future_actions[-1, 0]
        self.assertEqual(float(padded_value), 0.0)
        padded_value.add_(1.0)

        self.assertNotEqual(data.train_window_content_sha256, window_sha256)
        self.assertNotEqual(data.train_content_sha256, train_sha256)
        self.assertNotEqual(data.dataset_sha256, dataset_sha256)

    def test_training_loader_never_opens_held_out_shards(self) -> None:
        source = _consistent_synthetic_corpus()

        def record(commitment: GraspLiftRoleCommitment) -> dict[str, object]:
            if commitment.role == GRASP_LIFT_TRAINING_ROLE:
                split = "train"
                eligible = True
            elif commitment.role == GRASP_LIFT_VALIDATION_ROLE:
                split = "validation"
                eligible = True
            else:
                split = "validation"
                eligible = False
            return {
                "evidence_descriptor": (
                    f"shards/{commitment.trajectory_id}.grasp-lift-evidence.json"
                ),
                "evidence_sha256": commitment.evidence_sha256,
                "formal_gate_eligible": eligible,
                "reset_seed": commitment.reset_seed,
                "split": split,
                "trajectory_descriptor": (
                    f"shards/{commitment.trajectory_id}.stage0.json"
                ),
                "trajectory_id": commitment.trajectory_id,
                "trajectory_sha256": commitment.raw_trajectory_sha256,
            }

        commitments = (
            tuple(member.commitment for member in source.train)
            + source.validation_commitments
            + source.excluded_commitments
        )
        manifest = {
            "acceptance": {"passed": True},
            "composite_corpus_sha256": source.source_composite_corpus_sha256,
            "locked_test": {
                "artifact": None,
                "consumed": False,
                "policy": "not_created",
            },
            "members": [record(item) for item in commitments],
            "self_sha256": GRASP_LIFT_ACCEPTED_PILOT_MANIFEST_SHA256,
        }
        module = "rexpolicy.stage0.data.grasp_lift_training"
        train_trajectories = [item.trajectory for item in source.train]
        train_evidence = [item.evidence for item in source.train]
        with (
            patch(
                f"{module}.GRASP_LIFT_ACCEPTED_PILOT_CORPUS_SHA256",
                source.source_composite_corpus_sha256,
            ),
            patch(
                f"{module}._load_accepted_pilot_registry",
                return_value=(Path("/unread-synthetic-pilot"), manifest),
            ),
            patch(
                f"{module}._strict_train_shard_path",
                side_effect=lambda _root, relative, **_kwargs: Path(relative),
            ) as strict_path,
            patch(
                f"{module}.load_trajectory_shard",
                side_effect=train_trajectories,
            ) as load_trajectory,
            patch(
                f"{module}.load_grasp_lift_evidence_sidecar",
                side_effect=train_evidence,
            ) as load_evidence,
        ):
            loaded = load_grasp_lift_training_corpus("unused")

        self.assertEqual(len(loaded.train), 24)
        self.assertEqual(load_trajectory.call_count, 24)
        self.assertEqual(load_evidence.call_count, 24)
        self.assertEqual(strict_path.call_count, 48)
        opened_ids = {
            call.kwargs["trajectory_id"] for call in strict_path.call_args_list
        }
        self.assertEqual(opened_ids, {item.trajectory_id for item in source.train})
        self.assertTrue(
            opened_ids.isdisjoint(
                {
                    item.trajectory_id
                    for item in source.validation_commitments
                    + source.excluded_commitments
                }
            )
        )


if __name__ == "__main__":
    unittest.main()
