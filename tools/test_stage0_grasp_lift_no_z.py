"""Focused CPU tests for the pre-registered Grasp-Lift no-z baseline."""

from __future__ import annotations

import copy
import unittest
from dataclasses import replace
from pathlib import Path

import torch

from rexpolicy.stage0.envs.grasp_lift_action import (
    GRASP_LIFT_EFFECTIVE_ACTION_MASK,
)
from rexpolicy.stage0.envs.grasp_lift_state_view import (
    GRASP_LIFT_FIXED_PHASE_ONE_HOT,
)
from rexpolicy.stage0.envs.state_schema import DEFAULT_STAGE0_STATE_SCHEMA
from rexpolicy.stage0.grasp_lift_no_z import (
    GRASP_LIFT_NO_Z_TRAINING_SEEDS,
    GraspLiftNoZConfig,
    GraspLiftNoZSampler,
    build_grasp_lift_no_z_policy,
    load_grasp_lift_no_z_config,
    normalize_grasp_lift_no_z_batch_once,
)
from rexpolicy.stage0.normalization import Stage0Normalization
from rexpolicy.stage0.trainers.batch import Stage0WindowBatch


def _config_record() -> dict[str, object]:
    return {
        "data": {
            "expected_train_trajectories": 24,
            "expected_train_windows": 1030,
            "sampler_id": (
                "rexpolicy/global-no-replacement-cross-epoch-step-indexed/v1"
            ),
        },
        "model": {
            "action_dim": 19,
            "action_horizon": 8,
            "attention_heads": 4,
            "dropout": 0.0,
            "feedforward_dim": 256,
            "latent_dim": 32,
            "model_dim": 128,
            "state_dim": 79,
            "transformer_layers": 3,
        },
        "optimization": {
            "checkpoint_every_steps": 500,
            "cuda_fused_adamw": True,
            "global_batch_size": 2048,
            "gradient_clip_norm": 1.0,
            "learning_rate": 3.0e-4,
            "log_every_steps": 50,
            "total_steps": 10000,
            "weight_decay": 1.0e-4,
        },
        "runtime": {
            "allow_tf32": False,
            "cpu_threads": 16,
            "cublas_workspace_config": ":4096:8",
            "cudnn_benchmark": False,
            "deterministic_algorithms": True,
            "dtype": "float32",
            "interop_threads": 4,
            "nvidia_tf32_override": "0",
            "require_cuda": True,
            "torch_compile": False,
            "torch_allow_tf32_cublas_override": "0",
        },
        "schema_id": "rexpolicy/stage0-grasp-lift-no-z-config/v1",
        "schema_version": 1,
        "training_seeds": [31001, 31002, 31003],
    }


def _normalization() -> Stage0Normalization:
    state_mean = torch.linspace(-0.2, 0.2, 79, dtype=torch.float32)
    state_std = torch.linspace(0.5, 1.5, 79, dtype=torch.float32)
    action_mean = torch.zeros(19, dtype=torch.float32)
    action_std = torch.ones(19, dtype=torch.float32)
    action_mean[:3] = torch.tensor((0.05, -0.1, 0.2))
    action_std[:3] = torch.tensor((2.0, 4.0, 5.0))
    action_mean[9:] = torch.linspace(-0.3, 0.3, 10)
    action_std[9:] = torch.linspace(0.5, 1.4, 10)
    return Stage0Normalization(
        state_mean=state_mean,
        state_std=state_std,
        action_mean=action_mean,
        action_std=action_std,
        effective_action_mask=torch.tensor(
            GRASP_LIFT_EFFECTIVE_ACTION_MASK,
            dtype=torch.bool,
        ),
        state_sample_count=1054,
        action_sample_count=1030,
        epsilon=1.0e-4,
    )


def _raw_model_batch() -> Stage0WindowBatch:
    batch_size = 2
    action_horizon = 8
    future_horizon = 48
    states = torch.zeros((batch_size, 79), dtype=torch.float32)
    eef_slice = DEFAULT_STAGE0_STATE_SCHEMA.slice("eef_position")
    phase_slice = DEFAULT_STAGE0_STATE_SCHEMA.slice("task_phase_one_hot")
    states[:, eef_slice] = torch.tensor(
        ((100.0, 200.0, 300.0), (-100.0, -200.0, -300.0))
    )
    states[:, phase_slice] = torch.tensor(GRASP_LIFT_FIXED_PHASE_ONE_HOT)

    action_chunks = torch.zeros((batch_size, action_horizon, 19), dtype=torch.float32)
    action_chunks[..., :3] = torch.tensor((0.25, -0.5, 0.7))
    action_chunks[..., 9:] = torch.linspace(-0.2, 0.7, 10)
    future_actions = torch.zeros((batch_size, future_horizon, 19), dtype=torch.float32)
    future_actions[..., :3] = torch.tensor((-0.15, 0.3, 0.45))
    future_actions[..., 9:] = torch.linspace(0.1, 1.0, 10)
    future_states = states[:, None, :].expand(-1, future_horizon, -1).clone()
    action_mask = torch.ones((batch_size, action_horizon), dtype=torch.bool)
    future_mask = torch.ones((batch_size, future_horizon), dtype=torch.bool)
    return Stage0WindowBatch(
        current_states=states,
        action_chunks=action_chunks,
        action_mask=action_mask,
        future_actions=future_actions,
        future_states=future_states,
        future_mask=future_mask,
        trajectory_ids=("trajectory-a", "trajectory-b"),
        reset_group_ids=("reset-a", "reset-b"),
        starts=(0, 1),
        corpus_sha256="a" * 64,
        window_policy_sha256="b" * 64,
    )


class GraspLiftNoZTests(unittest.TestCase):
    def test_config_is_exactly_three_seeds_and_preregistered_budget(self) -> None:
        config = load_grasp_lift_no_z_config(
            Path(__file__).resolve().parents[1]
            / "configs"
            / "stage0"
            / "grasp_lift_no_z_bc.json"
        )
        self.assertEqual(config.to_record(), _config_record())
        self.assertEqual(config.training_seeds, GRASP_LIFT_NO_Z_TRAINING_SEEDS)
        self.assertEqual(config.latent_dim, 32)
        self.assertEqual(config.global_batch_size, 2048)
        self.assertEqual(config.total_steps, 10000)
        self.assertEqual(config.checkpoint_every_steps, 500)
        self.assertEqual(config.log_every_steps, 50)
        self.assertEqual(config.expected_train_windows, 1030)
        self.assertEqual(GraspLiftNoZConfig.from_record(config.to_record()), config)
        for seed in GRASP_LIFT_NO_Z_TRAINING_SEEDS:
            config.require_training_seed(seed)
            self.assertEqual(len(config.run_sha256(seed, world_size=4)), 64)

        changed = copy.deepcopy(_config_record())
        changed["training_seeds"] = [31001, 31002]
        with self.assertRaisesRegex(ValueError, "training seeds changed"):
            GraspLiftNoZConfig.from_record(changed)
        changed = copy.deepcopy(_config_record())
        changed["training_seeds"] = [31001, 31002, 31004]
        with self.assertRaisesRegex(ValueError, "training seeds changed"):
            GraspLiftNoZConfig.from_record(changed)
        changed = copy.deepcopy(_config_record())
        changed["unexpected"] = True
        with self.assertRaisesRegex(ValueError, "fields changed"):
            GraspLiftNoZConfig.from_record(changed)
        with self.assertRaisesRegex(ValueError, "not one of the three"):
            config.require_training_seed(31004)

    def test_no_z_uses_one_token_and_freezes_the_entire_latent_branch(self) -> None:
        config = GraspLiftNoZConfig.from_record(_config_record())
        policy, freeze = build_grasp_lift_no_z_policy(config, device="cpu")
        state = torch.randn(2, config.state_dim)
        no_z = policy.condition_encoder(state, None)
        self.assertEqual(tuple(no_z.tokens.shape), (2, 1, config.model_dim))
        self.assertTrue(no_z.attention_mask.all())

        frozen = tuple(freeze["frozen_parameter_names"])
        expected = tuple(
            name
            for name, _ in policy.named_parameters()
            if name == "condition_encoder.latent_token_type"
            or name.startswith("condition_encoder.latent_projector.")
        )
        self.assertEqual(frozen, expected)
        self.assertGreater(len(frozen), 1)
        for name, parameter in policy.named_parameters():
            self.assertEqual(parameter.requires_grad, name not in expected)

        with_z = policy.condition_encoder(
            state,
            torch.zeros(2, config.latent_dim),
        )
        self.assertEqual(tuple(with_z.tokens.shape), (2, 2, config.model_dim))

    def test_normalization_is_once_only_delta_preserving_and_13d(self) -> None:
        raw = _raw_model_batch()
        normalization = _normalization()
        self.assertEqual(
            int(normalization.effective_action_mask.count_nonzero()),
            13,
        )
        normalized = normalize_grasp_lift_no_z_batch_once(
            raw,
            normalization,
            source_corpus_sha256=raw.corpus_sha256,
        )

        expected_xyz = (
            raw.action_chunks[0, 0, :3] - normalization.action_mean[:3]
        ) / normalization.action_std[:3]
        torch.testing.assert_close(normalized.action_chunks[0, 0, :3], expected_xyz)
        # A second XYZ subtraction from the deliberately huge EEF state would
        # make these values enormous; the hybrid action is already a delta.
        self.assertLess(float(normalized.action_chunks[0, 0, :3].abs().max()), 1.0)
        self.assertFalse(bool(normalized.action_chunks[..., 3:9].count_nonzero()))
        self.assertFalse(bool(normalized.future_actions[..., 3:9].count_nonzero()))
        self.assertNotEqual(normalized.corpus_sha256, raw.corpus_sha256)
        with self.assertRaisesRegex(ValueError, "normalized twice"):
            normalize_grasp_lift_no_z_batch_once(
                normalized,
                normalization,
                source_corpus_sha256=raw.corpus_sha256,
            )

        wrong = replace(
            normalization,
            effective_action_mask=torch.ones(19, dtype=torch.bool),
        )
        with self.assertRaisesRegex(ValueError, "13D mask"):
            normalize_grasp_lift_no_z_batch_once(
                raw,
                wrong,
                source_corpus_sha256=raw.corpus_sha256,
            )

    def test_sampler_is_epochwise_without_replacement_and_step_resumable(self) -> None:
        sampler = GraspLiftNoZSampler(training_seed=31001)
        epoch_zero = sampler._permutation(0)
        epoch_one = sampler._permutation(1)
        expected_members = torch.arange(1030, dtype=torch.int64)
        torch.testing.assert_close(torch.sort(epoch_zero).values, expected_members)
        torch.testing.assert_close(torch.sort(epoch_one).values, expected_members)
        self.assertEqual(int(epoch_zero.unique().numel()), 1030)
        self.assertEqual(int(epoch_one.unique().numel()), 1030)
        self.assertFalse(torch.equal(epoch_zero, epoch_one))

        first = sampler.global_indices(0)
        torch.testing.assert_close(first[:1030], epoch_zero)
        torch.testing.assert_close(first[1030:], epoch_one[:1018])
        cursor = sampler.cursor(1)
        self.assertEqual(
            (cursor.optimizer_step, cursor.total_samples, cursor.epoch, cursor.offset),
            (1, 2048, 1, 1018),
        )
        second = sampler.global_indices(1)
        torch.testing.assert_close(second[:12], epoch_one[1018:])

        ranks = tuple(
            sampler.rank_indices(7, rank=rank, world_size=4) for rank in range(4)
        )
        self.assertTrue(all(value.numel() == 512 for value in ranks))
        torch.testing.assert_close(torch.cat(ranks), sampler.global_indices(7))

        # Resume is addressed by optimizer step, with no mutable iterator
        # state to guess or serialize.
        restored = GraspLiftNoZSampler(training_seed=31001)
        torch.testing.assert_close(
            restored.global_indices(7),
            sampler.global_indices(7),
        )
        different_seed = GraspLiftNoZSampler(training_seed=31002)
        self.assertFalse(
            torch.equal(different_seed.global_indices(7), sampler.global_indices(7))
        )

    def test_illegal_world_batch_and_sampler_coordinates_fail_closed(self) -> None:
        config = GraspLiftNoZConfig.from_record(_config_record())
        with self.assertRaisesRegex(ValueError, "divisible by world_size"):
            config.run_sha256(31001, world_size=3)
        with self.assertRaisesRegex(ValueError, "positive integer"):
            config.run_sha256(31001, world_size=0)
        with self.assertRaisesRegex(ValueError, "sampler shape changed"):
            GraspLiftNoZSampler(training_seed=31001, global_batch_size=1024)
        sampler = GraspLiftNoZSampler(training_seed=31001)
        with self.assertRaisesRegex(ValueError, "divisible by world_size"):
            sampler.rank_indices(0, rank=0, world_size=3)
        with self.assertRaisesRegex(ValueError, "rank must lie"):
            sampler.rank_indices(0, rank=4, world_size=4)
        with self.assertRaisesRegex(ValueError, "non-negative integer"):
            sampler.global_indices(-1)


if __name__ == "__main__":
    unittest.main()
