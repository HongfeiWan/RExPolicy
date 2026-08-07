#!/usr/bin/env python3
"""CPU tests for Grasp-Lift no-z validation diagnostics and conversion."""

from __future__ import annotations

import unittest

import torch
from torch import nn

from rexpolicy.stage0.envs.grasp_lift_oracle import GraspLiftSuccessOracle
from rexpolicy.stage0.envs.state_schema import DEFAULT_STAGE0_STATE_SCHEMA
from rexpolicy.stage0.evaluation.grasp_lift_no_z_evaluation import (
    evaluate_grasp_lift_offline_action_diagnostic,
    grasp_lift_rollout_to_selection_evidence,
)
from rexpolicy.stage0.evaluation.grasp_lift_rollout import (
    GraspLiftRolloutBudget,
    GraspLiftRolloutResult,
    grasp_lift_rollout_protocol,
)
from rexpolicy.stage0.grasp_lift_no_z_selection import (
    GRASP_LIFT_NO_Z_VALIDATION_RESET_SEEDS,
)
from rexpolicy.stage0.normalization import Stage0Normalization
from rexpolicy.stage0.trainers.batch import Stage0WindowBatch
from rexpolicy.stage0.types import canonical_fingerprint


def _normalization() -> Stage0Normalization:
    effective = torch.tensor(
        (True, True, True) + (False,) * 6 + (True,) * 10,
        dtype=torch.bool,
    )
    action_std = torch.ones(19, dtype=torch.float32)
    action_std[:3] = 0.01
    action_std[9:19] = 0.1
    return Stage0Normalization(
        state_mean=torch.zeros(79, dtype=torch.float32),
        state_std=torch.ones(79, dtype=torch.float32),
        action_mean=torch.zeros(19, dtype=torch.float32),
        action_std=action_std,
        effective_action_mask=effective,
        state_sample_count=100,
        action_sample_count=100,
    )


def _batch() -> Stage0WindowBatch:
    action = torch.zeros(3, 2, 19, dtype=torch.float32)
    action[:, :, :3] = 2.0
    action[:, :, 9:19] = -0.5
    mask = torch.tensor([[True, True], [True, False], [True, True]])
    future_mask = torch.cat((mask, torch.ones(3, 1, dtype=torch.bool)), dim=1)
    return Stage0WindowBatch(
        current_states=torch.zeros(3, 79, dtype=torch.float32),
        action_chunks=action,
        action_mask=mask,
        future_actions=torch.zeros(3, 3, 19, dtype=torch.float32),
        future_states=torch.zeros(3, 3, 79, dtype=torch.float32),
        future_mask=future_mask,
        trajectory_ids=("a", "b", "c"),
        reset_group_ids=("a", "b", "c"),
        starts=(0, 1, 2),
        corpus_sha256="a" * 64,
        window_policy_sha256="b" * 64,
    )


class _DiagnosticPolicy(nn.Module):
    def __init__(self, output: torch.Tensor) -> None:
        super().__init__()
        self.state_dim = 79
        self.action_dim = 19
        self.action_horizon = 2
        self.anchor = nn.Parameter(torch.zeros(()))
        self.output = output
        self.noises: list[torch.Tensor] = []

    def sample(
        self,
        state: torch.Tensor,
        *,
        success_latent: torch.Tensor | None,
        steps: int,
        action_mask: torch.Tensor,
        action_feature_mask: torch.Tensor,
        initial_noise: torch.Tensor,
    ) -> torch.Tensor:
        del state
        self.success_latent = success_latent
        self.steps = steps
        self.action_mask = action_mask.clone()
        self.action_feature_mask = action_feature_mask.clone()
        self.noises.append(initial_noise.clone())
        return self.output.clone()


def _validation_sha256() -> str:
    return canonical_fingerprint({"validation": "fixed"})


def _rollout_result() -> GraspLiftRolloutResult:
    groups = tuple(
        (f"reset-{seed}", seed)
        for seed in GRASP_LIFT_NO_Z_VALIDATION_RESET_SEEDS
    )
    budget = GraspLiftRolloutBudget.from_reset_groups(groups)
    count = budget.episode_count
    success = torch.ones(count, dtype=torch.bool)
    success[1] = False
    failure = torch.zeros(count, dtype=torch.bool)
    failure[1] = True
    zeros = torch.zeros(count, dtype=torch.bool)
    safety = zeros.clone()
    safety[1] = True
    oracle = zeros.clone()
    oracle[2] = True
    execution = zeros.clone()
    execution[3] = True
    success[2:4] = False
    failure[2:4] = True
    return GraspLiftRolloutResult(
        budget=budget,
        protocol=grasp_lift_rollout_protocol(
            _normalization(),
            GraspLiftSuccessOracle(count),
            action_horizon=8,
        ),
        success=success,
        failure=failure,
        timeout=zeros.clone(),
        completed_steps=torch.ones(count, dtype=torch.int64),
        forbidden_contact_violation=safety,
        lateral_displacement_violation=zeros.clone(),
        bottle_tilt_violation=zeros.clone(),
        dropped_grasp_violation=zeros.clone(),
        collision_buffer_overflow_violation=zeros.clone(),
        oracle_input_disagreement=oracle,
        execution_integrity_violation=execution,
        final_lift_height_m=torch.zeros(count),
        final_lateral_displacement_m=torch.zeros(count),
        final_bottle_tilt_rad=torch.zeros(count),
    )


class GraspLiftNoZEvaluationTest(unittest.TestCase):
    def test_exact_prediction_has_zero_physical_and_normalized_error(self) -> None:
        batch = _batch()
        policy = _DiagnosticPolicy(batch.action_chunks)
        diagnostic = evaluate_grasp_lift_offline_action_diagnostic(
            policy,
            batch,
            _normalization(),
            validation_data_sha256=_validation_sha256(),
        )

        self.assertEqual(diagnostic["normalized_effective_action_mse"], 0.0)
        self.assertEqual(diagnostic["normalized_mse_to_train_zero_ratio"], 0.0)
        self.assertEqual(diagnostic["xyz_physical_rmse_m"], 0.0)
        self.assertEqual(diagnostic["hand_physical_rmse_rad"], 0.0)
        self.assertGreater(
            diagnostic["train_zero_normalized_effective_action_mse"],
            0.0,
        )
        self.assertIsNone(policy.success_latent)
        self.assertEqual(policy.steps, 16)
        self.assertTrue(policy.training)

    def test_train_zero_prediction_has_ratio_one_and_fixed_paired_noise(self) -> None:
        batch = _batch()
        policies = (
            _DiagnosticPolicy(torch.zeros_like(batch.action_chunks)),
            _DiagnosticPolicy(torch.zeros_like(batch.action_chunks)),
        )
        diagnostics = tuple(
            evaluate_grasp_lift_offline_action_diagnostic(
                policy,
                batch,
                _normalization(),
                validation_data_sha256=_validation_sha256(),
            )
            for policy in policies
        )

        self.assertEqual(diagnostics[0], diagnostics[1])
        self.assertAlmostEqual(
            diagnostics[0]["normalized_mse_to_train_zero_ratio"],
            1.0,
        )
        torch.testing.assert_close(
            policies[0].noises[0],
            policies[1].noises[0],
            rtol=0.0,
            atol=0.0,
        )

    def test_rollout_conversion_preserves_canonical_cells_and_fault_classes(
        self,
    ) -> None:
        evidence = grasp_lift_rollout_to_selection_evidence(
            _rollout_result(),
            checkpoint_step=500,
            training_seed=31_001,
            model_sha256="c" * 64,
            offline_mse_diagnostic={"normalized": 0.5},
        )

        self.assertEqual(len(evidence.rollouts), 24)
        self.assertEqual(evidence.rollouts[0].inventory_key, (7009, 0))
        self.assertTrue(evidence.rollouts[1].safety_violation)
        self.assertTrue(evidence.rollouts[2].oracle_disagreement)
        self.assertFalse(evidence.rollouts[2].integrity_violation)
        self.assertTrue(evidence.rollouts[3].integrity_violation)
        self.assertFalse(evidence.rollouts[3].oracle_disagreement)
        self.assertFalse(evidence.rollouts[1].success)

    def test_rollout_conversion_rejects_nonformal_reset_grid(self) -> None:
        result = _rollout_result()
        wrong_budget = GraspLiftRolloutBudget.from_reset_groups(
            tuple((f"wrong-{index}", index) for index in range(6))
        )
        with self.assertRaisesRegex(ValueError, "canonical"):
            grasp_lift_rollout_to_selection_evidence(
                GraspLiftRolloutResult(
                    budget=wrong_budget,
                    protocol=result.protocol,
                    success=result.success,
                    failure=result.failure,
                    timeout=result.timeout,
                    completed_steps=result.completed_steps,
                    forbidden_contact_violation=result.forbidden_contact_violation,
                    lateral_displacement_violation=(
                        result.lateral_displacement_violation
                    ),
                    bottle_tilt_violation=result.bottle_tilt_violation,
                    dropped_grasp_violation=result.dropped_grasp_violation,
                    collision_buffer_overflow_violation=(
                        result.collision_buffer_overflow_violation
                    ),
                    oracle_input_disagreement=result.oracle_input_disagreement,
                    execution_integrity_violation=(
                        result.execution_integrity_violation
                    ),
                    final_lift_height_m=result.final_lift_height_m,
                    final_lateral_displacement_m=(
                        result.final_lateral_displacement_m
                    ),
                    final_bottle_tilt_rad=result.final_bottle_tilt_rad,
                ),
                checkpoint_step=500,
                training_seed=31_001,
                model_sha256="c" * 64,
                offline_mse_diagnostic={"normalized": 0.5},
            )


if __name__ == "__main__":
    unittest.main()
