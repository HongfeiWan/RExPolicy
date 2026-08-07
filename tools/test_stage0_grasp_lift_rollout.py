#!/usr/bin/env python3
"""CPU contract tests for learned Stage 0 Grasp-Lift rollouts."""

from __future__ import annotations

import unittest
from typing import Any

import torch
from torch import nn

from rexpolicy.stage0.envs.state_only import Stage0Transition
from rexpolicy.stage0.envs.state_schema import DEFAULT_STAGE0_STATE_SCHEMA
from rexpolicy.stage0.evaluation.grasp_lift_rollout import (
    GRASP_LIFT_FORMAL_NOISE_VARIANTS,
    GraspLiftRolloutBudget,
    rollout_grasp_lift_no_z_policy,
)
from rexpolicy.stage0.normalization import Stage0Normalization
from rexpolicy.stage0.types import Stage0Outcome


def _base_state(count: int) -> torch.Tensor:
    state = torch.zeros(
        count,
        DEFAULT_STAGE0_STATE_SCHEMA.dimension,
        dtype=torch.float32,
    )
    state[:, DEFAULT_STAGE0_STATE_SCHEMA.slice("eef_rotation_6d")] = torch.tensor(
        [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    )
    state[:, DEFAULT_STAGE0_STATE_SCHEMA.slice("object_rotation_6d")] = torch.tensor(
        [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    )
    state[:, DEFAULT_STAGE0_STATE_SCHEMA.slice("goal_position")] = torch.tensor(
        [0.0, 0.0, 0.1]
    )
    state[:, DEFAULT_STAGE0_STATE_SCHEMA.slice("object_to_goal")] = torch.tensor(
        [0.0, 0.0, 0.1]
    )
    state[:, DEFAULT_STAGE0_STATE_SCHEMA.slice("task_phase_one_hot")] = torch.tensor(
        [0.0, 0.0, 1.0, 0.0, 0.0]
    )
    return state


class _FakeGraspLiftEnv:
    def __init__(
        self,
        count: int,
        *,
        never_grasp: bool = False,
        forbidden_world: int | None = None,
        disagree_world: int | None = None,
        drift_world: int | None = None,
        nonzero_reward_world: int | None = None,
        overflow: bool = False,
    ) -> None:
        self.num_envs = count
        self.never_grasp = never_grasp
        self.forbidden_world = forbidden_world
        self.disagree_world = disagree_world
        self.drift_world = drift_world
        self.nonzero_reward_world = nonzero_reward_world
        self.overflow = overflow
        self.reset_seeds: tuple[int, ...] | None = None
        self.actions: list[torch.Tensor] = []
        self._state = _base_state(count)
        self._hold = torch.zeros(count, 19, dtype=torch.float32)
        self._hold[:, 3:9] = self._state[
            :, DEFAULT_STAGE0_STATE_SCHEMA.slice("eef_rotation_6d")
        ]
        self._actual = self._hold.clone()
        self._step = 0

    def reset(self, *, seed: list[int]) -> tuple[torch.Tensor, dict[str, Any]]:
        self.reset_seeds = tuple(seed)
        self.actions = []
        self._state = _base_state(self.num_envs)
        self._actual = self._hold.clone()
        self._step = 0
        return self._state.clone(), {}

    def hold_action_torch(self) -> torch.Tensor:
        return self._hold

    def project_effective_action_torch(self, action: torch.Tensor) -> torch.Tensor:
        projected = action.clone()
        projected[:, 3:9] = self._hold[:, 3:9]
        hand = self._state[
            :, DEFAULT_STAGE0_STATE_SCHEMA.slice("hand_joint_position")
        ]
        projected[:, 9:19] = torch.clamp(
            projected[:, 9:19], hand - 0.08, hand + 0.08
        )
        return projected

    def effective_action_torch(self) -> torch.Tensor:
        return self._actual

    def step(self, action: torch.Tensor) -> Stage0Transition:
        previous = self._state.clone()
        self.actions.append(action.detach().clone())
        self._actual = action.detach().clone()
        if self.drift_world is not None:
            self._actual[self.drift_world, 0] += 1.0e-4
        next_state = self._state.clone()
        next_state[:, DEFAULT_STAGE0_STATE_SCHEMA.slice("eef_position")] = action[
            :, :3
        ]
        next_state[
            :, DEFAULT_STAGE0_STATE_SCHEMA.slice("hand_joint_position")
        ] = action[:, 9:19]
        grasped = not self.never_grasp
        if grasped:
            next_state[
                :, DEFAULT_STAGE0_STATE_SCHEMA.slice("has_hand_contact")
            ] = 1.0
            next_state[:, DEFAULT_STAGE0_STATE_SCHEMA.slice("is_grasped")] = 1.0
            if self._step >= 1:
                next_state[
                    :, DEFAULT_STAGE0_STATE_SCHEMA.slice("object_position")
                ] = torch.tensor([0.0, 0.0, 0.04])
                next_state[
                    :, DEFAULT_STAGE0_STATE_SCHEMA.slice("object_to_goal")
                ] = torch.tensor([0.0, 0.0, 0.06])
        self._state = next_state
        contact = torch.full(
            (self.num_envs,), grasped, dtype=torch.bool
        )
        info_grasped = contact.clone()
        info_contact = contact.clone()
        if self.disagree_world is not None:
            info_grasped[self.disagree_world] = ~info_grasped[self.disagree_world]
        forbidden = torch.zeros(self.num_envs, dtype=torch.bool)
        if self.forbidden_world is not None and self._step == 0:
            forbidden[self.forbidden_world] = True
        overflow_count = torch.tensor(
            [int(self.overflow and self._step == 0)], dtype=torch.int64
        )
        reward = torch.zeros(self.num_envs, dtype=torch.float32)
        if self.nonzero_reward_world is not None:
            reward[self.nonzero_reward_world] = 1.0
        flags = torch.zeros(self.num_envs, dtype=torch.bool)
        self._step += 1
        return Stage0Transition(
            state=previous,
            action=action,
            next_state=next_state.clone(),
            reward=reward,
            terminated=flags.clone(),
            truncated=flags.clone(),
            info={
                "control_step_diagnostics": {
                    "rigid_contact_overflow_frame_count": overflow_count,
                    "triangle_pair_overflow_frame_count": torch.zeros_like(
                        overflow_count
                    ),
                },
                "forbidden_hand_contact_any_frame_this_control_step": forbidden,
                "had_hand_contact_this_control_step": contact,
                "has_hand_contact": info_contact,
                "is_grasped": info_grasped,
                "opposed_grasp_max_consecutive_physics_frames_this_control_step": (
                    torch.full(
                        (self.num_envs,),
                        6 if grasped else 0,
                        dtype=torch.int64,
                    )
                ),
            },
        )


class _FakePolicy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.state_dim = DEFAULT_STAGE0_STATE_SCHEMA.dimension
        self.latent_dim = 32
        self.action_dim = 19
        self.action_horizon = 2
        self.anchor = nn.Parameter(torch.zeros(()))
        self.states: list[torch.Tensor] = []
        self.noises: list[torch.Tensor] = []
        self.latents: list[torch.Tensor | None] = []

    def sample(
        self,
        current_state: torch.Tensor,
        *,
        success_latent: torch.Tensor | None,
        steps: int,
        action_feature_mask: torch.Tensor,
        initial_noise: torch.Tensor,
    ) -> torch.Tensor:
        self.states.append(current_state.detach().clone())
        self.noises.append(initial_noise.detach().clone())
        self.latents.append(success_latent)
        self.steps = steps
        self.mask = action_feature_mask.detach().clone()
        result = torch.zeros(
            current_state.shape[0],
            self.action_horizon,
            self.action_dim,
            dtype=current_state.dtype,
            device=current_state.device,
        )
        result[:, 0, 2] = 0.01
        result[:, 0, 9:19] = 0.5
        result[:, 1, :] = 1_000.0
        return result


def _normalization() -> Stage0Normalization:
    mask = torch.tensor(
        (True, True, True) + (False,) * 6 + (True,) * 10,
        dtype=torch.bool,
    )
    return Stage0Normalization(
        state_mean=torch.zeros(
            DEFAULT_STAGE0_STATE_SCHEMA.dimension, dtype=torch.float32
        ),
        state_std=torch.ones(
            DEFAULT_STAGE0_STATE_SCHEMA.dimension, dtype=torch.float32
        ),
        action_mean=torch.zeros(19, dtype=torch.float32),
        action_std=torch.ones(19, dtype=torch.float32),
        effective_action_mask=mask,
        state_sample_count=10,
        action_sample_count=10,
    )


def _budget(*, steps: int = 5) -> GraspLiftRolloutBudget:
    return GraspLiftRolloutBudget.from_reset_groups(
        (("reset-7009", 7009), ("reset-7014", 7014)),
        max_steps_per_episode=steps,
    )


class GraspLiftRolloutTest(unittest.TestCase):
    def test_budget_is_a_deterministic_reset_major_four_noise_grid(self) -> None:
        left = _budget()
        right = _budget()

        self.assertEqual(left, right)
        self.assertEqual(left.sha256, right.sha256)
        self.assertEqual(left.episode_count, 8)
        self.assertEqual(left.reset_count, 2)
        self.assertEqual(
            left.noise_variants,
            tuple(range(GRASP_LIFT_FORMAL_NOISE_VARIANTS)) * 2,
        )
        self.assertEqual(len(set(left.noise_seeds)), left.episode_count)
        with self.assertRaisesRegex(ValueError, "unique"):
            GraspLiftRolloutBudget.from_reset_groups(
                (("reset-7009", 7009), ("reset-7009", 7009))
            )

    def test_receding_horizon_applies_state_view_hybrid_decode_and_projection(
        self,
    ) -> None:
        budget = _budget()
        env = _FakeGraspLiftEnv(budget.episode_count)
        policy = _FakePolicy()
        result = rollout_grasp_lift_no_z_policy(
            env,
            policy,
            _normalization(),
            budget,
            flow_sample_steps=3,
        )

        self.assertEqual(
            result.outcomes,
            (Stage0Outcome.SUCCESS,) * budget.episode_count,
        )
        self.assertEqual(result.completed_steps.tolist(), [3] * budget.episode_count)
        self.assertEqual(env.reset_seeds, budget.reset_seeds)
        self.assertEqual(len(env.actions), 3)
        self.assertTrue(policy.training)
        self.assertEqual(policy.steps, 3)
        self.assertTrue(all(latent is None for latent in policy.latents))
        expected_phase = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0])
        phase_slice = DEFAULT_STAGE0_STATE_SCHEMA.slice("task_phase_one_hot")
        for state in policy.states:
            self.assertTrue(torch.equal(state[0, phase_slice], expected_phase))
        self.assertTrue(policy.mask[:3].all())
        self.assertFalse(policy.mask[3:9].any())
        self.assertTrue(policy.mask[9:19].all())
        for step, action in enumerate(env.actions, start=1):
            torch.testing.assert_close(
                action[:, 2],
                torch.full((budget.episode_count,), 0.01 * step),
                rtol=0.0,
                atol=1.0e-7,
            )
            torch.testing.assert_close(
                action[:, 9:19],
                torch.full((budget.episode_count, 10), 0.08 * step),
                rtol=0.0,
                atol=1.0e-7,
            )
        self.assertEqual(result.to_record()["safety_violation_count"], 0)

    def test_noise_is_paired_across_repeated_model_evaluations(self) -> None:
        budget = _budget()
        policies = (_FakePolicy(), _FakePolicy())
        for policy in policies:
            rollout_grasp_lift_no_z_policy(
                _FakeGraspLiftEnv(budget.episode_count),
                policy,
                _normalization(),
                budget,
            )
        self.assertEqual(len(policies[0].noises), len(policies[1].noises))
        for left, right in zip(policies[0].noises, policies[1].noises, strict=True):
            torch.testing.assert_close(left, right, rtol=0.0, atol=0.0)

    def test_timeout_is_explicit_when_grasp_never_occurs(self) -> None:
        budget = _budget(steps=2)
        result = rollout_grasp_lift_no_z_policy(
            _FakeGraspLiftEnv(budget.episode_count, never_grasp=True),
            _FakePolicy(),
            _normalization(),
            budget,
        )
        self.assertEqual(
            result.outcomes,
            (Stage0Outcome.TIMEOUT,) * budget.episode_count,
        )
        self.assertEqual(result.completed_steps.tolist(), [2] * budget.episode_count)

    def test_safety_oracle_failure_is_latched_per_world(self) -> None:
        budget = _budget()
        result = rollout_grasp_lift_no_z_policy(
            _FakeGraspLiftEnv(budget.episode_count, forbidden_world=0),
            _FakePolicy(),
            _normalization(),
            budget,
        )
        self.assertEqual(result.outcomes[0], Stage0Outcome.FAILURE)
        self.assertTrue(result.forbidden_contact_violation[0])
        self.assertTrue(all(not bool(value) for value in result.success[:1]))
        self.assertTrue(all(bool(value) for value in result.success[1:]))

    def test_global_collision_overflow_invalidates_the_vector_batch(self) -> None:
        budget = _budget()
        result = rollout_grasp_lift_no_z_policy(
            _FakeGraspLiftEnv(budget.episode_count, overflow=True),
            _FakePolicy(),
            _normalization(),
            budget,
        )
        self.assertTrue(bool(result.failure.all()))
        self.assertTrue(bool(result.collision_buffer_overflow_violation.all()))
        self.assertEqual(result.completed_steps.tolist(), [1] * budget.episode_count)

    def test_oracle_input_and_execution_faults_fail_only_affected_cells(self) -> None:
        budget = _budget()
        cases = (
            ("disagree_world", "oracle_input_disagreement"),
            ("drift_world", "execution_integrity_violation"),
            ("nonzero_reward_world", "execution_integrity_violation"),
        )
        for argument, field in cases:
            with self.subTest(argument=argument):
                env = _FakeGraspLiftEnv(
                    budget.episode_count,
                    **{argument: 1},
                )
                result = rollout_grasp_lift_no_z_policy(
                    env,
                    _FakePolicy(),
                    _normalization(),
                    budget,
                )
                self.assertEqual(result.outcomes[1], Stage0Outcome.FAILURE)
                self.assertTrue(getattr(result, field)[1])
                self.assertTrue(result.success[0])
                self.assertTrue(bool(result.success[2:].all()))


if __name__ == "__main__":
    unittest.main()
