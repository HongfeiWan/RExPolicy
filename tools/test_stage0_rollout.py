"""CPU contract tests for learned-policy Stage 0 Reach rollouts."""

from __future__ import annotations

import unittest
from typing import Any

import torch
from torch import nn

from rexpolicy.stage0.envs.oracles import ReachSuccessOracle
from rexpolicy.stage0.envs.state_only import Stage0Transition
from rexpolicy.stage0.envs.state_schema import DEFAULT_STAGE0_STATE_SCHEMA
from rexpolicy.stage0.evaluation.rollout import (
    ReachRolloutBudget,
    RolloutConditioningMode,
    rollout_reach_policy,
    rollout_reach_policy_controls,
)
from rexpolicy.stage0.normalization import Stage0Normalization
from rexpolicy.stage0.runtime import Stage0FlowPolicy
from rexpolicy.stage0.types import Stage0Outcome


class _FakeReachEnv:
    def __init__(
        self,
        goals: torch.Tensor,
        *,
        contact_on_first_step: torch.Tensor | None = None,
        object_displacement_on_first_step: torch.Tensor | None = None,
    ) -> None:
        self.num_envs = int(goals.shape[0])
        self.goals = goals.to(dtype=torch.float32)
        self.contact_on_first_step = contact_on_first_step
        self.object_displacement_on_first_step = object_displacement_on_first_step
        self.actions: list[torch.Tensor] = []
        self.reset_seed_history: list[tuple[int, ...]] = []
        self._state = torch.empty(0)
        self._step = 0

    def _new_state(self) -> torch.Tensor:
        state = torch.zeros(
            self.num_envs,
            DEFAULT_STAGE0_STATE_SCHEMA.dimension,
            dtype=torch.float32,
        )
        state[:, DEFAULT_STAGE0_STATE_SCHEMA.slice("goal_position")] = self.goals
        state[:, DEFAULT_STAGE0_STATE_SCHEMA.slice("object_position")] = torch.tensor(
            [0.5, 0.0, 0.0]
        )
        return state

    def reset(self, *, seed: list[int]) -> tuple[torch.Tensor, dict[str, Any]]:
        self.reset_seed_history.append(tuple(seed))
        self.actions = []
        self._step = 0
        self._state = self._new_state()
        return self._state.clone(), {}

    def project_effective_action(self, action: torch.Tensor) -> torch.Tensor:
        projected = action.clone()
        projected[:, 3:] = 7.0
        return projected

    def step(self, action: torch.Tensor) -> Stage0Transition:
        old_state = self._state.clone()
        self.actions.append(action.detach().clone())
        next_state = self._state.clone()
        next_state[:, DEFAULT_STAGE0_STATE_SCHEMA.slice("eef_position")] = action[:, :3]
        if self._step == 0 and self.contact_on_first_step is not None:
            next_state[:, DEFAULT_STAGE0_STATE_SCHEMA.slice("has_hand_contact")] = (
                self.contact_on_first_step[:, None].to(torch.float32)
            )
        if self._step == 0 and self.object_displacement_on_first_step is not None:
            next_state[:, DEFAULT_STAGE0_STATE_SCHEMA.slice("object_position")] += (
                self.object_displacement_on_first_step
            )
        self._state = next_state
        self._step += 1
        flags = torch.zeros(self.num_envs, dtype=torch.bool)
        return Stage0Transition(
            state=old_state,
            action=action,
            next_state=next_state.clone(),
            reward=torch.zeros(self.num_envs),
            terminated=flags.clone(),
            truncated=flags.clone(),
            info={},
        )


class _FakePolicy(nn.Module):
    def __init__(self, *, normalized_delta: float = 2.0) -> None:
        super().__init__()
        self.state_dim = DEFAULT_STAGE0_STATE_SCHEMA.dimension
        self.latent_dim = 2
        self.action_dim = 19
        self.action_horizon = 2
        self.anchor = nn.Parameter(torch.zeros(()))
        self.normalized_delta = normalized_delta
        self.states: list[torch.Tensor] = []
        self.latents: list[torch.Tensor | None] = []
        self.noises: list[torch.Tensor] = []

    def sample(
        self,
        current_state: torch.Tensor,
        *,
        success_latent: torch.Tensor | None,
        steps: int,
        action_feature_mask: torch.Tensor,
        initial_noise: torch.Tensor,
    ) -> torch.Tensor:
        del steps
        self.states.append(current_state.detach().clone())
        self.latents.append(
            None if success_latent is None else success_latent.detach().clone()
        )
        self.noises.append(initial_noise.detach().clone())
        self.assert_effective_mask = action_feature_mask.detach().clone()
        result = torch.zeros(
            current_state.shape[0],
            self.action_horizon,
            self.action_dim,
            dtype=current_state.dtype,
            device=current_state.device,
        )
        result[:, 0, 0] = self.normalized_delta
        # This deliberately dangerous second action must never be executed by
        # the receding-horizon controller.
        result[:, 1, :3] = 1_000.0
        return result


class _FakeSelector(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.calls = 0

    def predict(self, state: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        row = torch.arange(1, state.shape[0] + 1, dtype=state.dtype)[:, None]
        return torch.cat((row, row * 10.0), dim=1)


def _normalization() -> Stage0Normalization:
    state_dim = DEFAULT_STAGE0_STATE_SCHEMA.dimension
    action_mean = torch.zeros(19, dtype=torch.float32)
    action_std = torch.ones(19, dtype=torch.float32)
    action_std[:3] = 0.01
    effective = torch.zeros(19, dtype=torch.bool)
    effective[:3] = True
    return Stage0Normalization(
        state_mean=torch.ones(state_dim, dtype=torch.float32),
        state_std=torch.ones(state_dim, dtype=torch.float32),
        action_mean=action_mean,
        action_std=action_std,
        effective_action_mask=effective,
        state_sample_count=10,
        action_sample_count=10,
    )


def _budget(max_steps: int = 4) -> ReachRolloutBudget:
    return ReachRolloutBudget(
        reset_group_ids=("reset-a", "reset-b"),
        reset_seeds=(101, 202),
        noise_seeds=(303, 404),
        max_steps_per_episode=max_steps,
    )


def _oracle(worlds: int) -> ReachSuccessOracle:
    return ReachSuccessOracle(
        worlds,
        success_threshold_m=0.001,
        success_hold_steps=1,
        displacement_limit_m=0.01,
    )


class Stage0ReachRolloutTests(unittest.TestCase):
    def test_four_controls_share_one_vector_environment_without_semantic_drift(
        self,
    ) -> None:
        goals = torch.tensor([[0.20, 0.0, 0.0], [0.20, 0.0, 0.0]])
        oracle_latents = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        env = _FakeReachEnv(goals.repeat(4, 1))
        no_z_policy = _FakePolicy()
        conditional_policy = _FakePolicy()
        selector = _FakeSelector()
        no_z_policy.eval()
        selector.eval()

        batched = rollout_reach_policy_controls(
            env,
            no_z_policy,
            conditional_policy,
            _normalization(),
            _budget(max_steps=2),
            oracle=_oracle(8),
            oracle_latents=oracle_latents,
            selector=selector,
            latent_permutation=(1, 0),
        )

        sequential: dict[str, Any] = {}
        for mode in RolloutConditioningMode:
            kwargs: dict[str, Any] = {}
            if mode is RolloutConditioningMode.ORACLE_Z:
                kwargs["oracle_latents"] = oracle_latents
            elif mode is RolloutConditioningMode.SELECTOR_Z:
                kwargs["selector"] = _FakeSelector()
            elif mode is RolloutConditioningMode.PERMUTED_Z:
                kwargs["oracle_latents"] = oracle_latents
                kwargs["latent_permutation"] = (1, 0)
            sequential[mode.value] = rollout_reach_policy(
                _FakeReachEnv(goals),
                _FakePolicy(),
                _normalization(),
                _budget(max_steps=2),
                mode=mode,
                oracle=_oracle(2),
                **kwargs,
            )

        self.assertEqual(
            env.reset_seed_history,
            [(101, 202, 101, 202, 101, 202, 101, 202)],
        )
        self.assertEqual(len(env.actions), 2)
        self.assertEqual(selector.calls, 1)
        self.assertFalse(no_z_policy.training)
        self.assertTrue(conditional_policy.training)
        self.assertFalse(selector.training)
        for mode in RolloutConditioningMode:
            self.assertEqual(
                batched[mode.value].to_record(),
                sequential[mode.value].to_record(),
            )
        for step, no_z_noise in enumerate(no_z_policy.noises):
            conditional_noise = conditional_policy.noises[step]
            for block in conditional_noise.split(2, dim=0):
                torch.testing.assert_close(no_z_noise, block, rtol=0.0, atol=0.0)

    def test_real_flow_policy_is_signature_compatible(self) -> None:
        torch.manual_seed(13)
        env = _FakeReachEnv(torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]))
        policy = Stage0FlowPolicy(
            state_dim=DEFAULT_STAGE0_STATE_SCHEMA.dimension,
            latent_dim=2,
            action_dim=19,
            action_horizon=2,
            model_dim=16,
            transformer_layers=1,
            attention_heads=4,
            feedforward_dim=32,
        )
        result = rollout_reach_policy(
            env,
            policy,
            _normalization(),
            _budget(max_steps=1),
            mode="no-z",
            oracle=_oracle(2),
            flow_sample_steps=2,
        )
        self.assertEqual(
            result.outcomes,
            (Stage0Outcome.TIMEOUT, Stage0Outcome.TIMEOUT),
        )
        displacement = torch.linalg.vector_norm(env.actions[0][:, :3], dim=-1)
        self.assertTrue(bool((displacement <= 0.030001).all()))
        self.assertTrue(policy.training)

    def test_receding_horizon_normalizes_clamps_projects_and_masks(self) -> None:
        env = _FakeReachEnv(torch.tensor([[0.06, 0.0, 0.0], [0.09, 0.0, 0.0]]))
        policy = _FakePolicy(normalized_delta=10.0)
        result = rollout_reach_policy(
            env,
            policy,
            _normalization(),
            _budget(max_steps=3),
            mode=RolloutConditioningMode.NO_Z,
            oracle=_oracle(2),
            initial_active_mask=torch.tensor([True, False]),
            flow_sample_steps=2,
        )

        self.assertEqual(env.reset_seed_history, [(101, 202)])
        self.assertEqual(result.outcomes, (Stage0Outcome.SUCCESS, None))
        self.assertEqual(result.completed_steps.tolist(), [2, 0])
        self.assertEqual(result.trace_valid_mask.sum(dim=1).tolist(), [3, 0])
        self.assertAlmostEqual(float(env.actions[0][0, 0]), 0.03, places=6)
        self.assertAlmostEqual(float(env.actions[1][0, 0]), 0.06, places=6)
        self.assertTrue(torch.equal(env.actions[0][:, 3:], torch.full((2, 16), 7.0)))
        # The inactive world receives an EEF hold, not the sampled delta.
        self.assertTrue(torch.equal(env.actions[0][1, :3], torch.zeros(3)))
        # Raw zero reset state is normalized with mean one before policy input.
        eef_slice = DEFAULT_STAGE0_STATE_SCHEMA.slice("eef_position")
        self.assertTrue(torch.equal(policy.states[0][0, eef_slice], -torch.ones(3)))
        self.assertTrue(policy.assert_effective_mask[:3].all())
        self.assertFalse(policy.assert_effective_mask[3:].any())
        self.assertTrue(all(latent is not None for latent in policy.latents))
        for latent in policy.latents:
            torch.testing.assert_close(latent, torch.zeros_like(latent))
        self.assertIsNone(result.episode_latents)
        self.assertTrue(policy.training)

    def test_selector_latent_is_chosen_once_and_fixed_for_episode(self) -> None:
        env = _FakeReachEnv(torch.tensor([[0.04, 0.0, 0.0], [0.06, 0.0, 0.0]]))
        policy = _FakePolicy()
        selector = _FakeSelector()
        result = rollout_reach_policy(
            env,
            policy,
            _normalization(),
            _budget(max_steps=3),
            mode="selector-z",
            oracle=_oracle(2),
            selector=selector,
        )

        expected = torch.tensor([[1.0, 10.0], [2.0, 20.0]])
        self.assertEqual(selector.calls, 1)
        self.assertGreater(len(policy.latents), 1)
        for latent in policy.latents:
            torch.testing.assert_close(latent, expected)
        torch.testing.assert_close(result.episode_latents, expected)
        self.assertTrue(selector.training)

    def test_all_modes_share_step_noise_and_permuted_z_is_deranged(self) -> None:
        goals = torch.tensor([[0.04, 0.0, 0.0], [0.04, 0.0, 0.0]])
        no_z_policy = _FakePolicy()
        oracle_policy = _FakePolicy()
        permuted_policy = _FakePolicy()
        oracle_latents = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

        rollout_reach_policy(
            _FakeReachEnv(goals),
            no_z_policy,
            _normalization(),
            _budget(max_steps=2),
            mode="no-z",
            oracle=_oracle(2),
        )
        rollout_reach_policy(
            _FakeReachEnv(goals),
            oracle_policy,
            _normalization(),
            _budget(max_steps=2),
            mode="oracle-z",
            oracle=_oracle(2),
            oracle_latents=oracle_latents,
        )
        permuted = rollout_reach_policy(
            _FakeReachEnv(goals),
            permuted_policy,
            _normalization(),
            _budget(max_steps=2),
            mode="permuted-z",
            oracle=_oracle(2),
            oracle_latents=oracle_latents,
        )

        for left, right in zip(no_z_policy.noises, oracle_policy.noises):
            torch.testing.assert_close(left, right, rtol=0.0, atol=0.0)
        self.assertEqual(permuted.latent_permutation, (1, 0))
        expected = torch.tensor([[3.0, 4.0], [1.0, 2.0]])
        torch.testing.assert_close(permuted.episode_latents, expected)
        for latent in permuted_policy.latents:
            torch.testing.assert_close(latent, expected)

    def test_contact_and_displacement_are_failure_witnesses(self) -> None:
        env = _FakeReachEnv(
            torch.tensor([[0.20, 0.0, 0.0], [0.20, 0.0, 0.0]]),
            contact_on_first_step=torch.tensor([True, False]),
            object_displacement_on_first_step=torch.tensor(
                [[0.0, 0.0, 0.0], [0.02, 0.0, 0.0]]
            ),
        )
        result = rollout_reach_policy(
            env,
            _FakePolicy(),
            _normalization(),
            _budget(max_steps=4),
            mode="no-z",
            oracle=_oracle(2),
        )

        self.assertEqual(
            result.outcomes,
            (Stage0Outcome.FAILURE, Stage0Outcome.FAILURE),
        )
        self.assertEqual(result.completed_steps.tolist(), [1, 1])
        self.assertEqual(result.contact_violation.tolist(), [True, False])
        self.assertAlmostEqual(
            float(result.maximum_object_displacement_m[1]),
            0.02,
            places=6,
        )
        paths = result.eef_paths()
        self.assertEqual(paths[0].shape, (2, 3))
        self.assertEqual(paths[1].shape, (2, 3))
        record = result.to_record()
        self.assertEqual(record["trace_lengths"], [2, 2])
        self.assertIn("eef_trace_m", record)

    def test_timeout_and_budget_fingerprint_are_explicit(self) -> None:
        result = rollout_reach_policy(
            _FakeReachEnv(torch.tensor([[0.20, 0.0, 0.0], [0.20, 0.0, 0.0]])),
            _FakePolicy(),
            _normalization(),
            _budget(max_steps=2),
            mode="no-z",
            oracle=_oracle(2),
        )
        self.assertEqual(
            result.outcomes,
            (Stage0Outcome.TIMEOUT, Stage0Outcome.TIMEOUT),
        )
        self.assertEqual(result.executed_world_steps, 4)
        self.assertEqual(result.to_record()["budget_sha256"], _budget(2).fingerprint)

        with self.assertRaisesRegex(ValueError, "align"):
            ReachRolloutBudget(
                reset_group_ids=("a",),
                reset_seeds=(1, 2),
                noise_seeds=(3,),
                max_steps_per_episode=2,
            )
        with self.assertRaisesRegex(ValueError, "at least two"):
            rollout_reach_policy(
                _FakeReachEnv(torch.tensor([[0.1, 0.0, 0.0]])),
                _FakePolicy(),
                _normalization(),
                ReachRolloutBudget(("a",), (1,), (2,), 1),
                mode="permuted-z",
                oracle=_oracle(1),
                oracle_latents=torch.ones(1, 2),
            )


if __name__ == "__main__":
    unittest.main()
