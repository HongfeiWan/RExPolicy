"""Focused tests for Stage 0 generic conditioning and Flow-DiT."""

from __future__ import annotations

import unittest

import torch

from rexpolicy.stage0.models.conditioning import (
    ConditionTokens,
    StateLatentConditionEncoder,
)
from rexpolicy.stage0.models.flow_dit import FlowDiT


def _models() -> tuple[StateLatentConditionEncoder, FlowDiT]:
    conditioner = StateLatentConditionEncoder(
        state_dim=5,
        latent_dim=3,
        model_dim=16,
        hidden_dim=12,
    )
    policy = FlowDiT(
        action_dim=4,
        horizon=3,
        model_dim=16,
        transformer_layers=2,
        attention_heads=4,
        feedforward_dim=32,
        dropout=0.0,
    )
    return conditioner, policy


class Stage0ConditionTokensTest(unittest.TestCase):
    def test_no_z_and_success_z_contract(self) -> None:
        conditioner, _ = _models()
        states = torch.randn(4, 5)
        latents = torch.randn(4, 3)
        no_z = conditioner(states)
        with_z = conditioner(states, latents)
        self.assertEqual(tuple(no_z.tokens.shape), (4, 1, 16))
        self.assertEqual(tuple(with_z.tokens.shape), (4, 2, 16))
        self.assertTrue(no_z.attention_mask.all())
        self.assertTrue(with_z.attention_mask.all())
        with self.assertRaisesRegex(ValueError, "valid token"):
            ConditionTokens(
                tokens=torch.zeros(2, 1, 16),
                attention_mask=torch.zeros(2, 1, dtype=torch.bool),
            )
        with self.assertRaisesRegex(ValueError, "finite"):
            tokens = torch.zeros(2, 1, 16)
            tokens[0, 0, 0] = float("nan")
            ConditionTokens(
                tokens=tokens,
                attention_mask=torch.ones(2, 1, dtype=torch.bool),
            )


class Stage0FlowDiTTest(unittest.TestCase):
    def test_flow_equation_masks_and_condition_gradients(self) -> None:
        torch.manual_seed(31)
        conditioner, policy = _models()
        states = torch.randn(4, 5)
        latents = torch.randn(4, 3)
        conditions = conditioner(states, latents)
        actions = torch.randn(4, 3, 4)
        noise = torch.randn(4, 3, 4)
        timesteps = torch.tensor([0.0, 0.25, 0.5, 1.0])
        action_mask = torch.tensor(
            [
                [True, True, True],
                [True, True, False],
                [True, False, False],
                [True, True, True],
            ]
        )
        output = policy.flow_matching_loss(
            actions,
            conditions,
            action_mask=action_mask,
            noise=noise,
            timesteps=timesteps,
        )
        expected_noisy = (
            (1.0 - timesteps[:, None, None]) * noise
            + timesteps[:, None, None] * actions
        ).masked_fill(~action_mask.unsqueeze(-1), 0.0)
        expected_velocity = (actions - noise).masked_fill(
            ~action_mask.unsqueeze(-1),
            0.0,
        )
        torch.testing.assert_close(output.noisy_actions, expected_noisy)
        torch.testing.assert_close(output.target_velocity, expected_velocity)
        self.assertTrue(torch.isfinite(output.loss))
        self.assertTrue(
            torch.equal(
                output.predicted_velocity.masked_select((~action_mask).unsqueeze(-1)),
                torch.zeros((~action_mask).sum() * 4),
            )
        )
        output.loss.backward()
        for name in (
            "state_projector.0.weight",
            "latent_projector.0.weight",
        ):
            parameter = dict(conditioner.named_parameters())[name]
            self.assertIsNotNone(parameter.grad, name)
            self.assertGreater(float(parameter.grad.abs().sum()), 0.0, name)
        self.assertGreater(
            float(policy.action_projector.weight.grad.abs().sum()),
            0.0,
        )

    def test_generator_reproducibility_and_latent_intervention(self) -> None:
        torch.manual_seed(37)
        conditioner, policy = _models()
        policy.eval()
        states = torch.randn(3, 5)
        latent_a = torch.randn(3, 3)
        latent_b = latent_a.roll(1, dims=0)
        conditions_a = conditioner(states, latent_a)
        conditions_b = conditioner(states, latent_b)
        actions = torch.randn(3, 3, 4)
        first_batch = policy.flow_matching_loss(
            actions,
            conditions_a,
            generator=torch.Generator().manual_seed(41),
        )
        second_batch = policy.flow_matching_loss(
            actions,
            conditions_a,
            generator=torch.Generator().manual_seed(41),
        )
        torch.testing.assert_close(first_batch.noise, second_batch.noise)
        torch.testing.assert_close(first_batch.timesteps, second_batch.timesteps)
        torch.testing.assert_close(first_batch.loss, second_batch.loss)

        sample_a = policy.sample(
            conditions_a,
            steps=4,
            generator=torch.Generator().manual_seed(43),
        )
        sample_a_repeat = policy.sample(
            conditions_a,
            steps=4,
            generator=torch.Generator().manual_seed(43),
        )
        sample_b = policy.sample(
            conditions_b,
            steps=4,
            generator=torch.Generator().manual_seed(43),
        )
        torch.testing.assert_close(sample_a, sample_a_repeat)
        self.assertGreater(float((sample_a - sample_b).abs().max()), 1e-6)

    def test_strict_shape_finite_and_mask_validation(self) -> None:
        conditioner, policy = _models()
        conditions = conditioner(torch.zeros(2, 5), torch.zeros(2, 3))
        with self.assertRaisesRegex(ValueError, "horizon"):
            policy(torch.zeros(2, 2, 4), 0.5, conditions)
        with self.assertRaisesRegex(ValueError, "finite"):
            actions = torch.zeros(2, 3, 4)
            actions[0, 0, 0] = float("inf")
            policy(actions, 0.5, conditions)
        with self.assertRaisesRegex(ValueError, "valid step"):
            policy(
                torch.zeros(2, 3, 4),
                0.5,
                conditions,
                action_mask=torch.zeros(2, 3, dtype=torch.bool),
            )
        with self.assertRaisesRegex(ValueError, r"\[0, 1\]"):
            policy(torch.zeros(2, 3, 4), 1.1, conditions)


if __name__ == "__main__":
    unittest.main()
