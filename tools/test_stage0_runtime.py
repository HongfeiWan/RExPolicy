from __future__ import annotations

import unittest

import torch

from rexpolicy.stage0.runtime import Stage0FlowPolicy


class Stage0FlowPolicyTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(3)
        self.policy = Stage0FlowPolicy(
            state_dim=7,
            latent_dim=3,
            action_dim=2,
            action_horizon=4,
            model_dim=16,
            transformer_layers=1,
            attention_heads=4,
            feedforward_dim=32,
        )

    def test_flow_loss_reaches_every_projection(self) -> None:
        state = torch.randn(5, 7)
        latent = torch.randn(5, 3)
        actions = torch.randn(5, 4, 2)
        output = self.policy.flow_matching_loss(
            state,
            actions,
            success_latent=latent,
        )
        output.loss.backward()
        for name in (
            "condition_encoder.state_projector.0.weight",
            "condition_encoder.latent_projector.0.weight",
            "flow_dit.action_projector.weight",
            "flow_dit.velocity_head.1.weight",
        ):
            gradient = dict(self.policy.named_parameters())[name].grad
            self.assertIsNotNone(gradient, name)
            self.assertGreater(float(gradient.norm()), 0.0, name)

    def test_control_samples_share_noise_but_use_distinct_latents(self) -> None:
        state = torch.randn(2, 7)
        oracle = torch.randn(2, 3)
        selector = -oracle
        initial_noise = torch.randn(2, 4, 2)
        first = self.policy.sample_controls(
            state,
            oracle_latent=oracle,
            selector_latent=selector,
            steps=2,
            initial_noise=initial_noise,
        )
        second = self.policy.sample_controls(
            state,
            oracle_latent=oracle,
            selector_latent=selector,
            steps=2,
            initial_noise=initial_noise,
        )
        torch.testing.assert_close(first.oracle_z, second.oracle_z)
        self.assertFalse(torch.equal(first.no_z, first.oracle_z))
        self.assertFalse(torch.equal(first.oracle_z, first.selector_z))


if __name__ == "__main__":
    unittest.main()
