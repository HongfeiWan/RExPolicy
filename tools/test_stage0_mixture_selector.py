"""CPU checks for the Stage 0 multimodal success selector."""

from __future__ import annotations

import math
import unittest

import torch

from rexpolicy.stage0.models.mixture_selector import (
    MixtureSuccessModeSelector,
)
from rexpolicy.stage0.trainers import Stage0SelectorTrainer


class Stage0MixtureSelectorTest(unittest.TestCase):
    def test_shapes_sampling_determinism_and_finite_gradients(self) -> None:
        torch.manual_seed(11)
        selector = MixtureSuccessModeSelector(
            7,
            3,
            components=4,
            hidden_dim=16,
            layers=2,
            minimum_log_std=-3.0,
            maximum_log_std=1.0,
        )
        states = torch.randn(5, 7)
        targets = torch.randn(5, 3)
        distribution = selector(states)

        self.assertEqual(tuple(distribution.logits.shape), (5, 4))
        self.assertEqual(tuple(distribution.component_means.shape), (5, 4, 3))
        self.assertEqual(tuple(distribution.component_log_stds.shape), (5, 4, 3))
        self.assertEqual(tuple(distribution.mean.shape), (5, 3))
        self.assertEqual(tuple(distribution.mode.shape), (5, 3))
        torch.testing.assert_close(
            distribution.weights.sum(dim=-1),
            torch.ones(5),
        )
        self.assertLessEqual(float(distribution.logits.max()), 0.0)
        self.assertGreaterEqual(float(distribution.component_log_stds.min()), -3.0)
        self.assertLessEqual(float(distribution.component_log_stds.max()), 1.0)

        loss = distribution.negative_log_likelihood(targets)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        for name, parameter in selector.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)

        first = selector.sample(
            states,
            sample_shape=(3,),
            generator=torch.Generator().manual_seed(13),
        )
        second = selector.sample(
            states,
            sample_shape=(3,),
            generator=torch.Generator().manual_seed(13),
        )
        self.assertEqual(tuple(first.shape), (3, 5, 3))
        torch.testing.assert_close(first, second)
        self.assertFalse(first.requires_grad)
        self.assertTrue(
            selector(states)
            .rsample(generator=torch.Generator().manual_seed(17))
            .requires_grad
        )
        torch.testing.assert_close(
            selector.sample(states, deterministic=True),
            selector.predict(states),
        )

    def test_existing_selector_trainer_contract(self) -> None:
        selector = MixtureSuccessModeSelector(
            5,
            2,
            components=3,
            hidden_dim=12,
            layers=1,
        )
        trainer = Stage0SelectorTrainer(
            selector,
            torch.optim.AdamW(selector.parameters(), lr=1.0e-3),
        )
        metrics = trainer.step(torch.randn(8, 5), torch.randn(8, 2))
        self.assertEqual(metrics.optimizer_step, 1)
        self.assertTrue(math.isfinite(metrics.negative_log_likelihood))
        self.assertGreater(metrics.gradient_norm, 0.0)

    def test_same_state_four_mode_overfit(self) -> None:
        torch.manual_seed(23)
        modes = torch.tensor(
            [
                [-2.0, -2.0],
                [-2.0, 2.0],
                [2.0, -2.0],
                [2.0, 2.0],
            ]
        )
        states = torch.zeros(64, 3)
        targets = modes.repeat_interleave(16, dim=0)
        selector = MixtureSuccessModeSelector(
            3,
            2,
            components=4,
            hidden_dim=24,
            layers=1,
            minimum_log_std=-3.0,
            maximum_log_std=1.5,
        )
        # Mixture likelihood is permutation symmetric. A small deterministic
        # warm start keeps this unit test focused on multimodal capacity and
        # NLL optimization instead of stochastic dead-component recovery.
        offsets = torch.tensor(
            [
                [0.45, -0.35],
                [-0.40, 0.50],
                [0.50, 0.40],
                [-0.35, -0.45],
            ]
        )
        with torch.no_grad():
            selector.mean_head.weight.zero_()
            selector.mean_head.bias.copy_((modes + offsets).flatten())
        optimizer = torch.optim.Adam(selector.parameters(), lr=2.0e-2)

        initial_nll = float(selector(states).negative_log_likelihood(targets))
        for _ in range(600):
            optimizer.zero_grad(set_to_none=True)
            loss = selector(states).negative_log_likelihood(targets)
            loss.backward()
            optimizer.step()

        distribution = selector(states[:1])
        distances = torch.cdist(modes, distribution.component_means[0])
        nearest_component = distances.argmin(dim=1)
        final_nll = float(selector(states).negative_log_likelihood(targets))
        self.assertLess(final_nll, initial_nll - 2.0)
        self.assertEqual(nearest_component.unique().numel(), 4)
        self.assertLess(float(distances.min(dim=1).values.max()), 0.25)
        self.assertGreater(float(distribution.weights.min()), 0.15)


if __name__ == "__main__":
    unittest.main()
