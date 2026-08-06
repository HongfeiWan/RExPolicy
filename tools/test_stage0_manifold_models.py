"""Focused tests for independent Stage 0 manifold neural modules."""

from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path

import torch

from rexpolicy.stage0.models.future_encoder import (
    FutureTrajectoryEncoder,
    masked_reconstruction_loss,
)
from rexpolicy.stage0.models.selector import SuccessModeSelector


def _encoder() -> FutureTrajectoryEncoder:
    return FutureTrajectoryEncoder(
        7,
        4,
        latent_dim=6,
        projection_dim=5,
        model_dim=16,
        max_future_steps=5,
        transformer_layers=2,
        attention_heads=4,
        feedforward_dim=32,
        dropout=0.0,
    )


class Stage0FutureEncoderTest(unittest.TestCase):
    def test_import_has_no_vla_dependencies(self) -> None:
        repository = Path(__file__).resolve().parents[1]
        script = (
            "import sys; "
            f"sys.path.insert(0, {str(repository)!r}); "
            "import rexpolicy.stage0.models; "
            "forbidden={'gr00t','transformers','diffusers'}; "
            "loaded={name.split('.')[0] for name in sys.modules}; "
            "assert not forbidden.intersection(loaded)"
        )
        subprocess.run(
            [sys.executable, "-c", script],
            cwd=repository,
            check=True,
        )

    def test_transition_shapes_masks_and_all_branch_gradients(self) -> None:
        torch.manual_seed(13)
        model = _encoder()
        states = torch.randn(4, 5, 7)
        actions = torch.randn(4, 5, 4)
        padding = torch.tensor(
            [
                [False, False, False, False, False],
                [False, False, False, True, True],
                [False, False, False, False, True],
                [False, False, True, True, True],
            ]
        )
        reconstruction_mask = torch.tensor(
            [
                [True, False, True, False, False],
                [False, True, False, False, False],
                [True, False, False, True, False],
                [False, True, False, False, False],
            ]
        )
        output = model(
            states,
            actions,
            padding_mask=padding,
            reconstruction_mask=reconstruction_mask,
        )
        self.assertEqual(tuple(output.latent.shape), (4, 6))
        self.assertEqual(tuple(output.projection.shape), (4, 5))
        self.assertEqual(tuple(output.encoded_steps.shape), (4, 5, 16))
        self.assertEqual(tuple(output.reconstruction.states.shape), (4, 5, 7))
        self.assertEqual(tuple(output.reconstruction.actions.shape), (4, 5, 4))
        self.assertTrue(torch.isfinite(output.latent).all())
        self.assertTrue(
            torch.equal(
                output.encoded_steps.masked_select(padding.unsqueeze(-1)),
                torch.zeros(padding.sum() * 16),
            )
        )
        loss = masked_reconstruction_loss(output, states, actions)
        loss = loss + output.latent.square().mean()
        loss = loss + output.projection.square().mean()
        loss.backward()
        for parameter_name in (
            "state_projector.weight",
            "action_projector.weight",
            "latent_head.weight",
            "projection_head.1.weight",
            "decoder.3.weight",
        ):
            parameter = dict(model.named_parameters())[parameter_name]
            self.assertIsNotNone(parameter.grad, parameter_name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), parameter_name)
            self.assertGreater(float(parameter.grad.abs().sum()), 0.0)

    def test_mask_sampling_is_generator_reproducible_and_strict(self) -> None:
        model = _encoder()
        padding = torch.tensor(
            [
                [False, False, False, True, True],
                [False, False, False, False, False],
            ]
        )
        first = model.sample_reconstruction_mask(
            padding,
            generator=torch.Generator().manual_seed(71),
        )
        second = model.sample_reconstruction_mask(
            padding,
            generator=torch.Generator().manual_seed(71),
        )
        self.assertTrue(torch.equal(first, second))
        self.assertTrue(first.any(dim=1).all())
        self.assertFalse(bool((first & padding).any()))

        states = torch.randn(2, 5, 7)
        actions = torch.randn(2, 5, 4)
        with self.assertRaisesRegex(ValueError, "finite"):
            invalid = states.clone()
            invalid[0, 0, 0] = float("nan")
            model(invalid, actions)
        with self.assertRaisesRegex(ValueError, "valid step"):
            model(states, actions, padding_mask=torch.ones(2, 5, dtype=torch.bool))
        with self.assertRaisesRegex(ValueError, "cannot select padded"):
            model(
                states,
                actions,
                padding_mask=padding,
                reconstruction_mask=padding,
            )


class Stage0SelectorTest(unittest.TestCase):
    def test_distribution_loss_sampling_and_checkpoint(self) -> None:
        torch.manual_seed(19)
        selector = SuccessModeSelector(
            7,
            6,
            hidden_dim=12,
            layers=2,
        )
        state = torch.randn(5, 7)
        target = torch.randn(5, 6)
        distribution = selector(state)
        self.assertEqual(tuple(distribution.mean.shape), (5, 6))
        self.assertTrue(torch.isfinite(distribution.std).all())
        loss = distribution.negative_log_likelihood(target)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(
            sum(
                float(parameter.grad.abs().sum())
                for parameter in selector.parameters()
                if parameter.grad is not None
            ),
            0.0,
        )

        first = selector.sample(
            state,
            generator=torch.Generator().manual_seed(29),
        )
        second = selector.sample(
            state,
            generator=torch.Generator().manual_seed(29),
        )
        torch.testing.assert_close(first, second)
        self.assertFalse(first.requires_grad)
        torch.testing.assert_close(
            selector.sample(state, deterministic=True),
            selector.predict(state),
        )

        restored = SuccessModeSelector(7, 6, hidden_dim=12, layers=2)
        restored.load_state_dict(selector.state_dict())
        torch.testing.assert_close(restored.predict(state), selector.predict(state))

    def test_selector_rejects_non_finite_state_and_temperature(self) -> None:
        selector = SuccessModeSelector(7, 6)
        state = torch.zeros(2, 7)
        state[0, 0] = float("inf")
        with self.assertRaisesRegex(ValueError, "finite"):
            selector(state)
        with self.assertRaisesRegex(ValueError, "temperature"):
            selector.sample(torch.zeros(2, 7), temperature=0.0)


if __name__ == "__main__":
    unittest.main()
