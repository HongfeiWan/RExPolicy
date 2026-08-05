"""Tests for the v2.1 successor-model compatibility contract."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from rexpolicy.flywheel.distributed import DistributedContext
from rexpolicy.manifold.config import SuccessManifoldConfig
from rexpolicy.manifold.selector import SuccessModeSelector
from rexpolicy.manifold.successor import (
    SuccessorModel,
    SuccessorTrainingConfig,
    successor_losses,
)
from rexpolicy.manifold.successor_trainer import SuccessorDdpTrainer


def _manifold_config() -> SuccessManifoldConfig:
    return SuccessManifoldConfig(
        enabled=True,
        state_dim=3,
        action_dim=2,
        model_dim=8,
        latent_dim=2,
        condition_dim=4,
        projection_dim=4,
        max_future_steps=4,
        transformer_layers=1,
        attention_heads=2,
        feedforward_dim=16,
        dropout=0.0,
        selector_hidden_dim=8,
        selector_layers=1,
    )


class TestSuccessorModel(unittest.TestCase):
    def test_state_dict_is_bidirectionally_selector_compatible(self) -> None:
        torch.manual_seed(17)
        config = _manifold_config()
        selector = SuccessModeSelector(config)
        successor = SuccessorModel(config)
        successor.load_state_dict(selector.state_dict(), strict=True)
        states = torch.randn(5, config.state_dim)
        expected = selector(states)
        actual = successor(states)
        torch.testing.assert_close(actual.mean, expected.mean)
        torch.testing.assert_close(actual.log_std, expected.log_std)

        restored_selector = SuccessModeSelector(config)
        restored_selector.load_state_dict(successor.state_dict(), strict=True)
        torch.testing.assert_close(
            restored_selector(states).mean,
            successor(states).mean,
        )

    def test_losses_are_finite_and_calibration_has_gradient(self) -> None:
        torch.manual_seed(23)
        config = _manifold_config()
        model = SuccessorModel(config)
        states = torch.randn(8, config.state_dim)
        targets = torch.randn(8, config.latent_dim)
        losses = successor_losses(
            model(states),
            targets,
            config=SuccessorTrainingConfig(enabled=True),
        )
        self.assertTrue(torch.isfinite(losses.total))
        self.assertGreaterEqual(float(losses.calibration), 0.0)
        losses.total.backward()
        gradient = model.log_std_head.weight.grad
        self.assertIsNotNone(gradient)
        self.assertGreater(float(gradient.abs().sum()), 0.0)

    def test_training_config_is_strict_and_default_off(self) -> None:
        self.assertFalse(SuccessorTrainingConfig().enabled)
        with self.assertRaisesRegex(ValueError, "Unknown successor"):
            SuccessorTrainingConfig.from_mapping({"typo": 1})
        with self.assertRaisesRegex(ValueError, "positive"):
            SuccessorTrainingConfig(learning_rate=0.0)
        record = SuccessorTrainingConfig(enabled=True).to_record()
        self.assertEqual(
            SuccessorTrainingConfig.from_mapping(record).to_record(),
            record,
        )

    def test_standalone_trainer_checkpoint_resume_and_v2_export(self) -> None:
        from rexpolicy.manifold.encoder import FutureTrajectoryEncoder
        from rexpolicy.manifold.runtime import SuccessManifoldRuntime

        torch.manual_seed(31)
        config = _manifold_config()
        training_config = SuccessorTrainingConfig(enabled=True)
        context = DistributedContext(
            rank=0,
            local_rank=0,
            world_size=1,
            device=torch.device("cpu"),
            initialized=False,
        )
        trainer = SuccessorDdpTrainer(
            manifold_config=config,
            training_config=training_config,
            context=context,
        )
        metrics = trainer.train_batch(
            current_states=torch.randn(4, config.state_dim),
            target_latents=torch.randn(4, config.latent_dim),
        )
        self.assertEqual(metrics.optimizer_step, 1)
        self.assertEqual(metrics.samples_seen, 4)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "successor.pt"
            digest = trainer.save_checkpoint(checkpoint)
            restored = SuccessorDdpTrainer.load_checkpoint(
                checkpoint,
                expected_sha256=digest,
                context=context,
            )
            self.assertEqual(restored.step_count, 1)
            self.assertEqual(restored.samples_seen, 4)
            for expected, actual in zip(
                trainer.model.parameters(), restored.model.parameters()
            ):
                torch.testing.assert_close(expected, actual)

            bundle = Path(directory) / "manifold.pt"
            bundle_digest = restored.save_deployment_bundle(
                bundle,
                encoder=FutureTrajectoryEncoder(config),
            )
            runtime = SuccessManifoldRuntime.load(
                bundle,
                expected_sha256=bundle_digest,
                device="cpu",
            )
            self.assertEqual(runtime.config, config)

    def test_standalone_trainer_rejects_uneven_ddp_batches(self) -> None:
        config = _manifold_config()
        context = DistributedContext(
            rank=0,
            local_rank=0,
            world_size=2,
            device=torch.device("cpu"),
            initialized=False,
        )
        context.all_gather_objects = lambda _value: [4, 3]
        trainer = SuccessorDdpTrainer(
            manifold_config=config,
            training_config=SuccessorTrainingConfig(enabled=True),
            context=context,
        )
        with self.assertRaisesRegex(ValueError, "equal local batch"):
            trainer.train_batch(
                current_states=torch.randn(4, config.state_dim),
                target_latents=torch.randn(4, config.latent_dim),
            )


if __name__ == "__main__":
    unittest.main()
