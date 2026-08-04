"""Tests for staged, resumable success-manifold training."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from rexpolicy.flywheel.distributed import DistributedContext
from rexpolicy.manifold.config import SuccessManifoldConfig
from rexpolicy.manifold.runtime import SuccessManifoldRuntime
from rexpolicy.manifold.trainer import SuccessManifoldDdpTrainer


def _config() -> SuccessManifoldConfig:
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
        reconstruction_mask_probability=0.5,
        selector_hidden_dim=8,
        selector_layers=1,
        memory_capacity=8,
        novelty_threshold=0.0,
    )


def _context() -> DistributedContext:
    return DistributedContext(
        rank=0,
        local_rank=0,
        world_size=1,
        device=torch.device("cpu"),
        initialized=False,
    )


class TestSuccessManifoldTrainer(unittest.TestCase):
    def test_staged_updates_resume_and_export(self) -> None:
        torch.manual_seed(7)
        config = _config()
        trainer = SuccessManifoldDdpTrainer(
            config=config,
            context=_context(),
            gradient_clip_norm=10.0,
        )
        future_states = torch.randn(4, 3, config.state_dim)
        future_actions = torch.randn(4, 3, config.action_dim)
        encoder_metrics = trainer.train_encoder_batch(
            future_states=future_states,
            future_actions=future_actions,
            trajectory_ids=("a", "a", "b", "b"),
            generator=torch.Generator().manual_seed(11),
        )
        self.assertEqual(encoder_metrics.stage, "encoder")
        self.assertGreater(encoder_metrics.gradient_norm, 0.0)

        with torch.inference_mode():
            targets = trainer.encoder(
                future_states,
                future_actions,
            ).latent
        selector_metrics = trainer.train_selector_batch(
            current_states=future_states[:, 0],
            target_latents=targets,
        )
        self.assertEqual(selector_metrics.stage, "selector")
        self.assertGreater(selector_metrics.gradient_norm, 0.0)

        state = trainer.state_dict()
        restored = SuccessManifoldDdpTrainer(
            config=config,
            context=_context(),
        )
        restored.load_state_dict(state)
        self.assertEqual(restored.encoder_step_count, 1)
        self.assertEqual(restored.selector_step_count, 1)
        for expected, actual in zip(
            trainer.selector.parameters(), restored.selector.parameters()
        ):
            torch.testing.assert_close(expected, actual)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "success-manifold.pt"
            digest = restored.save_deployment_bundle(path)
            runtime = SuccessManifoldRuntime.load(
                path,
                expected_sha256=digest,
                device="cpu",
            )
            self.assertEqual(runtime.config, config)


if __name__ == "__main__":
    unittest.main()
