"""Focused tests for the independent Stage 0 training layer."""

from __future__ import annotations

import math
import subprocess
import sys
import unittest
from pathlib import Path

import torch

from rexpolicy.stage0.data.windows import Stage0SuccessWindow
from rexpolicy.stage0.models import FutureTrajectoryEncoder, SuccessModeSelector
from rexpolicy.stage0.runtime import Stage0FlowPolicy
from rexpolicy.stage0.trainers import (
    Stage0ManifoldTrainer,
    Stage0ManifoldTrainerConfig,
    Stage0PolicyTrainer,
    Stage0SelectorTrainer,
    collate_success_windows,
    temporal_group_contrastive_loss,
    variance_covariance_losses,
)


_DIGEST = "a" * 64
_POLICY_DIGEST = "b" * 64


def _window(
    trajectory_id: str,
    reset_group_id: str,
    start: int,
    *,
    valid_steps: int = 4,
) -> Stage0SuccessWindow:
    generator = torch.Generator().manual_seed(1000 + start + ord(trajectory_id[-1]))
    future_states = torch.zeros(4, 5)
    future_actions = torch.zeros(4, 3)
    future_states[:valid_steps] = torch.randn(
        valid_steps,
        5,
        generator=generator,
    )
    future_actions[:valid_steps] = torch.randn(
        valid_steps,
        3,
        generator=generator,
    )
    future_mask = torch.arange(4) < valid_steps
    return Stage0SuccessWindow(
        trajectory_id=trajectory_id,
        reset_group_id=reset_group_id,
        start=start,
        current_state=torch.randn(5, generator=generator),
        action_chunk=future_actions[:3].clone(),
        action_mask=future_mask[:3].clone(),
        future_actions=future_actions,
        future_states=future_states,
        future_mask=future_mask,
        trajectory_sha256=_DIGEST,
        corpus_sha256=_DIGEST,
        window_policy_sha256=_POLICY_DIGEST,
    )


def _batch():
    return collate_success_windows(
        (
            _window("trajectory-a", "reset-shared", 0),
            _window("trajectory-a", "reset-shared", 1, valid_steps=2),
            _window("trajectory-b", "reset-shared", 0),
            _window("trajectory-b", "reset-shared", 1),
            _window("trajectory-c", "reset-other", 0),
            _window("trajectory-c", "reset-other", 1),
        )
    )


def _gradient_norm(module: torch.nn.Module) -> float:
    squared = sum(
        float(parameter.grad.detach().square().sum())
        for parameter in module.parameters()
        if parameter.grad is not None
    )
    return math.sqrt(squared)


class RecordingEncoder(FutureTrajectoryEncoder):
    last_padding_mask: torch.Tensor | None = None

    def forward(self, *args, padding_mask=None, **kwargs):
        self.last_padding_mask = padding_mask.detach().clone()
        return super().forward(*args, padding_mask=padding_mask, **kwargs)


class Stage0BatchAndLossTest(unittest.TestCase):
    def test_import_has_no_vla_dependencies(self) -> None:
        repository = Path(__file__).resolve().parents[1]
        script = (
            "import sys; "
            f"sys.path.insert(0, {str(repository)!r}); "
            "import rexpolicy.stage0.trainers; "
            "forbidden={'gr00t','transformers','diffusers'}; "
            "loaded={name.split('.')[0] for name in sys.modules}; "
            "assert not forbidden.intersection(loaded)"
        )
        subprocess.run(
            [sys.executable, "-c", script],
            cwd=repository,
            check=True,
        )

    def test_collate_preserves_valid_masks_and_inverts_only_for_padding(self) -> None:
        batch = _batch()
        self.assertEqual(tuple(batch.current_states.shape), (6, 5))
        self.assertEqual(tuple(batch.action_chunks.shape), (6, 3, 3))
        self.assertEqual(tuple(batch.future_states.shape), (6, 4, 5))
        self.assertTrue(torch.equal(batch.future_padding_mask, ~batch.future_mask))
        self.assertTrue(batch.action_mask[1].tolist() == [True, True, False])
        self.assertTrue(batch.future_mask[1].tolist() == [True, True, False, False])
        with self.assertRaisesRegex(ValueError, "empty"):
            collate_success_windows(())

    def test_temporal_group_loss_rejects_degenerate_batches(self) -> None:
        embeddings = torch.randn(6, 4, requires_grad=True)
        loss = temporal_group_contrastive_loss(
            embeddings,
            ("a", "a", "b", "b", "c", "c"),
            ("shared", "shared", "shared", "shared", "other", "other"),
            (0, 1, 0, 1, 0, 1),
            temporal_radius=1,
            temperature=0.2,
        )
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertGreater(float(embeddings.grad.abs().sum()), 0.0)
        with self.assertRaisesRegex(ValueError, "temporal positive"):
            temporal_group_contrastive_loss(
                torch.randn(3, 4),
                ("a", "b", "c"),
                ("a", "b", "c"),
                (0, 0, 0),
                temporal_radius=1,
                temperature=0.2,
            )

    def test_variance_covariance_penalizes_collapse(self) -> None:
        collapsed = variance_covariance_losses(torch.zeros(6, 4))
        varied = variance_covariance_losses(torch.randn(64, 4) * 2.0)
        self.assertGreater(float(collapsed.variance), float(varied.variance))
        self.assertTrue(torch.isfinite(collapsed.covariance))


class Stage0TrainerStepTest(unittest.TestCase):
    def test_manifold_step_converts_mask_and_clips_all_gradients(self) -> None:
        torch.manual_seed(17)
        batch = _batch()
        model = RecordingEncoder(
            5,
            3,
            latent_dim=4,
            projection_dim=4,
            model_dim=16,
            max_future_steps=4,
            transformer_layers=1,
            attention_heads=4,
            feedforward_dim=24,
            dropout=0.0,
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3)
        trainer = Stage0ManifoldTrainer(
            model,
            optimizer,
            config=Stage0ManifoldTrainerConfig(
                reconstruction_probability=0.5,
                temporal_radius=1,
                gradient_clip_norm=0.05,
            ),
        )
        metrics = trainer.step(
            batch,
            generator=torch.Generator().manual_seed(19),
        )
        self.assertEqual(metrics.optimizer_step, 1)
        self.assertTrue(math.isfinite(metrics.loss))
        self.assertGreater(metrics.gradient_norm, 0.0)
        self.assertLessEqual(_gradient_norm(model), 0.050001)
        self.assertTrue(torch.equal(model.last_padding_mask, ~batch.future_mask))
        for name, parameter in model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)

    def test_selector_nll_step_detaches_targets_and_clips(self) -> None:
        torch.manual_seed(23)
        model = SuccessModeSelector(5, 4, hidden_dim=12, layers=2)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3)
        trainer = Stage0SelectorTrainer(
            model,
            optimizer,
            gradient_clip_norm=0.04,
        )
        states = torch.randn(8, 5)
        targets = torch.randn(8, 4, requires_grad=True)
        metrics = trainer.step(states, targets)
        self.assertEqual(metrics.loss, metrics.negative_log_likelihood)
        self.assertEqual(metrics.optimizer_step, 1)
        self.assertIsNone(targets.grad)
        self.assertGreater(metrics.gradient_norm, 0.0)
        self.assertLessEqual(_gradient_norm(model), 0.040001)

    def test_policy_flow_matching_step_uses_true_valid_action_mask(self) -> None:
        torch.manual_seed(29)
        batch = _batch()
        model = Stage0FlowPolicy(
            state_dim=5,
            latent_dim=4,
            action_dim=3,
            action_horizon=3,
            model_dim=16,
            transformer_layers=1,
            attention_heads=4,
            feedforward_dim=24,
            dropout=0.0,
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3)
        trainer = Stage0PolicyTrainer(
            model,
            optimizer,
            gradient_clip_norm=0.03,
        )
        latents = torch.randn(6, 4, requires_grad=True)
        metrics = trainer.step(
            batch,
            latents,
            generator=torch.Generator().manual_seed(31),
        )
        self.assertEqual(metrics.loss, metrics.flow_matching)
        self.assertEqual(metrics.optimizer_step, 1)
        self.assertIsNone(latents.grad)
        self.assertGreater(metrics.gradient_norm, 0.0)
        self.assertLessEqual(_gradient_norm(model), 0.030001)
        for name, parameter in model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)


if __name__ == "__main__":
    unittest.main()
