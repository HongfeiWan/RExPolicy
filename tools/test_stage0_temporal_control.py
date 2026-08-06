"""CPU tests for fixed-start-latent temporal-control diagnostics."""

from __future__ import annotations

import json
import unittest
from dataclasses import FrozenInstanceError

import torch
from torch import nn

from rexpolicy.stage0.evaluation.temporal_control import (
    evaluate_fixed_start_latent_first_action_mse,
)
from rexpolicy.stage0.trainers import Stage0WindowBatch


class _SpyPolicy(nn.Module):
    def __init__(self, *, fail: bool = False) -> None:
        super().__init__()
        self.state_dim = 2
        self.latent_dim = 2
        self.action_dim = 3
        self.action_horizon = 2
        self.anchor = nn.Parameter(torch.zeros(()))
        self.fail = fail
        self.latents: list[torch.Tensor] = []
        self.action_masks: list[torch.Tensor] = []
        self.feature_masks: list[torch.Tensor] = []
        self.noises: list[torch.Tensor] = []
        self.steps: list[int] = []

    def sample(
        self,
        current_state: torch.Tensor,
        *,
        success_latent: torch.Tensor,
        steps: int,
        action_mask: torch.Tensor,
        action_feature_mask: torch.Tensor,
        initial_noise: torch.Tensor,
    ) -> torch.Tensor:
        self.latents.append(success_latent.detach().clone())
        self.action_masks.append(action_mask.detach().clone())
        self.feature_masks.append(action_feature_mask.detach().clone())
        self.noises.append(initial_noise.detach().clone())
        self.steps.append(steps)
        if self.fail:
            raise RuntimeError("sample failed")
        result = torch.zeros(
            current_state.shape[0],
            self.action_horizon,
            self.action_dim,
            dtype=current_state.dtype,
            device=current_state.device,
        )
        result[:, 0, 0] = success_latent[:, 0]
        result[:, 0, 2] = success_latent[:, 1]
        return result


def _batch(
    *,
    starts: tuple[int, ...] = (0, 2, 0, 3),
    trajectory_ids: tuple[str, ...] = (
        "trajectory-a",
        "trajectory-a",
        "trajectory-b",
        "trajectory-b",
    ),
) -> Stage0WindowBatch:
    rows = len(starts)
    current_states = torch.arange(rows * 2, dtype=torch.float32).reshape(rows, 2)
    action_chunks = torch.zeros(rows, 2, 3, dtype=torch.float32)
    predictions = ((10.0, 20.0), (10.0, 20.0), (30.0, 40.0), (30.0, 40.0))
    errors = (0.0, 2.0, 1.0, 3.0)
    for index in range(rows):
        prediction = predictions[index]
        error = errors[index]
        action_chunks[index, 0, 0] = prediction[0] + error
        action_chunks[index, 0, 1] = 1_000.0
        action_chunks[index, 0, 2] = prediction[1] + error
    action_mask = torch.ones(rows, 2, dtype=torch.bool)
    return Stage0WindowBatch(
        current_states=current_states,
        action_chunks=action_chunks,
        action_mask=action_mask,
        future_actions=action_chunks.clone(),
        future_states=torch.zeros(rows, 2, 2),
        future_mask=action_mask.clone(),
        trajectory_ids=trajectory_ids,
        reset_group_ids=tuple(f"reset-{index // 2}" for index in range(rows)),
        starts=starts,
        corpus_sha256="a" * 64,
        window_policy_sha256="b" * 64,
    )


def _evaluate(policy: nn.Module, batch: Stage0WindowBatch | None = None):
    batch = _batch() if batch is None else batch
    noise = torch.arange(
        batch.batch_size * 2 * 3,
        dtype=torch.float32,
    ).reshape(batch.batch_size, 2, 3)
    return evaluate_fixed_start_latent_first_action_mse(
        policy,
        batch,
        episode_start_latents={
            "trajectory-a": torch.tensor([10.0, 20.0]),
            "trajectory-b": torch.tensor([30.0, 40.0]),
        },
        trajectory_mode_ids={
            "trajectory-a": "left",
            "trajectory-b": "right",
        },
        initial_noise=noise,
        effective_action_mask=torch.tensor([True, False, True]),
        flow_sample_steps=7,
        start_bin_width=2,
        sample_batch_size=2,
    )


class TemporalControlTest(unittest.TestCase):
    def test_fixed_start_latents_and_effective_first_action_mse(self) -> None:
        policy = _SpyPolicy()
        policy.train()
        batch = _batch()
        metrics = _evaluate(policy, batch)

        self.assertTrue(policy.training)
        self.assertEqual(metrics.overall.sample_count, 4)
        self.assertAlmostEqual(metrics.overall.first_action_mse, 3.5)
        self.assertAlmostEqual(metrics.start_zero.first_action_mse, 0.5)
        self.assertAlmostEqual(metrics.nonzero_start.first_action_mse, 6.5)
        per_mode = {value.mode_id: value.metrics for value in metrics.per_mode}
        self.assertAlmostEqual(per_mode["left"].first_action_mse, 2.0)
        self.assertAlmostEqual(per_mode["right"].first_action_mse, 5.0)
        per_bin = {
            value.start_inclusive: value.metrics for value in metrics.per_start_bin
        }
        self.assertAlmostEqual(per_bin[0].first_action_mse, 0.5)
        self.assertAlmostEqual(per_bin[2].first_action_mse, 6.5)

        observed_latents = torch.cat(policy.latents)
        self.assertTrue(
            torch.equal(
                observed_latents,
                torch.tensor([[10.0, 20.0], [10.0, 20.0], [30.0, 40.0], [30.0, 40.0]]),
            )
        )
        self.assertTrue(torch.equal(torch.cat(policy.action_masks), batch.action_mask))
        self.assertTrue(
            all(
                torch.equal(mask, torch.tensor([True, False, True]))
                for mask in policy.feature_masks
            )
        )
        expected_noise = torch.arange(24, dtype=torch.float32).reshape(4, 2, 3)
        self.assertTrue(torch.equal(torch.cat(policy.noises), expected_noise))
        self.assertEqual(policy.steps, [7, 7])

        with self.assertRaises(FrozenInstanceError):
            metrics.overall = metrics.nonzero_start  # type: ignore[misc]
        json.dumps(metrics.to_record(), allow_nan=False, sort_keys=True)

    def test_start_zero_only_and_mode_without_later_window_are_rejected(self) -> None:
        start_zero_only = _batch(
            starts=(0, 0),
            trajectory_ids=("trajectory-a", "trajectory-b"),
        )
        with self.assertRaisesRegex(ValueError, "start-zero-only"):
            _evaluate(_SpyPolicy(), start_zero_only)

        missing_right_later = _batch(
            starts=(0, 2, 0),
            trajectory_ids=("trajectory-a", "trajectory-a", "trajectory-b"),
        )
        with self.assertRaisesRegex(ValueError, "right.*nonzero-start"):
            _evaluate(_SpyPolicy(), missing_right_later)

    def test_policy_training_state_is_restored_after_sampling_error(self) -> None:
        policy = _SpyPolicy(fail=True)
        policy.train()
        with self.assertRaisesRegex(RuntimeError, "sample failed"):
            _evaluate(policy)
        self.assertTrue(policy.training)

        policy.eval()
        with self.assertRaisesRegex(RuntimeError, "sample failed"):
            _evaluate(policy)
        self.assertFalse(policy.training)


if __name__ == "__main__":
    unittest.main()
