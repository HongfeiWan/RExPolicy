"""Tests for the default-off Flow-DiT success-token boundary."""

from __future__ import annotations

import unittest


class TestSuccessTokenCollation(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        import torch

        cls.torch = torch

    def _condition(self, length: int, token=None):
        from rexpolicy.flywheel.groot_policy import CachedCondition

        torch = self.torch
        return CachedCondition(
            backbone_features=torch.arange(length * 4).reshape(length, 4),
            backbone_attention_mask=torch.ones(length, dtype=torch.bool),
            image_mask=torch.tensor([True] + [False] * (length - 1)),
            state=torch.zeros(1, 3),
            embodiment_id=7,
            success_latent_token=token,
        )

    def _policy_shell(self, *, enabled: bool):
        from rexpolicy.flywheel.groot_policy import GrootFlowDitPolicy

        policy = object.__new__(GrootFlowDitPolicy)
        policy.torch = self.torch
        policy.device = self.torch.device("cpu")
        policy.compute_dtype = self.torch.float32
        policy.success_conditioning_enabled = enabled
        return policy

    def test_disabled_path_preserves_original_sequence(self) -> None:
        policy = self._policy_shell(enabled=False)
        backbone, _ = policy._collate_conditions([self._condition(3)])
        self.assertEqual(tuple(backbone.backbone_features.shape), (1, 3, 4))
        self.assertEqual(backbone.backbone_attention_mask.sum().item(), 3)

    def test_enabled_path_appends_non_image_token_after_valid_tokens(self) -> None:
        torch = self.torch
        policy = self._policy_shell(enabled=True)
        token = torch.tensor([101.0, 102.0, 103.0, 104.0])
        backbone, _ = policy._collate_conditions(
            [self._condition(2, token), self._condition(3, token + 10)]
        )
        self.assertEqual(tuple(backbone.backbone_features.shape), (2, 4, 4))
        self.assertTrue(
            torch.equal(backbone.backbone_features[0, 2], token)
        )
        self.assertTrue(backbone.backbone_attention_mask[0, 2])
        self.assertFalse(backbone.image_mask[0, 2])
        self.assertFalse(backbone.backbone_attention_mask[0, 3])

    def test_flag_and_shape_fail_closed(self) -> None:
        torch = self.torch
        disabled = self._policy_shell(enabled=False)
        with self.assertRaisesRegex(RuntimeError, "enable_success_conditioning"):
            disabled._collate_conditions(
                [self._condition(2, torch.zeros(4))]
            )
        enabled = self._policy_shell(enabled=True)
        with self.assertRaisesRegex(RuntimeError, "every cached condition"):
            enabled._collate_conditions([self._condition(2)])
        with self.assertRaisesRegex(ValueError, "dimension"):
            enabled._collate_conditions(
                [self._condition(2, torch.zeros(3))]
            )

    def test_training_collator_supports_mixed_v1_and_v2_samples(self) -> None:
        from rexpolicy.flywheel.experience import (
            TrainingSample,
            collate_training_samples,
        )

        torch = self.torch

        def sample(token=None):
            return TrainingSample(
                backbone_features=torch.ones(2, 4),
                backbone_attention_mask=torch.ones(2, dtype=torch.bool),
                image_mask=torch.tensor([True, False]),
                state=torch.zeros(1, 3),
                embodiment_id=1,
                action=torch.zeros(2, 5),
                action_mask=torch.ones(2, 5),
                valid_steps=2,
                success_latent_token=token,
            )

        backbone, _ = collate_training_samples(
            [sample(torch.tensor([2.0, 3.0, 4.0, 5.0])), sample()],
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        self.assertEqual(tuple(backbone.backbone_features.shape), (2, 3, 4))
        self.assertTrue(backbone.backbone_attention_mask[0, 2])
        self.assertFalse(backbone.backbone_attention_mask[1, 2])
        self.assertFalse(backbone.image_mask[:, 2].any())


if __name__ == "__main__":
    unittest.main()
