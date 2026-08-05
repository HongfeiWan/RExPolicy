#!/usr/bin/env python3
"""Focused CPU tests for Stage 0 train-split normalization."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from rexpolicy.stage0.normalization import (
    fit_stage0_normalization,
    read_stage0_normalization,
    write_stage0_normalization,
)


class Stage0NormalizationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.states = torch.tensor(
            [[1.0, 10.0], [3.0, 10.0], [5.0, 10.0]], dtype=torch.float32
        )
        self.actions = torch.tensor(
            [[2.0, 40.0, 9.0], [4.0, 50.0, 8.0], [6.0, 60.0, 7.0]],
            dtype=torch.float32,
        )
        self.normalization = fit_stage0_normalization(
            self.states,
            self.actions,
            effective_action_mask=torch.tensor([True, True, False]),
        )

    def test_round_trip_and_effective_projection(self) -> None:
        normalized = self.normalization.normalize_actions(self.actions)
        self.assertTrue(torch.equal(normalized[:, 2], torch.zeros(3)))
        restored = self.normalization.denormalize_actions(normalized)
        self.assertTrue(torch.allclose(restored[:, :2], self.actions[:, :2]))
        self.assertTrue(torch.equal(restored[:, 2], torch.zeros(3)))
        normalized_states = self.normalization.normalize_states(self.states)
        self.assertLess(abs(float(normalized_states[:, 0].mean())), 1.0e-6)
        self.assertTrue(torch.equal(normalized_states[:, 1], torch.zeros(3)))

    def test_padding_is_zero_after_normalization(self) -> None:
        states = self.states[:2].reshape(1, 2, 2)
        actions = self.actions[:2].reshape(1, 2, 3)
        mask = torch.tensor([[True, False]])
        normalized_states = self.normalization.normalize_states(states, valid_mask=mask)
        normalized_actions = self.normalization.normalize_actions(
            actions, valid_mask=mask
        )
        self.assertTrue(torch.equal(normalized_states[:, 1], torch.zeros(1, 2)))
        self.assertTrue(torch.equal(normalized_actions[:, 1], torch.zeros(1, 3)))

    def test_record_round_trip_preserves_hash(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "normalization.json"
            write_stage0_normalization(path, self.normalization)
            restored = read_stage0_normalization(path)
        self.assertEqual(restored.sha256, self.normalization.sha256)
        self.assertEqual(restored.to_record(), self.normalization.to_record())

    def test_rejects_invalid_effective_mask(self) -> None:
        with self.assertRaisesRegex(ValueError, "effective_action_mask"):
            fit_stage0_normalization(
                self.states,
                self.actions,
                effective_action_mask=torch.zeros(3, dtype=torch.bool),
            )


if __name__ == "__main__":
    unittest.main()
