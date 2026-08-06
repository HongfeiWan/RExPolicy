"""Tests for the non-privileged Grasp-Lift state view."""

from __future__ import annotations

import unittest

import torch

from rexpolicy.stage0.envs.grasp_lift_state_view import (
    DEFAULT_GRASP_LIFT_STATE_VIEW,
    GRASP_LIFT_FIXED_PHASE_ONE_HOT,
    GraspLiftStateView,
    apply_grasp_lift_state_view,
)
from rexpolicy.stage0.envs.state_schema import DEFAULT_STAGE0_STATE_SCHEMA


SCHEMA = DEFAULT_STAGE0_STATE_SCHEMA


class GraspLiftStateViewTest(unittest.TestCase):
    def test_hash_binds_constant_phase_policy(self) -> None:
        view = DEFAULT_GRASP_LIFT_STATE_VIEW
        self.assertEqual(view.sha256, GraspLiftStateView().sha256)
        self.assertEqual(len(view.sha256), 64)
        self.assertEqual(
            view.to_record()["fixed_phase_one_hot"],
            list(GRASP_LIFT_FIXED_PHASE_ONE_HOT),
        )

    def test_replaces_only_legacy_phase_for_batches_and_windows(self) -> None:
        state = torch.arange(2 * 3 * SCHEMA.dimension, dtype=torch.float32).reshape(
            2, 3, SCHEMA.dimension
        )
        phase = SCHEMA.slice("task_phase_one_hot")
        original = state.clone()

        viewed = apply_grasp_lift_state_view(state)

        self.assertTrue(torch.equal(state, original))
        self.assertTrue(
            torch.equal(
                viewed[..., phase],
                torch.tensor(GRASP_LIFT_FIXED_PHASE_ONE_HOT).expand(2, 3, -1),
            )
        )
        before = viewed.clone()
        before[..., phase] = original[..., phase]
        self.assertTrue(torch.equal(before, original))

    def test_in_place_mode_is_explicit(self) -> None:
        state = torch.zeros(1, SCHEMA.dimension, dtype=torch.float32)
        result = apply_grasp_lift_state_view(state, clone=False)
        self.assertIs(result, state)
        self.assertTrue(
            torch.equal(
                state[:, SCHEMA.slice("task_phase_one_hot")],
                torch.tensor((GRASP_LIFT_FIXED_PHASE_ONE_HOT,)),
            )
        )

    def test_invalid_shape_and_non_finite_state_fail_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "last dimension"):
            apply_grasp_lift_state_view(torch.zeros(1, SCHEMA.dimension - 1))
        state = torch.zeros(1, SCHEMA.dimension)
        state[0, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "finite"):
            apply_grasp_lift_state_view(state)


if __name__ == "__main__":
    unittest.main()
