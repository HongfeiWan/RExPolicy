"""Focused tests for the reward-free Stage 0 grasp-lift oracle."""

from __future__ import annotations

import math
import unittest

import torch

from rexpolicy.stage0.envs.grasp_lift_oracle import (
    GRASP_LIFT_ORACLE_ID,
    GraspLiftSuccessOracle,
    GraspLiftTransientEvents,
    grasp_lift_transient_events_from_info,
)
from rexpolicy.stage0.envs.state_schema import DEFAULT_STAGE0_STATE_SCHEMA


SCHEMA = DEFAULT_STAGE0_STATE_SCHEMA


def _states(count: int) -> torch.Tensor:
    state = torch.zeros((count, SCHEMA.dimension), dtype=torch.float32)
    state[:, SCHEMA.slice("object_position")] = torch.tensor((0.4, -0.2, 0.1))
    state[:, SCHEMA.slice("object_to_goal")] = torch.tensor((0.0, 0.0, 0.03))
    state[:, SCHEMA.slice("object_rotation_6d")] = torch.tensor(
        (1.0, 0.0, 0.0, 0.0, 1.0, 0.0)
    )
    return state


def _events(
    count: int,
    *,
    forbidden: bool = False,
    streak: int = 0,
    contact: bool = False,
    overflow: bool = False,
) -> GraspLiftTransientEvents:
    return GraspLiftTransientEvents(
        forbidden_hand_contact=torch.full((count,), forbidden, dtype=torch.bool),
        opposed_grasp_max_consecutive_physics_frames=torch.full(
            (count,), streak, dtype=torch.int32
        ),
        had_hand_contact=torch.full((count,), contact, dtype=torch.bool),
        collision_buffer_overflow=torch.full((count,), overflow, dtype=torch.bool),
    )


class GraspLiftOracleTest(unittest.TestCase):
    def test_oracle_has_a_versioned_identity(self) -> None:
        self.assertEqual(
            GRASP_LIFT_ORACLE_ID,
            "rexpolicy/stage0-grasp-lift-oracle/v1",
        )

    def test_requires_confirmed_opposition_lift_and_two_step_hold(self) -> None:
        state = _states(1)
        oracle = GraspLiftSuccessOracle(1)
        oracle.reset(state)

        state[:, SCHEMA.slice("is_grasped")] = 1.0
        state[:, SCHEMA.slice("object_position")][:, 2] += 0.031
        first = oracle.evaluate(state, _events(1, streak=6, contact=True))
        self.assertTrue(first.grasp_confirmed.item())
        self.assertEqual(first.success_hold_steps.item(), 1)
        self.assertFalse(first.success.item())

        second = oracle.evaluate(state, _events(1, streak=6, contact=True))
        self.assertTrue(second.success.item())
        self.assertFalse(second.failure.item())

    def test_forbidden_contact_and_buffer_overflow_fail_closed(self) -> None:
        state = _states(2)
        oracle = GraspLiftSuccessOracle(2)
        oracle.reset(state)
        events = GraspLiftTransientEvents(
            forbidden_hand_contact=torch.tensor((True, False)),
            opposed_grasp_max_consecutive_physics_frames=torch.zeros(
                2, dtype=torch.int32
            ),
            had_hand_contact=torch.zeros(2, dtype=torch.bool),
            collision_buffer_overflow=torch.tensor((False, True)),
        )

        result = oracle.evaluate(state, events)

        self.assertTrue(torch.equal(result.failure, torch.tensor((True, True))))
        self.assertTrue(result.forbidden_contact_violation[0].item())
        self.assertTrue(result.collision_buffer_overflow_violation[1].item())

    def test_lateral_motion_and_tilt_are_independent_safety_gates(self) -> None:
        state = _states(2)
        oracle = GraspLiftSuccessOracle(2)
        oracle.reset(state)
        position_slice = SCHEMA.slice("object_position")
        state[0, position_slice.start] += 0.031
        angle = math.radians(31.0)
        state[1, SCHEMA.slice("object_rotation_6d")] = torch.tensor(
            (1.0, 0.0, 0.0, 0.0, math.cos(angle), -math.sin(angle))
        )

        result = oracle.evaluate(state, _events(2))

        self.assertTrue(result.lateral_displacement_violation[0].item())
        self.assertTrue(result.bottle_tilt_violation[1].item())
        self.assertTrue(torch.equal(result.failure, torch.tensor((True, True))))

    def test_yaw_does_not_count_as_bottle_tilt(self) -> None:
        state = _states(1)
        oracle = GraspLiftSuccessOracle(1)
        oracle.reset(state)
        angle = math.radians(125.0)
        state[:, SCHEMA.slice("object_rotation_6d")] = torch.tensor(
            (
                math.cos(angle),
                -math.sin(angle),
                0.0,
                math.sin(angle),
                math.cos(angle),
                0.0,
            )
        )

        result = oracle.evaluate(state, _events(1))

        self.assertAlmostEqual(result.bottle_tilt.item(), 0.0, places=6)
        self.assertFalse(result.failure.item())

    def test_lift_projects_onto_goal_axis_in_base_frame(self) -> None:
        state = _states(2)
        state[:, SCHEMA.slice("object_rotation_6d")] = torch.tensor(
            (0.0, 0.0, 1.0, 0.0, 1.0, 0.0)
        )
        state[:, SCHEMA.slice("object_to_goal")] = torch.tensor((0.03, 0.0, 0.0))
        oracle = GraspLiftSuccessOracle(2)
        oracle.reset(state)
        position_slice = SCHEMA.slice("object_position")
        state[0, position_slice.start] += 0.031
        state[1, position_slice.start + 2] += 0.031

        result = oracle.evaluate(state, _events(2))

        self.assertAlmostEqual(result.lift_height[0].item(), 0.031, places=6)
        self.assertAlmostEqual(result.lateral_displacement[0].item(), 0.0, places=6)
        self.assertAlmostEqual(result.lift_height[1].item(), 0.0, places=6)
        self.assertTrue(result.lateral_displacement_violation[1].item())

    def test_reversed_bottle_axis_does_not_reverse_lift_direction(self) -> None:
        state = _states(1)
        state[:, SCHEMA.slice("object_rotation_6d")] = torch.tensor(
            (-1.0, 0.0, 0.0, 0.0, 1.0, 0.0)
        )
        oracle = GraspLiftSuccessOracle(1)
        oracle.reset(state)
        position_slice = SCHEMA.slice("object_position")
        state[0, position_slice.start + 2] += 0.031

        result = oracle.evaluate(state, _events(1))

        self.assertAlmostEqual(result.lift_height.item(), 0.031, places=6)
        self.assertAlmostEqual(result.lateral_displacement.item(), 0.0, places=6)

    def test_confirmed_grasp_drop_uses_control_step_debounce(self) -> None:
        state = _states(1)
        state[:, SCHEMA.slice("is_grasped")] = 1.0
        oracle = GraspLiftSuccessOracle(1, drop_confirm_control_steps=2)
        oracle.reset(state)
        oracle.evaluate(state, _events(1, streak=6, contact=True))
        state[:, SCHEMA.slice("is_grasped")] = 0.0

        first_gap = oracle.evaluate(state, _events(1, contact=False))
        second_gap = oracle.evaluate(state, _events(1, contact=False))

        self.assertFalse(first_gap.dropped_grasp_violation.item())
        self.assertTrue(second_gap.dropped_grasp_violation.item())
        self.assertTrue(second_gap.failure.item())

    def test_partial_reset_only_clears_selected_world(self) -> None:
        state = _states(2)
        oracle = GraspLiftSuccessOracle(2)
        oracle.reset(state)
        failed = oracle.evaluate(state, _events(2, forbidden=True))
        self.assertTrue(failed.failure.all())

        oracle.reset(state, torch.tensor((True, False)))
        after = oracle.evaluate(state, _events(2))

        self.assertFalse(after.failure[0])
        self.assertTrue(after.failure[1])

    def test_terminal_success_cannot_be_rewritten_as_failure(self) -> None:
        state = _states(1)
        oracle = GraspLiftSuccessOracle(1, success_hold_control_steps=1)
        oracle.reset(state)
        state[:, SCHEMA.slice("is_grasped")] = 1.0
        state[:, SCHEMA.slice("object_position")][:, 2] += 0.031
        success = oracle.evaluate(state, _events(1, streak=6, contact=True))
        self.assertTrue(success.success.item())

        terminal = oracle.evaluate(state, _events(1, forbidden=True, overflow=True))

        self.assertTrue(terminal.success.item())
        self.assertFalse(terminal.failure.item())
        self.assertFalse(terminal.forbidden_contact_violation.item())
        self.assertFalse(terminal.collision_buffer_overflow_violation.item())

    def test_info_adapter_expands_batch_level_overflow(self) -> None:
        info = {
            "forbidden_hand_contact_any_frame_this_control_step": torch.zeros(
                3, dtype=torch.bool
            ),
            "opposed_grasp_max_consecutive_physics_frames_this_control_step": torch.zeros(
                3, dtype=torch.int32
            ),
            "had_hand_contact_this_control_step": torch.ones(3, dtype=torch.bool),
            "control_step_diagnostics": {
                "rigid_contact_overflow_frame_count": torch.tensor((0,)),
                "triangle_pair_overflow_frame_count": torch.tensor((1,)),
            },
        }

        events = grasp_lift_transient_events_from_info(
            info, count=3, device=torch.device("cpu")
        )

        self.assertTrue(events.collision_buffer_overflow.all())


if __name__ == "__main__":
    unittest.main()
