from __future__ import annotations

import unittest

import torch

from rexpolicy.stage0.envs.reach_modes import (
    DEFAULT_REACH_MODE_LIBRARY,
    ReachModeController,
    ReachModeSpec,
    build_reach_mode_plan,
    minimum_polyline_object_clearance,
)
from rexpolicy.stage0.envs.state_schema import DEFAULT_STAGE0_STATE_SCHEMA


def _state_batch(batch_size: int = 4) -> torch.Tensor:
    schema = DEFAULT_STAGE0_STATE_SCHEMA
    state = torch.zeros(batch_size, schema.dimension, dtype=torch.float32)
    state[:, schema.slice("eef_position")] = torch.tensor((0.30, -0.20, 0.30))
    state[:, schema.slice("object_position")] = torch.tensor((0.50, 0.00, 0.20))
    state[:, schema.slice("goal_position")] = torch.tensor((0.68, 0.08, 0.25))
    state[:, schema.slice("eef_to_object")] = (
        state[:, schema.slice("object_position")]
        - state[:, schema.slice("eef_position")]
    )
    state[:, schema.slice("object_to_goal")] = (
        state[:, schema.slice("goal_position")]
        - state[:, schema.slice("object_position")]
    )
    return state


def _open_hand_reference(state: torch.Tensor) -> torch.Tensor:
    action = torch.zeros(state.shape[0], 19, dtype=state.dtype, device=state.device)
    action[:, :3] = state[:, DEFAULT_STAGE0_STATE_SCHEMA.slice("eef_position")]
    action[:, 3:9] = torch.tensor(
        (1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
        dtype=state.dtype,
        device=state.device,
    )
    action[:, 9:] = torch.tensor(
        (0.18, 0.31, 0.0, 0.0, 0.0, 0.0, 0.05, 0.02, 0.11, 0.0),
        dtype=state.dtype,
        device=state.device,
    )
    return action


class Stage0ReachModesTest(unittest.TestCase):
    def test_library_has_four_hash_bound_modes(self) -> None:
        library = DEFAULT_REACH_MODE_LIBRARY
        self.assertEqual(len(library.mode_ids), 4)
        self.assertEqual(len(set(library.mode_ids)), 4)
        hashes = tuple(library.mode_hash(mode_id) for mode_id in library.mode_ids)
        self.assertEqual(len(set(hashes)), 4)
        self.assertTrue(all(len(value) == 64 for value in hashes))
        self.assertEqual(library.sha256, DEFAULT_REACH_MODE_LIBRARY.sha256)

    def test_mode_spec_rejects_unversioned_or_ambiguous_contracts(self) -> None:
        with self.assertRaisesRegex(ValueError, "mode_id"):
            ReachModeSpec("left", 0.05, 0.0, 0.0)
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            ReachModeSpec(
                "stage0/reach/test/v1",
                0.0,
                0.0,
                0.0,
                waypoint_progress=(0.5, 0.5, 1.0),
            )

    def test_same_reset_seed_has_repeatable_balanced_mode_order(self) -> None:
        library = DEFAULT_REACH_MODE_LIBRARY
        first = library.ordered_mode_ids(1729)
        second = library.ordered_mode_ids(1729)
        self.assertEqual(first, second)
        self.assertEqual(set(first), set(library.mode_ids))
        assigned_once = tuple(
            library.assign((1729,), slot)[0] for slot in range(len(library.mode_ids))
        )
        self.assertEqual(assigned_once, first)

    def test_same_reset_modes_are_distinct_safe_paths_to_same_goal(self) -> None:
        library = DEFAULT_REACH_MODE_LIBRARY
        state = _state_batch(len(library.mode_ids))
        plan = build_reach_mode_plan(state, library.mode_ids)
        schema = DEFAULT_STAGE0_STATE_SCHEMA
        goal = state[:, schema.slice("goal_position")]
        torch.testing.assert_close(plan.waypoints[:, -1], goal)
        self.assertTrue(bool((plan.minimum_object_clearance >= 0.050).all()))

        middle = plan.waypoints[:, 1]
        pairwise = torch.cdist(middle, middle)
        off_diagonal = pairwise[~torch.eye(len(library.mode_ids), dtype=torch.bool)]
        self.assertGreater(float(off_diagonal.min()), 0.025)
        self.assertEqual(plan.mode_ids, library.mode_ids)
        self.assertEqual(
            plan.mode_hashes,
            tuple(library.mode_hash(mode_id) for mode_id in library.mode_ids),
        )

    def test_controller_pins_rotation_and_open_hand_across_all_steps(self) -> None:
        library = DEFAULT_REACH_MODE_LIBRARY
        state = _state_batch(len(library.mode_ids))
        reference = _open_hand_reference(state)
        controller = ReachModeController(max_translation_step_m=0.020)
        first_plan = controller.reset(state.clone(), reference, library.mode_ids)
        first_waypoints = first_plan.waypoints.clone()

        eef_slice = DEFAULT_STAGE0_STATE_SCHEMA.slice("eef_position")
        eef_traces = [[state[row, eef_slice].clone()] for row in range(state.shape[0])]
        current = state.clone()
        for _ in range(80):
            action = controller.next_action(current)
            torch.testing.assert_close(action[:, 3:], reference[:, 3:])
            step = action[:, :3] - current[:, eef_slice]
            self.assertLessEqual(
                float(torch.linalg.vector_norm(step, dim=-1).max()), 0.020001
            )
            current[:, eef_slice].copy_(action[:, :3])
            for row in range(state.shape[0]):
                eef_traces[row].append(action[row, :3].clone())

        goal = state[:, DEFAULT_STAGE0_STATE_SCHEMA.slice("goal_position")]
        torch.testing.assert_close(current[:, eef_slice], goal, atol=1.0e-6, rtol=0.0)
        final_traces = torch.stack([torch.stack(rows) for rows in eef_traces])
        # All four ideal kinematic rollouts originate from an identical reset
        # but at least one commanded EEF point separates every mode pair.
        for left in range(len(library.mode_ids)):
            for right in range(left + 1, len(library.mode_ids)):
                separation = torch.linalg.vector_norm(
                    final_traces[left] - final_traces[right], dim=-1
                )
                self.assertGreater(float(separation.max()), 0.015)

        repeated = ReachModeController(max_translation_step_m=0.020)
        repeated_plan = repeated.reset(state.clone(), reference, library.mode_ids)
        torch.testing.assert_close(repeated_plan.waypoints, first_waypoints)
        independent = ReachModeController(max_translation_step_m=0.020)
        independent.reset(state.clone(), reference, library.mode_ids)
        torch.testing.assert_close(
            repeated.next_action(state.clone()),
            independent.next_action(state.clone()),
        )

    def test_polyline_clearance_matches_simple_cpu_geometry(self) -> None:
        start = torch.tensor(((0.0, 0.0, 0.0),), dtype=torch.float64)
        waypoints = torch.tensor(
            (((1.0, 0.0, 0.0), (2.0, 0.0, 0.0)),), dtype=torch.float64
        )
        obstacle = torch.tensor(((0.5, 0.3, 0.0),), dtype=torch.float64)
        clearance = minimum_polyline_object_clearance(start, waypoints, obstacle)
        torch.testing.assert_close(clearance, torch.tensor((0.3,), dtype=torch.float64))

    def test_unsafe_geometry_fails_closed(self) -> None:
        state = _state_batch(1)
        schema = DEFAULT_STAGE0_STATE_SCHEMA
        state[:, schema.slice("eef_position")] = torch.tensor((0.0, 0.0, 0.0))
        state[:, schema.slice("object_position")] = torch.tensor((0.5, 0.0, 0.0))
        state[:, schema.slice("goal_position")] = torch.tensor((1.0, 0.0, 0.0))
        with self.assertRaisesRegex(ValueError, "object clearance"):
            build_reach_mode_plan(
                state,
                ("stage0/reach/direct/v1",),
                minimum_object_clearance_m=0.10,
            )


if __name__ == "__main__":
    unittest.main()
