"""Tests for leakage-safe, transition-aligned Stage 0 future windows."""

from __future__ import annotations

import unittest

import torch

from rexpolicy.stage0.data import (
    Stage0SplitPolicy,
    Stage0Trajectory,
    Stage0TrajectoryProvenance,
    Stage0WindowPolicy,
    build_success_window_splits,
    extract_success_windows,
    split_trajectories,
)
from rexpolicy.stage0.types import Stage0Outcome


def _trajectory(
    trajectory_id: str,
    reset_group_id: str,
    *,
    outcome: Stage0Outcome = Stage0Outcome.SUCCESS,
) -> Stage0Trajectory:
    terminal = {
        Stage0Outcome.SUCCESS: (True, False, None),
        Stage0Outcome.FAILURE: (True, False, "oracle-failure"),
        Stage0Outcome.TIMEOUT: (False, True, None),
    }[outcome]
    provenance = Stage0TrajectoryProvenance(
        trajectory_id=trajectory_id,
        reset_group_id=reset_group_id,
        task_id="reach/v1",
        oracle_id="reach-oracle/v1",
        simulator_sha256="1" * 64,
        experiment_sha256="2" * 64,
        state_schema_sha256="3" * 64,
        action_schema_sha256="4" * 64,
        generation=0,
        rank=0,
        world=0,
        reset_seed=0,
    )
    return Stage0Trajectory(
        provenance=provenance,
        states=torch.tensor(
            [[0.0, 10.0], [1.0, 11.0], [2.0, 12.0], [3.0, 13.0]],
        ),
        actions=torch.tensor([[100.0], [101.0], [102.0]]),
        outcome=outcome,
        terminated=terminal[0],
        truncated=terminal[1],
        failure_reason=terminal[2],
    )


class Stage0WindowTests(unittest.TestCase):
    def test_windows_use_post_action_states_and_one_shared_mask(self) -> None:
        trajectory = _trajectory("success-000", "reset-000")
        windows = extract_success_windows(
            trajectory,
            Stage0WindowPolicy(action_horizon=2, future_horizon=4),
        )
        self.assertEqual(len(windows), 3)
        first = windows[0]
        self.assertTrue(torch.equal(first.current_state, trajectory.states[0]))
        self.assertTrue(
            torch.equal(
                first.future_actions[:, 0],
                torch.tensor([100.0, 101.0, 102.0, 0.0]),
            )
        )
        self.assertTrue(
            torch.equal(
                first.future_states,
                torch.tensor([[1.0, 11.0], [2.0, 12.0], [3.0, 13.0], [0.0, 0.0]]),
            )
        )
        self.assertTrue(
            torch.equal(first.future_mask, torch.tensor([True, True, True, False]))
        )
        self.assertTrue(torch.equal(first.action_chunk, first.future_actions[:2]))
        self.assertTrue(torch.equal(first.action_mask, first.future_mask[:2]))

        tail = windows[-1]
        self.assertEqual(tail.start, 2)
        self.assertTrue(torch.equal(tail.current_state, trajectory.states[2]))
        self.assertTrue(torch.equal(tail.future_states[0], trajectory.states[3]))
        self.assertEqual(int(tail.future_mask.sum().item()), 1)

    def test_failure_and_timeout_never_produce_training_windows(self) -> None:
        policy = Stage0WindowPolicy(action_horizon=1, future_horizon=2)
        for outcome in (Stage0Outcome.FAILURE, Stage0Outcome.TIMEOUT):
            trajectory = _trajectory(
                f"trajectory-{outcome.value}",
                f"reset-{outcome.value}",
                outcome=outcome,
            )
            self.assertEqual(extract_success_windows(trajectory, policy), ())

    def test_horizon_and_stride_are_strict(self) -> None:
        with self.assertRaisesRegex(ValueError, "<="):
            Stage0WindowPolicy(action_horizon=3, future_horizon=2)
        windows = extract_success_windows(
            _trajectory("success-000", "reset-000"),
            Stage0WindowPolicy(action_horizon=1, future_horizon=2, stride=2),
        )
        self.assertEqual([window.start for window in windows], [0, 2])

    def test_split_is_deterministic_and_never_leaks_reset_groups(self) -> None:
        trajectories = tuple(
            _trajectory(f"trajectory-{index:02d}", f"reset-{index // 2:02d}")
            for index in range(20)
        )
        policy = Stage0SplitPolicy(seed=91)
        first = split_trajectories(trajectories, policy)
        second = split_trajectories(reversed(trajectories), policy)
        for name in ("train", "validation", "test"):
            first_ids = [item.provenance.trajectory_id for item in getattr(first, name)]
            second_ids = [
                item.provenance.trajectory_id for item in getattr(second, name)
            ]
            self.assertEqual(first_ids, second_ids)
        group_sets = [
            {item.provenance.reset_group_id for item in getattr(first, name)}
            for name in ("train", "validation", "test")
        ]
        self.assertFalse(group_sets[0] & group_sets[1])
        self.assertFalse(group_sets[0] & group_sets[2])
        self.assertFalse(group_sets[1] & group_sets[2])
        self.assertEqual([len(groups) for groups in group_sets], [8, 1, 1])

    def test_split_precedes_windowing_and_retains_failure_for_accounting(self) -> None:
        success = _trajectory("success-000", "reset-shared")
        failure = _trajectory(
            "failure-000",
            "reset-shared",
            outcome=Stage0Outcome.FAILURE,
        )
        other = _trajectory("success-001", "reset-other")
        trajectories = (success, failure, other)
        split_policy = Stage0SplitPolicy(
            train_fraction=0.5,
            validation_fraction=0.0,
            test_fraction=0.5,
            seed=7,
        )
        trajectory_splits = split_trajectories(trajectories, split_policy)
        containing_success = next(
            split
            for split in (trajectory_splits.train, trajectory_splits.test)
            if success in split
        )
        self.assertIn(failure, containing_success)

        windows = build_success_window_splits(
            trajectories,
            split_policy=split_policy,
            window_policy=Stage0WindowPolicy(
                action_horizon=1,
                future_horizon=2,
            ),
        )
        all_windows = windows.train + windows.validation + windows.test
        self.assertEqual(
            {window.trajectory_id for window in all_windows},
            {"success-000", "success-001"},
        )
        self.assertEqual(
            {window.corpus_sha256 for window in all_windows},
            {windows.corpus_sha256},
        )


if __name__ == "__main__":
    unittest.main()
