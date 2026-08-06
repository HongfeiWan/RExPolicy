"""Focused tests for Grasp-Lift pilot runner bookkeeping."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import unittest

import torch

from tools.run_stage0_grasp_lift_pilot import (
    _build_newton_env_config,
    _failure_reason_from_tensors,
    _outcome_from_tensors,
    _split_assignments,
)
from rexpolicy.stage0.grasp_lift_pilot import (
    grasp_lift_pilot_environment_config_record,
    load_grasp_lift_pilot_config,
)
from rexpolicy.stage0.types import canonical_fingerprint


ROOT = Path(__file__).resolve().parents[1]


class GraspLiftPilotRunnerTest(unittest.TestCase):
    def test_complete_newton_config_matches_the_pilot_contract(self) -> None:
        from rexpolicy.envs.groot_newton_env import GrootNewtonEnvConfig

        config = load_grasp_lift_pilot_config(
            ROOT / "configs" / "stage0" / "grasp_lift_pilot.json"
        )
        actual = asdict(
            _build_newton_env_config(config.runtime, GrootNewtonEnvConfig)
        )
        expected = grasp_lift_pilot_environment_config_record(config.runtime)
        self.assertEqual(
            canonical_fingerprint(actual),
            canonical_fingerprint(expected),
        )

    def test_train_validation_split_is_deterministic_and_has_no_test(self) -> None:
        trajectory_groups = {
            f"grasp-{index:03d}": f"reset-{index:03d}" for index in range(32)
        }
        first = _split_assignments(
            trajectory_groups,
            validation_fraction=0.25,
        )
        second = _split_assignments(
            dict(reversed(tuple(trajectory_groups.items()))),
            validation_fraction=0.25,
        )
        self.assertEqual(first, second)
        self.assertEqual(set(first.values()), {"train", "validation"})
        self.assertEqual(sum(value == "validation" for value in first.values()), 8)

    def test_split_is_stable_across_trajectory_and_mode_identity(self) -> None:
        first_groups = {
            f"mode-a-trajectory-{index}": f"reset-{index // 2}"
            for index in range(16)
        }
        second_groups = {
            f"mode-b-rewritten-trajectory-{index}": f"reset-{index // 2}"
            for index in range(16)
        }
        first = _split_assignments(first_groups, validation_fraction=0.25)
        second = _split_assignments(second_groups, validation_fraction=0.25)
        for index in range(16):
            self.assertEqual(
                first[f"mode-a-trajectory-{index}"],
                second[f"mode-b-rewritten-trajectory-{index}"],
            )
        for index in range(0, 16, 2):
            self.assertEqual(
                first[f"mode-a-trajectory-{index}"],
                first[f"mode-a-trajectory-{index + 1}"],
            )

    def test_failure_reason_uses_last_recorded_transition(self) -> None:
        tensors = {
            "forbidden_contact_violation": torch.tensor((False, False)),
            "lateral_displacement_violation": torch.tensor((False, True)),
            "bottle_tilt_violation": torch.tensor((False, False)),
            "dropped_grasp_violation": torch.tensor((False, True)),
            "collision_buffer_overflow_violation": torch.tensor((False, False)),
        }
        self.assertEqual(
            _failure_reason_from_tensors(tensors),
            "lateral_displacement+dropped_grasp",
        )

    def test_outcome_uses_only_the_last_recorded_transition(self) -> None:
        tensors = {
            "oracle_success": torch.tensor((False, True)),
            "oracle_failure": torch.tensor((False, False)),
        }
        self.assertEqual(
            _outcome_from_tensors(tensors),
            ("success", True, False, None),
        )
        tensors.update(
            {
                "oracle_success": torch.tensor((False, False)),
                "oracle_failure": torch.tensor((False, True)),
                "forbidden_contact_violation": torch.tensor((False, False)),
                "lateral_displacement_violation": torch.tensor((False, True)),
                "bottle_tilt_violation": torch.tensor((False, False)),
                "dropped_grasp_violation": torch.tensor((False, False)),
                "collision_buffer_overflow_violation": torch.tensor(
                    (False, False)
                ),
            }
        )
        self.assertEqual(
            _outcome_from_tensors(tensors),
            ("failure", True, False, "lateral_displacement"),
        )
        tensors["oracle_failure"] = torch.tensor((False, False))
        self.assertEqual(
            _outcome_from_tensors(tensors),
            ("timeout", False, True, None),
        )


if __name__ == "__main__":
    unittest.main()
