"""Tests for explicit legacy-to-TaskSpec v2 runtime migration."""

from __future__ import annotations

import unittest
from dataclasses import replace

from rexpolicy.flywheel.task_runtime_adapter import (
    bind_legacy_task,
    load_production_reach_runtime_binding,
)
from rexpolicy.flywheel.task_spec import REACH_GREEN_CAP_V1


class TestRuntimeTaskBinding(unittest.TestCase):
    def test_production_binding_is_explicit_and_hash_bound(self) -> None:
        artifacts, binding = load_production_reach_runtime_binding()

        self.assertEqual(binding.legacy_task_id, "reach_green_cap/v1")
        self.assertEqual(binding.task_contract_id, (
            "rexpolicy_taskspec_v2/reach_green_cap/v2"
        ))
        self.assertEqual(binding.task_spec_fingerprint, artifacts.task.fingerprint)
        self.assertEqual(
            binding.task_contract_fingerprint,
            artifacts.contract.fingerprint,
        )
        self.assertEqual(binding.reward_profile_id, "reach_progress/v2")
        self.assertEqual(len(binding.fingerprint), 64)

    def test_matching_strings_cannot_bypass_the_migration_allowlist(self) -> None:
        artifacts, _ = load_production_reach_runtime_binding()
        unknown = replace(
            REACH_GREEN_CAP_V1,
            task_id="lookalike/v1",
        )
        with self.assertRaisesRegex(ValueError, "no trusted"):
            bind_legacy_task(unknown, artifacts)

    def test_instruction_mask_and_mode_must_remain_compatible(self) -> None:
        artifacts, _ = load_production_reach_runtime_binding()
        cases = (
            (
                replace(REACH_GREEN_CAP_V1, instruction="different"),
                "instruction",
            ),
            (
                replace(REACH_GREEN_CAP_V1, environment_task_mode="other"),
                "task mode",
            ),
            (
                replace(
                    REACH_GREEN_CAP_V1,
                    action_dimension_masks={
                        **REACH_GREEN_CAP_V1.action_dimension_masks,
                        "eef_9d": (True,) * 9,
                    },
                ),
                "projection",
            ),
        )
        for legacy, message in cases:
            with self.assertRaisesRegex(ValueError, message):
                bind_legacy_task(legacy, artifacts)


if __name__ == "__main__":
    unittest.main()
