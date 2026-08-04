"""Tests for Groot raw-signal extraction and TaskSpec shadow evaluation."""

from __future__ import annotations

import unittest

from rexpolicy.flywheel.task_runtime_adapter import (
    load_production_reach_runtime_binding,
)
from rexpolicy.flywheel.task_shadow import (
    begin_shadow_decision,
    compare_groot_step_parity,
    evaluate_shadow_transition,
    extract_groot_reach_metric_rows,
    require_identical_root_rows,
)
from rexpolicy.tasking.runtime import initial_runtime_state


class _ForbiddenPrevious:
    def __iter__(self):
        raise AssertionError("reach_previous_distance must not be read")


def _source(
    distances,
    *,
    contacts=None,
    displacements=None,
):
    count = len(distances)
    return {
        "reach_distance": list(distances),
        "reach_contact_violation": (
            [False] * count if contacts is None else list(contacts)
        ),
        "reach_displacement_violation": (
            [False] * count if displacements is None else list(displacements)
        ),
        "reach_previous_distance": _ForbiddenPrevious(),
        "reward": [999.0] * count,
    }


class TestTaskShadow(unittest.TestCase):
    def setUp(self) -> None:
        self.artifacts, self.binding = load_production_reach_runtime_binding()
        self.schema = self.artifacts.capabilities.event_schemas[0]
        self.root_state = initial_runtime_state(
            self.artifacts.contract,
            reward_profile_id=self.binding.reward_profile_id,
            catalog=self.artifacts.capabilities,
        )

    def test_adapter_extracts_only_certified_current_raw_signals(self) -> None:
        rows = extract_groot_reach_metric_rows(
            _source((0.1, 0.1)),
            event_schema=self.schema,
        )
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["reach.distance_m"], 0.1)
        self.assertNotIn("reward", rows[0])
        self.assertNotIn("reach_previous_distance", rows[0])

    def test_root_worlds_must_be_identical(self) -> None:
        rows = extract_groot_reach_metric_rows(
            _source((0.1, 0.2)),
            event_schema=self.schema,
        )
        with self.assertRaisesRegex(ValueError, "identical root"):
            require_identical_root_rows(rows)

    def test_active_worlds_advance_independently(self) -> None:
        branch = begin_shadow_decision(
            root_state=self.root_state,
            root_source=_source((0.1, 0.1)),
            event_schema=self.schema,
        )
        batch = evaluate_shadow_transition(
            branch,
            transition_source=_source(
                (0.08, 0.03),
                contacts=(False, True),
            ),
            active_before=(True, True),
            contract=self.artifacts.contract,
            catalog=self.artifacts.capabilities,
        )

        self.assertAlmostEqual(batch.results[0].reward, 2.0 / 3.0)
        self.assertTrue(batch.results[1].failure)
        self.assertEqual(batch.results[1].reward, -1.0)
        self.assertEqual(
            batch.next_branch.previous_rows[0]["reach.distance_m"],
            0.08,
        )

    def test_inactive_world_hold_steps_are_not_recorded_or_evaluated(self) -> None:
        branch = begin_shadow_decision(
            root_state=self.root_state,
            root_source=_source((0.1, 0.1)),
            event_schema=self.schema,
        )
        first = evaluate_shadow_transition(
            branch,
            transition_source=_source((0.08, 0.03), contacts=(False, True)),
            active_before=(True, True),
            contract=self.artifacts.contract,
            catalog=self.artifacts.capabilities,
        )
        second = evaluate_shadow_transition(
            first.next_branch,
            transition_source=_source((0.07, 0.02)),
            active_before=(True, False),
            contract=self.artifacts.contract,
            catalog=self.artifacts.capabilities,
        )
        self.assertIsNone(second.results[1])
        self.assertEqual(
            second.next_branch.previous_rows[1],
            first.next_branch.previous_rows[1],
        )
        self.assertEqual(
            second.next_branch.world_states[1],
            first.next_branch.world_states[1],
        )

    def test_environment_parity_reports_exact_mismatch_dimensions(self) -> None:
        branch = begin_shadow_decision(
            root_state=self.root_state,
            root_source=_source((0.1,)),
            event_schema=self.schema,
        )
        result = evaluate_shadow_transition(
            branch,
            transition_source=_source((0.08,)),
            active_before=(True,),
            contract=self.artifacts.contract,
            catalog=self.artifacts.capabilities,
        ).results[0]
        parity = compare_groot_step_parity(
            result,
            environment_reward=result.reward,
            environment_success=False,
            environment_failure=False,
            environment_terminated=False,
            environment_truncated=False,
        )
        self.assertTrue(parity.accepted)

        mismatch = compare_groot_step_parity(
            result,
            environment_reward=-1.0,
            environment_success=True,
            environment_failure=False,
            environment_terminated=False,
            environment_truncated=True,
        )
        self.assertFalse(mismatch.accepted)
        self.assertEqual(
            mismatch.mismatches,
            ("reward", "success", "truncated"),
        )


if __name__ == "__main__":
    unittest.main()
