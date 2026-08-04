"""Tests for deterministic terminal-safe compiled task evaluation."""

from __future__ import annotations

import copy
import unittest
from dataclasses import replace

from rexpolicy.tasking.capabilities import CapabilityCatalog
from rexpolicy.tasking.contract import compile_task_contract
from rexpolicy.tasking.expression import MetricFrame
from rexpolicy.tasking.model import TaskSpecV2
from rexpolicy.tasking.runtime import (
    ACTIVE,
    FAILURE,
    SUCCESS,
    TRUNCATED,
    TaskEvaluationError,
    evaluate_task_step,
    initial_runtime_state,
)
from tools.test_tasking_capabilities import capability_record
from tools.test_tasking_model import task_record


def _context(*, task_value=None, capability_value=None):
    catalog = CapabilityCatalog.from_record(
        capability_value if capability_value is not None else capability_record()
    )
    task = TaskSpecV2.from_record(
        task_value if task_value is not None else task_record()
    )
    contract = compile_task_contract(task, catalog=catalog)
    state = initial_runtime_state(
        contract,
        reward_profile_id=contract.reward_profiles[0].reward_profile_id,
        catalog=catalog,
    )
    return contract, catalog, state


def _frame(
    catalog,
    *,
    previous=0.10,
    current=0.08,
    contact=False,
    displacement=False,
):
    schema = catalog.event_schemas[0]
    return MetricFrame(
        event_schema_id=schema.event_schema_id,
        event_schema_fingerprint=schema.fingerprint,
        previous={"reach.distance_m": previous},
        current={
            "reach.distance_m": current,
            "reach.contact_violation": contact,
            "reach.displacement_violation": displacement,
        },
    )


class TestTaskRuntime(unittest.TestCase):
    def test_goal_requires_consecutive_hold_steps(self) -> None:
        contract, catalog, state = _context()
        first = evaluate_task_step(
            contract,
            state=state,
            frame=_frame(catalog, previous=0.05, current=0.03),
            catalog=catalog,
        )
        second = evaluate_task_step(
            contract,
            state=first.next_state,
            frame=_frame(catalog, previous=0.03, current=0.02),
            catalog=catalog,
        )

        self.assertEqual(first.status, ACTIVE)
        self.assertEqual(first.next_state.goal_streak, 1)
        self.assertEqual(second.status, SUCCESS)
        self.assertTrue(second.success)
        self.assertEqual(second.reward, 1.0)
        self.assertEqual(second.reward_terms, ())

    def test_false_goal_resets_the_streak(self) -> None:
        contract, catalog, state = _context()
        close = evaluate_task_step(
            contract,
            state=state,
            frame=_frame(catalog, current=0.03),
            catalog=catalog,
        )
        far = evaluate_task_step(
            contract,
            state=close.next_state,
            frame=_frame(catalog, previous=0.03, current=0.08),
            catalog=catalog,
        )
        self.assertEqual(far.status, ACTIVE)
        self.assertEqual(far.next_state.goal_streak, 0)

    def test_safety_failure_overrides_goal_and_shaping(self) -> None:
        contract, catalog, state = _context()
        result = evaluate_task_step(
            contract,
            state=state,
            frame=_frame(
                catalog,
                previous=0.10,
                current=0.03,
                contact=True,
            ),
            catalog=catalog,
        )

        self.assertTrue(result.goal_predicate)
        self.assertEqual(result.status, FAILURE)
        self.assertTrue(result.failure)
        self.assertTrue(result.safety_violation)
        self.assertFalse(result.success)
        self.assertEqual(result.failure_codes, ("forbidden_contact",))
        self.assertEqual(result.safety_codes, ("forbidden_contact",))
        self.assertEqual(result.reward, -1.0)
        self.assertIsNone(result.shaping_unclipped)
        self.assertEqual(result.reward_terms, ())

    def test_all_simultaneous_failure_codes_are_stable(self) -> None:
        contract, catalog, state = _context()
        result = evaluate_task_step(
            contract,
            state=state,
            frame=_frame(
                catalog,
                current=0.03,
                contact=True,
                displacement=True,
            ),
            catalog=catalog,
        )
        self.assertEqual(
            result.failure_codes,
            ("bottle_displacement", "forbidden_contact"),
        )
        self.assertEqual(result.safety_codes, result.failure_codes)

    def test_ordinary_failure_also_overrides_goal(self) -> None:
        task = task_record()
        task["terminal_rules"][0]["kind"] = "failure"
        capability = capability_record()
        capability["event_schemas"][0]["metrics"][1][
            "allowed_purposes"
        ] = ["failure", "process"]
        contract, catalog, state = _context(
            task_value=task,
            capability_value=capability,
        )
        result = evaluate_task_step(
            contract,
            state=state,
            frame=_frame(catalog, current=0.03, contact=True),
            catalog=catalog,
        )
        self.assertEqual(result.status, FAILURE)
        self.assertEqual(result.failure_codes, ("forbidden_contact",))
        self.assertEqual(result.safety_codes, ())
        self.assertTrue(result.goal_predicate)

    def test_weighted_shaping_uses_fixed_order_and_final_clip(self) -> None:
        task = task_record()
        duplicate = copy.deepcopy(task["reward_profiles"][0]["terms"][0])
        duplicate["term_id"] = "second_progress"
        task["reward_profiles"][0]["terms"].append(duplicate)
        contract, catalog, state = _context(task_value=task)
        result = evaluate_task_step(
            contract,
            state=state,
            frame=_frame(catalog, previous=0.10, current=0.07),
            catalog=catalog,
        )

        self.assertAlmostEqual(result.shaping_unclipped, 2.0)
        self.assertEqual(result.shaping_clipped, 1.0)
        self.assertEqual(result.reward, 1.0)
        self.assertEqual(
            tuple(item.term_id for item in result.reward_terms),
            ("distance_progress", "second_progress"),
        )

    def test_horizon_truncates_after_failure_and_success_checks(self) -> None:
        task = task_record()
        task["goal"]["hold_steps"] = 1
        capability = capability_record()
        capability["adapters"][0]["max_episode_control_steps"] = 1
        contract, catalog, state = _context(
            task_value=task,
            capability_value=capability,
        )
        truncated = evaluate_task_step(
            contract,
            state=state,
            frame=_frame(catalog, current=0.08),
            catalog=catalog,
        )
        self.assertEqual(truncated.status, TRUNCATED)
        self.assertTrue(truncated.truncated)
        self.assertIsNotNone(truncated.shaping_clipped)

        contract, catalog, state = _context(
            task_value=task,
            capability_value=capability,
        )
        success = evaluate_task_step(
            contract,
            state=state,
            frame=_frame(catalog, current=0.03),
            catalog=catalog,
        )
        self.assertEqual(success.status, SUCCESS)

        contract, catalog, state = _context(
            task_value=task,
            capability_value=capability,
        )
        failure = evaluate_task_step(
            contract,
            state=state,
            frame=_frame(catalog, current=0.03, contact=True),
            catalog=catalog,
        )
        self.assertEqual(failure.status, FAILURE)

    def test_terminal_state_cannot_receive_reward_twice(self) -> None:
        task = task_record()
        task["goal"]["hold_steps"] = 1
        contract, catalog, state = _context(task_value=task)
        terminal = evaluate_task_step(
            contract,
            state=state,
            frame=_frame(catalog, current=0.03),
            catalog=catalog,
        )
        with self.assertRaisesRegex(TaskEvaluationError, "Terminal runtime state"):
            evaluate_task_step(
                contract,
                state=terminal.next_state,
                frame=_frame(catalog, current=0.03),
                catalog=catalog,
            )

    def test_state_contract_profile_and_frame_mismatch_fail_closed(self) -> None:
        contract, catalog, state = _context()
        with self.assertRaisesRegex(TaskEvaluationError, "contract fingerprint"):
            evaluate_task_step(
                contract,
                state=replace(state, contract_fingerprint="0" * 64),
                frame=_frame(catalog),
                catalog=catalog,
            )
        with self.assertRaisesRegex(TaskEvaluationError, "reward profile"):
            evaluate_task_step(
                contract,
                state=replace(state, reward_profile_fingerprint="0" * 64),
                frame=_frame(catalog),
                catalog=catalog,
            )
        schema = catalog.event_schemas[0]
        invalid = MetricFrame(
            event_schema_id=schema.event_schema_id,
            event_schema_fingerprint=schema.fingerprint,
            previous={"reach.distance_m": 0.10},
            current={"reach.distance_m": 0.08},
        )
        with self.assertRaisesRegex(TaskEvaluationError, "evaluation failed"):
            evaluate_task_step(
                contract,
                state=state,
                frame=invalid,
                catalog=catalog,
            )

    def test_repeated_evaluation_is_deterministic_and_pure(self) -> None:
        contract, catalog, state = _context()
        frame = _frame(catalog)
        results = tuple(
            evaluate_task_step(
                contract,
                state=state,
                frame=frame,
                catalog=catalog,
            )
            for _ in range(2)
        )
        self.assertEqual(results[0], results[1])
        self.assertEqual(results[0].fingerprint, results[1].fingerprint)
        self.assertEqual(state.control_step, 0)
        self.assertEqual(frame.current["reach.distance_m"], 0.08)

    def test_reward_profile_choice_cannot_change_terminal_oracle(self) -> None:
        task = task_record()
        alternate = copy.deepcopy(task["reward_profiles"][0])
        alternate["reward_profile_id"] = "reach_progress_alt/v2"
        alternate["terms"][0]["weight"] = -1.0
        alternate["terminal_values"] = {"success": 2.0, "failure": -2.0}
        alternate["shaping_bounds"] = [-1.5, 1.5]
        task["reward_profiles"].append(alternate)
        catalog = CapabilityCatalog.from_record(capability_record())
        contract = compile_task_contract(
            TaskSpecV2.from_record(task),
            catalog=catalog,
        )
        frame = _frame(catalog, current=0.03, contact=True)
        outcomes = []
        for profile in contract.reward_profiles:
            state = initial_runtime_state(
                contract,
                reward_profile_id=profile.reward_profile_id,
                catalog=catalog,
            )
            outcomes.append(
                evaluate_task_step(
                    contract,
                    state=state,
                    frame=frame,
                    catalog=catalog,
                )
            )
        self.assertEqual(
            {(item.status, item.failure_codes) for item in outcomes},
            {(FAILURE, ("forbidden_contact",))},
        )
        self.assertEqual({item.reward for item in outcomes}, {-1.0, -2.0})


if __name__ == "__main__":
    unittest.main()
