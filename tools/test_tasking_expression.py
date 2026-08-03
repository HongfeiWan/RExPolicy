"""Tests for safe TaskSpec expression compilation and evaluation."""

from __future__ import annotations

import copy
import unittest

from rexpolicy.tasking.capabilities import CapabilityCatalog
from rexpolicy.tasking.expression import (
    CompileLimits,
    MetricFrame,
    compile_expression,
    evaluate_expression,
)
from rexpolicy.tasking.model import ExpressionSpec, TaskSpecV2
from tools.test_tasking_capabilities import capability_record
from tools.test_tasking_model import task_record


def _context():
    task = TaskSpecV2.from_record(task_record())
    schema = CapabilityCatalog.from_record(capability_record()).event_schemas[0]
    return task, schema


def _frame(schema, *, previous: float = 0.10, current: float = 0.08):
    return MetricFrame(
        event_schema_id=schema.event_schema_id,
        event_schema_fingerprint=schema.fingerprint,
        previous={"reach.distance_m": previous},
        current={
            "reach.distance_m": current,
            "reach.contact_violation": False,
            "reach.displacement_violation": False,
        },
    )


class TestTaskExpressionCompiler(unittest.TestCase):
    def test_reach_progress_compiles_and_matches_expected_reward(self) -> None:
        task, schema = _context()
        expression = task.reward_profiles[0].terms[0].expression
        program = compile_expression(
            expression,
            schema=schema,
            parameters=task.environment.parameters,
            purpose="reward",
            task_spec_fingerprint=task.fingerprint,
        )
        reward = evaluate_expression(program, frame=_frame(schema), schema=schema)

        self.assertAlmostEqual(reward, 2.0 / 3.0)
        self.assertEqual(program.referenced_metrics, ("reach.distance_m",))
        self.assertEqual(program.result_interval.minimum, -1.0)
        self.assertEqual(program.result_interval.maximum, 1.0)
        self.assertEqual(len(program.fingerprint), 64)

    def test_goal_and_safety_predicates_are_boolean(self) -> None:
        task, schema = _context()
        goal = compile_expression(
            task.goal.predicate,
            schema=schema,
            parameters=task.environment.parameters,
            purpose="goal",
            task_spec_fingerprint=task.fingerprint,
        )
        safety = compile_expression(
            task.terminal_rules[0].predicate,
            schema=schema,
            parameters=task.environment.parameters,
            purpose="terminal",
            task_spec_fingerprint=task.fingerprint,
        )
        close = _frame(schema, current=0.03)
        self.assertIs(evaluate_expression(goal, frame=close, schema=schema), True)
        self.assertIs(evaluate_expression(safety, frame=close, schema=schema), False)

    def test_unknown_metric_time_and_operator_are_rejected(self) -> None:
        task, schema = _context()
        cases = (
            (
                {"op": "metric", "name": "unknown.metric", "at": "current"},
                "unknown metric",
            ),
            (
                {
                    "op": "metric",
                    "name": "reach.contact_violation",
                    "at": "previous",
                },
                "unavailable",
            ),
            ({"op": "call", "args": []}, "Unsupported expression op"),
        )
        for record, message in cases:
            with self.assertRaisesRegex(ValueError, message):
                compile_expression(
                    ExpressionSpec.from_record(record, "$expr"),
                    schema=schema,
                    parameters=task.environment.parameters,
                    purpose="reward",
                    task_spec_fingerprint=task.fingerprint,
                )

    def test_division_requires_proven_nonzero_constant(self) -> None:
        task, schema = _context()
        expression = ExpressionSpec.from_record(
            {
                "op": "div",
                "args": [
                    {"op": "const", "value": 1.0},
                    {
                        "op": "metric",
                        "name": "reach.distance_m",
                        "at": "current",
                    },
                ],
            },
            "$expr",
        )
        with self.assertRaisesRegex(ValueError, "constant divisor"):
            compile_expression(
                expression,
                schema=schema,
                parameters=task.environment.parameters,
                purpose="reward",
                task_spec_fingerprint=task.fingerprint,
            )

    def test_metric_purpose_cannot_be_escalated_by_the_author(self) -> None:
        task, schema = _context()
        with self.assertRaisesRegex(ValueError, "reward expression cannot use"):
            compile_expression(
                task.terminal_rules[0].predicate,
                schema=schema,
                parameters=task.environment.parameters,
                purpose="reward",
                task_spec_fingerprint=task.fingerprint,
            )

    def test_type_and_node_limits_fail_closed(self) -> None:
        task, schema = _context()
        with self.assertRaisesRegex(ValueError, "reward expression must return number"):
            compile_expression(
                task.goal.predicate,
                schema=schema,
                parameters=task.environment.parameters,
                purpose="reward",
                task_spec_fingerprint=task.fingerprint,
            )
        with self.assertRaisesRegex(ValueError, "AST node limit"):
            compile_expression(
                task.reward_profiles[0].terms[0].expression,
                schema=schema,
                parameters=task.environment.parameters,
                purpose="reward",
                task_spec_fingerprint=task.fingerprint,
                limits=CompileLimits(max_nodes=2),
            )

    def test_frame_schema_and_values_are_strict(self) -> None:
        task, schema = _context()
        program = compile_expression(
            task.goal.predicate,
            schema=schema,
            parameters=task.environment.parameters,
            purpose="goal",
            task_spec_fingerprint=task.fingerprint,
        )
        missing = _frame(schema)
        missing.current.pop("reach.contact_violation")
        with self.assertRaisesRegex(ValueError, "fields do not match"):
            evaluate_expression(program, frame=missing, schema=schema)

        non_finite = _frame(schema)
        non_finite.current["reach.distance_m"] = float("nan")
        with self.assertRaisesRegex(ValueError, "finite"):
            evaluate_expression(program, frame=non_finite, schema=schema)

    def test_program_fingerprint_is_stable_for_key_order(self) -> None:
        task, schema = _context()
        original = task_record()["goal"]["predicate"]
        reordered = {
            "args": copy.deepcopy(original["args"]),
            "op": original["op"],
        }
        programs = [
            compile_expression(
                ExpressionSpec.from_record(record, "$expr"),
                schema=schema,
                parameters=task.environment.parameters,
                purpose="goal",
                task_spec_fingerprint=task.fingerprint,
            )
            for record in (original, reordered)
        ]
        self.assertEqual(programs[0].fingerprint, programs[1].fingerprint)


if __name__ == "__main__":
    unittest.main()
