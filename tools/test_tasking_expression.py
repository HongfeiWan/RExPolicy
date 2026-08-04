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
    catalog = CapabilityCatalog.from_record(capability_record())
    schema = catalog.event_schemas[0]
    parameters = {
        item.name: item for item in catalog.adapters[0].parameters
    }
    return task, schema, parameters


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
        task, schema, parameters = _context()
        expression = task.reward_profiles[0].terms[0].expression
        program = compile_expression(
            expression,
            schema=schema,
            parameters=task.environment.parameters,
            parameter_capabilities=parameters,
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
        task, schema, parameters = _context()
        goal = compile_expression(
            task.goal.predicate,
            schema=schema,
            parameters=task.environment.parameters,
            parameter_capabilities=parameters,
            purpose="goal",
            task_spec_fingerprint=task.fingerprint,
        )
        safety = compile_expression(
            task.terminal_rules[0].predicate,
            schema=schema,
            parameters=task.environment.parameters,
            parameter_capabilities=parameters,
            purpose="safety",
            task_spec_fingerprint=task.fingerprint,
        )
        close = _frame(schema, current=0.03)
        self.assertIs(evaluate_expression(goal, frame=close, schema=schema), True)
        self.assertIs(evaluate_expression(safety, frame=close, schema=schema), False)

    def test_process_predicates_are_boolean_and_purpose_limited(self) -> None:
        task, schema, parameters = _context()
        process_expression = ExpressionSpec.from_record(
            {
                "op": "any",
                "args": [
                    {
                        "op": "metric",
                        "name": "reach.contact_violation",
                        "at": "current",
                    },
                    {
                        "op": "metric",
                        "name": "reach.displacement_violation",
                        "at": "current",
                    },
                ],
            },
            "$process",
        )
        program = compile_expression(
            process_expression,
            schema=schema,
            parameters=task.environment.parameters,
            parameter_capabilities=parameters,
            purpose="process",
            task_spec_fingerprint=task.fingerprint,
        )

        self.assertEqual(program.result_type, "boolean")
        self.assertIs(
            evaluate_expression(program, frame=_frame(schema), schema=schema),
            False,
        )
        with self.assertRaisesRegex(ValueError, "process expression must return boolean"):
            compile_expression(
                task.reward_profiles[0].terms[0].expression,
                schema=schema,
                parameters=task.environment.parameters,
                parameter_capabilities=parameters,
                purpose="process",
                task_spec_fingerprint=task.fingerprint,
            )

    def test_unknown_metric_time_and_operator_are_rejected(self) -> None:
        task, schema, parameters = _context()
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
                    parameter_capabilities=parameters,
                    purpose="reward",
                    task_spec_fingerprint=task.fingerprint,
                )

    def test_division_requires_proven_nonzero_constant(self) -> None:
        task, schema, parameters = _context()
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
                parameter_capabilities=parameters,
                purpose="reward",
                task_spec_fingerprint=task.fingerprint,
            )

    def test_metric_purpose_cannot_be_escalated_by_the_author(self) -> None:
        task, schema, parameters = _context()
        with self.assertRaisesRegex(ValueError, "reward expression cannot use"):
            compile_expression(
                task.terminal_rules[0].predicate,
                schema=schema,
                parameters=task.environment.parameters,
                parameter_capabilities=parameters,
                purpose="reward",
                task_spec_fingerprint=task.fingerprint,
            )

    def test_type_and_node_limits_fail_closed(self) -> None:
        task, schema, parameters = _context()
        with self.assertRaisesRegex(ValueError, "reward expression must return number"):
            compile_expression(
                task.goal.predicate,
                schema=schema,
                parameters=task.environment.parameters,
                parameter_capabilities=parameters,
                purpose="reward",
                task_spec_fingerprint=task.fingerprint,
            )
        with self.assertRaisesRegex(ValueError, "AST node limit"):
            compile_expression(
                task.reward_profiles[0].terms[0].expression,
                schema=schema,
                parameters=task.environment.parameters,
                parameter_capabilities=parameters,
                purpose="reward",
                task_spec_fingerprint=task.fingerprint,
                limits=CompileLimits(max_nodes=2),
            )

    def test_frame_schema_and_values_are_strict(self) -> None:
        task, schema, parameters = _context()
        program = compile_expression(
            task.goal.predicate,
            schema=schema,
            parameters=task.environment.parameters,
            parameter_capabilities=parameters,
            purpose="goal",
            task_spec_fingerprint=task.fingerprint,
        )
        missing = MetricFrame(
            event_schema_id=schema.event_schema_id,
            event_schema_fingerprint=schema.fingerprint,
            previous={"reach.distance_m": 0.10},
            current={
                "reach.distance_m": 0.08,
                "reach.displacement_violation": False,
            },
        )
        with self.assertRaisesRegex(ValueError, "fields do not match"):
            evaluate_expression(program, frame=missing, schema=schema)

        non_finite = MetricFrame(
            event_schema_id=schema.event_schema_id,
            event_schema_fingerprint=schema.fingerprint,
            previous={"reach.distance_m": 0.10},
            current={
                "reach.distance_m": float("nan"),
                "reach.contact_violation": False,
                "reach.displacement_violation": False,
            },
        )
        with self.assertRaisesRegex(ValueError, "finite"):
            evaluate_expression(program, frame=non_finite, schema=schema)

    def test_frame_and_instruction_operands_are_deeply_immutable(self) -> None:
        task, schema, parameters = _context()
        previous = {"reach.distance_m": 0.10}
        current = {
            "reach.distance_m": 0.08,
            "reach.contact_violation": False,
            "reach.displacement_violation": False,
        }
        frame = MetricFrame(
            event_schema_id=schema.event_schema_id,
            event_schema_fingerprint=schema.fingerprint,
            previous=previous,
            current=current,
        )
        program = compile_expression(
            task.goal.predicate,
            schema=schema,
            parameters=task.environment.parameters,
            parameter_capabilities=parameters,
            purpose="goal",
            task_spec_fingerprint=task.fingerprint,
        )
        current["reach.distance_m"] = 1.0

        self.assertEqual(frame.current["reach.distance_m"], 0.08)
        with self.assertRaises(TypeError):
            frame.current["reach.distance_m"] = 1.0
        with self.assertRaisesRegex(ValueError, "invalid operand"):
            type(program.instructions[0])("load_metric", {"name": "x"})

    def test_units_are_checked_conservatively(self) -> None:
        task, schema, parameters = _context()
        reward = compile_expression(
            task.reward_profiles[0].terms[0].expression,
            schema=schema,
            parameters=task.environment.parameters,
            parameter_capabilities=parameters,
            purpose="reward",
            task_spec_fingerprint=task.fingerprint,
        )
        self.assertEqual(reward.result_unit, "1")

        mixed = ExpressionSpec.from_record(
            {
                "op": "add",
                "args": [
                    {
                        "op": "metric",
                        "name": "reach.distance_m",
                        "at": "current",
                    },
                    {"op": "const", "value": 1.0},
                ],
            },
            "$expr",
        )
        with self.assertRaisesRegex(ValueError, "mixes units"):
            compile_expression(
                mixed,
                schema=schema,
                parameters=task.environment.parameters,
                parameter_capabilities=parameters,
                purpose="reward",
                task_spec_fingerprint=task.fingerprint,
            )

        raw_distance = ExpressionSpec.from_record(
            {
                "op": "metric",
                "name": "reach.distance_m",
                "at": "current",
            },
            "$expr",
        )
        with self.assertRaisesRegex(ValueError, "unitless"):
            compile_expression(
                raw_distance,
                schema=schema,
                parameters=task.environment.parameters,
                parameter_capabilities=parameters,
                purpose="reward",
                task_spec_fingerprint=task.fingerprint,
            )

    def test_program_fingerprint_is_stable_for_key_order(self) -> None:
        task, schema, parameters = _context()
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
                parameter_capabilities=parameters,
                purpose="goal",
                task_spec_fingerprint=task.fingerprint,
            )
            for record in (original, reordered)
        ]
        self.assertEqual(programs[0].fingerprint, programs[1].fingerprint)


if __name__ == "__main__":
    unittest.main()
