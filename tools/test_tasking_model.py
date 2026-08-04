"""Tests for strict canonical TaskSpec v2 value objects."""

from __future__ import annotations

import copy
import json
import unittest

from rexpolicy.tasking.canonical import strict_json_loads
from rexpolicy.tasking.model import TaskSpecV2, decode_task_spec_json


def _expression_progress() -> dict:
    return {
        "op": "clip",
        "args": [
            {
                "op": "div",
                "args": [
                    {
                        "op": "sub",
                        "args": [
                            {
                                "op": "metric",
                                "name": "reach.distance_m",
                                "at": "previous",
                            },
                            {
                                "op": "metric",
                                "name": "reach.distance_m",
                                "at": "current",
                            },
                        ],
                    },
                    {"op": "parameter", "name": "reward_distance_scale_m"},
                ],
            },
            {"op": "const", "value": -1.0},
            {"op": "const", "value": 1.0},
        ],
    }


def task_record() -> dict:
    return {
        "schema_version": 2,
        "task_id": "reach_green_cap/v2",
        "version": 2,
        "family_id": "reach_green_cap",
        "event_schema_id": "groot_newton/reach_events/v1",
        "process_spec_id": "reach_pregrasp/v1",
        "curriculum_prerequisite_ids": [],
        "environment": {
            "adapter_id": "groot_newton/reach/v1",
            "task_mode": "reach_green_cap",
            "episode_control_steps": 8,
            "bindings": {"target": "green_cap"},
            "parameters": {
                "success_distance_m": 0.04,
                "reward_distance_scale_m": 0.03,
                "goal_offset_world_m": [0.18, 0.08, 0.05],
            },
            "required_assets": ["nero_bottle/v1"],
        },
        "instructions": {
            "train": ["Move the open right hand toward the green cap."],
            "promotion": ["Approach the green bottle cap without touching it."],
            "audit": {"count": 8, "sha256": "a" * 64},
        },
        "action_projection": {
            "eef_9d": [True, True, True, False, False, False, False, False, False],
            "hand_joint_target": [False] * 10,
            "arm_joint_target": [False] * 7,
        },
        "goal": {
            "hold_steps": 2,
            "predicate": {
                "op": "le",
                "args": [
                    {
                        "op": "metric",
                        "name": "reach.distance_m",
                        "at": "current",
                    },
                    {"op": "parameter", "name": "success_distance_m"},
                ],
            },
        },
        "terminal_rules": [
            {
                "code": "forbidden_contact",
                "kind": "safety",
                "predicate": {
                    "op": "metric",
                    "name": "reach.contact_violation",
                    "at": "current",
                },
            },
            {
                "code": "bottle_displacement",
                "kind": "safety",
                "predicate": {
                    "op": "metric",
                    "name": "reach.displacement_violation",
                    "at": "current",
                },
            },
        ],
        "reward_profiles": [
            {
                "reward_profile_id": "reach_progress/v2",
                "terms": [
                    {
                        "term_id": "distance_progress",
                        "weight": 1.0,
                        "expression": _expression_progress(),
                    }
                ],
                "shaping_bounds": [-1.0, 1.0],
                "terminal_values": {"success": 1.0, "failure": -1.0},
            }
        ],
        "validation_examples": [
            {
                "example_id": "active_progress",
                "initial_metrics": {"reach.distance_m": 0.10},
                "steps": [
                    {
                        "reach.distance_m": 0.08,
                        "reach.contact_violation": False,
                        "reach.displacement_violation": False,
                    }
                ],
                "reward_profile_id": "reach_progress/v2",
                "expected_status": "active",
                "expected_failure_codes": [],
                "expected_safety_codes": [],
            },
            {
                "example_id": "safe_success",
                "initial_metrics": {"reach.distance_m": 0.05},
                "steps": [
                    {
                        "reach.distance_m": 0.03,
                        "reach.contact_violation": False,
                        "reach.displacement_violation": False,
                    },
                    {
                        "reach.distance_m": 0.03,
                        "reach.contact_violation": False,
                        "reach.displacement_violation": False,
                    },
                ],
                "reward_profile_id": "reach_progress/v2",
                "expected_status": "success",
                "expected_failure_codes": [],
                "expected_safety_codes": [],
            },
            {
                "example_id": "contact_precedence",
                "initial_metrics": {"reach.distance_m": 0.05},
                "steps": [
                    {
                        "reach.distance_m": 0.03,
                        "reach.contact_violation": True,
                        "reach.displacement_violation": False,
                    }
                ],
                "reward_profile_id": "reach_progress/v2",
                "expected_status": "failure",
                "expected_failure_codes": ["forbidden_contact"],
                "expected_safety_codes": ["forbidden_contact"],
            },
            {
                "example_id": "displacement_failure",
                "initial_metrics": {"reach.distance_m": 0.10},
                "steps": [
                    {
                        "reach.distance_m": 0.08,
                        "reach.contact_violation": False,
                        "reach.displacement_violation": True,
                    }
                ],
                "reward_profile_id": "reach_progress/v2",
                "expected_status": "failure",
                "expected_failure_codes": ["bottle_displacement"],
                "expected_safety_codes": ["bottle_displacement"],
            },
        ],
    }


class TestTaskSpecModel(unittest.TestCase):
    def test_round_trip_and_fingerprint_are_canonical(self) -> None:
        record = task_record()
        first = TaskSpecV2.from_record(record)
        second = decode_task_spec_json(
            json.dumps(record, indent=2, sort_keys=False)
        )
        third = decode_task_spec_json(first.to_json())

        self.assertEqual(first, second)
        self.assertEqual(first, third)
        self.assertEqual(first.fingerprint, second.fingerprint)
        self.assertEqual(len(first.fingerprint), 64)
        self.assertNotIn(
            "Move the open",
            json.dumps(first.instructions.audit.to_record()),
        )

    def test_duplicate_and_non_finite_json_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Duplicate JSON key"):
            strict_json_loads('{"schema_version":2,"schema_version":2}')
        with self.assertRaisesRegex(ValueError, "Non-finite"):
            strict_json_loads('{"value":NaN}')

    def test_unknown_nested_field_is_rejected(self) -> None:
        record = task_record()
        record["goal"]["predicate"]["unexpected"] = True
        with self.assertRaisesRegex(ValueError, "unknown fields"):
            TaskSpecV2.from_record(record)

    def test_audit_text_cannot_cross_the_task_spec_boundary(self) -> None:
        record = task_record()
        record["instructions"]["audit"]["texts"] = ["secret audit prompt"]
        with self.assertRaisesRegex(ValueError, "unknown fields"):
            TaskSpecV2.from_record(record)

    def test_instruction_overlap_and_control_characters_are_rejected(self) -> None:
        overlap = task_record()
        overlap["instructions"]["promotion"] = list(
            overlap["instructions"]["train"]
        )
        with self.assertRaisesRegex(ValueError, "must be unique"):
            TaskSpecV2.from_record(overlap)

        control = task_record()
        control["instructions"]["train"] = ["unsafe\nsecond line"]
        with self.assertRaisesRegex(ValueError, "control characters"):
            TaskSpecV2.from_record(control)

    def test_fingerprint_changes_for_semantic_change(self) -> None:
        first_record = task_record()
        second_record = copy.deepcopy(first_record)
        second_record["environment"]["parameters"]["success_distance_m"] = 0.05
        self.assertNotEqual(
            TaskSpecV2.from_record(first_record).fingerprint,
            TaskSpecV2.from_record(second_record).fingerprint,
        )

    def test_task_id_and_version_cannot_diverge(self) -> None:
        record = task_record()
        record["version"] = 3
        with self.assertRaisesRegex(ValueError, "version suffix"):
            TaskSpecV2.from_record(record)

    def test_action_projection_requires_actual_booleans(self) -> None:
        record = task_record()
        record["action_projection"]["eef_9d"][0] = 1
        with self.assertRaisesRegex(ValueError, "must be Boolean"):
            TaskSpecV2.from_record(record)

    def test_nested_task_values_are_copied_and_deeply_immutable(self) -> None:
        record = task_record()
        task = TaskSpecV2.from_record(record)
        fingerprint = task.fingerprint
        record["environment"]["parameters"]["success_distance_m"] = 0.09

        self.assertEqual(task.fingerprint, fingerprint)
        with self.assertRaises(TypeError):
            task.environment.parameters["success_distance_m"] = 0.09
        with self.assertRaises(TypeError):
            task.environment.parameters["goal_offset_world_m"][0] = 0.0
        with self.assertRaises(TypeError):
            task.goal.predicate.payload["args"] = ()
        with self.assertRaises(TypeError):
            task.reward_profiles[0].terminal_values["success"] = 2.0
        with self.assertRaises(TypeError):
            task.validation_examples[0].steps[0]["reach.distance_m"] = 0.0

    def test_validation_examples_are_bounded_replay_traces(self) -> None:
        empty = task_record()
        empty["validation_examples"][0]["steps"] = []
        with self.assertRaisesRegex(ValueError, "non-empty array"):
            TaskSpecV2.from_record(empty)

        status = task_record()
        status["validation_examples"][0]["expected_status"] = "maybe"
        with self.assertRaisesRegex(ValueError, "unsupported"):
            TaskSpecV2.from_record(status)

        codes = task_record()
        codes["validation_examples"][2]["expected_safety_codes"] = [
            "made_up_rule"
        ]
        with self.assertRaisesRegex(ValueError, "also be failure"):
            TaskSpecV2.from_record(codes)


if __name__ == "__main__":
    unittest.main()
