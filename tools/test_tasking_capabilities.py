"""Tests for trusted EventSchema and adapter capability resolution."""

from __future__ import annotations

import copy
import unittest

from rexpolicy.tasking.capabilities import (
    CapabilityCatalog,
    validate_task_capabilities,
)
from rexpolicy.tasking.model import TaskSpecV2
from tools.test_tasking_model import task_record


def capability_record() -> dict:
    return {
        "schema_version": 1,
        "event_schemas": [
            {
                "schema_version": 1,
                "event_schema_id": "groot_newton/reach_events/v1",
                "metrics": [
                    {
                        "name": "reach.distance_m",
                        "value_type": "number",
                        "minimum": 0.0,
                        "maximum": 2.0,
                        "unit": "m",
                        "available_at": ["previous", "current"],
                    },
                    {
                        "name": "reach.contact_violation",
                        "value_type": "boolean",
                        "minimum": None,
                        "maximum": None,
                        "unit": "bool",
                        "available_at": ["current"],
                    },
                    {
                        "name": "reach.displacement_violation",
                        "value_type": "boolean",
                        "minimum": None,
                        "maximum": None,
                        "unit": "bool",
                        "available_at": ["current"],
                    },
                ],
            }
        ],
        "adapters": [
            {
                "adapter_id": "groot_newton/reach/v1",
                "task_modes": ["reach_green_cap"],
                "event_schema_id": "groot_newton/reach_events/v1",
                "process_spec_ids": ["reach_pregrasp/v1"],
                "allowed_assets": ["nero_bottle/v1"],
                "required_assets": ["nero_bottle/v1"],
                "bindings": {"target": ["green_cap"]},
                "parameters": [
                    {
                        "name": "success_distance_m",
                        "value_type": "number",
                        "minimum": 0.01,
                        "maximum": 0.10,
                        "required": True,
                        "vector_length": None,
                    },
                    {
                        "name": "reward_distance_scale_m",
                        "value_type": "number",
                        "minimum": 0.01,
                        "maximum": 0.10,
                        "required": True,
                        "vector_length": None,
                    },
                    {
                        "name": "goal_offset_world_m",
                        "value_type": "number_vector",
                        "minimum": -1.0,
                        "maximum": 1.0,
                        "required": True,
                        "vector_length": 3,
                    },
                ],
                "action_projection": copy.deepcopy(
                    task_record()["action_projection"]
                ),
            }
        ],
    }


class TestTaskCapabilities(unittest.TestCase):
    def test_resolves_exact_certified_adapter_contract(self) -> None:
        catalog = CapabilityCatalog.from_record(capability_record())
        task = TaskSpecV2.from_record(task_record())
        resolved = validate_task_capabilities(task, catalog)

        self.assertEqual(resolved.adapter.adapter_id, task.environment.adapter_id)
        self.assertEqual(
            resolved.event_schema.event_schema_id,
            task.event_schema_id,
        )
        self.assertEqual(len(catalog.fingerprint), 64)
        self.assertEqual(len(resolved.adapter.fingerprint), 64)
        self.assertEqual(len(resolved.event_schema.fingerprint), 64)

    def test_author_cannot_change_certified_action_mask(self) -> None:
        record = task_record()
        record["action_projection"]["eef_9d"][3] = True
        with self.assertRaisesRegex(ValueError, "certified adapter mask"):
            validate_task_capabilities(
                TaskSpecV2.from_record(record),
                CapabilityCatalog.from_record(capability_record()),
            )

    def test_unknown_process_asset_and_binding_are_rejected(self) -> None:
        changes = (
            ("process_spec_id", "unknown_process/v1", "process spec"),
            ("required_assets", ["unknown_asset/v1"], "required asset"),
        )
        for field, value, message in changes:
            record = task_record()
            if field == "required_assets":
                record["environment"][field] = value
            else:
                record[field] = value
            with self.assertRaisesRegex(ValueError, message):
                validate_task_capabilities(
                    TaskSpecV2.from_record(record),
                    CapabilityCatalog.from_record(capability_record()),
                )

        record = task_record()
        record["environment"]["bindings"]["target"] = "red_cap"
        with self.assertRaisesRegex(ValueError, "outside the allowlist"):
            validate_task_capabilities(
                TaskSpecV2.from_record(record),
                CapabilityCatalog.from_record(capability_record()),
            )

    def test_parameter_range_and_unknown_parameter_are_rejected(self) -> None:
        record = task_record()
        record["environment"]["parameters"]["success_distance_m"] = 0.5
        with self.assertRaisesRegex(ValueError, "capability maximum"):
            validate_task_capabilities(
                TaskSpecV2.from_record(record),
                CapabilityCatalog.from_record(capability_record()),
            )

        record = task_record()
        record["environment"]["parameters"]["made_up_force"] = 1.0
        with self.assertRaisesRegex(ValueError, "outside the adapter allowlist"):
            validate_task_capabilities(
                TaskSpecV2.from_record(record),
                CapabilityCatalog.from_record(capability_record()),
            )

        record = task_record()
        record["environment"]["parameters"]["goal_offset_world_m"] = [0.1, 0.2]
        with self.assertRaisesRegex(ValueError, "wrong vector length"):
            validate_task_capabilities(
                TaskSpecV2.from_record(record),
                CapabilityCatalog.from_record(capability_record()),
            )

    def test_duplicate_metric_and_unknown_schema_are_rejected(self) -> None:
        record = capability_record()
        record["event_schemas"][0]["metrics"].append(
            copy.deepcopy(record["event_schemas"][0]["metrics"][0])
        )
        with self.assertRaisesRegex(ValueError, "metric names must be unique"):
            CapabilityCatalog.from_record(record)

        record = capability_record()
        record["adapters"][0]["event_schema_id"] = "missing/events/v1"
        with self.assertRaisesRegex(ValueError, "unknown event schema"):
            CapabilityCatalog.from_record(record)


if __name__ == "__main__":
    unittest.main()
