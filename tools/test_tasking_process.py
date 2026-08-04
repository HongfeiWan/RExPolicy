"""Tests for strict deterministic ProcessSpec value objects."""

from __future__ import annotations

import json
import unittest

from rexpolicy.tasking.process import ProcessSpecV1


def process_record() -> dict:
    return {
        "schema_version": 1,
        "process_spec_id": "reach_pregrasp/v1",
        "task_id": "reach_green_cap/v2",
        "event_schema_id": "groot_newton/reach_events/v1",
        "default_stage": {
            "stage_id": "far",
            "kind": "progress",
            "ordinal": 0,
            "description": "The hand remains outside the approach region.",
        },
        "stages": [
            {
                "stage_id": "unsafe",
                "kind": "unsafe",
                "ordinal": None,
                "description": "A certified safety predicate fired.",
                "predicate": {
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
            },
            {
                "stage_id": "goal_zone",
                "kind": "progress",
                "ordinal": 2,
                "description": "The hand is inside the certified goal region.",
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
            {
                "stage_id": "approach",
                "kind": "progress",
                "ordinal": 1,
                "description": "The hand is in the approach region.",
                "predicate": {
                    "op": "le",
                    "args": [
                        {
                            "op": "metric",
                            "name": "reach.distance_m",
                            "at": "current",
                        },
                        {
                            "op": "mul",
                            "args": [
                                {"op": "const", "value": 2.5},
                                {"op": "parameter", "name": "success_distance_m"},
                            ],
                        },
                    ],
                },
            },
        ],
        "segmentation": {"max_segment_steps": 4},
    }


class TestProcessSpec(unittest.TestCase):
    def test_round_trip_and_fingerprint_are_stable(self) -> None:
        spec = ProcessSpecV1.from_record(process_record())
        restored = ProcessSpecV1.from_json(spec.to_json())

        self.assertEqual(restored, spec)
        self.assertEqual(restored.fingerprint, spec.fingerprint)
        self.assertEqual(spec.stages[0].stage_id, "unsafe")
        self.assertEqual(spec.default_stage.ordinal, 0)
        json.loads(spec.to_json())

    def test_priority_order_changes_the_identity(self) -> None:
        original = ProcessSpecV1.from_record(process_record())
        changed_record = process_record()
        changed_record["stages"][1], changed_record["stages"][2] = (
            changed_record["stages"][2],
            changed_record["stages"][1],
        )
        changed = ProcessSpecV1.from_record(changed_record)

        self.assertNotEqual(original.fingerprint, changed.fingerprint)

    def test_stage_ids_and_progress_ordinals_are_strict(self) -> None:
        duplicate_id = process_record()
        duplicate_id["stages"][1]["stage_id"] = "far"
        with self.assertRaisesRegex(ValueError, "stage IDs must be unique"):
            ProcessSpecV1.from_record(duplicate_id)

        ordinal_gap = process_record()
        ordinal_gap["stages"][2]["ordinal"] = 3
        with self.assertRaisesRegex(ValueError, "contiguous from zero"):
            ProcessSpecV1.from_record(ordinal_gap)

        unsafe_ordinal = process_record()
        unsafe_ordinal["stages"][0]["ordinal"] = 9
        with self.assertRaisesRegex(ValueError, "must be null"):
            ProcessSpecV1.from_record(unsafe_ordinal)

        unsafe_after_progress = process_record()
        unsafe_after_progress["stages"].append(
            unsafe_after_progress["stages"].pop(0)
        )
        with self.assertRaisesRegex(ValueError, "unsafe stages must precede"):
            ProcessSpecV1.from_record(unsafe_after_progress)

        nonzero_default = process_record()
        nonzero_default["default_stage"]["ordinal"] = 3
        with self.assertRaisesRegex(ValueError, "default progress ordinal"):
            ProcessSpecV1.from_record(nonzero_default)

    def test_unknown_fields_and_unbounded_segments_fail_closed(self) -> None:
        unknown = process_record()
        unknown["reward"] = 1.0
        with self.assertRaisesRegex(ValueError, "unknown fields: reward"):
            ProcessSpecV1.from_record(unknown)

        unbounded = process_record()
        unbounded["segmentation"]["max_segment_steps"] = 257
        with self.assertRaisesRegex(ValueError, "between 1 and 256"):
            ProcessSpecV1.from_record(unbounded)

    def test_nested_predicates_are_immutable(self) -> None:
        record = process_record()
        spec = ProcessSpecV1.from_record(record)
        record["stages"][0]["predicate"]["args"][0]["name"] = "changed"

        self.assertEqual(
            spec.stages[0].predicate.payload["args"][0].payload["name"],
            "reach.contact_violation",
        )
        with self.assertRaises(TypeError):
            spec.stages[0].predicate.payload["args"] = ()

    def test_duplicate_json_keys_are_rejected(self) -> None:
        payload = json.dumps(process_record())
        duplicate = payload.replace(
            '"schema_version": 1',
            '"schema_version": 1, "schema_version": 1',
            1,
        )
        with self.assertRaisesRegex(ValueError, "Duplicate JSON key"):
            ProcessSpecV1.from_json(duplicate)


if __name__ == "__main__":
    unittest.main()
