"""Tests for reusable, strictly bound ProcessTemplate artifacts."""

from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path

from rexpolicy.tasking.process_template import (
    ProcessTemplateV1,
    instantiate_process_spec,
)

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE_PATH = ROOT / "configs/process_templates/reach_pregrasp.v1.json"
PROCESS_SPEC_PATH = ROOT / "configs/process_specs/reach_pregrasp.v2.json"


def production_record() -> dict:
    return json.loads(TEMPLATE_PATH.read_text(encoding="utf-8"))


class TestProcessTemplateV1(unittest.TestCase):
    def test_round_trip_and_fingerprint_are_deterministic(self) -> None:
        template = ProcessTemplateV1.from_record(production_record())
        restored = ProcessTemplateV1.from_json(template.to_json())

        self.assertEqual(restored, template)
        self.assertEqual(restored.to_record(), template.to_record())
        self.assertEqual(restored.fingerprint, template.fingerprint)
        self.assertEqual(len(template.fingerprint), 64)
        json.loads(template.to_json())

    def test_template_is_task_agnostic_and_strict(self) -> None:
        with_task_id = production_record()
        with_task_id["task_id"] = "reach_green_cap/v3"
        with self.assertRaisesRegex(ValueError, "unknown fields: task_id"):
            ProcessTemplateV1.from_record(with_task_id)

        with_process_id = production_record()
        with_process_id["process_spec_id"] = "reach_pregrasp/v2"
        with self.assertRaisesRegex(ValueError, "unknown fields: process_spec_id"):
            ProcessTemplateV1.from_record(with_process_id)

        unknown = production_record()
        unknown["reward_profile_id"] = "forbidden/v1"
        with self.assertRaisesRegex(ValueError, "unknown fields: reward_profile_id"):
            ProcessTemplateV1.from_record(unknown)

    def test_process_spec_stage_semantics_are_reused(self) -> None:
        unsafe_after_progress = production_record()
        unsafe_after_progress["stages"].append(
            unsafe_after_progress["stages"].pop(0)
        )
        with self.assertRaisesRegex(ValueError, "unsafe stages must precede"):
            ProcessTemplateV1.from_record(unsafe_after_progress)

        ordinal_gap = production_record()
        ordinal_gap["stages"][2]["ordinal"] = 3
        with self.assertRaisesRegex(ValueError, "contiguous from zero"):
            ProcessTemplateV1.from_record(ordinal_gap)

        unbounded = production_record()
        unbounded["segmentation"]["max_segment_steps"] = 257
        with self.assertRaisesRegex(ValueError, "between 1 and 256"):
            ProcessTemplateV1.from_record(unbounded)

    def test_template_identity_drift_fails_closed(self) -> None:
        record = production_record()
        original = ProcessTemplateV1.from_record(record)
        record["template_id"] = "reach_pregrasp/v2"
        drifted = ProcessTemplateV1.from_record(record)

        self.assertNotEqual(original.fingerprint, drifted.fingerprint)
        with self.assertRaisesRegex(ValueError, "template identity binding mismatch"):
            instantiate_process_spec(
                drifted,
                "reach_green_cap/v3",
                "reach_pregrasp/v2",
                expected_template_id="reach_pregrasp/v1",
            )

        content_drift = production_record()
        content_drift["segmentation"]["max_segment_steps"] = 5
        with self.assertRaisesRegex(ValueError, "fingerprint mismatch"):
            ProcessTemplateV1.from_record(
                content_drift,
                expected_fingerprint=original.fingerprint,
            )

    def test_family_and_event_schema_bindings_fail_closed(self) -> None:
        template = ProcessTemplateV1.from_record(production_record())

        with self.assertRaisesRegex(ValueError, "family binding mismatch"):
            instantiate_process_spec(
                template,
                "reach_green_cap/v3",
                "reach_pregrasp/v2",
                expected_family_id="another_family",
            )
        with self.assertRaisesRegex(ValueError, "EventSchema binding mismatch"):
            instantiate_process_spec(
                template,
                "reach_green_cap/v3",
                "reach_pregrasp/v2",
                expected_event_schema_id="groot_newton/other_events/v1",
            )

    def test_target_ids_are_caller_selected_and_validated(self) -> None:
        template = ProcessTemplateV1.from_record(production_record())
        first = instantiate_process_spec(
            template,
            "reach_green_cap/v3",
            "reach_pregrasp/v2",
        )
        second = instantiate_process_spec(
            template,
            "reach_green_cap/v4",
            "reach_pregrasp/v3",
        )

        self.assertEqual(first.task_id, "reach_green_cap/v3")
        self.assertEqual(first.process_spec_id, "reach_pregrasp/v2")
        self.assertEqual(second.task_id, "reach_green_cap/v4")
        self.assertEqual(second.process_spec_id, "reach_pregrasp/v3")
        self.assertEqual(first.event_schema_id, second.event_schema_id)
        self.assertEqual(first.stages, second.stages)
        with self.assertRaisesRegex(ValueError, "task_id"):
            instantiate_process_spec(
                template,
                "Reach Green Cap v3",
                "reach_pregrasp/v2",
            )
        with self.assertRaisesRegex(ValueError, "process_spec_id"):
            instantiate_process_spec(
                template,
                "reach_green_cap/v3",
                "REACH_PREGRASP/v2",
            )

    def test_production_template_materializes_exact_v2_spec(self) -> None:
        template = ProcessTemplateV1.from_json(
            TEMPLATE_PATH.read_text(encoding="utf-8")
        )
        actual = instantiate_process_spec(
            template,
            "reach_green_cap/v3",
            "reach_pregrasp/v2",
            expected_template_id="reach_pregrasp/v1",
            expected_family_id="reach_green_cap",
            expected_event_schema_id="groot_newton/reach_events/v1",
        )
        expected = json.loads(PROCESS_SPEC_PATH.read_text(encoding="utf-8"))

        self.assertEqual(actual.to_record(), expected)
        self.assertEqual(
            actual.to_json(),
            json.dumps(
                expected,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
            ),
        )

    def test_input_mutation_and_json_tampering_are_rejected(self) -> None:
        record = production_record()
        template = ProcessTemplateV1.from_record(record)
        original_fingerprint = template.fingerprint
        record["stages"][0]["predicate"]["args"][0]["name"] = "tampered"

        self.assertEqual(template.fingerprint, original_fingerprint)
        self.assertEqual(
            template.stages[0].predicate.payload["args"][0].payload["name"],
            "reach.contact_violation",
        )

        payload = json.dumps(production_record())
        duplicate = payload.replace(
            '"template_id": "reach_pregrasp/v1"',
            '"template_id": "reach_pregrasp/v1", '
            '"template_id": "reach_pregrasp/v2"',
            1,
        )
        with self.assertRaisesRegex(ValueError, "Duplicate JSON key"):
            ProcessTemplateV1.from_json(duplicate)

        missing = copy.deepcopy(production_record())
        del missing["family_id"]
        with self.assertRaisesRegex(ValueError, "missing fields: family_id"):
            ProcessTemplateV1.from_record(missing)


if __name__ == "__main__":
    unittest.main()
