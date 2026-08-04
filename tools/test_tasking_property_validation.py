"""Tests for deterministic authored-task property validation."""

from __future__ import annotations

import copy
import json
import unittest
from dataclasses import replace

from rexpolicy.tasking.capabilities import CapabilityCatalog
from rexpolicy.tasking.contract import compile_task_contract
from rexpolicy.tasking.model import TaskSpecV2
from rexpolicy.tasking.property_validation import (
    DEFAULT_PROPERTY_VALIDATION_POLICY,
    validate_task_properties,
)
from tools.test_tasking_capabilities import capability_record
from tools.test_tasking_model import task_record


def _validate(record=None):
    catalog = CapabilityCatalog.from_record(capability_record())
    task = TaskSpecV2.from_record(record if record is not None else task_record())
    contract = compile_task_contract(task, catalog=catalog)
    return validate_task_properties(
        task,
        contract,
        catalog=catalog,
    )


class TestTaskPropertyValidation(unittest.TestCase):
    def test_reach_validation_traces_pass_the_trusted_gate(self) -> None:
        reports = (_validate(), _validate())
        self.assertTrue(reports[0].accepted)
        self.assertEqual(reports[0].issues, ())
        self.assertEqual(reports[0].example_count, 4)
        self.assertEqual(reports[0].evaluated_step_count, 5)
        self.assertEqual(reports[0], reports[1])
        self.assertEqual(reports[0].fingerprint, reports[1].fingerprint)
        json.dumps(reports[0].to_record(), allow_nan=False)

    def test_safe_success_is_a_required_witness(self) -> None:
        record = task_record()
        record["validation_examples"] = [
            item
            for item in record["validation_examples"]
            if item["example_id"] != "safe_success"
        ]
        report = _validate(record)
        self.assertFalse(report.accepted)
        self.assertIn("safe_success_missing", {item.code for item in report.issues})

    def test_every_terminal_rule_needs_positive_and_negative_coverage(self) -> None:
        record = task_record()
        record["validation_examples"] = [
            item
            for item in record["validation_examples"]
            if item["example_id"] != "displacement_failure"
        ]
        report = _validate(record)
        uncovered = [
            item for item in report.issues if item.code == "terminal_rule_uncovered"
        ]
        self.assertEqual(len(uncovered), 1)
        self.assertIn("bottle_displacement", uncovered[0].detail)

    def test_expected_outcome_and_frame_errors_are_author_issues(self) -> None:
        mismatch = task_record()
        mismatch["validation_examples"][0]["expected_status"] = "success"
        report = _validate(mismatch)
        self.assertIn(
            "example_outcome_mismatch",
            {item.code for item in report.issues},
        )

        invalid = task_record()
        del invalid["validation_examples"][0]["steps"][0][
            "reach.contact_violation"
        ]
        report = _validate(invalid)
        self.assertIn(
            "example_replay_invalid",
            {item.code for item in report.issues},
        )

    def test_failure_precedence_requires_an_overlap_witness(self) -> None:
        record = task_record()
        contact = next(
            item
            for item in record["validation_examples"]
            if item["example_id"] == "contact_precedence"
        )
        contact["steps"][0]["reach.distance_m"] = 0.08
        report = _validate(record)
        self.assertIn(
            "failure_precedence_unproven",
            {item.code for item in report.issues},
        )

    def test_validation_limits_are_policy_bound(self) -> None:
        catalog = CapabilityCatalog.from_record(capability_record())
        task = TaskSpecV2.from_record(task_record())
        contract = compile_task_contract(task, catalog=catalog)
        policy = replace(
            DEFAULT_PROPERTY_VALIDATION_POLICY,
            max_examples=2,
            max_total_steps=2,
        )
        report = validate_task_properties(
            task,
            contract,
            catalog=catalog,
            policy=policy,
        )
        self.assertFalse(report.accepted)
        self.assertIn("example_limit_exceeded", {item.code for item in report.issues})
        self.assertLessEqual(report.evaluated_step_count, 2)

    def test_task_contract_integrity_mismatch_raises(self) -> None:
        catalog = CapabilityCatalog.from_record(capability_record())
        task = TaskSpecV2.from_record(task_record())
        contract = compile_task_contract(task, catalog=catalog)
        changed = copy.deepcopy(task_record())
        changed["goal"]["hold_steps"] = 3
        with self.assertRaisesRegex(ValueError, "fingerprint mismatch"):
            validate_task_properties(
                TaskSpecV2.from_record(changed),
                contract,
                catalog=catalog,
            )


if __name__ == "__main__":
    unittest.main()
