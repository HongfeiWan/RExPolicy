"""Tests for the strict Grasp-Lift pilot config contract."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from rexpolicy.stage0.grasp_lift_pilot import (
    GraspLiftPilotConfig,
    load_grasp_lift_pilot_config,
    load_grasp_lift_pilot_manifest,
    seal_grasp_lift_pilot_record,
    verify_grasp_lift_pilot_record,
)


ROOT = Path(__file__).resolve().parents[1]


class GraspLiftPilotConfigTest(unittest.TestCase):
    def test_repository_pilot_config_is_valid_and_hash_bound(self) -> None:
        config = load_grasp_lift_pilot_config(
            ROOT / "configs" / "stage0" / "grasp_lift_pilot.json"
        )
        self.assertEqual(config.runtime.cpu_threads, 16)
        self.assertEqual(config.runtime.num_envs, 8)
        self.assertEqual(config.runtime.reset_groups, 32)
        self.assertEqual(config.acceptance.locked_test_policy, "not_created")
        self.assertEqual(len(config.sha256), 64)
        self.assertEqual(
            config.sha256,
            GraspLiftPilotConfig.from_mapping(config.to_record()).sha256,
        )

    def test_unknown_fields_and_locked_test_creation_fail_closed(self) -> None:
        source = json.loads(
            (ROOT / "configs" / "stage0" / "grasp_lift_pilot.json").read_text()
        )
        source["runtime"]["surprise"] = 1
        with self.assertRaisesRegex(ValueError, "unexpected"):
            GraspLiftPilotConfig.from_mapping(source)
        source["runtime"].pop("surprise")
        source["acceptance"]["locked_test_policy"] = "create"
        with self.assertRaisesRegex(ValueError, "not_created"):
            GraspLiftPilotConfig.from_mapping(source)

    def test_reset_groups_must_match_world_batch(self) -> None:
        source = json.loads(
            (ROOT / "configs" / "stage0" / "grasp_lift_pilot.json").read_text()
        )
        source["runtime"]["reset_groups"] = 31
        with self.assertRaisesRegex(ValueError, "divisible"):
            GraspLiftPilotConfig.from_mapping(source)

    def test_validation_fraction_is_fixed_and_type_checked(self) -> None:
        source = json.loads(
            (ROOT / "configs" / "stage0" / "grasp_lift_pilot.json").read_text()
        )
        source["acceptance"]["validation_fraction"] = 0.2
        with self.assertRaisesRegex(ValueError, "fixed"):
            GraspLiftPilotConfig.from_mapping(source)
        source["acceptance"]["validation_fraction"] = "0.25"
        with self.assertRaisesRegex(ValueError, "numeric"):
            GraspLiftPilotConfig.from_mapping(source)

    def test_sealed_record_detects_tampering(self) -> None:
        record = seal_grasp_lift_pilot_record(
            {"schema_id": "test/pilot-record/v1", "value": 1}
        )
        verify_grasp_lift_pilot_record(record, "test record")
        record["value"] = 2
        with self.assertRaisesRegex(ValueError, "self_sha256 mismatch"):
            verify_grasp_lift_pilot_record(record, "test record")

    def test_loader_rejects_an_incomplete_publication(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "pilot commit"):
                load_grasp_lift_pilot_manifest(directory)

    def test_loader_rejects_non_json(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            path.write_text("not-json", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "cannot read"):
                load_grasp_lift_pilot_config(path)


if __name__ == "__main__":
    unittest.main()
