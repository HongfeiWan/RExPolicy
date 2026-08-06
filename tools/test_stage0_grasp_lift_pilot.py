"""Tests for the strict Grasp-Lift pilot config contract."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from rexpolicy.stage0.grasp_lift_pilot import (
    GRASP_LIFT_PILOT_COMMIT_SCHEMA_ID,
    GRASP_LIFT_PILOT_OUTPUT_SCHEMA_ID,
    GRASP_LIFT_PILOT_STATUS_SCHEMA_ID,
    GraspLiftPilotConfig,
    assign_grasp_lift_pilot_splits,
    grasp_lift_pilot_split_policy,
    load_grasp_lift_pilot_config,
    load_grasp_lift_pilot_manifest,
    seal_grasp_lift_pilot_record,
)
from rexpolicy.stage0.types import canonical_fingerprint


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

    def test_loader_verifies_commit_chain_and_detects_manifest_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            split_policy = grasp_lift_pilot_split_policy()
            splits = assign_grasp_lift_pilot_splits(
                {
                    "trajectory-a": "reset-a",
                    "trajectory-b": "reset-b",
                },
                validation_fraction=0.25,
            )
            manifest = seal_grasp_lift_pilot_record(
                {
                    "acceptance": {"passed": True},
                    "locked_test": {
                        "artifact": None,
                        "consumed": False,
                        "policy": "not_created",
                    },
                    "members": [
                        {
                            "reset_group_id": "reset-a",
                            "split": splits["trajectory-a"],
                            "trajectory_id": "trajectory-a",
                        },
                        {
                            "reset_group_id": "reset-b",
                            "split": splits["trajectory-b"],
                            "trajectory_id": "trajectory-b",
                        },
                    ],
                    "outcome_counts": {"success": 2},
                    "schema_id": GRASP_LIFT_PILOT_OUTPUT_SCHEMA_ID,
                    "split_policy": split_policy,
                    "split_policy_sha256": canonical_fingerprint(split_policy),
                }
            )
            status = seal_grasp_lift_pilot_record(
                {
                    "acceptance_passed": True,
                    "complete": True,
                    "manifest_sha256": manifest["self_sha256"],
                    "outcome_counts": {"success": 2},
                    "schema_id": GRASP_LIFT_PILOT_STATUS_SCHEMA_ID,
                }
            )
            commit = seal_grasp_lift_pilot_record(
                {
                    "manifest_file": "manifest.json",
                    "manifest_sha256": manifest["self_sha256"],
                    "schema_id": GRASP_LIFT_PILOT_COMMIT_SCHEMA_ID,
                    "status_file": "status.json",
                    "status_sha256": status["self_sha256"],
                }
            )
            for name, record in (
                ("manifest.json", manifest),
                ("status.json", status),
                ("commit.json", commit),
            ):
                (root / name).write_text(json.dumps(record), encoding="utf-8")
            self.assertEqual(
                load_grasp_lift_pilot_manifest(root)["self_sha256"],
                manifest["self_sha256"],
            )
            manifest["members"][0]["split"] = (
                "train"
                if manifest["members"][0]["split"] == "validation"
                else "validation"
            )
            (root / "manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "self_sha256 mismatch"):
                load_grasp_lift_pilot_manifest(root)

    def test_loader_rejects_non_json(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            path.write_text("not-json", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "cannot read"):
                load_grasp_lift_pilot_config(path)


if __name__ == "__main__":
    unittest.main()
