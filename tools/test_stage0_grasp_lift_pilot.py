"""Tests for the strict Grasp-Lift pilot config contract."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from rexpolicy.stage0.grasp_lift_pilot import (
    GRASP_LIFT_PILOT_FORMAL_CONFIG_SHA256,
    GraspLiftPilotConfig,
    evaluate_grasp_lift_close7_authoring_gate,
    load_grasp_lift_pilot_config,
    load_grasp_lift_pilot_manifest,
    seal_grasp_lift_pilot_record,
    verify_grasp_lift_pilot_record,
    verify_grasp_lift_pilot_formal_config,
)


ROOT = Path(__file__).resolve().parents[1]


class GraspLiftPilotConfigTest(unittest.TestCase):
    @staticmethod
    def _paired_members(
        *,
        train_timeout_conversions: int,
        validation_timeout_conversions: int,
        validation_rejection_from_phase: int | None = None,
        engineering_smoke_successful: bool = True,
    ) -> list[dict[str, object]]:
        train_success = (
            7003, 7004, 7007, 7008, 7010, 7015, 7016, 7017,
            7018, 7022, 7023, 7024, 7027, 7029, 7030,
        )
        train_timeout = (
            7000, 7002, 7006, 7011, 7012, 7013, 7020, 7021, 7028,
        )
        engineering_smoke_excluded = (7001, 7005)
        validation_success = (7009, 7019, 7025, 7026)
        validation_timeout = (7014, 7031)
        members = []
        for split, seeds, successful_count in (
            ("train", train_success, len(train_success)),
            ("train", train_timeout, train_timeout_conversions),
            (
                "validation",
                engineering_smoke_excluded,
                (
                    len(engineering_smoke_excluded)
                    if engineering_smoke_successful
                    else 0
                ),
            ),
            ("validation", validation_success, len(validation_success)),
            (
                "validation",
                validation_timeout,
                validation_timeout_conversions,
            ),
        ):
            for index, seed in enumerate(seeds):
                success = index < successful_count
                members.append(
                    {
                        "final_authoring_phase": 5 if success else 7,
                        "formal_gate_eligible": seed not in {7001, 7005},
                        "outcome": "success" if success else "timeout",
                        "reset_seed": seed,
                        "split": split,
                        "terminal_transition_from_phase": (
                            validation_rejection_from_phase
                            if split == "validation"
                            and not success
                            and validation_rejection_from_phase is not None
                            else 2
                        ),
                    }
                )
        return members

    def test_close7_gate_passes_only_the_precommitted_paired_thresholds(
        self,
    ) -> None:
        failed = evaluate_grasp_lift_close7_authoring_gate(
            self._paired_members(
                train_timeout_conversions=5,
                validation_timeout_conversions=1,
            )
        )
        self.assertFalse(failed["passed"])

        passed = evaluate_grasp_lift_close7_authoring_gate(
            self._paired_members(
                train_timeout_conversions=6,
                validation_timeout_conversions=2,
                engineering_smoke_successful=False,
            )
        )
        self.assertTrue(passed["passed"])

    def test_close7_gate_rejects_post_lift_instability(self) -> None:
        for phase in (3, 4):
            gate = evaluate_grasp_lift_close7_authoring_gate(
                self._paired_members(
                    train_timeout_conversions=9,
                    validation_timeout_conversions=1,
                    validation_rejection_from_phase=phase,
                )
            )
            self.assertFalse(gate["passed"])
            self.assertEqual(gate["validation"]["lift_to_rejected"], 1)

    def test_repository_pilot_config_is_valid_and_hash_bound(self) -> None:
        config = load_grasp_lift_pilot_config(
            ROOT / "configs" / "stage0" / "grasp_lift_pilot.json"
        )
        self.assertEqual(config.runtime.cpu_threads, 16)
        self.assertEqual(config.runtime.num_envs, 8)
        self.assertEqual(config.runtime.reset_groups, 32)
        self.assertEqual(config.acceptance.locked_test_policy, "not_created")
        self.assertEqual(len(config.sha256), 64)
        self.assertEqual(config.sha256, GRASP_LIFT_PILOT_FORMAL_CONFIG_SHA256)
        verify_grasp_lift_pilot_formal_config(config)
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

    def test_formal_config_rejects_a_runtime_override(self) -> None:
        source = json.loads(
            (ROOT / "configs" / "stage0" / "grasp_lift_pilot.json").read_text()
        )
        source["runtime"]["max_episode_steps"] = 73
        with self.assertRaisesRegex(ValueError, "pre-registered"):
            verify_grasp_lift_pilot_formal_config(
                GraspLiftPilotConfig.from_mapping(source)
            )

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
