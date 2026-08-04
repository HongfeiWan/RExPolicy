"""Tests for production task artifact loading and validation."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from rexpolicy.tasking.repository import (
    PRODUCTION_CAPABILITIES_PATH,
    PRODUCTION_REACH_PROCESS_PATH,
    PRODUCTION_REACH_TASK_PATH,
    load_capability_catalog,
    load_production_reach_artifacts,
    load_production_reach_process_artifacts,
    load_process_spec,
    load_task_spec,
)


class TestTaskRepository(unittest.TestCase):
    def test_production_reach_artifacts_compile_and_validate(self) -> None:
        artifacts = load_production_reach_artifacts()

        self.assertEqual(artifacts.task.task_id, "reach_green_cap/v2")
        self.assertEqual(
            artifacts.contract.task_contract_id,
            "rexpolicy_taskspec_v2/reach_green_cap/v2",
        )
        self.assertEqual(artifacts.contract.episode_control_steps, 8)
        self.assertTrue(artifacts.property_report.accepted)
        self.assertIn(
            "move the open right hand to the safe pre-grasp position beside "
            "the green bottle cap without moving the bottle",
            artifacts.task.instructions.train,
        )
        self.assertEqual(artifacts.task.instructions.audit.count, 8)
        self.assertEqual(len(artifacts.task.instructions.audit.sha256), 64)

    def test_production_files_round_trip_canonically(self) -> None:
        catalog = load_capability_catalog(PRODUCTION_CAPABILITIES_PATH)
        task = load_task_spec(PRODUCTION_REACH_TASK_PATH)
        process = load_process_spec(PRODUCTION_REACH_PROCESS_PATH)

        self.assertEqual(
            type(catalog).from_record(catalog.to_record()).fingerprint,
            catalog.fingerprint,
        )
        self.assertEqual(type(task).from_record(task.to_record()), task)
        self.assertEqual(type(process).from_record(process.to_record()), process)
        json.loads(task.to_json())
        json.loads(process.to_json())

    def test_production_process_is_bound_to_exact_task_artifacts(self) -> None:
        tasks = load_production_reach_artifacts()
        process = load_production_reach_process_artifacts(tasks)

        self.assertIs(process.task_artifacts, tasks)
        self.assertEqual(process.process_spec.process_spec_id, "reach_pregrasp/v1")
        self.assertEqual(
            process.contract.task_contract_fingerprint,
            tasks.contract.fingerprint,
        )
        self.assertEqual(
            process.contract.process_spec_fingerprint,
            process.process_spec.fingerprint,
        )
        self.assertNotIn(
            "reward_distance_scale_m",
            process.process_spec.to_json(),
        )

    def test_duplicate_json_keys_fail_at_the_file_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "task.json"
            path.write_text(
                '{"schema_version":2,"schema_version":2}',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "Duplicate JSON key"):
                load_task_spec(path)

            process_path = Path(directory) / "process.json"
            process_path.write_text(
                '{"schema_version":1,"schema_version":1}',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "Duplicate JSON key"):
                load_process_spec(process_path)


if __name__ == "__main__":
    unittest.main()
