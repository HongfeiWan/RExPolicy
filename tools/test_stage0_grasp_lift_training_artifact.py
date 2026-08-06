"""Tests for immutable, tensor-free Grasp-Lift training artifacts."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rexpolicy.stage0.data.grasp_lift_training_artifact import (
    GRASP_LIFT_TRAINING_ARTIFACT_STORAGE_POLICY,
    load_grasp_lift_training_artifact,
    publish_grasp_lift_training_artifact,
)
from tools.test_stage0_grasp_lift_training import _consistent_synthetic_corpus


_MODULE = "rexpolicy.stage0.data.grasp_lift_training_artifact"


class GraspLiftTrainingArtifactTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.pilot = self.root / "synthetic-pilot-is-never-opened"
        self.corpus = _consistent_synthetic_corpus()
        self.loader = patch(
            f"{_MODULE}.load_grasp_lift_training_corpus",
            return_value=self.corpus,
        )
        self.loader.start()
        self.addCleanup(self.loader.stop)
        self.addCleanup(self.temporary.cleanup)

    def test_three_file_round_trip_rebuilds_train_content(self) -> None:
        target = self.root / "training-artifact"
        published = publish_grasp_lift_training_artifact(
            target,
            pilot_directory=self.pilot,
        )
        loaded = load_grasp_lift_training_artifact(
            target,
            pilot_directory=self.pilot,
        )

        self.assertEqual(
            {path.name for path in target.iterdir()},
            {"commit.json", "manifest.json", "normalization.json"},
        )
        self.assertFalse(any(target.glob("*.pt")))
        self.assertEqual(published.artifact_sha256, loaded.artifact_sha256)
        self.assertEqual(
            loaded.train_content_sha256,
            loaded.data.train_content_sha256,
        )
        self.assertEqual(
            loaded.audit_dataset_sha256,
            loaded.data.dataset_sha256,
        )
        self.assertEqual(
            loaded.manifest["storage_policy"],
            GRASP_LIFT_TRAINING_ARTIFACT_STORAGE_POLICY,
        )
        self.assertFalse(loaded.data.window_splits.validation)
        self.assertFalse(loaded.data.window_splits.test)

    def test_existing_target_and_extra_files_fail_closed(self) -> None:
        target = self.root / "training-artifact"
        publish_grasp_lift_training_artifact(
            target,
            pilot_directory=self.pilot,
        )
        with self.assertRaises(FileExistsError):
            publish_grasp_lift_training_artifact(
                target,
                pilot_directory=self.pilot,
            )
        (target / "held-out.pt").write_bytes(b"must never be accepted")
        with self.assertRaisesRegex(ValueError, "file set changed"):
            load_grasp_lift_training_artifact(
                target,
                pilot_directory=self.pilot,
            )

    def test_normalization_tamper_is_rejected(self) -> None:
        target = self.root / "training-artifact"
        publish_grasp_lift_training_artifact(
            target,
            pilot_directory=self.pilot,
        )
        path = target / "normalization.json"
        record = json.loads(path.read_text(encoding="utf-8"))
        record["state_mean"][0] += 1.0
        path.write_text(
            json.dumps(record, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        with self.assertRaisesRegex(ValueError, "file hash changed"):
            load_grasp_lift_training_artifact(
                target,
                pilot_directory=self.pilot,
            )

    def test_failed_prepublication_reload_leaves_no_target(self) -> None:
        target = self.root / "training-artifact"
        with patch(
            f"{_MODULE}._load_grasp_lift_training_artifact",
            side_effect=RuntimeError("injected strict reload failure"),
        ):
            with self.assertRaisesRegex(RuntimeError, "strict reload failure"):
                publish_grasp_lift_training_artifact(
                    target,
                    pilot_directory=self.pilot,
                )
        self.assertFalse(target.exists())
        self.assertFalse(any(self.root.glob(".*.partial-*")))


if __name__ == "__main__":
    unittest.main()
