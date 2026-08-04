"""Tests for immutable storage of rebuildable flywheel sidecars."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from rexpolicy.flywheel.derived_views import write_derived_view
from rexpolicy.tasking.process_label_view import materialize_process_label_view
from rexpolicy.tasking.repository import load_production_reach_process_artifacts
from rexpolicy.tasking.reward_view import materialize_reward_view
from tools.test_tasking_reward_view import _inputs


def _views():
    tasks, ledger = _inputs()
    process = load_production_reach_process_artifacts(tasks)
    process_view = materialize_process_label_view(
        ledger,
        task_contract=tasks.contract,
        process_contract=process.contract,
        catalog=tasks.capabilities,
    )
    reward_view = materialize_reward_view(
        ledger,
        scoring_contract=tasks.contract,
        reward_profile_id="reach_progress/v2",
        catalog=tasks.capabilities,
    )
    return process_view, reward_view


class TestDerivedViewArchive(unittest.TestCase):
    def test_views_are_content_addressed_without_touching_source_pairs(self) -> None:
        process_view, reward_view = _views()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            process_write = write_derived_view(
                run_dir=root,
                view=process_view,
            )
            reward_write = write_derived_view(
                run_dir=root,
                view=reward_view,
            )

            self.assertTrue(process_write.created)
            self.assertTrue(reward_write.created)
            self.assertIn(process_view.fingerprint, process_write.relative_path)
            self.assertIn(reward_view.fingerprint, reward_write.relative_path)
            self.assertEqual(
                json.loads(
                    (root / process_write.relative_path).read_text(
                        encoding="ascii"
                    )
                ),
                process_view.to_record(),
            )
            self.assertFalse((root / "archive" / "attempts").exists())

    def test_idempotent_retry_never_overwrites_existing_content(self) -> None:
        process_view, _ = _views()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = write_derived_view(run_dir=root, view=process_view)
            second = write_derived_view(run_dir=root, view=process_view)
            self.assertTrue(first.created)
            self.assertFalse(second.created)

            path = root / first.relative_path
            path.write_text("{}\n", encoding="ascii")
            with self.assertRaisesRegex(FileExistsError, "does not match"):
                write_derived_view(run_dir=root, view=process_view)


if __name__ == "__main__":
    unittest.main()
