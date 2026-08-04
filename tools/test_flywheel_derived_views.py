"""Tests for immutable storage of rebuildable flywheel sidecars."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from rexpolicy.flywheel.derived_views import write_derived_view
from rexpolicy.tasking.process_reward import (
    ProcessRewardModelManifest,
    materialize_process_reward_shadow,
)
from rexpolicy.tasking.reward_view import materialize_reward_view
from tools.test_tasking_process_reward import (
    _artifacts,
    _manifest_record,
    _outputs,
)
from tools.test_tasking_reward_view import _inputs as _reward_inputs


def _views():
    tasks, process, ledger, process_view = _artifacts()
    manifest = ProcessRewardModelManifest.from_record(
        _manifest_record(tasks, process, process_view)
    )
    shadow_view = materialize_process_reward_shadow(
        process_view,
        manifest=manifest,
        checkpoint_sha256="a" * 64,
        outputs=_outputs(process_view),
    )
    reward_tasks, reward_ledger = _reward_inputs()
    reward_view = materialize_reward_view(
        reward_ledger,
        scoring_contract=reward_tasks.contract,
        reward_profile_id="reach_progress/v2",
        catalog=reward_tasks.capabilities,
    )
    return process_view, reward_view, shadow_view


class TestDerivedViewArchive(unittest.TestCase):
    def test_views_are_content_addressed_without_touching_source_pairs(self) -> None:
        process_view, reward_view, shadow_view = _views()
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
            shadow_write = write_derived_view(
                run_dir=root,
                view=shadow_view,
            )

            self.assertTrue(process_write.created)
            self.assertTrue(reward_write.created)
            self.assertTrue(shadow_write.created)
            self.assertIn(process_view.fingerprint, process_write.relative_path)
            self.assertIn(reward_view.fingerprint, reward_write.relative_path)
            self.assertIn(shadow_view.fingerprint, shadow_write.relative_path)
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
        process_view, _, _ = _views()
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
