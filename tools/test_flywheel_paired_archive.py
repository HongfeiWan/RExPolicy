"""Tests for aligned episode/Event Ledger shard commit descriptors."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rexpolicy.flywheel.experience import EpisodeExperience
from rexpolicy.flywheel.paired_archive import write_generation_archive_pair
from rexpolicy.tasking.event_ledger import EpisodeEventLedger
from tools.test_tasking_event_ledger import ledger_record
from tools.test_tasking_capabilities import capability_record
from rexpolicy.tasking.capabilities import CapabilityCatalog


def _episode() -> EpisodeExperience:
    return EpisodeExperience(
        generation=1,
        sampling_policy_generation=0,
        rank=0,
        episode=0,
        reset_seed=123,
        instruction="reach",
        task_id="reach_green_cap/v1",
        reward_profile_id="reach_progress/v1",
        score=0.0,
        success=False,
        initial_state={},
        simulator_fingerprint="a" * 64,
        decisions=[],
        selected_actions=[],
        rewards=[],
        samples=[],
    )


def _ledger() -> EpisodeEventLedger:
    schema = CapabilityCatalog.from_record(capability_record()).event_schemas[0]
    return EpisodeEventLedger.from_record(
        ledger_record(),
        event_schema=schema,
    )


class TestPairedArchive(unittest.TestCase):
    def test_descriptor_commits_aligned_immutable_shards(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = write_generation_archive_pair(
                run_dir=root,
                attempt_id="attempt-1",
                generation=1,
                rank=0,
                episodes=[_episode()],
                event_ledgers=[_ledger()],
            )
            descriptor = json.loads(
                (root / result.pair_descriptor_path).read_text(encoding="utf-8")
            )
            self.assertEqual(descriptor["schema_version"], 1)
            self.assertEqual(descriptor["episode_shard"]["record_count"], 1)
            self.assertEqual(descriptor["event_shard"]["record_count"], 1)
            self.assertEqual(
                descriptor["episode_shard"]["sha256"],
                result.episode_sha256,
            )
            self.assertEqual(
                descriptor["event_shard"]["sha256"],
                result.event_sha256,
            )
            with self.assertRaises(FileExistsError):
                write_generation_archive_pair(
                    run_dir=root,
                    attempt_id="attempt-1",
                    generation=1,
                    rank=0,
                    episodes=[_episode()],
                    event_ledgers=[_ledger()],
                )

    def test_provenance_mismatch_is_rejected_before_writes(self) -> None:
        episode = _episode()
        episode.episode = 9
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(ValueError, "provenance mismatch"):
                write_generation_archive_pair(
                    run_dir=root,
                    attempt_id="attempt-1",
                    generation=1,
                    rank=0,
                    episodes=[episode],
                    event_ledgers=[_ledger()],
                )
            self.assertFalse((root / "archive").exists())

    def test_second_shard_failure_never_publishes_descriptor(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            from rexpolicy.flywheel import paired_archive

            original = paired_archive._atomic_write_jsonl
            calls = 0

            def fail_second(path, records):
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise OSError("injected event-shard failure")
                original(path, records)

            with patch.object(
                paired_archive,
                "_atomic_write_jsonl",
                side_effect=fail_second,
            ):
                with self.assertRaisesRegex(OSError, "injected"):
                    write_generation_archive_pair(
                        run_dir=root,
                        attempt_id="attempt-1",
                        generation=1,
                        rank=0,
                        episodes=[_episode()],
                        event_ledgers=[_ledger()],
                    )
            pair = root / "archive" / "attempts" / "attempt-1"
            pair = pair / "generation-000001" / "rank-00000.pair.json"
            self.assertFalse(pair.exists())


if __name__ == "__main__":
    unittest.main()
