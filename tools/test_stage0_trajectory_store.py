"""Tests for strict, immutable Stage 0 trajectory shards."""

from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import torch

from rexpolicy.stage0.data import (
    Stage0ShardIntegrityError,
    Stage0Trajectory,
    Stage0TrajectoryProvenance,
    load_trajectory_shard,
    trajectory_sha256,
    write_trajectory_shard,
)
from rexpolicy.stage0.types import Stage0Outcome


def _provenance(trajectory_id: str = "trajectory-000") -> Stage0TrajectoryProvenance:
    return Stage0TrajectoryProvenance(
        trajectory_id=trajectory_id,
        reset_group_id="reset-000",
        task_id="reach/v1",
        oracle_id="reach-oracle/v1",
        simulator_sha256="1" * 64,
        experiment_sha256="2" * 64,
        state_schema_sha256="3" * 64,
        action_schema_sha256="4" * 64,
        generation=1,
        rank=0,
        world=7,
        reset_seed=42,
    )


def _trajectory(
    *,
    outcome: Stage0Outcome = Stage0Outcome.SUCCESS,
    trajectory_id: str = "trajectory-000",
) -> Stage0Trajectory:
    terminal = {
        Stage0Outcome.SUCCESS: (True, False, None),
        Stage0Outcome.FAILURE: (True, False, "unsafe-contact"),
        Stage0Outcome.TIMEOUT: (False, True, None),
    }[outcome]
    return Stage0Trajectory(
        provenance=_provenance(trajectory_id),
        states=torch.arange(15, dtype=torch.float32).reshape(5, 3),
        actions=torch.arange(8, dtype=torch.float32).reshape(4, 2),
        outcome=outcome,
        terminated=terminal[0],
        truncated=terminal[1],
        failure_reason=terminal[2],
    )


class Stage0TrajectoryStoreTests(unittest.TestCase):
    def test_success_failure_and_timeout_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for index, outcome in enumerate(Stage0Outcome):
                source = _trajectory(
                    outcome=outcome,
                    trajectory_id=f"trajectory-{index:03d}",
                )
                descriptor = write_trajectory_shard(root, source)
                restored = load_trajectory_shard(descriptor)
                self.assertEqual(restored.metadata_record(), source.metadata_record())
                self.assertTrue(torch.equal(restored.states, source.states))
                self.assertTrue(torch.equal(restored.actions, source.actions))
                self.assertEqual(trajectory_sha256(restored), trajectory_sha256(source))

    def test_shards_are_immutable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            trajectory = _trajectory()
            descriptor = write_trajectory_shard(temporary, trajectory)
            payload = descriptor.with_name("trajectory-000.stage0.pt")
            descriptor_bytes = descriptor.read_bytes()
            payload_bytes = payload.read_bytes()
            with self.assertRaises(FileExistsError):
                write_trajectory_shard(temporary, trajectory)
            self.assertEqual(descriptor.read_bytes(), descriptor_bytes)
            self.assertEqual(payload.read_bytes(), payload_bytes)

    def test_payload_corruption_is_detected_before_loading(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            descriptor = write_trajectory_shard(temporary, _trajectory())
            record = json.loads(descriptor.read_text(encoding="utf-8"))
            payload = descriptor.parent / record["payload_file"]
            changed = bytearray(payload.read_bytes())
            changed[len(changed) // 2] ^= 1
            payload.write_bytes(changed)
            with self.assertRaisesRegex(
                Stage0ShardIntegrityError,
                "SHA-256 mismatch",
            ):
                load_trajectory_shard(descriptor)

    def test_descriptor_is_strict_and_cannot_redirect_payload(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            descriptor = write_trajectory_shard(temporary, _trajectory())
            record = json.loads(descriptor.read_text(encoding="utf-8"))
            record["unexpected"] = True
            descriptor.write_text(json.dumps(record), encoding="utf-8")
            with self.assertRaisesRegex(Stage0ShardIntegrityError, "extra"):
                load_trajectory_shard(descriptor)

    def test_transition_and_terminal_contracts_are_strict(self) -> None:
        source = _trajectory()
        with self.assertRaisesRegex(ValueError, "transition aligned"):
            replace(source, states=source.states[:-1])
        with self.assertRaisesRegex(ValueError, "terminal contract"):
            replace(source, terminated=False)
        with self.assertRaisesRegex(ValueError, "non-finite"):
            bad_states = source.states.clone()
            bad_states[0, 0] = torch.nan
            replace(source, states=bad_states)


if __name__ == "__main__":
    unittest.main()
