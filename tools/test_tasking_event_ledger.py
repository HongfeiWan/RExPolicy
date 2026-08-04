"""Tests for immutable reward-agnostic Event Ledger v1 contracts."""

from __future__ import annotations

import copy
import json
import unittest

from rexpolicy.tasking.capabilities import CapabilityCatalog
from rexpolicy.tasking.event_ledger import EpisodeEventLedger
from tools.test_tasking_capabilities import capability_record


def _signals(distance: float) -> dict:
    return {
        "reach.distance_m": distance,
        "reach.contact_violation": False,
        "reach.displacement_violation": False,
    }


def ledger_record() -> dict:
    schema = CapabilityCatalog.from_record(capability_record()).event_schemas[0]
    sample_00 = "g000001-r00000-e00000-d00000-w00000"
    sample_01 = "g000001-r00000-e00000-d00000-w00001"
    sample_10 = "g000001-r00000-e00000-d00001-w00000"
    sample_11 = "g000001-r00000-e00000-d00001-w00001"
    return {
        "schema_version": 1,
        "episode_id": "g000001-r00000-e00000",
        "bindings": {
            "task_contract_id": "reach_green_cap/v2",
            "task_contract_sha256": "a" * 64,
            "task_oracle_sha256": "f" * 64,
            "dynamics_contract_sha256": "b" * 64,
            "observation_contract_sha256": "c" * 64,
            "event_schema_id": schema.event_schema_id,
            "event_schema_sha256": schema.fingerprint,
        },
        "collection": {
            "data_generation": 1,
            "sampling_policy_generation": 0,
            "rank": 0,
            "episode": 0,
            "instruction_variant_id": "train/reach_safe/v1",
        },
        "reset_recipe": {
            "environment_adapter_id": "groot_newton/reach/v1",
            "seed": 123,
        },
        "events": [
            {
                "sequence": 0,
                "kind": "decision_root",
                "decision_index": 0,
                "root_control_step": 0,
                "dynamics_digest_sha256": "d" * 64,
                "signals": _signals(0.10),
            },
            {
                "sequence": 1,
                "kind": "transition",
                "sample_id": sample_00,
                "decision_index": 0,
                "world": 0,
                "branch_step": 0,
                "effective_action_19d": [0.0] * 19,
                "post_dynamics_digest_sha256": "e" * 64,
                "post_signals": _signals(0.08),
                "terminated": False,
                "truncated": False,
            },
            {
                "sequence": 2,
                "kind": "transition",
                "sample_id": sample_01,
                "decision_index": 0,
                "world": 1,
                "branch_step": 0,
                "effective_action_19d": [0.01] * 19,
                "post_dynamics_digest_sha256": "f" * 64,
                "post_signals": _signals(0.09),
                "terminated": False,
                "truncated": False,
            },
            {
                "sequence": 3,
                "kind": "decision_closed",
                "decision_index": 0,
                "continued_sample_id": sample_00,
            },
            {
                "sequence": 4,
                "kind": "decision_root",
                "decision_index": 1,
                "root_control_step": 1,
                "dynamics_digest_sha256": "e" * 64,
                "signals": _signals(0.08),
            },
            {
                "sequence": 5,
                "kind": "transition",
                "sample_id": sample_10,
                "decision_index": 1,
                "world": 0,
                "branch_step": 0,
                "effective_action_19d": [0.02] * 19,
                "post_dynamics_digest_sha256": "1" * 64,
                "post_signals": _signals(0.03),
                "terminated": True,
                "truncated": False,
            },
            {
                "sequence": 6,
                "kind": "transition",
                "sample_id": sample_11,
                "decision_index": 1,
                "world": 1,
                "branch_step": 0,
                "effective_action_19d": [0.0] * 19,
                "post_dynamics_digest_sha256": "2" * 64,
                "post_signals": _signals(0.07),
                "terminated": False,
                "truncated": False,
            },
            {
                "sequence": 7,
                "kind": "decision_closed",
                "decision_index": 1,
                "continued_sample_id": sample_10,
            },
        ],
    }


class TestEpisodeEventLedger(unittest.TestCase):
    def setUp(self) -> None:
        self.schema = CapabilityCatalog.from_record(
            capability_record()
        ).event_schemas[0]

    def test_round_trip_is_stable_and_reward_agnostic(self) -> None:
        ledger = EpisodeEventLedger.from_record(
            ledger_record(),
            event_schema=self.schema,
        )
        ledger.assert_reward_agnostic()
        restored = EpisodeEventLedger.from_record(
            json.loads(ledger.to_json()),
            event_schema=self.schema,
        )

        self.assertEqual(restored, ledger)
        self.assertEqual(restored.fingerprint, ledger.fingerprint)
        self.assertNotIn('"reward"', ledger.to_json())
        self.assertNotIn('"score"', ledger.to_json())
        self.assertNotIn('"advantage"', ledger.to_json())

    def test_sequence_continuation_and_root_digest_are_strict(self) -> None:
        cases = []
        sequence = ledger_record()
        sequence["events"][2]["sequence"] = 9
        cases.append((sequence, "sequence must be contiguous"))
        dangling = ledger_record()
        dangling["events"][3]["continued_sample_id"] = (
            "g000001-r00000-e00000-d00000-w00009"
        )
        cases.append((dangling, "references no candidate"))
        digest = ledger_record()
        digest["events"][4]["dynamics_digest_sha256"] = "9" * 64
        cases.append((digest, "digest does not match continuation"))
        for record, message in cases:
            with self.assertRaisesRegex(ValueError, message):
                EpisodeEventLedger.from_record(
                    record,
                    event_schema=self.schema,
                )

    def test_continuation_signals_and_sample_provenance_are_strict(self) -> None:
        signals = ledger_record()
        signals["events"][4]["signals"] = _signals(0.081)
        with self.assertRaisesRegex(ValueError, "signals do not match"):
            EpisodeEventLedger.from_record(signals, event_schema=self.schema)

        provenance = ledger_record()
        provenance["events"][1]["sample_id"] = (
            "g000001-r00000-e00000-d00001-w00000"
        )
        with self.assertRaisesRegex(ValueError, "sample_id/provenance"):
            EpisodeEventLedger.from_record(provenance, event_schema=self.schema)

    def test_signal_frames_are_deeply_immutable(self) -> None:
        ledger = EpisodeEventLedger.from_record(
            ledger_record(),
            event_schema=self.schema,
        )
        root = ledger.events[0]
        with self.assertRaises(TypeError):
            root.signals[0] = ("reach.distance_m", 0.0)

    def test_transition_after_termination_is_rejected(self) -> None:
        record = ledger_record()
        terminal = copy.deepcopy(record["events"][5])
        terminal["sequence"] = 6
        terminal["branch_step"] = 1
        terminal["post_dynamics_digest_sha256"] = "3" * 64
        record["events"].insert(6, terminal)
        for sequence, event in enumerate(record["events"]):
            event["sequence"] = sequence
        with self.assertRaisesRegex(ValueError, "after termination"):
            EpisodeEventLedger.from_record(record, event_schema=self.schema)

    def test_action_and_signal_values_are_strict(self) -> None:
        short_action = ledger_record()
        short_action["events"][1]["effective_action_19d"] = [0.0] * 18
        with self.assertRaisesRegex(ValueError, "length 19"):
            EpisodeEventLedger.from_record(
                short_action,
                event_schema=self.schema,
            )

        non_finite = ledger_record()
        non_finite["events"][1]["post_signals"][
            "reach.distance_m"
        ] = float("nan")
        with self.assertRaisesRegex(ValueError, "finite"):
            EpisodeEventLedger.from_record(
                non_finite,
                event_schema=self.schema,
            )

    def test_derived_training_fields_cannot_enter_events(self) -> None:
        record = ledger_record()
        record["events"][1]["reward"] = 1.0
        with self.assertRaisesRegex(ValueError, "unknown fields"):
            EpisodeEventLedger.from_record(record, event_schema=self.schema)


if __name__ == "__main__":
    unittest.main()
