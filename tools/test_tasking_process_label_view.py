"""Tests for deterministic ProcessLabelView materialization."""

from __future__ import annotations

import json
import unittest

from rexpolicy.tasking.event_ledger import EpisodeEventLedger
from rexpolicy.tasking.process_label_view import ProcessLabelView, materialize_process_label_view
from rexpolicy.tasking.repository import (
    load_production_reach_artifacts,
    load_production_reach_process_artifacts,
)
from tools.test_tasking_event_ledger import ledger_record


def _inputs(*, unsafe_goal: bool = False):
    tasks = load_production_reach_artifacts()
    process = load_production_reach_process_artifacts(tasks)
    record = ledger_record()
    record["bindings"]["task_contract_id"] = tasks.contract.task_contract_id
    record["bindings"]["task_contract_sha256"] = tasks.contract.fingerprint
    record["bindings"]["task_oracle_sha256"] = (
        tasks.contract.oracle_fingerprint
    )
    record["events"][0]["signals"]["reach.distance_m"] = 0.12
    if unsafe_goal:
        record["events"][5]["post_signals"]["reach.contact_violation"] = True
    schema = tasks.capabilities.event_schemas[0]
    ledger = EpisodeEventLedger.from_record(record, event_schema=schema)
    return tasks, process, ledger


class TestProcessLabelView(unittest.TestCase):
    def test_every_parallel_transition_is_labeled_from_its_decision_root(self) -> None:
        tasks, process, ledger = _inputs()
        view = materialize_process_label_view(
            ledger,
            task_contract=tasks.contract,
            process_contract=process.contract,
            catalog=tasks.capabilities,
        )

        self.assertEqual(len(view.transition_labels), 4)
        first, second, third, fourth = view.transition_labels
        self.assertEqual(
            (first.stage_before, first.stage_after, first.transition_code),
            ("far", "approach", "advance"),
        )
        self.assertEqual(
            (second.stage_before, second.stage_after, second.transition_code),
            ("far", "approach", "advance"),
        )
        self.assertEqual(
            (third.stage_before, third.stage_after, third.transition_code),
            ("approach", "goal_zone", "advance"),
        )
        self.assertEqual(
            (fourth.stage_before, fourth.stage_after, fourth.transition_code),
            ("approach", "approach", "hold"),
        )
        self.assertEqual(
            {label.source_sequence for label in view.transition_labels},
            {1, 2, 5, 6},
        )
        self.assertEqual(
            view.task_oracle_sha256,
            tasks.contract.oracle_fingerprint,
        )

    def test_unsafe_stage_precedes_overlapping_goal_stage(self) -> None:
        tasks, process, ledger = _inputs(unsafe_goal=True)
        view = materialize_process_label_view(
            ledger,
            task_contract=tasks.contract,
            process_contract=process.contract,
            catalog=tasks.capabilities,
        )
        unsafe = next(
            label for label in view.transition_labels if label.source_sequence == 5
        )

        self.assertEqual(unsafe.stage_after, "unsafe")
        self.assertEqual(unsafe.transition_code, "entered_unsafe")
        self.assertNotIn("success", unsafe.to_record())

    def test_segments_cover_candidates_without_cross_world_merging(self) -> None:
        tasks, process, ledger = _inputs()
        view = materialize_process_label_view(
            ledger,
            task_contract=tasks.contract,
            process_contract=process.contract,
            catalog=tasks.capabilities,
        )

        self.assertEqual(len(view.segments), 4)
        self.assertEqual(
            {(segment.decision_index, segment.world) for segment in view.segments},
            {(0, 0), (0, 1), (1, 0), (1, 1)},
        )
        self.assertTrue(all(segment.step_count == 1 for segment in view.segments))
        self.assertEqual(
            next(segment for segment in view.segments if segment.world == 0 and segment.decision_index == 1).end_reason,
            "terminal",
        )

    def test_segments_split_at_the_policy_bound_length(self) -> None:
        tasks = load_production_reach_artifacts()
        process = load_production_reach_process_artifacts(tasks)
        record = ledger_record()
        record["bindings"]["task_contract_id"] = tasks.contract.task_contract_id
        record["bindings"]["task_contract_sha256"] = tasks.contract.fingerprint
        record["bindings"]["task_oracle_sha256"] = (
            tasks.contract.oracle_fingerprint
        )
        root = record["events"][0]
        root["signals"]["reach.distance_m"] = 0.08
        sample_id = "g000001-r00000-e00000-d00000-w00000"
        transitions = []
        for branch_step in range(5):
            transitions.append(
                {
                    "sequence": branch_step + 1,
                    "kind": "transition",
                    "sample_id": sample_id,
                    "decision_index": 0,
                    "world": 0,
                    "branch_step": branch_step,
                    "effective_action_19d": [0.0] * 19,
                    "post_dynamics_digest_sha256": str(branch_step + 3) * 64,
                    "post_signals": {
                        "reach.distance_m": 0.08,
                        "reach.contact_violation": False,
                        "reach.displacement_violation": False,
                    },
                    "terminated": False,
                    "truncated": False,
                }
            )
        record["events"] = [
            root,
            *transitions,
            {
                "sequence": 6,
                "kind": "decision_closed",
                "decision_index": 0,
                "continued_sample_id": None,
            },
        ]
        ledger = EpisodeEventLedger.from_record(
            record,
            event_schema=tasks.capabilities.event_schemas[0],
        )
        view = materialize_process_label_view(
            ledger,
            task_contract=tasks.contract,
            process_contract=process.contract,
            catalog=tasks.capabilities,
        )

        self.assertEqual(
            [label.segment_index for label in view.transition_labels],
            [0, 0, 0, 0, 1],
        )
        self.assertEqual(
            [(segment.step_count, segment.end_reason) for segment in view.segments],
            [(4, "max_length"), (1, "candidate_end")],
        )

    def test_round_trip_recomputes_instead_of_trusting_derived_labels(self) -> None:
        tasks, process, ledger = _inputs()
        view = materialize_process_label_view(
            ledger,
            task_contract=tasks.contract,
            process_contract=process.contract,
            catalog=tasks.capabilities,
        )
        restored = ProcessLabelView.from_json(
            view.to_json(),
            ledger=ledger,
            task_contract=tasks.contract,
            process_contract=process.contract,
            catalog=tasks.capabilities,
        )
        self.assertEqual(restored, view)
        self.assertEqual(restored.fingerprint, view.fingerprint)

        tampered = json.loads(view.to_json())
        tampered["transition_labels"][0]["transition_code"] = "hold"
        with self.assertRaisesRegex(ValueError, "deterministic materialization"):
            ProcessLabelView.from_record(
                tampered,
                ledger=ledger,
                task_contract=tasks.contract,
                process_contract=process.contract,
                catalog=tasks.capabilities,
            )

    def test_task_contract_and_source_ledger_bindings_fail_closed(self) -> None:
        tasks, process, ledger = _inputs()
        record = ledger.to_record()
        record["bindings"]["task_contract_sha256"] = "0" * 64
        mismatched = EpisodeEventLedger.from_record(
            record,
            event_schema=tasks.capabilities.event_schemas[0],
        )
        with self.assertRaisesRegex(ValueError, "TaskContract binding mismatch"):
            materialize_process_label_view(
                mismatched,
                task_contract=tasks.contract,
                process_contract=process.contract,
                catalog=tasks.capabilities,
            )

    def test_process_view_contains_no_reward_fields(self) -> None:
        tasks, process, ledger = _inputs()
        view = materialize_process_label_view(
            ledger,
            task_contract=tasks.contract,
            process_contract=process.contract,
            catalog=tasks.capabilities,
        )
        payload = view.to_json()

        self.assertNotIn('"reward"', payload)
        self.assertNotIn('"score"', payload)
        self.assertNotIn('"advantage"', payload)


if __name__ == "__main__":
    unittest.main()
