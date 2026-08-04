"""Tests for deterministic, re-scorable RewardView sidecars."""

from __future__ import annotations

import copy
import json
import unittest

from rexpolicy.tasking.contract import compile_task_contract
from rexpolicy.tasking.event_ledger import EpisodeEventLedger
from rexpolicy.tasking.model import TaskSpecV2
from rexpolicy.tasking.repository import load_production_reach_artifacts
from rexpolicy.tasking.reward_view import RewardView, materialize_reward_view


def _signals(distance: float, *, contact: bool = False) -> dict:
    return {
        "reach.distance_m": distance,
        "reach.contact_violation": contact,
        "reach.displacement_violation": False,
    }


def _ledger_record(tasks) -> dict:
    sample_0 = "g000001-r00000-e00000-d00000-w00000"
    sample_1 = "g000001-r00000-e00000-d00000-w00001"
    return {
        "schema_version": 1,
        "episode_id": "g000001-r00000-e00000",
        "bindings": {
            "task_contract_id": tasks.contract.task_contract_id,
            "task_contract_sha256": tasks.contract.fingerprint,
            "task_oracle_sha256": tasks.contract.oracle_fingerprint,
            "dynamics_contract_sha256": "b" * 64,
            "observation_contract_sha256": "c" * 64,
            "event_schema_id": tasks.contract.event_schema_id,
            "event_schema_sha256": tasks.contract.event_schema_fingerprint,
        },
        "collection": {
            "data_generation": 1,
            "sampling_policy_generation": 0,
            "rank": 0,
            "episode": 0,
            "instruction_variant_id": "train/reach_safe/v1",
        },
        "reset_recipe": {
            "environment_adapter_id": tasks.contract.adapter_id,
            "seed": 123,
        },
        "events": [
            {
                "sequence": 0,
                "kind": "decision_root",
                "decision_index": 0,
                "root_control_step": 0,
                "dynamics_digest_sha256": "d" * 64,
                "signals": _signals(0.12),
            },
            {
                "sequence": 1,
                "kind": "transition",
                "sample_id": sample_0,
                "decision_index": 0,
                "world": 0,
                "branch_step": 0,
                "effective_action_19d": [0.0] * 19,
                "post_dynamics_digest_sha256": "e" * 64,
                "post_signals": _signals(0.03),
                "terminated": False,
                "truncated": False,
            },
            {
                "sequence": 2,
                "kind": "transition",
                "sample_id": sample_0,
                "decision_index": 0,
                "world": 0,
                "branch_step": 1,
                "effective_action_19d": [0.0] * 19,
                "post_dynamics_digest_sha256": "f" * 64,
                "post_signals": _signals(0.03),
                "terminated": True,
                "truncated": False,
            },
            {
                "sequence": 3,
                "kind": "transition",
                "sample_id": sample_1,
                "decision_index": 0,
                "world": 1,
                "branch_step": 0,
                "effective_action_19d": [0.0] * 19,
                "post_dynamics_digest_sha256": "1" * 64,
                "post_signals": _signals(0.10),
                "terminated": False,
                "truncated": False,
            },
            {
                "sequence": 4,
                "kind": "transition",
                "sample_id": sample_1,
                "decision_index": 0,
                "world": 1,
                "branch_step": 1,
                "effective_action_19d": [0.0] * 19,
                "post_dynamics_digest_sha256": "2" * 64,
                "post_signals": _signals(0.09),
                "terminated": False,
                "truncated": False,
            },
            {
                "sequence": 5,
                "kind": "decision_closed",
                "decision_index": 0,
                "continued_sample_id": sample_0,
            },
        ],
    }


def _inputs():
    tasks = load_production_reach_artifacts()
    ledger = EpisodeEventLedger.from_record(
        _ledger_record(tasks),
        event_schema=tasks.capabilities.event_schemas[0],
    )
    return tasks, ledger


class TestRewardView(unittest.TestCase):
    def test_replays_parallel_branches_with_failure_precedence_runtime(self) -> None:
        tasks, ledger = _inputs()
        view = materialize_reward_view(
            ledger,
            scoring_contract=tasks.contract,
            reward_profile_id="reach_progress/v2",
            catalog=tasks.capabilities,
        )

        self.assertEqual(len(view.transitions), 4)
        self.assertEqual(
            [item.status for item in view.transitions],
            ["active", "success", "active", "active"],
        )
        self.assertEqual(view.transitions[1].reward, 1.0)
        self.assertEqual(len(view.candidates), 2)
        self.assertTrue(view.candidates[0].continued_from_root)
        self.assertFalse(view.candidates[1].continued_from_root)
        self.assertEqual(view.task_oracle_sha256, tasks.contract.oracle_fingerprint)

    def test_same_ledger_can_be_rescored_without_changing_task_truth(self) -> None:
        tasks, ledger = _inputs()
        changed = tasks.task.to_record()
        changed["reward_profiles"][0]["terms"][0]["weight"] = 0.5
        changed["reward_profiles"][0]["shaping_bounds"] = [-0.5, 0.5]
        alternate = compile_task_contract(
            TaskSpecV2.from_record(changed),
            catalog=tasks.capabilities,
        )
        original_view = materialize_reward_view(
            ledger,
            scoring_contract=tasks.contract,
            reward_profile_id="reach_progress/v2",
            catalog=tasks.capabilities,
        )
        alternate_view = materialize_reward_view(
            ledger,
            scoring_contract=alternate,
            reward_profile_id="reach_progress/v2",
            catalog=tasks.capabilities,
        )

        self.assertNotEqual(tasks.contract.fingerprint, alternate.fingerprint)
        self.assertEqual(
            tasks.contract.oracle_fingerprint,
            alternate.oracle_fingerprint,
        )
        self.assertNotEqual(
            [item.reward for item in original_view.transitions],
            [item.reward for item in alternate_view.transitions],
        )
        self.assertEqual(
            [item.status for item in original_view.transitions],
            [item.status for item in alternate_view.transitions],
        )
        self.assertEqual(
            [item.oracle_state_after_sha256 for item in original_view.transitions],
            [item.oracle_state_after_sha256 for item in alternate_view.transitions],
        )

    def test_terminal_flags_must_match_the_task_oracle(self) -> None:
        tasks, _ = _inputs()
        record = _ledger_record(tasks)
        record["events"][2]["terminated"] = False
        ledger = EpisodeEventLedger.from_record(
            record,
            event_schema=tasks.capabilities.event_schemas[0],
        )
        with self.assertRaisesRegex(ValueError, "terminal flags"):
            materialize_reward_view(
                ledger,
                scoring_contract=tasks.contract,
                reward_profile_id="reach_progress/v2",
                catalog=tasks.capabilities,
            )

    def test_oracle_drift_cannot_be_hidden_by_a_new_reward_profile(self) -> None:
        tasks, ledger = _inputs()
        changed = tasks.task.to_record()
        changed["goal"]["hold_steps"] = 3
        drifted = compile_task_contract(
            TaskSpecV2.from_record(changed),
            catalog=tasks.capabilities,
        )
        with self.assertRaisesRegex(ValueError, "oracle fingerprint mismatch"):
            materialize_reward_view(
                ledger,
                scoring_contract=drifted,
                reward_profile_id="reach_progress/v2",
                catalog=tasks.capabilities,
            )

    def test_round_trip_recomputes_and_rejects_tampered_rewards(self) -> None:
        tasks, ledger = _inputs()
        view = materialize_reward_view(
            ledger,
            scoring_contract=tasks.contract,
            reward_profile_id="reach_progress/v2",
            catalog=tasks.capabilities,
        )
        restored = RewardView.from_json(
            view.to_json(),
            ledger=ledger,
            scoring_contract=tasks.contract,
            reward_profile_id="reach_progress/v2",
            catalog=tasks.capabilities,
        )
        self.assertEqual(restored, view)

        tampered = json.loads(view.to_json())
        tampered["transitions"][0]["reward"] = 99.0
        with self.assertRaisesRegex(ValueError, "deterministic replay"):
            RewardView.from_record(
                tampered,
                ledger=ledger,
                scoring_contract=tasks.contract,
                reward_profile_id="reach_progress/v2",
                catalog=tasks.capabilities,
            )

    def test_materialization_never_changes_the_fact_ledger(self) -> None:
        tasks, ledger = _inputs()
        before = ledger.to_json()
        materialize_reward_view(
            ledger,
            scoring_contract=tasks.contract,
            reward_profile_id="reach_progress/v2",
            catalog=tasks.capabilities,
        )

        self.assertEqual(ledger.to_json(), before)
        self.assertNotIn('"reward"', ledger.to_json())


if __name__ == "__main__":
    unittest.main()
