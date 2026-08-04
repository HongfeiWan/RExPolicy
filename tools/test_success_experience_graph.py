"""Tests for metadata-only future windows over terminal success paths."""

from __future__ import annotations

import copy
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from rexpolicy.flywheel.derived_views import write_derived_view
from rexpolicy.flywheel.experience import (
    DIRECT_SUCCESS_ROLE,
    SUCCESS_PATH_ROLE,
)
from rexpolicy.flywheel.success_archive import SuccessReference
from rexpolicy.replay.success_graph import (
    FutureWindowPolicy,
    SuccessExperienceGraph,
    compile_success_experience_graph,
)
from rexpolicy.tasking.capabilities import CapabilityCatalog
from rexpolicy.tasking.event_ledger import EpisodeEventLedger
from tools.test_tasking_capabilities import capability_record
from tools.test_tasking_event_ledger import ledger_record

_SAMPLE_00 = "g000001-r00000-e00000-d00000-w00000"
_SAMPLE_01 = "g000001-r00000-e00000-d00000-w00001"
_SAMPLE_10 = "g000001-r00000-e00000-d00001-w00000"
_SAMPLE_11 = "g000001-r00000-e00000-d00001-w00001"


def _candidate(
    sample_id: str,
    *,
    world: int,
    roles: tuple[str, ...] = (),
    success: bool = False,
    chosen: bool = False,
    action_value: float = 7.0,
) -> dict:
    return {
        "sample_id": sample_id,
        "success_roles": list(roles),
        "world": world,
        "score": 99.0 if success else -3.0,
        "success": success,
        "failure": False,
        "safety_violation": False,
        "failure_reasons": [],
        "terminated": success,
        "truncated": False,
        "valid_steps": 1,
        "rewards": [41.0],
        "advantage": 2.0,
        "sample_weight": 1.0,
        "selected_for_training": bool(roles),
        "chosen_for_continuation": chosen,
        "action": {"action_19d": [[action_value] * 19]},
    }


def _decision(index: int, candidates: list[dict]) -> dict:
    return {
        "decision_index": index,
        "start_control_step": index,
        "physical_replay_gate": {"schema_version": 3},
        "advantage_baseline": 0.0,
        "selection_mode": "success" if index else "advantage",
        "success_constraint_active": index == 1,
        "safe_candidate_count": len(candidates),
        "selected_chunk_count": sum(
            bool(item["selected_for_training"]) for item in candidates
        ),
        "normalized_action_pairwise_rms_mean": 0.1,
        "normalized_action_pairwise_rms_max": 0.2,
        "candidates": candidates,
    }


def _episode_record() -> dict:
    return {
        "schema_version": 5,
        "generation": 1,
        "data_generation": 1,
        "sampling_policy_generation": 0,
        "policy_version": "generation-000000",
        "rank": 0,
        "episode": 0,
        "task_id": "reach_green_cap/v1",
        "reward_profile_id": "reach_progress/v1",
        "simulator_fingerprint": "9" * 64,
        "reset_recipe": {
            "environment": "GrootNewtonEnv",
            "seed": 123,
            "initial_state": {"ignored_by_graph": [1.0, 2.0]},
        },
        "instruction": "reach safely",
        "selected_path": {
            "score": 123.0,
            "success": True,
            "rewards": [40.0, 41.0],
            "actions": [{"action_19d": [[8.0] * 19]}],
        },
        "decisions": [
            _decision(
                0,
                [
                    _candidate(
                        _SAMPLE_00,
                        world=0,
                        roles=(SUCCESS_PATH_ROLE,),
                        chosen=True,
                    ),
                    _candidate(_SAMPLE_01, world=1),
                ],
            ),
            _decision(
                1,
                [
                    _candidate(
                        _SAMPLE_10,
                        world=0,
                        roles=(DIRECT_SUCCESS_ROLE, SUCCESS_PATH_ROLE),
                        success=True,
                        chosen=True,
                    ),
                    _candidate(
                        _SAMPLE_11,
                        world=1,
                        roles=(DIRECT_SUCCESS_ROLE,),
                        success=True,
                    ),
                ],
            ),
        ],
        "success_samples": [
            {
                "sample_id": _SAMPLE_00,
                "generation": 1,
                "rank": 0,
                "episode": 0,
                "decision": 0,
                "world": 0,
                "roles": [SUCCESS_PATH_ROLE],
                "weight": 1.0,
            },
            {
                "sample_id": _SAMPLE_10,
                "generation": 1,
                "rank": 0,
                "episode": 0,
                "decision": 1,
                "world": 0,
                "roles": [DIRECT_SUCCESS_ROLE, SUCCESS_PATH_ROLE],
                "weight": 1.0,
            },
            {
                "sample_id": _SAMPLE_11,
                "generation": 1,
                "rank": 0,
                "episode": 0,
                "decision": 1,
                "world": 1,
                "roles": [DIRECT_SUCCESS_ROLE],
                "weight": 1.0,
            },
        ],
    }


def _ledger(*, changed: bool = False) -> EpisodeEventLedger:
    record = ledger_record()
    record["events"][6]["terminated"] = True
    if changed:
        record["events"][6]["post_dynamics_digest_sha256"] = "3" * 64
    schema = CapabilityCatalog.from_record(capability_record()).event_schemas[0]
    return EpisodeEventLedger.from_record(record, event_schema=schema)


def _reference(sample_id: str, roles: tuple[str, ...]) -> SuccessReference:
    decision = 0 if sample_id == _SAMPLE_00 else 1
    world = 0 if sample_id in {_SAMPLE_00, _SAMPLE_10} else 1
    reference = SuccessReference(
        schema_version=2,
        sample_id=sample_id,
        source_generation=1,
        source_rank=0,
        source_episode=0,
        decision=decision,
        world=world,
        success_roles=roles,
        episode_path="archive/generation-000001/rank-00000.episodes.jsonl",
        episode_sha256="a" * 64,
        episode_record_index=0,
        archived_weight=1.0,
        task_id="reach_green_cap/v1",
        reward_profile_id="reach_progress/v1",
        simulator_fingerprint="9" * 64,
    )
    reference.validate()
    return reference


def _references() -> tuple[SuccessReference, ...]:
    return (
        _reference(_SAMPLE_00, (SUCCESS_PATH_ROLE,)),
        _reference(
            _SAMPLE_10,
            (DIRECT_SUCCESS_ROLE, SUCCESS_PATH_ROLE),
        ),
        _reference(_SAMPLE_11, (DIRECT_SUCCESS_ROLE,)),
    )


def _graph() -> SuccessExperienceGraph:
    return compile_success_experience_graph(
        episode_record=_episode_record(),
        event_ledger=_ledger(),
        success_references=_references(),
        window_policy=FutureWindowPolicy(
            action_horizon=2,
            future_horizon=2,
        ),
    )


class TestSuccessExperienceGraph(unittest.TestCase):
    def test_compiles_shared_prefix_into_multiple_terminal_paths(self) -> None:
        graph = _graph()

        self.assertEqual(len(graph.terminal_paths), 2)
        self.assertEqual(graph.task_id, "reach_green_cap/v1")
        self.assertEqual(graph.task_contract_id, "reach_green_cap/v2")
        by_terminal = {
            item.terminal_sample_id: item.transition_sequences
            for item in graph.terminal_paths
        }
        self.assertEqual(by_terminal[_SAMPLE_10], (1, 5))
        self.assertEqual(by_terminal[_SAMPLE_11], (1, 6))
        self.assertEqual(len(graph.experiences), 4)
        self.assertEqual(
            {item.locator.current_transition_sequence for item in graph.experiences},
            {1, 5, 6},
        )

    def test_graph_is_canonical_and_persists_only_locators(self) -> None:
        graph = _graph()
        encoded = graph.to_json()
        restored = SuccessExperienceGraph.from_json(encoded)

        self.assertEqual(restored, graph)
        self.assertEqual(restored.fingerprint, graph.fingerprint)
        for forbidden in (
            '"action"',
            '"action_chunk"',
            '"current_state"',
            '"feature"',
            '"future_state_sequence"',
            '"image"',
            '"reward"',
            '"score"',
        ):
            self.assertNotIn(forbidden, encoded)

        unknown = copy.deepcopy(graph.to_record())
        unknown["score"] = 1.0
        with self.assertRaisesRegex(ValueError, "unknown fields"):
            SuccessExperienceGraph.from_record(unknown)
        changed_id = copy.deepcopy(graph.to_record())
        changed_id["experiences"][0]["experience_id"] = "f" * 64
        with self.assertRaisesRegex(ValueError, "identity mismatch"):
            SuccessExperienceGraph.from_record(changed_id)

    def test_graph_publishes_as_idempotent_derived_view(self) -> None:
        graph = _graph()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = write_derived_view(run_dir=root, view=graph)
            second = write_derived_view(run_dir=root, view=graph)
            self.assertTrue(first.created)
            self.assertFalse(second.created)
            self.assertIn(graph.fingerprint, first.relative_path)
            restored = SuccessExperienceGraph.from_json(
                (root / first.relative_path).read_text(encoding="ascii")
            )
            self.assertEqual(restored, graph)

    def test_materialization_uses_ledger_not_episode_payload(self) -> None:
        graph = _graph()
        experience = next(
            item
            for item in graph.experiences
            if item.locator.terminal_sample_id == _SAMPLE_11
            and item.locator.current_transition_sequence == 1
        )
        materialized = experience.materialize(_ledger())

        self.assertEqual(materialized.current_witness.control_step, 0)
        self.assertEqual(materialized.current_witness.dynamics_digest_sha256, "d" * 64)
        self.assertEqual(materialized.action_chunk[0], (0.0,) * 19)
        self.assertEqual(materialized.action_chunk[1], (0.0,) * 19)
        self.assertEqual(
            tuple(
                item.dynamics_digest_sha256
                for item in materialized.future_witnesses
            ),
            ("e" * 64, "2" * 64),
        )
        self.assertEqual(materialized.future_witness.control_step, 2)
        with self.assertRaisesRegex(ValueError, "fingerprint mismatch"):
            experience.materialize(_ledger(changed=True))

    def test_rejects_incomplete_or_noncanonical_success_authority(self) -> None:
        failure = _episode_record()
        failure["selected_path"]["success"] = False
        with self.assertRaisesRegex(ValueError, "successful episode"):
            compile_success_experience_graph(
                episode_record=failure,
                event_ledger=_ledger(),
                success_references=_references(),
                window_policy=FutureWindowPolicy(1, 2),
            )

        with self.assertRaisesRegex(ValueError, "exactly cover"):
            compile_success_experience_graph(
                episode_record=_episode_record(),
                event_ledger=_ledger(),
                success_references=_references()[:-1],
                window_policy=FutureWindowPolicy(1, 2),
            )

        references = list(_references())
        references[1] = replace(
            references[1],
            success_roles=(SUCCESS_PATH_ROLE, DIRECT_SUCCESS_ROLE),
        )
        with self.assertRaisesRegex(ValueError, "sorted and unique"):
            compile_success_experience_graph(
                episode_record=_episode_record(),
                event_ledger=_ledger(),
                success_references=references,
                window_policy=FutureWindowPolicy(1, 2),
            )


if __name__ == "__main__":
    unittest.main()
