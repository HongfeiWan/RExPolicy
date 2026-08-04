"""Tests for deterministic Event Ledger to QD descriptor compilation."""

from __future__ import annotations

import copy
import hashlib
import math
import unittest

from rexpolicy.flywheel.quality_descriptor_compiler import (
    InitialStateGroupBinding,
    QualityDescriptorCompilationPolicy,
    compile_success_behavior_descriptor,
    compile_success_behavior_descriptors,
)
from rexpolicy.flywheel.quality_diversity import (
    QualityDiversityIndex,
    QualityDiversityPolicy,
    SuccessBehaviorDescriptor,
)
from rexpolicy.flywheel.success_archive import SuccessReference
from rexpolicy.tasking.canonical import canonical_fingerprint
from rexpolicy.tasking.capabilities import CapabilityCatalog
from rexpolicy.tasking.event_ledger import EpisodeEventLedger
from tools.test_tasking_capabilities import capability_record
from tools.test_tasking_event_ledger import ledger_record


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _schema():
    return CapabilityCatalog.from_record(capability_record()).event_schemas[0]


def _ledger(record: dict | None = None) -> EpisodeEventLedger:
    return EpisodeEventLedger.from_record(
        ledger_record() if record is None else record,
        event_schema=_schema(),
    )


def _policy(*, action_maximum: float = 1.0) -> QualityDiversityPolicy:
    return QualityDiversityPolicy.from_record(
        {
            "schema_version": 1,
            "policy_id": "reach_qd/ledger_v1",
            "task_id": "reach_green_cap/v3",
            "task_oracle_sha256": "f" * 64,
            "dimensions": [
                {
                    "dimension_id": "reach.action_path_l2",
                    "minimum": 0.0,
                    "maximum": action_maximum,
                    "bins": 2,
                },
                {
                    "dimension_id": "reach.control_step_count",
                    "minimum": 0.0,
                    "maximum": 64.0,
                    "bins": 2,
                },
                {
                    "dimension_id": "reach.distance_delta_m",
                    "minimum": -1.0,
                    "maximum": 1.0,
                    "bins": 2,
                },
            ],
            "quality_metric_id": "reach.final_distance_m",
            "maximize_quality": False,
            "balance_axes": [
                "behavior_cell",
                "initial_state_group",
                "reward_profile",
            ],
        }
    )


def _compiler_policy(
    policy: QualityDiversityPolicy,
) -> QualityDescriptorCompilationPolicy:
    return QualityDescriptorCompilationPolicy.from_record(
        {
            "schema_version": 1,
            "compiler_policy_id": "reach_qd/ledger_compiler_v1",
            "quality_diversity_policy_sha256": policy.fingerprint,
            "metrics": [
                {
                    "metric_id": "reach.action_path_l2",
                    "operator": "effective_action_19d.path_length_l2",
                    "signal_name": None,
                    "minimum": 0.0,
                    "maximum": policy.dimensions[0].maximum,
                },
                {
                    "metric_id": "reach.control_step_count",
                    "operator": "effective_action_19d.control_step_count",
                    "signal_name": None,
                    "minimum": 0.0,
                    "maximum": 64.0,
                },
                {
                    "metric_id": "reach.distance_delta_m",
                    "operator": "signal.delta",
                    "signal_name": "reach.distance_m",
                    "minimum": -1.0,
                    "maximum": 1.0,
                },
                {
                    "metric_id": "reach.final_distance_m",
                    "operator": "signal.final",
                    "signal_name": "reach.distance_m",
                    "minimum": 0.0,
                    "maximum": 2.0,
                },
            ],
        }
    )


def _reference(
    *,
    decision: int,
    world: int,
    archived_weight: float = 1.0,
) -> SuccessReference:
    sample_id = (
        f"g000001-r00000-e00000-d{decision:05d}-w{world:05d}"
    )
    return SuccessReference.from_record(
        {
            "schema_version": 2,
            "sample_id": sample_id,
            "source_generation": 1,
            "source_rank": 0,
            "source_episode": 0,
            "decision": decision,
            "world": world,
            "success_roles": ["direct_branch"],
            "episode_path": "archive/episode.jsonl",
            "episode_sha256": _hash("episode"),
            "episode_record_index": 0,
            "archived_weight": archived_weight,
            "task_id": "reach_green_cap/v3",
            "reward_profile_id": "reach_progress/v3",
            "simulator_fingerprint": "newton/test-v1",
        }
    )


def _state_binding(ledger: EpisodeEventLedger) -> InitialStateGroupBinding:
    return InitialStateGroupBinding.for_ledger(
        initial_state_group_id="reach_reset/centre_v1",
        initial_state_group_sha256=_hash("reach-reset-centre-v1"),
        event_ledger=ledger,
    )


def _compile(
    reference: SuccessReference,
    *,
    ledger: EpisodeEventLedger | None = None,
    policy: QualityDiversityPolicy | None = None,
    compiler_policy: QualityDescriptorCompilationPolicy | None = None,
    state_binding: InitialStateGroupBinding | None = None,
) -> SuccessBehaviorDescriptor:
    actual_ledger = _ledger() if ledger is None else ledger
    actual_policy = _policy() if policy is None else policy
    actual_compiler = (
        _compiler_policy(actual_policy)
        if compiler_policy is None
        else compiler_policy
    )
    actual_binding = (
        _state_binding(actual_ledger)
        if state_binding is None
        else state_binding
    )
    return compile_success_behavior_descriptor(
        policy=actual_policy,
        compiler_policy=actual_compiler,
        success_reference=reference,
        event_ledger=actual_ledger,
        event_schema=_schema(),
        reward_profile_sha256=_hash("reach-progress-v3"),
        initial_state_group=actual_binding,
    )


class TestQualityDescriptorCompiler(unittest.TestCase):
    def test_exact_hashes_and_allowlisted_aggregations(self) -> None:
        ledger = _ledger()
        reference = _reference(decision=1, world=0)
        descriptor = _compile(reference, ledger=ledger)
        metrics = dict(descriptor.metrics)

        self.assertEqual(
            descriptor.success_reference_sha256,
            canonical_fingerprint(reference.to_record()),
        )
        self.assertEqual(descriptor.event_ledger_sha256, ledger.fingerprint)
        self.assertEqual(
            descriptor.compilation_policy_sha256,
            _compiler_policy(_policy()).fingerprint,
        )
        self.assertEqual(descriptor.task_oracle_sha256, "f" * 64)
        self.assertEqual(
            descriptor.reward_profile_sha256,
            _hash("reach-progress-v3"),
        )
        self.assertEqual(
            descriptor.initial_state_group_sha256,
            _hash("reach-reset-centre-v1"),
        )
        self.assertAlmostEqual(metrics["reach.distance_delta_m"], -0.05)
        self.assertAlmostEqual(metrics["reach.final_distance_m"], 0.03)
        self.assertEqual(metrics["reach.control_step_count"], 1.0)
        self.assertAlmostEqual(
            metrics["reach.action_path_l2"],
            math.sqrt(19 * 0.02**2),
        )

    def test_policy_binding_and_output_round_trip_are_canonical(self) -> None:
        policy = _policy()
        compiler = _compiler_policy(policy)
        ledger = _ledger()
        binding = _state_binding(ledger)
        descriptor = _compile(
            _reference(decision=0, world=1),
            ledger=ledger,
            policy=policy,
            compiler_policy=compiler,
            state_binding=binding,
        )

        self.assertEqual(
            QualityDescriptorCompilationPolicy.from_json(compiler.to_json()),
            compiler,
        )
        self.assertEqual(
            InitialStateGroupBinding.from_json(binding.to_json()),
            binding,
        )
        self.assertEqual(
            SuccessBehaviorDescriptor.from_record(descriptor.to_record()),
            descriptor,
        )

        changed = copy.deepcopy(compiler.to_record())
        changed["quality_diversity_policy_sha256"] = _hash("other-policy")
        with self.assertRaisesRegex(ValueError, "another QD policy"):
            _compile(
                _reference(decision=0, world=1),
                ledger=ledger,
                policy=policy,
                compiler_policy=QualityDescriptorCompilationPolicy.from_record(
                    changed
                ),
                state_binding=binding,
            )

    def test_equal_metric_values_still_bind_the_exact_compiler_policy(self) -> None:
        policy = _policy()
        reference = _reference(decision=0, world=0)
        final_compiler = _compiler_policy(policy)
        minimum_record = copy.deepcopy(final_compiler.to_record())
        minimum_record["metrics"][-1]["operator"] = "signal.minimum"
        minimum_compiler = QualityDescriptorCompilationPolicy.from_record(
            minimum_record
        )

        final_descriptor = _compile(
            reference,
            policy=policy,
            compiler_policy=final_compiler,
        )
        minimum_descriptor = _compile(
            reference,
            policy=policy,
            compiler_policy=minimum_compiler,
        )

        self.assertEqual(final_descriptor.metrics, minimum_descriptor.metrics)
        self.assertNotEqual(
            final_descriptor.compilation_policy_sha256,
            minimum_descriptor.compilation_policy_sha256,
        )
        self.assertNotEqual(
            final_descriptor.fingerprint,
            minimum_descriptor.fingerprint,
        )

    def test_ledger_state_oracle_and_reference_tampering_fail_closed(self) -> None:
        ledger = _ledger()
        binding_record = _state_binding(ledger).to_record()
        binding_record["event_ledger_sha256"] = _hash("other-ledger")
        with self.assertRaisesRegex(ValueError, "another Event Ledger"):
            _compile(
                _reference(decision=0, world=0),
                ledger=ledger,
                state_binding=InitialStateGroupBinding.from_record(binding_record),
            )

        changed_ledger = ledger_record()
        changed_ledger["bindings"]["task_oracle_sha256"] = "a" * 64
        other_ledger = _ledger(changed_ledger)
        with self.assertRaisesRegex(ValueError, "task oracle"):
            _compile(
                _reference(decision=0, world=0),
                ledger=other_ledger,
            )

        changed_reference = _reference(decision=0, world=0).to_record()
        changed_reference["decision"] = 1
        with self.assertRaisesRegex(ValueError, "sample_id/decision/world"):
            _compile(SuccessReference.from_record(changed_reference))

        changed_signals = ledger_record()
        changed_signals["events"][2]["post_signals"]["reach.distance_m"] = 0.095
        altered_ledger = _ledger(changed_signals)
        with self.assertRaisesRegex(ValueError, "another Event Ledger"):
            _compile(
                _reference(decision=0, world=1),
                ledger=altered_ledger,
                state_binding=_state_binding(ledger),
            )

    def test_metric_program_is_exact_numeric_and_range_checked(self) -> None:
        policy = _policy()
        compiler_record = _compiler_policy(policy).to_record()
        compiler_record["metrics"][0]["operator"] = "model.embedding"
        with self.assertRaisesRegex(ValueError, "Unsupported behavior aggregation"):
            QualityDescriptorCompilationPolicy.from_record(compiler_record)

        boolean_record = _compiler_policy(policy).to_record()
        boolean_record["metrics"][3]["signal_name"] = (
            "reach.contact_violation"
        )
        with self.assertRaisesRegex(ValueError, "numeric signals"):
            _compile(
                _reference(decision=0, world=0),
                policy=policy,
                compiler_policy=QualityDescriptorCompilationPolicy.from_record(
                    boolean_record
                ),
            )

        missing_record = _compiler_policy(policy).to_record()
        missing_record["metrics"].pop()
        with self.assertRaisesRegex(ValueError, "does not exactly match"):
            _compile(
                _reference(decision=0, world=0),
                policy=policy,
                compiler_policy=QualityDescriptorCompilationPolicy.from_record(
                    missing_record
                ),
            )

        mismatched_range = _compiler_policy(policy).to_record()
        mismatched_range["metrics"][0]["maximum"] = 0.5
        with self.assertRaisesRegex(ValueError, "does not match QD dimension"):
            _compile(
                _reference(decision=0, world=0),
                policy=policy,
                compiler_policy=QualityDescriptorCompilationPolicy.from_record(
                    mismatched_range
                ),
            )

        narrow = _policy(action_maximum=0.001)
        with self.assertRaisesRegex(ValueError, "compiler-policy range"):
            _compile(
                _reference(decision=0, world=1),
                policy=narrow,
                compiler_policy=_compiler_policy(narrow),
            )

    def test_all_signal_operators_use_only_candidate_frames(self) -> None:
        policy = QualityDiversityPolicy.from_record(
            {
                "schema_version": 1,
                "policy_id": "reach_qd/signal_operators_v1",
                "task_id": "reach_green_cap/v3",
                "task_oracle_sha256": "f" * 64,
                "dimensions": [
                    {
                        "dimension_id": "reach.distance_delta_m",
                        "minimum": -2.0,
                        "maximum": 2.0,
                        "bins": 2,
                    },
                    {
                        "dimension_id": "reach.distance_initial_m",
                        "minimum": 0.0,
                        "maximum": 2.0,
                        "bins": 2,
                    },
                    {
                        "dimension_id": "reach.distance_maximum_m",
                        "minimum": 0.0,
                        "maximum": 2.0,
                        "bins": 2,
                    },
                    {
                        "dimension_id": "reach.distance_minimum_m",
                        "minimum": 0.0,
                        "maximum": 2.0,
                        "bins": 2,
                    },
                ],
                "quality_metric_id": "reach.distance_final_m",
                "maximize_quality": False,
                "balance_axes": [
                    "behavior_cell",
                    "initial_state_group",
                    "reward_profile",
                ],
            }
        )
        compiler = QualityDescriptorCompilationPolicy.from_record(
            {
                "schema_version": 1,
                "compiler_policy_id": "reach_qd/signal_compiler_v1",
                "quality_diversity_policy_sha256": policy.fingerprint,
                "metrics": [
                    {
                        "metric_id": "reach.distance_delta_m",
                        "operator": "signal.delta",
                        "signal_name": "reach.distance_m",
                        "minimum": -2.0,
                        "maximum": 2.0,
                    },
                    {
                        "metric_id": "reach.distance_final_m",
                        "operator": "signal.final",
                        "signal_name": "reach.distance_m",
                        "minimum": 0.0,
                        "maximum": 2.0,
                    },
                    {
                        "metric_id": "reach.distance_initial_m",
                        "operator": "signal.initial",
                        "signal_name": "reach.distance_m",
                        "minimum": 0.0,
                        "maximum": 2.0,
                    },
                    {
                        "metric_id": "reach.distance_maximum_m",
                        "operator": "signal.maximum",
                        "signal_name": "reach.distance_m",
                        "minimum": 0.0,
                        "maximum": 2.0,
                    },
                    {
                        "metric_id": "reach.distance_minimum_m",
                        "operator": "signal.minimum",
                        "signal_name": "reach.distance_m",
                        "minimum": 0.0,
                        "maximum": 2.0,
                    },
                ],
            }
        )
        metrics = dict(
            _compile(
                _reference(decision=0, world=0),
                policy=policy,
                compiler_policy=compiler,
            ).metrics
        )

        self.assertAlmostEqual(metrics["reach.distance_initial_m"], 0.10)
        self.assertAlmostEqual(metrics["reach.distance_final_m"], 0.08)
        self.assertAlmostEqual(metrics["reach.distance_minimum_m"], 0.08)
        self.assertAlmostEqual(metrics["reach.distance_maximum_m"], 0.10)
        self.assertAlmostEqual(metrics["reach.distance_delta_m"], -0.02)

    def test_batch_retains_all_successes_and_non_elites(self) -> None:
        ledger = _ledger()
        policy = _policy()
        references = (
            _reference(decision=0, world=1),
            _reference(decision=0, world=0),
        )
        descriptors = compile_success_behavior_descriptors(
            policy=policy,
            compiler_policy=_compiler_policy(policy),
            success_references=references,
            event_ledger=ledger,
            event_schema=_schema(),
            reward_profile_sha256_by_id={
                "reach_progress/v3": _hash("reach-progress-v3")
            },
            initial_state_group=_state_binding(ledger),
        )
        index = QualityDiversityIndex(policy=policy, descriptors=descriptors)

        self.assertEqual(len(descriptors), 2)
        self.assertEqual(
            tuple(item.sample_id for item in descriptors),
            tuple(sorted(item.sample_id for item in references)),
        )
        self.assertEqual(len(index.cells), 1)
        self.assertEqual(len(index.cells[0].member_descriptor_sha256s), 2)
        elite = next(
            item
            for item in descriptors
            if item.fingerprint == index.cells[0].elite_descriptor_sha256
        )
        self.assertEqual(elite.sample_id, references[1].sample_id)

        with self.assertRaisesRegex(ValueError, "fingerprint set must be exact"):
            compile_success_behavior_descriptors(
                policy=policy,
                compiler_policy=_compiler_policy(policy),
                success_references=references,
                event_ledger=ledger,
                event_schema=_schema(),
                reward_profile_sha256_by_id={},
                initial_state_group=_state_binding(ledger),
            )

    def test_reference_weights_cannot_change_behavior_metrics(self) -> None:
        normal = _compile(_reference(decision=0, world=1, archived_weight=1.0))
        reweighted = _compile(
            _reference(decision=0, world=1, archived_weight=999.0)
        )

        self.assertEqual(normal.metrics, reweighted.metrics)
        self.assertNotEqual(
            normal.success_reference_sha256,
            reweighted.success_reference_sha256,
        )


if __name__ == "__main__":
    unittest.main()
