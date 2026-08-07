"""CPU-only tests for the fixed Grasp-Lift no-z validation selector."""

from __future__ import annotations

import ast
import copy
import json
import math
import unittest
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from rexpolicy.stage0.grasp_lift_no_z_selection import (
    GRASP_LIFT_NO_Z_POLICY_NOISE_INDICES,
    GRASP_LIFT_NO_Z_SELECTION_PROTOCOL_SHA256,
    GRASP_LIFT_NO_Z_SELECTION_SCHEMA_ID,
    GRASP_LIFT_NO_Z_VALIDATION_RESET_SEEDS,
    GRASP_LIFT_NO_Z_VALIDATION_STEPS,
    GRASP_LIFT_NO_Z_VALIDATION_TRAINING_SEEDS,
    GraspLiftNoZSeedCheckpointEvidence,
    GraspLiftNoZValidationRollout,
    grasp_lift_no_z_selection_protocol_record,
    select_grasp_lift_no_z_checkpoint,
    verify_grasp_lift_no_z_selection_record,
)
from rexpolicy.stage0.types import canonical_fingerprint


def _model_sha256(step: int, seed: int) -> str:
    return canonical_fingerprint({"checkpoint_step": step, "training_seed": seed})


def _reseal(record: dict[str, Any]) -> None:
    unsealed = dict(record)
    unsealed.pop("self_sha256")
    record["self_sha256"] = canonical_fingerprint(unsealed)


def _rollouts(
    *,
    failure_keys: frozenset[tuple[int, int]] = frozenset(),
    flag: str | None = None,
    flag_key: tuple[int, int] | None = None,
) -> tuple[GraspLiftNoZValidationRollout, ...]:
    values = []
    for reset_seed in GRASP_LIFT_NO_Z_VALIDATION_RESET_SEEDS:
        for noise_index in GRASP_LIFT_NO_Z_POLICY_NOISE_INDICES:
            key = (reset_seed, noise_index)
            values.append(
                GraspLiftNoZValidationRollout(
                    reset_seed=reset_seed,
                    policy_noise_index=noise_index,
                    success=key not in failure_keys,
                    safety_violation=flag == "safety" and key == flag_key,
                    integrity_violation=flag == "integrity" and key == flag_key,
                    oracle_disagreement=(
                        flag == "oracle_disagreement" and key == flag_key
                    ),
                )
            )
    return tuple(values)


def _evidence(
    step: int,
    seed: int,
    *,
    failure_keys: frozenset[tuple[int, int]] = frozenset(),
    flag: str | None = None,
    flag_key: tuple[int, int] | None = None,
    diagnostic: Mapping[str, Any] | None = None,
) -> GraspLiftNoZSeedCheckpointEvidence:
    return GraspLiftNoZSeedCheckpointEvidence(
        checkpoint_step=step,
        training_seed=seed,
        model_sha256=_model_sha256(step, seed),
        rollouts=_rollouts(
            failure_keys=failure_keys,
            flag=flag,
            flag_key=flag_key,
        ),
        offline_mse_diagnostic=(
            {"normalized": 0.5, "train_mean": 0.4, "ratio": 1.25}
            if diagnostic is None
            else diagnostic
        ),
    )


def _inventory(
    overrides: Mapping[tuple[int, int], Mapping[str, Any]] | None = None,
) -> tuple[GraspLiftNoZSeedCheckpointEvidence, ...]:
    changes = {} if overrides is None else dict(overrides)
    return tuple(
        _evidence(step, seed, **changes.get((step, seed), {}))
        for step in GRASP_LIFT_NO_Z_VALIDATION_STEPS
        for seed in GRASP_LIFT_NO_Z_VALIDATION_TRAINING_SEEDS
    )


def _two_failures() -> frozenset[tuple[int, int]]:
    return frozenset(
        {
            (GRASP_LIFT_NO_Z_VALIDATION_RESET_SEEDS[0], 0),
            (GRASP_LIFT_NO_Z_VALIDATION_RESET_SEEDS[1], 0),
        }
    )


def _only_steps_eligible(
    eligible_steps: set[int],
) -> tuple[GraspLiftNoZSeedCheckpointEvidence, ...]:
    overrides = {
        (step, seed): {"failure_keys": _two_failures()}
        for step in GRASP_LIFT_NO_Z_VALIDATION_STEPS
        if step not in eligible_steps
        for seed in GRASP_LIFT_NO_Z_VALIDATION_TRAINING_SEEDS
    }
    return _inventory(overrides)


class GraspLiftNoZSelectionTest(unittest.TestCase):
    def test_public_protocol_record_has_one_stable_fingerprint(self) -> None:
        first = grasp_lift_no_z_selection_protocol_record()
        second = grasp_lift_no_z_selection_protocol_record()

        self.assertEqual(first, second)
        self.assertIsNot(first, second)
        self.assertEqual(
            canonical_fingerprint(first),
            GRASP_LIFT_NO_Z_SELECTION_PROTOCOL_SHA256,
        )
        first["training_seeds"].append(-1)
        self.assertEqual(
            tuple(second["training_seeds"]),
            GRASP_LIFT_NO_Z_VALIDATION_TRAINING_SEEDS,
        )

    def test_complete_passing_inventory_selects_one_same_step_seed_family(
        self,
    ) -> None:
        selection = select_grasp_lift_no_z_checkpoint(_inventory())

        self.assertTrue(selection.gate_passed)
        self.assertEqual(selection.selected_checkpoint_step, 5_000)
        self.assertEqual(selection.eligible_runs, (GRASP_LIFT_NO_Z_VALIDATION_STEPS,))
        self.assertEqual(
            tuple(item.training_seed for item in selection.selected_family),
            GRASP_LIFT_NO_Z_VALIDATION_TRAINING_SEEDS,
        )
        self.assertTrue(
            all(
                item.checkpoint_step == selection.selected_checkpoint_step
                for item in selection.selected_family
            )
        )
        self.assertEqual(len(selection.evidence), 60)
        self.assertEqual(len(selection.checkpoint_results), 20)
        self.assertTrue(all(item.eligible for item in selection.checkpoint_results))
        self.assertTrue(
            all(item.success_count == 72 for item in selection.checkpoint_results)
        )

        record = selection.to_record()
        self.assertEqual(record["schema_id"], GRASP_LIFT_NO_Z_SELECTION_SCHEMA_ID)
        self.assertEqual(record["inventory"]["rollout_cells"], 1_440)
        self.assertEqual(record["inventory"]["seed_checkpoint_entries"], 60)
        self.assertEqual(record["self_sha256"], selection.sha256)
        verify_grasp_lift_no_z_selection_record(record)
        json.dumps(record, allow_nan=False, sort_keys=True)

    def test_longest_run_lower_median_wins_and_equal_length_prefers_earlier(
        self,
    ) -> None:
        longest = select_grasp_lift_no_z_checkpoint(
            _only_steps_eligible({500, 1_000, 1_500, 5_000, 5_500, 6_000, 6_500})
        )
        self.assertEqual(
            longest.eligible_runs,
            ((500, 1_000, 1_500), (5_000, 5_500, 6_000, 6_500)),
        )
        self.assertEqual(longest.selected_checkpoint_step, 5_500)

        tied = select_grasp_lift_no_z_checkpoint(
            _only_steps_eligible({500, 1_000, 1_500, 5_000, 5_500, 6_000})
        )
        self.assertEqual(tied.selected_checkpoint_step, 1_000)

    def test_two_passing_seeds_and_exact_reset_floor_are_eligible(self) -> None:
        reset0 = GRASP_LIFT_NO_Z_VALIDATION_RESET_SEEDS[0]
        seed0, seed1, seed2 = GRASP_LIFT_NO_Z_VALIDATION_TRAINING_SEEDS
        overrides = {
            (500, seed0): {"failure_keys": frozenset({(reset0, 0)})},
            (500, seed1): {"failure_keys": frozenset({(reset0, 1)})},
            (500, seed2): {
                "failure_keys": frozenset(
                    (reset_seed, 2)
                    for reset_seed in GRASP_LIFT_NO_Z_VALIDATION_RESET_SEEDS
                )
            },
        }
        result = select_grasp_lift_no_z_checkpoint(_inventory(overrides))
        checkpoint = result.checkpoint_results[0]

        self.assertTrue(checkpoint.eligible)
        self.assertEqual(checkpoint.passing_seed_count, 2)
        self.assertEqual(
            tuple(item.success_count for item in checkpoint.seed_results),
            (23, 23, 18),
        )
        self.assertEqual(checkpoint.successes_by_reset[0], 9)
        self.assertEqual(checkpoint.admission_failures, ())

    def test_seed_total_and_per_reset_floors_are_both_explicit(self) -> None:
        seed0 = GRASP_LIFT_NO_Z_VALIDATION_TRAINING_SEEDS[0]
        reset0 = GRASP_LIFT_NO_Z_VALIDATION_RESET_SEEDS[0]
        overrides = {
            (500, seed0): {"failure_keys": frozenset({(reset0, 0), (reset0, 1)})}
        }
        selection = select_grasp_lift_no_z_checkpoint(_inventory(overrides))
        seed_result = selection.checkpoint_results[0].seed_results[0]

        self.assertFalse(seed_result.success_gate_passed)
        self.assertEqual(seed_result.success_count, 22)
        self.assertEqual(seed_result.successes_by_reset[0], 2)
        self.assertEqual(
            seed_result.admission_failures,
            (
                "minimum_seed_successes_not_met",
                "minimum_seed_reset_successes_not_met",
            ),
        )

    def test_cross_seed_reset_floor_is_not_replaced_by_passing_seed_count(
        self,
    ) -> None:
        reset0 = GRASP_LIFT_NO_Z_VALIDATION_RESET_SEEDS[0]
        seed0, seed1, seed2 = GRASP_LIFT_NO_Z_VALIDATION_TRAINING_SEEDS
        overrides = {
            (500, seed0): {"failure_keys": frozenset({(reset0, 0)})},
            (500, seed1): {"failure_keys": frozenset({(reset0, 1)})},
            (500, seed2): {
                "failure_keys": frozenset(
                    (reset0, noise) for noise in GRASP_LIFT_NO_Z_POLICY_NOISE_INDICES
                )
            },
        }
        checkpoint = select_grasp_lift_no_z_checkpoint(
            _inventory(overrides)
        ).checkpoint_results[0]

        self.assertEqual(checkpoint.passing_seed_count, 2)
        self.assertEqual(checkpoint.successes_by_reset[0], 6)
        self.assertFalse(checkpoint.eligible)
        self.assertEqual(
            checkpoint.admission_failures,
            ("minimum_checkpoint_reset_successes_not_met",),
        )

    def test_any_of_72_hard_failures_rejects_the_common_step(self) -> None:
        seed = GRASP_LIFT_NO_Z_VALIDATION_TRAINING_SEEDS[2]
        cell = (GRASP_LIFT_NO_Z_VALIDATION_RESET_SEEDS[-1], 3)
        expected = {
            "safety": "safety_violation",
            "integrity": "integrity_violation",
            "oracle_disagreement": "oracle_disagreement",
        }
        for flag, failure in expected.items():
            with self.subTest(flag=flag):
                result = select_grasp_lift_no_z_checkpoint(
                    _inventory({(500, seed): {"flag": flag, "flag_key": cell}})
                ).checkpoint_results[0]
                self.assertFalse(result.eligible)
                self.assertIn(failure, result.admission_failures)
                self.assertTrue(
                    all(item.success_gate_passed for item in result.seed_results)
                )

    def test_isolated_or_two_checkpoint_success_fails_closed(self) -> None:
        selection = select_grasp_lift_no_z_checkpoint(
            _only_steps_eligible({500, 1_000, 3_000})
        )

        self.assertFalse(selection.gate_passed)
        self.assertIsNone(selection.selected_checkpoint_step)
        self.assertEqual(selection.selected_family, ())
        self.assertEqual(selection.eligible_runs, ((500, 1_000), (3_000,)))
        self.assertIn("at least three", selection.selection_reason)

    def test_offline_mse_is_finite_preserved_and_selection_irrelevant(self) -> None:
        low = _inventory(
            {
                (step, seed): {
                    "diagnostic": {
                        "normalized": 0.001,
                        "xyz": 0.002,
                        "hand": 0.003,
                        "baseline": {"train_mean": 100.0},
                        "ratio": 0.00001,
                    }
                }
                for step in GRASP_LIFT_NO_Z_VALIDATION_STEPS
                for seed in GRASP_LIFT_NO_Z_VALIDATION_TRAINING_SEEDS
            }
        )
        high = _inventory(
            {
                (step, seed): {
                    "diagnostic": {
                        "normalized": 1_000_000.0,
                        "xyz": 20.0,
                        "hand": 30.0,
                        "baseline": {"train_mean": 0.0001},
                        "ratio": 10_000_000_000.0,
                    }
                }
                for step in GRASP_LIFT_NO_Z_VALIDATION_STEPS
                for seed in GRASP_LIFT_NO_Z_VALIDATION_TRAINING_SEEDS
            }
        )
        low_result = select_grasp_lift_no_z_checkpoint(low)
        high_result = select_grasp_lift_no_z_checkpoint(high)

        self.assertEqual(
            low_result.selected_checkpoint_step,
            high_result.selected_checkpoint_step,
        )
        self.assertEqual(
            tuple(item.eligible for item in low_result.checkpoint_results),
            tuple(item.eligible for item in high_result.checkpoint_results),
        )
        self.assertEqual(
            low_result.to_record()["protocol"]["diagnostics"]["offline_mse"],
            "recorded_only_never_admission_or_ranking/v1",
        )
        self.assertEqual(
            high_result.to_record()["evidence"][0]["offline_mse_diagnostic"]["ratio"],
            10_000_000_000.0,
        )

    def test_inventory_order_duplicates_and_missing_entries_fail_closed(self) -> None:
        values = _inventory()
        with self.assertRaisesRegex(ValueError, "20-step by 3-seed"):
            select_grasp_lift_no_z_checkpoint(values[:-1])
        with self.assertRaisesRegex(ValueError, "20-step by 3-seed"):
            select_grasp_lift_no_z_checkpoint((values[1], values[0], *values[2:]))
        with self.assertRaisesRegex(ValueError, "20-step by 3-seed"):
            select_grasp_lift_no_z_checkpoint((values[0], values[0], *values[2:]))
        with self.assertRaisesRegex(ValueError, "canonically ordered"):
            select_grasp_lift_no_z_checkpoint(
                values,
                validation_reset_seeds=tuple(
                    reversed(GRASP_LIFT_NO_Z_VALIDATION_RESET_SEEDS)
                ),
            )

        first = values[0]
        with self.assertRaisesRegex(ValueError, "6-reset by 4-noise"):
            GraspLiftNoZSeedCheckpointEvidence(
                checkpoint_step=first.checkpoint_step,
                training_seed=first.training_seed,
                model_sha256=first.model_sha256,
                rollouts=(first.rollouts[1], first.rollouts[0], *first.rollouts[2:]),
            )
        with self.assertRaisesRegex(ValueError, "6-reset by 4-noise"):
            GraspLiftNoZSeedCheckpointEvidence(
                checkpoint_step=first.checkpoint_step,
                training_seed=first.training_seed,
                model_sha256=first.model_sha256,
                rollouts=first.rollouts[:-1],
            )

    def test_cell_identity_boolean_digest_and_diagnostic_types_are_strict(self) -> None:
        base = {
            "reset_seed": GRASP_LIFT_NO_Z_VALIDATION_RESET_SEEDS[0],
            "policy_noise_index": 0,
            "success": True,
            "safety_violation": False,
            "integrity_violation": False,
            "oracle_disagreement": False,
        }
        with self.assertRaisesRegex(TypeError, "reset_seed"):
            GraspLiftNoZValidationRollout(**{**base, "reset_seed": True})
        with self.assertRaisesRegex(ValueError, "policy_noise_index"):
            GraspLiftNoZValidationRollout(**{**base, "policy_noise_index": 4})
        with self.assertRaisesRegex(TypeError, "success"):
            GraspLiftNoZValidationRollout(**{**base, "success": 1})

        with self.assertRaisesRegex(ValueError, "model_sha256"):
            GraspLiftNoZSeedCheckpointEvidence(
                checkpoint_step=500,
                training_seed=31_001,
                model_sha256="A" * 64,
                rollouts=_rollouts(),
            )
        for diagnostic in (
            {"normalized": math.nan},
            {"normalized": math.inf},
            {"normalized": -0.1},
            {"normalized": True},
            {},
        ):
            with self.subTest(diagnostic=diagnostic):
                with self.assertRaises((TypeError, ValueError)):
                    _evidence(500, 31_001, diagnostic=diagnostic)

    def test_self_hash_detects_nested_tampering_and_extra_fields(self) -> None:
        record = select_grasp_lift_no_z_checkpoint(_inventory()).to_record()
        verify_grasp_lift_no_z_selection_record(record)

        tampered = copy.deepcopy(record)
        tampered["evidence"][0]["rollouts"][0]["success"] = False
        with self.assertRaisesRegex(ValueError, "self_sha256 mismatch"):
            verify_grasp_lift_no_z_selection_record(tampered)

        extended = dict(record)
        extended["uncommitted_note"] = "not allowed"
        with self.assertRaisesRegex(ValueError, "fields changed"):
            verify_grasp_lift_no_z_selection_record(extended)

    def test_resealed_nested_tampering_is_rejected_by_reconstruction(self) -> None:
        record = select_grasp_lift_no_z_checkpoint(_inventory()).to_record()

        changed_evidence = copy.deepcopy(record)
        changed_evidence["evidence"][0]["rollouts"][0]["success"] = False
        _reseal(changed_evidence)
        with self.assertRaisesRegex(ValueError, "evidence-derived selection"):
            verify_grasp_lift_no_z_selection_record(changed_evidence)

        changed_result = copy.deepcopy(record)
        changed_result["checkpoint_results"][0]["success_count"] = 71
        _reseal(changed_result)
        with self.assertRaisesRegex(ValueError, "evidence-derived selection"):
            verify_grasp_lift_no_z_selection_record(changed_result)

        changed_ranking = copy.deepcopy(record)
        changed_ranking["selected_checkpoint_step"] = 5_500
        _reseal(changed_ranking)
        with self.assertRaisesRegex(ValueError, "evidence-derived selection"):
            verify_grasp_lift_no_z_selection_record(changed_ranking)

        changed_diagnostic = copy.deepcopy(record)
        changed_diagnostic["evidence"][0]["offline_mse_diagnostic"][
            "normalized"
        ] = 99.0
        _reseal(changed_diagnostic)
        with self.assertRaisesRegex(ValueError, "evidence-derived selection"):
            verify_grasp_lift_no_z_selection_record(changed_diagnostic)

    def test_resealed_nested_shape_and_schema_tampering_is_rejected(self) -> None:
        record = select_grasp_lift_no_z_checkpoint(_inventory()).to_record()

        changed_schema = copy.deepcopy(record)
        changed_schema["evidence"][0]["rollouts"][0]["schema_id"] = "wrong/v1"
        _reseal(changed_schema)
        with self.assertRaisesRegex(ValueError, "schema_id changed"):
            verify_grasp_lift_no_z_selection_record(changed_schema)

        extended_rollout = copy.deepcopy(record)
        extended_rollout["evidence"][0]["rollouts"][0]["note"] = "injected"
        _reseal(extended_rollout)
        with self.assertRaisesRegex(ValueError, "fields changed"):
            verify_grasp_lift_no_z_selection_record(extended_rollout)

    def test_module_has_no_torch_newton_or_data_import(self) -> None:
        source_path = (
            Path(__file__).resolve().parents[1]
            / "rexpolicy"
            / "stage0"
            / "grasp_lift_no_z_selection.py"
        )
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                imported.add(node.module)
        self.assertFalse(
            any(
                name == "torch"
                or "newton" in name.lower()
                or name.startswith("rexpolicy.stage0.data")
                for name in imported
            ),
            imported,
        )


if __name__ == "__main__":
    unittest.main()
