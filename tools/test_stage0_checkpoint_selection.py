"""CPU-only tests for hard-safe Stage 0 checkpoint selection."""

from __future__ import annotations

import json
import unittest

from rexpolicy.stage0.evaluation.checkpoint_selection import (
    STAGE0_VALIDATION_CHECKPOINT_SELECTION_SCHEMA_ID,
    Stage0ValidationCheckpointCandidate,
    select_stage0_validation_checkpoint,
)


def _candidate(
    step: int,
    *,
    success: float,
    path_f1: float,
    contact: bool = False,
    displacement: float = 0.0,
) -> Stage0ValidationCheckpointCandidate:
    return Stage0ValidationCheckpointCandidate(
        conditional_step=step,
        success_rate=success,
        contact_violation=contact,
        maximum_object_displacement_m=displacement,
        path_macro_f1=path_f1,
    )


class Stage0CheckpointSelectionTest(unittest.TestCase):
    def test_hard_safety_precedes_success_and_path_quality(self) -> None:
        result = select_stage0_validation_checkpoint(
            (
                _candidate(
                    3_000,
                    success=1.0,
                    path_f1=1.0,
                    contact=True,
                ),
                _candidate(2_500, success=0.95, path_f1=0.99),
                _candidate(2_000, success=0.96, path_f1=0.80),
            ),
            maximum_object_displacement_m=0.005,
        )

        self.assertTrue(result.gate_passed)
        self.assertEqual(result.selected_conditional_step, 2_000)
        self.assertEqual(
            tuple(item.candidate.conditional_step for item in result.ranking),
            (2_000, 2_500, 3_000),
        )
        self.assertEqual(
            tuple(item.hard_safety_passed for item in result.ranking),
            (True, True, False),
        )
        self.assertEqual(result.ranking[-1].safety_failures, ("contact_violation",))
        self.assertTrue(result.ranking[0].selected)
        self.assertFalse(result.ranking[1].selected)

    def test_path_f1_and_earliest_step_break_success_ties_deterministically(
        self,
    ) -> None:
        candidates = (
            _candidate(2_500, success=0.95, path_f1=0.90),
            _candidate(1_500, success=0.95, path_f1=0.95),
            _candidate(1_000, success=0.95, path_f1=0.95),
        )
        forward = select_stage0_validation_checkpoint(
            candidates,
            maximum_object_displacement_m=0.005,
        )
        reverse = select_stage0_validation_checkpoint(
            tuple(reversed(candidates)),
            maximum_object_displacement_m=0.005,
        )

        self.assertEqual(forward, reverse)
        self.assertEqual(forward.selected_conditional_step, 1_000)
        self.assertEqual(
            tuple(item.candidate.conditional_step for item in forward.ranking),
            (1_000, 1_500, 2_500),
        )

    def test_no_safe_candidate_returns_explicit_gate_failure(self) -> None:
        result = select_stage0_validation_checkpoint(
            (
                _candidate(
                    3_000,
                    success=1.0,
                    path_f1=1.0,
                    contact=True,
                ),
                _candidate(
                    2_500,
                    success=0.99,
                    path_f1=0.99,
                    displacement=0.005_001,
                ),
            ),
            maximum_object_displacement_m=0.005,
        )

        self.assertFalse(result.gate_passed)
        self.assertIsNone(result.selected_conditional_step)
        self.assertIn("no validation checkpoint", result.selection_reason)
        self.assertEqual(len(result.ranking), 2)
        self.assertFalse(any(item.selected for item in result.ranking))
        self.assertFalse(any(item.hard_safety_passed for item in result.ranking))
        self.assertEqual(
            result.ranking[1].safety_failures,
            ("object_displacement_limit_exceeded",),
        )

        record = result.to_record()
        self.assertEqual(
            record["schema_id"],
            STAGE0_VALIDATION_CHECKPOINT_SELECTION_SCHEMA_ID,
        )
        self.assertFalse(record["gate_passed"])
        self.assertIsNone(record["selected_conditional_step"])
        json.dumps(record, allow_nan=False, sort_keys=True)

    def test_empty_input_is_a_serializable_gate_failure(self) -> None:
        result = select_stage0_validation_checkpoint(
            (),
            maximum_object_displacement_m=0.005,
        )
        self.assertFalse(result.gate_passed)
        self.assertEqual(result.ranking, ())
        json.dumps(result.to_record(), allow_nan=False, sort_keys=True)

    def test_candidate_and_protocol_inputs_are_strict(self) -> None:
        with self.assertRaisesRegex(ValueError, "success_rate"):
            _candidate(1, success=float("nan"), path_f1=0.5)
        with self.assertRaisesRegex(ValueError, "path_macro_f1"):
            _candidate(1, success=0.5, path_f1=1.1)
        with self.assertRaisesRegex(TypeError, "contact_violation"):
            Stage0ValidationCheckpointCandidate(
                conditional_step=1,
                success_rate=0.5,
                contact_violation=1,  # type: ignore[arg-type]
                maximum_object_displacement_m=0.0,
                path_macro_f1=0.5,
            )
        with self.assertRaisesRegex(ValueError, "unique"):
            select_stage0_validation_checkpoint(
                (
                    _candidate(1, success=0.5, path_f1=0.5),
                    _candidate(1, success=0.6, path_f1=0.6),
                ),
                maximum_object_displacement_m=0.005,
            )
        with self.assertRaisesRegex(ValueError, "non-negative"):
            select_stage0_validation_checkpoint(
                (_candidate(1, success=0.5, path_f1=0.5),),
                maximum_object_displacement_m=-0.001,
            )


if __name__ == "__main__":
    unittest.main()
