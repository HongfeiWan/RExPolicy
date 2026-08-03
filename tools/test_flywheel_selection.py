"""Pure selection-contract tests that do not require a simulator runtime."""

from __future__ import annotations

import unittest

from rexpolicy.flywheel.experience import (
    BranchOutcomeAccumulator,
    ChunkCandidate,
    TrainingSample,
    select_advantage_chunks,
)


def _candidate(
    world: int,
    score: float,
    *,
    success: bool = False,
    failure: bool = False,
    safety_violation: bool = False,
) -> ChunkCandidate:
    sample = TrainingSample(
        backbone_features=None,
        backbone_attention_mask=None,
        image_mask=None,
        state=None,
        embodiment_id=0,
        action=None,
        action_mask=None,
        valid_steps=1,
    )
    return ChunkCandidate(
        world=world,
        score=score,
        success=success,
        terminated=success or failure,
        truncated=False,
        valid_steps=1,
        rewards=[score],
        action={},
        sample=sample,
        failure=failure,
        safety_violation=safety_violation,
        failure_reasons=("test_failure",) if failure else (),
    )


class TestFailureAwareChunkSelection(unittest.TestCase):
    def test_transient_contact_is_sticky_but_inactive_frames_are_ignored(self) -> None:
        outcome = BranchOutcomeAccumulator()
        outcome.observe(
            executed=True,
            failure=True,
            contact_violation=True,
            displacement_violation=False,
        )
        outcome.observe(
            executed=True,
            failure=False,
            contact_violation=False,
            displacement_violation=False,
        )
        outcome.observe(
            executed=False,
            failure=True,
            contact_violation=False,
            displacement_violation=True,
        )

        self.assertTrue(outcome.failure)
        self.assertTrue(outcome.contact_violation)
        self.assertFalse(outcome.displacement_violation)
        self.assertTrue(outcome.safety_violation)

    def test_failure_cannot_beat_lower_scoring_safe_candidate(self) -> None:
        failed_progress = _candidate(
            0,
            0.0,
            failure=True,
            safety_violation=True,
        )
        safe_retreat = _candidate(1, -0.1)

        selection = select_advantage_chunks(
            [failed_progress, safe_retreat],
            fraction=1.0,
            temperature=1.0,
        )

        self.assertEqual(selection.mode, "positive_advantage")
        self.assertIs(selection.continuation, safe_retreat)
        self.assertEqual(selection.selected, [safe_retreat])
        self.assertFalse(failed_progress.selected_for_training)
        self.assertFalse(failed_progress.chosen_for_continuation)

    def test_all_unsafe_candidates_abstain(self) -> None:
        candidates = [
            _candidate(0, 1.0, failure=True, safety_violation=True),
            _candidate(1, -1.0, failure=True),
        ]

        selection = select_advantage_chunks(
            candidates,
            fraction=0.5,
            temperature=1.0,
        )

        self.assertEqual(selection.mode, "no_safe_candidate")
        self.assertEqual(selection.selected, [])
        self.assertIsNone(selection.continuation)
        self.assertFalse(any(item.selected_for_training for item in candidates))
        self.assertFalse(any(item.chosen_for_continuation for item in candidates))

    def test_success_cannot_also_be_failure_or_unsafe(self) -> None:
        contradictory = _candidate(0, 1.0, success=True, failure=True)
        with self.assertRaisesRegex(ValueError, "both success and failure"):
            select_advantage_chunks(
                [contradictory],
                fraction=1.0,
                temperature=1.0,
            )

        unsafe_success = _candidate(1, 1.0, success=True, safety_violation=True)
        with self.assertRaisesRegex(ValueError, "both success and unsafe"):
            select_advantage_chunks(
                [unsafe_success],
                fraction=1.0,
                temperature=1.0,
            )

    def test_archive_record_preserves_failure_taxonomy(self) -> None:
        candidate = _candidate(0, 0.0, failure=True, safety_violation=True)
        record = candidate.archive_record()

        self.assertTrue(record["failure"])
        self.assertTrue(record["safety_violation"])
        self.assertEqual(record["failure_reasons"], ["test_failure"])


if __name__ == "__main__":
    unittest.main()
