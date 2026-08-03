"""Tests for exact archived candidate re-execution validation."""

from __future__ import annotations

import unittest

import numpy as np

from rexpolicy.flywheel.replay import (
    CandidateReplayResult,
    validate_candidate_replay,
)


def _archived_candidate() -> dict[str, object]:
    return {
        "success": False,
        "failure": False,
        "safety_violation": False,
        "terminated": False,
        "truncated": False,
        "rewards": [0.25, 0.5],
        "action": {
            "action_19d": np.asarray(
                [[0.0] * 19, [0.1] * 19],
                dtype=np.float32,
            )
        },
    }


def _replayed_result(**overrides: object) -> CandidateReplayResult:
    values: dict[str, object] = {
        "rewards": (0.25, 0.5),
        "success": False,
        "failure": False,
        "safety_violation": False,
        "terminated": False,
        "truncated": False,
        "executed_action": np.asarray(
            [[0.0] * 19, [0.1] * 19],
            dtype=np.float32,
        ),
    }
    values.update(overrides)
    return CandidateReplayResult(**values)


class TestCandidateReplayValidation(unittest.TestCase):
    def test_exact_replay_passes(self) -> None:
        validate_candidate_replay(
            _archived_candidate(),
            _replayed_result(),
            float_tolerance=1.0e-6,
        )

    def test_effective_action_drift_is_rejected(self) -> None:
        changed = np.asarray([[0.0] * 19, [0.1] * 19], dtype=np.float32)
        changed[1, 0] += 0.01
        with self.assertRaisesRegex(RuntimeError, "effective action changed"):
            validate_candidate_replay(
                _archived_candidate(),
                _replayed_result(executed_action=changed),
                float_tolerance=1.0e-6,
            )

    def test_reward_drift_is_rejected(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "rewards changed"):
            validate_candidate_replay(
                _archived_candidate(),
                _replayed_result(rewards=(0.25, 0.75)),
                float_tolerance=1.0e-6,
            )

    def test_non_finite_archive_is_rejected(self) -> None:
        archived = _archived_candidate()
        archived["rewards"] = [0.25, float("nan")]
        with self.assertRaisesRegex(FloatingPointError, "archive contains"):
            validate_candidate_replay(
                archived,
                _replayed_result(),
                float_tolerance=1.0e-6,
            )

    def test_outcome_drift_is_rejected(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "outcome changed"):
            validate_candidate_replay(
                _archived_candidate(),
                _replayed_result(
                    failure=True,
                    safety_violation=True,
                    terminated=True,
                ),
                float_tolerance=1.0e-6,
            )

    def test_legacy_terminated_failure_is_inferred(self) -> None:
        archived = _archived_candidate()
        archived.pop("failure")
        archived.pop("safety_violation")
        archived["terminated"] = True
        validate_candidate_replay(
            archived,
            _replayed_result(
                failure=True,
                safety_violation=True,
                terminated=True,
            ),
            float_tolerance=1.0e-6,
        )


if __name__ == "__main__":
    unittest.main()
