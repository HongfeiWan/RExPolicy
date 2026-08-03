"""Tests for exact archived candidate re-execution validation."""

from __future__ import annotations

import unittest

import numpy as np

from rexpolicy.flywheel.replay import (
    CandidateReplayMismatch,
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


def _validate(
    archived: dict[str, object],
    replayed: CandidateReplayResult,
    *,
    action_tolerance: float = 1.0e-6,
    reward_tolerance: float = 1.0e-3,
) -> None:
    validate_candidate_replay(
        archived,
        replayed,
        action_tolerance=action_tolerance,
        reward_tolerance=reward_tolerance,
    )


class TestCandidateReplayValidation(unittest.TestCase):
    def test_exact_replay_passes(self) -> None:
        _validate(_archived_candidate(), _replayed_result())

    def test_effective_action_drift_is_rejected(self) -> None:
        changed = np.asarray([[0.0] * 19, [0.1] * 19], dtype=np.float32)
        changed[1, 0] += 0.01
        with self.assertRaisesRegex(
            CandidateReplayMismatch,
            "effective action changed",
        ):
            _validate(
                _archived_candidate(),
                _replayed_result(executed_action=changed),
            )

    def test_reward_drift_is_rejected(self) -> None:
        with self.assertRaisesRegex(CandidateReplayMismatch, "rewards changed"):
            _validate(
                _archived_candidate(),
                _replayed_result(rewards=(0.25, 0.75)),
            )

    def test_reward_and_action_tolerances_are_independent(self) -> None:
        changed = np.asarray([[0.0] * 19, [0.1] * 19], dtype=np.float32)
        changed[1, 0] += 5.0e-5
        _validate(
            _archived_candidate(),
            _replayed_result(
                rewards=(0.2505, 0.5005),
                executed_action=changed,
            ),
            action_tolerance=1.0e-4,
            reward_tolerance=7.0e-4,
        )

    def test_tolerances_must_be_positive_and_finite(self) -> None:
        for action_tolerance, reward_tolerance in (
            (0.0, 1.0e-3),
            (1.0e-6, float("inf")),
        ):
            with self.assertRaisesRegex(ValueError, "positive and finite"):
                _validate(
                    _archived_candidate(),
                    _replayed_result(),
                    action_tolerance=action_tolerance,
                    reward_tolerance=reward_tolerance,
                )

    def test_non_finite_archive_is_rejected(self) -> None:
        archived = _archived_candidate()
        archived["rewards"] = [0.25, float("nan")]
        with self.assertRaisesRegex(CandidateReplayMismatch, "archive contains"):
            _validate(
                archived,
                _replayed_result(),
            )

    def test_outcome_drift_is_rejected(self) -> None:
        with self.assertRaisesRegex(CandidateReplayMismatch, "outcome changed"):
            _validate(
                _archived_candidate(),
                _replayed_result(
                    failure=True,
                    safety_violation=True,
                    terminated=True,
                ),
            )

    def test_legacy_terminated_failure_is_inferred(self) -> None:
        archived = _archived_candidate()
        archived.pop("failure")
        archived.pop("safety_violation")
        archived["terminated"] = True
        _validate(
            archived,
            _replayed_result(
                failure=True,
                safety_violation=True,
                terminated=True,
            ),
        )


if __name__ == "__main__":
    unittest.main()
