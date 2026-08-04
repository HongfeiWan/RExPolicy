"""Tests for independent reward-free dynamics branch digests."""

from __future__ import annotations

import unittest

import numpy as np

from rexpolicy.flywheel.replay import (
    fingerprint_each_world,
    validate_replay_fingerprint,
)


def _tree() -> dict:
    return {
        "body": {
            "q": np.asarray(
                [
                    [[0.1, 0.2], [0.3, 0.4]],
                    [[0.1, 0.2], [0.3, 0.4]],
                ],
                dtype=np.float32,
            )
        },
        "task": {
            "step": np.asarray([[1], [1]], dtype=np.int32),
        },
    }


class TestDynamicsFingerprint(unittest.TestCase):
    def test_identical_world_digest_matches_same_state_digest(self) -> None:
        tree = _tree()
        branch_digests = fingerprint_each_world(
            tree,
            world_count=2,
            float_tolerance=1.0e-6,
        )
        same_state = validate_replay_fingerprint(
            tree,
            world_count=2,
            float_tolerance=1.0e-6,
        )
        self.assertEqual(branch_digests, (same_state.digest, same_state.digest))

    def test_diverged_branches_have_independent_digests(self) -> None:
        tree = _tree()
        tree["body"]["q"][1, 0, 0] += 0.01
        digests = fingerprint_each_world(
            tree,
            world_count=2,
            float_tolerance=1.0e-6,
        )
        self.assertNotEqual(digests[0], digests[1])
        with self.assertRaisesRegex(RuntimeError, "diverged"):
            validate_replay_fingerprint(
                tree,
                world_count=2,
                float_tolerance=1.0e-6,
            )

    def test_non_finite_and_wrong_world_shapes_fail_closed(self) -> None:
        non_finite = _tree()
        non_finite["body"]["q"][0, 0, 0] = np.nan
        with self.assertRaises(FloatingPointError):
            fingerprint_each_world(
                non_finite,
                world_count=2,
                float_tolerance=1.0e-6,
            )
        with self.assertRaisesRegex(ValueError, "world dimension"):
            fingerprint_each_world(
                {"x": np.zeros((1, 2), dtype=np.float32)},
                world_count=2,
                float_tolerance=1.0e-6,
            )


if __name__ == "__main__":
    unittest.main()
