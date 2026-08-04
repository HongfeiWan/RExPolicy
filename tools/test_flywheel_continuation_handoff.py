"""Tests for one-shot canonical continuation replay and root caching."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from tools.run_flywheel_ddp import _replay_canonical_continuation


class TestCanonicalContinuationHandoff(unittest.TestCase):
    def test_two_decision_handoff_replays_once_and_returns_next_root(self) -> None:
        observation = {"state": object()}
        cached_root = object()
        actions = [np.zeros(19, dtype=np.float32)]
        with (
            patch(
                "tools.run_flywheel_ddp._reset_and_replay",
                return_value=(observation, False),
            ) as replay,
            patch(
                "tools.run_flywheel_ddp._capture_canonical_decision_root",
                return_value=cached_root,
            ) as capture,
        ):
            result = _replay_canonical_continuation(
                args=SimpleNamespace(),
                env=object(),
                context=object(),
                reset_seed=17,
                root_actions=actions,
                simulator_fingerprint="a" * 64,
                observation_contract_sha256="b" * 64,
                event_schema=object(),
            )

        self.assertIs(result, cached_root)
        replay.assert_called_once()
        capture.assert_called_once()
        self.assertIs(capture.call_args.kwargs["observation"], observation)

    def test_terminated_replay_never_commits_a_root(self) -> None:
        with (
            patch(
                "tools.run_flywheel_ddp._reset_and_replay",
                return_value=({}, True),
            ) as replay,
            patch(
                "tools.run_flywheel_ddp._capture_canonical_decision_root"
            ) as capture,
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                "terminated despite a non-terminal selected branch",
            ):
                _replay_canonical_continuation(
                    args=SimpleNamespace(),
                    env=object(),
                    context=object(),
                    reset_seed=17,
                    root_actions=[np.zeros(19, dtype=np.float32)],
                    simulator_fingerprint="a" * 64,
                    observation_contract_sha256="b" * 64,
                    event_schema=object(),
                )

        replay.assert_called_once()
        capture.assert_not_called()


if __name__ == "__main__":
    unittest.main()
