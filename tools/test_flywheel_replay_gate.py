"""Contract tests for the versioned physical historical replay gate."""

from __future__ import annotations

import copy
import json
import unittest
from typing import Any

import numpy as np

from rexpolicy.flywheel.replay import (
    HistoricalReplayGateMismatch,
    PHYSICAL_REPLAY_GATE_TASK_FIELDS,
    build_physical_replay_gate_record,
    physical_replay_gate_tree,
    validate_historical_physical_replay_gate,
    validate_physical_replay_gate,
)


SIMULATOR_SHA256 = "b" * 64
OBSERVATION_SHA256 = "a" * 64
SAME_STATE_TOLERANCE = 1.0e-5
HISTORICAL_REPLAY_STATE_TOLERANCE = 1.0e-4
RENDER_CONTRACT = {
    "render_contract_id": "test/v1",
    "outputs": {"ego_view": {"shape_per_world": [4, 4, 3]}},
}


def _worlds(value: object, *, dtype: object) -> np.ndarray:
    return np.asarray([[value], [value]], dtype=dtype)


def _replay_tree() -> dict[str, Any]:
    task = {
        name: _worlds(False, dtype=np.bool_)
        for name in PHYSICAL_REPLAY_GATE_TASK_FIELDS
    }
    for name in (
        "task_phase",
        "reach_success_hold_steps",
        "grasp_contact_frames",
        "grasp_support_gap_frames",
        "contact_gap_frames",
        "settle_frames",
        "episode_step",
    ):
        task[name] = _worlds(0, dtype=np.int32)
    return {
        "task": task,
        "contacts": {
            "reach_contact_violation": _worlds(False, dtype=np.bool_),
            "reach_displacement_violation": _worlds(False, dtype=np.bool_),
            "finger_contact_counts": np.zeros((2, 5), dtype=np.int32),
        },
        "images": {
            "ego_view": np.zeros((2, 4, 4, 3), dtype=np.uint8),
            "wrist_view": np.zeros((2, 4, 4, 3), dtype=np.uint8),
        },
    }


def _dynamics_tree() -> dict[str, Any]:
    return {
        "body": {"q": np.zeros((2, 2, 7), dtype=np.float32)},
        "control": {
            "effective_action": np.zeros((2, 19), dtype=np.float32),
            "effective_action_mask": np.ones((2, 19), dtype=np.bool_),
        },
        "task": {"obj_pose": np.zeros((2, 7), dtype=np.float32)},
    }


def _fingerprint(
    replay: dict[str, Any] | None = None,
    dynamics: dict[str, Any] | None = None,
):
    tree = physical_replay_gate_tree(
        replay=_replay_tree() if replay is None else replay,
        dynamics=_dynamics_tree() if dynamics is None else dynamics,
    )
    return validate_physical_replay_gate(
        tree,
        world_count=2,
        float_tolerance=1.0e-5,
    )


def _gate_record(fingerprint):
    return json.loads(
        json.dumps(
            build_physical_replay_gate_record(
                fingerprint,
                same_state_tolerance=SAME_STATE_TOLERANCE,
                historical_replay_state_tolerance=(
                    HISTORICAL_REPLAY_STATE_TOLERANCE
                ),
                simulator_fingerprint_sha256=SIMULATOR_SHA256,
                observation_contract_sha256=OBSERVATION_SHA256,
                render_contract=RENDER_CONTRACT,
            ),
            allow_nan=False,
        )
    )


def _validate_gate(record, fingerprint) -> None:
    validate_historical_physical_replay_gate(
        record,
        current_fingerprint=fingerprint,
        same_state_tolerance=SAME_STATE_TOLERANCE,
        historical_replay_state_tolerance=(
            HISTORICAL_REPLAY_STATE_TOLERANCE
        ),
        simulator_fingerprint_sha256=SIMULATOR_SHA256,
        observation_contract_sha256=OBSERVATION_SHA256,
        render_contract=RENDER_CONTRACT,
    )


class TestPhysicalReplayGate(unittest.TestCase):
    def test_historical_tolerance_cannot_be_stricter_than_same_state(self) -> None:
        with self.assertRaisesRegex(ValueError, "greater than or equal"):
            build_physical_replay_gate_record(
                _fingerprint(),
                same_state_tolerance=SAME_STATE_TOLERANCE,
                historical_replay_state_tolerance=1.0e-6,
                simulator_fingerprint_sha256=SIMULATOR_SHA256,
                observation_contract_sha256=OBSERVATION_SHA256,
                render_contract=RENDER_CONTRACT,
            )

    def test_rgb_changes_do_not_change_physical_gate(self) -> None:
        replay = _replay_tree()
        first = _fingerprint(replay=replay)
        replay["images"]["ego_view"].fill(255)
        replay["images"]["wrist_view"][0, 0, 0, 0] = 17
        second = _fingerprint(replay=replay)
        self.assertEqual(first.digest, second.digest)

    def test_physical_and_discrete_safety_task_changes_change_gate(self) -> None:
        baseline = _fingerprint()

        dynamics = _dynamics_tree()
        dynamics["body"]["q"][:, 0, 0] = 2.0e-5
        self.assertNotEqual(baseline.digest, _fingerprint(dynamics=dynamics).digest)

        control = _dynamics_tree()
        control["control"]["effective_action_mask"][:, 0] = False
        self.assertNotEqual(baseline.digest, _fingerprint(dynamics=control).digest)

        safety = _replay_tree()
        safety["contacts"]["reach_contact_violation"][:] = True
        self.assertNotEqual(baseline.digest, _fingerprint(replay=safety).digest)

        task = _replay_tree()
        task["task"]["task_phase"][:] = 1
        self.assertNotEqual(baseline.digest, _fingerprint(replay=task).digest)

    def test_quantization_boundary_is_commitment_not_tolerance(self) -> None:
        boundary = np.float64(0.5e-5)
        below = np.nextafter(boundary, np.float64(0.0))
        above = np.nextafter(boundary, np.float64(1.0))
        self.assertLess(float(above - below), 1.0e-5)
        archived_dynamics = _dynamics_tree()
        archived_dynamics["body"]["q"] = archived_dynamics["body"]["q"].astype(
            np.float64
        )
        archived_dynamics["body"]["q"][:, 0, 0] = below
        current_dynamics = _dynamics_tree()
        current_dynamics["body"]["q"] = current_dynamics["body"]["q"].astype(
            np.float64
        )
        current_dynamics["body"]["q"][:, 0, 0] = above
        archived = _fingerprint(dynamics=archived_dynamics)
        current = _fingerprint(dynamics=current_dynamics)
        self.assertNotEqual(archived.digest, current.digest)
        _validate_gate(_gate_record(archived), current)

    def test_historical_tolerance_is_independent_from_same_state(self) -> None:
        archived = _fingerprint()
        within_historical = _dynamics_tree()
        within_historical["body"]["q"][:, 0, 0] = 5.0e-5
        _validate_gate(
            _gate_record(archived),
            _fingerprint(dynamics=within_historical),
        )

        changed = _dynamics_tree()
        changed["body"]["q"][:, 0, 0] = 2.0e-4
        with self.assertRaisesRegex(
            HistoricalReplayGateMismatch,
            "continuous witness diverged",
        ):
            _validate_gate(_gate_record(archived), _fingerprint(dynamics=changed))

    def test_discrete_and_shape_changes_are_rejected_exactly(self) -> None:
        archived = _fingerprint()
        record = _gate_record(archived)

        safety = _replay_tree()
        safety["contacts"]["reach_contact_violation"][:] = True
        with self.assertRaisesRegex(
            HistoricalReplayGateMismatch,
            "discrete state changed",
        ):
            _validate_gate(record, _fingerprint(replay=safety))

        shape = _dynamics_tree()
        shape["body"]["q"] = np.zeros((2, 3, 7), dtype=np.float32)
        with self.assertRaisesRegex(
            HistoricalReplayGateMismatch,
            "field schema changed",
        ):
            _validate_gate(record, _fingerprint(dynamics=shape))

    def test_old_or_missing_gate_schema_is_rejected(self) -> None:
        fingerprint = _fingerprint()
        render_contract = copy.deepcopy(RENDER_CONTRACT)
        kwargs = {
            "current_fingerprint": fingerprint,
            "same_state_tolerance": SAME_STATE_TOLERANCE,
            "historical_replay_state_tolerance": (
                HISTORICAL_REPLAY_STATE_TOLERANCE
            ),
            "simulator_fingerprint_sha256": SIMULATOR_SHA256,
            "observation_contract_sha256": OBSERVATION_SHA256,
            "render_contract": render_contract,
        }
        record = build_physical_replay_gate_record(
            fingerprint,
            same_state_tolerance=SAME_STATE_TOLERANCE,
            historical_replay_state_tolerance=(
                HISTORICAL_REPLAY_STATE_TOLERANCE
            ),
            simulator_fingerprint_sha256=SIMULATOR_SHA256,
            observation_contract_sha256=OBSERVATION_SHA256,
            render_contract=render_contract,
        )
        validate_historical_physical_replay_gate(record, **kwargs)
        self.assertFalse(
            record["observation_render_provenance"]["pixel_equality_asserted"]
        )
        changed_render = copy.deepcopy(render_contract)
        changed_render["outputs"]["ego_view"]["shape_per_world"] = [8, 8, 3]
        with self.assertRaisesRegex(
            HistoricalReplayGateMismatch,
            "observation_render_provenance",
        ):
            validate_historical_physical_replay_gate(
                record,
                **{**kwargs, "render_contract": changed_render},
            )
        with self.assertRaisesRegex(
            HistoricalReplayGateMismatch,
            "physical_provenance",
        ):
            validate_historical_physical_replay_gate(
                record,
                **{**kwargs, "simulator_fingerprint_sha256": "c" * 64},
            )

        with self.assertRaisesRegex(
            HistoricalReplayGateMismatch,
            "explicit migration",
        ):
            validate_historical_physical_replay_gate(
                {"replay_fingerprint_sha256": fingerprint.digest},
                **kwargs,
            )
        legacy = copy.deepcopy(record)
        legacy["schema_version"] = 1
        with self.assertRaisesRegex(
            HistoricalReplayGateMismatch,
            "explicit migration",
        ):
            validate_historical_physical_replay_gate(legacy, **kwargs)


if __name__ == "__main__":
    unittest.main()
