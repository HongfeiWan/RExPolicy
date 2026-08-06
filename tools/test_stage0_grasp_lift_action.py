"""Tests for the hash-bound Grasp-Lift physical/model action contract."""

from __future__ import annotations

import unittest

import torch

from rexpolicy.stage0.envs.grasp_lift_action import (
    DEFAULT_GRASP_LIFT_ACTION_SCHEMA,
    GRASP_LIFT_DEFAULT_CLOSE_HAND_TARGET,
    GRASP_LIFT_EFFECTIVE_ACTION_MASK,
    GRASP_LIFT_HAND_JOINT_NAMES,
    GraspLiftActionSchema,
    grasp_lift_default_close_hand_target_like,
    grasp_lift_model_to_physical_action,
    grasp_lift_physical_to_model_action,
)
from rexpolicy.stage0.envs.state_schema import DEFAULT_STAGE0_STATE_SCHEMA


SCHEMA = DEFAULT_STAGE0_STATE_SCHEMA


def _state(batch_size: int = 3) -> torch.Tensor:
    state = torch.zeros(batch_size, SCHEMA.dimension, dtype=torch.float32)
    state[:, SCHEMA.slice("eef_position")] = torch.tensor((0.2, -0.1, 0.4))
    return state


def _reference(state: torch.Tensor) -> torch.Tensor:
    action = torch.zeros(state.shape[0], 19, dtype=state.dtype)
    action[:, :3] = state[:, SCHEMA.slice("eef_position")]
    action[:, 3:9] = torch.tensor((1.0, 0.0, 0.0, 0.0, 1.0, 0.0))
    action[:, 9:19] = torch.linspace(0.01, 0.10, 10)
    return action


class GraspLiftActionContractTest(unittest.TestCase):
    def test_schema_binds_mask_joint_order_and_close_target(self) -> None:
        schema = DEFAULT_GRASP_LIFT_ACTION_SCHEMA
        self.assertEqual(schema.sha256, GraspLiftActionSchema().sha256)
        self.assertEqual(len(schema.sha256), 64)
        self.assertEqual(len(GRASP_LIFT_EFFECTIVE_ACTION_MASK), 19)
        self.assertEqual(sum(GRASP_LIFT_EFFECTIVE_ACTION_MASK), 13)
        self.assertEqual(len(GRASP_LIFT_HAND_JOINT_NAMES), 10)
        self.assertEqual(len(GRASP_LIFT_DEFAULT_CLOSE_HAND_TARGET), 10)
        self.assertEqual(GRASP_LIFT_HAND_JOINT_NAMES[0], "thumb_cmc_pitch")
        self.assertEqual(GRASP_LIFT_HAND_JOINT_NAMES[-1], "thumb_cmc_roll")

    def test_default_close_target_is_materialized_per_world(self) -> None:
        state = _state(2)
        target = grasp_lift_default_close_hand_target_like(state)
        self.assertEqual(tuple(target.shape), (2, 10))
        self.assertTrue(
            torch.equal(target[0], torch.tensor(GRASP_LIFT_DEFAULT_CLOSE_HAND_TARGET))
        )
        target[0, 0] = -1.0
        self.assertNotEqual(target[0, 0].item(), target[1, 0].item())

    def test_hybrid_representation_round_trips_executed_action(self) -> None:
        state = _state()
        reference = _reference(state)
        executed = reference.clone()
        executed[:, :3] += torch.tensor((0.01, -0.02, 0.03))
        executed[:, 9:19] = torch.linspace(0.2, 1.1, 10)

        model = grasp_lift_physical_to_model_action(state, executed, reference)
        self.assertTrue(
            torch.allclose(model[:, :3], torch.tensor(((0.01, -0.02, 0.03),) * 3))
        )
        self.assertTrue(torch.count_nonzero(model[:, 3:9]).item() == 0)
        self.assertTrue(torch.equal(model[:, 9:19], executed[:, 9:19]))
        decoded = grasp_lift_model_to_physical_action(state, model, reference)
        self.assertTrue(torch.equal(decoded, executed))

    def test_rotation_contract_fails_closed(self) -> None:
        state = _state(1)
        reference = _reference(state)
        executed = reference.clone()
        executed[:, 3] = 0.5
        with self.assertRaisesRegex(ValueError, "rotation"):
            grasp_lift_physical_to_model_action(state, executed, reference)

        model = reference.clone()
        model[:, 3:9] = 1.0
        with self.assertRaisesRegex(ValueError, "zero"):
            grasp_lift_model_to_physical_action(state, model, reference)


if __name__ == "__main__":
    unittest.main()
