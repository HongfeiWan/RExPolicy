from __future__ import annotations

import unittest

import torch

from rexpolicy.stage0.envs.state_only import (
    canonicalize_stage0_components,
    pack_stage0_state,
)
from rexpolicy.stage0.envs.state_schema import DEFAULT_STAGE0_STATE_SCHEMA


class Stage0StateAdapterTest(unittest.TestCase):
    def test_world_translation_does_not_change_canonical_state(self) -> None:
        batch_size = 2
        qpos = (
            torch.arange(batch_size * 17, dtype=torch.float32).reshape(batch_size, 17)
            / 10.0
        )
        qvel = -qpos
        eef = torch.tensor(
            [[0.50, 0.10, 0.30, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0]] * batch_size,
            dtype=torch.float32,
        )
        base_position = torch.tensor(
            [[0.0, 0.0, 0.0], [5.0, -3.0, 2.0]], dtype=torch.float32
        )
        object_relative = torch.tensor([[0.60, 0.20, 0.25]] * batch_size)
        goal_relative = torch.tensor([[0.70, 0.25, 0.30]] * batch_size)
        identity = torch.tensor([[0.0, 0.0, 0.0, 1.0]] * batch_size)
        zeros6 = torch.zeros(batch_size, 6)
        fields = canonicalize_stage0_components(
            {
                "agent_qpos": qpos,
                "agent_qvel": qvel,
                "eef_9d_base": eef,
                "object_pose_world": torch.cat(
                    (base_position + object_relative, identity), dim=-1
                ),
                "object_twist_world": zeros6,
                "base_pose_world": torch.cat((base_position, identity), dim=-1),
                "base_twist_world": zeros6,
                "goal_position_world": base_position + goal_relative,
                "finger_contact_counts": torch.zeros(batch_size, 5),
                "has_hand_contact": torch.zeros(batch_size, dtype=torch.bool),
                "is_grasped": torch.zeros(batch_size, dtype=torch.bool),
                "task_phase": torch.zeros(batch_size, dtype=torch.int32),
            }
        )
        packed = pack_stage0_state(fields)
        self.assertEqual(packed.shape, (2, 79))
        # qpos intentionally differs; every geometry slice must remain equal.
        geometry_start = DEFAULT_STAGE0_STATE_SCHEMA.slice("eef_position").start
        torch.testing.assert_close(
            packed[0, geometry_start:], packed[1, geometry_start:]
        )

    def test_quaternion_sign_is_rotation_6d_invariant(self) -> None:
        def components(sign: float) -> dict[str, torch.Tensor]:
            return {
                "agent_qpos": torch.zeros(1, 17),
                "agent_qvel": torch.zeros(1, 17),
                "eef_9d_base": torch.tensor(
                    [[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0]]
                ),
                "object_pose_world": torch.tensor(
                    [[0.1, 0.2, 0.3, 0.0, 0.0, 0.0, sign]]
                ),
                "object_twist_world": torch.zeros(1, 6),
                "base_pose_world": torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]),
                "base_twist_world": torch.zeros(1, 6),
                "goal_position_world": torch.tensor([[0.2, 0.3, 0.4]]),
                "finger_contact_counts": torch.zeros(1, 5),
                "has_hand_contact": torch.zeros(1, dtype=torch.bool),
                "is_grasped": torch.zeros(1, dtype=torch.bool),
                "task_phase": torch.zeros(1, dtype=torch.int32),
            }

        positive = pack_stage0_state(canonicalize_stage0_components(components(1.0)))
        negative = pack_stage0_state(canonicalize_stage0_components(components(-1.0)))
        torch.testing.assert_close(positive, negative)

    def test_rejects_non_finite_components(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-finite"):
            canonicalize_stage0_components(
                {
                    "agent_qpos": torch.full((1, 17), torch.nan),
                    "agent_qvel": torch.zeros(1, 17),
                    "eef_9d_base": torch.zeros(1, 9),
                    "object_pose_world": torch.zeros(1, 7),
                    "object_twist_world": torch.zeros(1, 6),
                    "base_pose_world": torch.zeros(1, 7),
                    "base_twist_world": torch.zeros(1, 6),
                    "goal_position_world": torch.zeros(1, 3),
                    "finger_contact_counts": torch.zeros(1, 5),
                }
            )


if __name__ == "__main__":
    unittest.main()
