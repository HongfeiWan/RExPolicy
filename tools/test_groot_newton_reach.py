# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Contract and optional production-GPU tests for reach_green_cap/v1."""

from __future__ import annotations

import math
import os
import unittest

import numpy as np
import warp as wp

from rexpolicy.envs import groot_newton_env as env_module
from rexpolicy.flywheel.replay import validate_replay_fingerprint


class TestGrootNewtonReach(unittest.TestCase):
    device = "cpu"

    def _array(self, values, dtype):
        return wp.array(values, dtype=dtype, device=self.device)

    def _zeros(self, count, dtype):
        return wp.zeros(count, dtype=dtype, device=self.device)

    def test_default_transfer_and_production_reach_contract(self):
        default = env_module.GrootNewtonEnvConfig()
        self.assertEqual(default.task_mode, "bottle_transfer")

        reach = env_module.GrootNewtonEnvConfig(task_mode="reach_green_cap")
        self.assertEqual((reach.control_hz, reach.simulation_hz, reach.substeps_per_frame), (10, 60, 16))
        self.assertEqual(reach.bottle_settle_frames, 60)
        self.assertEqual(reach.reach_goal_offset_world, (0.180, 0.080, 0.050))
        self.assertEqual(reach.reach_success_hold_steps, 2)
        self.assertAlmostEqual(reach.reach_success_threshold, 0.040)
        self.assertAlmostEqual(reach.reach_bottle_displacement_limit, 0.010)
        self.assertAlmostEqual(reach.reach_reset_xy_jitter_m, 0.010)
        self.assertEqual(
            env_module.REACH_GREEN_CAP_INSTRUCTION,
            "move the open right hand to the safe pre-grasp position beside the green bottle cap "
            "without moving the bottle",
        )

        with self.assertRaisesRegex(ValueError, "Unsupported task_mode"):
            env_module.GrootNewtonEnvConfig(task_mode="unknown")
        with self.assertRaisesRegex(ValueError, "pd_eef_pose_abs"):
            env_module.GrootNewtonEnvConfig(task_mode="reach_green_cap", control_mode="pd_joint_pos")
        with self.assertRaisesRegex(ValueError, "reach_success_hold_steps"):
            env_module.GrootNewtonEnvConfig(reach_success_hold_steps=0)
        with self.assertRaisesRegex(ValueError, "reach_reset_xy_jitter_m"):
            env_module.GrootNewtonEnvConfig(reach_reset_xy_jitter_m=-0.001)

    def test_reset_recipe_offset_is_stable_bounded_and_distinct(self):
        first = env_module._reach_reset_xy_offset(1234, 0.010)
        repeated = env_module._reach_reset_xy_offset(1234, 0.010)
        second = env_module._reach_reset_xy_offset(1235, 0.010)

        self.assertEqual(first, repeated)
        self.assertNotEqual(first, second)
        self.assertTrue(all(abs(value) <= 0.010 for value in first))
        self.assertEqual(env_module._reach_reset_xy_offset(1234, 0.0), (0.0, 0.0))

    def test_goal_is_world_offset_transformed_into_right_world_base(self):
        angle = math.pi / 2.0
        transforms = [
            wp.transform(wp.vec3(1.0, 2.0, 3.0), wp.quat_identity()),
            wp.transform(
                wp.vec3(0.25, -0.50, 1.0),
                wp.quat(0.0, 0.0, math.sin(angle / 2.0), math.cos(angle / 2.0)),
            ),
        ]
        body_q = wp.array(transforms, dtype=wp.transform, device=self.device)
        starts = self._array((0, 2), wp.int32)
        goal_world = self._zeros((1, 3), wp.float32)
        goal_base = self._zeros((1, 3), wp.float32)
        initial_pose = self._zeros((1, 7), wp.float32)
        max_z = self._zeros(1, wp.float32)

        wp.launch(
            env_module._initialize_reach_task_goal,
            dim=1,
            inputs=[
                body_q,
                starts,
                0,
                1,
                wp.vec3(0.18, 0.08, 0.05),
                None,
                goal_world,
                goal_base,
                initial_pose,
                max_z,
            ],
            device=self.device,
        )

        expected_world = np.asarray((1.18, 2.08, 3.05), dtype=np.float32)
        relative = expected_world - np.asarray((0.25, -0.50, 1.0), dtype=np.float32)
        expected_base = np.asarray((relative[1], -relative[0], relative[2]), dtype=np.float32)
        np.testing.assert_allclose(goal_world.numpy()[0], expected_world, atol=1.0e-6)
        np.testing.assert_allclose(goal_base.numpy()[0], expected_base, atol=1.0e-6)
        np.testing.assert_allclose(initial_pose.numpy()[0, :3], (1.0, 2.0, 3.0), atol=1.0e-6)

    def test_delta_reward_two_step_success_and_failure_override(self):
        count = 5
        episode_step = self._zeros(count, wp.int32)
        episode_return = self._zeros(count, wp.float32)
        success_once = self._zeros(count, wp.bool)
        distance = self._array((0.17, 0.215, 0.030, 0.030, 0.10), wp.float32)
        previous = self._array((0.20, 0.20, 0.040, 0.040, 0.11), wp.float32)
        hold_count = self._array((0, 0, 0, 1, 0), wp.int32)
        phase = self._zeros(count, wp.int32)
        success = self._zeros(count, wp.bool)
        fail = self._array((False, False, False, False, True), wp.bool)
        delta = self._zeros(count, wp.float32)
        dense = self._zeros(count, wp.float32)
        reward = self._zeros(count, wp.float32)
        terminated = self._zeros(count, wp.bool)
        truncated = self._zeros(count, wp.bool)

        wp.launch(
            env_module._advance_reach_episode,
            dim=count,
            inputs=[
                episode_step,
                episode_return,
                success_once,
                distance,
                previous,
                hold_count,
                0.040,
                2,
                0.030,
                phase,
                success,
                fail,
                delta,
                dense,
                reward,
                terminated,
                truncated,
                8,
                env_module._REWARD_MODE_NORMALIZED_DENSE,
                True,
                True,
            ],
            device=self.device,
        )

        np.testing.assert_allclose(reward.numpy(), (1.0, -0.5, 1.0 / 3.0, 1.0, -1.0), atol=1.0e-6)
        np.testing.assert_array_equal(success.numpy(), (False, False, False, True, False))
        np.testing.assert_array_equal(hold_count.numpy(), (0, 0, 1, 2, 0))
        np.testing.assert_array_equal(terminated.numpy(), (False, False, False, True, True))
        np.testing.assert_allclose(previous.numpy(), distance.numpy(), atol=0.0)

    def test_contact_or_displacement_fails_on_a_physics_frame(self):
        phase = self._zeros(4, wp.int32)
        fail = self._array((False, False, False, True), wp.bool)
        wp.launch(
            env_module._advance_reach_physics,
            dim=4,
            inputs=[
                self._array((False, True, False, False), wp.bool),
                self._array((False, False, True, False), wp.bool),
                phase,
                self._zeros(4, wp.bool),
                fail,
            ],
            device=self.device,
        )
        np.testing.assert_array_equal(fail.numpy(), (False, True, True, True))
        np.testing.assert_array_equal(
            phase.numpy(),
            (
                env_module._TASK_PHASE_APPROACH,
                env_module._TASK_PHASE_FAIL,
                env_module._TASK_PHASE_FAIL,
                env_module._TASK_PHASE_APPROACH,
            ),
        )

    def test_effective_action_projection_retains_only_xyz(self):
        import torch

        env = object.__new__(env_module.GrootNewtonEnv)
        env.task_mode = "reach_green_cap"
        env.num_envs = 2
        env.action_size = env_module.ACTION_SIZE
        env.device = wp.get_device(self.device)
        env._action = wp.zeros((2, env_module.ACTION_SIZE), dtype=wp.float32, device=self.device)
        hold = np.arange(2 * env_module.ACTION_SIZE, dtype=np.float32).reshape(2, -1)
        env._reach_hold_action = wp.array(hold, dtype=wp.float32, device=self.device)
        mask = np.zeros(env_module.ACTION_SIZE, dtype=np.bool_)
        mask[:3] = True
        env._effective_action_mask = wp.array(mask, dtype=wp.bool, device=self.device)

        raw = torch.full((2, env_module.ACTION_SIZE), -7.0, dtype=torch.float32)
        raw[0, :3] = torch.tensor((1.0, float("nan"), 3.0))
        raw[1, :3] = torch.tensor((4.0, 5.0, 6.0))
        projected = env.project_effective_action_torch(raw)

        np.testing.assert_allclose(projected[:, 3:].numpy(), hold[:, 3:], atol=0.0)
        np.testing.assert_allclose(projected[0, :3].numpy(), (1.0, hold[0, 1], 3.0), atol=0.0)
        np.testing.assert_allclose(projected[1, :3].numpy(), (4.0, 5.0, 6.0), atol=0.0)
        np.testing.assert_array_equal(env.effective_action_mask_torch().numpy(), mask)
        self.assertTrue(torch.equal(raw[:, 3:], torch.full_like(raw[:, 3:], -7.0)))

    @unittest.skipUnless(
        wp.is_cuda_available() and os.environ.get("REXPOLICY_RUN_GPU_ORACLE") == "1",
        "Set REXPOLICY_RUN_GPU_ORACLE=1 to run the production-physics Reach oracle",
    )
    def test_production_gpu_oracle_reaches_without_touching_bottle(self):
        import torch

        env = env_module.GrootNewtonEnv(
            task_mode="reach_green_cap",
            num_envs=4,
            device="cuda:0",
            control_hz=10,
            simulation_hz=60,
            substeps_per_frame=16,
            bottle_settle_frames=60,
            max_episode_steps=8,
            obs_mode="state_dict+rgb",
            render_images=True,
            camera_textures=True,
            load_scene_visuals=False,
            hydroelastic_contacts=False,
            # Graph capture is an execution optimization, not part of the
            # production physics contract, and is hardware/backend dependent.
            capture_graph=False,
            terminate_on_success=True,
            terminate_on_fail=True,
        )

        def rollout(seed=None):
            _, info = env.reset(seed=seed)
            env.canonicalize_branch_state_torch()
            reset_fingerprint = {
                f"{group_name}.{field_name}": value.clone()
                for group_name, group in env.replay_fingerprint_torch().items()
                for field_name, value in group.items()
            }
            for name, value in reset_fingerprint.items():
                if value.is_floating_point():
                    torch.testing.assert_close(value, value[:1].expand_as(value), atol=1.0e-5, rtol=0.0)
                else:
                    self.assertTrue(torch.equal(value, value[:1].expand_as(value)), name)
            goal = info["reach_goal_pos_base"].clone()
            rows = []
            branch_states = []
            branch_digests = []
            success_step = None
            previous_distance = float(env.evaluate()["reach_distance"][0])
            for step in range(1, 9):
                branch_fingerprint = env.replay_fingerprint_torch()
                branch_digests.append(
                    validate_replay_fingerprint(
                        branch_fingerprint,
                        world_count=env.num_envs,
                        float_tolerance=1.0e-5,
                    ).digest
                )
                branch_states.append(
                    {
                        f"{group_name}.{field_name}": value[0].detach().cpu().clone()
                        for group_name, group in branch_fingerprint.items()
                        for field_name, value in group.items()
                    }
                )
                for name, group in branch_fingerprint.items():
                    for field_name, value in group.items():
                        label = f"{name}.{field_name}"
                        if value.is_floating_point():
                            torch.testing.assert_close(
                                value,
                                value[:1].expand_as(value),
                                atol=1.0e-5,
                                rtol=0.0,
                                msg=label,
                            )
                        else:
                            self.assertTrue(
                                torch.equal(value, value[:1].expand_as(value)),
                                label,
                            )
                action = env.hold_action_torch().clone()
                action[:, :3].copy_(goal)
                _, reward, terminated, truncated, info = env.step(action)
                current_distance = float(info["reach_distance"][0])
                if bool(info["success"][0]):
                    expected_reward = 1.0
                elif bool(info["fail"][0]):
                    expected_reward = -1.0
                else:
                    expected_reward = float(
                        np.clip(
                            (previous_distance - current_distance) / 0.030,
                            -1.0,
                            1.0,
                        )
                    )
                self.assertAlmostEqual(
                    float(reward[0]),
                    expected_reward,
                    delta=1.0e-6,
                )
                rows.append(
                    (
                        current_distance,
                        float(info["reach_bottle_displacement"][0]),
                        float(reward[0]),
                        int(info["reach_success_hold_steps"][0]),
                    )
                )
                self.assertFalse(bool(info["reach_has_hand_contact"].any()))
                self.assertTrue(bool((info["success"] == info["success"][0]).all()))
                self.assertTrue(bool((info["fail"] == info["fail"][0]).all()))
                if bool(info["success"].all()):
                    success_step = step
                    break
                self.assertFalse(bool((terminated | truncated).any()))
                env.canonicalize_branch_state_torch()
                previous_distance = float(env.evaluate()["reach_distance"][0])
            return (
                success_step,
                np.asarray(rows, dtype=np.float32),
                reset_fingerprint,
                branch_states,
                branch_digests,
            )

        try:
            _, seeded_info = env.reset(seed=41)
            env.canonicalize_branch_state_torch()
            first_seeded_offset = seeded_info["reach_reset_xy_offset"].clone()
            first_seeded_pose = env.replay_fingerprint_torch()["task"][
                "obj_pose"
            ].clone()
            self.assertTrue(
                torch.equal(
                    first_seeded_offset,
                    first_seeded_offset[:1].expand_as(first_seeded_offset),
                )
            )
            _, repeated_info = env.reset(seed=41)
            env.canonicalize_branch_state_torch()
            torch.testing.assert_close(
                repeated_info["reach_reset_xy_offset"],
                first_seeded_offset,
                atol=0.0,
                rtol=0.0,
            )
            torch.testing.assert_close(
                env.replay_fingerprint_torch()["task"]["obj_pose"],
                first_seeded_pose,
                atol=1.0e-5,
                rtol=0.0,
            )
            _, distinct_info = env.reset(seed=42)
            env.canonicalize_branch_state_torch()
            self.assertFalse(
                torch.equal(
                    distinct_info["reach_reset_xy_offset"],
                    first_seeded_offset,
                )
            )
            (
                first_step,
                first_rows,
                first_reset,
                first_branches,
                first_digests,
            ) = rollout()
            (
                second_step,
                second_rows,
                second_reset,
                second_branches,
                second_digests,
            ) = rollout()
            seeded_step, seeded_rows, _, _, _ = rollout(seed=41)
            self.assertIsNotNone(first_step)
            self.assertLessEqual(first_step, 8)
            self.assertEqual(first_step, 6)
            self.assertEqual(second_step, first_step)
            self.assertIsNotNone(seeded_step)
            self.assertLessEqual(seeded_step, 8)
            self.assertLessEqual(float(first_rows[:, 1].max()), 0.005)
            self.assertLessEqual(float(seeded_rows[:, 1].max()), 0.005)
            np.testing.assert_allclose(second_rows[:, :2], first_rows[:, :2], atol=1.0e-5, rtol=0.0)
            reward_replay_tolerance = 2.0 * 1.0e-5 / 0.030 + 1.0e-6
            np.testing.assert_allclose(
                second_rows[:, 2],
                first_rows[:, 2],
                atol=reward_replay_tolerance,
                rtol=0.0,
            )
            np.testing.assert_array_equal(second_rows[:, 3], first_rows[:, 3])
            self.assertEqual(second_reset.keys(), first_reset.keys())
            for name in first_reset:
                if first_reset[name].is_floating_point():
                    torch.testing.assert_close(second_reset[name], first_reset[name], atol=1.0e-5, rtol=0.0)
                else:
                    self.assertTrue(torch.equal(second_reset[name], first_reset[name]), name)
            self.assertEqual(len(second_branches), len(first_branches))
            self.assertEqual(second_digests, first_digests)
            for step, (first_branch, second_branch) in enumerate(
                zip(first_branches, second_branches, strict=True),
                start=1,
            ):
                self.assertEqual(second_branch.keys(), first_branch.keys())
                for name in first_branch:
                    if first_branch[name].is_floating_point():
                        torch.testing.assert_close(
                            second_branch[name],
                            first_branch[name],
                            atol=1.0e-5,
                            rtol=0.0,
                            msg=f"replay step={step} field={name}",
                        )
                    else:
                        self.assertTrue(
                            torch.equal(second_branch[name], first_branch[name]),
                            f"replay step={step} field={name}",
                        )

            fingerprint = env.replay_fingerprint_torch()
            for group in ("joint", "body", "control", "task", "contacts"):
                self.assertIn(group, fingerprint)
            self.assertTrue(all(isinstance(value, torch.Tensor) for value in fingerprint["joint"].values()))
            for group in fingerprint.values():
                for value in group.values():
                    self.assertEqual(value.shape[0], env.num_envs)
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
