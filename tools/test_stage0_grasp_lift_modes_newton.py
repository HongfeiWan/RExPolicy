#!/usr/bin/env python3
"""CUDA/Newton integration gate for Stage 0 Grasp-Lift authoring."""

from __future__ import annotations

import argparse
import json

import torch

from rexpolicy.envs.groot_newton_env import GrootNewtonEnv, GrootNewtonEnvConfig
from rexpolicy.stage0.envs.grasp_lift_action import (
    DEFAULT_GRASP_LIFT_ACTION_SCHEMA,
    GRASP_LIFT_EFFECTIVE_ACTION_MASK,
)
from rexpolicy.stage0.envs.grasp_lift_modes import (
    DEFAULT_GRASP_LIFT_AUTHORING_MODE,
    GraspLiftAuthoringController,
    GraspLiftAuthoringPhase,
)
from rexpolicy.stage0.envs.grasp_lift_oracle import (
    GraspLiftSuccessOracle,
    grasp_lift_transient_events_from_info,
)
from rexpolicy.stage0.envs.state_only import Stage0StateOnlyEnv


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=7000)
    parser.add_argument("--max-episode-steps", type=int, default=72)
    parser.add_argument("--minimum-success-rate", type=float, default=1.0)
    parser.add_argument("--rigid-contacts-per-env", type=int, default=2048)
    parser.add_argument("--triangle-pairs-per-env", type=int, default=1_500_000)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if min(
        args.num_envs,
        args.max_episode_steps,
        args.rigid_contacts_per_env,
        args.triangle_pairs_per_env,
    ) < 1:
        raise ValueError("world, horizon, and collision capacities must be positive")
    if not 0.0 <= args.minimum_success_rate <= 1.0:
        raise ValueError("minimum-success-rate must be in [0, 1]")
    seeds = tuple(args.seed + world for world in range(args.num_envs))
    config = GrootNewtonEnvConfig(
        task_mode="grasp_lift_green_bottle",
        num_envs=args.num_envs,
        device=args.device,
        max_episode_steps=args.max_episode_steps,
        obs_mode="state",
        reward_mode="none",
        render_images=False,
        camera_textures=False,
        load_scene_visuals=False,
        hydroelastic_contacts=False,
        capture_graph=False,
        terminate_on_success=False,
        terminate_on_fail=False,
        rigid_contacts_per_env=args.rigid_contacts_per_env,
        triangle_pairs_per_env=args.triangle_pairs_per_env,
    )

    env = GrootNewtonEnv(config)
    try:
        adapter = Stage0StateOnlyEnv(env)
        state, _ = adapter.reset(seed=list(seeds))
        controller = GraspLiftAuthoringController()
        controller.reset(state, env.hold_action_torch().clone())
        oracle = GraspLiftSuccessOracle(
            args.num_envs,
            lift_height_m=config.grasp_lift_height,
            success_hold_control_steps=config.grasp_lift_hold_control_steps,
            lateral_displacement_limit_m=config.grasp_lateral_displacement_limit,
            bottle_tilt_limit_rad=config.grasp_bottle_tilt_limit_rad,
            grasp_confirm_physics_frames=config.grasp_confirm_frames,
        )
        oracle.reset(state)

        expected_mask = torch.tensor(
            GRASP_LIFT_EFFECTIVE_ACTION_MASK,
            dtype=torch.bool,
            device=state.device,
        )
        if not torch.equal(env.effective_action_mask_torch(), expected_mask):
            raise RuntimeError("Newton effective action mask changed")
        if env.frames_per_action != DEFAULT_GRASP_LIFT_AUTHORING_MODE.full_opposed_physics_frames:
            raise RuntimeError("authoring mode physics-frame contract changed")

        maximum_projection_delta = 0.0
        reward_nonzero_count = 0
        projection_drift_step_count = 0
        executed_step_count = 0
        latest = None
        latest_events = None
        for _ in range(args.max_episode_steps):
            proposal = controller.propose(state)
            projected = env.project_effective_action_torch(proposal)
            projection_delta = torch.amax(torch.abs(projected - proposal))
            maximum_projection_delta = max(
                maximum_projection_delta, float(projection_delta)
            )
            projection_drift_step_count += int(bool(projection_delta > 0.0))
            transition = adapter.step(projected)
            executed = env.effective_action_torch().detach().clone()
            if not torch.equal(executed, projected):
                raise RuntimeError("executed action drifted after projection")
            executed_step_count += 1
            reward_nonzero_count += int(torch.count_nonzero(transition.reward))
            latest_events = grasp_lift_transient_events_from_info(
                transition.info,
                count=args.num_envs,
                device=state.device,
            )
            latest = oracle.evaluate(transition.next_state, latest_events)
            evaluation = env.evaluate()
            controller.observe(
                executed_action=executed,
                next_state=transition.next_state,
                oracle_result=latest,
                events=latest_events,
                is_obj_static=evaluation["is_obj_static"],
                is_robot_static=evaluation["is_robot_static"],
            )
            state = transition.next_state
            if bool(controller.terminal_mask.all()):
                break

        if latest is None or latest_events is None:
            raise RuntimeError("integration gate executed no transitions")
        phases = controller.phase.detach().cpu().tolist()
        successes = latest.success.detach().cpu()
        failures = latest.failure.detach().cpu()
        success_count = int(successes.sum())
        failure_count = int(failures.sum())
        success_rate = success_count / float(args.num_envs)
        violations = {
            "bottle_tilt": int(latest.bottle_tilt_violation.sum()),
            "collision_buffer_overflow": int(
                latest.collision_buffer_overflow_violation.sum()
            ),
            "dropped_grasp": int(latest.dropped_grasp_violation.sum()),
            "forbidden_contact": int(latest.forbidden_contact_violation.sum()),
            "lateral_displacement": int(
                latest.lateral_displacement_violation.sum()
            ),
        }
        result = {
            "action_schema_sha256": DEFAULT_GRASP_LIFT_ACTION_SCHEMA.sha256,
            "controller_mode_id": DEFAULT_GRASP_LIFT_AUTHORING_MODE.mode_id,
            "controller_mode_sha256": DEFAULT_GRASP_LIFT_AUTHORING_MODE.sha256,
            "executed_step_count": executed_step_count,
            "failure_count": failure_count,
            "final_phase": phases,
            "frames_per_action": env.frames_per_action,
            "gate_passed": False,
            "maximum_bottle_tilt_rad": float(latest.bottle_tilt.max()),
            "maximum_lateral_displacement_m": float(
                latest.lateral_displacement.max()
            ),
            "maximum_lift_height_m": float(latest.lift_height.max()),
            "maximum_projection_delta": maximum_projection_delta,
            "minimum_success_rate": args.minimum_success_rate,
            "num_envs": args.num_envs,
            "projection_drift_step_count": projection_drift_step_count,
            "reward_nonzero_count": reward_nonzero_count,
            "seeds": list(seeds),
            "success_count": success_count,
            "success_rate": success_rate,
            "violations": violations,
        }
        succeeded_phase = int(GraspLiftAuthoringPhase.SUCCEEDED)
        result["gate_passed"] = bool(
            success_rate >= args.minimum_success_rate
            and failure_count == 0
            and reward_nonzero_count == 0
            and not any(violations.values())
            and all(
                phase == succeeded_phase
                for phase, success in zip(phases, successes.tolist(), strict=True)
                if success
            )
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        if not result["gate_passed"]:
            raise SystemExit(2)
    finally:
        env.close()


if __name__ == "__main__":
    main()
