#!/usr/bin/env python3
"""Run the minimal no-image Stage 0 Reach validation on Newton."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import warp as wp

from rexpolicy.envs.groot_newton_env import GrootNewtonEnv, GrootNewtonEnvConfig
from rexpolicy.stage0.envs.oracles import ReachSuccessOracle
from rexpolicy.stage0.envs.reset_sampler import Stage0ResetSampler
from rexpolicy.stage0.envs.state_only import Stage0StateOnlyEnv
from rexpolicy.stage0.envs.state_schema import DEFAULT_STAGE0_STATE_SCHEMA


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260805)
    parser.add_argument("--capture-graph", action="store_true")
    parser.add_argument("--hydroelastic-contacts", action="store_true")
    parser.add_argument("--require-success", action="store_true")
    parser.add_argument("--report-json", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_envs < 1 or args.steps < 1:
        raise ValueError("--num-envs and --steps must be positive")

    config = GrootNewtonEnvConfig(
        num_envs=args.num_envs,
        device=args.device,
        task_mode="reach_green_cap",
        obs_mode="state",
        control_mode="pd_eef_pose_abs",
        reward_mode="none",
        max_episode_steps=args.steps,
        terminate_on_success=True,
        terminate_on_fail=True,
        capture_graph=args.capture_graph,
        render_images=False,
        camera_textures=False,
        load_scene_visuals=False,
        hydroelastic_contacts=args.hydroelastic_contacts,
    )
    env = GrootNewtonEnv(config)
    try:
        stage0_env = Stage0StateOnlyEnv(env)
        sampler = Stage0ResetSampler(args.seed, args.num_envs)
        state, _ = stage0_env.reset(seed=sampler.next())
        oracle = ReachSuccessOracle(
            args.num_envs,
            success_threshold_m=config.reach_success_threshold,
            success_hold_steps=config.reach_success_hold_steps,
            displacement_limit_m=config.reach_bottle_displacement_limit,
        )
        oracle.reset(state)

        torch.cuda.synchronize(state.device)
        started = time.perf_counter()
        oracle_result = oracle.evaluate(state)
        transition_count = 0
        for _ in range(args.steps):
            # The environment already clamps this absolute target through its
            # production IK limits; using the final goal matches its Reach
            # replay/oracle contract and normally succeeds within eight steps.
            action = stage0_env.reach_heuristic_action(max_translation_step_m=10.0)
            transition = stage0_env.step(action)
            transition_count += args.num_envs
            oracle_result = oracle.evaluate(transition.next_state)
            env_success = transition.info["success"].to(dtype=torch.bool)
            env_failure = transition.info["fail"].to(dtype=torch.bool)
            if not torch.equal(oracle_result.success, env_success):
                raise RuntimeError(
                    "Stage 0 Reach oracle disagrees with Newton success flags"
                )
            if not torch.equal(oracle_result.failure, env_failure):
                raise RuntimeError(
                    "Stage 0 Reach oracle disagrees with Newton failure flags"
                )
            if bool((oracle_result.success | oracle_result.failure).all()):
                break
        torch.cuda.synchronize(state.device)
        elapsed = time.perf_counter() - started

        report = {
            "system_path": "stage0_state",
            "task": "reach",
            "device": str(state.device),
            "num_envs": args.num_envs,
            "state_shape": list(state.shape),
            "action_shape": [args.num_envs, 19],
            "state_schema_id": DEFAULT_STAGE0_STATE_SCHEMA.schema_id,
            "state_schema_sha256": DEFAULT_STAGE0_STATE_SCHEMA.sha256,
            "steps_executed": transition_count // args.num_envs,
            "world_steps": transition_count,
            "elapsed_seconds": elapsed,
            "world_steps_per_second": transition_count / elapsed,
            "success_count": int(oracle_result.success.sum().item()),
            "failure_count": int(oracle_result.failure.sum().item()),
            "mean_final_distance_m": float(oracle_result.distance.mean().item()),
            "max_object_displacement_m": float(
                oracle_result.object_displacement.max().item()
            ),
            "finite_state": bool(torch.isfinite(stage0_env.observe()).all()),
            "contiguous_state": bool(stage0_env.observe().is_contiguous()),
            "images_enabled": False,
            "groot_enabled": False,
        }
        rendered = json.dumps(report, indent=2, sort_keys=True)
        print(rendered)
        if args.report_json is not None:
            args.report_json.parent.mkdir(parents=True, exist_ok=True)
            args.report_json.write_text(rendered + "\n", encoding="utf-8")
        if args.require_success and report["success_count"] != args.num_envs:
            raise SystemExit(2)
    finally:
        env.close()
        if wp.is_cuda_available():
            wp.synchronize()


if __name__ == "__main__":
    main()
