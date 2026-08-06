#!/usr/bin/env python3
"""GPU integration gate for repeated same-reset Stage 0 Reach modes."""

from __future__ import annotations

import argparse
import json

import torch

from rexpolicy.envs.groot_newton_env import GrootNewtonEnv, GrootNewtonEnvConfig
from rexpolicy.stage0.envs.oracles import ReachSuccessOracle
from rexpolicy.stage0.envs.reach_modes import (
    DEFAULT_REACH_MODE_LIBRARY,
    ReachModeController,
)
from rexpolicy.stage0.envs.state_only import Stage0StateOnlyEnv
from rexpolicy.stage0.envs.state_schema import DEFAULT_STAGE0_STATE_SCHEMA


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260806)
    parser.add_argument("--max-episode-steps", type=int, default=32)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.num_envs < 1 or args.max_episode_steps < 1:
        raise ValueError("num-envs and max-episode-steps must be positive")
    config = GrootNewtonEnvConfig(
        num_envs=args.num_envs,
        device=args.device,
        task_mode="reach_green_cap",
        obs_mode="state",
        control_mode="pd_eef_pose_abs",
        reward_mode="none",
        max_episode_steps=args.max_episode_steps,
        terminate_on_success=True,
        terminate_on_fail=True,
        capture_graph=False,
        render_images=False,
        camera_textures=False,
        load_scene_visuals=False,
        hydroelastic_contacts=False,
    )
    seeds = tuple(args.seed + world for world in range(args.num_envs))
    library = DEFAULT_REACH_MODE_LIBRARY
    schema = DEFAULT_STAGE0_STATE_SCHEMA
    eef_slice = schema.slice("eef_position")
    initial_states: list[torch.Tensor] = []
    traces: list[torch.Tensor] = []
    mode_successes = {mode_id: 0 for mode_id in library.mode_ids}
    mode_failures = {mode_id: 0 for mode_id in library.mode_ids}
    mode_timeouts = {mode_id: 0 for mode_id in library.mode_ids}
    mode_lengths: dict[str, list[int]] = {mode_id: [] for mode_id in library.mode_ids}
    maximum_displacement = 0.0
    maximum_terminal_state_distance = 0.0
    contact_count = 0

    env = GrootNewtonEnv(config)
    try:
        adapter = Stage0StateOnlyEnv(env)
        for slot in range(len(library.mode_ids)):
            mode_ids = library.assign(seeds, slot)
            state, _ = adapter.reset(seed=list(seeds))
            initial_states.append(state.detach().cpu().clone())
            controller = ReachModeController()
            controller.reset(state, env.hold_action_torch().clone(), mode_ids)
            oracle = ReachSuccessOracle(
                args.num_envs,
                success_threshold_m=config.reach_success_threshold,
                success_hold_steps=config.reach_success_hold_steps,
                displacement_limit_m=config.reach_bottle_displacement_limit,
            )
            oracle.reset(state)
            active = torch.ones(args.num_envs, dtype=torch.bool, device=state.device)
            outcomes: list[str | None] = [None] * args.num_envs
            lengths = [0] * args.num_envs
            trace = [state[:, eef_slice].detach().cpu().clone()]
            for _ in range(args.max_episode_steps):
                action = controller.next_action(state, active)
                transition = adapter.step(action)
                state = transition.next_state
                result = oracle.evaluate(state)
                env_success = transition.info["success"].to(dtype=torch.bool)
                if bool(env_success.any()):
                    maximum_terminal_state_distance = max(
                        maximum_terminal_state_distance,
                        float(result.distance[env_success].max()),
                    )
                if not torch.equal(
                    result.failure,
                    transition.info["fail"].to(dtype=torch.bool),
                ):
                    raise RuntimeError(
                        "independent oracle disagrees with Newton failure"
                    )
                maximum_displacement = max(
                    maximum_displacement,
                    float(result.object_displacement.max()),
                )
                contact_count += int((result.contact_violation & active).sum())
                trace.append(state[:, eef_slice].detach().cpu().clone())
                for world in torch.nonzero(active, as_tuple=False).flatten().tolist():
                    lengths[world] += 1
                    if bool(env_success[world]):
                        outcomes[world] = "success"
                        active[world] = False
                    elif bool(result.failure[world]):
                        outcomes[world] = "failure"
                        active[world] = False
                if not bool(active.any()):
                    break
            for world in torch.nonzero(active, as_tuple=False).flatten().tolist():
                outcomes[world] = "timeout"
            for world, outcome in enumerate(outcomes):
                if outcome is None:
                    raise RuntimeError("world has no terminal outcome")
                mode_id = mode_ids[world]
                if outcome == "success":
                    mode_successes[mode_id] += 1
                elif outcome == "failure":
                    mode_failures[mode_id] += 1
                else:
                    mode_timeouts[mode_id] += 1
                mode_lengths[mode_id].append(lengths[world])
            traces.append(torch.stack(trace))
    finally:
        env.close()

    initial_reference = initial_states[0]
    initial_max_delta = max(
        float((state - initial_reference).abs().max()) for state in initial_states[1:]
    )
    pairwise_rms: dict[str, float] = {}
    for left in range(len(traces)):
        for right in range(left + 1, len(traces)):
            common = min(traces[left].shape[0], traces[right].shape[0])
            delta = traces[left][:common] - traces[right][:common]
            rms_by_world = delta.square().sum(dim=-1).mean(dim=0).sqrt()
            key = f"slot-{left}-vs-{right}"
            pairwise_rms[key] = float(rms_by_world.mean())
    result = {
        "contact_count": contact_count,
        "initial_state_max_abs_delta": initial_max_delta,
        "maximum_object_displacement_m": maximum_displacement,
        "maximum_terminal_state_distance_m": maximum_terminal_state_distance,
        "minimum_pairwise_eef_rms_m": min(pairwise_rms.values()),
        "mode_failures": mode_failures,
        "mode_lengths": {
            mode_id: {
                "maximum": max(lengths),
                "minimum": min(lengths),
            }
            for mode_id, lengths in mode_lengths.items()
        },
        "mode_successes": mode_successes,
        "mode_timeouts": mode_timeouts,
        "pairwise_eef_rms_m": pairwise_rms,
        "reset_count": args.num_envs,
    }
    result["gate_passed"] = bool(
        initial_max_delta <= 1.0e-6
        and all(value == args.num_envs for value in mode_successes.values())
        and not any(mode_failures.values())
        and not any(mode_timeouts.values())
        and contact_count == 0
        and maximum_displacement <= 0.005
        and min(pairwise_rms.values()) >= 0.030
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["gate_passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
