#!/usr/bin/env python3
"""Collect, train, diagnose, and checkpoint the independent Stage 0 loop."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch

from rexpolicy.envs.groot_newton_env import GrootNewtonEnv, GrootNewtonEnvConfig
from rexpolicy.stage0.checkpoint import (
    Stage0CheckpointHashes,
    Stage0CheckpointManager,
)
from rexpolicy.stage0.config import Stage0Config
from rexpolicy.stage0.data import (
    Stage0SplitPolicy,
    Stage0Trajectory,
    Stage0TrajectoryProvenance,
    Stage0WindowPolicy,
    build_success_window_splits,
    trajectory_corpus_sha256,
    write_trajectory_shard,
)
from rexpolicy.stage0.envs.oracles import ReachSuccessOracle
from rexpolicy.stage0.envs.reset_sampler import Stage0ResetSampler
from rexpolicy.stage0.envs.state_only import Stage0StateOnlyEnv
from rexpolicy.stage0.envs.state_schema import DEFAULT_STAGE0_STATE_SCHEMA
from rexpolicy.stage0.evaluation import analyze_latents, deterministic_pca
from rexpolicy.stage0.models import FutureTrajectoryEncoder, SuccessModeSelector
from rexpolicy.stage0.runtime import Stage0FlowPolicy
from rexpolicy.stage0.trainers import (
    Stage0ManifoldTrainer,
    Stage0ManifoldTrainerConfig,
    Stage0PolicyTrainer,
    Stage0SelectorTrainer,
    collate_success_windows,
)
from rexpolicy.stage0.types import Stage0Outcome, canonical_fingerprint


@dataclass(frozen=True)
class CollectionResult:
    trajectories: tuple[Stage0Trajectory, ...]
    reset_seeds: tuple[int, ...]
    elapsed_seconds: float
    world_steps: int


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/stage0/reach_smoke.json")
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--manifold-steps", type=int, default=20)
    parser.add_argument("--selector-steps", type=int, default=20)
    parser.add_argument("--policy-steps", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--capture-graph", action="store_true")
    return parser.parse_args()


def _load_config(path: Path) -> Stage0Config:
    raw = json.loads(path.read_text(encoding="utf-8"))
    config = Stage0Config.from_mapping(raw)
    config.require_enabled_stage0()
    if config.task.value != "reach":
        raise RuntimeError("The first Stage 0 gate supports Reach only")
    return config


def _default_output_dir() -> Path:
    return Path("outputs/stage0") / f"reach-{time.time_ns()}"


def _environment_config(
    config: Stage0Config, device: str, *, capture_graph: bool
) -> GrootNewtonEnvConfig:
    return GrootNewtonEnvConfig(
        num_envs=config.num_envs,
        device=device,
        task_mode="reach_green_cap",
        obs_mode="state",
        control_mode="pd_eef_pose_abs",
        reward_mode="none",
        max_episode_steps=config.max_episode_steps,
        terminate_on_success=True,
        terminate_on_fail=True,
        capture_graph=capture_graph,
        render_images=False,
        camera_textures=False,
        load_scene_visuals=False,
        hydroelastic_contacts=False,
    )


def _failure_reason(result: Any, world: int) -> str:
    if bool(result.contact_violation[world]):
        return "unsafe-hand-object-contact"
    if bool(result.displacement_violation[world]):
        return "object-displacement-limit"
    return "newton-oracle-failure"


def _collect_reach(
    config: Stage0Config,
    *,
    device: str,
    capture_graph: bool,
) -> CollectionResult:
    env_config = _environment_config(config, device, capture_graph=capture_graph)
    env = GrootNewtonEnv(env_config)
    try:
        adapter = Stage0StateOnlyEnv(env)
        sampler = Stage0ResetSampler(config.seed, config.num_envs)
        reset_seeds = tuple(sampler.next())
        state, _ = adapter.reset(seed=list(reset_seeds))
        oracle = ReachSuccessOracle(
            config.num_envs,
            success_threshold_m=env_config.reach_success_threshold,
            success_hold_steps=env_config.reach_success_hold_steps,
            displacement_limit_m=env_config.reach_bottle_displacement_limit,
        )
        oracle.reset(state)

        state_rows: list[list[torch.Tensor]] = [
            [state[world].detach().cpu().clone()] for world in range(config.num_envs)
        ]
        action_rows: list[list[torch.Tensor]] = [[] for _ in range(config.num_envs)]
        outcomes: list[Stage0Outcome | None] = [None] * config.num_envs
        failure_reasons: list[str | None] = [None] * config.num_envs
        active = torch.ones(config.num_envs, dtype=torch.bool, device=state.device)

        torch.cuda.synchronize(state.device)
        started = time.perf_counter()
        world_steps = 0
        for _ in range(config.max_episode_steps):
            action = adapter.reach_heuristic_action(max_translation_step_m=10.0)
            transition = adapter.step(action)
            result = oracle.evaluate(transition.next_state)
            env_success = transition.info["success"].to(dtype=torch.bool)
            env_failure = transition.info["fail"].to(dtype=torch.bool)
            if not torch.equal(result.success, env_success):
                raise RuntimeError(
                    "Independent Reach oracle disagrees with Newton success"
                )
            if not torch.equal(result.failure, env_failure):
                raise RuntimeError(
                    "Independent Reach oracle disagrees with Newton failure"
                )

            active_worlds = torch.nonzero(active, as_tuple=False).flatten().tolist()
            for world in active_worlds:
                action_rows[world].append(action[world].detach().cpu().clone())
                state_rows[world].append(
                    transition.next_state[world].detach().cpu().clone()
                )
                world_steps += 1
                if bool(result.success[world]):
                    outcomes[world] = Stage0Outcome.SUCCESS
                    active[world] = False
                elif bool(result.failure[world]):
                    outcomes[world] = Stage0Outcome.FAILURE
                    failure_reasons[world] = _failure_reason(result, world)
                    active[world] = False
            if not bool(active.any()):
                break
        torch.cuda.synchronize(state.device)
        elapsed = time.perf_counter() - started

        for world in torch.nonzero(active, as_tuple=False).flatten().tolist():
            outcomes[world] = Stage0Outcome.TIMEOUT
        simulator_sha256 = canonical_fingerprint(
            {
                "schema_id": "rexpolicy/stage0-newton-runtime/v1",
                "config": asdict(env_config),
            }
        )
        action_schema_sha256 = canonical_fingerprint(
            {
                "schema_id": "rexpolicy/stage0-eef-action/v1",
                "dimension": config.action_dim,
                "fields": ["absolute_eef_xyz", "rotation_6d", "absolute_hand_q"],
            }
        )
        trajectories = []
        for world, outcome in enumerate(outcomes):
            if outcome is None:
                raise RuntimeError(f"World {world} did not receive a terminal outcome")
            trajectories.append(
                Stage0Trajectory(
                    provenance=Stage0TrajectoryProvenance(
                        trajectory_id=f"reach-r0-w{world:05d}-s{reset_seeds[world]:010d}",
                        reset_group_id=f"reset-s{reset_seeds[world]:010d}",
                        task_id="reach_green_cap/v1",
                        oracle_id="stage0/reach-oracle/v1",
                        simulator_sha256=simulator_sha256,
                        experiment_sha256=config.fingerprint,
                        state_schema_sha256=DEFAULT_STAGE0_STATE_SCHEMA.sha256,
                        action_schema_sha256=action_schema_sha256,
                        generation=0,
                        rank=0,
                        world=world,
                        reset_seed=reset_seeds[world],
                    ),
                    states=torch.stack(state_rows[world]),
                    actions=torch.stack(action_rows[world]),
                    outcome=outcome,
                    terminated=outcome is not Stage0Outcome.TIMEOUT,
                    truncated=outcome is Stage0Outcome.TIMEOUT,
                    failure_reason=failure_reasons[world],
                )
            )
        return CollectionResult(
            trajectories=tuple(trajectories),
            reset_seeds=reset_seeds,
            elapsed_seconds=elapsed,
            world_steps=world_steps,
        )
    finally:
        env.close()


def _write_corpus(output_dir: Path, trajectories: Iterable[Stage0Trajectory]) -> None:
    corpus_dir = output_dir / "corpus"
    for trajectory in trajectories:
        write_trajectory_shard(corpus_dir, trajectory)


def _sample_windows(
    windows: Sequence[Any],
    *,
    batch_size: int,
    generator: torch.Generator,
) -> list[Any]:
    if not windows:
        raise RuntimeError("Stage 0 training split has no success windows")
    indices = torch.randint(len(windows), (batch_size,), generator=generator).tolist()
    return [windows[index] for index in indices]


def _sample_contrastive_windows(
    windows: Sequence[Any],
    *,
    batch_size: int,
    generator: torch.Generator,
) -> list[Any]:
    """Sample adjacent pairs from at least two successful trajectories."""
    if batch_size < 4 or batch_size % 2:
        raise ValueError("manifold batch_size must be an even integer >= 4")
    by_trajectory: dict[str, list[Any]] = {}
    for window in windows:
        by_trajectory.setdefault(window.trajectory_id, []).append(window)
    eligible = [
        sorted(items, key=lambda item: item.start)
        for items in by_trajectory.values()
        if len(items) >= 2
    ]
    if len(eligible) < 2:
        raise RuntimeError(
            "manifold training requires two trajectories with adjacent windows"
        )
    order = torch.randperm(len(eligible), generator=generator).tolist()
    pairs = batch_size // 2
    result: list[Any] = []
    for pair_index in range(pairs):
        trajectory = eligible[order[pair_index % len(order)]]
        start = int(
            torch.randint(len(trajectory) - 1, (1,), generator=generator).item()
        )
        left, right = trajectory[start], trajectory[start + 1]
        if right.start - left.start > 1:
            raise RuntimeError(
                "contrastive pair is outside the configured temporal radius"
            )
        result.extend((left, right))
    return result


@torch.no_grad()
def _encode_windows(
    encoder: FutureTrajectoryEncoder,
    windows: Sequence[Any],
    *,
    device: torch.device,
) -> tuple[Any, torch.Tensor]:
    batch = collate_success_windows(windows).to(device)
    encoding = encoder(
        batch.future_states,
        batch.future_actions,
        padding_mask=batch.future_padding_mask,
    )
    return batch, encoding.latent


def _masked_mse(
    prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> float:
    error = (prediction - target).square()
    return float(error.masked_select(mask.unsqueeze(-1)).mean())


def _train_stage0(
    config: Stage0Config,
    window_splits: Any,
    *,
    device: torch.device,
    manifold_steps: int,
    selector_steps: int,
    policy_steps: int,
    batch_size: int,
) -> tuple[
    FutureTrajectoryEncoder,
    SuccessModeSelector,
    Stage0FlowPolicy,
    dict[str, torch.optim.Optimizer],
    dict[str, Any],
]:
    cpu_generator = torch.Generator().manual_seed(config.seed)
    model_generator = torch.Generator(device=device).manual_seed(config.seed)
    encoder = FutureTrajectoryEncoder(
        config.state_dim,
        config.action_dim,
        latent_dim=config.latent_dim,
        projection_dim=config.latent_dim,
        model_dim=config.model_dim,
        max_future_steps=config.future_horizon,
        transformer_layers=config.transformer_layers,
        attention_heads=config.attention_heads,
        feedforward_dim=config.feedforward_dim,
        dropout=config.dropout,
    ).to(device)
    selector = SuccessModeSelector(
        config.state_dim,
        config.latent_dim,
        hidden_dim=config.model_dim,
        layers=2,
    ).to(device)
    policy = Stage0FlowPolicy(
        state_dim=config.state_dim,
        latent_dim=config.latent_dim,
        action_dim=config.action_dim,
        action_horizon=config.action_horizon,
        model_dim=config.model_dim,
        transformer_layers=config.transformer_layers,
        attention_heads=config.attention_heads,
        feedforward_dim=config.feedforward_dim,
        dropout=config.dropout,
    ).to(device)
    optimizers = {
        "future_encoder": torch.optim.AdamW(encoder.parameters(), lr=3.0e-4),
        "selector": torch.optim.AdamW(selector.parameters(), lr=3.0e-4),
        "policy": torch.optim.AdamW(policy.parameters(), lr=3.0e-4),
    }
    manifold_trainer = Stage0ManifoldTrainer(
        encoder,
        optimizers["future_encoder"],
        config=Stage0ManifoldTrainerConfig(
            reconstruction_probability=0.25,
            temporal_radius=1,
        ),
    )
    manifold_metrics = None
    for _ in range(manifold_steps):
        windows = _sample_contrastive_windows(
            window_splits.train,
            batch_size=batch_size,
            generator=cpu_generator,
        )
        manifold_metrics = manifold_trainer.step(
            collate_success_windows(windows),
            generator=model_generator,
        )

    encoder.eval()
    selector_trainer = Stage0SelectorTrainer(selector, optimizers["selector"])
    selector_metrics = None
    for _ in range(selector_steps):
        windows = _sample_windows(
            window_splits.train,
            batch_size=batch_size,
            generator=cpu_generator,
        )
        batch, target_latents = _encode_windows(encoder, windows, device=device)
        selector_metrics = selector_trainer.step(batch.current_states, target_latents)

    policy_trainer = Stage0PolicyTrainer(policy, optimizers["policy"])
    policy_metrics = None
    for _ in range(policy_steps):
        windows = _sample_windows(
            window_splits.train,
            batch_size=batch_size,
            generator=cpu_generator,
        )
        batch, success_latents = _encode_windows(encoder, windows, device=device)
        policy_metrics = policy_trainer.step(
            batch,
            success_latents,
            generator=model_generator,
        )
    if manifold_metrics is None or selector_metrics is None or policy_metrics is None:
        raise RuntimeError("Stage 0 training did not execute all phases")
    return (
        encoder,
        selector,
        policy,
        optimizers,
        {
            "manifold": asdict(manifold_metrics),
            "selector": asdict(selector_metrics),
            "policy": asdict(policy_metrics),
        },
    )


@torch.no_grad()
def _evaluate_offline(
    encoder: FutureTrajectoryEncoder,
    selector: SuccessModeSelector,
    policy: Stage0FlowPolicy,
    windows: Sequence[Any],
    *,
    device: torch.device,
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    selected = list(windows[: min(len(windows), 64)])
    if len(selected) < 2:
        raise RuntimeError("held-out latent diagnostics require at least two windows")
    encoder.eval()
    selector.eval()
    policy.eval()
    batch, oracle_latent = _encode_windows(encoder, selected, device=device)
    selector_distribution = selector(batch.current_states)
    selector_latent = selector_distribution.mode
    latent_statistics = analyze_latents(oracle_latent)
    pca = deterministic_pca(oracle_latent, components=2)

    control_count = min(8, batch.batch_size)
    state = batch.current_states[:control_count]
    oracle = oracle_latent[:control_count]
    selected_latent = selector_latent[:control_count]
    action_mask = batch.action_mask[:control_count]
    target_actions = batch.action_chunks[:control_count]
    generator = torch.Generator(device=device).manual_seed(seed + 1)
    initial_noise = torch.randn(
        target_actions.shape,
        dtype=target_actions.dtype,
        device=device,
        generator=generator,
    )
    controls = policy.sample_controls(
        state,
        oracle_latent=oracle,
        selector_latent=selected_latent,
        steps=8,
        action_mask=action_mask,
        initial_noise=initial_noise,
    )
    offline = {
        "authority": "offline_smoke_only",
        "not_a_rollout_success_claim": True,
        "no_z_action_mse": _masked_mse(controls.no_z, target_actions, action_mask),
        "oracle_z_action_mse": _masked_mse(
            controls.oracle_z, target_actions, action_mask
        ),
        "selector_z_action_mse": _masked_mse(
            controls.selector_z, target_actions, action_mask
        ),
        "oracle_vs_no_z_mean_abs_delta": float(
            (controls.oracle_z - controls.no_z).abs().mean()
        ),
        "selector_vs_oracle_mean_abs_delta": float(
            (controls.selector_z - controls.oracle_z).abs().mean()
        ),
        "selector_nll": float(
            selector_distribution.negative_log_likelihood(oracle_latent)
        ),
        "selector_cosine_alignment": float(
            torch.nn.functional.cosine_similarity(selector_latent, oracle_latent).mean()
        ),
    }
    diagnostics = latent_statistics.to_record()
    diagnostics["pca"] = pca.to_record()
    return diagnostics, offline


def main() -> None:
    args = _parse_args()
    if min(args.manifold_steps, args.selector_steps, args.policy_steps) < 1:
        raise ValueError("all training step counts must be positive")
    if args.batch_size < 2:
        raise ValueError("--batch-size must be at least two")
    config = _load_config(args.config)
    output_dir = args.output_dir or _default_output_dir()
    if output_dir.exists():
        raise FileExistsError(f"Stage 0 output directory already exists: {output_dir}")
    output_dir.mkdir(parents=True)

    collection = _collect_reach(
        config, device=args.device, capture_graph=args.capture_graph
    )
    _write_corpus(output_dir, collection.trajectories)
    success_count = sum(item.is_success for item in collection.trajectories)
    if success_count < max(8, config.num_envs // 2):
        raise RuntimeError(f"Insufficient verified Reach successes: {success_count}")

    window_splits = build_success_window_splits(
        collection.trajectories,
        split_policy=Stage0SplitPolicy(seed=config.seed),
        window_policy=Stage0WindowPolicy(
            action_horizon=config.action_horizon,
            future_horizon=config.future_horizon,
            stride=config.window_stride,
        ),
    )
    device = torch.device(args.device)
    encoder, selector, policy, optimizers, train_metrics = _train_stage0(
        config,
        window_splits,
        device=device,
        manifold_steps=args.manifold_steps,
        selector_steps=args.selector_steps,
        policy_steps=args.policy_steps,
        batch_size=args.batch_size,
    )
    held_out = window_splits.validation or window_splits.test
    latent_diagnostics, offline_controls = _evaluate_offline(
        encoder,
        selector,
        policy,
        held_out,
        device=device,
        seed=config.seed,
    )
    split_sha256 = canonical_fingerprint(
        {
            "split_policy_sha256": window_splits.split_policy_sha256,
            "window_policy_sha256": window_splits.window_policy_sha256,
        }
    )
    checkpoint_hashes = Stage0CheckpointHashes(
        config_sha256=config.fingerprint,
        state_schema_sha256=DEFAULT_STAGE0_STATE_SCHEMA.sha256,
        dataset_sha256=window_splits.corpus_sha256,
        split_sha256=split_sha256,
    )
    checkpoint_manager = Stage0CheckpointManager(output_dir)
    component_steps = {
        "global": args.manifold_steps + args.selector_steps + args.policy_steps,
        "future_encoder": args.manifold_steps,
        "selector": args.selector_steps,
        "policy": args.policy_steps,
    }
    models = {
        "future_encoder": encoder,
        "selector": selector,
        "policy": policy,
    }
    checkpoint = checkpoint_manager.save(
        sequence=component_steps["global"],
        phase="smoke_complete",
        steps=component_steps,
        hashes=checkpoint_hashes,
        models=models,
        optimizers=optimizers,
        rank_state={"rank": 0, "sampler_cursor": component_steps["global"]},
        extra={
            "latent_collapse_gate_passed": latent_diagnostics["collapse_gate_passed"],
            "offline_conditioning_authority": offline_controls["authority"],
        },
        cuda_device=device,
    )
    resumed = checkpoint_manager.load(
        hashes=checkpoint_hashes,
        models=models,
        optimizers=optimizers,
        checkpoint=checkpoint,
        restore_rng=False,
        cuda_device=device,
    )
    (output_dir / "latent-diagnostics.json").write_text(
        json.dumps(latent_diagnostics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report = {
        "config_sha256": config.fingerprint,
        "corpus_sha256": trajectory_corpus_sha256(collection.trajectories),
        "state_schema_sha256": DEFAULT_STAGE0_STATE_SCHEMA.sha256,
        "success_count": success_count,
        "failure_count": sum(
            item.outcome is Stage0Outcome.FAILURE for item in collection.trajectories
        ),
        "timeout_count": sum(
            item.outcome is Stage0Outcome.TIMEOUT for item in collection.trajectories
        ),
        "world_steps": collection.world_steps,
        "world_steps_per_second": collection.world_steps / collection.elapsed_seconds,
        "train_windows": len(window_splits.train),
        "validation_windows": len(window_splits.validation),
        "test_windows": len(window_splits.test),
        "train_metrics": train_metrics,
        "latent_summary": {
            "collapse_gate_passed": latent_diagnostics["collapse_gate_passed"],
            "collapse_reasons": latent_diagnostics["collapse_reasons"],
            "effective_rank": latent_diagnostics["effective_rank"],
            "active_dimension_fraction": latent_diagnostics[
                "active_dimension_fraction"
            ],
        },
        "offline_conditioning_controls": offline_controls,
        "checkpoint": str(checkpoint),
        "checkpoint_resume_phase": resumed.phase,
        "checkpoint_resume_steps": resumed.steps,
        "output_dir": str(output_dir),
    }
    (output_dir / "stage0-report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
