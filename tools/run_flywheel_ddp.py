#!/usr/bin/env python3
"""Run same-state Reach branches, globally weighted DiT updates, and gates."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import random
import subprocess
import sys
import time
import uuid
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np

from rexpolicy.envs import GrootNewtonEnv, GrootNewtonEnvConfig
from rexpolicy.flywheel.archive_replay import admit_candidate_replay
from rexpolicy.flywheel.checkpoint import (
    CheckpointManager,
    atomic_write_json,
    capture_rng_state,
    file_sha256,
    restore_rng_state,
)
from rexpolicy.flywheel.distributed import DistributedContext
from rexpolicy.flywheel.derived_views import write_derived_view
from rexpolicy.flywheel.evaluation import (
    TRAIN_RESET_SEED_DOMAIN,
    HeldOutEpisodeResult,
    HeldOutMetrics,
    domain_seed,
    evaluate_non_regression,
    fixed_evaluation_suite,
)
from rexpolicy.flywheel.experience import (
    BranchOutcomeAccumulator,
    DIRECT_SUCCESS_ROLE,
    ChunkCandidate,
    EpisodeExperience,
    TrainingSample,
    format_sample_id,
    select_advantage_chunks,
    validate_candidate_archive_record,
)
from rexpolicy.flywheel.event_ledger import (
    EpisodeEventLedgerBuilder,
    production_ledger_bindings,
)
from rexpolicy.flywheel.groot_policy import (
    GrootFlowDitPolicy,
    RawPolicyObservation,
)
from rexpolicy.flywheel.operations import (
    GpuPreflightError,
    RunOperations,
    preflight_selected_gpus,
)
from rexpolicy.flywheel.replay import (
    CandidateReplayResult,
    build_physical_replay_gate_record,
    fingerprint_each_world,
    validate_historical_physical_replay_gate,
    validate_physical_replay_gate,
    validate_replay_fingerprint,
)
from rexpolicy.flywheel.paired_archive import (
    CollectedEpisode,
    PairedArchiveWrite,
    write_generation_archive_pair,
)
from rexpolicy.flywheel.success_archive import (
    SuccessArchive,
    SuccessReference,
    load_episode_record,
)
from rexpolicy.flywheel.task_spec import (
    REACH_GREEN_CAP_V1,
    FlywheelTaskSpec,
    get_task_spec,
)
from rexpolicy.flywheel.task_runtime_adapter import (
    RuntimeTaskBinding,
    load_production_reach_runtime_binding,
)
from rexpolicy.flywheel.task_shadow import (
    begin_shadow_decision,
    compare_groot_step_parity,
    evaluate_shadow_transition,
    extract_groot_reach_metric_rows,
    require_identical_root_rows,
)
from rexpolicy.flywheel.trainer import DitDdpTrainer, UpdateMetrics
from rexpolicy.manifold.runtime import SuccessManifoldRuntime
from rexpolicy.replay.success_graph import (
    FutureWindowPolicy,
    SuccessExperienceGraph,
    compile_success_experience_graph,
)
from rexpolicy.tasking.canonical import canonical_fingerprint
from rexpolicy.tasking.event_ledger import (
    CollectionProvenance,
    EpisodeEventLedger,
    ResetRecipe,
)
from rexpolicy.tasking.repository import TaskArtifacts
from rexpolicy.tasking.runtime import initial_runtime_state

REPO_ROOT = Path(__file__).resolve().parents[1]
REACH_BOTTLE_DISPLACEMENT_LIMIT_M = 0.010


def _first_existing(*paths: Path) -> Path:
    for path in paths:
        if path.exists():
            return path
    return paths[0]


DEFAULT_GROOT_ROOT = _first_existing(
    REPO_ROOT / "external" / "Isaac-GR00T",
    REPO_ROOT.parent / "Isaac-GR00T",
)
DEFAULT_CHECKPOINT = REPO_ROOT / "checkpoints" / "groot" / "checkpoint-200000"
DEFAULT_VLM = REPO_ROOT / "checkpoints" / "nvidia" / "Cosmos-Reason2-2B"


def _environment_path(name: str, fallback: Path) -> Path:
    value = os.environ.get(name)
    return Path(value).expanduser() if value else fallback


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--isaac-groot-root",
        type=Path,
        default=_environment_path("ISAAC_GROOT_ROOT", DEFAULT_GROOT_ROOT),
    )
    parser.add_argument(
        "--policy-checkpoint",
        type=Path,
        default=_environment_path("GROOT_POLICY_CHECKPOINT", DEFAULT_CHECKPOINT),
    )
    parser.add_argument(
        "--vlm-model",
        type=Path,
        default=_environment_path("GROOT_VLM_MODEL", DEFAULT_VLM),
    )
    parser.add_argument("--dit-overlay", type=Path)
    parser.add_argument("--success-manifold-bundle", type=Path)
    parser.add_argument("--expected-success-manifold-bundle-sha256")
    parser.add_argument(
        "--manifold-memory-candidates",
        type=int,
        default=0,
        help=(
            "Maximum same-state worlds conditioned from discovered latent "
            "memory; remaining worlds sample the state-conditioned selector."
        ),
    )
    parser.add_argument("--run-dir", "--output-dir", dest="run_dir", type=Path)
    parser.add_argument("--task-id", default=REACH_GREEN_CAP_V1.task_id)
    parser.add_argument("--instruction")
    parser.add_argument(
        "--max-generations",
        "--generations",
        dest="max_generations",
        type=int,
        default=1,
        help="Total 1-based generation target, including resumed generations.",
    )
    parser.add_argument("--episodes-per-generation", type=int, default=1)
    parser.add_argument(
        "--candidates-per-state",
        "--num-envs-per-rank",
        dest="candidates_per_state",
        type=int,
        default=2,
    )
    parser.add_argument("--episode-control-steps", type=int, default=8)
    parser.add_argument("--execution-horizon", type=int, default=2)
    parser.add_argument(
        "--chunk-elite-fraction",
        "--elite-fraction",
        dest="chunk_elite_fraction",
        type=float,
        default=0.5,
    )
    parser.add_argument("--advantage-temperature", type=float, default=0.1)
    parser.add_argument("--same-state-tolerance", type=float, default=1.0e-5)
    parser.add_argument(
        "--historical-replay-state-tolerance",
        type=float,
        default=1.0e-4,
        help=(
            "Absolute tolerance only for archived-vs-current continuous "
            "physical-state witnesses."
        ),
    )
    parser.add_argument(
        "--candidate-replay-reward-tolerance",
        type=float,
        default=7.0e-4,
        help="Absolute reward tolerance, separate from state/action replay.",
    )
    parser.add_argument("--train-steps-per-generation", type=int, default=4)
    parser.add_argument("--train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=3.0e-6)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--gradient-clip-norm", type=float, default=1.0)
    parser.add_argument("--parameter-probe-size", type=int, default=65_536)
    parser.add_argument("--parameter-sync-tolerance", type=float, default=0.0)
    parser.add_argument("--success-replay-per-rank", type=int, default=4)
    parser.add_argument(
        "--success-experience-graph",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Publish rebuildable future-window graphs for successful episodes.",
    )
    parser.add_argument("--success-future-horizon", type=int, default=8)
    parser.add_argument("--success-action-horizon", type=int, default=2)
    parser.add_argument("--success-window-stride", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260722)

    parser.add_argument(
        "--checkpoint-every",
        "--save-every",
        dest="checkpoint_every",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--save",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write resumable checkpoints; enabled for every managed run.",
    )
    resume = parser.add_mutually_exclusive_group()
    resume.add_argument("--resume-latest", action="store_true")
    resume.add_argument("--resume-from", type=Path)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument(
        "--eval-at-start",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--eval-at-end",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--eval-reset-recipes", type=int, default=8)
    parser.add_argument("--eval-diffusion-seeds", type=int, default=4)
    parser.add_argument(
        "--optimizer-state-offload",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--offload-vlm-during-update",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    parser.add_argument(
        "--camera-textures", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--scene-visuals", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--capture-graph", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--hydroelastic", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--bottle-settle-frames", type=int, default=60)
    parser.add_argument("--substeps-per-frame", type=int, default=16)

    parser.add_argument("--heartbeat-seconds", type=float, default=30.0)
    parser.add_argument("--gpu-monitor-seconds", type=float, default=5.0)
    parser.add_argument("--minimum-free-gpu-mib", type=int, default=2048)
    parser.add_argument(
        "--strict-gpu-preflight",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--validate-only", action="store_true")
    return parser


def _validate_args(
    args: argparse.Namespace,
    task: FlywheelTaskSpec,
) -> None:
    if args.max_generations < 1:
        raise ValueError("max-generations must be positive")
    if args.episodes_per_generation < 1:
        raise ValueError("episodes-per-generation must be positive")
    if args.candidates_per_state < 2:
        raise ValueError(
            "candidates-per-state must be at least 2 for comparative training"
        )
    if min(args.episode_control_steps, args.execution_horizon) < 1:
        raise ValueError("control steps and execution horizon must be positive")
    if not 0.0 < args.chunk_elite_fraction <= 1.0:
        raise ValueError("chunk-elite-fraction must be in (0, 1]")
    if args.advantage_temperature <= 0.0:
        raise ValueError("advantage-temperature must be positive")
    if (
        not np.isfinite(args.same_state_tolerance)
        or args.same_state_tolerance <= 0.0
    ):
        raise ValueError("same-state-tolerance must be positive")
    if (
        not np.isfinite(args.historical_replay_state_tolerance)
        or args.historical_replay_state_tolerance <= 0.0
    ):
        raise ValueError(
            "historical-replay-state-tolerance must be positive"
        )
    if args.historical_replay_state_tolerance < args.same_state_tolerance:
        raise ValueError(
            "historical-replay-state-tolerance must be greater than or "
            "equal to same-state-tolerance"
        )
    if (
        not np.isfinite(args.candidate_replay_reward_tolerance)
        or args.candidate_replay_reward_tolerance <= 0.0
    ):
        raise ValueError("candidate-replay-reward-tolerance must be positive")
    if args.train_steps_per_generation < 0:
        raise ValueError("train-steps-per-generation cannot be negative")
    if min(args.train_batch_size, args.gradient_accumulation) < 1:
        raise ValueError("training batch and accumulation must be positive")
    if args.learning_rate <= 0.0 or args.weight_decay < 0.0:
        raise ValueError("invalid optimizer settings")
    if args.gradient_clip_norm <= 0.0:
        raise ValueError("gradient-clip-norm must be positive")
    if args.parameter_probe_size < 1:
        raise ValueError("parameter-probe-size must be positive")
    if args.parameter_sync_tolerance < 0.0:
        raise ValueError("parameter-sync-tolerance cannot be negative")
    if args.success_replay_per_rank < 0:
        raise ValueError("success-replay-per-rank cannot be negative")
    if min(
        args.success_future_horizon,
        args.success_action_horizon,
        args.success_window_stride,
    ) < 1:
        raise ValueError("Success graph horizons and stride must be positive")
    if args.success_action_horizon > args.success_future_horizon:
        raise ValueError(
            "success-action-horizon cannot exceed success-future-horizon"
        )
    has_manifold = args.success_manifold_bundle is not None
    has_manifold_hash = (
        args.expected_success_manifold_bundle_sha256 is not None
    )
    if has_manifold != has_manifold_hash:
        raise ValueError(
            "--success-manifold-bundle and its expected SHA-256 are required together"
        )
    if has_manifold_hash and (
        len(args.expected_success_manifold_bundle_sha256) != 64
        or any(
            character not in "0123456789abcdef"
            for character in args.expected_success_manifold_bundle_sha256
        )
    ):
        raise ValueError(
            "expected-success-manifold-bundle-sha256 must be lowercase SHA-256"
        )
    if not 0 <= args.manifold_memory_candidates <= args.candidates_per_state:
        raise ValueError(
            "manifold-memory-candidates must be between zero and candidates-per-state"
        )
    if not has_manifold and args.manifold_memory_candidates:
        raise ValueError(
            "manifold-memory-candidates requires a success-manifold bundle"
        )
    if args.checkpoint_every < 1:
        raise ValueError("checkpoint-every must be positive")
    if args.eval_every < 0:
        raise ValueError("eval-every cannot be negative")
    if min(args.eval_reset_recipes, args.eval_diffusion_seeds) < 1:
        raise ValueError("held-out suite dimensions must be positive")
    if min(args.heartbeat_seconds, args.gpu_monitor_seconds) <= 0.0:
        raise ValueError("monitor intervals must be positive")
    if args.minimum_free_gpu_mib < 0:
        raise ValueError("minimum-free-gpu-mib cannot be negative")
    if not args.validate_only and args.run_dir is None:
        raise ValueError("--run-dir is required for a managed flywheel run")
    if args.substeps_per_frame != 16 or args.bottle_settle_frames != 60:
        raise ValueError(
            "reach_green_cap/v1 uses the validated production physics "
            "configuration: --substeps-per-frame 16 and "
            "--bottle-settle-frames 60"
        )
    if args.instruction is not None and args.instruction != task.instruction:
        raise ValueError(
            "The canonical Reach soak does not permit an instruction override; "
            f"expected {task.instruction!r}"
        )


def _validate_artifacts(args: argparse.Namespace) -> None:
    GrootFlowDitPolicy._validate_layout(
        args.isaac_groot_root.expanduser().resolve(),
        args.policy_checkpoint.expanduser().resolve(),
        args.vlm_model.expanduser().resolve(),
    )
    if args.success_manifold_bundle is not None:
        bundle = args.success_manifold_bundle.expanduser().resolve()
        if not bundle.is_file():
            raise FileNotFoundError(
                f"Success-manifold bundle is missing: {bundle}"
            )
        actual = file_sha256(bundle)
        if actual != args.expected_success_manifold_bundle_sha256:
            raise ValueError(
                "Success-manifold bundle checksum mismatch: "
                f"expected={args.expected_success_manifold_bundle_sha256}, "
                f"actual={actual}"
            )


def _raw_observations(
    observation: dict[str, Any],
    *,
    instruction: str,
) -> list[RawPolicyObservation]:
    eef = observation["extra"]["eef_9d"].detach().cpu().numpy()
    hand = observation["agent"]["hand_joint_pos"].detach().cpu().numpy()
    arm = observation["agent"]["arm_joint_pos"].detach().cpu().numpy()
    ego = observation["sensor_data"]["ego_view"]["rgb"].detach().cpu().numpy()
    wrist = observation["sensor_data"]["wrist_view"]["rgb"].detach().cpu().numpy()
    return [
        RawPolicyObservation(
            images={
                "ego_view": np.ascontiguousarray(ego[index][None]),
                "wrist_view": np.ascontiguousarray(wrist[index][None]),
            },
            state={
                "eef_9d": np.ascontiguousarray(
                    eef[index][None], dtype=np.float32
                ),
                "hand_joint_pos": np.ascontiguousarray(
                    hand[index][None], dtype=np.float32
                ),
                "arm_joint_pos": np.ascontiguousarray(
                    arm[index][None], dtype=np.float32
                ),
            },
            instruction=instruction,
        )
        for index in range(eef.shape[0])
    ]


def _copy_initial_state(observation: RawPolicyObservation) -> dict[str, np.ndarray]:
    return {
        key: np.asarray(value[-1], dtype=np.float32).copy()
        for key, value in observation.state.items()
    }


def _create_environment(
    args: argparse.Namespace,
    context: DistributedContext,
    task: FlywheelTaskSpec,
    *,
    num_envs: int | None = None,
) -> GrootNewtonEnv:
    env = GrootNewtonEnv(
        GrootNewtonEnvConfig(
            num_envs=args.candidates_per_state if num_envs is None else num_envs,
            device=str(context.device),
            max_episode_steps=args.episode_control_steps,
            obs_mode="state_dict+rgb",
            control_mode="pd_eef_pose_abs",
            reward_mode="normalized_dense",
            task_mode=task.environment_task_mode,
            substeps_per_frame=args.substeps_per_frame,
            bottle_settle_frames=args.bottle_settle_frames,
            terminate_on_success=True,
            terminate_on_fail=True,
            capture_graph=args.capture_graph,
            render_images=True,
            camera_textures=args.camera_textures,
            load_scene_visuals=args.scene_visuals,
            hydroelastic_contacts=args.hydroelastic,
        )
    )
    try:
        metadata = env.task_metadata
        expected_metadata = {
            "task_id": task.task_id,
            "instruction": task.instruction,
            "goal_offset_world_m": (0.180, 0.080, 0.050),
            "success_distance_m": 0.040,
            "success_hold_control_steps": 2,
            "bottle_displacement_limit_m": 0.010,
            "reward_distance_scale_m": 0.030,
            "reset_xy_jitter_m": 0.010,
        }
        for key, expected in expected_metadata.items():
            if metadata.get(key) != expected:
                raise RuntimeError(
                    f"Environment task contract mismatch for {key}: "
                    f"expected={expected!r}, actual={metadata.get(key)!r}"
                )
        expected_mask = (
            *task.action_dimension_masks["eef_9d"],
            *task.action_dimension_masks["hand_joint_target"],
        )
        actual_mask = tuple(
            bool(value)
            for value in env.effective_action_mask_torch()
            .detach()
            .cpu()
            .tolist()
        )
        if actual_mask != expected_mask:
            raise RuntimeError(
                "Environment effective-action mask does not match the "
                f"training task: expected={expected_mask}, actual={actual_mask}"
            )
    except Exception:
        env.close()
        raise
    return env


def _validate_run_directory_mode(
    *,
    run_dir: Path,
    context: DistributedContext,
    resume: bool,
    resume_from: Path | None,
) -> None:
    """Reject accidental reuse before any flywheel-owned file is created."""
    error = None
    if context.is_main:
        try:
            if resume_from is not None:
                checkpoint = resume_from.expanduser().resolve()
                expected_parent = (run_dir / "checkpoints").resolve()
                if checkpoint.parent != expected_parent:
                    raise ValueError(
                        "--resume-from must select a checkpoint inside "
                        f"{expected_parent}"
                    )
            if resume:
                if not run_dir.is_dir():
                    raise FileNotFoundError(
                        f"Resume run directory does not exist: {run_dir}"
                    )
            elif run_dir.exists():
                owned_names = {
                    "archive",
                    "checkpoints",
                    "heartbeats",
                    "launches",
                    "logs",
                    "metrics.jsonl",
                    "pids",
                    "run_manifest.json",
                }
                collisions = sorted(
                    path.name
                    for path in run_dir.iterdir()
                    if path.name in owned_names
                    or path.name.startswith("exit-rank-")
                )
                if collisions:
                    raise FileExistsError(
                        "Run directory already contains flywheel artifacts; "
                        "use --resume-latest or choose a new directory: "
                        + ", ".join(collisions)
                    )
        except Exception as caught:
            error = f"{type(caught).__name__}: {caught}"
    error = context.broadcast_object(error, source=0)
    if error is not None:
        raise RuntimeError("Run-directory validation failed: " + error)


def _reset_and_replay(
    *,
    env: GrootNewtonEnv,
    context: DistributedContext,
    reset_seed: int,
    root_actions: list[np.ndarray],
) -> tuple[dict[str, Any], bool]:
    """Rebuild one root state from reset seed and effective action history."""
    import torch

    observation, _ = env.reset(seed=reset_seed)
    for root_action in root_actions:
        tiled = np.repeat(root_action[None], env.num_envs, axis=0)
        proposed = torch.as_tensor(
            tiled,
            dtype=torch.float32,
            device=context.device,
        )
        effective = env.project_effective_action_torch(proposed)
        observation, _, terminated, truncated, _ = env.step(effective)
        if bool((terminated | truncated).any().item()):
            return observation, True
        env.canonicalize_branch_state_torch(source_world=0)
        observation = env.observation_torch()
    if not root_actions:
        env.canonicalize_branch_state_torch(source_world=0)
    return env.observation_torch(), False


def _normalized_action_diversity(
    candidates: list[ChunkCandidate],
) -> tuple[float, float]:
    """Return pairwise RMS over only dimensions used by the training loss."""
    pairwise_rms = []
    for left_index, left in enumerate(candidates):
        left_action = left.sample.action.detach().float().cpu().numpy()
        left_mask = left.sample.action_mask.detach().bool().cpu().numpy()
        for right in candidates[left_index + 1 :]:
            right_action = right.sample.action.detach().float().cpu().numpy()
            right_mask = right.sample.action_mask.detach().bool().cpu().numpy()
            common_mask = left_mask & right_mask
            if not np.any(common_mask):
                continue
            difference = left_action[common_mask] - right_action[common_mask]
            pairwise_rms.append(float(np.sqrt(np.mean(np.square(difference)))))
    if not pairwise_rms:
        return 0.0, 0.0
    return float(np.mean(pairwise_rms)), max(pairwise_rms)


def _sample_policy_batch(
    *,
    policy: GrootFlowDitPolicy,
    observations: list[RawPolicyObservation],
    manifold: SuccessManifoldRuntime | None,
    memory_candidates: int,
):
    """Keep v1 sampling exact or add selector/memory latents in v2."""
    if manifold is None:
        return policy.sample(observations)
    conditions = policy.encode_conditions(observations)
    conditioned = manifold.condition(
        policy=policy,
        conditions=conditions,
        memory_candidates=memory_candidates,
    )
    sampled = policy.sample_from_conditions(
        conditions=list(conditioned.conditions),
        observations=observations,
    )
    return replace(
        sampled,
        success_latents=conditioned.latents,
        latent_sources=conditioned.sources,
    )


def _collect_episode(
    *,
    args: argparse.Namespace,
    context: DistributedContext,
    policy: GrootFlowDitPolicy,
    env: GrootNewtonEnv,
    task: FlywheelTaskSpec,
    generation: int,
    episode_index: int,
    simulator_fingerprint: str,
    task_artifacts: TaskArtifacts,
    runtime_binding: RuntimeTaskBinding,
    observation_contract_sha256: str,
    manifold: SuccessManifoldRuntime | None,
) -> CollectedEpisode:
    """Build one root path while retaining all selected branch chunks."""
    import torch

    reset_seed = domain_seed(
        args.seed,
        TRAIN_RESET_SEED_DOMAIN,
        generation,
        context.rank,
        episode_index,
    )
    observation, replay_ended = _reset_and_replay(
        env=env,
        context=context,
        reset_seed=reset_seed,
        root_actions=[],
    )
    if replay_ended:
        raise RuntimeError("Freshly reset Reach episode terminated unexpectedly")
    initial_raw = _raw_observations(
        observation,
        instruction=task.instruction,
    )
    initial_state = _copy_initial_state(initial_raw[0])
    root_actions: list[np.ndarray] = []
    root_rewards: list[float] = []
    selected_actions: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    selected_samples: list[TrainingSample] = []
    path_samples: list[TrainingSample] = []
    selected_return = 0.0
    selected_success = False
    episode_done = False
    decision_index = 0
    event_schema = task_artifacts.capabilities.event_schemas[0]
    task_state = initial_runtime_state(
        task_artifacts.contract,
        reward_profile_id=runtime_binding.reward_profile_id,
        catalog=task_artifacts.capabilities,
    )
    ledger_builder = EpisodeEventLedgerBuilder(
        bindings=production_ledger_bindings(
            task_contract_id=task_artifacts.contract.task_contract_id,
            task_contract_sha256=task_artifacts.contract.fingerprint,
            task_oracle_sha256=task_artifacts.contract.oracle_fingerprint,
            observation_contract_sha256=observation_contract_sha256,
            event_schema=event_schema,
        ),
        collection=CollectionProvenance.from_record(
            {
                "data_generation": generation,
                "sampling_policy_generation": generation - 1,
                "rank": context.rank,
                "episode": episode_index,
                "instruction_variant_id": "reach_green_cap/train_00000/v2",
            }
        ),
        reset_recipe=ResetRecipe.from_record(
            {
                "environment_adapter_id": task_artifacts.contract.adapter_id,
                "seed": reset_seed,
            }
        ),
        event_schema=event_schema,
    )

    while len(root_actions) < args.episode_control_steps and not episode_done:
        if decision_index > 0:
            observation, replay_ended = _reset_and_replay(
                env=env,
                context=context,
                reset_seed=reset_seed,
                root_actions=root_actions,
            )
            if replay_ended:
                raise RuntimeError(
                    "Root replay terminated before a previously reached decision"
                )
        physical_replay_gate = validate_physical_replay_gate(
            env.physical_replay_gate_torch(),
            world_count=env.num_envs,
            float_tolerance=args.same_state_tolerance,
        )
        physical_replay_gate_record = build_physical_replay_gate_record(
            physical_replay_gate,
            same_state_tolerance=args.same_state_tolerance,
            historical_replay_state_tolerance=(
                args.historical_replay_state_tolerance
            ),
            simulator_fingerprint_sha256=simulator_fingerprint,
            observation_contract_sha256=observation_contract_sha256,
            render_contract=env.observation_render_contract(),
        )
        dynamics_fingerprint = validate_replay_fingerprint(
            env.dynamics_fingerprint_torch(),
            world_count=env.num_envs,
            float_tolerance=args.same_state_tolerance,
        )
        root_evaluation = env.evaluate()
        root_metric_rows = extract_groot_reach_metric_rows(
            root_evaluation,
            event_schema=event_schema,
        )
        root_metrics = require_identical_root_rows(root_metric_rows)
        shadow_branch = begin_shadow_decision(
            root_state=task_state,
            root_source=root_evaluation,
            event_schema=event_schema,
        )
        ledger_builder.open_decision(
            decision_index=decision_index,
            root_control_step=len(root_actions),
            dynamics_digest_sha256=dynamics_fingerprint.digest,
            current_metrics=root_metrics,
        )
        raw = _raw_observations(observation, instruction=task.instruction)
        sampled = _sample_policy_batch(
            policy=policy,
            observations=raw,
            manifold=manifold,
            memory_candidates=args.manifold_memory_candidates,
        )
        decoded = sampled.decoded_action
        horizon = min(
            args.execution_horizon,
            args.episode_control_steps - len(root_actions),
            *(int(decoded[key].shape[1]) for key in policy.action_keys),
        )
        active = torch.ones(env.num_envs, dtype=torch.bool, device=context.device)
        terminated_any = torch.zeros_like(active)
        truncated_any = torch.zeros_like(active)
        chunk_scores = torch.zeros(
            env.num_envs,
            dtype=torch.float32,
            device=context.device,
        )
        valid_steps = [0 for _ in range(env.num_envs)]
        chunk_rewards: list[list[float]] = [[] for _ in range(env.num_envs)]
        executed_steps: list[list[np.ndarray]] = [
            [] for _ in range(env.num_envs)
        ]
        branch_outcomes = [
            BranchOutcomeAccumulator() for _ in range(env.num_envs)
        ]

        for chunk_step in range(horizon):
            active_before = active.clone()
            candidate_np = np.concatenate(
                (
                    decoded["eef_9d"][:, chunk_step],
                    decoded["hand_joint_target"][:, chunk_step],
                ),
                axis=-1,
            ).astype(np.float32)
            if not np.isfinite(candidate_np).all():
                raise FloatingPointError("DiT sampled a non-finite action")
            proposed = torch.as_tensor(
                candidate_np,
                dtype=torch.float32,
                device=context.device,
            )
            hold = env.hold_action_torch().clone()
            mixed = torch.where(active_before[:, None], proposed, hold)
            effective = env.project_effective_action_torch(mixed)
            observation, reward, terminated, truncated, info = env.step(effective)
            executed = (
                env.effective_action_torch().detach().float().cpu().numpy().copy()
            )
            reward_values = reward.detach().float().cpu().tolist()
            active_cpu = active_before.detach().cpu().tolist()
            failure_step = info["fail"].detach().cpu().tolist()
            contact_step = (
                info["had_hand_contact_this_control_step"]
                .detach()
                .cpu()
                .tolist()
            )
            displacement_step = (
                info["reach_max_bottle_displacement"]
                .gt(REACH_BOTTLE_DISPLACEMENT_LIMIT_M)
                .detach()
                .cpu()
                .tolist()
            )
            success_step = info["success"].detach().cpu().tolist()
            terminated_step = terminated.detach().cpu().tolist()
            truncated_step = truncated.detach().cpu().tolist()
            post_dynamics_digests = fingerprint_each_world(
                env.dynamics_fingerprint_torch(),
                world_count=env.num_envs,
                float_tolerance=args.same_state_tolerance,
            )
            shadow_batch = evaluate_shadow_transition(
                shadow_branch,
                transition_source=info,
                active_before=tuple(bool(value) for value in active_cpu),
                contract=task_artifacts.contract,
                catalog=task_artifacts.capabilities,
            )
            shadow_branch = shadow_batch.next_branch
            chunk_scores.add_(reward * active_before)
            for world, was_active in enumerate(active_cpu):
                branch_outcomes[world].observe(
                    executed=bool(was_active),
                    failure=bool(failure_step[world]),
                    contact_violation=bool(contact_step[world]),
                    displacement_violation=bool(displacement_step[world]),
                )
                if was_active:
                    shadow_result = shadow_batch.results[world]
                    if shadow_result is None:
                        raise RuntimeError("Active world has no TaskSpec shadow result")
                    parity = compare_groot_step_parity(
                        shadow_result,
                        environment_reward=float(reward_values[world]),
                        environment_success=bool(success_step[world]),
                        environment_failure=bool(failure_step[world]),
                        environment_terminated=bool(terminated_step[world]),
                        environment_truncated=bool(truncated_step[world]),
                    )
                    if not parity.accepted:
                        raise RuntimeError(
                            "TaskSpec v2 shadow mismatch for world "
                            f"{world}: {', '.join(parity.mismatches)}"
                        )
                    ledger_builder.append_transition(
                        decision_index=decision_index,
                        world=world,
                        branch_step=valid_steps[world],
                        effective_action_19d=executed[world],
                        post_dynamics_digest_sha256=(
                            post_dynamics_digests[world]
                        ),
                        current_metrics=shadow_batch.current_rows[world],
                        terminated=bool(terminated_step[world]),
                        truncated=bool(truncated_step[world]),
                    )
                    valid_steps[world] += 1
                    chunk_rewards[world].append(float(reward_values[world]))
                    executed_steps[world].append(executed[world])
            terminated_any.logical_or_(terminated)
            truncated_any.logical_or_(truncated)
            active.logical_and_(~(terminated | truncated))

        evaluation = env.evaluate()
        score_values = chunk_scores.detach().float().cpu().tolist()
        success_values = evaluation["success"].detach().cpu().tolist()
        final_failure_values = evaluation["fail"].detach().cpu().tolist()
        final_contact_values = (
            evaluation["reach_contact_violation"].detach().cpu().tolist()
        )
        final_displacement_values = (
            evaluation["reach_displacement_violation"].detach().cpu().tolist()
        )
        for world, outcome in enumerate(branch_outcomes):
            outcome.observe(
                executed=valid_steps[world] > 0,
                failure=bool(final_failure_values[world]),
                contact_violation=bool(final_contact_values[world]),
                displacement_violation=bool(final_displacement_values[world]),
            )
        terminated_values = terminated_any.detach().cpu().tolist()
        truncated_values = truncated_any.detach().cpu().tolist()
        candidates = []
        for world, count in enumerate(valid_steps):
            if count == 0:
                continue
            effective_action = np.stack(executed_steps[world], axis=0).astype(
                np.float32
            )
            executed_action = {
                "eef_9d": np.ascontiguousarray(effective_action[:, :9]),
                "hand_joint_target": np.ascontiguousarray(
                    effective_action[:, 9:19]
                ),
                "arm_joint_target": np.repeat(
                    raw[world].state["arm_joint_pos"][-1][None],
                    count,
                    axis=0,
                ).astype(np.float32),
            }
            training_sample = policy.make_training_sample(
                condition=sampled.conditions[world],
                raw_state=raw[world].state,
                executed_action=executed_action,
                action_dimension_masks=task.action_dimension_masks,
                sample_metadata={
                    "generation": generation,
                    "rank": context.rank,
                    "episode": episode_index,
                    "decision": decision_index,
                    "world": world,
                    "task_id": task.task_id,
                    "reward_profile_id": task.reward_profile_id,
                    "source": "current",
                },
            )
            expected_sample_id = format_sample_id(
                generation,
                context.rank,
                episode_index,
                decision_index,
                world,
            )
            if training_sample.sample_id != expected_sample_id:
                raise RuntimeError(
                    "Training/Event Ledger sample identity mismatch: "
                    f"{training_sample.sample_id!r} != {expected_sample_id!r}"
                )
            if bool(success_values[world]):
                training_sample.add_success_role(DIRECT_SUCCESS_ROLE)
            failure_reasons = []
            outcome = branch_outcomes[world]
            if outcome.contact_violation:
                failure_reasons.append("hand_contact")
            if outcome.displacement_violation:
                failure_reasons.append("bottle_displacement")
            if outcome.failure and not failure_reasons:
                failure_reasons.append("environment_failure")
            candidates.append(
                ChunkCandidate(
                    world=world,
                    score=float(score_values[world]),
                    success=bool(success_values[world]),
                    terminated=bool(terminated_values[world]),
                    truncated=bool(truncated_values[world]),
                    valid_steps=count,
                    rewards=chunk_rewards[world],
                    action={
                        **executed_action,
                        "action_19d": effective_action,
                    },
                    sample=training_sample,
                    failure=outcome.failure,
                    safety_violation=outcome.safety_violation,
                    failure_reasons=tuple(failure_reasons),
                )
            )
        if not candidates:
            raise RuntimeError("No branch executed a valid action step")

        selection = select_advantage_chunks(
            candidates,
            fraction=args.chunk_elite_fraction,
            temperature=args.advantage_temperature,
        )
        diversity_mean, diversity_max = _normalized_action_diversity(candidates)
        selected_samples.extend(candidate.sample for candidate in selection.selected)
        continuation = selection.continuation
        ledger_builder.close_decision(
            decision_index=decision_index,
            root_continuation_sample_id=(
                None if continuation is None else continuation.sample.sample_id
            ),
        )
        if continuation is None:
            decisions.append(
                {
                    "decision_index": decision_index,
                    "start_control_step": len(root_actions),
                    "physical_replay_gate": physical_replay_gate_record,
                    "advantage_baseline": selection.baseline,
                    "selection_mode": selection.mode,
                    "success_constraint_active": False,
                    "safe_candidate_count": 0,
                    "selected_chunk_count": 0,
                    "normalized_action_pairwise_rms_mean": diversity_mean,
                    "normalized_action_pairwise_rms_max": diversity_max,
                    "candidates": [
                        candidate.archive_record() for candidate in candidates
                    ],
                }
            )
            episode_done = True
            decision_index += 1
            continue
        task_state = shadow_branch.world_states[continuation.world]
        path_samples.append(continuation.sample)
        selected_return += continuation.score
        root_rewards.extend(continuation.rewards)
        continuation_actions = np.asarray(
            continuation.action["action_19d"],
            dtype=np.float32,
        )
        root_actions.extend(row.copy() for row in continuation_actions)
        selected_actions.append(
            {
                "decision_index": decision_index,
                "start_control_step": len(root_actions) - continuation.valid_steps,
                "valid_steps": continuation.valid_steps,
                "action_19d": continuation_actions,
                "rewards": continuation.rewards,
            }
        )
        decisions.append(
            {
                "decision_index": decision_index,
                "start_control_step": len(root_actions) - continuation.valid_steps,
                "physical_replay_gate": physical_replay_gate_record,
                "advantage_baseline": selection.baseline,
                "selection_mode": selection.mode,
                "success_constraint_active": any(
                    candidate.success for candidate in candidates
                ),
                "safe_candidate_count": sum(
                    not candidate.failure and not candidate.safety_violation
                    for candidate in candidates
                ),
                "selected_chunk_count": len(selection.selected),
                "normalized_action_pairwise_rms_mean": diversity_mean,
                "normalized_action_pairwise_rms_max": diversity_max,
                "candidates": [candidate.archive_record() for candidate in candidates],
            }
        )
        selected_success = continuation.success
        episode_done = (
            continuation.success
            or continuation.terminated
            or continuation.truncated
            or len(root_actions) >= args.episode_control_steps
        )
        decision_index += 1

    episode = EpisodeExperience(
        generation=generation,
        sampling_policy_generation=generation - 1,
        rank=context.rank,
        episode=episode_index,
        reset_seed=reset_seed,
        instruction=task.instruction,
        task_id=task.task_id,
        reward_profile_id=task.reward_profile_id,
        score=selected_return,
        success=selected_success,
        initial_state=initial_state,
        simulator_fingerprint=simulator_fingerprint,
        decisions=decisions,
        selected_actions=selected_actions,
        rewards=root_rewards,
        samples=selected_samples,
    )
    episode.mark_success_path(path_samples)
    roles_by_id = {
        sample.sample_id: list(sample.success_roles) for sample in episode.samples
    }
    for decision in episode.decisions:
        for candidate in decision["candidates"]:
            candidate["success_roles"] = roles_by_id.get(
                candidate["sample_id"],
                candidate.get("success_roles", []),
            )
    if manifold is not None and episode.success_samples:
        missing_latents = [
            sample.sample_id
            for sample in episode.success_samples
            if sample.success_latent is None
        ]
        if missing_latents:
            raise RuntimeError(
                "Success-conditioned samples lost their latent provenance: "
                + ", ".join(missing_latents)
            )
        manifold.record_successes(
            sample_ids=[sample.sample_id for sample in episode.success_samples],
            latents=torch.stack(
                [sample.success_latent for sample in episode.success_samples]
            ),
            metadata=[
                {
                    "generation": sample.generation,
                    "rank": sample.rank,
                    "episode": sample.episode,
                    "decision": sample.decision,
                    "world": sample.world,
                    "task_id": sample.task_id,
                }
                for sample in episode.success_samples
            ],
        )
    return CollectedEpisode(
        experience=episode,
        event_ledger=ledger_builder.finish(),
    )


def _collect_generation(
    *,
    args: argparse.Namespace,
    context: DistributedContext,
    policy: GrootFlowDitPolicy,
    env: GrootNewtonEnv,
    task: FlywheelTaskSpec,
    generation: int,
    simulator_fingerprint: str,
    task_artifacts: TaskArtifacts,
    runtime_binding: RuntimeTaskBinding,
    observation_contract_sha256: str,
    manifold: SuccessManifoldRuntime | None,
) -> list[CollectedEpisode]:
    return [
        _collect_episode(
            args=args,
            context=context,
            policy=policy,
            env=env,
            task=task,
            generation=generation,
            episode_index=episode_index,
            simulator_fingerprint=simulator_fingerprint,
            task_artifacts=task_artifacts,
            runtime_binding=runtime_binding,
            observation_contract_sha256=observation_contract_sha256,
            manifold=manifold,
        )
        for episode_index in range(args.episodes_per_generation)
    ]


def _write_generation_archive(
    *,
    run_dir: Path,
    attempt_id: str,
    generation: int,
    rank: int,
    episodes: list[EpisodeExperience],
    event_ledgers: list[EpisodeEventLedger],
    window_policy: FutureWindowPolicy | None = None,
) -> tuple[
    PairedArchiveWrite,
    list[SuccessReference],
    list[SuccessExperienceGraph],
]:
    pair = write_generation_archive_pair(
        run_dir=run_dir,
        attempt_id=attempt_id,
        generation=generation,
        rank=rank,
        episodes=episodes,
        event_ledgers=event_ledgers,
    )
    references = []
    references_by_episode: dict[int, list[SuccessReference]] = {}
    for record_index, episode in enumerate(episodes):
        for sample in episode.success_samples:
            reference = SuccessReference.from_sample(
                sample,
                episode_path=pair.episode_path,
                episode_sha256=pair.episode_sha256,
                episode_record_index=record_index,
                simulator_fingerprint=episode.simulator_fingerprint,
            )
            references.append(reference)
            references_by_episode.setdefault(record_index, []).append(reference)

    graphs = []
    if window_policy is not None:
        for record_index, (episode, event_ledger) in enumerate(
            zip(episodes, event_ledgers)
        ):
            episode_references = references_by_episode.get(record_index, [])
            if not episode.success:
                if episode_references:
                    raise RuntimeError(
                        "Unsuccessful episode unexpectedly produced success references"
                    )
                continue
            graph = compile_success_experience_graph(
                episode_record=episode.archive_record(),
                event_ledger=event_ledger,
                success_references=episode_references,
                window_policy=window_policy,
            )
            graphs.append(graph)
    return pair, references, graphs


def _materialize_historical_samples(
    *,
    references: list[SuccessReference],
    run_dir: Path,
    args: argparse.Namespace,
    context: DistributedContext,
    policy: GrootFlowDitPolicy,
    env: GrootNewtonEnv,
    task: FlywheelTaskSpec,
    simulator_fingerprint: str,
    observation_contract_sha256: str,
    archive: SuccessArchive,
    generation: int,
) -> tuple[list[TrainingSample], list[dict[str, Any]]]:
    """Reconstruct historical conditions from reset/action recipes."""
    import torch

    samples = []
    quarantines = []
    for reference in references:
        if reference.task_id != task.task_id:
            raise RuntimeError(
                f"Historical task mismatch for {reference.sample_id}"
            )
        if reference.reward_profile_id != task.reward_profile_id:
            raise RuntimeError(
                f"Historical reward-profile mismatch for {reference.sample_id}"
            )
        if reference.simulator_fingerprint != simulator_fingerprint:
            raise RuntimeError(
                f"Historical simulator fingerprint mismatch for {reference.sample_id}"
            )
        record = load_episode_record(run_dir, reference)
        if int(record.get("schema_version", -1)) != 5:
            raise RuntimeError(
                f"Historical episode schema is not v5 for {reference.sample_id}"
            )
        decision = next(
            (
                item
                for item in record["decisions"]
                if int(item["decision_index"]) == reference.decision
            ),
            None,
        )
        if decision is None:
            raise RuntimeError(
                f"Historical decision is missing for {reference.sample_id}"
            )
        root_actions = []
        for selected in record["selected_path"]["actions"]:
            if int(selected["decision_index"]) >= reference.decision:
                break
            root_actions.extend(
                np.asarray(selected["action_19d"], dtype=np.float32)
            )
        observation, replay_ended = _reset_and_replay(
            env=env,
            context=context,
            reset_seed=int(record["reset_recipe"]["seed"]),
            root_actions=root_actions,
        )
        if replay_ended:
            raise RuntimeError(
                f"Historical replay ended before {reference.sample_id}"
            )
        fingerprint = validate_physical_replay_gate(
            env.physical_replay_gate_torch(),
            world_count=env.num_envs,
            float_tolerance=args.same_state_tolerance,
        )
        validate_historical_physical_replay_gate(
            decision.get("physical_replay_gate"),
            current_fingerprint=fingerprint,
            same_state_tolerance=args.same_state_tolerance,
            historical_replay_state_tolerance=(
                args.historical_replay_state_tolerance
            ),
            simulator_fingerprint_sha256=simulator_fingerprint,
            observation_contract_sha256=observation_contract_sha256,
            render_contract=env.observation_render_contract(),
        )
        candidate = next(
            (
                item
                for item in decision["candidates"]
                if item["sample_id"] == reference.sample_id
            ),
            None,
        )
        if candidate is None:
            raise RuntimeError(
                f"Historical candidate is missing for {reference.sample_id}"
            )
        validate_candidate_archive_record(candidate)
        replay_world = int(reference.world)
        if replay_world < 0 or replay_world >= env.num_envs:
            raise RuntimeError(
                f"Historical world is unavailable for {reference.sample_id}: "
                f"world={replay_world}, envs={env.num_envs}"
            )
        raw = _raw_observations(
            observation,
            instruction=task.instruction,
        )[replay_world]
        condition = policy.encode_conditions([raw])[0]
        archived_action = candidate["action"]
        candidate_action_19d = np.asarray(
            archived_action["action_19d"],
            dtype=np.float32,
        )
        replayed_rewards = []
        replayed_actions = []
        terminated_any = False
        truncated_any = False
        replay_outcome = BranchOutcomeAccumulator()
        for action_row in candidate_action_19d:
            tiled = np.repeat(action_row[None], env.num_envs, axis=0)
            proposed = torch.as_tensor(
                tiled,
                dtype=torch.float32,
                device=context.device,
            )
            effective = env.project_effective_action_torch(proposed)
            _, reward, terminated, truncated, info = env.step(effective)
            replay_outcome.observe(
                executed=True,
                failure=bool(info["fail"][replay_world].item()),
                contact_violation=bool(
                    info["had_hand_contact_this_control_step"][
                        replay_world
                    ].item()
                ),
                displacement_violation=bool(
                    info["reach_max_bottle_displacement"][
                        replay_world
                    ].item()
                    > REACH_BOTTLE_DISPLACEMENT_LIMIT_M
                ),
            )
            replayed_actions.append(
                env.effective_action_torch()[replay_world]
                .detach()
                .float()
                .cpu()
                .numpy()
                .copy()
            )
            replayed_rewards.append(float(reward[replay_world].item()))
            terminated_any = terminated_any or bool(
                terminated[replay_world].item()
            )
            truncated_any = truncated_any or bool(
                truncated[replay_world].item()
            )
            if terminated_any or truncated_any:
                break
        replay_evaluation = env.evaluate()
        replay_outcome.observe(
            executed=True,
            failure=bool(replay_evaluation["fail"][replay_world].item()),
            contact_violation=bool(
                replay_evaluation["reach_contact_violation"][
                    replay_world
                ].item()
            ),
            displacement_violation=bool(
                replay_evaluation["reach_displacement_violation"][
                    replay_world
                ].item()
            ),
        )
        admission = admit_candidate_replay(
            archive=archive,
            reference=reference,
            archived=candidate,
            replayed=CandidateReplayResult(
                rewards=tuple(replayed_rewards),
                success=bool(
                    replay_evaluation["success"][replay_world].item()
                ),
                failure=replay_outcome.failure,
                safety_violation=replay_outcome.safety_violation,
                terminated=terminated_any,
                truncated=truncated_any,
                executed_action=np.asarray(replayed_actions, dtype=np.float32),
            ),
            detected_generation=generation,
            action_tolerance=args.same_state_tolerance,
            reward_tolerance=args.candidate_replay_reward_tolerance,
        )
        if not admission.accepted:
            quarantines.append(
                {
                    "sample_id": reference.sample_id,
                    "source_generation": reference.source_generation,
                    "reason_code": admission.reason_code,
                    "detail": admission.detail,
                    "created": admission.quarantine_created,
                }
            )
            continue
        executed_action = {
            key: np.asarray(archived_action[key], dtype=np.float32)
            for key in ("eef_9d", "hand_joint_target", "arm_joint_target")
        }
        sample = policy.make_training_sample(
            condition=condition,
            raw_state=raw.state,
            executed_action=executed_action,
            action_dimension_masks=task.action_dimension_masks,
            sample_metadata={
                "generation": reference.source_generation,
                "rank": reference.source_rank,
                "episode": reference.source_episode,
                "decision": reference.decision,
                "world": reference.world,
                "task_id": reference.task_id,
                "reward_profile_id": reference.reward_profile_id,
                "source": "historical",
            },
        )
        if sample.sample_id != reference.sample_id:
            raise RuntimeError(
                f"Historical sample identity changed for {reference.sample_id}"
            )
        sample.success_roles = tuple(reference.success_roles)
        sample.sample_weight = float(reference.archived_weight)
        sample.advantage = float(candidate["advantage"])
        samples.append(sample)
    return samples, quarantines


def _evaluate_held_out(
    *,
    args: argparse.Namespace,
    context: DistributedContext,
    policy: GrootFlowDitPolicy,
    task: FlywheelTaskSpec,
    generation: int,
    manifold: SuccessManifoldRuntime | None,
) -> HeldOutMetrics:
    """Run the fixed paired suite with exactly one action sample per state."""
    import torch

    saved_rng = capture_rng_state(cuda_device=context.device)
    suite = fixed_evaluation_suite(
        base_seed=args.seed,
        reset_recipes=args.eval_reset_recipes,
        diffusion_seeds_per_recipe=args.eval_diffusion_seeds,
    )
    local_pairs = suite[context.rank :: context.world_size]
    local_results: list[dict[str, Any]] = []
    env = _create_environment(args, context, task, num_envs=1)
    try:
        for reset_seed, diffusion_seed in local_pairs:
            random.seed(diffusion_seed)
            np.random.seed(diffusion_seed % (2**32))
            torch.manual_seed(diffusion_seed)
            torch.cuda.manual_seed(diffusion_seed)
            observation, _ = env.reset(seed=reset_seed)
            episode_return = 0.0
            invalid = False
            control_steps = 0
            done = False
            while control_steps < args.episode_control_steps and not done:
                raw = _raw_observations(
                    observation,
                    instruction=task.instruction,
                )
                sampled = _sample_policy_batch(
                    policy=policy,
                    observations=raw,
                    manifold=manifold,
                    memory_candidates=0,
                )
                decoded = sampled.decoded_action
                horizon = min(
                    args.execution_horizon,
                    args.episode_control_steps - control_steps,
                    *(int(decoded[key].shape[1]) for key in policy.action_keys),
                )
                for chunk_step in range(horizon):
                    proposed_np = np.concatenate(
                        (
                            decoded["eef_9d"][:, chunk_step],
                            decoded["hand_joint_target"][:, chunk_step],
                        ),
                        axis=-1,
                    ).astype(np.float32)
                    if not np.isfinite(proposed_np).all():
                        invalid = True
                        done = True
                        break
                    proposed = torch.as_tensor(
                        proposed_np,
                        dtype=torch.float32,
                        device=context.device,
                    )
                    effective = env.project_effective_action_torch(proposed)
                    if not bool(torch.isfinite(effective).all()):
                        invalid = True
                        done = True
                        break
                    observation, reward, terminated, truncated, _ = env.step(
                        effective
                    )
                    episode_return += float(reward[0].item())
                    control_steps += 1
                    done = bool((terminated | truncated)[0].item())
                    if done:
                        break
            evaluation = env.evaluate()
            success = bool(evaluation["success"][0].item()) and not invalid
            fail = bool(evaluation["fail"][0].item())
            contact = bool(
                evaluation["reach_contact_violation"][0].item()
            )
            displacement_failure = bool(
                evaluation["reach_displacement_violation"][0].item()
            )
            result = HeldOutEpisodeResult(
                reset_seed=reset_seed,
                diffusion_seed=diffusion_seed,
                success=success,
                safety_failure=fail or contact or displacement_failure,
                invalid_action=invalid,
                final_distance=float(evaluation["reach_distance"][0].item()),
                episode_return=episode_return,
                object_displacement=float(
                    evaluation["reach_max_bottle_displacement"][0].item()
                ),
                control_steps=control_steps,
            )
            local_results.append(asdict(result))
    finally:
        env.close()
        restore_rng_state(saved_rng, cuda_device=context.device)

    gathered = context.all_gather_objects(local_results)
    if context.is_main:
        combined = [
            HeldOutEpisodeResult(**record)
            for rank_records in gathered
            for record in rank_records
        ]
        combined.sort(key=lambda item: (item.reset_seed, item.diffusion_seed))
        metrics_record = HeldOutMetrics.from_episodes(
            generation=generation,
            episodes=combined,
        ).to_record()
    else:
        metrics_record = None
    return HeldOutMetrics.from_record(
        context.broadcast_object(metrics_record, source=0)
    )


def _file_descriptor(path: Path) -> dict[str, Any]:
    path = path.expanduser().resolve()
    return {
        "path": str(path),
        "size": path.stat().st_size,
        "sha256": file_sha256(path),
    }


def _directory_descriptors(path: Path) -> list[dict[str, Any]]:
    """Hash every regular file under one immutable runtime artifact."""
    path = path.expanduser().resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"Artifact directory is missing: {path}")
    descriptors = []
    for candidate in sorted(path.rglob("*")):
        if not candidate.is_file():
            continue
        descriptor = _file_descriptor(candidate)
        descriptor["relative_path"] = str(candidate.relative_to(path))
        descriptors.append(descriptor)
    if not descriptors:
        raise FileNotFoundError(f"Artifact directory is empty: {path}")
    return descriptors


_EPHEMERAL_SOURCE_DIRECTORIES = frozenset(
    {
        ".cache",
        ".git",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        "__pycache__",
    }
)
_EPHEMERAL_SOURCE_SUFFIXES = frozenset({".pyc", ".pyo"})


def _source_tree_descriptors(path: Path) -> list[dict[str, Any]]:
    """Hash a non-Git source copy while excluding generated interpreter state."""
    path = path.expanduser().resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"Source directory is missing: {path}")
    descriptors = []
    for candidate in sorted(path.rglob("*")):
        relative = candidate.relative_to(path)
        if any(
            part in _EPHEMERAL_SOURCE_DIRECTORIES for part in relative.parts
        ):
            continue
        if (
            not candidate.is_file()
            or candidate.suffix in _EPHEMERAL_SOURCE_SUFFIXES
        ):
            continue
        descriptor = _file_descriptor(candidate)
        descriptor["relative_path"] = str(relative)
        descriptors.append(descriptor)
    if not descriptors:
        raise FileNotFoundError(f"Source directory is empty: {path}")
    return descriptors


def _git_source_descriptor(path: Path) -> dict[str, Any]:
    """Fingerprint committed and dirty source without copying the repository."""
    path = path.expanduser().resolve()

    def source_tree_descriptor() -> dict[str, Any]:
        files = _source_tree_descriptors(path)
        digest_payload = [
            (item["relative_path"], item["size"], item["sha256"])
            for item in files
        ]
        return {
            "path": str(path),
            "repository_root": str(path),
            "git_available": False,
            "commit": None,
            "source_tree_sha256": hashlib.sha256(
                json.dumps(
                    digest_payload,
                    separators=(",", ":"),
                ).encode()
            ).hexdigest(),
            "files": files,
            "clean": None,
        }

    def run_at(base: Path, *arguments: str) -> bytes:
        return subprocess.run(
            ["git", "-C", str(base), *arguments],
            check=True,
            capture_output=True,
        ).stdout

    root_probe = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "--show-toplevel"],
        check=False,
        capture_output=True,
    )
    if root_probe.returncode != 0:
        return source_tree_descriptor()

    root = Path(root_probe.stdout.decode("utf-8").strip()).resolve()
    try:
        source_scope = path.relative_to(root)
    except ValueError as error:
        raise RuntimeError(
            f"Source path {path} is outside its Git root {root}"
        ) from error
    scope_argument = source_scope.as_posix() or "."
    tracked = run_at(
        root,
        "ls-files",
        "-z",
        "--",
        scope_argument,
    )
    if not tracked:
        return source_tree_descriptor()
    commit = run_at(root, "rev-parse", "HEAD").decode("ascii").strip()
    diff = run_at(
        root,
        "diff",
        "--binary",
        "--no-ext-diff",
        "HEAD",
        "--",
        scope_argument,
    )
    untracked_output = run_at(
        root,
        "ls-files",
        "--others",
        "--exclude-standard",
        "-z",
        "--",
        scope_argument,
    )
    untracked = []
    for encoded in sorted(item for item in untracked_output.split(b"\0") if item):
        relative = encoded.decode("utf-8", errors="surrogateescape")
        candidate = root / relative
        relative_path = Path(relative)
        if any(
            part in _EPHEMERAL_SOURCE_DIRECTORIES
            for part in relative_path.parts
        ) or relative_path.suffix in _EPHEMERAL_SOURCE_SUFFIXES:
            continue
        if candidate.is_file():
            untracked.append(
                {
                    "path": relative,
                    "size": candidate.stat().st_size,
                    "sha256": file_sha256(candidate),
                }
            )
    submodules = run_at(
        root,
        "submodule",
        "status",
        "--recursive",
        "--",
        scope_argument,
    ).decode("utf-8", errors="replace")
    return {
        "path": str(path),
        "repository_root": str(root),
        "source_scope": scope_argument,
        "git_available": True,
        "commit": commit,
        "tracked_diff_sha256": hashlib.sha256(diff).hexdigest(),
        "untracked": untracked,
        "submodules_sha256": hashlib.sha256(submodules.encode()).hexdigest(),
        "clean": not diff and not untracked,
    }


def _asset_descriptors(args: argparse.Namespace) -> list[dict[str, Any]]:
    paths = [
        *sorted((REPO_ROOT / "assets").rglob("*")),
        *sorted((REPO_ROOT / "configs").rglob("*")),
        REPO_ROOT / "debug" / "scene_collision_boxes.json",
        REPO_ROOT / "debug" / "dynamic_bottle_body.json",
    ]
    if args.scene_visuals:
        paths.append(REPO_ROOT / "scene" / "scene.glb")
    descriptors = []
    seen = set()
    for path in paths:
        if not path.is_file():
            continue
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        descriptor = _file_descriptor(resolved)
        descriptor["repository_relative_path"] = str(
            path.relative_to(REPO_ROOT)
        )
        descriptors.append(descriptor)
    return descriptors


def _newton_source_root() -> Path:
    spec = importlib.util.find_spec("newton")
    if spec is None or spec.origin is None:
        raise RuntimeError("Cannot resolve the active Newton source tree")
    return Path(spec.origin).resolve().parents[1]


def _artifact_descriptor(args: argparse.Namespace) -> dict[str, Any]:
    checkpoint = args.policy_checkpoint.expanduser().resolve()
    vlm = args.vlm_model.expanduser().resolve()
    descriptor = {
        "policy_directory": _directory_descriptors(checkpoint),
        "vlm_directory": _directory_descriptors(vlm),
        "initial_dit_overlay": (
            _directory_descriptors(args.dit_overlay)
            if args.dit_overlay is not None
            else None
        ),
        "sources": {
            "rexpolicy": _git_source_descriptor(REPO_ROOT),
            "isaac_groot": _git_source_descriptor(args.isaac_groot_root),
            "newton": _git_source_descriptor(
                _newton_source_root() / "newton"
            ),
        },
        "simulator_assets": _asset_descriptors(args),
    }
    if args.success_manifold_bundle is not None:
        descriptor["success_manifold_bundle"] = _file_descriptor(
            args.success_manifold_bundle
        )
    return descriptor


def _git_commit() -> str:
    result = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _build_run_manifest(
    *,
    args: argparse.Namespace,
    context: DistributedContext,
    task: FlywheelTaskSpec,
    task_artifacts: TaskArtifacts,
    runtime_binding: RuntimeTaskBinding,
) -> dict[str, Any]:
    evaluation_suite = fixed_evaluation_suite(
        base_seed=args.seed,
        reset_recipes=args.eval_reset_recipes,
        diffusion_seeds_per_recipe=args.eval_diffusion_seeds,
    )
    evaluation_hash = hashlib.sha256(
        json.dumps(evaluation_suite, separators=(",", ":")).encode()
    ).hexdigest()
    manifest = {
        "schema_version": 1,
        "code_commit": _git_commit(),
        "world_size": context.world_size,
        "task": {
            **asdict(task),
            "fingerprint": task.fingerprint,
            "environment_contract": {
                "goal_offset_world_m": [0.180, 0.080, 0.050],
                "success_distance_m": 0.040,
                "success_hold_control_steps": 2,
                "bottle_displacement_limit_m": 0.010,
                "reward_distance_scale_m": 0.030,
                "reset_xy_jitter_m": 0.010,
            },
        },
        "task_v2": {
            "task_contract_id": task_artifacts.contract.task_contract_id,
            "task_contract_sha256": task_artifacts.contract.fingerprint,
            "task_oracle_sha256": task_artifacts.contract.oracle_fingerprint,
            "task_spec_sha256": task_artifacts.task.fingerprint,
            "compiler_policy_sha256": (
                task_artifacts.contract.compiler_policy_fingerprint
            ),
            "capability_catalog_sha256": (
                task_artifacts.capabilities.fingerprint
            ),
            "adapter_id": task_artifacts.contract.adapter_id,
            "adapter_sha256": task_artifacts.contract.adapter_fingerprint,
            "event_schema_id": task_artifacts.contract.event_schema_id,
            "event_schema_sha256": (
                task_artifacts.contract.event_schema_fingerprint
            ),
            "process_spec_id": task_artifacts.contract.process_spec_id,
            "reward_profile_id": runtime_binding.reward_profile_id,
            "reward_profile_sha256": (
                runtime_binding.reward_profile_fingerprint
            ),
            "property_report_sha256": (
                task_artifacts.property_report.fingerprint
            ),
            "legacy_binding_sha256": runtime_binding.fingerprint,
        },
        "precision_contract": "fp32_dit_master_bf16_autocast_v1",
        "update_rule": "global_advantage_weighted_flow_matching_v1",
        "artifacts": _artifact_descriptor(args),
        "evaluation_suite_sha256": evaluation_hash,
        "config": {
            "max_generations": args.max_generations,
            "episodes_per_generation": args.episodes_per_generation,
            "candidates_per_state": args.candidates_per_state,
            "episode_control_steps": args.episode_control_steps,
            "execution_horizon": args.execution_horizon,
            "chunk_elite_fraction": args.chunk_elite_fraction,
            "advantage_temperature": args.advantage_temperature,
            "same_state_tolerance": args.same_state_tolerance,
            "historical_replay_state_tolerance": (
                args.historical_replay_state_tolerance
            ),
            "candidate_replay_reward_tolerance": (
                args.candidate_replay_reward_tolerance
            ),
            "train_steps_per_generation": args.train_steps_per_generation,
            "train_batch_size": args.train_batch_size,
            "gradient_accumulation": args.gradient_accumulation,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "gradient_clip_norm": args.gradient_clip_norm,
            "parameter_probe_size": args.parameter_probe_size,
            "parameter_sync_tolerance": args.parameter_sync_tolerance,
            "success_replay_per_rank": args.success_replay_per_rank,
            "seed": args.seed,
            "checkpoint_every": args.checkpoint_every,
            "eval_every": args.eval_every,
            "eval_at_start": args.eval_at_start,
            "eval_at_end": args.eval_at_end,
            "eval_reset_recipes": args.eval_reset_recipes,
            "eval_diffusion_seeds": args.eval_diffusion_seeds,
            "optimizer_state_offload": args.optimizer_state_offload,
            "offload_vlm_during_update": args.offload_vlm_during_update,
            "camera_textures": args.camera_textures,
            "scene_visuals": args.scene_visuals,
            "capture_graph": args.capture_graph,
            "hydroelastic": args.hydroelastic,
            "bottle_settle_frames": args.bottle_settle_frames,
            "substeps_per_frame": args.substeps_per_frame,
            "heartbeat_interval_seconds": args.heartbeat_seconds,
            "gpu_monitor_interval_seconds": args.gpu_monitor_seconds,
        },
    }
    if args.success_manifold_bundle is not None:
        manifest["success_manifold"] = {
            "schema_version": 1,
            "mode": "selector_with_bounded_latent_memory",
            "bundle_sha256": (
                args.expected_success_manifold_bundle_sha256
            ),
            "memory_candidates_per_state": args.manifold_memory_candidates,
            "oracle_authority": "none",
        }
    if args.success_experience_graph:
        policy = FutureWindowPolicy(
            action_horizon=args.success_action_horizon,
            future_horizon=args.success_future_horizon,
            stride=args.success_window_stride,
        )
        manifest["success_experience_graph"] = {
            "schema_version": 1,
            "mode": "derived_metadata_only",
            "window_policy": policy.to_record(),
            "window_policy_sha256": policy.fingerprint,
        }
    return manifest


def _simulator_fingerprint(
    manifest: dict[str, Any],
) -> str:
    """Bind physical execution separately from render/observation settings."""
    payload = {
        "code_commit": manifest["code_commit"],
        "task": manifest["task"],
        "sources": manifest["artifacts"]["sources"],
        "simulator_assets": manifest["artifacts"]["simulator_assets"],
        "physics": {
            key: manifest["config"][key]
            for key in (
                "episode_control_steps",
                "hydroelastic",
                "bottle_settle_frames",
                "substeps_per_frame",
            )
        },
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _observation_contract_fingerprint(manifest: dict[str, Any]) -> str:
    """Bind policy-visible modalities separately from physics and reward."""
    return canonical_fingerprint(
        {
            "schema_version": 1,
            "observation_contract_id": "groot/reach_observation/v1",
            "modalities": ["state_dict", "ego_rgb", "wrist_rgb", "language"],
            "action_conditioning": [
                "eef_9d",
                "hand_joint_pos",
                "arm_joint_pos",
            ],
            "camera_textures": manifest["config"]["camera_textures"],
            "scene_visuals": manifest["config"]["scene_visuals"],
            "vlm_artifacts": manifest["artifacts"]["vlm_directory"],
            "policy_artifacts": manifest["artifacts"]["policy_directory"],
            "isaac_groot_source": manifest["artifacts"]["sources"][
                "isaac_groot"
            ],
        }
    )


def _physical_gpu_token(local_rank: int) -> str | int:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible:
        tokens = [token.strip() for token in visible.split(",") if token.strip()]
        if local_rank >= len(tokens):
            raise RuntimeError("LOCAL_RANK exceeds CUDA_VISIBLE_DEVICES")
        return tokens[local_rank]
    return local_rank


def _coordinated_gpu_preflight(
    *,
    args: argparse.Namespace,
    context: DistributedContext,
) -> None:
    if not args.strict_gpu_preflight:
        return
    worker_pids = [
        int(pid) for pid in context.all_gather_objects(os.getpid())
    ]
    error = None
    try:
        preflight_selected_gpus(
            [_physical_gpu_token(context.local_rank)],
            minimum_free_mib=args.minimum_free_gpu_mib,
            allowed_pids=worker_pids,
        )
    except Exception as caught:
        error = f"{type(caught).__name__}: {caught}"
    errors = [item for item in context.all_gather_objects(error) if item]
    if errors:
        if any("foreign compute processes" in item for item in errors):
            raise GpuPreflightError("; ".join(errors))
        raise RuntimeError("GPU preflight failed: " + "; ".join(errors))


def _early_gpu_preflight(args: argparse.Namespace) -> None:
    """Reject occupied GPUs before creating a CUDA or NCCL context."""
    if not args.strict_gpu_preflight:
        return
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    preflight_selected_gpus(
        [_physical_gpu_token(local_rank)],
        minimum_free_mib=args.minimum_free_gpu_mib,
        allowed_pids=[os.getpid()],
    )


def _generation_metrics(
    *,
    context: DistributedContext,
    episodes: list[EpisodeExperience],
    current_samples: list[TrainingSample],
    historical_references: list[SuccessReference],
    historical_samples: list[TrainingSample],
    update: UpdateMetrics,
    archive: SuccessArchive,
    collect_seconds: float,
    historical_replay_seconds: float,
    archive_write_seconds: float,
    peak_allocated: int,
    peak_reserved: int,
    vlm_restore_seconds: float,
    vlm_offload_seconds: float,
    archive_commit_seconds: float,
) -> dict[str, Any]:
    local_decisions = sum(len(episode.decisions) for episode in episodes)
    local_candidates = sum(
        len(decision["candidates"])
        for episode in episodes
        for decision in episode.decisions
    )
    local_selected = sum(len(episode.samples) for episode in episodes)
    local_successes = sum(episode.success for episode in episodes)
    local_direct_or_path = sum(
        len(episode.success_samples) for episode in episodes
    )
    diversity_values = [
        decision["normalized_action_pairwise_rms_mean"]
        for episode in episodes
        for decision in episode.decisions
    ]
    local_diversity = (
        sum(diversity_values) / len(diversity_values)
        if diversity_values
        else 0.0
    )
    exposure_by_rank = context.all_gather_objects(
        {
            "rank": context.rank,
            "selected_sample_ids": [
                sample.sample_id for sample in current_samples
            ],
            "new_success_sample_ids": [
                sample.sample_id for sample in current_samples if sample.is_success
            ],
            "historical_replay_sample_ids": [
                reference.sample_id for reference in historical_references
            ],
            "historical_materialized_sample_ids": [
                sample.sample_id for sample in historical_samples
            ],
            "archive_size": archive.size,
            "archive_eligible_size": archive.eligible_size,
            "archive_quarantined_size": archive.quarantined_size,
            "archive_cursor": archive.cursor,
            "optimizer_state_device": update.optimizer_state_device,
            "stage_seconds": {
                "vlm_restore": vlm_restore_seconds,
                "historical_replay": historical_replay_seconds,
                "collection": collect_seconds,
                "archive_write": archive_write_seconds,
                "vlm_offload": vlm_offload_seconds,
                "optimizer_restore": update.optimizer_restore_seconds,
                "update": update.elapsed_seconds,
                "optimizer_offload": update.optimizer_offload_seconds,
                "archive_commit": archive_commit_seconds,
            },
        }
    )
    stage_seconds_by_rank = [
        {
            "rank": record["rank"],
            **record["stage_seconds"],
        }
        for record in exposure_by_rank
    ]

    def stage_summary(name: str) -> tuple[float, float]:
        values = [
            float(record["stage_seconds"][name])
            for record in exposure_by_rank
        ]
        return sum(values) / len(values), max(values)

    stage_summaries = {
        name: stage_summary(name)
        for name in (
            "vlm_restore",
            "historical_replay",
            "collection",
            "archive_write",
            "vlm_offload",
            "optimizer_restore",
            "update",
            "optimizer_offload",
            "archive_commit",
        )
    }
    return {
        "episodes": context.sum_int(len(episodes)),
        "decisions": context.sum_int(local_decisions),
        "candidate_chunks": context.sum_int(local_candidates),
        "selected_chunks": context.sum_int(local_selected),
        "successful_episodes": context.sum_int(local_successes),
        "new_success_chunks": context.sum_int(local_direct_or_path),
        "success_archive_size": context.sum_int(archive.size),
        "success_archive_eligible_size": context.sum_int(
            archive.eligible_size
        ),
        "success_archive_quarantined_size": context.sum_int(
            archive.quarantined_size
        ),
        "archive_cursor_by_rank": [
            record["archive_cursor"] for record in exposure_by_rank
        ],
        "optimizer_state_device_by_rank": [
            record["optimizer_state_device"] for record in exposure_by_rank
        ],
        "sample_exposure_by_rank": exposure_by_rank,
        "stage_seconds_by_rank": stage_seconds_by_rank,
        "mean_action_diversity": context.mean(local_diversity),
        "collect_seconds_mean": stage_summaries["collection"][0],
        "collect_seconds_max": stage_summaries["collection"][1],
        "historical_replay_seconds_mean": stage_summaries[
            "historical_replay"
        ][0],
        "historical_replay_seconds_max": stage_summaries[
            "historical_replay"
        ][1],
        "archive_write_seconds_mean": stage_summaries["archive_write"][0],
        "archive_write_seconds_max": stage_summaries["archive_write"][1],
        "loss": update.mean_loss,
        "requested_optimizer_steps": update.requested_optimizer_steps,
        "effective_optimizer_steps": update.optimizer_steps,
        "global_optimizer_step": update.global_optimizer_step,
        "global_current_samples": update.global_current_samples,
        "global_current_success_samples": (
            update.global_current_success_samples
        ),
        "global_historical_samples": update.global_historical_samples,
        "global_current_exposed": update.global_current_exposed,
        "current_coverage": update.current_coverage,
        "global_current_success_exposed": (
            update.global_current_success_exposed
        ),
        "current_success_coverage": update.current_success_coverage,
        "global_weight": update.global_weight,
        "gradient_norm": update.gradient_norm,
        "gradient_norm_max": update.gradient_norm_max,
        "gradient_norm_rank_spread": update.gradient_norm_rank_spread,
        "parameter_update_rms": update.update_rms,
        "parameter_update_max_abs": update.update_max_abs,
        "parameter_changed_fraction": update.changed_fraction,
        "parameter_update_rms_rank_spread": (
            update.update_rms_rank_spread
        ),
        "parameter_update_max_abs_rank_spread": (
            update.update_max_abs_rank_spread
        ),
        "parameter_changed_fraction_rank_spread": (
            update.changed_fraction_rank_spread
        ),
        "parameter_probe_size": update.parameter_probe_size,
        "parameter_sync_max_abs": update.parameter_sync_max_abs,
        "update_seconds_mean": stage_summaries["update"][0],
        "update_seconds_max": stage_summaries["update"][1],
        "optimizer_restore_seconds_mean": stage_summaries[
            "optimizer_restore"
        ][0],
        "optimizer_restore_seconds_max": stage_summaries[
            "optimizer_restore"
        ][1],
        "optimizer_offload_seconds_mean": stage_summaries[
            "optimizer_offload"
        ][0],
        "optimizer_offload_seconds_max": stage_summaries[
            "optimizer_offload"
        ][1],
        "vlm_restore_seconds_mean": stage_summaries["vlm_restore"][0],
        "vlm_restore_seconds_max": stage_summaries["vlm_restore"][1],
        "vlm_offload_seconds_mean": stage_summaries["vlm_offload"][0],
        "vlm_offload_seconds_max": stage_summaries["vlm_offload"][1],
        "archive_commit_seconds_mean": stage_summaries["archive_commit"][0],
        "archive_commit_seconds_max": stage_summaries["archive_commit"][1],
        "cuda_peak_allocated_bytes_max": context.max_int(peak_allocated),
        "cuda_peak_reserved_bytes_max": context.max_int(peak_reserved),
    }


def _checkpoint(
    *,
    manager: CheckpointManager,
    context: DistributedContext,
    policy: GrootFlowDitPolicy,
    trainer: DitDdpTrainer,
    archive: SuccessArchive,
    manifold: SuccessManifoldRuntime | None,
    generation: int,
    manifest: dict[str, Any],
    last_good: HeldOutMetrics | None,
    candidate_eval: HeldOutMetrics | None = None,
    accepted: bool = True,
) -> Path:
    extra_state = {
        "last_good_evaluation": (
            last_good.to_record() if last_good is not None else None
        ),
        "candidate_evaluation": (
            candidate_eval.to_record() if candidate_eval is not None else None
        ),
        "candidate_accepted": bool(accepted),
    }
    return manager.save(
        dit=policy.dit if context.is_main else None,
        optimizer=trainer.optimizer if context.is_main else None,
        completed_generation=generation,
        global_optimizer_step=trainer.global_optimizer_step,
        run_manifest=manifest,
        replay_cursor=archive.cursor,
        extra_rank_state={
            "success_archive": archive.state_dict(),
            **(
                {"success_manifold": manifold.state_dict()}
                if manifold is not None
                else {}
            ),
        },
        extra_state=extra_state,
        cuda_device=context.device,
        update_latest=accepted,
        accepted=accepted,
    )


def _write_last_good(
    *,
    run_dir: Path,
    checkpoint: Path,
    metrics: HeldOutMetrics,
) -> None:
    atomic_write_json(
        run_dir / "checkpoints" / "last_good.json",
        {
            "checkpoint": checkpoint.name,
            "generation": metrics.generation,
            "evaluation": metrics.to_record(),
        },
    )


def _initialize_operations(
    *,
    run_dir: Path,
    context: DistributedContext,
    args: argparse.Namespace,
) -> RunOperations:
    operation = None
    error = None
    try:
        operation = RunOperations(
            run_dir,
            rank=context.rank,
            world_size=context.world_size,
            heartbeat_interval_seconds=args.heartbeat_seconds,
        )
    except Exception as caught:
        error = f"{type(caught).__name__}: {caught}"
    errors = [item for item in context.all_gather_objects(error) if item]
    if errors:
        if operation is not None:
            operation.close()
        raise RuntimeError("Run initialization failed: " + "; ".join(errors))
    assert operation is not None
    return operation


def main() -> None:
    args = create_parser().parse_args()
    task = get_task_spec(args.task_id)
    task_artifacts, runtime_binding = load_production_reach_runtime_binding()
    _validate_args(args, task)
    if args.episode_control_steps != task_artifacts.contract.episode_control_steps:
        raise ValueError(
            "episode-control-steps must match the TaskSpec v2 contract: "
            f"{task_artifacts.contract.episode_control_steps}"
        )
    _early_gpu_preflight(args)
    context = DistributedContext.initialize()
    operations: RunOperations | None = None
    env: GrootNewtonEnv | None = None
    completed_generation = 0
    terminal_status = "failed"
    terminal_reason = "unhandled failure"
    try:
        _validate_artifacts(args)
        if args.validate_only:
            _coordinated_gpu_preflight(args=args, context=context)
            context.barrier()
            if context.is_main:
                print(
                    f"DDP validation passed: world_size={context.world_size}",
                    flush=True,
                )
            return

        assert args.run_dir is not None
        run_dir = args.run_dir.expanduser().resolve()
        _validate_run_directory_mode(
            run_dir=run_dir,
            context=context,
            resume=args.resume_latest or args.resume_from is not None,
            resume_from=args.resume_from,
        )
        operations = _initialize_operations(
            run_dir=run_dir,
            context=context,
            args=args,
        )
        if args.resume_latest or args.resume_from is not None:
            operations.clear_stop_request()
        operations.install_generation_boundary_signal_handlers()
        operations.update_heartbeat(status="preflight")
        _coordinated_gpu_preflight(args=args, context=context)
        operations.start_background(
            gpu_monitor_interval_seconds=args.gpu_monitor_seconds
        )

        rank_seed = args.seed + context.rank * 1_000_003
        random.seed(rank_seed)
        np.random.seed(rank_seed % (2**32))
        import torch

        torch.manual_seed(rank_seed)
        torch.cuda.manual_seed(rank_seed)

        manifest = context.broadcast_object(
            (
                _build_run_manifest(
                    args=args,
                    context=context,
                    task=task,
                    task_artifacts=task_artifacts,
                    runtime_binding=runtime_binding,
                )
                if context.is_main
                else None
            ),
            source=0,
        )
        simulator_fingerprint = _simulator_fingerprint(manifest)
        observation_contract_sha256 = _observation_contract_fingerprint(
            manifest
        )
        operations.write_manifest(
            config=manifest,
            command=sys.argv,
            environment=os.environ,
            extra={
                "simulator_fingerprint": simulator_fingerprint,
                "observation_contract_sha256": observation_contract_sha256,
                "task_contract_sha256": task_artifacts.contract.fingerprint,
                "task_oracle_sha256": (
                    task_artifacts.contract.oracle_fingerprint
                ),
            },
            repo_root=REPO_ROOT,
        )
        operations.log(
            "loading frozen GR00T VLM and FP32 Flow-DiT",
            task_id=task.task_id,
            device=str(context.device),
        )
        operations.update_heartbeat(status="loading_model")
        policy = GrootFlowDitPolicy(
            isaac_groot_root=args.isaac_groot_root,
            checkpoint=args.policy_checkpoint,
            vlm_model=args.vlm_model,
            device=context.device,
            dit_overlay=args.dit_overlay,
            enable_success_conditioning=(
                args.success_manifold_bundle is not None
            ),
        )
        policy.assert_precision_contract()
        manifold = None
        if args.success_manifold_bundle is not None:
            manifold = SuccessManifoldRuntime.load(
                args.success_manifold_bundle,
                expected_sha256=(
                    args.expected_success_manifold_bundle_sha256
                ),
                device=context.device,
            )
            if manifold.config.condition_dim != policy.success_token_dimension:
                raise ValueError(
                    "Success-manifold condition_dim does not match frozen "
                    "GR00T backbone width"
                )
        trainer = DitDdpTrainer(
            action_head=policy.action_head,
            context=context,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            gradient_clip_norm=args.gradient_clip_norm,
            optimizer_state_offload=args.optimizer_state_offload,
            parameter_probe_size=args.parameter_probe_size,
            parameter_sync_tolerance=args.parameter_sync_tolerance,
        )
        checkpoint_manager = CheckpointManager(
            run_dir,
            rank=context.rank,
            world_size=context.world_size,
            barrier=context.barrier,
        )
        attempt_id = context.broadcast_object(
            (
                f"{int(time.time())}-{uuid.uuid4().hex[:8]}"
                if context.is_main
                else None
            ),
            source=0,
        )

        last_good: HeldOutMetrics | None = None
        if args.resume_latest or args.resume_from is not None:
            operations.update_heartbeat(status="resuming")
            policy.offload_backbone()
            resume = checkpoint_manager.load(
                dit=policy.dit,
                optimizer=trainer.optimizer,
                current_manifest=manifest,
                checkpoint=args.resume_from,
                cuda_device=context.device,
                optimizer_device="cpu",
            )
            trainer.global_optimizer_step = resume.global_optimizer_step
            trainer.offload_optimizer_state()
            archive_state = resume.rank_state.get("success_archive")
            if archive_state is None:
                raise RuntimeError("Checkpoint is missing Success Archive state")
            archive = SuccessArchive(
                run_dir=run_dir,
                rank=context.rank,
                attempt_id=attempt_id,
                state=archive_state,
            )
            if manifold is not None:
                manifold_state = resume.rank_state.get("success_manifold")
                if manifold_state is None:
                    raise RuntimeError(
                        "Checkpoint is missing Success Manifold runtime state"
                    )
                manifold.load_state_dict(manifold_state)
            if resume.replay_cursor != archive.cursor:
                raise RuntimeError(
                    "Checkpoint replay cursor disagrees with Success Archive "
                    f"state: rank_state={archive.cursor}, "
                    f"checkpoint={resume.replay_cursor}"
                )
            completed_generation = resume.completed_generation
            start_generation = resume.next_generation
            last_good_record = resume.extra_state.get("last_good_evaluation")
            if last_good_record is not None:
                last_good = HeldOutMetrics.from_record(last_good_record)
            policy.restore_backbone()
            operations.log(
                "resumed complete checkpoint",
                checkpoint=str(resume.checkpoint_path),
                completed_generation=completed_generation,
                next_generation=start_generation,
                optimizer_step=trainer.global_optimizer_step,
            )
        else:
            if (run_dir / "checkpoints" / "latest.json").exists():
                raise RuntimeError(
                    "Run directory already contains checkpoints; use --resume-latest"
                )
            archive = SuccessArchive(
                run_dir=run_dir,
                rank=context.rank,
                attempt_id=attempt_id,
            )
            start_generation = 1
            if args.eval_at_start and args.eval_every > 0:
                operations.update_heartbeat(
                    status="evaluating",
                    generation=0,
                )
                last_good = _evaluate_held_out(
                    args=args,
                    context=context,
                    policy=policy,
                    task=task,
                    generation=0,
                    manifold=manifold,
                )
                operations.metric("held_out_evaluation", **last_good.to_record())
            if args.save:
                policy.offload_backbone()
                baseline_checkpoint = _checkpoint(
                    manager=checkpoint_manager,
                    context=context,
                    policy=policy,
                    trainer=trainer,
                    archive=archive,
                    manifold=manifold,
                    generation=0,
                    manifest=manifest,
                    last_good=last_good,
                )
                if context.is_main and last_good is not None:
                    _write_last_good(
                        run_dir=run_dir,
                        checkpoint=baseline_checkpoint,
                        metrics=last_good,
                    )
                policy.restore_backbone()
                operations.log(
                    "saved generation-zero baseline",
                    checkpoint=str(baseline_checkpoint),
                )

        if start_generation > args.max_generations:
            terminal_status = "completed"
            terminal_reason = (
                f"checkpoint already reached generation {completed_generation}"
            )
            return

        for generation in range(start_generation, args.max_generations + 1):
            stop_now = context.max_int(int(operations.stop_requested())) > 0
            if stop_now:
                terminal_status = "stopped"
                terminal_reason = "stop requested before next generation"
                break
            operations.update_heartbeat(
                status="collecting",
                generation=generation,
            )
            operations.log("starting generation", generation=generation)
            torch.cuda.reset_peak_memory_stats(context.device)
            vlm_restore_started = time.perf_counter()
            policy.restore_backbone()
            vlm_restore_seconds = time.perf_counter() - vlm_restore_started
            if env is None:
                env = _create_environment(args, context, task)

            historical_started = time.perf_counter()
            historical_references = archive.plan(
                replay_per_rank=args.success_replay_per_rank,
                max_generation_exclusive=generation,
            )
            (
                historical_samples,
                replay_quarantines,
            ) = _materialize_historical_samples(
                references=historical_references,
                run_dir=run_dir,
                args=args,
                context=context,
                policy=policy,
                env=env,
                task=task,
                simulator_fingerprint=simulator_fingerprint,
                observation_contract_sha256=observation_contract_sha256,
                archive=archive,
                generation=generation,
            )
            for quarantine in replay_quarantines:
                operations.metric(
                    "archive_quarantine",
                    generation=generation,
                    rank=context.rank,
                    **quarantine,
                )
                operations.log(
                    "quarantined historical replay",
                    generation=generation,
                    rank=context.rank,
                    **quarantine,
                )
            historical_seconds = time.perf_counter() - historical_started
            collect_started = time.perf_counter()
            collected_episodes = _collect_generation(
                args=args,
                context=context,
                policy=policy,
                env=env,
                task=task,
                generation=generation,
                simulator_fingerprint=simulator_fingerprint,
                task_artifacts=task_artifacts,
                runtime_binding=runtime_binding,
                observation_contract_sha256=observation_contract_sha256,
                manifold=manifold,
            )
            episodes = [item.experience for item in collected_episodes]
            event_ledgers = [item.event_ledger for item in collected_episodes]
            collect_seconds = time.perf_counter() - collect_started
            archive_started = time.perf_counter()
            (
                archive_pair,
                success_references,
                success_graphs,
            ) = _write_generation_archive(
                run_dir=run_dir,
                attempt_id=attempt_id,
                generation=generation,
                rank=context.rank,
                episodes=episodes,
                event_ledgers=event_ledgers,
                window_policy=(
                    FutureWindowPolicy(
                        action_horizon=args.success_action_horizon,
                        future_horizon=args.success_future_horizon,
                        stride=args.success_window_stride,
                    )
                    if args.success_experience_graph
                    else None
                ),
            )
            pending_success_shard = archive.write_generation(
                generation=generation,
                references=success_references,
            )
            archive_write_seconds = time.perf_counter() - archive_started
            current_samples = [
                sample for episode in episodes for sample in episode.samples
            ]
            if not current_samples:
                raise RuntimeError("Collector produced no selected training samples")

            operations.update_heartbeat(
                status="updating",
                generation=generation,
            )
            vlm_offload_started = time.perf_counter()
            if args.offload_vlm_during_update:
                policy.offload_backbone()
            vlm_offload_seconds = time.perf_counter() - vlm_offload_started
            update = trainer.update(
                current_samples,
                historical_samples=historical_samples,
                optimizer_steps=args.train_steps_per_generation,
                batch_size=args.train_batch_size,
                gradient_accumulation=args.gradient_accumulation,
                seed=args.seed + generation * 97,
                dtype=policy.compute_dtype,
                dummy_sample=current_samples[0],
            )
            archive_commit_started = time.perf_counter()
            if pending_success_shard is not None:
                archive.commit_shard(pending_success_shard)
            success_graph_writes = [
                write_derived_view(run_dir=run_dir, view=graph)
                for graph in success_graphs
            ]
            archive_commit_seconds = (
                time.perf_counter() - archive_commit_started
            )
            completed_generation = generation
            peak_allocated = torch.cuda.max_memory_allocated(context.device)
            peak_reserved = torch.cuda.max_memory_reserved(context.device)
            metrics = _generation_metrics(
                context=context,
                episodes=episodes,
                current_samples=current_samples,
                historical_references=historical_references,
                historical_samples=historical_samples,
                update=update,
                archive=archive,
                collect_seconds=collect_seconds,
                historical_replay_seconds=historical_seconds,
                archive_write_seconds=archive_write_seconds,
                peak_allocated=peak_allocated,
                peak_reserved=peak_reserved,
                vlm_restore_seconds=vlm_restore_seconds,
                vlm_offload_seconds=vlm_offload_seconds,
                archive_commit_seconds=archive_commit_seconds,
            )
            operations.metric(
                "generation",
                generation=generation,
                episode_archive=archive_pair.episode_path,
                event_archive=archive_pair.event_path,
                archive_pair_descriptor=archive_pair.pair_descriptor_path,
                historical_replay_samples=len(historical_samples),
                **(
                    {
                        "success_experience_graphs": [
                            item.relative_path
                            for item in success_graph_writes
                        ]
                    }
                    if args.success_experience_graph
                    else {}
                ),
                **metrics,
            )
            operations.log(
                "completed synchronized FP32 update",
                generation=generation,
                loss=update.mean_loss,
                effective_optimizer_steps=update.optimizer_steps,
                coverage=update.current_coverage,
                update_rms=update.update_rms,
                update_max_abs=update.update_max_abs,
                changed_fraction=update.changed_fraction,
                sync_max_abs=update.parameter_sync_max_abs,
            )

            evaluation_due = args.eval_every > 0 and (
                generation % args.eval_every == 0
                or (args.eval_at_end and generation == args.max_generations)
            )
            candidate_eval = None
            accepted = True
            if evaluation_due:
                evaluation_started = time.perf_counter()
                if env is not None:
                    env.close()
                    env = None
                policy.restore_backbone()
                operations.update_heartbeat(
                    status="evaluating",
                    generation=generation,
                )
                candidate_eval = _evaluate_held_out(
                    args=args,
                    context=context,
                    policy=policy,
                    task=task,
                    generation=generation,
                    manifold=manifold,
                )
                operations.metric(
                    "held_out_evaluation",
                    elapsed_seconds=time.perf_counter() - evaluation_started,
                    **candidate_eval.to_record(),
                )
                if last_good is not None:
                    decision = evaluate_non_regression(
                        last_good=last_good,
                        candidate=candidate_eval,
                    )
                    accepted = decision.accepted
                    operations.metric(
                        "held_out_gate",
                        generation=generation,
                        accepted=accepted,
                        reasons=list(decision.reasons),
                    )
                    if not accepted:
                        operations.log(
                            "held-out gate rejected candidate",
                            level="ERROR",
                            generation=generation,
                            reasons=list(decision.reasons),
                        )
                if accepted:
                    last_good = candidate_eval

            stop_after_generation = (
                context.max_int(int(operations.stop_requested())) > 0
            )
            checkpoint_due = args.save and (
                generation % args.checkpoint_every == 0
                or generation == args.max_generations
                or evaluation_due
                or stop_after_generation
                or not accepted
            )
            checkpoint_path = None
            if checkpoint_due:
                checkpoint_started = time.perf_counter()
                operations.update_heartbeat(
                    status="checkpointing",
                    generation=generation,
                )
                if not policy.backbone_offloaded:
                    policy.offload_backbone()
                checkpoint_path = _checkpoint(
                    manager=checkpoint_manager,
                    context=context,
                    policy=policy,
                    trainer=trainer,
                    archive=archive,
                    manifold=manifold,
                    generation=generation,
                    manifest=manifest,
                    last_good=last_good,
                    candidate_eval=candidate_eval,
                    accepted=accepted,
                )
                operations.metric(
                    "checkpoint",
                    generation=generation,
                    path=str(checkpoint_path),
                    accepted=accepted,
                    elapsed_seconds=time.perf_counter() - checkpoint_started,
                )
                if (
                    context.is_main
                    and accepted
                    and candidate_eval is not None
                ):
                    _write_last_good(
                        run_dir=run_dir,
                        checkpoint=checkpoint_path,
                        metrics=candidate_eval,
                    )
            if not accepted:
                terminal_status = "rejected"
                terminal_reason = (
                    f"held-out evaluation rejected generation {generation}"
                )
                break
            if stop_after_generation:
                terminal_status = "stopped"
                terminal_reason = (
                    f"stop requested after generation {generation}"
                )
                break
        else:
            terminal_status = "completed"
            terminal_reason = (
                f"reached max_generations={args.max_generations}"
            )
    except Exception as error:
        terminal_status = "failed"
        terminal_reason = f"{type(error).__name__}: {error}"
        if operations is not None:
            operations.log(
                "flywheel run failed",
                level="ERROR",
                error=terminal_reason,
            )
        raise
    finally:
        if env is not None:
            env.close()
        if operations is not None:
            operations.mark_exit(
                status=terminal_status,
                reason=terminal_reason,
                completed_generation=completed_generation,
            )
            operations.close()
        context.close()


if __name__ == "__main__":
    main()
