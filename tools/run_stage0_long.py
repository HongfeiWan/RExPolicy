#!/usr/bin/env python3
"""Collect a multimode Reach corpus and run resumable Stage 0 long training.

The process deliberately imports Torch/Newton only after the CPU thread limit
is installed.  Checkpoints contain four independent components: the future
encoder, selector, success-conditioned policy, and an equal-step no-z policy.
Periodic evaluations are offline diagnostics; rollout success remains a
separate scientific gate and is never inferred from action reconstruction.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import signal
import sys
import time
import uuid
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

from rexpolicy.stage0.long_run import (
    AtomicJsonlLog,
    LongRunCursor,
    LongRunPhase,
    Stage0LongRunConfig,
    WallClockBudget,
    atomic_write_json,
    load_long_run_config,
)
from rexpolicy.stage0.types import Stage0Outcome, canonical_fingerprint


_CORPUS_MODE_SCHEMA_ID = "rexpolicy/stage0-reach-mode-corpus/v1"
_RUN_CONTRACT_SCHEMA_ID = "rexpolicy/stage0-long-run-contract/v1"
_RANK_STATE_SCHEMA_ID = "rexpolicy/stage0-long-run-rank-state/v1"
_MODE_DIAGNOSTIC_PROTOCOL = {
    "classification_nmi_minimum": 0.65,
    "euclidean_silhouette_minimum": 0.25,
    "macro_f1_minimum": 0.75,
    "mode_to_reset_distance_ratio_minimum": 2.0,
    "reset_xy_linear_probe_r2_maximum": 0.20,
    "ridge_alpha": 1.0e-3,
    "schema_id": "rexpolicy/stage0-held-out-mode-gate/v1",
    "selection": "complete-reset-groups-start-zero/v1",
}
_SELECTOR_DIAGNOSTIC_PROTOCOL = {
    "all_components_cover_all_modes_per_reset": True,
    "component_count": "one_per_configured_mode",
    "maximum_normalized_mode_to_component_distance": 1.0,
    "minimum_component_probability_per_reset": 0.10,
    "nll_must_improve_over_held_out_global_diagonal_gaussian": True,
    "schema_id": "rexpolicy/stage0-selector-gate/v2",
    "warm_start": "train-start-zero-mode-centroids/v1",
}
_ROLLOUT_PROTOCOL = {
    "maximum_evaluation_windows": 64,
    "maximum_object_displacement_m": 0.005,
    "minimum_oracle_z_success_rate": 0.80,
    "minimum_selector_z_success_rate": 0.80,
    "schema_id": "rexpolicy/stage0-newton-rollout-gate/v1",
}
_PATH_ADHERENCE_PROTOCOL = {
    "all_selector_modes_must_be_sampled": True,
    "oracle_z_macro_f1_minimum": 0.75,
    "permuted_z_macro_f1_minimum": 0.75,
    "resample_points": 64,
    "schema_id": "rexpolicy/stage0-rollout-path-adherence-gate/v2",
    "selector_average_sampled_mode_coverage_minimum": 2.5,
    "selector_z_macro_f1_minimum": 0.70,
}
_MANIFOLD_TRAINER_CONFIG = {
    "contrastive_temperature": 0.1,
    "contrastive_weight": 1.0,
    "covariance_weight": 1.0,
    "gather_distributed": False,
    "gradient_clip_norm": 1.0,
    "mode_alignment_weight": 1.0,
    "reconstruction_probability": 0.25,
    "reconstruction_weight": 1.0,
    "temporal_radius": 1,
    "variance_epsilon": 1.0e-4,
    "variance_target_std": 1.0,
    "variance_weight": 1.0,
}
_STOP_REASON: str | None = None


@dataclass(frozen=True)
class CollectionSummary:
    trajectory_count: int
    success_count: int
    failure_count: int
    timeout_count: int
    world_steps: int
    elapsed_seconds: float
    maximum_object_displacement_m: float
    maximum_terminal_state_distance_m: float
    success_state_oracle_disagreement_count: int
    mode_outcomes: dict[str, dict[str, int]]

    def to_record(self) -> dict[str, Any]:
        return asdict(self)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/stage0/reach_long.json"),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--corpus-dir",
        type=Path,
        help="materialize and verify an existing multimode corpus instead of collecting",
    )
    parser.add_argument(
        "--resume",
        nargs="?",
        const="latest",
        metavar="CHECKPOINT",
        help="resume the latest checkpoint, or the explicit checkpoint path",
    )
    parser.add_argument("--max-wall-seconds", type=float)
    parser.add_argument("--capture-graph", action="store_true")
    parser.add_argument("--collect-only", action="store_true")
    parser.add_argument("--validate-config-only", action="store_true")
    return parser.parse_args()


def _default_output_dir() -> Path:
    return Path("outputs/stage0") / f"reach-long-{time.time_ns()}"


def _configure_cpu_threads(count: int) -> None:
    text = str(count)
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ[name] = text
    os.environ["TOKENIZERS_PARALLELISM"] = "false"


def _install_stop_handlers() -> None:
    def request_stop(signum: int, _frame: Any) -> None:
        global _STOP_REASON
        _STOP_REASON = signal.Signals(signum).name.lower()

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)


def _run_contract(config: Stage0LongRunConfig) -> dict[str, Any]:
    return {
        "config": config.to_record(),
        "config_sha256": config.fingerprint,
        "schema_id": _RUN_CONTRACT_SCHEMA_ID,
    }


def _prepare_output(
    output_dir: Path,
    config: Stage0LongRunConfig,
    *,
    resume: bool,
) -> None:
    contract_path = output_dir / "run-contract.json"
    expected = _run_contract(config)
    if resume:
        if not output_dir.is_dir() or not contract_path.is_file():
            raise FileNotFoundError(
                "--resume requires an existing output directory with run-contract.json"
            )
        actual = json.loads(contract_path.read_text(encoding="ascii"))
        if actual != expected:
            raise RuntimeError("resume run contract differs from the requested config")
        return
    if output_dir.exists():
        raise FileExistsError(f"long-run output directory already exists: {output_dir}")
    output_dir.mkdir(parents=True)
    atomic_write_json(contract_path, expected)


def _environment_config(
    config: Stage0LongRunConfig, device: str, *, capture_graph: bool
):
    from rexpolicy.envs.groot_newton_env import GrootNewtonEnvConfig

    stage0 = config.stage0
    return GrootNewtonEnvConfig(
        num_envs=stage0.num_envs,
        device=device,
        task_mode="reach_green_cap",
        obs_mode="state",
        control_mode="pd_eef_pose_abs",
        reward_mode="none",
        max_episode_steps=stage0.max_episode_steps,
        terminate_on_success=True,
        terminate_on_fail=True,
        capture_graph=capture_graph,
        render_images=False,
        camera_textures=False,
        load_scene_visuals=False,
        hydroelastic_contacts=False,
    )


def _synchronize(torch: Any, device: Any) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _failure_reason(result: Any, world: int) -> str:
    if bool(result.contact_violation[world]):
        return "unsafe-hand-object-contact"
    if bool(result.displacement_violation[world]):
        return "object-displacement-limit"
    return "newton-oracle-failure"


def _safe_mode_slug(mode_id: str) -> str:
    return "-".join(part for part in mode_id.replace("_", "-").split("/") if part)


def _empty_mode_counts(mode_ids: Sequence[str]) -> dict[str, dict[str, int]]:
    return {
        mode_id: {outcome.value: 0 for outcome in Stage0Outcome} for mode_id in mode_ids
    }


def _mode_manifest(
    config: Stage0LongRunConfig,
    *,
    library_sha256: str,
    mode_hashes: Mapping[str, str],
    trajectory_modes: Mapping[str, Mapping[str, str]],
) -> dict[str, Any]:
    return {
        "corpus_config_sha256": canonical_fingerprint(config.corpus.to_record()),
        "library_sha256": library_sha256,
        "mode_hashes": dict(sorted(mode_hashes.items())),
        "mode_ids": list(config.corpus.mode_ids),
        "schema_id": _CORPUS_MODE_SCHEMA_ID,
        "stage0_experiment_sha256": config.stage0.fingerprint,
        "trajectory_modes": {
            key: dict(value) for key, value in sorted(trajectory_modes.items())
        },
    }


def _publish_corpus_directory(partial: Path, target: Path) -> None:
    if target.exists():
        raise FileExistsError(f"corpus directory already exists: {target}")
    os.rename(partial, target)
    descriptor = os.open(target.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _collect_multimode_corpus(
    config: Stage0LongRunConfig,
    target: Path,
    *,
    device_name: str,
    capture_graph: bool,
) -> CollectionSummary:
    import torch

    from rexpolicy.envs.groot_newton_env import GrootNewtonEnv
    from rexpolicy.stage0.data import (
        Stage0Trajectory,
        Stage0TrajectoryProvenance,
        write_trajectory_shard,
    )
    from rexpolicy.stage0.envs.oracles import ReachSuccessOracle
    from rexpolicy.stage0.envs.reset_sampler import Stage0ResetSampler
    from rexpolicy.stage0.envs.state_only import Stage0StateOnlyEnv
    from rexpolicy.stage0.envs.state_schema import DEFAULT_STAGE0_STATE_SCHEMA

    try:
        from rexpolicy.stage0.envs.reach_modes import (
            DEFAULT_REACH_MODE_LIBRARY,
            ReachModeController,
        )
    except ImportError as error:
        raise RuntimeError(
            "multimode collection requires rexpolicy.stage0.envs.reach_modes "
            "with DEFAULT_REACH_MODE_LIBRARY and ReachModeController"
        ) from error

    requested_modes = tuple(config.corpus.mode_ids)
    library_modes = tuple(DEFAULT_REACH_MODE_LIBRARY.mode_ids)
    if set(requested_modes) != set(library_modes):
        raise RuntimeError(
            "long-run mode_ids must exactly match DEFAULT_REACH_MODE_LIBRARY.mode_ids"
        )
    mode_hashes = {
        mode_id: DEFAULT_REACH_MODE_LIBRARY.mode_hash(mode_id)
        for mode_id in requested_modes
    }
    partial = target.with_name(f".corpus-partial-{uuid.uuid4().hex}")
    partial.mkdir(parents=True)
    trajectory_modes: dict[str, dict[str, str]] = {}
    mode_outcomes = _empty_mode_counts(requested_modes)
    trajectory_count = 0
    world_steps = 0
    maximum_displacement = 0.0
    maximum_terminal_state_distance = 0.0
    success_state_oracle_disagreement_count = 0
    env_config = _environment_config(config, device_name, capture_graph=capture_graph)
    simulator_sha256 = canonical_fingerprint(
        {
            "config": asdict(env_config),
            "schema_id": "rexpolicy/stage0-newton-runtime/v1",
        }
    )
    action_schema_sha256 = canonical_fingerprint(
        {
            "dimension": config.stage0.action_dim,
            "fields": ["absolute_eef_xyz", "rotation_6d", "absolute_hand_q"],
            "schema_id": "rexpolicy/stage0-eef-action/v1",
        }
    )
    env = GrootNewtonEnv(env_config)
    device = torch.device(device_name)
    started = time.perf_counter()
    try:
        adapter = Stage0StateOnlyEnv(env)
        sampler = Stage0ResetSampler(config.stage0.seed, config.stage0.num_envs)
        batches = config.corpus.reset_groups // config.stage0.num_envs
        for batch_index in range(batches):
            reset_seeds = tuple(sampler.next())
            orders = tuple(
                DEFAULT_REACH_MODE_LIBRARY.ordered_mode_ids(seed)
                for seed in reset_seeds
            )
            if any(set(order) != set(requested_modes) for order in orders):
                raise RuntimeError(
                    "ordered reach modes changed the configured mode set"
                )
            for pass_index in range(len(requested_modes)):
                world_modes = tuple(order[pass_index] for order in orders)
                state, _ = adapter.reset(seed=list(reset_seeds))
                open_hand_action = env.hold_action_torch().clone()
                controller = ReachModeController()
                controller.reset(state, open_hand_action, world_modes)
                oracle = ReachSuccessOracle(
                    config.stage0.num_envs,
                    success_threshold_m=env_config.reach_success_threshold,
                    success_hold_steps=env_config.reach_success_hold_steps,
                    displacement_limit_m=env_config.reach_bottle_displacement_limit,
                )
                oracle.reset(state)
                state_rows: list[list[Any]] = [
                    [state[world].detach().cpu().clone()]
                    for world in range(config.stage0.num_envs)
                ]
                action_rows: list[list[Any]] = [
                    [] for _ in range(config.stage0.num_envs)
                ]
                outcomes: list[Stage0Outcome | None] = [None] * config.stage0.num_envs
                reasons: list[str | None] = [None] * config.stage0.num_envs
                active = torch.ones(
                    config.stage0.num_envs,
                    dtype=torch.bool,
                    device=state.device,
                )
                for _ in range(config.stage0.max_episode_steps):
                    action = controller.action(state, active_mask=active)
                    transition = adapter.step(action)
                    state = transition.next_state
                    result = oracle.evaluate(state)
                    env_success = transition.info["success"].to(dtype=torch.bool)
                    env_failure = transition.info["fail"].to(dtype=torch.bool)
                    success_state_oracle_disagreement_count += int(
                        ((result.success != env_success) & active).sum()
                    )
                    if not torch.equal(result.failure, env_failure):
                        raise RuntimeError(
                            "independent Reach oracle disagrees with Newton failure"
                        )
                    maximum_displacement = max(
                        maximum_displacement,
                        float(result.object_displacement.max()),
                    )
                    if bool(env_success.any()):
                        maximum_terminal_state_distance = max(
                            maximum_terminal_state_distance,
                            float(result.distance[env_success].max()),
                        )
                    for world in (
                        torch.nonzero(active, as_tuple=False).flatten().tolist()
                    ):
                        action_rows[world].append(action[world].detach().cpu().clone())
                        state_rows[world].append(state[world].detach().cpu().clone())
                        world_steps += 1
                        if bool(env_success[world]):
                            outcomes[world] = Stage0Outcome.SUCCESS
                            active[world] = False
                        elif bool(result.failure[world]):
                            outcomes[world] = Stage0Outcome.FAILURE
                            reasons[world] = _failure_reason(result, world)
                            active[world] = False
                    if not bool(active.any()):
                        break
                for world in torch.nonzero(active, as_tuple=False).flatten().tolist():
                    outcomes[world] = Stage0Outcome.TIMEOUT

                for world, outcome in enumerate(outcomes):
                    if outcome is None:
                        raise RuntimeError("multimode collector left an outcome unset")
                    group_index = batch_index * config.stage0.num_envs + world
                    mode_id = world_modes[world]
                    mode_index = requested_modes.index(mode_id)
                    reset_group_id = (
                        f"reset-g{group_index:06d}-s{reset_seeds[world]:010d}"
                    )
                    trajectory_id = (
                        f"reach-g{group_index:06d}-m{mode_index:02d}-"
                        f"{_safe_mode_slug(mode_id)}-"
                        f"h{mode_hashes[mode_id][:12]}-s{reset_seeds[world]:010d}"
                    )
                    trajectory = Stage0Trajectory(
                        provenance=Stage0TrajectoryProvenance(
                            trajectory_id=trajectory_id,
                            reset_group_id=reset_group_id,
                            task_id="reach_green_cap/v1",
                            oracle_id="stage0/reach-oracle/v1",
                            simulator_sha256=simulator_sha256,
                            experiment_sha256=config.stage0.fingerprint,
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
                        failure_reason=reasons[world],
                    )
                    write_trajectory_shard(partial, trajectory)
                    trajectory_modes[trajectory_id] = {
                        "mode_id": mode_id,
                        "mode_sha256": mode_hashes[mode_id],
                        "reset_group_id": reset_group_id,
                    }
                    mode_outcomes[mode_id][outcome.value] += 1
                    trajectory_count += 1
        _synchronize(torch, device)
        elapsed = time.perf_counter() - started
        atomic_write_json(
            partial / "corpus-modes.json",
            _mode_manifest(
                config,
                library_sha256=DEFAULT_REACH_MODE_LIBRARY.sha256,
                mode_hashes=mode_hashes,
                trajectory_modes=trajectory_modes,
            ),
        )
        summary = CollectionSummary(
            trajectory_count=trajectory_count,
            success_count=sum(
                values[Stage0Outcome.SUCCESS.value] for values in mode_outcomes.values()
            ),
            failure_count=sum(
                values[Stage0Outcome.FAILURE.value] for values in mode_outcomes.values()
            ),
            timeout_count=sum(
                values[Stage0Outcome.TIMEOUT.value] for values in mode_outcomes.values()
            ),
            world_steps=world_steps,
            elapsed_seconds=elapsed,
            maximum_object_displacement_m=maximum_displacement,
            maximum_terminal_state_distance_m=maximum_terminal_state_distance,
            success_state_oracle_disagreement_count=(
                success_state_oracle_disagreement_count
            ),
            mode_outcomes=mode_outcomes,
        )
        atomic_write_json(partial / "collection-summary.json", summary.to_record())
        _publish_corpus_directory(partial, target)
        return summary
    finally:
        env.close()


def _load_corpus(directory: Path) -> tuple[tuple[Any, ...], dict[str, Any]]:
    from rexpolicy.stage0.data import load_trajectory_shard

    if not directory.is_dir():
        raise FileNotFoundError(f"Stage 0 corpus directory does not exist: {directory}")
    descriptors = sorted(directory.glob("*.stage0.json"))
    if not descriptors:
        raise RuntimeError(f"Stage 0 corpus has no committed descriptors: {directory}")
    trajectories = tuple(load_trajectory_shard(path) for path in descriptors)
    manifest_path = directory / "corpus-modes.json"
    if not manifest_path.is_file():
        raise RuntimeError("multimode corpus is missing corpus-modes.json")
    manifest = json.loads(manifest_path.read_text(encoding="ascii"))
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema_id") != _CORPUS_MODE_SCHEMA_ID
    ):
        raise RuntimeError("multimode corpus manifest schema changed")
    return trajectories, manifest


def _verify_multimode_corpus(
    config: Stage0LongRunConfig,
    trajectories: Sequence[Any],
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    import torch

    from rexpolicy.stage0.envs.reach_modes import DEFAULT_REACH_MODE_LIBRARY
    from rexpolicy.stage0.envs.state_schema import DEFAULT_STAGE0_STATE_SCHEMA

    expected_manifest_keys = {
        "corpus_config_sha256",
        "library_sha256",
        "mode_hashes",
        "mode_ids",
        "schema_id",
        "stage0_experiment_sha256",
        "trajectory_modes",
    }
    if set(manifest) != expected_manifest_keys:
        raise RuntimeError("multimode corpus manifest keys changed")
    if manifest["mode_ids"] != list(config.corpus.mode_ids):
        raise RuntimeError("multimode corpus mode IDs differ from the run contract")
    if manifest["corpus_config_sha256"] != canonical_fingerprint(
        config.corpus.to_record()
    ):
        raise RuntimeError("multimode corpus config hash differs from the run contract")
    if manifest["stage0_experiment_sha256"] != config.stage0.fingerprint:
        raise RuntimeError("multimode corpus Stage 0 experiment hash changed")
    if manifest["library_sha256"] != DEFAULT_REACH_MODE_LIBRARY.sha256:
        raise RuntimeError("multimode corpus library hash changed")
    trajectory_modes = manifest["trajectory_modes"]
    if not isinstance(trajectory_modes, Mapping):
        raise RuntimeError("multimode trajectory map must be an object")
    identifiers = {item.provenance.trajectory_id for item in trajectories}
    if set(trajectory_modes) != identifiers:
        raise RuntimeError("multimode trajectory map does not match committed shards")
    expected_count = config.corpus.reset_groups * len(config.corpus.mode_ids)
    if len(trajectories) != expected_count:
        raise RuntimeError(
            f"multimode corpus has {len(trajectories)} trajectories; expected {expected_count}"
        )
    groups: dict[str, list[Any]] = {}
    successful_modes: dict[str, set[str]] = {}
    outcome_counts = {outcome.value: 0 for outcome in Stage0Outcome}
    mode_hashes = manifest["mode_hashes"]
    if not isinstance(mode_hashes, Mapping) or set(mode_hashes) != set(
        config.corpus.mode_ids
    ):
        raise RuntimeError("multimode corpus mode hashes do not match mode IDs")
    expected_mode_hashes = {
        mode_id: DEFAULT_REACH_MODE_LIBRARY.mode_hash(mode_id)
        for mode_id in config.corpus.mode_ids
    }
    if dict(mode_hashes) != expected_mode_hashes:
        raise RuntimeError("multimode corpus mode definitions changed")
    for trajectory in trajectories:
        if trajectory.state_dim != config.stage0.state_dim:
            raise RuntimeError("multimode corpus state dimension changed")
        if trajectory.action_dim != config.stage0.action_dim:
            raise RuntimeError("multimode corpus action dimension changed")
        provenance = trajectory.provenance
        if provenance.experiment_sha256 != config.stage0.fingerprint:
            raise RuntimeError("trajectory experiment hash changed")
        groups.setdefault(provenance.reset_group_id, []).append(trajectory)
        record = trajectory_modes[provenance.trajectory_id]
        if not isinstance(record, Mapping) or set(record) != {
            "mode_id",
            "mode_sha256",
            "reset_group_id",
        }:
            raise RuntimeError("trajectory mode record keys changed")
        mode_id = record["mode_id"]
        if (
            mode_id not in config.corpus.mode_ids
            or record["mode_sha256"] != mode_hashes[mode_id]
            or record["reset_group_id"] != provenance.reset_group_id
        ):
            raise RuntimeError("trajectory mode provenance changed")
        if (
            _safe_mode_slug(mode_id) not in provenance.trajectory_id
            or f"h{record['mode_sha256'][:12]}" not in provenance.trajectory_id
        ):
            raise RuntimeError("trajectory ID does not bind its mode hash")
        if trajectory.is_success:
            successful_modes.setdefault(provenance.reset_group_id, set()).add(mode_id)
        outcome_counts[trajectory.outcome.value] += 1
    if len(groups) != config.corpus.reset_groups:
        raise RuntimeError("multimode corpus reset-group count changed")
    eef_slice = DEFAULT_STAGE0_STATE_SCHEMA.slice("eef_position")
    object_slice = DEFAULT_STAGE0_STATE_SCHEMA.slice("object_position")
    contact_slice = DEFAULT_STAGE0_STATE_SCHEMA.slice("has_hand_contact")
    maximum_initial_state_delta = 0.0
    maximum_object_displacement = 0.0
    contact_state_count = 0
    pairwise_eef_rms: list[float] = []
    pairwise_eef_rms_by_mode_pair: dict[tuple[str, str], list[float]] = {}
    fully_successful_groups = 0
    for group_id, items in groups.items():
        if len(items) != len(config.corpus.mode_ids):
            raise RuntimeError(f"reset group {group_id} does not contain every mode")
        seeds = {item.provenance.reset_seed for item in items}
        modes = {
            trajectory_modes[item.provenance.trajectory_id]["mode_id"] for item in items
        }
        if len(seeds) != 1 or modes != set(config.corpus.mode_ids):
            raise RuntimeError(f"reset group {group_id} is not a same-reset mode set")
        if len(successful_modes.get(group_id, set())) < (
            config.corpus.minimum_success_modes_per_reset
        ):
            raise RuntimeError(
                f"reset group {group_id} has too few verified successful modes"
            )
        reference = items[0].states[0]
        maximum_initial_state_delta = max(
            maximum_initial_state_delta,
            max(float((item.states[0] - reference).abs().max()) for item in items),
        )
        if all(item.is_success for item in items):
            fully_successful_groups += 1
        successful = [item for item in items if item.is_success]
        for item in items:
            displacement = torch.linalg.vector_norm(
                item.states[:, object_slice] - item.states[0, object_slice],
                dim=-1,
            )
            maximum_object_displacement = max(
                maximum_object_displacement,
                float(displacement.max()),
            )
            contact_state_count += int(
                (item.states[:, contact_slice].squeeze(-1) > 0.5).sum()
            )
        for left_index, left in enumerate(successful):
            for right in successful[left_index + 1 :]:
                common = min(left.states.shape[0], right.states.shape[0])
                delta = (
                    left.states[:common, eef_slice] - right.states[:common, eef_slice]
                )
                rms = float(delta.square().sum(dim=-1).mean().sqrt())
                pairwise_eef_rms.append(rms)
                left_mode = trajectory_modes[left.provenance.trajectory_id]["mode_id"]
                right_mode = trajectory_modes[right.provenance.trajectory_id]["mode_id"]
                pair = tuple(sorted((left_mode, right_mode)))
                pairwise_eef_rms_by_mode_pair.setdefault(pair, []).append(rms)
    if maximum_initial_state_delta > 1.0e-6:
        raise RuntimeError("same-reset mode initial states differ")
    if contact_state_count:
        raise RuntimeError("verified success corpus contains hand-object contact")
    if maximum_object_displacement > 0.005:
        raise RuntimeError("verified success corpus displaced the object above 5 mm")
    mean_pairwise_eef_rms = {
        "::".join(pair): sum(values) / len(values)
        for pair, values in sorted(pairwise_eef_rms_by_mode_pair.items())
    }
    if (
        not pairwise_eef_rms
        or min(pairwise_eef_rms) < 0.025
        or min(mean_pairwise_eef_rms.values()) < 0.030
    ):
        raise RuntimeError("same-reset success paths are not geometrically distinct")
    fully_successful_fraction = fully_successful_groups / len(groups)
    if fully_successful_fraction < 0.95:
        raise RuntimeError("fewer than 95% of reset groups contain all success modes")
    mode_success_counts = {
        mode_id: sum(
            item.is_success
            and trajectory_modes[item.provenance.trajectory_id]["mode_id"] == mode_id
            for item in trajectories
        )
        for mode_id in config.corpus.mode_ids
    }
    if any(
        count / config.corpus.reset_groups < 0.98
        for count in mode_success_counts.values()
    ):
        raise RuntimeError("a Reach mode has less than 98% verified success")
    return {
        "contact_state_count": contact_state_count,
        "fully_successful_group_fraction": fully_successful_fraction,
        "maximum_initial_state_delta": maximum_initial_state_delta,
        "maximum_object_displacement_m": maximum_object_displacement,
        "minimum_pairwise_eef_rms_m": min(pairwise_eef_rms),
        "minimum_mode_pair_mean_eef_rms_m": min(mean_pairwise_eef_rms.values()),
        "mode_pair_mean_eef_rms_m": mean_pairwise_eef_rms,
        "mode_success_counts": mode_success_counts,
        "outcome_counts": outcome_counts,
        "mode_manifest_sha256": canonical_fingerprint(manifest),
        "reset_group_count": len(groups),
        "trajectory_count": len(trajectories),
        "verified_same_reset_multimode": True,
    }


def _materialize_existing_corpus(
    config: Stage0LongRunConfig,
    source: Path,
    target: Path,
) -> None:
    from rexpolicy.stage0.data import write_trajectory_shard

    trajectories, manifest = _load_corpus(source)
    _verify_multimode_corpus(config, trajectories, manifest)
    partial = target.with_name(f".corpus-partial-{uuid.uuid4().hex}")
    partial.mkdir(parents=True)
    try:
        for trajectory in sorted(
            trajectories,
            key=lambda item: item.provenance.trajectory_id,
        ):
            write_trajectory_shard(partial, trajectory)
        atomic_write_json(partial / "corpus-modes.json", manifest)
        source_summary = source / "collection-summary.json"
        if source_summary.is_file():
            shutil.copy2(source_summary, partial / "collection-summary.json")
        _publish_corpus_directory(partial, target)
    except Exception:
        # Retain the hidden partial for forensic inspection; it is never read as
        # a committed corpus because only the final directory is authoritative.
        raise


def _ensure_corpus(
    config: Stage0LongRunConfig,
    output_dir: Path,
    *,
    source: Path | None,
    device: str,
    capture_graph: bool,
) -> tuple[tuple[Any, ...], dict[str, Any], dict[str, Any]]:
    corpus_dir = output_dir / "corpus"
    if not corpus_dir.exists():
        if source is None:
            collection = _collect_multimode_corpus(
                config,
                corpus_dir,
                device_name=device,
                capture_graph=capture_graph,
            )
            collection_record = collection.to_record()
        else:
            if source.expanduser().resolve() == corpus_dir.resolve():
                raise RuntimeError("--corpus-dir cannot name an absent output corpus")
            _materialize_existing_corpus(
                config, source.expanduser().resolve(), corpus_dir
            )
            collection_record = {"authority": "materialized_existing_corpus"}
    else:
        collection_record = {"authority": "existing_output_corpus"}
    trajectories, manifest = _load_corpus(corpus_dir)
    verification = _verify_multimode_corpus(config, trajectories, manifest)
    atomic_write_json(
        output_dir / "corpus-verification.json",
        {**verification, "collection": collection_record},
    )
    return trajectories, manifest, verification


def _sample_windows(
    windows: Sequence[Any],
    *,
    batch_size: int,
    generator: Any,
) -> list[Any]:
    import torch

    if not windows:
        raise RuntimeError("Stage 0 training split has no success windows")
    indices = torch.randint(len(windows), (batch_size,), generator=generator).tolist()
    return [windows[index] for index in indices]


def _sample_contrastive_windows(
    windows: Sequence[Any],
    trajectory_modes: Mapping[str, Mapping[str, str]],
    *,
    batch_size: int,
    generator: Any,
) -> list[Any]:
    import torch

    by_group: dict[str, dict[str, list[Any]]] = {}
    for window in windows:
        mode_id = trajectory_modes[window.trajectory_id]["mode_id"]
        by_group.setdefault(window.reset_group_id, {}).setdefault(
            mode_id,
            [],
        ).append(window)
    eligible = [
        {
            mode_id: sorted(rows, key=lambda item: item.start)
            for mode_id, rows in modes.items()
            if len(rows) >= 2
        }
        for _, modes in sorted(by_group.items())
    ]
    eligible = [modes for modes in eligible if len(modes) >= 2]
    if len(eligible) < 2:
        raise RuntimeError(
            "manifold training requires two reset groups sharing two success modes"
        )
    compatible_pairs = [
        (left_group, right_group, tuple(sorted(set(left_group) & set(right_group))))
        for left_index, left_group in enumerate(eligible)
        for right_group in eligible[left_index + 1 :]
        if len(set(left_group) & set(right_group)) >= 2
    ]
    if not compatible_pairs:
        raise RuntimeError(
            "manifold training requires a reset-group pair sharing two success modes"
        )
    if batch_size % 8:
        raise ValueError("manifold batch_size must be divisible by eight")
    pair_order = torch.randperm(
        len(compatible_pairs),
        generator=generator,
    ).tolist()
    result: list[Any] = []
    for octet_index in range(batch_size // 8):
        left_group, right_group, common_modes = compatible_pairs[
            pair_order[octet_index % len(pair_order)]
        ]
        mode_order = torch.randperm(
            len(common_modes),
            generator=generator,
        ).tolist()
        for mode_index in mode_order[:2]:
            mode_id = common_modes[mode_index]
            left_rows = left_group[mode_id]
            right_rows = right_group[mode_id]
            start = int(
                torch.randint(
                    min(len(left_rows), len(right_rows)) - 1,
                    (1,),
                    generator=generator,
                )
            )
            result.extend(
                (
                    left_rows[start],
                    left_rows[start + 1],
                    right_rows[start],
                    right_rows[start + 1],
                )
            )
    return result


def _generator(torch: Any, *, seed: int, device: Any = "cpu") -> Any:
    return torch.Generator(device=device).manual_seed(seed)


def _step_seed(base_seed: int, phase: str, stream: str, local_step: int) -> int:
    from rexpolicy.stage0.seed import derive_stage0_seed

    return derive_stage0_seed(
        base_seed,
        phase,
        stream,
        local_step,
        namespace="stage0-long-run",
    )


def _normalize_batch(batch: Any, normalization: Any) -> Any:
    from rexpolicy.stage0.envs.state_schema import DEFAULT_STAGE0_STATE_SCHEMA

    eef_slice = DEFAULT_STAGE0_STATE_SCHEMA.slice("eef_position")
    previous_eef = __import__("torch").cat(
        (
            batch.current_states[:, None, eef_slice],
            batch.future_states[:, :-1, eef_slice],
        ),
        dim=1,
    )
    relative_future_actions = batch.future_actions.clone()
    relative_future_actions[:, :, :3].sub_(previous_eef)
    relative_action_chunks = relative_future_actions[
        :, : batch.action_chunks.shape[1]
    ].clone()
    normalized_current_states = normalization.normalize_states(batch.current_states)
    normalized_future_states = normalization.normalize_states(
        batch.future_states,
        valid_mask=batch.future_mask,
    )
    relative_future_states = (
        normalized_future_states - (normalized_current_states[:, None, :])
    )
    relative_future_states.masked_fill_(
        ~batch.future_mask.unsqueeze(-1),
        0.0,
    )
    return replace(
        batch,
        current_states=normalized_current_states,
        action_chunks=normalization.normalize_actions(
            relative_action_chunks,
            valid_mask=batch.action_mask,
        ),
        future_actions=normalization.normalize_actions(
            relative_future_actions,
            valid_mask=batch.future_mask,
        ),
        future_states=relative_future_states,
    )


def _encode_windows(
    encoder: Any,
    windows: Sequence[Any],
    *,
    device: Any,
    normalization: Any,
):
    from rexpolicy.stage0.trainers import collate_success_windows

    batch = _normalize_batch(collate_success_windows(windows), normalization).to(device)
    encoder.eval()
    with __import__("torch").no_grad():
        encoding = encoder(
            batch.future_states,
            batch.future_actions,
            padding_mask=batch.future_padding_mask,
        )
    return batch, encoding.latent


def _window_key(window: Any) -> tuple[str, int]:
    return (window.trajectory_id, window.start)


def _precompute_window_latents(
    encoder: Any,
    windows: Sequence[Any],
    *,
    batch_size: int,
    device: Any,
    normalization: Any,
) -> dict[tuple[str, int], Any]:
    import torch

    cache: dict[tuple[str, int], Any] = {}
    encoder.eval()
    with torch.inference_mode():
        for start in range(0, len(windows), batch_size):
            items = windows[start : start + batch_size]
            _, latents = _encode_windows(
                encoder,
                items,
                device=device,
                normalization=normalization,
            )
            for window, latent in zip(items, latents.detach().float().cpu()):
                key = _window_key(window)
                if key in cache:
                    raise RuntimeError("training windows repeat a trajectory/start key")
                cache[key] = latent.contiguous()
    if len(cache) != len(windows):
        raise RuntimeError("precomputed latent cache omitted training windows")
    return cache


def _cached_window_batch(
    windows: Sequence[Any],
    latent_cache: Mapping[tuple[str, int], Any],
    *,
    device: Any,
    normalization: Any,
) -> tuple[Any, Any]:
    import torch

    from rexpolicy.stage0.trainers import collate_success_windows

    batch = _normalize_batch(collate_success_windows(windows), normalization).to(device)
    try:
        latents = torch.stack(
            tuple(latent_cache[_window_key(window)] for window in windows)
        ).to(device=device, dtype=batch.current_states.dtype)
    except KeyError as error:
        raise RuntimeError("training latent cache is incomplete") from error
    return batch, latents


def _warm_start_selector(
    selector: Any,
    windows: Sequence[Any],
    latent_cache: Mapping[tuple[str, int], Any],
    trajectory_modes: Mapping[str, Mapping[str, str]],
    mode_ids: Sequence[str],
) -> dict[str, Any]:
    import torch

    rows: dict[str, list[Any]] = {mode_id: [] for mode_id in mode_ids}
    for window in windows:
        if window.start != 0:
            continue
        mode_id = trajectory_modes[window.trajectory_id]["mode_id"]
        rows[mode_id].append(latent_cache[_window_key(window)])
    if any(len(values) < 2 for values in rows.values()):
        raise RuntimeError("selector warm start requires two train resets per mode")
    centroids = torch.stack(
        tuple(torch.stack(rows[mode_id]).mean(dim=0) for mode_id in mode_ids)
    )
    scales = torch.stack(
        tuple(
            torch.stack(rows[mode_id]).std(dim=0, unbiased=False)
            for mode_id in mode_ids
        )
    ).clamp(min=0.05, max=1.0)
    if selector.components != len(mode_ids):
        raise RuntimeError("selector components must equal configured success modes")
    with torch.no_grad():
        selector.logits_head.weight.zero_()
        selector.logits_head.bias.zero_()
        selector.mean_head.weight.zero_()
        selector.mean_head.bias.copy_(
            centroids.flatten().to(
                device=selector.mean_head.bias.device,
                dtype=selector.mean_head.bias.dtype,
            )
        )
        selector.log_std_head.weight.zero_()
        selector.log_std_head.bias.copy_(
            scales.log()
            .flatten()
            .to(
                device=selector.log_std_head.bias.device,
                dtype=selector.log_std_head.bias.dtype,
            )
        )
    return {
        "mode_counts": {mode_id: len(rows[mode_id]) for mode_id in mode_ids},
        "mode_ids": list(mode_ids),
        "protocol": _SELECTOR_DIAGNOSTIC_PROTOCOL["warm_start"],
    }


def _fit_normalization(trajectory_splits: Any, output_dir: Path) -> Any:
    import torch

    from rexpolicy.stage0.normalization import (
        fit_stage0_normalization,
        read_stage0_normalization,
        write_stage0_normalization,
    )
    from rexpolicy.stage0.envs.state_schema import DEFAULT_STAGE0_STATE_SCHEMA

    successful = tuple(item for item in trajectory_splits.train if item.is_success)
    if len(successful) < 2:
        raise RuntimeError("normalization requires two successful train trajectories")
    states = torch.cat(tuple(item.states for item in successful), dim=0)
    eef_slice = DEFAULT_STAGE0_STATE_SCHEMA.slice("eef_position")
    relative_action_rows = []
    for trajectory in successful:
        relative = trajectory.actions.clone()
        relative[:, :3].sub_(trajectory.states[:-1, eef_slice])
        relative[:, 3:].zero_()
        relative_action_rows.append(relative)
    actions = torch.cat(tuple(relative_action_rows), dim=0)
    effective_action_mask = torch.zeros(actions.shape[1], dtype=torch.bool)
    effective_action_mask[:3] = True
    fitted = fit_stage0_normalization(
        states,
        actions,
        effective_action_mask=effective_action_mask,
    )
    path = output_dir / "normalization.json"
    if path.exists():
        persisted = read_stage0_normalization(path)
        if persisted.sha256 != fitted.sha256:
            raise RuntimeError("persisted train-split normalization hash changed")
        return persisted
    write_stage0_normalization(path, fitted)
    return fitted


def _create_training_components(
    config: Stage0LongRunConfig,
    *,
    device: Any,
    action_feature_mask: Any,
):
    import torch

    from rexpolicy.stage0.models import (
        FutureTrajectoryEncoder,
        MixtureSuccessModeSelector,
    )
    from rexpolicy.stage0.runtime import Stage0FlowPolicy
    from rexpolicy.stage0.trainers import (
        Stage0ManifoldTrainer,
        Stage0ManifoldTrainerConfig,
        Stage0PolicyTrainer,
        Stage0SelectorTrainer,
    )

    stage0 = config.stage0
    torch.manual_seed(config.runtime.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed(config.runtime.seed)
    encoder = FutureTrajectoryEncoder(
        stage0.state_dim,
        stage0.action_dim,
        latent_dim=stage0.latent_dim,
        projection_dim=stage0.latent_dim,
        model_dim=stage0.model_dim,
        max_future_steps=stage0.future_horizon,
        transformer_layers=stage0.transformer_layers,
        attention_heads=stage0.attention_heads,
        feedforward_dim=stage0.feedforward_dim,
        dropout=stage0.dropout,
    ).to(device)
    selector = MixtureSuccessModeSelector(
        stage0.state_dim,
        stage0.latent_dim,
        components=len(config.corpus.mode_ids),
        hidden_dim=stage0.model_dim,
        layers=2,
    ).to(device)

    def policy() -> Any:
        return Stage0FlowPolicy(
            state_dim=stage0.state_dim,
            latent_dim=stage0.latent_dim,
            action_dim=stage0.action_dim,
            action_horizon=stage0.action_horizon,
            model_dim=stage0.model_dim,
            transformer_layers=stage0.transformer_layers,
            attention_heads=stage0.attention_heads,
            feedforward_dim=stage0.feedforward_dim,
            dropout=stage0.dropout,
        ).to(device)

    conditional_policy = policy()
    no_z_policy = policy()
    no_z_policy.load_state_dict(conditional_policy.state_dict(), strict=True)
    models = {
        "conditional_policy": conditional_policy,
        "future_encoder": encoder,
        "no_z_policy": no_z_policy,
        "selector": selector,
    }
    optimizers = {
        name: torch.optim.AdamW(
            (parameter for parameter in model.parameters() if parameter.requires_grad),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
        )
        for name, model in models.items()
    }
    trainers = {
        "future_encoder": Stage0ManifoldTrainer(
            encoder,
            optimizers["future_encoder"],
            config=Stage0ManifoldTrainerConfig(**_MANIFOLD_TRAINER_CONFIG),
        ),
        "selector": Stage0SelectorTrainer(selector, optimizers["selector"]),
        "conditional_policy": Stage0PolicyTrainer(
            conditional_policy,
            optimizers["conditional_policy"],
            action_feature_mask=action_feature_mask,
        ),
        "no_z_policy": Stage0PolicyTrainer(
            no_z_policy,
            optimizers["no_z_policy"],
            action_feature_mask=action_feature_mask,
        ),
    }
    return models, optimizers, trainers


def _masked_mse(
    prediction: Any,
    target: Any,
    mask: Any,
    feature_mask: Any,
) -> float:
    value_mask = mask.unsqueeze(-1) & feature_mask.reshape(1, 1, -1)
    return float((prediction - target).square().masked_select(value_mask).mean())


def _select_complete_mode_groups(
    windows: Sequence[Any],
    trajectory_modes: Mapping[str, Mapping[str, str]],
    mode_ids: Sequence[str],
    *,
    maximum_windows: int,
) -> list[Any]:
    by_group: dict[str, dict[str, Any]] = {}
    expected_modes = tuple(mode_ids)
    for window in windows:
        if window.start != 0:
            continue
        mode_id = trajectory_modes[window.trajectory_id]["mode_id"]
        group = by_group.setdefault(window.reset_group_id, {})
        if mode_id in group:
            raise RuntimeError("held-out reset group repeats a start-zero mode")
        group[mode_id] = window
    complete = [
        (group_id, group)
        for group_id, group in sorted(by_group.items())
        if set(group) == set(expected_modes)
    ]
    maximum_groups = maximum_windows // len(expected_modes)
    if len(complete) < 3 or maximum_groups < 3:
        raise RuntimeError(
            "held-out diagnostics require at least three complete reset-mode groups"
        )
    selected_count = min(len(complete), maximum_groups)
    indices = [
        index * len(complete) // selected_count for index in range(selected_count)
    ]
    return [
        complete[index][1][mode_id] for index in indices for mode_id in expected_modes
    ]


def _evaluate_offline(
    config: Stage0LongRunConfig,
    models: Mapping[str, Any],
    windows: Sequence[Any],
    trajectory_modes: Mapping[str, Mapping[str, str]],
    *,
    device: Any,
    global_step: int,
    normalization: Any,
    split_name: str,
) -> dict[str, Any]:
    import torch

    from rexpolicy.stage0.envs.state_schema import DEFAULT_STAGE0_STATE_SCHEMA
    from rexpolicy.stage0.evaluation import analyze_latent_modes, analyze_latents

    selected = _select_complete_mode_groups(
        windows,
        trajectory_modes,
        config.corpus.mode_ids,
        maximum_windows=config.runtime.eval_window_count,
    )
    encoder = models["future_encoder"]
    selector = models["selector"]
    conditional_policy = models["conditional_policy"]
    no_z_policy = models["no_z_policy"]
    for model in models.values():
        model.eval()
    batch, oracle_latent = _encode_windows(
        encoder,
        selected,
        device=device,
        normalization=normalization,
    )
    with torch.no_grad():
        distribution = selector(batch.current_states)
        selector_latent = distribution.sample(
            generator=_generator(
                torch,
                seed=_step_seed(
                    config.runtime.seed,
                    "offline_evaluation",
                    "selector_latent",
                    global_step,
                ),
                device=device,
            )
        )
        latent_record = analyze_latents(oracle_latent).to_record()
        mode_ids = tuple(
            trajectory_modes[window.trajectory_id]["mode_id"] for window in selected
        )
        reset_xy = torch.stack(
            tuple(
                window.current_state[
                    DEFAULT_STAGE0_STATE_SCHEMA.slice("object_position")
                ][:2]
                for window in selected
            )
        )
        mode_record = analyze_latent_modes(
            oracle_latent,
            mode_ids,
            tuple(window.reset_group_id for window in selected),
            reset_xy=reset_xy,
            ridge_alpha=_MODE_DIAGNOSTIC_PROTOCOL["ridge_alpha"],
        ).to_record()
        mode_geometry = mode_record["geometry"]
        mode_centroid = mode_record["nearest_centroid"]
        reset_probe = mode_record["reset_xy_linear_probe"]
        mode_gate = bool(
            mode_centroid["macro_f1"] >= _MODE_DIAGNOSTIC_PROTOCOL["macro_f1_minimum"]
            and mode_centroid["normalized_mutual_information"]
            >= _MODE_DIAGNOSTIC_PROTOCOL["classification_nmi_minimum"]
            and mode_geometry["euclidean_silhouette"]
            >= _MODE_DIAGNOSTIC_PROTOCOL["euclidean_silhouette_minimum"]
            and mode_geometry["mode_to_reset_distance_ratio"]
            >= _MODE_DIAGNOSTIC_PROTOCOL["mode_to_reset_distance_ratio_minimum"]
            and reset_probe is not None
            and reset_probe["r2"]
            <= _MODE_DIAGNOSTIC_PROTOCOL["reset_xy_linear_probe_r2_maximum"]
        )
        control_count = min(32, batch.batch_size)
        state = batch.current_states[:control_count]
        oracle = oracle_latent[:control_count]
        selected_latent = selector_latent[:control_count]
        action_mask = batch.action_mask[:control_count]
        target_actions = batch.action_chunks[:control_count]
        feature_mask = normalization.effective_action_mask.to(device=device)
        noise_generator = _generator(
            torch,
            seed=_step_seed(
                config.runtime.seed,
                "offline_evaluation",
                "initial_noise",
                global_step,
            ),
            device=device,
        )
        initial_noise = torch.randn(
            target_actions.shape,
            dtype=target_actions.dtype,
            device=device,
            generator=noise_generator,
        )
        no_z_actions = no_z_policy.sample(
            state,
            success_latent=torch.zeros_like(oracle),
            steps=config.runtime.flow_sample_steps,
            action_mask=action_mask,
            action_feature_mask=feature_mask,
            initial_noise=initial_noise,
        )
        oracle_actions = conditional_policy.sample(
            state,
            success_latent=oracle,
            steps=config.runtime.flow_sample_steps,
            action_mask=action_mask,
            action_feature_mask=feature_mask,
            initial_noise=initial_noise,
        )
        selector_actions = conditional_policy.sample(
            state,
            success_latent=selected_latent,
            steps=config.runtime.flow_sample_steps,
            action_mask=action_mask,
            action_feature_mask=feature_mask,
            initial_noise=initial_noise,
        )
        no_z_mse = _masked_mse(
            no_z_actions,
            target_actions,
            action_mask,
            feature_mask,
        )
        oracle_mse = _masked_mse(
            oracle_actions,
            target_actions,
            action_mask,
            feature_mask,
        )
        selector_mse = _masked_mse(
            selector_actions,
            target_actions,
            action_mask,
            feature_mask,
        )
        physical_target = normalization.denormalize_actions(
            target_actions,
            valid_mask=action_mask,
        )
        physical_no_z = normalization.denormalize_actions(
            no_z_actions,
            valid_mask=action_mask,
        )
        physical_oracle = normalization.denormalize_actions(
            oracle_actions,
            valid_mask=action_mask,
        )
        physical_selector = normalization.denormalize_actions(
            selector_actions,
            valid_mask=action_mask,
        )
        selector_nll = float(distribution.negative_log_likelihood(oracle_latent))
        global_mean = oracle_latent.mean(dim=0)
        global_std = oracle_latent.std(dim=0, unbiased=False).clamp_min(1.0e-4)
        global_gaussian_nll = float(
            (
                0.5 * ((oracle_latent - global_mean) / global_std).square()
                + global_std.log()
                + 0.5 * math.log(2.0 * math.pi)
            )
            .sum(dim=-1)
            .mean()
        )
        mode_count = len(config.corpus.mode_ids)
        group_count = len(selected) // mode_count
        component_means = distribution.component_means[::mode_count]
        oracle_modes = oracle_latent.reshape(group_count, mode_count, -1)
        component_mode_coverages: list[int] = []
        maximum_mode_to_component_distances: list[float] = []
        normalized_mode_to_component_distances: list[float] = []
        for group_index in range(group_count):
            distances = torch.cdist(
                component_means[group_index],
                oracle_modes[group_index],
            )
            component_mode_coverages.append(
                int(distances.argmin(dim=1).unique().numel())
            )
            maximum_mode_to_component_distances.append(
                float(distances.min(dim=0).values.max())
            )
            oracle_distances = torch.cdist(
                oracle_modes[group_index],
                oracle_modes[group_index],
            )
            minimum_mode_separation = oracle_distances.masked_fill(
                torch.eye(mode_count, dtype=torch.bool, device=device),
                float("inf"),
            ).min()
            normalized_mode_to_component_distances.append(
                float(
                    distances.min(dim=0).values.max()
                    / minimum_mode_separation.clamp_min(1.0e-6)
                )
            )
        average_component_mode_coverage = sum(component_mode_coverages) / len(
            component_mode_coverages
        )
        component_probabilities = distribution.weights[::mode_count]
        minimum_component_probabilities = component_probabilities.min(dim=1).values
        selector_gate = bool(
            distribution.logits.shape[1] == mode_count
            and all(coverage == mode_count for coverage in component_mode_coverages)
            and max(normalized_mode_to_component_distances)
            <= _SELECTOR_DIAGNOSTIC_PROTOCOL[
                "maximum_normalized_mode_to_component_distance"
            ]
            and float(minimum_component_probabilities.min())
            >= _SELECTOR_DIAGNOSTIC_PROTOCOL["minimum_component_probability_per_reset"]
            and selector_nll < global_gaussian_nll
        )
        selector_cosine = float(
            torch.nn.functional.cosine_similarity(
                selector_latent,
                oracle_latent,
            ).mean()
        )
    return {
        "authority": "offline_diagnostic_only",
        "conditional_and_no_z_equal_training_budget": (
            config.training.conditional_policy_steps
            == config.training.no_z_policy_steps
        ),
        "global_step": global_step,
        "latent": latent_record,
        "latent_mode_gate_passed": mode_gate,
        "mode_diagnostic_protocol": dict(_MODE_DIAGNOSTIC_PROTOCOL),
        "mode_diagnostics": mode_record,
        "not_a_rollout_success_claim": True,
        "policy_controls": {
            "control_window_count": control_count,
            "effective_action_dimensions": 3,
            "no_z_action_mse": no_z_mse,
            "no_z_xyz_mse_m2": _masked_mse(
                physical_no_z,
                physical_target,
                action_mask,
                feature_mask,
            ),
            "oracle_z_action_mse": oracle_mse,
            "oracle_z_xyz_mse_m2": _masked_mse(
                physical_oracle,
                physical_target,
                action_mask,
                feature_mask,
            ),
            "oracle_z_improves_over_no_z": oracle_mse < no_z_mse,
            "selector_z_action_mse": selector_mse,
            "selector_z_xyz_mse_m2": _masked_mse(
                physical_selector,
                physical_target,
                action_mask,
                feature_mask,
            ),
            "selector_z_improves_over_no_z": selector_mse < no_z_mse,
        },
        "rollout_gate_evaluated": False,
        "selector": {
            "average_component_mode_coverage": (average_component_mode_coverage),
            "component_count": int(distribution.logits.shape[1]),
            "component_mode_coverages": component_mode_coverages,
            "cosine_alignment": selector_cosine,
            "distribution_family": "diagonal_gaussian_mixture",
            "gate_passed": selector_gate,
            "global_diagonal_gaussian_nll": global_gaussian_nll,
            "maximum_mode_to_component_distance_by_reset": (
                maximum_mode_to_component_distances
            ),
            "minimum_component_probability_by_reset": (
                minimum_component_probabilities.tolist()
            ),
            "mean_weight_entropy": float(
                -(distribution.weights * distribution.log_weights).sum(dim=-1).mean()
            ),
            "negative_log_likelihood": selector_nll,
            "normalized_mode_to_component_distance_by_reset": (
                normalized_mode_to_component_distances
            ),
            "protocol": dict(_SELECTOR_DIAGNOSTIC_PROTOCOL),
        },
        "split": split_name,
        "window_count": len(selected),
        "reset_group_count": len(selected) // len(config.corpus.mode_ids),
        "window_selection": "held_out_reset_groups_start_zero/v1",
    }


def _evaluation_summary(record: Mapping[str, Any]) -> dict[str, Any]:
    latent = record["latent"]
    controls = record["policy_controls"]
    return {
        "collapse_gate_passed": latent["collapse_gate_passed"],
        "effective_rank": latent["effective_rank"],
        "latent_mode_gate_passed": record["latent_mode_gate_passed"],
        "mode_macro_f1": record["mode_diagnostics"]["nearest_centroid"]["macro_f1"],
        "no_z_action_mse": controls["no_z_action_mse"],
        "oracle_z_action_mse": controls["oracle_z_action_mse"],
        "selector_nll": record["selector"]["negative_log_likelihood"],
        "selector_gate_passed": record["selector"]["gate_passed"],
        "selector_z_action_mse": controls["selector_z_action_mse"],
    }


def _evaluate_newton_rollouts(
    config: Stage0LongRunConfig,
    models: Mapping[str, Any],
    windows: Sequence[Any],
    trajectories: Sequence[Any],
    trajectory_modes: Mapping[str, Mapping[str, str]],
    *,
    device_name: str,
    normalization: Any,
) -> dict[str, Any]:
    import torch

    from rexpolicy.envs.groot_newton_env import GrootNewtonEnv
    from rexpolicy.stage0.envs.oracles import ReachSuccessOracle
    from rexpolicy.stage0.envs.state_only import Stage0StateOnlyEnv
    from rexpolicy.stage0.envs.state_schema import DEFAULT_STAGE0_STATE_SCHEMA
    from rexpolicy.stage0.evaluation import (
        ReachRolloutBudget,
        analyze_path_adherence,
        rollout_reach_policy,
    )

    selected = _select_complete_mode_groups(
        windows,
        trajectory_modes,
        config.corpus.mode_ids,
        maximum_windows=min(
            config.runtime.eval_window_count,
            _ROLLOUT_PROTOCOL["maximum_evaluation_windows"],
        ),
    )
    trajectory_by_id = {item.provenance.trajectory_id: item for item in trajectories}
    reset_seeds = tuple(
        trajectory_by_id[window.trajectory_id].provenance.reset_seed
        for window in selected
    )
    noise_seeds = tuple(
        _step_seed(
            config.runtime.seed,
            "final_test_rollout",
            "flow_noise",
            index,
        )
        for index in range(len(selected))
    )
    budget = ReachRolloutBudget(
        reset_group_ids=tuple(window.reset_group_id for window in selected),
        reset_seeds=reset_seeds,
        noise_seeds=noise_seeds,
        max_steps_per_episode=config.stage0.max_episode_steps,
    )
    device = torch.device(device_name)
    _, oracle_latents = _encode_windows(
        models["future_encoder"],
        selected,
        device=device,
        normalization=normalization,
    )
    mode_count = len(config.corpus.mode_ids)
    permutation = tuple(
        group_start + (mode_index + 1) % mode_count
        for group_start in range(0, len(selected), mode_count)
        for mode_index in range(mode_count)
    )
    env_config = replace(
        _environment_config(config, device_name, capture_graph=False),
        num_envs=len(selected),
    )
    env = GrootNewtonEnv(env_config)
    adapter = Stage0StateOnlyEnv(env)

    def oracle() -> Any:
        return ReachSuccessOracle(
            len(selected),
            success_threshold_m=env_config.reach_success_threshold,
            success_hold_steps=env_config.reach_success_hold_steps,
            displacement_limit_m=env_config.reach_bottle_displacement_limit,
        )

    try:
        results = {
            "no-z": rollout_reach_policy(
                adapter,
                models["no_z_policy"],
                normalization,
                budget,
                mode="no-z",
                oracle=oracle(),
                flow_sample_steps=config.runtime.flow_sample_steps,
            ),
            "oracle-z": rollout_reach_policy(
                adapter,
                models["conditional_policy"],
                normalization,
                budget,
                mode="oracle-z",
                oracle=oracle(),
                oracle_latents=oracle_latents,
                flow_sample_steps=config.runtime.flow_sample_steps,
            ),
            "selector-z": rollout_reach_policy(
                adapter,
                models["conditional_policy"],
                normalization,
                budget,
                mode="selector-z",
                oracle=oracle(),
                selector=models["selector"],
                selector_generator=_generator(
                    torch,
                    seed=_step_seed(
                        config.runtime.seed,
                        "final_test_rollout",
                        "selector_latent",
                        0,
                    ),
                    device=device,
                ),
                flow_sample_steps=config.runtime.flow_sample_steps,
            ),
            "permuted-z": rollout_reach_policy(
                adapter,
                models["conditional_policy"],
                normalization,
                budget,
                mode="permuted-z",
                oracle=oracle(),
                oracle_latents=oracle_latents,
                latent_permutation=permutation,
                flow_sample_steps=config.runtime.flow_sample_steps,
            ),
        }
    finally:
        env.close()
    eef_slice = DEFAULT_STAGE0_STATE_SCHEMA.slice("eef_position")
    reference_paths: dict[str, dict[str, Any]] = {}
    expected_modes = tuple(
        trajectory_modes[window.trajectory_id]["mode_id"] for window in selected
    )
    for window, mode_id in zip(selected, expected_modes):
        trajectory = trajectory_by_id[window.trajectory_id]
        reference_paths.setdefault(window.reset_group_id, {})[mode_id] = (
            trajectory.states[:, eef_slice]
        )
    permuted_modes = tuple(expected_modes[index] for index in permutation)
    oracle_by_group = (
        oracle_latents.detach()
        .float()
        .cpu()
        .reshape(
            -1,
            mode_count,
            oracle_latents.shape[-1],
        )
    )
    selector_latents = results["selector-z"].episode_latents
    if selector_latents is None:
        raise RuntimeError("selector rollout did not retain its fixed latents")
    selector_modes: list[str] = []
    selector_coverages: list[int] = []
    for group_index in range(len(selected) // mode_count):
        start = group_index * mode_count
        stop = start + mode_count
        distances = torch.cdist(
            selector_latents[start:stop],
            oracle_by_group[group_index],
        )
        assignments = distances.argmin(dim=1).tolist()
        selector_coverages.append(len(set(assignments)))
        selector_modes.extend(config.corpus.mode_ids[index] for index in assignments)
    expected_by_condition = {
        "no-z": expected_modes,
        "oracle-z": expected_modes,
        "selector-z": tuple(selector_modes),
        "permuted-z": permuted_modes,
    }
    path_metrics = {
        name: analyze_path_adherence(
            result.eef_paths(),
            successful=result.success,
            reset_group_ids=budget.reset_group_ids,
            expected_mode_ids=expected_by_condition[name],
            reference_paths=reference_paths,
            mode_ids=config.corpus.mode_ids,
            resample_points=_PATH_ADHERENCE_PROTOCOL["resample_points"],
        )
        for name, result in results.items()
    }
    average_selector_coverage = sum(selector_coverages) / len(selector_coverages)
    selector_mode_support = {
        mode_id: selector_modes.count(mode_id) for mode_id in config.corpus.mode_ids
    }
    path_gate = bool(
        path_metrics["oracle-z"].macro_f1
        >= _PATH_ADHERENCE_PROTOCOL["oracle_z_macro_f1_minimum"]
        and path_metrics["permuted-z"].macro_f1
        >= _PATH_ADHERENCE_PROTOCOL["permuted_z_macro_f1_minimum"]
        and path_metrics["selector-z"].macro_f1
        >= _PATH_ADHERENCE_PROTOCOL["selector_z_macro_f1_minimum"]
        and average_selector_coverage
        >= _PATH_ADHERENCE_PROTOCOL["selector_average_sampled_mode_coverage_minimum"]
        and all(support > 0 for support in selector_mode_support.values())
    )
    safety_gate = all(
        not bool(result.contact_violation.any())
        and float(result.maximum_object_displacement_m.max())
        <= _ROLLOUT_PROTOCOL["maximum_object_displacement_m"]
        for result in results.values()
    )
    success_gate = bool(
        results["oracle-z"].success_rate
        >= _ROLLOUT_PROTOCOL["minimum_oracle_z_success_rate"]
        and results["selector-z"].success_rate
        >= _ROLLOUT_PROTOCOL["minimum_selector_z_success_rate"]
    )
    return {
        "budget": budget.to_record(),
        "conditioning_results": {
            name: result.to_record() for name, result in results.items()
        },
        "gate_passed": safety_gate and success_gate and path_gate,
        "mode_ids": [
            trajectory_modes[window.trajectory_id]["mode_id"] for window in selected
        ],
        "protocol": dict(_ROLLOUT_PROTOCOL),
        "path_adherence": {
            name: metrics.to_record() for name, metrics in path_metrics.items()
        },
        "path_adherence_gate_passed": path_gate,
        "path_adherence_protocol": dict(_PATH_ADHERENCE_PROTOCOL),
        "safety_gate_passed": safety_gate,
        "schema_id": "rexpolicy/stage0-final-newton-rollouts/v1",
        "split": "test",
        "success_gate_passed": success_gate,
        "selector_sampled_mode_coverage_by_reset": selector_coverages,
        "selector_average_sampled_mode_coverage": average_selector_coverage,
        "selector_sampled_mode_support": selector_mode_support,
    }


def _checkpoint_hashes(
    config: Stage0LongRunConfig,
    window_splits: Any,
    normalization: Any,
    mode_manifest_sha256: str,
):
    from rexpolicy.stage0.checkpoint import Stage0CheckpointHashes
    from rexpolicy.stage0.envs.state_schema import DEFAULT_STAGE0_STATE_SCHEMA

    split_sha256 = canonical_fingerprint(
        {
            "split_policy_sha256": window_splits.split_policy_sha256,
            "normalization_sha256": normalization.sha256,
            "action_representation": "eef_delta_xyz/v1",
            "future_state_representation": "normalized_delta_from_current/v1",
            "manifold_sampling": "paired-mode-cross-reset-octets/v1",
            "mode_invariant_alignment": "raw-latent-squared-distance/v1",
            "manifold_trainer_config": dict(_MANIFOLD_TRAINER_CONFIG),
            "mode_diagnostic_protocol": _MODE_DIAGNOSTIC_PROTOCOL,
            "no_z_control": "zero_latent_same_two_token_architecture/v1",
            "selector_diagnostic_protocol": _SELECTOR_DIAGNOSTIC_PROTOCOL,
            "selector_training_windows": "start-zero-only/v1",
            "rollout_protocol": _ROLLOUT_PROTOCOL,
            "path_adherence_protocol": _PATH_ADHERENCE_PROTOCOL,
            "window_policy_sha256": window_splits.window_policy_sha256,
        }
    )
    return Stage0CheckpointHashes(
        config_sha256=config.fingerprint,
        state_schema_sha256=DEFAULT_STAGE0_STATE_SCHEMA.sha256,
        dataset_sha256=canonical_fingerprint(
            {
                "mode_manifest_sha256": mode_manifest_sha256,
                "trajectory_corpus_sha256": window_splits.corpus_sha256,
            }
        ),
        split_sha256=split_sha256,
    )


def _rank_state(config: Stage0LongRunConfig, cursor: LongRunCursor) -> dict[str, Any]:
    return {
        "config_sha256": config.fingerprint,
        "cursor": cursor.to_steps(),
        "schema_id": _RANK_STATE_SCHEMA_ID,
        "seed": config.runtime.seed,
        "seed_derivation": "stateless-stage0-long-run/v1",
    }


def _validate_rank_state(
    config: Stage0LongRunConfig,
    cursor: LongRunCursor,
    value: Mapping[str, Any],
) -> None:
    expected = _rank_state(config, cursor)
    if dict(value) != expected:
        raise RuntimeError("checkpoint rank-state cursor or seed contract changed")


def _phase_seed_name(phase: LongRunPhase) -> str:
    if phase in (LongRunPhase.CONDITIONAL_POLICY, LongRunPhase.NO_Z_POLICY):
        return "policy_equal_budget"
    return phase.value


def _phase_training_windows(
    windows: Sequence[Any],
    phase: LongRunPhase,
) -> Sequence[Any]:
    if phase is not LongRunPhase.SELECTOR:
        return windows
    start_zero = tuple(window for window in windows if window.start == 0)
    if not start_zero:
        raise RuntimeError("selector training requires episode-start windows")
    return start_zero


def _train_one_step(
    config: Stage0LongRunConfig,
    cursor: LongRunCursor,
    phase: LongRunPhase,
    models: Mapping[str, Any],
    trainers: Mapping[str, Any],
    train_windows: Sequence[Any],
    latent_cache: Mapping[tuple[str, int], Any] | None,
    trajectory_modes: Mapping[str, Mapping[str, str]],
    *,
    device: Any,
    normalization: Any,
) -> dict[str, Any]:
    import torch

    from rexpolicy.stage0.trainers import collate_success_windows

    local_step = getattr(cursor, phase.value)
    seed_phase = _phase_seed_name(phase)
    batch_generator = _generator(
        torch,
        seed=_step_seed(config.runtime.seed, seed_phase, "batch", local_step),
    )
    model_generator = _generator(
        torch,
        seed=_step_seed(config.runtime.seed, seed_phase, "model", local_step),
        device=device,
    )
    if phase is LongRunPhase.FUTURE_ENCODER:
        windows = _sample_contrastive_windows(
            train_windows,
            trajectory_modes,
            batch_size=config.training.batch_size,
            generator=batch_generator,
        )
        metrics = trainers[phase.value].step(
            _normalize_batch(collate_success_windows(windows), normalization),
            mode_ids=tuple(
                trajectory_modes[window.trajectory_id]["mode_id"] for window in windows
            ),
            generator=model_generator,
        )
        return asdict(metrics)
    candidate_windows = _phase_training_windows(train_windows, phase)
    windows = _sample_windows(
        candidate_windows,
        batch_size=config.training.batch_size,
        generator=batch_generator,
    )
    if latent_cache is None:
        raise RuntimeError("non-encoder training requires a frozen latent cache")
    batch, target_latents = _cached_window_batch(
        windows,
        latent_cache,
        device=device,
        normalization=normalization,
    )
    if phase is LongRunPhase.SELECTOR:
        return asdict(trainers[phase.value].step(batch.current_states, target_latents))
    if phase is LongRunPhase.CONDITIONAL_POLICY:
        return asdict(
            trainers[phase.value].step(
                batch,
                target_latents,
                generator=model_generator,
            )
        )
    if phase is not LongRunPhase.NO_Z_POLICY:
        raise RuntimeError(f"unsupported training phase: {phase.value}")
    return asdict(
        trainers[phase.value].step(
            batch,
            torch.zeros_like(target_latents),
            generator=model_generator,
        )
    )


def _run_training(
    config: Stage0LongRunConfig,
    trajectories: Sequence[Any],
    mode_manifest: Mapping[str, Any],
    verification: Mapping[str, Any],
    output_dir: Path,
    *,
    device_name: str,
    resume_checkpoint: str | None,
    max_wall_seconds: float,
) -> dict[str, Any]:
    import torch

    from rexpolicy.stage0.checkpoint import Stage0CheckpointManager
    from rexpolicy.stage0.data import (
        Stage0SplitPolicy,
        Stage0WindowPolicy,
        build_success_window_splits,
        split_trajectories,
    )

    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but CUDA is unavailable: {device}")
    torch.set_num_threads(config.runtime.cpu_threads)
    try:
        torch.set_num_interop_threads(min(4, config.runtime.cpu_threads))
    except RuntimeError:
        pass
    split_policy = Stage0SplitPolicy(seed=config.runtime.seed)
    trajectory_splits = split_trajectories(trajectories, split_policy)
    normalization = _fit_normalization(trajectory_splits, output_dir)
    window_policy = Stage0WindowPolicy(
        action_horizon=config.stage0.action_horizon,
        future_horizon=config.stage0.future_horizon,
        stride=config.stage0.window_stride,
    )
    window_splits = build_success_window_splits(
        trajectories,
        split_policy=split_policy,
        window_policy=window_policy,
    )
    if not window_splits.train:
        raise RuntimeError("long-run training split has no successful windows")
    held_out = window_splits.validation
    if len(held_out) < 2:
        raise RuntimeError("long-run held-out split has fewer than two windows")
    trajectory_modes = mode_manifest["trajectory_modes"]
    if not isinstance(trajectory_modes, Mapping):
        raise RuntimeError("mode manifest trajectory map must be an object")
    _select_complete_mode_groups(
        window_splits.validation,
        trajectory_modes,
        config.corpus.mode_ids,
        maximum_windows=config.runtime.eval_window_count,
    )
    _select_complete_mode_groups(
        window_splits.test,
        trajectory_modes,
        config.corpus.mode_ids,
        maximum_windows=config.runtime.eval_window_count,
    )
    models, optimizers, trainers = _create_training_components(
        config,
        device=device,
        action_feature_mask=normalization.effective_action_mask,
    )
    hashes = _checkpoint_hashes(
        config,
        window_splits,
        normalization,
        verification["mode_manifest_sha256"],
    )
    manager = Stage0CheckpointManager(output_dir)
    cursor = LongRunCursor()
    resumed_from: str | None = None
    latest_checkpoint = manager.latest_checkpoint()
    if resume_checkpoint == "latest" and latest_checkpoint is None:
        # A verified --collect-only output can become a training run without
        # recollecting or weakening the run contract.
        resume_checkpoint = None
    if resume_checkpoint is not None:
        selected = None if resume_checkpoint == "latest" else Path(resume_checkpoint)
        if selected is not None:
            selected = selected.expanduser().resolve()
            if latest_checkpoint is None or selected != latest_checkpoint:
                raise RuntimeError(
                    "in-place long-run resume permits only the latest complete "
                    "checkpoint"
                )
        resumed = manager.load(
            hashes=hashes,
            models=models,
            optimizers=optimizers,
            checkpoint=selected,
            restore_rng=True,
            cuda_device=device,
        )
        cursor = LongRunCursor.from_steps(resumed.steps)
        _validate_rank_state(config, cursor, resumed.rank_state)
        expected_phase = cursor.next_phase(config.training).value
        if resumed.phase != expected_phase:
            raise RuntimeError(
                f"checkpoint phase {resumed.phase!r} != recovered phase {expected_phase!r}"
            )
        resumed_from = str(resumed.checkpoint_path)
    for name, trainer in trainers.items():
        trainer.optimizer_step = getattr(cursor, name)

    heartbeat = AtomicJsonlLog(output_dir / "heartbeat.jsonl")
    event_index = len(heartbeat.records())
    budget = WallClockBudget(max_wall_seconds)
    latest_eval: dict[str, Any] | None = None
    latest_eval_summary: dict[str, Any] | None = None
    latest_metrics: dict[str, Any] | None = None
    latent_cache: dict[tuple[str, int], Any] | None = None
    selector_warm_started = cursor.selector > 0
    last_checkpoint_step = cursor.global_step if resume_checkpoint is not None else -1

    def emit(event: str, *, detail: Mapping[str, Any] | None = None) -> None:
        nonlocal event_index
        record = {
            "config_sha256": config.fingerprint,
            "cursor": cursor.to_steps(),
            "detail": dict(detail or {}),
            "elapsed_seconds": budget.elapsed_seconds,
            "event": event,
            "event_index": event_index,
            "phase": cursor.next_phase(config.training).value,
            "schema_id": "rexpolicy/stage0-long-run-heartbeat/v1",
            "utc_unix_seconds": time.time(),
        }
        heartbeat.append(record)
        atomic_write_json(output_dir / "status.json", record)
        print(json.dumps(record, sort_keys=True), flush=True)
        event_index += 1

    def evaluate() -> None:
        nonlocal latest_eval, latest_eval_summary
        latest_eval = _evaluate_offline(
            config,
            models,
            held_out,
            trajectory_modes,
            device=device,
            global_step=cursor.global_step,
            normalization=normalization,
            split_name="validation",
        )
        latest_eval_summary = _evaluation_summary(latest_eval)
        atomic_write_json(
            output_dir / "evaluations" / f"eval-{cursor.global_step:012d}.json",
            latest_eval,
        )
        emit("evaluation", detail=latest_eval_summary)

    def save_checkpoint(*, stop_reason: str | None = None) -> Path | None:
        nonlocal last_checkpoint_step
        if cursor.global_step == last_checkpoint_step:
            return manager.latest_checkpoint()
        phase = cursor.next_phase(config.training).value
        checkpoint = manager.save(
            sequence=cursor.global_step,
            phase=phase,
            steps=cursor.to_steps(),
            hashes=hashes,
            models=models,
            optimizers=optimizers,
            rank_state=_rank_state(config, cursor),
            extra={
                "latest_evaluation": latest_eval_summary,
                "offline_evaluation_only": True,
                "normalization_sha256": normalization.sha256,
                "stop_reason": stop_reason,
            },
            cuda_device=device,
        )
        last_checkpoint_step = cursor.global_step
        emit("checkpoint", detail={"checkpoint": str(checkpoint)})
        return checkpoint

    if resume_checkpoint is None:
        save_checkpoint()
    emit(
        "resumed" if resumed_from else "started",
        detail={
            "resumed_from": resumed_from,
            "train_windows": len(window_splits.train),
            "validation_windows": len(window_splits.validation),
            "test_windows": len(window_splits.test),
            "verification": dict(verification),
        },
    )
    if (
        cursor.global_step == 0
        or cursor.global_step % config.training.eval_every_steps == 0
    ):
        evaluate()

    stop_reason: str | None = None
    while cursor.next_phase(config.training) is not LongRunPhase.COMPLETE:
        if budget.expired:
            stop_reason = "max_wall_seconds"
            break
        if _STOP_REASON is not None:
            stop_reason = _STOP_REASON
            break
        phase = cursor.next_phase(config.training)
        if phase is not LongRunPhase.FUTURE_ENCODER and latent_cache is None:
            latent_cache = _precompute_window_latents(
                models["future_encoder"],
                window_splits.train,
                batch_size=config.training.batch_size,
                device=device,
                normalization=normalization,
            )
            emit(
                "latent_cache_ready",
                detail={"window_count": len(latent_cache)},
            )
        if (
            phase is LongRunPhase.SELECTOR
            and not selector_warm_started
            and latent_cache is not None
        ):
            warm_start = _warm_start_selector(
                models["selector"],
                window_splits.train,
                latent_cache,
                trajectory_modes,
                config.corpus.mode_ids,
            )
            selector_warm_started = True
            emit("selector_warm_started", detail=warm_start)
        before_phase = phase
        latest_metrics = _train_one_step(
            config,
            cursor,
            phase,
            models,
            trainers,
            window_splits.train,
            latent_cache,
            trajectory_modes,
            device=device,
            normalization=normalization,
        )
        cursor = cursor.advance(config.training, phase)
        next_phase = cursor.next_phase(config.training)
        phase_boundary = next_phase is not before_phase
        if cursor.global_step % config.training.log_every_steps == 0 or phase_boundary:
            emit(
                "training",
                detail={"metrics": latest_metrics, "trained_phase": phase.value},
            )
        should_evaluate = (
            cursor.global_step % config.training.eval_every_steps == 0
            or phase_boundary
            or next_phase is LongRunPhase.COMPLETE
        )
        if should_evaluate:
            evaluate()
        should_checkpoint = (
            cursor.global_step % config.training.checkpoint_every_steps == 0
            or phase_boundary
            or next_phase is LongRunPhase.COMPLETE
        )
        if should_checkpoint:
            save_checkpoint()

    if stop_reason is not None:
        if latest_eval is None:
            evaluate()
        checkpoint = save_checkpoint(stop_reason=stop_reason)
        emit(
            "paused",
            detail={"checkpoint": str(checkpoint), "stop_reason": stop_reason},
        )
        status = "paused"
    else:
        if latest_eval is None or latest_eval["global_step"] != cursor.global_step:
            evaluate()
        checkpoint = save_checkpoint()
        emit("complete", detail={"checkpoint": str(checkpoint)})
        status = "complete"

    final_test_eval: dict[str, Any] | None = None
    final_rollout_eval: dict[str, Any] | None = None
    if status == "complete":
        final_test_eval = _evaluate_offline(
            config,
            models,
            window_splits.test,
            trajectory_modes,
            device=device,
            global_step=cursor.global_step,
            normalization=normalization,
            split_name="test",
        )
        atomic_write_json(
            output_dir / "evaluations" / f"test-final-{cursor.global_step:012d}.json",
            final_test_eval,
        )
        emit(
            "final_test_evaluation",
            detail=_evaluation_summary(final_test_eval),
        )
        final_rollout_eval = _evaluate_newton_rollouts(
            config,
            models,
            window_splits.test,
            trajectories,
            trajectory_modes,
            device_name=device_name,
            normalization=normalization,
        )
        atomic_write_json(
            output_dir / "final-newton-rollouts.json",
            final_rollout_eval,
        )
        emit(
            "final_newton_rollouts",
            detail={
                "gate_passed": final_rollout_eval["gate_passed"],
                "path_adherence_gate_passed": final_rollout_eval[
                    "path_adherence_gate_passed"
                ],
                "safety_gate_passed": final_rollout_eval["safety_gate_passed"],
                "success_gate_passed": final_rollout_eval["success_gate_passed"],
            },
        )
    gate_evaluation = final_test_eval or {}
    controls = gate_evaluation.get("policy_controls", {})
    latent = gate_evaluation.get("latent", {})
    offline_gate = bool(
        status == "complete"
        and latent.get("collapse_gate_passed", False)
        and gate_evaluation.get("latent_mode_gate_passed", False)
        and gate_evaluation.get("selector", {}).get("gate_passed", False)
        and controls.get("oracle_z_improves_over_no_z", False)
    )
    report = {
        "checkpoint": str(checkpoint),
        "config_sha256": config.fingerprint,
        "cursor": cursor.to_steps(),
        "elapsed_seconds": budget.elapsed_seconds,
        "latest_evaluation": latest_eval,
        "final_test_evaluation": final_test_eval,
        "final_newton_rollouts": final_rollout_eval,
        "offline_representation_gate_passed": offline_gate,
        "normalization_sha256": normalization.sha256,
        "resumed_from": resumed_from,
        "rollout_feasibility_gate": (
            "not_evaluated"
            if final_rollout_eval is None
            else final_rollout_eval["gate_passed"]
        ),
        "scientific_limitations": [
            "offline action error is not rollout success",
            "selector samples may represent a different valid mode than a paired demo",
            "Newton learned-policy rollouts are the authoritative feasibility gate",
        ],
        "scientific_feasibility_status": (
            "supported"
            if offline_gate
            and final_rollout_eval is not None
            and final_rollout_eval["gate_passed"]
            else "not_supported"
            if status == "complete"
            else "training_incomplete"
        ),
        "status": status,
        "stop_reason": stop_reason,
        "verification": dict(verification),
        "window_counts": {
            "test": len(window_splits.test),
            "train": len(window_splits.train),
            "validation": len(window_splits.validation),
        },
    }
    atomic_write_json(output_dir / "long-run-report.json", report)
    return report


def main() -> None:
    args = _parse_args()
    config = load_long_run_config(args.config)
    if args.max_wall_seconds is not None and args.max_wall_seconds <= 0.0:
        raise ValueError("--max-wall-seconds must be positive")
    if args.validate_config_only:
        print(
            json.dumps(
                {"config": config.to_record(), "config_sha256": config.fingerprint},
                indent=2,
                sort_keys=True,
            )
        )
        return
    if args.resume is not None and args.output_dir is None:
        raise ValueError("--resume requires --output-dir")
    _configure_cpu_threads(config.runtime.cpu_threads)
    _install_stop_handlers()
    output_dir = (args.output_dir or _default_output_dir()).expanduser().resolve()
    _prepare_output(output_dir, config, resume=args.resume is not None)
    trajectories, manifest, verification = _ensure_corpus(
        config,
        output_dir,
        source=args.corpus_dir,
        device=args.device,
        capture_graph=args.capture_graph,
    )
    if args.collect_only:
        result = {
            "config_sha256": config.fingerprint,
            "output_dir": str(output_dir),
            "status": "corpus_complete",
            "verification": verification,
        }
        atomic_write_json(output_dir / "long-run-report.json", result)
        print(json.dumps(result, indent=2, sort_keys=True))
        return
    report = _run_training(
        config,
        trajectories,
        manifest,
        verification,
        output_dir,
        device_name=args.device,
        resume_checkpoint=args.resume,
        max_wall_seconds=(
            config.runtime.max_wall_seconds
            if args.max_wall_seconds is None
            else args.max_wall_seconds
        ),
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(
            f"Stage 0 long run failed: {type(error).__name__}: {error}", file=sys.stderr
        )
        raise
