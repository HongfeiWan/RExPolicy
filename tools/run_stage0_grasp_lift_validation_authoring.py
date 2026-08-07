#!/usr/bin/env python3
"""Author one preregistered six-member close7 validation cohort on CUDA."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any

from rexpolicy.stage0.data.grasp_lift_validation_cohort import (
    GRASP_LIFT_VALIDATION_AUTHORING_COMMIT_SCHEMA_ID,
    GRASP_LIFT_VALIDATION_AUTHORING_MANIFEST_SCHEMA_ID,
    GRASP_LIFT_VALIDATION_AUTHORING_STATUS_SCHEMA_ID,
    GRASP_LIFT_VALIDATION_LOCKED_TEST,
    grasp_lift_validation_cohort_slot_id,
    grasp_lift_validation_environment_config_record,
    grasp_lift_validation_preregistration_record,
    load_grasp_lift_validation_authoring_config,
    load_grasp_lift_validation_authoring_manifest,
    validation_authoring_file_sha256,
)
from rexpolicy.stage0.grasp_lift_pilot import (
    grasp_lift_pilot_oracle_record,
    grasp_lift_pilot_task_metadata_record,
    seal_grasp_lift_pilot_record,
)
from rexpolicy.stage0.implementation_fingerprint import (
    capture_stage0_implementation_fingerprint,
)
from rexpolicy.stage0.types import canonical_fingerprint
from tools.run_stage0_grasp_lift_pilot import (
    _append_evidence,
    _fsync_directory,
    _outcome_from_tensors,
    _restrict_cpu_threads,
    _write_json,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/stage0/grasp_lift_validation_cohort_v2.json",
    )
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def _persist_authoring_claim_once(
    *,
    repository_root: Path,
    cohort_slot_id: str,
    config_sha256: str,
    preregistration_sha256: str,
    implementation_sha256: str,
) -> dict[str, Any]:
    """Burn this exact seed slot before Torch or Newton can observe it."""
    common = subprocess.run(
        ["git", "rev-parse", "--git-common-dir"],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    common_dir = Path(common)
    if not common_dir.is_absolute():
        common_dir = (repository_root / common_dir).resolve()
    claim = seal_grasp_lift_pilot_record(
        {
            "cohort_slot_id": cohort_slot_id,
            "config_sha256": config_sha256,
            "implementation_sha256": implementation_sha256,
            "locked_test": dict(GRASP_LIFT_VALIDATION_LOCKED_TEST),
            "preregistration_sha256": preregistration_sha256,
            "schema_id": ("rexpolicy/stage0-grasp-lift-validation-authoring-claim/v1"),
        }
    )
    registry = common_dir / "rexpolicy-validation-authoring-claims"
    existed = registry.exists()
    registry.mkdir(parents=True, exist_ok=True)
    if registry.is_symlink() or not registry.is_dir():
        raise RuntimeError("validation authoring claim registry is not a directory")
    if not existed:
        _fsync_directory(common_dir)
    try:
        _write_json(registry / f"{cohort_slot_id}.json", claim)
    except FileExistsError as error:
        raise RuntimeError(
            "this validation cohort seed slot was already authored or attempted"
        ) from error
    _fsync_directory(registry)
    return claim


def main() -> None:
    args = _parse_args()
    repository_root = Path(__file__).resolve().parents[1]
    config_path = Path(args.config).expanduser().resolve()
    config = load_grasp_lift_validation_authoring_config(config_path)
    target = Path(args.output).expanduser().resolve()
    if target.is_symlink() or target.exists():
        raise FileExistsError(f"validation authoring output exists: {target}")
    implementation = capture_stage0_implementation_fingerprint(repository_root)
    affinity = _restrict_cpu_threads(config.runtime.cpu_threads)
    preregistration = grasp_lift_validation_preregistration_record(config)
    preregistration_sha256 = canonical_fingerprint(preregistration)
    cohort_slot_id = grasp_lift_validation_cohort_slot_id(config)
    _persist_authoring_claim_once(
        repository_root=repository_root,
        cohort_slot_id=cohort_slot_id,
        config_sha256=config.sha256,
        preregistration_sha256=preregistration_sha256,
        implementation_sha256=implementation.sha256,
    )

    target.parent.mkdir(parents=True, exist_ok=True)
    if target.parent.is_symlink() or not target.parent.is_dir():
        raise ValueError("validation authoring output parent must be a real directory")
    partial = target.with_name(f".{target.name}.partial-{uuid.uuid4().hex}")
    partial.mkdir()
    shards = partial / "shards"
    shards.mkdir()
    _write_json(partial / "config.json", config.to_record())
    started = time.perf_counter()
    env = None
    try:
        import torch

        torch.set_num_threads(config.runtime.cpu_threads)
        from rexpolicy.envs.groot_newton_env import (
            GRASP_LIFT_GREEN_BOTTLE_TASK_ID,
            GrootNewtonEnv,
            GrootNewtonEnvConfig,
        )
        from rexpolicy.stage0.data.grasp_lift_evidence import (
            GRASP_LIFT_EVIDENCE_TENSOR_SPECS,
            GraspLiftEvidence,
            GraspLiftEvidenceMetadata,
            grasp_lift_composite_corpus_sha256,
            grasp_lift_evidence_sha256,
            load_grasp_lift_evidence_sidecar,
            write_grasp_lift_evidence_sidecar,
        )
        from rexpolicy.stage0.data.store import (
            load_trajectory_shard,
            write_trajectory_shard,
        )
        from rexpolicy.stage0.data.trajectory import (
            Stage0Trajectory,
            Stage0TrajectoryProvenance,
            trajectory_sha256,
        )
        from rexpolicy.stage0.envs.grasp_lift_action import (
            DEFAULT_GRASP_LIFT_ACTION_SCHEMA,
            GRASP_LIFT_EFFECTIVE_ACTION_MASK,
        )
        from rexpolicy.stage0.envs.grasp_lift_modes import (
            DEFAULT_GRASP_LIFT_AUTHORING_MODE,
            GraspLiftAuthoringController,
        )
        from rexpolicy.stage0.envs.grasp_lift_oracle import (
            GRASP_LIFT_ORACLE_ID,
            GraspLiftSuccessOracle,
            grasp_lift_transient_events_from_info,
        )
        from rexpolicy.stage0.envs.grasp_lift_state_view import (
            DEFAULT_GRASP_LIFT_STATE_VIEW,
        )
        from rexpolicy.stage0.envs.state_only import Stage0StateOnlyEnv
        from rexpolicy.stage0.envs.state_schema import DEFAULT_STAGE0_STATE_SCHEMA
        from rexpolicy.stage0.types import Stage0Outcome

        expected_environment = grasp_lift_validation_environment_config_record(config)
        env_config = GrootNewtonEnvConfig(**expected_environment)
        if asdict(env_config) != expected_environment:
            raise RuntimeError("Newton validation environment config drifted")
        env = GrootNewtonEnv(env_config)
        expected_task = grasp_lift_pilot_task_metadata_record()
        if canonical_fingerprint(env.task_metadata) != canonical_fingerprint(
            expected_task
        ):
            raise RuntimeError("Newton validation task metadata drifted")
        oracle_record = grasp_lift_pilot_oracle_record()
        simulator_record = {
            "action_schema_sha256": DEFAULT_GRASP_LIFT_ACTION_SCHEMA.sha256,
            "authoring_mode_sha256": DEFAULT_GRASP_LIFT_AUTHORING_MODE.sha256,
            "environment_config": expected_environment,
            "implementation_sha256": implementation.sha256,
            "oracle": oracle_record,
            "oracle_id": GRASP_LIFT_ORACLE_ID,
            "schema_id": "rexpolicy/stage0-grasp-lift-newton-runtime/v1",
            "task_metadata": expected_task,
        }
        simulator_sha256 = canonical_fingerprint(simulator_record)
        adapter = Stage0StateOnlyEnv(env)
        expected_mask = torch.tensor(
            GRASP_LIFT_EFFECTIVE_ACTION_MASK,
            dtype=torch.bool,
            device=config.runtime.device,
        )
        if not torch.equal(env.effective_action_mask_torch(), expected_mask):
            raise RuntimeError("Newton validation effective action mask drifted")
        if (
            env.frames_per_action
            != DEFAULT_GRASP_LIFT_AUTHORING_MODE.full_opposed_physics_frames
        ):
            raise RuntimeError("Newton validation frames-per-action drifted")

        state, _ = adapter.reset(seed=list(config.seeds))
        controller = GraspLiftAuthoringController()
        controller.reset(state, env.hold_action_torch().clone())
        oracle = GraspLiftSuccessOracle(
            config.runtime.num_envs,
            lift_height_m=oracle_record["lift_height_m"],
            success_hold_control_steps=oracle_record["success_hold_control_steps"],
            lateral_displacement_limit_m=oracle_record["lateral_displacement_limit_m"],
            bottle_tilt_limit_rad=oracle_record["bottle_tilt_limit_rad"],
            grasp_confirm_physics_frames=oracle_record["grasp_confirm_physics_frames"],
            drop_confirm_control_steps=oracle_record["drop_confirm_control_steps"],
        )
        oracle.reset(state)
        reset_axis = state[
            :, DEFAULT_STAGE0_STATE_SCHEMA.slice("object_to_goal")
        ].clone()
        reset_axis /= torch.linalg.vector_norm(reset_axis, dim=-1, keepdim=True)
        state_rows = [
            [state[world].detach().cpu().clone()]
            for world in range(config.runtime.num_envs)
        ]
        action_rows: list[list[Any]] = [[] for _ in range(config.runtime.num_envs)]
        evidence_rows: list[dict[str, list[Any]]] = [
            {name: [] for name in GRASP_LIFT_EVIDENCE_TENSOR_SPECS}
            for _ in range(config.runtime.num_envs)
        ]
        world_steps = 0
        nonzero_reward_transition_count = 0
        collision_buffer_overflow_count = 0
        latest = None
        for _ in range(config.runtime.max_episode_steps):
            active = ~controller.terminal_mask
            before = controller.snapshot()
            proposal = controller.propose(state)
            projected = env.project_effective_action_torch(proposal)
            transition = adapter.step(projected)
            executed = env.effective_action_torch().detach().clone()
            if not torch.equal(executed, projected):
                raise RuntimeError("Newton validation executed action drifted")
            reward_nonzero = int(torch.count_nonzero(transition.reward).item())
            nonzero_reward_transition_count += reward_nonzero
            if reward_nonzero:
                raise RuntimeError("reward_mode=none produced non-zero reward")
            events = grasp_lift_transient_events_from_info(
                transition.info,
                count=config.runtime.num_envs,
                device=state.device,
            )
            overflow = int(events.collision_buffer_overflow.sum().item())
            collision_buffer_overflow_count += overflow
            if overflow:
                raise RuntimeError("collision buffer overflow invalidated cohort")
            latest = oracle.evaluate(transition.next_state, events)
            evaluation = env.evaluate()
            after = controller.observe(
                executed_action=executed,
                next_state=transition.next_state,
                oracle_result=latest,
                events=events,
                is_obj_static=evaluation["is_obj_static"],
                is_robot_static=evaluation["is_robot_static"],
            )
            diagnostics = transition.info["control_step_diagnostics"]
            for world in torch.nonzero(active, as_tuple=False).flatten().tolist():
                action_rows[world].append(executed[world].detach().cpu().clone())
                state_rows[world].append(
                    transition.next_state[world].detach().cpu().clone()
                )
                _append_evidence(
                    evidence_rows[world],
                    world=world,
                    proposal=proposal,
                    executed=executed,
                    next_state=transition.next_state,
                    before=before,
                    after=after,
                    events=events,
                    result=latest,
                    evaluation=evaluation,
                    diagnostics=diagnostics,
                    schema=DEFAULT_STAGE0_STATE_SCHEMA,
                    torch=torch,
                )
                world_steps += 1
            state = transition.next_state
            if bool(controller.terminal_mask.all()):
                break
        if latest is None:
            raise RuntimeError("validation authoring executed no transition")

        corpus = []
        members: list[dict[str, Any]] = []
        outcome_counts = {item.value: 0 for item in Stage0Outcome}
        safety_violation_count = 0
        for world, seed in enumerate(config.seeds):
            tensors = {
                name: torch.stack(evidence_rows[world][name])
                for name in GRASP_LIFT_EVIDENCE_TENSOR_SPECS
            }
            outcome_name, terminated, truncated, failure_reason = _outcome_from_tensors(
                tensors
            )
            outcome = Stage0Outcome(outcome_name)
            outcome_counts[outcome.value] += 1
            violation_keys = (
                "forbidden_contact_violation",
                "lateral_displacement_violation",
                "bottle_tilt_violation",
                "dropped_grasp_violation",
                "collision_buffer_overflow_violation",
            )
            safety_violation = any(bool(tensors[name][-1]) for name in violation_keys)
            safety_violation_count += int(safety_violation)
            reset_group_id = f"grasp-validation-v2-g{world:06d}-s{seed:010d}"
            trajectory_id = (
                f"grasp-validation-v2-g{world:06d}-thumb-index-side-close7-"
                f"h{DEFAULT_GRASP_LIFT_AUTHORING_MODE.sha256[:12]}-s{seed:010d}"
            )
            trajectory = Stage0Trajectory(
                provenance=Stage0TrajectoryProvenance(
                    trajectory_id=trajectory_id,
                    reset_group_id=reset_group_id,
                    task_id=GRASP_LIFT_GREEN_BOTTLE_TASK_ID,
                    oracle_id=GRASP_LIFT_ORACLE_ID,
                    simulator_sha256=simulator_sha256,
                    experiment_sha256=config.sha256,
                    state_schema_sha256=DEFAULT_STAGE0_STATE_SCHEMA.sha256,
                    action_schema_sha256=DEFAULT_GRASP_LIFT_ACTION_SCHEMA.sha256,
                    generation=1,
                    rank=0,
                    world=world,
                    reset_seed=seed,
                ),
                states=torch.stack(state_rows[world]),
                actions=torch.stack(action_rows[world]),
                outcome=outcome,
                terminated=terminated,
                truncated=truncated,
                failure_reason=failure_reason,
            )
            evidence = GraspLiftEvidence(
                metadata=GraspLiftEvidenceMetadata.from_trajectory(
                    trajectory,
                    mode_id=DEFAULT_GRASP_LIFT_AUTHORING_MODE.mode_id,
                    mode_sha256=DEFAULT_GRASP_LIFT_AUTHORING_MODE.sha256,
                    reset_lift_axis=tuple(float(value) for value in reset_axis[world]),
                    implementation_sha256=implementation.sha256,
                ),
                tensors=tensors,
            )
            trajectory_descriptor = write_trajectory_shard(shards, trajectory)
            evidence_descriptor = write_grasp_lift_evidence_sidecar(
                shards, evidence, trajectory=trajectory
            )
            restored_trajectory = load_trajectory_shard(trajectory_descriptor)
            restored_evidence = load_grasp_lift_evidence_sidecar(
                evidence_descriptor, trajectory=restored_trajectory
            )
            corpus.append((restored_trajectory, restored_evidence))
            members.append(
                {
                    "evidence_descriptor": str(
                        evidence_descriptor.relative_to(partial)
                    ),
                    "evidence_sha256": grasp_lift_evidence_sha256(evidence),
                    "final_authoring_phase": int(tensors["authoring_phase_after"][-1]),
                    "outcome": outcome.value,
                    "reset_group_id": reset_group_id,
                    "reset_seed": seed,
                    "role": "formal_validation",
                    "safety_violation": safety_violation,
                    "split": "validation",
                    "terminal_transition_from_phase": int(
                        tensors["authoring_phase_before"][-1]
                    ),
                    "trajectory_descriptor": str(
                        trajectory_descriptor.relative_to(partial)
                    ),
                    "trajectory_id": trajectory_id,
                    "trajectory_sha256": trajectory_sha256(trajectory),
                }
            )

        passed = bool(
            outcome_counts == {"failure": 0, "success": 6, "timeout": 0}
            and safety_violation_count == 0
            and collision_buffer_overflow_count == 0
            and nonzero_reward_transition_count == 0
        )
        if not passed:
            raise RuntimeError("preregistered validation authoring acceptance failed")
        acceptance = {
            "collision_buffer_overflow_count": collision_buffer_overflow_count,
            "nonzero_reward_transition_count": nonzero_reward_transition_count,
            "outcome_counts": outcome_counts,
            "passed": True,
            "policy": dict(config.acceptance),
            "safety_violation_count": safety_violation_count,
        }
        elapsed = time.perf_counter() - started
        manifest = seal_grasp_lift_pilot_record(
            {
                "acceptance": acceptance,
                "action_schema": DEFAULT_GRASP_LIFT_ACTION_SCHEMA.to_record(),
                "action_schema_sha256": DEFAULT_GRASP_LIFT_ACTION_SCHEMA.sha256,
                "authoring_mode": DEFAULT_GRASP_LIFT_AUTHORING_MODE.to_record(),
                "authoring_mode_sha256": DEFAULT_GRASP_LIFT_AUTHORING_MODE.sha256,
                "cohort_slot_id": cohort_slot_id,
                "composite_corpus_sha256": grasp_lift_composite_corpus_sha256(corpus),
                "config_sha256": config.sha256,
                "cpu_affinity": list(affinity),
                "elapsed_seconds": elapsed,
                "environment_config": expected_environment,
                "environment_config_sha256": canonical_fingerprint(
                    expected_environment
                ),
                "implementation": implementation.to_record(),
                "implementation_sha256": implementation.sha256,
                "locked_test": dict(GRASP_LIFT_VALIDATION_LOCKED_TEST),
                "members": members,
                "oracle": oracle_record,
                "oracle_sha256": canonical_fingerprint(oracle_record),
                "preregistration": preregistration,
                "preregistration_sha256": preregistration_sha256,
                "purpose": config.purpose,
                "reset_seeds": list(config.seeds),
                "schema_id": GRASP_LIFT_VALIDATION_AUTHORING_MANIFEST_SCHEMA_ID,
                "seed_derivation": config.seed_derivation,
                "seed_namespace": config.seed_namespace,
                "simulator_record": simulator_record,
                "simulator_sha256": simulator_sha256,
                "state_view": DEFAULT_GRASP_LIFT_STATE_VIEW.to_record(),
                "state_view_sha256": DEFAULT_GRASP_LIFT_STATE_VIEW.sha256,
                "task_metadata": expected_task,
                "task_metadata_sha256": canonical_fingerprint(expected_task),
                "visuals": {
                    "camera_textures": False,
                    "scene_asset": None,
                    "scene_visuals": False,
                },
                "world_steps": world_steps,
            }
        )
        _write_json(partial / "manifest.json", manifest)
        status = seal_grasp_lift_pilot_record(
            {
                "accepted": True,
                "cohort_slot_id": cohort_slot_id,
                "collision_buffer_overflow_count": 0,
                "complete": True,
                "manifest_sha256": manifest["self_sha256"],
                "nonzero_reward_transition_count": 0,
                "outcome_counts": outcome_counts,
                "safety_violation_count": 0,
                "schema_id": GRASP_LIFT_VALIDATION_AUTHORING_STATUS_SCHEMA_ID,
            }
        )
        _write_json(partial / "status.json", status)
        commit = seal_grasp_lift_pilot_record(
            {
                "config_file": "config.json",
                "config_file_sha256": validation_authoring_file_sha256(
                    partial / "config.json"
                ),
                "config_sha256": config.sha256,
                "manifest_file": "manifest.json",
                "manifest_file_sha256": validation_authoring_file_sha256(
                    partial / "manifest.json"
                ),
                "manifest_sha256": manifest["self_sha256"],
                "schema_id": GRASP_LIFT_VALIDATION_AUTHORING_COMMIT_SCHEMA_ID,
                "status_file": "status.json",
                "status_file_sha256": validation_authoring_file_sha256(
                    partial / "status.json"
                ),
                "status_sha256": status["self_sha256"],
            }
        )
        _write_json(partial / "commit.json", commit)
        _fsync_directory(partial)
        load_grasp_lift_validation_authoring_manifest(partial)
        partial.rename(target)
        _fsync_directory(target.parent)
        print(json.dumps(manifest, indent=2, sort_keys=True))
    except BaseException as error:
        if partial.exists() and not (partial / "error.json").exists():
            try:
                _write_json(
                    partial / "error.json",
                    {
                        "cohort_slot_id": cohort_slot_id,
                        "error_type": type(error).__name__,
                        "schema_id": (
                            "rexpolicy/stage0-grasp-lift-validation-authoring-error/v1"
                        ),
                    },
                )
                _fsync_directory(partial)
            except BaseException:
                pass
        raise
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    main()
