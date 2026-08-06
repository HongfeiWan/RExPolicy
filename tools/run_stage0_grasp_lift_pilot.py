#!/usr/bin/env python3
"""Author an immutable train/validation-only Stage 0 Grasp-Lift pilot corpus."""

from __future__ import annotations

import argparse
import json
import os
import time
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

from rexpolicy.stage0.grasp_lift_pilot import (
    GRASP_LIFT_PILOT_COMMIT_SCHEMA_ID,
    GRASP_LIFT_PILOT_OUTPUT_SCHEMA_ID,
    GRASP_LIFT_PILOT_STATUS_SCHEMA_ID,
    assign_grasp_lift_pilot_splits,
    grasp_lift_pilot_environment_config_record,
    grasp_lift_pilot_oracle_record,
    grasp_lift_pilot_split_policy,
    grasp_lift_pilot_task_metadata_record,
    load_grasp_lift_pilot_config,
    load_grasp_lift_pilot_manifest,
    seal_grasp_lift_pilot_record,
)
from rexpolicy.stage0.implementation_fingerprint import (
    capture_stage0_implementation_fingerprint,
)
from rexpolicy.stage0.types import canonical_fingerprint


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/stage0/grasp_lift_pilot.json"
    )
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def _write_json(path: Path, record: Mapping[str, Any]) -> None:
    encoded = (
        json.dumps(record, allow_nan=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    with path.open("xb") as file:
        file.write(encoded)
        file.flush()
        os.fsync(file.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _restrict_cpu_threads(cpu_threads: int) -> tuple[int, ...]:
    os.environ["OMP_NUM_THREADS"] = str(cpu_threads)
    os.environ["MKL_NUM_THREADS"] = str(cpu_threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(cpu_threads)
    if not hasattr(os, "sched_getaffinity") or not hasattr(os, "sched_setaffinity"):
        raise RuntimeError("pilot authoring requires Linux CPU affinity support")
    available = tuple(sorted(os.sched_getaffinity(0)))
    if len(available) < cpu_threads:
        raise RuntimeError(
            f"pilot requires {cpu_threads} CPUs, only {len(available)} are available"
        )
    selected = available[:cpu_threads]
    if len(available) > cpu_threads:
        os.sched_setaffinity(0, selected)
    restricted = tuple(sorted(os.sched_getaffinity(0)))
    if len(restricted) != cpu_threads:
        raise RuntimeError("pilot CPU affinity restriction did not take effect")
    return restricted


def _diagnostic_scalar(
    diagnostics: Mapping[str, Any],
    name: str,
    *,
    torch: Any,
    dtype: Any,
) -> Any:
    if name not in diagnostics:
        raise KeyError(f"control_step_diagnostics is missing {name!r}")
    value = diagnostics[name]
    if isinstance(value, torch.Tensor):
        flattened = value.reshape(-1)
        if flattened.numel() < 1:
            raise ValueError(f"diagnostic {name} is empty")
        if dtype is torch.bool:
            scalar = flattened.to(dtype=torch.bool).any()
        else:
            scalar = flattened.to(dtype=torch.int64).max()
        return scalar.detach().to(device="cpu", dtype=dtype).clone()
    if dtype is torch.bool and isinstance(value, bool):
        return torch.tensor(value, dtype=torch.bool)
    if dtype is torch.int64 and isinstance(value, int) and not isinstance(value, bool):
        return torch.tensor(value, dtype=torch.int64)
    raise TypeError(f"diagnostic {name} has unsupported type {type(value).__name__}")


def _append_evidence(
    rows: dict[str, list[Any]],
    *,
    world: int,
    proposal: Any,
    executed: Any,
    next_state: Any,
    before: Any,
    after: Any,
    events: Any,
    result: Any,
    evaluation: Mapping[str, Any],
    diagnostics: Mapping[str, Any],
    schema: Any,
    torch: Any,
) -> None:
    cpu_float = lambda value: value.detach().to(
        device="cpu", dtype=torch.float32
    ).clone()
    cpu_bool = lambda value: value.detach().to(
        device="cpu", dtype=torch.bool
    ).clone()
    cpu_int = lambda value: value.detach().to(
        device="cpu", dtype=torch.int64
    ).clone()

    rows["proposal_action"].append(cpu_float(proposal[world]))
    rows["executed_action"].append(cpu_float(executed[world]))
    rows["measured_eef_position"].append(
        cpu_float(next_state[world, schema.slice("eef_position")])
    )
    rows["measured_hand_joint_position"].append(
        cpu_float(next_state[world, schema.slice("hand_joint_position")])
    )
    rows["oracle_lift_height"].append(cpu_float(result.lift_height[world]))
    rows["oracle_lateral_displacement"].append(
        cpu_float(result.lateral_displacement[world])
    )
    rows["oracle_bottle_tilt"].append(cpu_float(result.bottle_tilt[world]))

    boolean_values = {
        "authoring_has_tentative_target": after.has_tentative_target[world],
        "authoring_has_frozen_target": after.has_frozen_target[world],
        "current_opposed_grasp": result.current_opposed_grasp[world],
        "grasp_confirmed": result.grasp_confirmed[world],
        "had_hand_contact": events.had_hand_contact[world],
        "forbidden_hand_contact": events.forbidden_hand_contact[world],
        "collision_buffer_overflow": events.collision_buffer_overflow[world],
        "oracle_success": result.success[world],
        "oracle_failure": result.failure[world],
        "forbidden_contact_violation": result.forbidden_contact_violation[world],
        "lateral_displacement_violation": result.lateral_displacement_violation[
            world
        ],
        "bottle_tilt_violation": result.bottle_tilt_violation[world],
        "dropped_grasp_violation": result.dropped_grasp_violation[world],
        "collision_buffer_overflow_violation": (
            result.collision_buffer_overflow_violation[world]
        ),
    }
    for name, value in boolean_values.items():
        rows[name].append(cpu_bool(value))
    rows["triangle_pair_buffer_available"].append(
        _diagnostic_scalar(
            diagnostics,
            "triangle_pair_buffer_available",
            torch=torch,
            dtype=torch.bool,
        )
    )

    integer_values = {
        "authoring_phase_before": before.phase[world],
        "authoring_phase_after": after.phase[world],
        "authoring_approach_control_steps": after.approach_control_steps[world],
        "authoring_close_control_steps": after.close_control_steps[world],
        "authoring_close_elapsed_control_steps": (
            after.close_elapsed_control_steps[world]
        ),
        "authoring_stable_grasp_control_steps": (
            after.stable_grasp_control_steps[world]
        ),
        "authoring_confirm_hold_control_steps": (
            after.confirm_hold_control_steps[world]
        ),
        "authoring_lift_control_steps": after.lift_control_steps[world],
        "authoring_success_hold_control_steps": (
            after.success_hold_control_steps[world]
        ),
        "opposed_grasp_max_consecutive_physics_frames": (
            events.opposed_grasp_max_consecutive_physics_frames[world]
        ),
        "finger_contact_counts": evaluation["finger_contact_counts"][world],
        "touching_finger_count": evaluation["touching_finger_count"][world],
        "forbidden_hand_contact_count": evaluation[
            "forbidden_hand_contact_count"
        ][world],
        "oracle_success_hold_steps": result.success_hold_steps[world],
        "oracle_drop_gap_steps": result.drop_gap_steps[world],
    }
    for name, value in integer_values.items():
        rows[name].append(cpu_int(value))
    for name in (
        "rigid_contact_capacity",
        "rigid_contact_frame_max",
        "rigid_contact_overflow_frame_count",
        "rigid_contact_overflow_excess_count",
        "triangle_pair_capacity",
        "triangle_pair_frame_max",
        "triangle_pair_overflow_frame_count",
        "triangle_pair_overflow_excess_count",
    ):
        rows[name].append(
            _diagnostic_scalar(diagnostics, name, torch=torch, dtype=torch.int64)
        )


def _failure_reason_from_tensors(tensors: Mapping[str, Any]) -> str:
    checks = (
        ("forbidden_contact", "forbidden_contact_violation"),
        ("lateral_displacement", "lateral_displacement_violation"),
        ("bottle_tilt", "bottle_tilt_violation"),
        ("dropped_grasp", "dropped_grasp_violation"),
        ("collision_buffer_overflow", "collision_buffer_overflow_violation"),
    )
    reasons = [name for name, key in checks if bool(tensors[key][-1])]
    return "+".join(reasons) if reasons else "oracle_failure"


def _outcome_from_tensors(
    tensors: Mapping[str, Any],
) -> tuple[str, bool, bool, str | None]:
    success = bool(tensors["oracle_success"][-1])
    failure = bool(tensors["oracle_failure"][-1])
    if success and failure:
        raise ValueError("oracle success and failure cannot both be true")
    if success:
        return "success", True, False, None
    if failure:
        return "failure", True, False, _failure_reason_from_tensors(tensors)
    return "timeout", False, True, None


def _build_newton_env_config(runtime: Any, env_config_type: Any) -> Any:
    return env_config_type(
        task_mode="grasp_lift_green_bottle",
        num_envs=runtime.num_envs,
        device=runtime.device,
        max_episode_steps=runtime.max_episode_steps,
        obs_mode="state",
        reward_mode="none",
        render_images=False,
        camera_textures=False,
        load_scene_visuals=False,
        hydroelastic_contacts=False,
        capture_graph=False,
        terminate_on_success=False,
        terminate_on_fail=False,
        rigid_contacts_per_env=runtime.rigid_contacts_per_env,
        triangle_pairs_per_env=runtime.triangle_pairs_per_env,
    )


def _split_assignments(
    trajectory_groups: Mapping[str, str], *, validation_fraction: float
) -> dict[str, str]:
    return assign_grasp_lift_pilot_splits(
        trajectory_groups,
        validation_fraction=validation_fraction,
    )


def main() -> None:
    args = _parse_args()
    config_path = Path(args.config).expanduser().resolve()
    config = load_grasp_lift_pilot_config(config_path)
    target = Path(args.output).expanduser().resolve()
    if target.exists():
        raise FileExistsError(f"pilot output already exists: {target}")

    implementation = capture_stage0_implementation_fingerprint(
        Path(__file__).resolve().parents[1]
    )
    affinity = _restrict_cpu_threads(config.runtime.cpu_threads)

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
        GraspLiftAuthoringPhase,
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

    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(f".{target.name}.partial-{uuid.uuid4().hex}")
    partial.mkdir()
    shards = partial / "shards"
    shards.mkdir()
    started = time.perf_counter()
    env_config = _build_newton_env_config(
        config.runtime,
        GrootNewtonEnvConfig,
    )

    env = None
    try:
        env = GrootNewtonEnv(env_config)
        expected_environment = grasp_lift_pilot_environment_config_record(
            config.runtime
        )
        if canonical_fingerprint(asdict(env_config)) != canonical_fingerprint(
            expected_environment
        ):
            raise RuntimeError("Newton environment config drifted from pilot contract")
        expected_task_metadata = grasp_lift_pilot_task_metadata_record()
        if canonical_fingerprint(env.task_metadata) != canonical_fingerprint(
            expected_task_metadata
        ):
            raise RuntimeError("Newton task metadata drifted from pilot contract")
        oracle_record = grasp_lift_pilot_oracle_record()
        simulator_record = {
            "action_schema_sha256": DEFAULT_GRASP_LIFT_ACTION_SCHEMA.sha256,
            "authoring_mode_sha256": DEFAULT_GRASP_LIFT_AUTHORING_MODE.sha256,
            "environment_config": expected_environment,
            "implementation_sha256": implementation.sha256,
            "oracle": oracle_record,
            "oracle_id": GRASP_LIFT_ORACLE_ID,
            "schema_id": "rexpolicy/stage0-grasp-lift-newton-runtime/v1",
            "task_metadata": expected_task_metadata,
        }
        simulator_sha256 = canonical_fingerprint(simulator_record)
        adapter = Stage0StateOnlyEnv(env)
        expected_mask = torch.tensor(
            GRASP_LIFT_EFFECTIVE_ACTION_MASK,
            dtype=torch.bool,
            device=config.runtime.device,
        )
        if not torch.equal(env.effective_action_mask_torch(), expected_mask):
            raise RuntimeError("Newton Grasp-Lift effective action mask changed")
        if (
            env.frames_per_action
            != DEFAULT_GRASP_LIFT_AUTHORING_MODE.full_opposed_physics_frames
        ):
            raise RuntimeError("Grasp-Lift frames-per-action contract changed")

        corpus: list[tuple[Any, Any]] = []
        member_records: list[dict[str, Any]] = []
        outcome_counts = {item.value: 0 for item in Stage0Outcome}
        safety_violation_count = 0
        world_steps = 0
        batches = config.runtime.reset_groups // config.runtime.num_envs
        for batch_index in range(batches):
            first_group = batch_index * config.runtime.num_envs
            seeds = tuple(
                config.runtime.seed + first_group + world
                for world in range(config.runtime.num_envs)
            )
            state, _ = adapter.reset(seed=list(seeds))
            controller = GraspLiftAuthoringController()
            controller.reset(state, env.hold_action_torch().clone())
            oracle = GraspLiftSuccessOracle(
                config.runtime.num_envs,
                lift_height_m=oracle_record["lift_height_m"],
                success_hold_control_steps=oracle_record[
                    "success_hold_control_steps"
                ],
                lateral_displacement_limit_m=oracle_record[
                    "lateral_displacement_limit_m"
                ],
                bottle_tilt_limit_rad=oracle_record[
                    "bottle_tilt_limit_rad"
                ],
                grasp_confirm_physics_frames=oracle_record[
                    "grasp_confirm_physics_frames"
                ],
                drop_confirm_control_steps=oracle_record[
                    "drop_confirm_control_steps"
                ],
            )
            oracle.reset(state)
            reset_axis = state[
                :, DEFAULT_STAGE0_STATE_SCHEMA.slice("object_to_goal")
            ].clone()
            reset_axis /= torch.linalg.vector_norm(
                reset_axis, dim=-1, keepdim=True
            )
            state_rows: list[list[Any]] = [
                [state[world].detach().cpu().clone()]
                for world in range(config.runtime.num_envs)
            ]
            action_rows: list[list[Any]] = [
                [] for _ in range(config.runtime.num_envs)
            ]
            evidence_rows: list[dict[str, list[Any]]] = [
                {name: [] for name in GRASP_LIFT_EVIDENCE_TENSOR_SPECS}
                for _ in range(config.runtime.num_envs)
            ]
            latest = None
            for _ in range(config.runtime.max_episode_steps):
                active = ~controller.terminal_mask
                before = controller.snapshot()
                proposal = controller.propose(state)
                projected = env.project_effective_action_torch(proposal)
                transition = adapter.step(projected)
                executed = env.effective_action_torch().detach().clone()
                if not torch.equal(executed, projected):
                    raise RuntimeError("executed action drifted after projection")
                if bool(torch.count_nonzero(transition.reward)):
                    raise RuntimeError("reward_mode=none produced non-zero reward")
                events = grasp_lift_transient_events_from_info(
                    transition.info,
                    count=config.runtime.num_envs,
                    device=state.device,
                )
                if bool(events.collision_buffer_overflow.any()):
                    raise RuntimeError(
                        "collision buffer overflow invalidated the pilot batch"
                    )
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
                for world in torch.nonzero(
                    active, as_tuple=False
                ).flatten().tolist():
                    action_rows[world].append(
                        executed[world].detach().cpu().clone()
                    )
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
                raise RuntimeError("pilot batch executed no transitions")

            for world in range(config.runtime.num_envs):
                tensors = {
                    name: torch.stack(evidence_rows[world][name])
                    for name in GRASP_LIFT_EVIDENCE_TENSOR_SPECS
                }
                (
                    outcome_name,
                    terminated,
                    truncated,
                    failure_reason,
                ) = _outcome_from_tensors(tensors)
                outcome = Stage0Outcome(outcome_name)
                outcome_counts[outcome.value] += 1
                final_violation_keys = (
                    "forbidden_contact_violation",
                    "lateral_displacement_violation",
                    "bottle_tilt_violation",
                    "dropped_grasp_violation",
                    "collision_buffer_overflow_violation",
                )
                safety_violation_count += int(
                    any(bool(tensors[key][-1]) for key in final_violation_keys)
                )
                group_index = first_group + world
                reset_group_id = f"grasp-reset-g{group_index:06d}-s{seeds[world]:010d}"
                trajectory_id = (
                    f"grasp-g{group_index:06d}-thumb-index-side-"
                    f"h{DEFAULT_GRASP_LIFT_AUTHORING_MODE.sha256[:12]}-"
                    f"s{seeds[world]:010d}"
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
                        action_schema_sha256=(
                            DEFAULT_GRASP_LIFT_ACTION_SCHEMA.sha256
                        ),
                        generation=0,
                        rank=0,
                        world=world,
                        reset_seed=seeds[world],
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
                        reset_lift_axis=tuple(
                            float(value) for value in reset_axis[world]
                        ),
                        implementation_sha256=implementation.sha256,
                    ),
                    tensors=tensors,
                )
                trajectory_descriptor = write_trajectory_shard(
                    shards, trajectory
                )
                evidence_descriptor = write_grasp_lift_evidence_sidecar(
                    shards, evidence, trajectory=trajectory
                )
                restored_trajectory = load_trajectory_shard(
                    trajectory_descriptor
                )
                restored_evidence = load_grasp_lift_evidence_sidecar(
                    evidence_descriptor, trajectory=restored_trajectory
                )
                corpus.append((restored_trajectory, restored_evidence))
                member_records.append(
                    {
                        "evidence_descriptor": str(
                            evidence_descriptor.relative_to(partial)
                        ),
                        "evidence_sha256": grasp_lift_evidence_sha256(evidence),
                        "final_authoring_phase": int(
                            tensors["authoring_phase_after"][-1]
                        ),
                        "outcome": outcome.value,
                        "reset_group_id": reset_group_id,
                        "reset_seed": seeds[world],
                        "trajectory_descriptor": str(
                            trajectory_descriptor.relative_to(partial)
                        ),
                        "trajectory_id": trajectory_id,
                        "trajectory_sha256": trajectory_sha256(trajectory),
                    }
                )

        split_by_id = _split_assignments(
            {
                record["trajectory_id"]: record["reset_group_id"]
                for record in member_records
            },
            validation_fraction=config.acceptance.validation_fraction,
        )
        for record in member_records:
            record["split"] = split_by_id[record["trajectory_id"]]
        split_counts = {
            split: {
                outcome: sum(
                    1
                    for record in member_records
                    if record["split"] == split and record["outcome"] == outcome
                )
                for outcome in outcome_counts
            }
            for split in ("train", "validation")
        }
        success_rate = outcome_counts[Stage0Outcome.SUCCESS.value] / float(
            config.runtime.reset_groups
        )
        acceptance_passed = bool(
            success_rate >= config.acceptance.minimum_success_rate
            and (
                not config.acceptance.require_zero_oracle_failures
                or outcome_counts[Stage0Outcome.FAILURE.value] == 0
            )
            and (
                not config.acceptance.require_zero_safety_violations
                or safety_violation_count == 0
            )
        )
        elapsed = time.perf_counter() - started
        split_policy = grasp_lift_pilot_split_policy()
        manifest = seal_grasp_lift_pilot_record(
            {
                "acceptance": {
                    "passed": acceptance_passed,
                    "policy": config.acceptance.to_record(),
                    "safety_violation_count": safety_violation_count,
                    "success_rate": success_rate,
                },
                "action_schema": DEFAULT_GRASP_LIFT_ACTION_SCHEMA.to_record(),
                "action_schema_sha256": DEFAULT_GRASP_LIFT_ACTION_SCHEMA.sha256,
                "authoring_mode": DEFAULT_GRASP_LIFT_AUTHORING_MODE.to_record(),
                "authoring_mode_sha256": (
                    DEFAULT_GRASP_LIFT_AUTHORING_MODE.sha256
                ),
                "composite_corpus_sha256": (
                    grasp_lift_composite_corpus_sha256(corpus)
                ),
                "cpu_affinity": list(affinity),
                "elapsed_seconds": elapsed,
                "experiment_sha256": config.sha256,
                "implementation": implementation.to_record(),
                "implementation_sha256": implementation.sha256,
                "locked_test": {
                    "artifact": None,
                    "consumed": False,
                    "policy": config.acceptance.locked_test_policy,
                },
                "members": sorted(
                    member_records, key=lambda record: record["trajectory_id"]
                ),
                "outcome_counts": outcome_counts,
                "schema_id": GRASP_LIFT_PILOT_OUTPUT_SCHEMA_ID,
                "simulator_record": simulator_record,
                "simulator_sha256": simulator_sha256,
                "split_counts": split_counts,
                "split_policy": split_policy,
                "split_policy_sha256": canonical_fingerprint(split_policy),
                "state_view": DEFAULT_GRASP_LIFT_STATE_VIEW.to_record(),
                "state_view_sha256": DEFAULT_GRASP_LIFT_STATE_VIEW.sha256,
                "visuals": {
                    "camera_textures": False,
                    "scene_asset": None,
                    "scene_visuals": False,
                },
                "world_steps": world_steps,
            }
        )
        _write_json(partial / "config.json", config.to_record())
        _write_json(partial / "manifest.json", manifest)
        status = seal_grasp_lift_pilot_record(
            {
                "acceptance_passed": acceptance_passed,
                "complete": True,
                "manifest_sha256": manifest["self_sha256"],
                "outcome_counts": outcome_counts,
                "schema_id": GRASP_LIFT_PILOT_STATUS_SCHEMA_ID,
            }
        )
        _write_json(partial / "status.json", status)
        commit = seal_grasp_lift_pilot_record(
            {
                "manifest_file": "manifest.json",
                "manifest_sha256": manifest["self_sha256"],
                "config_file": "config.json",
                "config_sha256": config.sha256,
                "schema_id": GRASP_LIFT_PILOT_COMMIT_SCHEMA_ID,
                "status_file": "status.json",
                "status_sha256": status["self_sha256"],
            }
        )
        _write_json(partial / "commit.json", commit)
        _fsync_directory(partial)
        load_grasp_lift_pilot_manifest(partial)
        partial.rename(target)
        _fsync_directory(target.parent)
        print(json.dumps(manifest, indent=2, sort_keys=True))
        if not acceptance_passed:
            raise SystemExit(2)
    except Exception as error:
        if partial.exists() and not (partial / "error.json").exists():
            try:
                _write_json(
                    partial / "error.json",
                    {
                        "error_type": type(error).__name__,
                        "schema_id": GRASP_LIFT_PILOT_OUTPUT_SCHEMA_ID,
                    },
                )
            except Exception:
                pass
        raise
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    main()
