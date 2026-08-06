"""Strict configuration contract for reward-free Grasp-Lift pilot authoring."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from rexpolicy.stage0.types import canonical_fingerprint


GRASP_LIFT_PILOT_CONFIG_SCHEMA_VERSION = 1
GRASP_LIFT_PILOT_CONFIG_SCHEMA_ID = "rexpolicy/stage0-grasp-lift-pilot/v1"
GRASP_LIFT_PILOT_OUTPUT_SCHEMA_ID = "rexpolicy/stage0-grasp-lift-pilot-output/v5"
GRASP_LIFT_PILOT_STATUS_SCHEMA_ID = "rexpolicy/stage0-grasp-lift-pilot-status/v1"
GRASP_LIFT_PILOT_COMMIT_SCHEMA_ID = "rexpolicy/stage0-grasp-lift-pilot-commit/v5"
GRASP_LIFT_PILOT_VALIDATION_CLAIM_SCHEMA_ID = (
    "rexpolicy/stage0-grasp-lift-validation-claim/v1"
)
GRASP_LIFT_PILOT_FORMAL_CONFIG_SHA256 = (
    "cdf0ca2c3b11325c804852f464930df1d31a115c6fae8136d4314d4295e6cc8f"
)
GRASP_LIFT_PILOT_SPLIT_NAMESPACE = "rexpolicy/grasp-lift-split/20260806-v1"
GRASP_LIFT_PILOT_VALIDATION_FRACTION = 0.25
GRASP_LIFT_PILOT_SPLIT_POLICY_ID = (
    "rexpolicy/stage0-grasp-lift-reset-group-split/v1"
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SAFE_PILOT_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_PILOT_OUTCOMES = ("failure", "success", "timeout")
_PILOT_MANIFEST_KEYS = {
    "acceptance",
    "action_schema",
    "action_schema_sha256",
    "authoring_mode",
    "authoring_mode_sha256",
    "composite_corpus_sha256",
    "cpu_affinity",
    "elapsed_seconds",
    "experiment_sha256",
    "implementation",
    "implementation_sha256",
    "locked_test",
    "authoring_lineage",
    "authoring_lineage_sha256",
    "members",
    "outcome_counts",
    "schema_id",
    "self_sha256",
    "simulator_record",
    "simulator_sha256",
    "split_counts",
    "split_policy",
    "split_policy_sha256",
    "state_view",
    "state_view_sha256",
    "validation_claim",
    "validation_claim_sha256",
    "visuals",
    "world_steps",
}
_PILOT_MEMBER_KEYS = {
    "evidence_descriptor",
    "evidence_sha256",
    "final_authoring_phase",
    "formal_gate_eligible",
    "outcome",
    "reset_group_id",
    "reset_seed",
    "safety_violation",
    "split",
    "trajectory_descriptor",
    "trajectory_id",
    "trajectory_sha256",
    "terminal_transition_from_phase",
}

_BASELINE_TRAIN_SUCCESS_SEEDS = (
    7003,
    7004,
    7007,
    7008,
    7010,
    7015,
    7016,
    7017,
    7018,
    7022,
    7023,
    7024,
    7027,
    7029,
    7030,
)
_BASELINE_TRAIN_TIMEOUT_SEEDS = (
    7000,
    7002,
    7006,
    7011,
    7012,
    7013,
    7020,
    7021,
    7028,
)
_ENGINEERING_SMOKE_EXCLUDED_VALIDATION_SEEDS = (7001, 7005)
_BASELINE_VALIDATION_SUCCESS_SEEDS = (7009, 7019, 7025, 7026)
_BASELINE_VALIDATION_TIMEOUT_SEEDS = (7014, 7031)


def _exact_mapping(value: Any, keys: set[str], context: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must be an object")
    actual = set(value)
    if actual != keys:
        raise ValueError(
            f"{context} keys changed: missing={sorted(keys - actual)}, "
            f"unexpected={sorted(actual - keys)}"
        )
    return dict(value)


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _fraction(value: Any, name: str, *, allow_one: bool = True) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    upper_ok = result <= 1.0 if allow_one else result < 1.0
    if not math.isfinite(result) or result <= 0.0 or not upper_ok:
        upper = "]" if allow_one else ")"
        raise ValueError(f"{name} must be finite in (0, 1{upper}")
    return result


def _require_sha256(value: Any, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _parse_json_object(path: Path, context: str) -> dict[str, Any]:
    if path.is_symlink():
        raise ValueError(f"{context} cannot be a symlink")
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=lambda constant: (_ for _ in ()).throw(
                ValueError(f"invalid JSON constant: {constant}")
            ),
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"cannot read {context} {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{context} must contain an object")
    return value


@dataclass(frozen=True)
class GraspLiftPilotRuntimeConfig:
    device: str
    cpu_threads: int
    num_envs: int
    reset_groups: int
    seed: int
    max_episode_steps: int
    rigid_contacts_per_env: int
    triangle_pairs_per_env: int

    def __post_init__(self) -> None:
        if not isinstance(self.device, str) or not self.device.startswith("cuda:"):
            raise ValueError("runtime.device must name one CUDA device")
        for name in (
            "cpu_threads",
            "num_envs",
            "reset_groups",
            "max_episode_steps",
            "rigid_contacts_per_env",
            "triangle_pairs_per_env",
        ):
            _positive_int(getattr(self, name), f"runtime.{name}")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("runtime.seed must be a non-negative integer")
        if self.reset_groups % self.num_envs:
            raise ValueError("runtime.reset_groups must be divisible by num_envs")

    @classmethod
    def from_mapping(cls, value: Any) -> GraspLiftPilotRuntimeConfig:
        keys = {
            "cpu_threads",
            "device",
            "max_episode_steps",
            "num_envs",
            "reset_groups",
            "rigid_contacts_per_env",
            "seed",
            "triangle_pairs_per_env",
        }
        return cls(**_exact_mapping(value, keys, "pilot runtime config"))

    def to_record(self) -> dict[str, Any]:
        return {
            "cpu_threads": self.cpu_threads,
            "device": self.device,
            "max_episode_steps": self.max_episode_steps,
            "num_envs": self.num_envs,
            "reset_groups": self.reset_groups,
            "rigid_contacts_per_env": self.rigid_contacts_per_env,
            "seed": self.seed,
            "triangle_pairs_per_env": self.triangle_pairs_per_env,
        }


@dataclass(frozen=True)
class GraspLiftPilotAcceptanceConfig:
    minimum_success_rate: float
    validation_fraction: float
    locked_test_policy: str
    require_zero_oracle_failures: bool
    require_zero_safety_violations: bool

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "minimum_success_rate",
            _fraction(self.minimum_success_rate, "acceptance.minimum_success_rate"),
        )
        validation_fraction = _fraction(
            self.validation_fraction,
            "acceptance.validation_fraction",
            allow_one=False,
        )
        if not math.isclose(
            validation_fraction,
            GRASP_LIFT_PILOT_VALIDATION_FRACTION,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            raise ValueError(
                "pilot validation_fraction is fixed at "
                f"{GRASP_LIFT_PILOT_VALIDATION_FRACTION}"
            )
        object.__setattr__(
            self,
            "validation_fraction",
            validation_fraction,
        )
        if self.locked_test_policy != "not_created":
            raise ValueError("pilot locked_test_policy must be 'not_created'")
        for name in (
            "require_zero_oracle_failures",
            "require_zero_safety_violations",
        ):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"acceptance.{name} must be boolean")

    @classmethod
    def from_mapping(cls, value: Any) -> GraspLiftPilotAcceptanceConfig:
        keys = {
            "locked_test_policy",
            "minimum_success_rate",
            "require_zero_oracle_failures",
            "require_zero_safety_violations",
            "validation_fraction",
        }
        return cls(**_exact_mapping(value, keys, "pilot acceptance config"))

    def to_record(self) -> dict[str, Any]:
        return {
            "locked_test_policy": self.locked_test_policy,
            "minimum_success_rate": self.minimum_success_rate,
            "require_zero_oracle_failures": self.require_zero_oracle_failures,
            "require_zero_safety_violations": self.require_zero_safety_violations,
            "validation_fraction": self.validation_fraction,
        }


@dataclass(frozen=True)
class GraspLiftPilotConfig:
    runtime: GraspLiftPilotRuntimeConfig
    acceptance: GraspLiftPilotAcceptanceConfig
    schema_version: int = GRASP_LIFT_PILOT_CONFIG_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != GRASP_LIFT_PILOT_CONFIG_SCHEMA_VERSION:
            raise ValueError(
                f"schema_version must be {GRASP_LIFT_PILOT_CONFIG_SCHEMA_VERSION}"
            )
        if not isinstance(self.runtime, GraspLiftPilotRuntimeConfig):
            raise TypeError("runtime must be GraspLiftPilotRuntimeConfig")
        if not isinstance(self.acceptance, GraspLiftPilotAcceptanceConfig):
            raise TypeError("acceptance must be GraspLiftPilotAcceptanceConfig")

    @classmethod
    def from_mapping(cls, value: Any) -> GraspLiftPilotConfig:
        record = _exact_mapping(
            value,
            {"acceptance", "runtime", "schema_version"},
            "Grasp-Lift pilot config",
        )
        return cls(
            schema_version=record["schema_version"],
            runtime=GraspLiftPilotRuntimeConfig.from_mapping(record["runtime"]),
            acceptance=GraspLiftPilotAcceptanceConfig.from_mapping(
                record["acceptance"]
            ),
        )

    def to_record(self) -> dict[str, Any]:
        return {
            "acceptance": self.acceptance.to_record(),
            "runtime": self.runtime.to_record(),
            "schema_version": self.schema_version,
        }

    @property
    def sha256(self) -> str:
        return canonical_fingerprint(
            {"schema_id": GRASP_LIFT_PILOT_CONFIG_SCHEMA_ID, **self.to_record()}
        )


def load_grasp_lift_pilot_config(path: str | Path) -> GraspLiftPilotConfig:
    source = Path(path)
    value = _parse_json_object(source, "Grasp-Lift pilot config")
    return GraspLiftPilotConfig.from_mapping(value)


def grasp_lift_pilot_split_policy() -> dict[str, Any]:
    """Return the immutable train/validation assignment policy."""
    return {
        "assignment_unit": "reset_group_id",
        "count_rule": "round-half-even-clamped-to-[1,group-count-1]",
        "hash_algorithm": "sha256",
        "namespace": GRASP_LIFT_PILOT_SPLIT_NAMESPACE,
        "ordering": "sha256(namespace + ':' + reset_group_id)",
        "policy_id": GRASP_LIFT_PILOT_SPLIT_POLICY_ID,
        "validation_fraction": GRASP_LIFT_PILOT_VALIDATION_FRACTION,
    }


def grasp_lift_pilot_environment_config_record(
    runtime: GraspLiftPilotRuntimeConfig,
) -> dict[str, Any]:
    """Return every Newton environment field allowed by the pilot contract."""
    if not isinstance(runtime, GraspLiftPilotRuntimeConfig):
        raise TypeError("runtime must be GraspLiftPilotRuntimeConfig")
    return {
        "arm_action_delta": 0.1,
        "bottle_lift_height": 0.1,
        "bottle_min_xy_displacement": 0.1,
        "bottle_settle_frames": 60,
        "camera_textures": False,
        "capture_graph": False,
        "contact_max_separation": 0.0002,
        "control_hz": 10,
        "control_mode": "pd_eef_pose_abs",
        "device": runtime.device,
        "ego_height": 180,
        "ego_width": 320,
        "final_orientation_threshold_rad": math.radians(15.0),
        "final_z_threshold": 0.01,
        "finger_root_closing_sign": [1.0, 1.0, 1.0, 1.0, 1.0],
        "finger_root_load_bias": [0.0, 0.0, 0.0, 0.0, 0.0],
        "finger_root_load_scale": None,
        "goal_threshold": 0.005,
        "grasp_bottle_tilt_limit_rad": math.radians(30.0),
        "grasp_confirm_frames": 6,
        "grasp_finger_count": 2,
        "grasp_lateral_displacement_limit": 0.030,
        "grasp_lift_height": 0.030,
        "grasp_lift_hold_control_steps": 2,
        "hand_action_delta": 0.1,
        "hand_max_joint_step_rad": 0.08,
        "hydroelastic_contacts": False,
        "ik_damping_lambda": 0.02,
        "ik_iterations": 4,
        "ik_max_joint_step_rad": 0.045,
        "ik_max_rotation_step_rad": math.radians(5.0),
        "ik_max_task_step_m": 0.03,
        "ik_orientation_weight": 1.0,
        "ik_position_weight": 3.0,
        "initial_hand_q": [
            0.1848468184,
            0.3151794076,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0581703521,
            0.0262242742,
            0.1140341759,
            0.0,
        ],
        "load_scene_visuals": False,
        "max_episode_steps": runtime.max_episode_steps,
        "mujoco_nconmax": 1024,
        "mujoco_njmax": 2048,
        "num_envs": runtime.num_envs,
        "object_angular_velocity_threshold": 0.5,
        "object_linear_velocity_threshold": 0.02,
        "obs_mode": "state",
        "reach_bottle_displacement_limit": 0.010,
        "reach_goal_offset_world": [0.180, 0.080, 0.050],
        "reach_reset_xy_jitter_m": 0.010,
        "reach_reward_distance_scale": 0.030,
        "reach_success_hold_steps": 2,
        "reach_success_threshold": 0.040,
        "release_confirm_frames": 6,
        "render_images": False,
        "request_finger_root_load": False,
        "reward_mode": "none",
        "rigid_contacts_per_env": runtime.rigid_contacts_per_env,
        "scene_glb_path": None,
        "settle_confirm_frames": 12,
        "simulation_hz": 60,
        "static_velocity_threshold": 0.2,
        "substeps_per_frame": 16,
        "task_mode": "grasp_lift_green_bottle",
        "terminate_on_fail": False,
        "terminate_on_success": False,
        "transport_start_distance": 0.01,
        "triangle_pairs_per_env": runtime.triangle_pairs_per_env,
        "wrist_height": 480,
        "wrist_width": 640,
    }


def grasp_lift_pilot_task_metadata_record() -> dict[str, Any]:
    """Return the exact simulator task metadata allowed for authoring."""
    return {
        "bottle_tilt_limit_rad": math.radians(30.0),
        "effective_action": (
            "absolute_eef_xyz_and_absolute_hand_targets; reset_rotation_held"
        ),
        "effective_action_mask": (
            [True, True, True] + [False] * 6 + [True] * 10
        ),
        "instruction": (
            "grasp the green bottle with opposed finger contact and lift it "
            "while keeping the grasp stable"
        ),
        "lateral_displacement_limit_m": 0.030,
        "lift_height_m": 0.030,
        "reset_xy_jitter_m": 0.010,
        "success_authority": "independent_state_and_transient_event_oracle",
        "success_hold_control_steps": 2,
        "task_id": "grasp_lift_green_bottle/v1",
    }


def grasp_lift_pilot_oracle_record() -> dict[str, Any]:
    """Return every independent reward-free oracle threshold."""
    from rexpolicy.stage0.envs.grasp_lift_oracle import GRASP_LIFT_ORACLE_ID
    from rexpolicy.stage0.envs.state_schema import DEFAULT_STAGE0_STATE_SCHEMA

    return {
        "bottle_tilt_limit_rad": math.radians(30.0),
        "drop_confirm_control_steps": 2,
        "grasp_confirm_physics_frames": 6,
        "lateral_displacement_limit_m": 0.030,
        "lift_height_m": 0.030,
        "oracle_id": GRASP_LIFT_ORACLE_ID,
        "state_schema_sha256": DEFAULT_STAGE0_STATE_SCHEMA.sha256,
        "success_hold_control_steps": 2,
    }


def grasp_lift_pilot_authoring_lineage_record() -> dict[str, Any]:
    """Bind the close7 candidate to train-only evidence from the failed pilot."""
    from rexpolicy.stage0.envs.grasp_lift_modes import (
        DEFAULT_GRASP_LIFT_AUTHORING_MODE,
        GRASP_LIFT_AUTHORING_MODE_CLOSE4,
    )

    return {
        "candidate": {
            "mode_id": DEFAULT_GRASP_LIFT_AUTHORING_MODE.mode_id,
            "mode_sha256": DEFAULT_GRASP_LIFT_AUTHORING_MODE.sha256,
        },
        "decision_id": "stage0/grasp_lift/close7_from_train_pilot/v2",
        "intervention": {
            "baseline_value": 4,
            "candidate_value": 7,
            "field": "minimum_tentative_close_step",
            "scope": "single_authoring_mode_field",
        },
        "locked_test": {
            "artifact": None,
            "consumed": False,
            "policy": "not_created",
        },
        "parent": {
            "authoring_mode_id": GRASP_LIFT_AUTHORING_MODE_CLOSE4.mode_id,
            "authoring_mode_sha256": GRASP_LIFT_AUTHORING_MODE_CLOSE4.sha256,
            "composite_corpus_sha256": (
                "527ba0299c0b1d4eac5a397663a15406e344b8d536163f61d1d84aa6d311881c"
            ),
            "implementation_sha256": (
                "aa9d72d26b92088cb7412a3e718268c5fae3ac9388d5d5b0f9ff7b94e6aba636"
            ),
            "manifest_sha256": (
                "f1acd4f35915613ddbe8e041de1c16e7cafaa3e33a97e81b94e5d81ddcbd9187"
            ),
        },
        "schema_id": "rexpolicy/stage0-grasp-lift-authoring-lineage/v2",
        "selection_evidence": {
            "mechanism": (
                "all train successes froze at close step 7; all train timeouts "
                "froze at close step 6 and lost opposed grasp during lift"
            ),
            "minimum_former_timeout_conversions": 6,
            "require_all_baseline_successes_retained": True,
            "split": "train",
            "success_seeds": list(_BASELINE_TRAIN_SUCCESS_SEEDS),
            "timeout_seeds": list(_BASELINE_TRAIN_TIMEOUT_SEEDS),
        },
        "validation_protocol": {
            "candidate_evaluation_budget": 1,
            "engineering_smoke_exclusions": {
                "exposure_ledger": [
                    {
                        "controller_mode_sha256": (
                            "c09b682944d2ea03c91ba12ba97e0a9c5b45ea985da6a5543e05604a053f1c18"
                        ),
                        "date": "2026-08-06",
                        "evidence_artifact": None,
                        "git_head": (
                            "c2407c634766804e6eb6cc8bee0af893e0aa60a3"
                        ),
                        "mode_patch_sha256": (
                            "ea9e7f3d054ec6afce6c2b48bf28f70331b367dd42369930aae1dfe378c6c806"
                        ),
                        "observed_outcomes": {
                            "7001": "success",
                            "7005": "success",
                        },
                        "run_id": "close7-precommit-newton-smoke-1",
                        "seeds": [7001, 7005],
                    },
                    {
                        "controller_mode_sha256": (
                            "c09b682944d2ea03c91ba12ba97e0a9c5b45ea985da6a5543e05604a053f1c18"
                        ),
                        "date": "2026-08-06",
                        "evidence_artifact": None,
                        "git_head": (
                            "c2407c634766804e6eb6cc8bee0af893e0aa60a3"
                        ),
                        "mode_patch_sha256": (
                            "ea9e7f3d054ec6afce6c2b48bf28f70331b367dd42369930aae1dfe378c6c806"
                        ),
                        "observed_outcomes": {
                            "7001": "success",
                            "7005": "success",
                        },
                        "run_id": "close7-precommit-newton-smoke-2",
                        "seeds": [7001, 7005],
                    },
                ],
                "reason": (
                    "close7 engineering smoke ran seeds 7000 through 7007 "
                    "before the formal claim was introduced"
                ),
                "seeds": list(_ENGINEERING_SMOKE_EXCLUDED_VALIDATION_SEEDS),
            },
            "minimum_former_timeout_conversions": 2,
            "formal_experiment_sha256": (
                GRASP_LIFT_PILOT_FORMAL_CONFIG_SHA256
            ),
            "require_all_baseline_successes_retained": True,
            "require_zero_lift_to_rejected": True,
            "split": "validation",
            "success_seeds": list(_BASELINE_VALIDATION_SUCCESS_SEEDS),
            "timeout_seeds": list(_BASELINE_VALIDATION_TIMEOUT_SEEDS),
        },
    }


def grasp_lift_pilot_formal_gate_eligible(reset_seed: int) -> bool:
    """Exclude validation seeds already exposed to the close7 candidate."""
    if isinstance(reset_seed, bool) or not isinstance(reset_seed, int):
        raise TypeError("reset_seed must be an integer")
    return reset_seed not in _ENGINEERING_SMOKE_EXCLUDED_VALIDATION_SEEDS


def evaluate_grasp_lift_close7_authoring_gate(
    members: list[Mapping[str, Any]],
) -> dict[str, Any]:
    """Evaluate the precommitted paired train and one-shot validation gates."""
    if not isinstance(members, list):
        raise TypeError("members must be a list")
    by_seed: dict[int, Mapping[str, Any]] = {}
    for member in members:
        if not isinstance(member, Mapping):
            raise ValueError("authoring gate member must be an object")
        seed = member.get("reset_seed")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed in by_seed:
            raise ValueError("authoring gate member seed is invalid or duplicated")
        if member.get("outcome") not in _PILOT_OUTCOMES:
            raise ValueError("authoring gate member outcome changed")
        if member.get("split") not in {"train", "validation"}:
            raise ValueError("authoring gate member split changed")
        eligibility = member.get("formal_gate_eligible")
        if not isinstance(eligibility, bool):
            raise ValueError("authoring gate member eligibility changed")
        phase = member.get("final_authoring_phase")
        terminal_from_phase = member.get("terminal_transition_from_phase")
        if (
            isinstance(phase, bool)
            or not isinstance(phase, int)
            or isinstance(terminal_from_phase, bool)
            or not isinstance(terminal_from_phase, int)
        ):
            raise ValueError("authoring gate member phase changed")
        by_seed[seed] = member
    expected_seeds = set(
        _BASELINE_TRAIN_SUCCESS_SEEDS
        + _BASELINE_TRAIN_TIMEOUT_SEEDS
        + _ENGINEERING_SMOKE_EXCLUDED_VALIDATION_SEEDS
        + _BASELINE_VALIDATION_SUCCESS_SEEDS
        + _BASELINE_VALIDATION_TIMEOUT_SEEDS
    )
    if set(by_seed) != expected_seeds:
        raise ValueError("authoring gate seed coverage changed")

    groups = (
        (_BASELINE_TRAIN_SUCCESS_SEEDS, "train"),
        (_BASELINE_TRAIN_TIMEOUT_SEEDS, "train"),
        (_ENGINEERING_SMOKE_EXCLUDED_VALIDATION_SEEDS, "validation"),
        (_BASELINE_VALIDATION_SUCCESS_SEEDS, "validation"),
        (_BASELINE_VALIDATION_TIMEOUT_SEEDS, "validation"),
    )
    for seeds, expected_split in groups:
        if any(by_seed[seed]["split"] != expected_split for seed in seeds):
            raise ValueError("authoring gate split assignment changed")
    for seed, member in by_seed.items():
        expected_eligibility = grasp_lift_pilot_formal_gate_eligible(seed)
        if member["formal_gate_eligible"] is not expected_eligibility:
            raise ValueError("authoring gate eligibility changed")
    def is_success(seed: int) -> bool:
        return by_seed[seed]["outcome"] == "success"

    train_retained = sum(is_success(seed) for seed in _BASELINE_TRAIN_SUCCESS_SEEDS)
    train_converted = sum(is_success(seed) for seed in _BASELINE_TRAIN_TIMEOUT_SEEDS)
    validation_retained = sum(
        is_success(seed) for seed in _BASELINE_VALIDATION_SUCCESS_SEEDS
    )
    validation_converted = sum(
        is_success(seed) for seed in _BASELINE_VALIDATION_TIMEOUT_SEEDS
    )
    validation_lift_to_rejected = sum(
        member["split"] == "validation"
        and member["formal_gate_eligible"]
        and member["final_authoring_phase"] == 7
        and member["terminal_transition_from_phase"] in {3, 4}
        for member in members
    )
    passed = bool(
        train_retained == len(_BASELINE_TRAIN_SUCCESS_SEEDS)
        and train_converted >= 6
        and validation_retained == len(_BASELINE_VALIDATION_SUCCESS_SEEDS)
        and validation_converted == len(_BASELINE_VALIDATION_TIMEOUT_SEEDS)
        and validation_lift_to_rejected == 0
    )
    return {
        "passed": passed,
        "schema_id": "rexpolicy/stage0-grasp-lift-close7-paired-gate/v2",
        "train": {
            "baseline_success_regressions": (
                len(_BASELINE_TRAIN_SUCCESS_SEEDS) - train_retained
            ),
            "baseline_success_retained": train_retained,
            "baseline_success_required": len(_BASELINE_TRAIN_SUCCESS_SEEDS),
            "former_timeout_converted": train_converted,
            "former_timeout_conversion_required": 6,
            "former_timeout_total": len(_BASELINE_TRAIN_TIMEOUT_SEEDS),
        },
        "validation": {
            "baseline_success_regressions": (
                len(_BASELINE_VALIDATION_SUCCESS_SEEDS) - validation_retained
            ),
            "baseline_success_retained": validation_retained,
            "baseline_success_required": len(
                _BASELINE_VALIDATION_SUCCESS_SEEDS
            ),
            "former_timeout_converted": validation_converted,
            "former_timeout_conversion_required": 2,
            "former_timeout_total": len(_BASELINE_VALIDATION_TIMEOUT_SEEDS),
            "engineering_smoke_excluded_seeds": list(
                _ENGINEERING_SMOKE_EXCLUDED_VALIDATION_SEEDS
            ),
            "engineering_smoke_excluded_total": len(
                _ENGINEERING_SMOKE_EXCLUDED_VALIDATION_SEEDS
            ),
            "lift_to_rejected": validation_lift_to_rejected,
            "lift_to_rejected_required": 0,
        },
    }


def grasp_lift_pilot_validation_claim_identity() -> dict[str, Any]:
    """Return the stable identity consumed by one formal validation attempt."""
    lineage = grasp_lift_pilot_authoring_lineage_record()
    protocol = lineage["validation_protocol"]
    return {
        "candidate_mode_sha256": lineage["candidate"]["mode_sha256"],
        "coordination_scope": "single_git_common_dir",
        "evaluation_budget": protocol["candidate_evaluation_budget"],
        "formal_experiment_sha256": protocol["formal_experiment_sha256"],
        "parent_manifest_sha256": lineage["parent"]["manifest_sha256"],
        "schema_id": "rexpolicy/stage0-grasp-lift-validation-claim-identity/v1",
        "validation_seeds": sorted(
            protocol["success_seeds"] + protocol["timeout_seeds"]
        ),
    }


def grasp_lift_pilot_validation_claim_record(
    *,
    experiment_sha256: str,
    implementation_sha256: str,
) -> dict[str, Any]:
    """Build the deterministic record written with exclusive-create semantics."""
    _require_sha256(experiment_sha256, "validation claim experiment hash")
    _require_sha256(implementation_sha256, "validation claim implementation hash")
    if experiment_sha256 != GRASP_LIFT_PILOT_FORMAL_CONFIG_SHA256:
        raise ValueError("validation claim requires the formal pilot config")
    identity = grasp_lift_pilot_validation_claim_identity()
    return {
        "experiment_sha256": experiment_sha256,
        "identity": identity,
        "identity_sha256": canonical_fingerprint(identity),
        "implementation_sha256": implementation_sha256,
        "schema_id": GRASP_LIFT_PILOT_VALIDATION_CLAIM_SCHEMA_ID,
    }


def verify_grasp_lift_pilot_formal_config(
    config: GraspLiftPilotConfig,
) -> None:
    """Fail before claim creation unless every scientific/runtime field is fixed."""
    if not isinstance(config, GraspLiftPilotConfig):
        raise TypeError("config must be GraspLiftPilotConfig")
    if config.sha256 != GRASP_LIFT_PILOT_FORMAL_CONFIG_SHA256:
        raise ValueError("pilot config differs from the pre-registered formal config")


def assign_grasp_lift_pilot_splits(
    trajectory_groups: Mapping[str, str],
    *,
    validation_fraction: float,
) -> dict[str, str]:
    """Assign complete reset groups using only the immutable split namespace."""
    if not isinstance(trajectory_groups, Mapping):
        raise TypeError("trajectory_groups must be a mapping")
    fraction = _fraction(
        validation_fraction,
        "pilot validation_fraction",
        allow_one=False,
    )
    if not math.isclose(
        fraction,
        GRASP_LIFT_PILOT_VALIDATION_FRACTION,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise ValueError("pilot validation fraction changed")
    for trajectory_id, group_id in trajectory_groups.items():
        if not isinstance(trajectory_id, str) or not trajectory_id:
            raise ValueError("pilot trajectory IDs must be non-empty strings")
        if not isinstance(group_id, str) or not group_id:
            raise ValueError("pilot reset group IDs must be non-empty strings")
    groups = set(trajectory_groups.values())
    if len(groups) < 2:
        raise ValueError("pilot split requires at least two reset groups")
    ordered_groups = sorted(
        groups,
        key=lambda group_id: (
            hashlib.sha256(
                f"{GRASP_LIFT_PILOT_SPLIT_NAMESPACE}:{group_id}".encode("utf-8")
            ).digest(),
            group_id,
        ),
    )
    validation_count = max(1, round(len(ordered_groups) * fraction))
    validation_count = min(validation_count, len(ordered_groups) - 1)
    validation_groups = set(ordered_groups[:validation_count])
    return {
        trajectory_id: (
            "validation" if group_id in validation_groups else "train"
        )
        for trajectory_id, group_id in trajectory_groups.items()
    }


def seal_grasp_lift_pilot_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """Add a self hash computed over every other record field."""
    if not isinstance(record, Mapping):
        raise TypeError("record must be a mapping")
    if "self_sha256" in record:
        raise ValueError("record is already sealed")
    sealed = dict(record)
    sealed["self_sha256"] = canonical_fingerprint(sealed)
    return sealed


def verify_grasp_lift_pilot_record(record: Mapping[str, Any], context: str) -> None:
    """Recompute and verify a sealed pilot record."""
    if not isinstance(record, Mapping):
        raise ValueError(f"{context} must be an object")
    expected = record.get("self_sha256")
    if not isinstance(expected, str):
        raise ValueError(f"{context} is missing self_sha256")
    unsealed = dict(record)
    unsealed.pop("self_sha256")
    if canonical_fingerprint(unsealed) != expected:
        raise ValueError(f"{context} self_sha256 mismatch")


def _read_pilot_json(path: Path, context: str) -> dict[str, Any]:
    return _parse_json_object(path, context)


def _pilot_count_record(value: Any, context: str) -> dict[str, int]:
    record = _exact_mapping(value, set(_PILOT_OUTCOMES), context)
    for name, count in record.items():
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError(f"{context}.{name} must be a non-negative integer")
    return record


def _pilot_member_path(
    root: Path,
    value: Any,
    *,
    expected_name: str,
    context: str,
) -> Path:
    if not isinstance(value, str):
        raise ValueError(f"{context} must be a relative path")
    relative = Path(value)
    expected = Path("shards") / expected_name
    if relative != expected or relative.is_absolute():
        raise ValueError(f"{context} changed from {expected.as_posix()!r}")
    shards = root / "shards"
    candidate = root / relative
    if shards.is_symlink() or not shards.is_dir():
        raise ValueError("pilot shards directory is missing or a symlink")
    if candidate.is_symlink() or not candidate.is_file():
        raise ValueError(f"{context} is missing or a symlink")
    return candidate


def _implementation_fingerprint_from_record(value: Any) -> Any:
    from rexpolicy.stage0.implementation_fingerprint import (
        STAGE0_IMPLEMENTATION_DISTRIBUTIONS,
        STAGE0_IMPLEMENTATION_FINGERPRINT_SCHEMA_ID,
        STAGE0_IMPLEMENTATION_FINGERPRINT_SCHEMA_VERSION,
        Stage0DistributionVersion,
        Stage0ImplementationFingerprint,
    )

    record = _exact_mapping(
        value,
        {
            "distributions",
            "git",
            "python",
            "schema_id",
            "schema_version",
            "version_source",
        },
        "pilot implementation",
    )
    distributions = _exact_mapping(
        record["distributions"],
        {component for component, _ in STAGE0_IMPLEMENTATION_DISTRIBUTIONS},
        "pilot implementation distributions",
    )
    items = []
    for component, distribution in STAGE0_IMPLEMENTATION_DISTRIBUTIONS:
        item = _exact_mapping(
            distributions[component],
            {"distribution", "status", "version"},
            f"pilot implementation distributions.{component}",
        )
        if item["distribution"] != distribution:
            raise ValueError("pilot implementation distribution changed")
        expected_status = "missing" if item["version"] is None else "installed"
        if item["status"] != expected_status:
            raise ValueError("pilot implementation distribution status changed")
        items.append(
            Stage0DistributionVersion(
                component=component,
                distribution=distribution,
                version=item["version"],
            )
        )
    git = _exact_mapping(
        record["git"],
        {"head", "worktree_policy", "worktree_status"},
        "pilot implementation git",
    )
    python = _exact_mapping(
        record["python"],
        {"implementation", "version"},
        "pilot implementation python",
    )
    if (
        record["schema_id"] != STAGE0_IMPLEMENTATION_FINGERPRINT_SCHEMA_ID
        or record["schema_version"]
        != STAGE0_IMPLEMENTATION_FINGERPRINT_SCHEMA_VERSION
        or record["version_source"] != "importlib.metadata.version/v1"
    ):
        raise ValueError("pilot implementation schema changed")
    fingerprint = Stage0ImplementationFingerprint(
        git_head=git["head"],
        python_implementation=python["implementation"],
        python_version=python["version"],
        distributions=tuple(items),
    )
    if fingerprint.to_record() != record:
        raise ValueError("pilot implementation fingerprint record changed")
    return fingerprint


def _verify_pilot_runtime_record(
    manifest: Mapping[str, Any],
    config: GraspLiftPilotConfig,
) -> None:
    from rexpolicy.stage0.envs.grasp_lift_action import (
        DEFAULT_GRASP_LIFT_ACTION_SCHEMA,
    )
    from rexpolicy.stage0.envs.grasp_lift_modes import (
        DEFAULT_GRASP_LIFT_AUTHORING_MODE,
    )
    from rexpolicy.stage0.envs.grasp_lift_oracle import GRASP_LIFT_ORACLE_ID
    from rexpolicy.stage0.envs.grasp_lift_state_view import (
        DEFAULT_GRASP_LIFT_STATE_VIEW,
    )

    bindings = (
        (
            "action schema",
            manifest["action_schema"],
            DEFAULT_GRASP_LIFT_ACTION_SCHEMA.to_record(),
            manifest["action_schema_sha256"],
            DEFAULT_GRASP_LIFT_ACTION_SCHEMA.sha256,
        ),
        (
            "authoring mode",
            manifest["authoring_mode"],
            DEFAULT_GRASP_LIFT_AUTHORING_MODE.to_record(),
            manifest["authoring_mode_sha256"],
            DEFAULT_GRASP_LIFT_AUTHORING_MODE.sha256,
        ),
        (
            "state view",
            manifest["state_view"],
            DEFAULT_GRASP_LIFT_STATE_VIEW.to_record(),
            manifest["state_view_sha256"],
            DEFAULT_GRASP_LIFT_STATE_VIEW.sha256,
        ),
    )
    for name, record, expected_record, digest, expected_digest in bindings:
        if record != expected_record or digest != expected_digest:
            raise ValueError(f"pilot {name} binding changed")

    implementation = _implementation_fingerprint_from_record(
        manifest["implementation"]
    )
    if implementation.sha256 != manifest["implementation_sha256"]:
        raise ValueError("pilot implementation hash changed")

    simulator = _exact_mapping(
        manifest["simulator_record"],
        {
            "action_schema_sha256",
            "authoring_mode_sha256",
            "environment_config",
            "implementation_sha256",
            "oracle",
            "oracle_id",
            "schema_id",
            "task_metadata",
        },
        "pilot simulator record",
    )
    if canonical_fingerprint(simulator) != manifest["simulator_sha256"]:
        raise ValueError("pilot simulator hash changed")
    if simulator["schema_id"] != "rexpolicy/stage0-grasp-lift-newton-runtime/v1":
        raise ValueError("pilot simulator schema changed")
    if simulator["action_schema_sha256"] != manifest["action_schema_sha256"]:
        raise ValueError("pilot simulator action schema changed")
    if simulator["authoring_mode_sha256"] != manifest["authoring_mode_sha256"]:
        raise ValueError("pilot simulator authoring mode changed")
    if simulator["implementation_sha256"] != manifest["implementation_sha256"]:
        raise ValueError("pilot simulator implementation changed")
    if simulator["oracle_id"] != GRASP_LIFT_ORACLE_ID:
        raise ValueError("pilot simulator oracle changed")
    expected_environment = grasp_lift_pilot_environment_config_record(
        config.runtime
    )
    if simulator["environment_config"] != expected_environment:
        raise ValueError("pilot environment config changed")
    if simulator["task_metadata"] != grasp_lift_pilot_task_metadata_record():
        raise ValueError("pilot task metadata changed")
    oracle = grasp_lift_pilot_oracle_record()
    if simulator["oracle"] != oracle or simulator["oracle_id"] != oracle["oracle_id"]:
        raise ValueError("pilot oracle contract changed")


def load_grasp_lift_pilot_manifest(directory: str | Path) -> dict[str, Any]:
    """Verify every committed pilot record, trajectory, and evidence sidecar."""
    requested = Path(directory).expanduser()
    if requested.is_symlink() or not requested.is_dir():
        raise ValueError("pilot output must be a directory and cannot be a symlink")
    root = requested.resolve()
    commit = _read_pilot_json(root / "commit.json", "pilot commit")
    verify_grasp_lift_pilot_record(commit, "pilot commit")
    _exact_mapping(
        commit,
        {
            "config_file",
            "config_sha256",
            "manifest_file",
            "manifest_sha256",
            "schema_id",
            "self_sha256",
            "status_file",
            "status_sha256",
        },
        "pilot commit",
    )
    if commit["schema_id"] != GRASP_LIFT_PILOT_COMMIT_SCHEMA_ID:
        raise ValueError("pilot commit schema changed")
    if (
        commit["config_file"] != "config.json"
        or commit["manifest_file"] != "manifest.json"
        or commit["status_file"] != "status.json"
    ):
        raise ValueError("pilot commit file names changed")
    for name in ("config_sha256", "manifest_sha256", "status_sha256"):
        _require_sha256(commit[name], f"pilot commit {name}")

    config = load_grasp_lift_pilot_config(root / "config.json")
    verify_grasp_lift_pilot_formal_config(config)
    if config.sha256 != commit["config_sha256"]:
        raise ValueError("pilot commit does not bind config")
    manifest = _read_pilot_json(root / "manifest.json", "pilot manifest")
    status = _read_pilot_json(root / "status.json", "pilot status")
    verify_grasp_lift_pilot_record(manifest, "pilot manifest")
    verify_grasp_lift_pilot_record(status, "pilot status")
    _exact_mapping(manifest, _PILOT_MANIFEST_KEYS, "pilot manifest")
    _exact_mapping(
        status,
        {
            "acceptance_passed",
            "complete",
            "manifest_sha256",
            "outcome_counts",
            "schema_id",
            "self_sha256",
        },
        "pilot status",
    )
    if manifest["schema_id"] != GRASP_LIFT_PILOT_OUTPUT_SCHEMA_ID:
        raise ValueError("pilot manifest schema changed")
    if status["schema_id"] != GRASP_LIFT_PILOT_STATUS_SCHEMA_ID:
        raise ValueError("pilot status schema changed")
    if commit["manifest_sha256"] != manifest["self_sha256"]:
        raise ValueError("pilot commit does not bind the manifest")
    if commit["status_sha256"] != status["self_sha256"]:
        raise ValueError("pilot commit does not bind status")
    if status["manifest_sha256"] != manifest["self_sha256"]:
        raise ValueError("pilot status does not bind the manifest")
    if status["complete"] is not True:
        raise ValueError("pilot status is not complete")
    if manifest["experiment_sha256"] != config.sha256:
        raise ValueError("pilot manifest does not bind config")

    for name in (
        "action_schema_sha256",
        "authoring_mode_sha256",
        "composite_corpus_sha256",
        "experiment_sha256",
        "implementation_sha256",
        "authoring_lineage_sha256",
        "simulator_sha256",
        "split_policy_sha256",
        "state_view_sha256",
        "validation_claim_sha256",
    ):
        _require_sha256(manifest[name], f"pilot manifest {name}")
    _verify_pilot_runtime_record(manifest, config)

    expected_lineage = grasp_lift_pilot_authoring_lineage_record()
    if manifest["authoring_lineage"] != expected_lineage:
        raise ValueError("pilot authoring lineage changed")
    if manifest["authoring_lineage_sha256"] != canonical_fingerprint(
        expected_lineage
    ):
        raise ValueError("pilot authoring lineage hash changed")

    validation_claim = manifest["validation_claim"]
    verify_grasp_lift_pilot_record(validation_claim, "pilot validation claim")
    _exact_mapping(
        validation_claim,
        {
            "experiment_sha256",
            "identity",
            "identity_sha256",
            "implementation_sha256",
            "schema_id",
            "self_sha256",
        },
        "pilot validation claim",
    )
    expected_validation_claim = seal_grasp_lift_pilot_record(
        grasp_lift_pilot_validation_claim_record(
            experiment_sha256=config.sha256,
            implementation_sha256=manifest["implementation_sha256"],
        )
    )
    if validation_claim != expected_validation_claim:
        raise ValueError("pilot validation claim changed")
    if manifest["validation_claim_sha256"] != validation_claim["self_sha256"]:
        raise ValueError("pilot validation claim hash changed")

    expected_split_policy = grasp_lift_pilot_split_policy()
    if manifest["split_policy"] != expected_split_policy:
        raise ValueError("pilot split policy changed")
    if manifest["split_policy_sha256"] != canonical_fingerprint(
        expected_split_policy
    ):
        raise ValueError("pilot split policy hash changed")
    if manifest["locked_test"] != {
        "artifact": None,
        "consumed": False,
        "policy": "not_created",
    }:
        raise ValueError("pilot manifest illegally created or consumed locked test")
    if manifest["visuals"] != {
        "camera_textures": False,
        "scene_asset": None,
        "scene_visuals": False,
    }:
        raise ValueError("pilot authoring unexpectedly used visual observations")
    affinity = manifest["cpu_affinity"]
    if (
        not isinstance(affinity, list)
        or len(affinity) != config.runtime.cpu_threads
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in affinity
        )
        or affinity != sorted(set(affinity))
    ):
        raise ValueError("pilot CPU affinity does not match the runtime contract")
    elapsed = manifest["elapsed_seconds"]
    if (
        isinstance(elapsed, bool)
        or not isinstance(elapsed, (int, float))
        or not math.isfinite(float(elapsed))
        or float(elapsed) < 0.0
    ):
        raise ValueError("pilot elapsed_seconds must be finite and non-negative")

    from rexpolicy.stage0.data.grasp_lift_evidence import (
        grasp_lift_composite_corpus_sha256,
        grasp_lift_evidence_sha256,
        load_grasp_lift_evidence_sidecar,
    )
    from rexpolicy.stage0.data.store import load_trajectory_shard
    from rexpolicy.stage0.data.trajectory import trajectory_sha256
    from rexpolicy.stage0.envs.grasp_lift_action import (
        DEFAULT_GRASP_LIFT_ACTION_SCHEMA,
    )
    from rexpolicy.stage0.envs.grasp_lift_modes import (
        DEFAULT_GRASP_LIFT_AUTHORING_MODE,
        GraspLiftAuthoringPhase,
    )
    from rexpolicy.stage0.envs.grasp_lift_oracle import GRASP_LIFT_ORACLE_ID
    from rexpolicy.stage0.envs.grasp_lift_state_view import (
        DEFAULT_GRASP_LIFT_STATE_VIEW,
    )
    from rexpolicy.stage0.envs.state_schema import DEFAULT_STAGE0_STATE_SCHEMA
    from rexpolicy.stage0.types import Stage0Outcome

    members = manifest["members"]
    if not isinstance(members, list) or len(members) != config.runtime.reset_groups:
        raise ValueError("pilot manifest member count changed")
    corpus: list[tuple[Any, Any]] = []
    trajectory_groups: dict[str, str] = {}
    actual_splits: dict[str, str] = {}
    observed_seeds: set[int] = set()
    outcomes = {name: 0 for name in _PILOT_OUTCOMES}
    split_counts = {
        split: {name: 0 for name in _PILOT_OUTCOMES}
        for split in ("train", "validation")
    }
    safety_violation_count = 0
    world_steps = 0
    listed_ids: list[str] = []
    expected_shard_files: set[str] = set()
    violation_keys = (
        "forbidden_contact_violation",
        "lateral_displacement_violation",
        "bottle_tilt_violation",
        "dropped_grasp_violation",
        "collision_buffer_overflow_violation",
    )
    for raw_member in members:
        member = _exact_mapping(raw_member, _PILOT_MEMBER_KEYS, "pilot member")
        trajectory_id = member["trajectory_id"]
        group_id = member["reset_group_id"]
        if (
            not isinstance(trajectory_id, str)
            or _SAFE_PILOT_ID.fullmatch(trajectory_id) is None
            or trajectory_id in trajectory_groups
        ):
            raise ValueError("pilot manifest has an invalid trajectory ID")
        if (
            not isinstance(group_id, str)
            or _SAFE_PILOT_ID.fullmatch(group_id) is None
        ):
            raise ValueError("pilot manifest has an invalid reset group ID")
        split = member["split"]
        outcome = member["outcome"]
        if split not in {"train", "validation"} or outcome not in _PILOT_OUTCOMES:
            raise ValueError("pilot member split or outcome changed")
        reset_seed = member["reset_seed"]
        final_phase = member["final_authoring_phase"]
        terminal_from_phase = member["terminal_transition_from_phase"]
        formal_gate_eligible = member["formal_gate_eligible"]
        recorded_safety_violation = member["safety_violation"]
        if (
            isinstance(reset_seed, bool)
            or not isinstance(reset_seed, int)
            or isinstance(final_phase, bool)
            or not isinstance(final_phase, int)
            or isinstance(terminal_from_phase, bool)
            or not isinstance(terminal_from_phase, int)
            or not isinstance(formal_gate_eligible, bool)
            or not isinstance(recorded_safety_violation, bool)
        ):
            raise ValueError("pilot member seed or phase changed")
        for name in ("trajectory_sha256", "evidence_sha256"):
            _require_sha256(member[name], f"pilot member {name}")
        group_index = reset_seed - config.runtime.seed
        if group_index < 0 or group_index >= config.runtime.reset_groups:
            raise ValueError("pilot member reset seed is outside the fixed range")
        expected_eligibility = grasp_lift_pilot_formal_gate_eligible(reset_seed)
        if formal_gate_eligible is not expected_eligibility:
            raise ValueError("pilot member formal gate eligibility changed")
        expected_group = f"grasp-reset-g{group_index:06d}-s{reset_seed:010d}"
        expected_trajectory = (
            f"grasp-g{group_index:06d}-thumb-index-side-close7-"
            f"h{DEFAULT_GRASP_LIFT_AUTHORING_MODE.sha256[:12]}-"
            f"s{reset_seed:010d}"
        )
        if group_id != expected_group or trajectory_id != expected_trajectory:
            raise ValueError("pilot member identity changed")
        if reset_seed in observed_seeds:
            raise ValueError("pilot reset seed is duplicated")
        observed_seeds.add(reset_seed)
        expected_shard_files.update(
            {
                f"{trajectory_id}.stage0.json",
                f"{trajectory_id}.stage0.pt",
                f"{trajectory_id}.grasp-lift-evidence.json",
                f"{trajectory_id}.grasp-lift-evidence.pt",
            }
        )

        trajectory_descriptor = _pilot_member_path(
            root,
            member["trajectory_descriptor"],
            expected_name=f"{trajectory_id}.stage0.json",
            context="pilot trajectory descriptor",
        )
        _pilot_member_path(
            root,
            f"shards/{trajectory_id}.stage0.pt",
            expected_name=f"{trajectory_id}.stage0.pt",
            context="pilot trajectory payload",
        )
        evidence_descriptor = _pilot_member_path(
            root,
            member["evidence_descriptor"],
            expected_name=f"{trajectory_id}.grasp-lift-evidence.json",
            context="pilot evidence descriptor",
        )
        _pilot_member_path(
            root,
            f"shards/{trajectory_id}.grasp-lift-evidence.pt",
            expected_name=f"{trajectory_id}.grasp-lift-evidence.pt",
            context="pilot evidence payload",
        )
        trajectory = load_trajectory_shard(trajectory_descriptor)
        if trajectory_sha256(trajectory) != member["trajectory_sha256"]:
            raise ValueError("pilot member trajectory hash changed")
        provenance = trajectory.provenance
        expected_provenance = {
            "action_schema_sha256": DEFAULT_GRASP_LIFT_ACTION_SCHEMA.sha256,
            "experiment_sha256": config.sha256,
            "generation": 0,
            "oracle_id": GRASP_LIFT_ORACLE_ID,
            "rank": 0,
            "reset_group_id": group_id,
            "reset_seed": reset_seed,
            "simulator_sha256": manifest["simulator_sha256"],
            "state_schema_sha256": DEFAULT_STAGE0_STATE_SCHEMA.sha256,
            "task_id": "grasp_lift_green_bottle/v1",
            "trajectory_id": trajectory_id,
            "world": group_index % config.runtime.num_envs,
        }
        if provenance.to_record() != expected_provenance:
            raise ValueError("pilot trajectory provenance changed")
        if trajectory.outcome.value != outcome:
            raise ValueError("pilot member outcome changed")

        evidence = load_grasp_lift_evidence_sidecar(
            evidence_descriptor,
            trajectory=trajectory,
        )
        if grasp_lift_evidence_sha256(evidence) != member["evidence_sha256"]:
            raise ValueError("pilot member evidence hash changed")
        metadata = evidence.metadata
        if (
            metadata.mode_id != DEFAULT_GRASP_LIFT_AUTHORING_MODE.mode_id
            or metadata.mode_sha256 != DEFAULT_GRASP_LIFT_AUTHORING_MODE.sha256
            or metadata.action_schema_sha256
            != DEFAULT_GRASP_LIFT_ACTION_SCHEMA.sha256
            or metadata.state_view_sha256 != DEFAULT_GRASP_LIFT_STATE_VIEW.sha256
            or metadata.implementation_sha256 != manifest["implementation_sha256"]
            or metadata.experiment_sha256 != config.sha256
        ):
            raise ValueError("pilot evidence provenance changed")
        recorded_phase = int(evidence.tensors["authoring_phase_after"][-1])
        recorded_terminal_from_phase = int(
            evidence.tensors["authoring_phase_before"][-1]
        )
        if final_phase != recorded_phase:
            raise ValueError("pilot member final authoring phase changed")
        if terminal_from_phase != recorded_terminal_from_phase:
            raise ValueError("pilot member terminal transition changed")
        allowed_phases = {int(phase) for phase in GraspLiftAuthoringPhase}
        for name in ("authoring_phase_before", "authoring_phase_after"):
            if not set(evidence.tensors[name].tolist()).issubset(allowed_phases):
                raise ValueError("pilot evidence contains an invalid authoring phase")
        if bool(evidence.tensors["collision_buffer_overflow"].any()):
            raise ValueError("pilot evidence contains a collision buffer overflow")
        oracle_success = bool(evidence.tensors["oracle_success"][-1])
        oracle_failure = bool(evidence.tensors["oracle_failure"][-1])
        if outcome == Stage0Outcome.SUCCESS.value:
            valid_terminal = (
                oracle_success
                and not oracle_failure
                and final_phase == int(GraspLiftAuthoringPhase.SUCCEEDED)
            )
        elif outcome == Stage0Outcome.FAILURE.value:
            valid_terminal = (
                oracle_failure
                and not oracle_success
                and final_phase == int(GraspLiftAuthoringPhase.ORACLE_FAILED)
            )
        else:
            valid_terminal = not oracle_success and not oracle_failure
        if not valid_terminal:
            raise ValueError("pilot member terminal evidence changed")

        failure_reasons = (
            ("forbidden_contact", "forbidden_contact_violation"),
            ("lateral_displacement", "lateral_displacement_violation"),
            ("bottle_tilt", "bottle_tilt_violation"),
            ("dropped_grasp", "dropped_grasp_violation"),
            ("collision_buffer_overflow", "collision_buffer_overflow_violation"),
        )
        active_reasons = [
            label
            for label, name in failure_reasons
            if bool(evidence.tensors[name][-1])
        ]
        expected_failure_reason = (
            "+".join(active_reasons) if active_reasons else "oracle_failure"
        )
        if outcome == Stage0Outcome.FAILURE.value:
            if trajectory.failure_reason != expected_failure_reason:
                raise ValueError("pilot trajectory failure reason changed")
        elif trajectory.failure_reason is not None:
            raise ValueError("pilot non-failure trajectory has a failure reason")

        member_safety_violation = any(
            bool(evidence.tensors[name][-1]) for name in violation_keys
        )
        if recorded_safety_violation is not member_safety_violation:
            raise ValueError("pilot member safety violation changed")
        safety_violation_count += int(member_safety_violation)
        world_steps += trajectory.transition_count
        outcomes[outcome] += 1
        split_counts[split][outcome] += 1
        trajectory_groups[trajectory_id] = group_id
        actual_splits[trajectory_id] = split
        listed_ids.append(trajectory_id)
        corpus.append((trajectory, evidence))

    if listed_ids != sorted(listed_ids):
        raise ValueError("pilot manifest members are not canonically ordered")
    expected_root_files = {
        "commit.json",
        "config.json",
        "manifest.json",
        "shards",
        "status.json",
    }
    if {entry.name for entry in root.iterdir()} != expected_root_files:
        raise ValueError("pilot output contains uncommitted root artifacts")
    shards = root / "shards"
    if {entry.name for entry in shards.iterdir()} != expected_shard_files:
        raise ValueError("pilot output contains uncommitted shard artifacts")
    expected_seeds = set(
        range(
            config.runtime.seed,
            config.runtime.seed + config.runtime.reset_groups,
        )
    )
    if observed_seeds != expected_seeds:
        raise ValueError("pilot reset seed coverage changed")
    expected_splits = assign_grasp_lift_pilot_splits(
        trajectory_groups,
        validation_fraction=config.acceptance.validation_fraction,
    )
    if actual_splits != expected_splits:
        raise ValueError("pilot manifest split assignments violate policy")
    if grasp_lift_composite_corpus_sha256(corpus) != manifest[
        "composite_corpus_sha256"
    ]:
        raise ValueError("pilot composite corpus hash changed")
    if _pilot_count_record(manifest["outcome_counts"], "pilot outcomes") != outcomes:
        raise ValueError("pilot outcome counts changed")
    manifest_split_counts = _exact_mapping(
        manifest["split_counts"],
        {"train", "validation"},
        "pilot split counts",
    )
    for split in ("train", "validation"):
        if _pilot_count_record(
            manifest_split_counts[split], f"pilot split counts.{split}"
        ) != split_counts[split]:
            raise ValueError("pilot split counts changed")
    if (
        isinstance(manifest["world_steps"], bool)
        or not isinstance(manifest["world_steps"], int)
        or manifest["world_steps"] != world_steps
    ):
        raise ValueError("pilot world_steps changed")

    eligible_member_count = sum(
        member["formal_gate_eligible"] for member in members
    )
    eligible_success_count = sum(
        member["formal_gate_eligible"]
        and member["outcome"] == Stage0Outcome.SUCCESS.value
        for member in members
    )
    eligible_failure_count = sum(
        member["formal_gate_eligible"]
        and member["outcome"] == Stage0Outcome.FAILURE.value
        for member in members
    )
    eligible_safety_violation_count = sum(
        member["formal_gate_eligible"] and member["safety_violation"]
        for member in members
    )
    excluded_member_count = len(members) - eligible_member_count
    success_rate = eligible_success_count / float(eligible_member_count)
    acceptance = _exact_mapping(
        manifest["acceptance"],
        {
            "authoring_gate",
            "eligible_member_count",
            "eligible_failure_count",
            "eligible_safety_violation_count",
            "eligible_success_count",
            "engineering_smoke_excluded_member_count",
            "passed",
            "policy",
            "safety_violation_count",
            "success_rate",
        },
        "pilot acceptance",
    )
    base_acceptance_passed = bool(
        success_rate >= config.acceptance.minimum_success_rate
        and (
            not config.acceptance.require_zero_oracle_failures
            or eligible_failure_count == 0
        )
        and (
            not config.acceptance.require_zero_safety_violations
            or eligible_safety_violation_count == 0
        )
    )
    authoring_gate = evaluate_grasp_lift_close7_authoring_gate(members)
    expected_passed = bool(base_acceptance_passed and authoring_gate["passed"])
    for name in (
        "eligible_member_count",
        "eligible_failure_count",
        "eligible_safety_violation_count",
        "eligible_success_count",
        "engineering_smoke_excluded_member_count",
    ):
        value = acceptance[name]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"pilot acceptance {name} changed")
    if (
        not isinstance(acceptance["passed"], bool)
        or isinstance(acceptance["safety_violation_count"], bool)
        or not isinstance(acceptance["safety_violation_count"], int)
        or isinstance(acceptance["success_rate"], bool)
        or not isinstance(acceptance["success_rate"], (int, float))
        or acceptance["eligible_member_count"] != eligible_member_count
        or acceptance["eligible_failure_count"] != eligible_failure_count
        or acceptance["eligible_safety_violation_count"]
        != eligible_safety_violation_count
        or acceptance["eligible_success_count"] != eligible_success_count
        or acceptance["engineering_smoke_excluded_member_count"]
        != excluded_member_count
        or acceptance["authoring_gate"] != authoring_gate
        or acceptance["policy"] != config.acceptance.to_record()
        or acceptance["safety_violation_count"] != safety_violation_count
        or acceptance["success_rate"] != success_rate
        or acceptance["passed"] is not expected_passed
    ):
        raise ValueError("pilot acceptance calculation changed")
    if (
        not isinstance(status["acceptance_passed"], bool)
        or status["acceptance_passed"] is not expected_passed
        or _pilot_count_record(status["outcome_counts"], "pilot status outcomes")
        != outcomes
    ):
        raise ValueError("pilot status disagrees with verified artifacts")
    return manifest


__all__ = [
    "GRASP_LIFT_PILOT_CONFIG_SCHEMA_ID",
    "GRASP_LIFT_PILOT_CONFIG_SCHEMA_VERSION",
    "GRASP_LIFT_PILOT_FORMAL_CONFIG_SHA256",
    "GRASP_LIFT_PILOT_COMMIT_SCHEMA_ID",
    "GRASP_LIFT_PILOT_OUTPUT_SCHEMA_ID",
    "GRASP_LIFT_PILOT_SPLIT_NAMESPACE",
    "GRASP_LIFT_PILOT_SPLIT_POLICY_ID",
    "GRASP_LIFT_PILOT_STATUS_SCHEMA_ID",
    "GRASP_LIFT_PILOT_VALIDATION_FRACTION",
    "GRASP_LIFT_PILOT_VALIDATION_CLAIM_SCHEMA_ID",
    "GraspLiftPilotAcceptanceConfig",
    "GraspLiftPilotConfig",
    "GraspLiftPilotRuntimeConfig",
    "assign_grasp_lift_pilot_splits",
    "evaluate_grasp_lift_close7_authoring_gate",
    "grasp_lift_pilot_authoring_lineage_record",
    "grasp_lift_pilot_environment_config_record",
    "grasp_lift_pilot_formal_gate_eligible",
    "grasp_lift_pilot_oracle_record",
    "grasp_lift_pilot_split_policy",
    "grasp_lift_pilot_task_metadata_record",
    "grasp_lift_pilot_validation_claim_identity",
    "grasp_lift_pilot_validation_claim_record",
    "load_grasp_lift_pilot_config",
    "load_grasp_lift_pilot_manifest",
    "seal_grasp_lift_pilot_record",
    "verify_grasp_lift_pilot_record",
    "verify_grasp_lift_pilot_formal_config",
]
