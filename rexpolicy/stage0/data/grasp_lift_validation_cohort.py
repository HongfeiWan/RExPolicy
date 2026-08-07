"""Strict write-once artifact contract for a new Grasp-Lift validation cohort.

This module describes authored oracle demonstrations only.  It deliberately has
no checkpoint, learned-policy, optimizer, or model-loading surface.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from rexpolicy.stage0.grasp_lift_pilot import (
    GraspLiftPilotRuntimeConfig,
    grasp_lift_pilot_environment_config_record,
    grasp_lift_pilot_oracle_record,
    grasp_lift_pilot_task_metadata_record,
    verify_grasp_lift_pilot_record,
)
from rexpolicy.stage0.types import canonical_fingerprint


GRASP_LIFT_VALIDATION_AUTHORING_CONFIG_SCHEMA_ID = (
    "rexpolicy/stage0-grasp-lift-validation-authoring-config/v1"
)
GRASP_LIFT_VALIDATION_AUTHORING_MANIFEST_SCHEMA_ID = (
    "rexpolicy/stage0-grasp-lift-validation-authoring-manifest/v1"
)
GRASP_LIFT_VALIDATION_AUTHORING_STATUS_SCHEMA_ID = (
    "rexpolicy/stage0-grasp-lift-validation-authoring-status/v1"
)
GRASP_LIFT_VALIDATION_AUTHORING_COMMIT_SCHEMA_ID = (
    "rexpolicy/stage0-grasp-lift-validation-authoring-commit/v1"
)
GRASP_LIFT_VALIDATION_AUTHORING_PREREGISTRATION_SCHEMA_ID = (
    "rexpolicy/stage0-grasp-lift-validation-authoring-preregistration/v1"
)
GRASP_LIFT_VALIDATION_COHORT_SLOT_SCHEMA_ID = (
    "rexpolicy/stage0-grasp-lift-validation-cohort-slot/v1"
)
GRASP_LIFT_VALIDATION_PURPOSE = "grasp_lift_no_z_model_selection/v1"
GRASP_LIFT_VALIDATION_SEED_NAMESPACE = (
    "rexpolicy/grasp-lift-formal-validation/20260807-v2"
)
GRASP_LIFT_VALIDATION_SEED_DERIVATION = (
    "int.from_bytes(sha256(f'{namespace}/{index}').digest()[:4], 'big') "
    "& 0x7fffffff; canonical_numeric_sort"
)
GRASP_LIFT_VALIDATION_RESET_SEEDS = (
    374383479,
    630355668,
    1195800966,
    1354723473,
    1579845348,
    2014581365,
)
GRASP_LIFT_VALIDATION_MEMBER_COUNT = 6
GRASP_LIFT_VALIDATION_LOCKED_TEST = {
    "artifact": None,
    "consumed": False,
    "policy": "not_created",
}

_CONFIG_KEYS = {"acceptance", "cohort", "runtime", "schema_version"}
_COHORT_KEYS = {"purpose", "seed_derivation", "seed_namespace", "seeds"}
_RUNTIME_KEYS = {
    "cpu_threads",
    "device",
    "max_episode_steps",
    "num_envs",
    "rigid_contacts_per_env",
    "triangle_pairs_per_env",
}
_ACCEPTANCE_KEYS = {
    "require_all_success",
    "require_zero_collision_buffer_overflows",
    "require_zero_nonzero_rewards",
    "require_zero_oracle_failures",
    "require_zero_safety_violations",
    "require_zero_timeouts",
}
_MANIFEST_KEYS = {
    "acceptance",
    "action_schema",
    "action_schema_sha256",
    "authoring_mode",
    "authoring_mode_sha256",
    "cohort_slot_id",
    "composite_corpus_sha256",
    "config_sha256",
    "cpu_affinity",
    "elapsed_seconds",
    "environment_config",
    "environment_config_sha256",
    "implementation",
    "implementation_sha256",
    "locked_test",
    "members",
    "oracle",
    "oracle_sha256",
    "preregistration",
    "preregistration_sha256",
    "purpose",
    "reset_seeds",
    "schema_id",
    "seed_derivation",
    "seed_namespace",
    "self_sha256",
    "simulator_record",
    "simulator_sha256",
    "state_view",
    "state_view_sha256",
    "task_metadata",
    "task_metadata_sha256",
    "visuals",
    "world_steps",
}
_MEMBER_KEYS = {
    "evidence_descriptor",
    "evidence_sha256",
    "final_authoring_phase",
    "outcome",
    "reset_group_id",
    "reset_seed",
    "role",
    "safety_violation",
    "split",
    "terminal_transition_from_phase",
    "trajectory_descriptor",
    "trajectory_id",
    "trajectory_sha256",
}
_STATUS_KEYS = {
    "accepted",
    "cohort_slot_id",
    "collision_buffer_overflow_count",
    "complete",
    "manifest_sha256",
    "nonzero_reward_transition_count",
    "outcome_counts",
    "safety_violation_count",
    "schema_id",
    "self_sha256",
}
_COMMIT_KEYS = {
    "config_file",
    "config_file_sha256",
    "config_sha256",
    "manifest_file",
    "manifest_file_sha256",
    "manifest_sha256",
    "schema_id",
    "self_sha256",
    "status_file",
    "status_file_sha256",
    "status_sha256",
}


def _exact_mapping(value: Any, keys: set[str], context: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != keys:
        raise ValueError(f"{context} fields changed")
    return dict(value)


def _positive_integer(value: Any, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{context} must be a positive integer")
    return value


def _require_sha256(value: Any, context: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{context} must be a lowercase SHA-256 digest")
    return value


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _read_json(path: Path, context: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{context} is missing or a symlink")
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=lambda constant: (_ for _ in ()).throw(
                ValueError(f"invalid JSON constant: {constant}")
            ),
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"cannot read {context}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{context} must contain an object")
    return value


def validation_authoring_file_sha256(path: str | Path) -> str:
    """Hash one regular non-symlink artifact file without model dependencies."""
    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise ValueError("validation authoring hash source is missing or a symlink")
    digest = hashlib.sha256()
    with source.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def derive_grasp_lift_validation_seeds(
    namespace: str = GRASP_LIFT_VALIDATION_SEED_NAMESPACE,
) -> tuple[int, ...]:
    """Derive the committed six reset seeds without observing outcomes."""
    if namespace != GRASP_LIFT_VALIDATION_SEED_NAMESPACE:
        raise ValueError("validation seed namespace changed")
    derived = (
        int.from_bytes(
            hashlib.sha256(f"{namespace}/{index}".encode("utf-8")).digest()[:4],
            "big",
        )
        & 0x7FFFFFFF
        for index in range(GRASP_LIFT_VALIDATION_MEMBER_COUNT)
    )
    result = tuple(sorted(derived))
    if len(set(result)) != GRASP_LIFT_VALIDATION_MEMBER_COUNT:
        raise RuntimeError("derived validation seeds collided")
    return result


@dataclass(frozen=True)
class GraspLiftValidationAuthoringRuntime:
    cpu_threads: int
    device: str
    max_episode_steps: int
    num_envs: int
    rigid_contacts_per_env: int
    triangle_pairs_per_env: int

    def __post_init__(self) -> None:
        if self.device != "cuda:0":
            raise ValueError("validation authoring device is fixed at cuda:0")
        expected = {
            "cpu_threads": 16,
            "max_episode_steps": 72,
            "num_envs": GRASP_LIFT_VALIDATION_MEMBER_COUNT,
            "rigid_contacts_per_env": 2048,
            "triangle_pairs_per_env": 200000,
        }
        for name, value in expected.items():
            _positive_integer(getattr(self, name), f"runtime.{name}")
            if getattr(self, name) != value:
                raise ValueError(f"validation authoring runtime.{name} changed")

    @classmethod
    def from_mapping(cls, value: Any) -> GraspLiftValidationAuthoringRuntime:
        return cls(**_exact_mapping(value, _RUNTIME_KEYS, "authoring runtime"))

    def to_record(self) -> dict[str, Any]:
        return {
            "cpu_threads": self.cpu_threads,
            "device": self.device,
            "max_episode_steps": self.max_episode_steps,
            "num_envs": self.num_envs,
            "rigid_contacts_per_env": self.rigid_contacts_per_env,
            "triangle_pairs_per_env": self.triangle_pairs_per_env,
        }


@dataclass(frozen=True)
class GraspLiftValidationAuthoringConfig:
    runtime: GraspLiftValidationAuthoringRuntime
    acceptance: Mapping[str, bool]
    purpose: str
    seed_namespace: str
    seed_derivation: str
    seeds: tuple[int, ...]
    schema_version: int = 1

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("validation authoring schema_version must be 1")
        if not isinstance(self.runtime, GraspLiftValidationAuthoringRuntime):
            raise TypeError("runtime has the wrong type")
        acceptance = _exact_mapping(
            self.acceptance, _ACCEPTANCE_KEYS, "authoring acceptance"
        )
        if any(value is not True for value in acceptance.values()):
            raise ValueError("every validation authoring acceptance gate is mandatory")
        object.__setattr__(self, "acceptance", acceptance)
        if self.purpose != GRASP_LIFT_VALIDATION_PURPOSE:
            raise ValueError("validation purpose changed")
        if self.seed_namespace != GRASP_LIFT_VALIDATION_SEED_NAMESPACE:
            raise ValueError("validation seed namespace changed")
        if self.seed_derivation != GRASP_LIFT_VALIDATION_SEED_DERIVATION:
            raise ValueError("validation seed derivation changed")
        if self.seeds != derive_grasp_lift_validation_seeds():
            raise ValueError("validation seeds are not the preregistered derivation")

    @classmethod
    def from_mapping(cls, value: Any) -> GraspLiftValidationAuthoringConfig:
        record = _exact_mapping(value, _CONFIG_KEYS, "authoring config")
        cohort = _exact_mapping(record["cohort"], _COHORT_KEYS, "cohort config")
        seeds = cohort["seeds"]
        if not isinstance(seeds, list):
            raise ValueError("cohort seeds must be a JSON list")
        return cls(
            runtime=GraspLiftValidationAuthoringRuntime.from_mapping(record["runtime"]),
            acceptance=_exact_mapping(
                record["acceptance"], _ACCEPTANCE_KEYS, "authoring acceptance"
            ),
            purpose=cohort["purpose"],
            seed_namespace=cohort["seed_namespace"],
            seed_derivation=cohort["seed_derivation"],
            seeds=tuple(seeds),
            schema_version=record["schema_version"],
        )

    def to_record(self) -> dict[str, Any]:
        return {
            "acceptance": dict(self.acceptance),
            "cohort": {
                "purpose": self.purpose,
                "seed_derivation": self.seed_derivation,
                "seed_namespace": self.seed_namespace,
                "seeds": list(self.seeds),
            },
            "runtime": self.runtime.to_record(),
            "schema_version": self.schema_version,
        }

    @property
    def sha256(self) -> str:
        return canonical_fingerprint(
            {
                "schema_id": GRASP_LIFT_VALIDATION_AUTHORING_CONFIG_SCHEMA_ID,
                **self.to_record(),
            }
        )


def load_grasp_lift_validation_authoring_config(
    path: str | Path,
) -> GraspLiftValidationAuthoringConfig:
    return GraspLiftValidationAuthoringConfig.from_mapping(
        _read_json(Path(path), "validation authoring config")
    )


def grasp_lift_validation_environment_config_record(
    config: GraspLiftValidationAuthoringConfig,
) -> dict[str, Any]:
    """Reuse the accepted state-only Newton contract with exactly six worlds."""
    if not isinstance(config, GraspLiftValidationAuthoringConfig):
        raise TypeError("config has the wrong type")
    runtime = config.runtime
    pilot_runtime = GraspLiftPilotRuntimeConfig(
        device=runtime.device,
        cpu_threads=runtime.cpu_threads,
        num_envs=runtime.num_envs,
        reset_groups=runtime.num_envs,
        seed=config.seeds[0],
        max_episode_steps=runtime.max_episode_steps,
        rigid_contacts_per_env=runtime.rigid_contacts_per_env,
        triangle_pairs_per_env=runtime.triangle_pairs_per_env,
    )
    return grasp_lift_pilot_environment_config_record(pilot_runtime)


def grasp_lift_validation_cohort_slot_record(
    config: GraspLiftValidationAuthoringConfig,
) -> dict[str, Any]:
    if not isinstance(config, GraspLiftValidationAuthoringConfig):
        raise TypeError("config has the wrong type")
    return {
        "purpose": config.purpose,
        "reset_seeds": list(config.seeds),
        "schema_id": GRASP_LIFT_VALIDATION_COHORT_SLOT_SCHEMA_ID,
        "seed_namespace": config.seed_namespace,
    }


def grasp_lift_validation_cohort_slot_id(
    config: GraspLiftValidationAuthoringConfig,
) -> str:
    return canonical_fingerprint(grasp_lift_validation_cohort_slot_record(config))


def grasp_lift_validation_preregistration_record(
    config: GraspLiftValidationAuthoringConfig,
) -> dict[str, Any]:
    """Build the tensor-free scientific contract committed before CUDA authoring."""
    from rexpolicy.stage0.envs.grasp_lift_action import (
        DEFAULT_GRASP_LIFT_ACTION_SCHEMA,
    )
    from rexpolicy.stage0.envs.grasp_lift_modes import (
        DEFAULT_GRASP_LIFT_AUTHORING_MODE,
    )
    from rexpolicy.stage0.envs.grasp_lift_state_view import (
        DEFAULT_GRASP_LIFT_STATE_VIEW,
    )

    environment = grasp_lift_validation_environment_config_record(config)
    task = grasp_lift_pilot_task_metadata_record()
    oracle = grasp_lift_pilot_oracle_record()
    return {
        "acceptance_policy": dict(config.acceptance),
        "action_schema_sha256": DEFAULT_GRASP_LIFT_ACTION_SCHEMA.sha256,
        "authoring_mode_sha256": DEFAULT_GRASP_LIFT_AUTHORING_MODE.sha256,
        "cohort_slot_id": grasp_lift_validation_cohort_slot_id(config),
        "environment_config_sha256": canonical_fingerprint(environment),
        "locked_test": dict(GRASP_LIFT_VALIDATION_LOCKED_TEST),
        "member_count": GRASP_LIFT_VALIDATION_MEMBER_COUNT,
        "oracle_sha256": canonical_fingerprint(oracle),
        "purpose": config.purpose,
        "reset_seeds": list(config.seeds),
        "role": "formal_validation",
        "schema_id": GRASP_LIFT_VALIDATION_AUTHORING_PREREGISTRATION_SCHEMA_ID,
        "seed_derivation": config.seed_derivation,
        "seed_namespace": config.seed_namespace,
        "split": "validation",
        "state_view_sha256": DEFAULT_GRASP_LIFT_STATE_VIEW.sha256,
        "task_metadata_sha256": canonical_fingerprint(task),
    }


def _strict_member_path(
    root: Path,
    value: Any,
    *,
    trajectory_id: str,
    evidence: bool,
) -> Path:
    suffix = ".grasp-lift-evidence" if evidence else ".stage0"
    expected = Path("shards") / f"{trajectory_id}{suffix}.json"
    if not isinstance(value, str) or Path(value) != expected:
        raise ValueError("validation member descriptor path changed")
    shards = root / "shards"
    descriptor = root / expected
    payload = descriptor.with_suffix(".pt")
    if shards.is_symlink() or not shards.is_dir():
        raise ValueError("validation shards directory is missing or a symlink")
    for path in (descriptor, payload):
        if path.is_symlink() or not path.is_file():
            raise ValueError("validation shard is missing or a symlink")
    return descriptor


def load_grasp_lift_validation_authoring_metadata(
    directory: str | Path,
) -> dict[str, Any]:
    """Verify committed metadata without listing, stating, or opening shards."""
    requested = Path(directory).expanduser()
    if requested.is_symlink() or not requested.is_dir():
        raise ValueError("validation authoring artifact must be a real directory")
    root = requested.resolve()
    config_path = root / "config.json"
    manifest_path = root / "manifest.json"
    status_path = root / "status.json"
    commit_path = root / "commit.json"
    config = load_grasp_lift_validation_authoring_config(config_path)
    commit = _read_json(commit_path, "validation authoring commit")
    status = _read_json(status_path, "validation authoring status")
    manifest = _read_json(manifest_path, "validation authoring manifest")
    _exact_mapping(commit, _COMMIT_KEYS, "validation authoring commit")
    _exact_mapping(status, _STATUS_KEYS, "validation authoring status")
    _exact_mapping(manifest, _MANIFEST_KEYS, "validation authoring manifest")
    verify_grasp_lift_pilot_record(commit, "validation authoring commit")
    verify_grasp_lift_pilot_record(status, "validation authoring status")
    verify_grasp_lift_pilot_record(manifest, "validation authoring manifest")
    if commit["schema_id"] != GRASP_LIFT_VALIDATION_AUTHORING_COMMIT_SCHEMA_ID:
        raise ValueError("validation authoring commit schema changed")
    if status["schema_id"] != GRASP_LIFT_VALIDATION_AUTHORING_STATUS_SCHEMA_ID:
        raise ValueError("validation authoring status schema changed")
    if manifest["schema_id"] != GRASP_LIFT_VALIDATION_AUTHORING_MANIFEST_SCHEMA_ID:
        raise ValueError("validation authoring manifest schema changed")
    if (
        commit["config_file"] != "config.json"
        or commit["manifest_file"] != "manifest.json"
        or commit["status_file"] != "status.json"
        or commit["config_file_sha256"] != validation_authoring_file_sha256(config_path)
        or commit["manifest_file_sha256"]
        != validation_authoring_file_sha256(manifest_path)
        or commit["status_file_sha256"] != validation_authoring_file_sha256(status_path)
        or commit["config_sha256"] != config.sha256
        or commit["manifest_sha256"] != manifest["self_sha256"]
        or commit["status_sha256"] != status["self_sha256"]
    ):
        raise ValueError("validation authoring commit binding changed")

    from rexpolicy.stage0.envs.grasp_lift_action import (
        DEFAULT_GRASP_LIFT_ACTION_SCHEMA,
    )
    from rexpolicy.stage0.envs.grasp_lift_modes import (
        DEFAULT_GRASP_LIFT_AUTHORING_MODE,
    )
    from rexpolicy.stage0.envs.grasp_lift_state_view import (
        DEFAULT_GRASP_LIFT_STATE_VIEW,
    )

    preregistration = grasp_lift_validation_preregistration_record(config)
    environment = grasp_lift_validation_environment_config_record(config)
    task = grasp_lift_pilot_task_metadata_record()
    oracle = grasp_lift_pilot_oracle_record()
    expected_bindings = {
        "action_schema": DEFAULT_GRASP_LIFT_ACTION_SCHEMA.to_record(),
        "action_schema_sha256": DEFAULT_GRASP_LIFT_ACTION_SCHEMA.sha256,
        "authoring_mode": DEFAULT_GRASP_LIFT_AUTHORING_MODE.to_record(),
        "authoring_mode_sha256": DEFAULT_GRASP_LIFT_AUTHORING_MODE.sha256,
        "cohort_slot_id": grasp_lift_validation_cohort_slot_id(config),
        "config_sha256": config.sha256,
        "environment_config": environment,
        "environment_config_sha256": canonical_fingerprint(environment),
        "locked_test": GRASP_LIFT_VALIDATION_LOCKED_TEST,
        "oracle": oracle,
        "oracle_sha256": canonical_fingerprint(oracle),
        "preregistration": preregistration,
        "preregistration_sha256": canonical_fingerprint(preregistration),
        "purpose": config.purpose,
        "reset_seeds": list(config.seeds),
        "seed_derivation": config.seed_derivation,
        "seed_namespace": config.seed_namespace,
        "state_view": DEFAULT_GRASP_LIFT_STATE_VIEW.to_record(),
        "state_view_sha256": DEFAULT_GRASP_LIFT_STATE_VIEW.sha256,
        "task_metadata": task,
        "task_metadata_sha256": canonical_fingerprint(task),
        "visuals": {
            "camera_textures": False,
            "scene_asset": None,
            "scene_visuals": False,
        },
    }
    if any(manifest[key] != value for key, value in expected_bindings.items()):
        raise ValueError("validation authoring semantic binding changed")
    _require_sha256(manifest["implementation_sha256"], "implementation hash")
    _require_sha256(manifest["simulator_sha256"], "simulator hash")
    _require_sha256(manifest["composite_corpus_sha256"], "composite corpus hash")
    if (
        canonical_fingerprint(manifest["implementation"])
        != manifest["implementation_sha256"]
    ):
        raise ValueError("validation authoring implementation hash changed")
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
        "validation simulator record",
    )
    if (
        canonical_fingerprint(simulator) != manifest["simulator_sha256"]
        or simulator["action_schema_sha256"] != DEFAULT_GRASP_LIFT_ACTION_SCHEMA.sha256
        or simulator["authoring_mode_sha256"]
        != DEFAULT_GRASP_LIFT_AUTHORING_MODE.sha256
        or simulator["environment_config"] != environment
        or simulator["task_metadata"] != task
        or simulator["oracle"] != oracle
        or simulator["oracle_id"] != oracle["oracle_id"]
        or simulator["implementation_sha256"] != manifest["implementation_sha256"]
        or simulator["schema_id"] != "rexpolicy/stage0-grasp-lift-newton-runtime/v1"
    ):
        raise ValueError("validation simulator binding changed")

    members = manifest["members"]
    if not isinstance(members, list) or len(members) != 6:
        raise ValueError("validation authoring member count changed")
    for index, raw_member in enumerate(members):
        member = _exact_mapping(raw_member, _MEMBER_KEYS, "validation member")
        seed = config.seeds[index]
        expected_group = f"grasp-validation-v2-g{index:06d}-s{seed:010d}"
        expected_id = (
            f"grasp-validation-v2-g{index:06d}-thumb-index-side-close7-"
            f"h{DEFAULT_GRASP_LIFT_AUTHORING_MODE.sha256[:12]}-s{seed:010d}"
        )
        if (
            member["reset_seed"] != seed
            or member["reset_group_id"] != expected_group
            or member["trajectory_id"] != expected_id
            or member["trajectory_descriptor"] != f"shards/{expected_id}.stage0.json"
            or member["evidence_descriptor"]
            != f"shards/{expected_id}.grasp-lift-evidence.json"
            or member["role"] != "formal_validation"
            or member["split"] != "validation"
            or member["outcome"] != "success"
            or member["safety_violation"] is not False
        ):
            raise ValueError("validation member metadata changed")
        _require_sha256(member["trajectory_sha256"], "trajectory hash")
        _require_sha256(member["evidence_sha256"], "evidence hash")
        for name in ("final_authoring_phase", "terminal_transition_from_phase"):
            if isinstance(member[name], bool) or not isinstance(member[name], int):
                raise ValueError("validation member phase metadata changed")

    expected_counts = {"failure": 0, "success": 6, "timeout": 0}
    acceptance = _exact_mapping(
        manifest["acceptance"],
        {
            "collision_buffer_overflow_count",
            "nonzero_reward_transition_count",
            "outcome_counts",
            "passed",
            "policy",
            "safety_violation_count",
        },
        "validation authoring acceptance",
    )
    if acceptance != {
        "collision_buffer_overflow_count": 0,
        "nonzero_reward_transition_count": 0,
        "outcome_counts": expected_counts,
        "passed": True,
        "policy": dict(config.acceptance),
        "safety_violation_count": 0,
    }:
        raise ValueError("validation authoring acceptance changed")
    if (
        status["complete"] is not True
        or status["accepted"] is not True
        or status["cohort_slot_id"] != manifest["cohort_slot_id"]
        or status["manifest_sha256"] != manifest["self_sha256"]
        or status["outcome_counts"] != expected_counts
        or status["safety_violation_count"] != 0
        or status["collision_buffer_overflow_count"] != 0
        or status["nonzero_reward_transition_count"] != 0
    ):
        raise ValueError("validation authoring status changed")
    if (
        not isinstance(manifest["cpu_affinity"], list)
        or len(manifest["cpu_affinity"]) != config.runtime.cpu_threads
        or any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in manifest["cpu_affinity"]
        )
        or isinstance(manifest["world_steps"], bool)
        or not isinstance(manifest["world_steps"], int)
        or manifest["world_steps"] < 1
        or not isinstance(manifest["elapsed_seconds"], (int, float))
        or isinstance(manifest["elapsed_seconds"], bool)
        or not math.isfinite(float(manifest["elapsed_seconds"]))
        or manifest["elapsed_seconds"] <= 0
    ):
        raise ValueError("validation authoring runtime metadata changed")
    return {
        "commit": commit,
        "config": config.to_record(),
        "config_sha256": config.sha256,
        "manifest": manifest,
        "status": status,
    }


def load_grasp_lift_validation_authoring_manifest(
    directory: str | Path,
) -> dict[str, Any]:
    """Strictly reload every committed file and all six authored shards."""
    metadata = load_grasp_lift_validation_authoring_metadata(directory)
    requested = Path(directory).expanduser()
    if requested.is_symlink() or not requested.is_dir():
        raise ValueError("validation authoring artifact must be a real directory")
    root = requested.resolve()
    if {entry.name for entry in root.iterdir()} != {
        "commit.json",
        "config.json",
        "manifest.json",
        "shards",
        "status.json",
    }:
        raise ValueError("validation authoring artifact file set changed")
    config = GraspLiftValidationAuthoringConfig.from_mapping(metadata["config"])
    manifest = metadata["manifest"]
    status = metadata["status"]

    from rexpolicy.stage0.data.grasp_lift_evidence import (
        grasp_lift_composite_corpus_sha256,
        grasp_lift_evidence_sha256,
        load_grasp_lift_evidence_sidecar,
        validate_grasp_lift_evidence_trajectory,
    )
    from rexpolicy.stage0.data.store import load_trajectory_shard
    from rexpolicy.stage0.data.trajectory import trajectory_sha256
    from rexpolicy.stage0.envs.grasp_lift_action import (
        DEFAULT_GRASP_LIFT_ACTION_SCHEMA,
    )
    from rexpolicy.stage0.envs.grasp_lift_modes import (
        DEFAULT_GRASP_LIFT_AUTHORING_MODE,
    )
    from rexpolicy.stage0.envs.grasp_lift_state_view import (
        DEFAULT_GRASP_LIFT_STATE_VIEW,
    )

    preregistration = grasp_lift_validation_preregistration_record(config)
    environment = grasp_lift_validation_environment_config_record(config)
    task = grasp_lift_pilot_task_metadata_record()
    oracle = grasp_lift_pilot_oracle_record()
    expected_bindings = {
        "action_schema": DEFAULT_GRASP_LIFT_ACTION_SCHEMA.to_record(),
        "action_schema_sha256": DEFAULT_GRASP_LIFT_ACTION_SCHEMA.sha256,
        "authoring_mode": DEFAULT_GRASP_LIFT_AUTHORING_MODE.to_record(),
        "authoring_mode_sha256": DEFAULT_GRASP_LIFT_AUTHORING_MODE.sha256,
        "cohort_slot_id": grasp_lift_validation_cohort_slot_id(config),
        "config_sha256": config.sha256,
        "environment_config": environment,
        "environment_config_sha256": canonical_fingerprint(environment),
        "locked_test": GRASP_LIFT_VALIDATION_LOCKED_TEST,
        "oracle": oracle,
        "oracle_sha256": canonical_fingerprint(oracle),
        "preregistration": preregistration,
        "preregistration_sha256": canonical_fingerprint(preregistration),
        "purpose": config.purpose,
        "reset_seeds": list(config.seeds),
        "seed_derivation": config.seed_derivation,
        "seed_namespace": config.seed_namespace,
        "state_view": DEFAULT_GRASP_LIFT_STATE_VIEW.to_record(),
        "state_view_sha256": DEFAULT_GRASP_LIFT_STATE_VIEW.sha256,
        "task_metadata": task,
        "task_metadata_sha256": canonical_fingerprint(task),
        "visuals": {
            "camera_textures": False,
            "scene_asset": None,
            "scene_visuals": False,
        },
    }
    if any(manifest[key] != value for key, value in expected_bindings.items()):
        raise ValueError("validation authoring semantic binding changed")
    _require_sha256(manifest["implementation_sha256"], "implementation hash")
    if (
        canonical_fingerprint(manifest["implementation"])
        != manifest["implementation_sha256"]
    ):
        raise ValueError("validation authoring implementation hash changed")
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
        "validation simulator record",
    )
    if (
        canonical_fingerprint(simulator) != manifest["simulator_sha256"]
        or simulator["environment_config"] != environment
        or simulator["task_metadata"] != task
        or simulator["oracle"] != oracle
        or simulator["implementation_sha256"] != manifest["implementation_sha256"]
    ):
        raise ValueError("validation simulator binding changed")

    members = manifest["members"]
    if not isinstance(members, list) or len(members) != 6:
        raise ValueError("validation authoring member count changed")
    corpus = []
    shard_files: set[str] = set()
    observed_seeds: list[int] = []
    for index, raw_member in enumerate(members):
        member = _exact_mapping(raw_member, _MEMBER_KEYS, "validation member")
        seed = member["reset_seed"]
        expected_group = f"grasp-validation-v2-g{index:06d}-s{seed:010d}"
        expected_id = (
            f"grasp-validation-v2-g{index:06d}-thumb-index-side-close7-"
            f"h{DEFAULT_GRASP_LIFT_AUTHORING_MODE.sha256[:12]}-s{seed:010d}"
        )
        if (
            seed != config.seeds[index]
            or member["reset_group_id"] != expected_group
            or member["trajectory_id"] != expected_id
            or member["role"] != "formal_validation"
            or member["split"] != "validation"
            or member["outcome"] != "success"
            or member["safety_violation"] is not False
        ):
            raise ValueError("validation member identity or acceptance changed")
        observed_seeds.append(seed)
        trajectory_path = _strict_member_path(
            root,
            member["trajectory_descriptor"],
            trajectory_id=expected_id,
            evidence=False,
        )
        evidence_path = _strict_member_path(
            root,
            member["evidence_descriptor"],
            trajectory_id=expected_id,
            evidence=True,
        )
        for path in (trajectory_path, evidence_path):
            shard_files.update({path.name, path.with_suffix(".pt").name})
        trajectory = load_trajectory_shard(trajectory_path)
        evidence = load_grasp_lift_evidence_sidecar(
            evidence_path, trajectory=trajectory
        )
        validate_grasp_lift_evidence_trajectory(evidence, trajectory)
        if (
            trajectory_sha256(trajectory) != member["trajectory_sha256"]
            or grasp_lift_evidence_sha256(evidence) != member["evidence_sha256"]
            or trajectory.provenance.reset_seed != seed
            or trajectory.provenance.reset_group_id != expected_group
            or trajectory.provenance.trajectory_id != expected_id
            or trajectory.provenance.simulator_sha256 != manifest["simulator_sha256"]
            or trajectory.provenance.experiment_sha256 != config.sha256
            or trajectory.outcome.value != "success"
            or not bool(evidence.tensors["oracle_success"][-1])
            or bool(evidence.tensors["oracle_failure"][-1])
        ):
            raise ValueError("validation member shard content changed")
        corpus.append((trajectory, evidence))
    if tuple(observed_seeds) != config.seeds:
        raise ValueError("validation member ordering changed")
    if {entry.name for entry in (root / "shards").iterdir()} != shard_files:
        raise ValueError("validation artifact contains uncommitted shard files")
    if (
        grasp_lift_composite_corpus_sha256(corpus)
        != manifest["composite_corpus_sha256"]
    ):
        raise ValueError("validation composite corpus hash changed")
    expected_counts = {"failure": 0, "success": 6, "timeout": 0}
    acceptance = _exact_mapping(
        manifest["acceptance"],
        {
            "collision_buffer_overflow_count",
            "nonzero_reward_transition_count",
            "outcome_counts",
            "passed",
            "policy",
            "safety_violation_count",
        },
        "validation authoring acceptance",
    )
    if acceptance != {
        "collision_buffer_overflow_count": 0,
        "nonzero_reward_transition_count": 0,
        "outcome_counts": expected_counts,
        "passed": True,
        "policy": dict(config.acceptance),
        "safety_violation_count": 0,
    }:
        raise ValueError("validation authoring acceptance changed")
    if (
        status["complete"] is not True
        or status["accepted"] is not True
        or status["cohort_slot_id"] != manifest["cohort_slot_id"]
        or status["manifest_sha256"] != manifest["self_sha256"]
        or status["outcome_counts"] != expected_counts
        or status["safety_violation_count"] != 0
        or status["collision_buffer_overflow_count"] != 0
        or status["nonzero_reward_transition_count"] != 0
    ):
        raise ValueError("validation authoring status changed")
    if (
        isinstance(manifest["world_steps"], bool)
        or not isinstance(manifest["world_steps"], int)
        or manifest["world_steps"] < 1
        or not isinstance(manifest["elapsed_seconds"], (int, float))
        or isinstance(manifest["elapsed_seconds"], bool)
        or not math.isfinite(float(manifest["elapsed_seconds"]))
        or manifest["elapsed_seconds"] <= 0
    ):
        raise ValueError("validation authoring runtime metrics changed")
    return manifest


__all__ = [
    "GRASP_LIFT_VALIDATION_AUTHORING_COMMIT_SCHEMA_ID",
    "GRASP_LIFT_VALIDATION_AUTHORING_CONFIG_SCHEMA_ID",
    "GRASP_LIFT_VALIDATION_AUTHORING_MANIFEST_SCHEMA_ID",
    "GRASP_LIFT_VALIDATION_AUTHORING_STATUS_SCHEMA_ID",
    "GRASP_LIFT_VALIDATION_LOCKED_TEST",
    "GRASP_LIFT_VALIDATION_PURPOSE",
    "GRASP_LIFT_VALIDATION_RESET_SEEDS",
    "GRASP_LIFT_VALIDATION_SEED_DERIVATION",
    "GRASP_LIFT_VALIDATION_SEED_NAMESPACE",
    "GraspLiftValidationAuthoringConfig",
    "derive_grasp_lift_validation_seeds",
    "grasp_lift_validation_cohort_slot_id",
    "grasp_lift_validation_cohort_slot_record",
    "grasp_lift_validation_environment_config_record",
    "grasp_lift_validation_preregistration_record",
    "load_grasp_lift_validation_authoring_config",
    "load_grasp_lift_validation_authoring_manifest",
    "load_grasp_lift_validation_authoring_metadata",
    "validation_authoring_file_sha256",
]
