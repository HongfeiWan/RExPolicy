"""Strict configuration contract for reward-free Grasp-Lift pilot authoring."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from rexpolicy.stage0.types import canonical_fingerprint


GRASP_LIFT_PILOT_CONFIG_SCHEMA_VERSION = 1
GRASP_LIFT_PILOT_CONFIG_SCHEMA_ID = "rexpolicy/stage0-grasp-lift-pilot/v1"
GRASP_LIFT_PILOT_OUTPUT_SCHEMA_ID = "rexpolicy/stage0-grasp-lift-pilot-output/v1"
GRASP_LIFT_PILOT_STATUS_SCHEMA_ID = "rexpolicy/stage0-grasp-lift-pilot-status/v1"
GRASP_LIFT_PILOT_COMMIT_SCHEMA_ID = "rexpolicy/stage0-grasp-lift-pilot-commit/v1"
GRASP_LIFT_PILOT_SPLIT_NAMESPACE = "rexpolicy/grasp-lift-split/20260806-v1"
GRASP_LIFT_PILOT_VALIDATION_FRACTION = 0.25
GRASP_LIFT_PILOT_SPLIT_POLICY_ID = (
    "rexpolicy/stage0-grasp-lift-reset-group-split/v1"
)


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
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read Grasp-Lift pilot config {source}: {error}") from error
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
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read {context} {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{context} must contain an object")
    return value


def load_grasp_lift_pilot_manifest(directory: str | Path) -> dict[str, Any]:
    """Verify the final commit/status/manifest chain for a pilot output."""
    root = Path(directory)
    commit = _read_pilot_json(root / "commit.json", "pilot commit")
    verify_grasp_lift_pilot_record(commit, "pilot commit")
    expected_commit_keys = {
        "manifest_file",
        "manifest_sha256",
        "schema_id",
        "self_sha256",
        "status_file",
        "status_sha256",
    }
    if set(commit) != expected_commit_keys:
        raise ValueError("pilot commit keys changed")
    if commit["schema_id"] != GRASP_LIFT_PILOT_COMMIT_SCHEMA_ID:
        raise ValueError("pilot commit schema changed")
    if (
        commit["manifest_file"] != "manifest.json"
        or commit["status_file"] != "status.json"
    ):
        raise ValueError("pilot commit file names changed")

    manifest = _read_pilot_json(root / "manifest.json", "pilot manifest")
    status = _read_pilot_json(root / "status.json", "pilot status")
    verify_grasp_lift_pilot_record(manifest, "pilot manifest")
    verify_grasp_lift_pilot_record(status, "pilot status")
    if manifest.get("schema_id") != GRASP_LIFT_PILOT_OUTPUT_SCHEMA_ID:
        raise ValueError("pilot manifest schema changed")
    if status.get("schema_id") != GRASP_LIFT_PILOT_STATUS_SCHEMA_ID:
        raise ValueError("pilot status schema changed")
    if commit["manifest_sha256"] != manifest["self_sha256"]:
        raise ValueError("pilot commit does not bind the manifest")
    if commit["status_sha256"] != status["self_sha256"]:
        raise ValueError("pilot commit does not bind status")
    if status.get("manifest_sha256") != manifest["self_sha256"]:
        raise ValueError("pilot status does not bind the manifest")
    if status.get("complete") is not True:
        raise ValueError("pilot status is not complete")
    acceptance = manifest.get("acceptance")
    if not isinstance(acceptance, Mapping):
        raise ValueError("pilot manifest acceptance must be an object")
    if status.get("acceptance_passed") != acceptance.get("passed"):
        raise ValueError("pilot status acceptance disagrees with the manifest")
    if status.get("outcome_counts") != manifest.get("outcome_counts"):
        raise ValueError("pilot status outcomes disagree with the manifest")
    expected_split_policy = grasp_lift_pilot_split_policy()
    if manifest.get("split_policy") != expected_split_policy:
        raise ValueError("pilot split policy changed")
    if manifest.get("split_policy_sha256") != canonical_fingerprint(
        expected_split_policy
    ):
        raise ValueError("pilot split policy hash changed")
    members = manifest.get("members")
    if not isinstance(members, list) or not members:
        raise ValueError("pilot manifest members must be a non-empty list")
    split_by_group: dict[str, str] = {}
    trajectory_groups: dict[str, str] = {}
    actual_splits: dict[str, str] = {}
    for member in members:
        if not isinstance(member, Mapping):
            raise ValueError("pilot manifest member must be an object")
        trajectory_id = member.get("trajectory_id")
        group = member.get("reset_group_id")
        split = member.get("split")
        if (
            not isinstance(trajectory_id, str)
            or not trajectory_id
            or trajectory_id in trajectory_groups
        ):
            raise ValueError("pilot manifest has an invalid trajectory ID")
        if not isinstance(group, str) or split not in {"train", "validation"}:
            raise ValueError("pilot manifest member has an invalid split")
        trajectory_groups[trajectory_id] = group
        actual_splits[trajectory_id] = split
        previous = split_by_group.setdefault(group, split)
        if previous != split:
            raise ValueError("pilot reset group crosses train/validation splits")
    expected_splits = assign_grasp_lift_pilot_splits(
        trajectory_groups,
        validation_fraction=GRASP_LIFT_PILOT_VALIDATION_FRACTION,
    )
    if actual_splits != expected_splits:
        raise ValueError("pilot manifest split assignments violate policy")
    locked_test = manifest.get("locked_test")
    if locked_test != {
        "artifact": None,
        "consumed": False,
        "policy": "not_created",
    }:
        raise ValueError("pilot manifest illegally created or consumed locked test")
    return manifest


__all__ = [
    "GRASP_LIFT_PILOT_CONFIG_SCHEMA_ID",
    "GRASP_LIFT_PILOT_CONFIG_SCHEMA_VERSION",
    "GRASP_LIFT_PILOT_COMMIT_SCHEMA_ID",
    "GRASP_LIFT_PILOT_OUTPUT_SCHEMA_ID",
    "GRASP_LIFT_PILOT_SPLIT_NAMESPACE",
    "GRASP_LIFT_PILOT_SPLIT_POLICY_ID",
    "GRASP_LIFT_PILOT_STATUS_SCHEMA_ID",
    "GRASP_LIFT_PILOT_VALIDATION_FRACTION",
    "GraspLiftPilotAcceptanceConfig",
    "GraspLiftPilotConfig",
    "GraspLiftPilotRuntimeConfig",
    "assign_grasp_lift_pilot_splits",
    "grasp_lift_pilot_split_policy",
    "load_grasp_lift_pilot_config",
    "load_grasp_lift_pilot_manifest",
    "seal_grasp_lift_pilot_record",
    "verify_grasp_lift_pilot_record",
]
