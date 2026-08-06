"""Strict configuration contract for reward-free Grasp-Lift pilot authoring."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from rexpolicy.stage0.types import canonical_fingerprint


GRASP_LIFT_PILOT_CONFIG_SCHEMA_VERSION = 1
GRASP_LIFT_PILOT_CONFIG_SCHEMA_ID = "rexpolicy/stage0-grasp-lift-pilot/v1"


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
        object.__setattr__(
            self,
            "validation_fraction",
            _fraction(
                self.validation_fraction,
                "acceptance.validation_fraction",
                allow_one=False,
            ),
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


__all__ = [
    "GRASP_LIFT_PILOT_CONFIG_SCHEMA_ID",
    "GRASP_LIFT_PILOT_CONFIG_SCHEMA_VERSION",
    "GraspLiftPilotAcceptanceConfig",
    "GraspLiftPilotConfig",
    "GraspLiftPilotRuntimeConfig",
    "load_grasp_lift_pilot_config",
]
