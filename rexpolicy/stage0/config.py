"""Strict, dependency-free configuration for RExPolicy Stage 0."""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from typing import Any, Mapping

from .types import (
    Stage0SystemPath,
    Stage0Task,
    canonical_fingerprint,
    parse_string_enum,
)


STAGE0_CONFIG_SCHEMA_VERSION = 1
STAGE0_STATE_SCHEMA_ID = "stage0/newton-state/v1"
STAGE0_STATE_DIM = 79
STAGE0_ACTION_DIM = 19


def _exact_int(
    value: Any,
    name: str,
    *,
    minimum: int,
    maximum: int | None = None,
) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    if maximum is not None and value > maximum:
        raise ValueError(f"{name} must be <= {maximum}")
    return value


def _finite_number(
    value: Any,
    name: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    if maximum is not None and result > maximum:
        raise ValueError(f"{name} must be <= {maximum}")
    return result


@dataclass(frozen=True)
class Stage0ExperimentConfig:
    """One complete, fingerprinted Stage 0 experiment contract.

    The feature is default-off. Enabling it requires the isolated state-only
    system path; image rendering and GR00T imports are forbidden by this
    schema instead of being silently ignored.
    """

    schema_version: int = STAGE0_CONFIG_SCHEMA_VERSION
    enabled: bool = False
    system_path: Stage0SystemPath = Stage0SystemPath.STAGE0_STATE
    task: Stage0Task = Stage0Task.REACH
    seed: int = 0
    num_envs: int = 1
    max_episode_steps: int = 100
    state_schema_id: str = STAGE0_STATE_SCHEMA_ID
    state_dim: int = STAGE0_STATE_DIM
    action_dim: int = STAGE0_ACTION_DIM
    action_horizon: int = 8
    future_horizon: int = 32
    window_stride: int = 1
    latent_dim: int = 32
    model_dim: int = 128
    attention_heads: int = 4
    transformer_layers: int = 2
    feedforward_dim: int = 256
    dropout: float = 0.0
    render_images: bool = False
    use_groot: bool = False

    def __post_init__(self) -> None:
        if (
            type(self.schema_version) is not int
            or self.schema_version != STAGE0_CONFIG_SCHEMA_VERSION
        ):
            raise ValueError(
                "Stage0ExperimentConfig schema_version must be "
                f"{STAGE0_CONFIG_SCHEMA_VERSION}"
            )
        if type(self.enabled) is not bool:
            raise ValueError("enabled must be a boolean")
        if type(self.render_images) is not bool:
            raise ValueError("render_images must be a boolean")
        if type(self.use_groot) is not bool:
            raise ValueError("use_groot must be a boolean")

        system_path = parse_string_enum(
            Stage0SystemPath,
            self.system_path,
            "system_path",
        )
        task = parse_string_enum(Stage0Task, self.task, "task")
        object.__setattr__(self, "system_path", system_path)
        object.__setattr__(self, "task", task)

        _exact_int(self.seed, "seed", minimum=0, maximum=(1 << 63) - 1)
        for name in (
            "num_envs",
            "max_episode_steps",
            "state_dim",
            "action_dim",
            "action_horizon",
            "future_horizon",
            "window_stride",
            "latent_dim",
            "model_dim",
            "attention_heads",
            "transformer_layers",
            "feedforward_dim",
        ):
            _exact_int(getattr(self, name), name, minimum=1)
        if self.state_schema_id != STAGE0_STATE_SCHEMA_ID:
            raise ValueError(f"state_schema_id must be {STAGE0_STATE_SCHEMA_ID!r}")
        if self.state_dim != STAGE0_STATE_DIM:
            raise ValueError(f"state_dim must be {STAGE0_STATE_DIM}")
        if self.action_dim != STAGE0_ACTION_DIM:
            raise ValueError(f"action_dim must be {STAGE0_ACTION_DIM}")
        if self.action_horizon > self.future_horizon:
            raise ValueError("action_horizon must be <= future_horizon")
        if self.model_dim % self.attention_heads != 0:
            raise ValueError("model_dim must be divisible by attention_heads")
        dropout = _finite_number(
            self.dropout,
            "dropout",
            minimum=0.0,
            maximum=1.0,
        )
        if dropout >= 1.0:
            raise ValueError("dropout must be less than 1")

        if self.render_images:
            raise ValueError("Stage 0 forbids image rendering")
        if self.use_groot:
            raise ValueError("Stage 0 forbids GR00T dependencies")
        if self.enabled and self.system_path is not Stage0SystemPath.STAGE0_STATE:
            raise ValueError("enabled Stage 0 requires system_path='stage0_state'")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> Stage0ExperimentConfig:
        """Construct a config without accepting missing-type coercions or typos."""
        if not isinstance(value, Mapping):
            raise ValueError("Stage 0 config must be an object")
        known = {item.name for item in fields(cls)}
        unknown = sorted(set(value).difference(known))
        if unknown:
            raise ValueError("Unknown Stage 0 config fields: " + ", ".join(unknown))
        return cls(**dict(value))

    def to_record(self) -> dict[str, Any]:
        """Return every semantic field in a JSON-compatible record."""
        record = {item.name: getattr(self, item.name) for item in fields(self)}
        record["system_path"] = self.system_path.value
        record["task"] = self.task.value
        return record

    @property
    def fingerprint(self) -> str:
        """Hash the complete semantic configuration."""
        return canonical_fingerprint(self.to_record())

    def require_enabled_stage0(self) -> None:
        """Fail closed at Stage 0 entry points."""
        if not self.enabled:
            raise RuntimeError("Stage 0 is disabled")
        if self.system_path is not Stage0SystemPath.STAGE0_STATE:
            raise RuntimeError("Stage 0 entry point requires stage0_state")


# Short public spelling for runners and configuration loaders.
Stage0Config = Stage0ExperimentConfig


__all__ = [
    "STAGE0_ACTION_DIM",
    "STAGE0_CONFIG_SCHEMA_VERSION",
    "STAGE0_STATE_DIM",
    "STAGE0_STATE_SCHEMA_ID",
    "Stage0Config",
    "Stage0ExperimentConfig",
]
