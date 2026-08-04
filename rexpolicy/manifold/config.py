"""Strict, versioned configuration for the optional success manifold.

The default record is deliberately disabled.  Importing this module has no
PyTorch dependency, so task authoring and archive inspection stay lightweight.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from typing import Any, Mapping


SUCCESS_MANIFOLD_CONFIG_SCHEMA_VERSION = 1


def _exact_int(value: Any, name: str, *, minimum: int) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
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
class SuccessManifoldConfig:
    """One complete v1 contract for the success-manifold components.

    ``state_dim=0`` is valid only while the feature is disabled.  This lets a
    v1 runner construct the default config without knowing any model shapes.
    Enabling the feature requires an explicit state dimension.
    """

    schema_version: int = SUCCESS_MANIFOLD_CONFIG_SCHEMA_VERSION
    enabled: bool = False
    state_dim: int = 0
    action_dim: int = 19
    visual_dim: int = 0
    model_dim: int = 256
    latent_dim: int = 64
    condition_dim: int = 256
    projection_dim: int = 128
    max_future_steps: int = 64
    transformer_layers: int = 4
    attention_heads: int = 8
    feedforward_dim: int = 1024
    dropout: float = 0.1
    reconstruction_mask_probability: float = 0.3
    info_nce_temperature: float = 0.1
    variance_target_std: float = 1.0
    reconstruction_weight: float = 1.0
    info_nce_weight: float = 1.0
    variance_weight: float = 0.1
    covariance_weight: float = 0.01
    selector_hidden_dim: int = 256
    selector_layers: int = 2
    selector_min_log_std: float = -5.0
    selector_max_log_std: float = 2.0
    memory_capacity: int = 4096
    novelty_threshold: float = 0.25

    def __post_init__(self) -> None:
        if self.schema_version != SUCCESS_MANIFOLD_CONFIG_SCHEMA_VERSION:
            raise ValueError(
                "SuccessManifoldConfig schema_version must be "
                f"{SUCCESS_MANIFOLD_CONFIG_SCHEMA_VERSION}"
            )
        if type(self.enabled) is not bool:
            raise ValueError("enabled must be a boolean")

        _exact_int(self.state_dim, "state_dim", minimum=0)
        if self.enabled and self.state_dim == 0:
            raise ValueError("state_dim must be positive when the manifold is enabled")
        for name in (
            "action_dim",
            "model_dim",
            "latent_dim",
            "condition_dim",
            "projection_dim",
            "max_future_steps",
            "transformer_layers",
            "attention_heads",
            "feedforward_dim",
            "selector_hidden_dim",
            "selector_layers",
            "memory_capacity",
        ):
            _exact_int(getattr(self, name), name, minimum=1)
        _exact_int(self.visual_dim, "visual_dim", minimum=0)
        if self.model_dim % self.attention_heads != 0:
            raise ValueError("model_dim must be divisible by attention_heads")

        _finite_number(self.dropout, "dropout", minimum=0.0, maximum=1.0)
        if float(self.dropout) >= 1.0:
            raise ValueError("dropout must be less than 1")
        _finite_number(
            self.reconstruction_mask_probability,
            "reconstruction_mask_probability",
            minimum=0.0,
            maximum=1.0,
        )
        _finite_number(
            self.info_nce_temperature,
            "info_nce_temperature",
            minimum=0.0,
        )
        if float(self.info_nce_temperature) == 0.0:
            raise ValueError("info_nce_temperature must be positive")
        _finite_number(
            self.variance_target_std,
            "variance_target_std",
            minimum=0.0,
        )
        if float(self.variance_target_std) == 0.0:
            raise ValueError("variance_target_std must be positive")
        for name in (
            "reconstruction_weight",
            "info_nce_weight",
            "variance_weight",
            "covariance_weight",
            "novelty_threshold",
        ):
            _finite_number(getattr(self, name), name, minimum=0.0)
        minimum_log_std = _finite_number(
            self.selector_min_log_std,
            "selector_min_log_std",
        )
        maximum_log_std = _finite_number(
            self.selector_max_log_std,
            "selector_max_log_std",
        )
        if minimum_log_std >= maximum_log_std:
            raise ValueError(
                "selector_min_log_std must be below selector_max_log_std"
            )

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> SuccessManifoldConfig:
        """Build a config without accepting misspelled or unknown fields."""
        if not isinstance(value, Mapping):
            raise ValueError("Success manifold config must be an object")
        known = {item.name for item in fields(cls)}
        unknown = sorted(set(value).difference(known))
        if unknown:
            raise ValueError(
                "Unknown success manifold config fields: " + ", ".join(unknown)
            )
        return cls(**dict(value))

    def to_record(self) -> dict[str, Any]:
        """Return a JSON-compatible record containing every semantic field."""
        return {item.name: getattr(self, item.name) for item in fields(self)}
