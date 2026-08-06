"""Compatible successor objective for reachable success latents.

RExPolicy v2 already trains ``SuccessModeSelector`` as a state-conditioned
Gaussian over successful futures.  v2.1 names that contract explicitly as a
successor model and adds uncertainty calibration without introducing a second
set of deployment weights.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from typing import Any, Mapping

import torch

from .config import SuccessManifoldConfig
from .losses import success_selector_losses
from .selector import DiagonalGaussian, SuccessModeSelector


SUCCESSOR_TRAINING_CONFIG_SCHEMA_VERSION = 1


def _finite_non_negative(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return result


@dataclass(frozen=True)
class SuccessorTrainingConfig:
    """Versioned training-only settings, kept outside the v2 bundle hash."""

    schema_version: int = SUCCESSOR_TRAINING_CONFIG_SCHEMA_VERSION
    enabled: bool = False
    cosine_weight: float = 1.0
    calibration_weight: float = 0.1
    learning_rate: float = 1.0e-4
    weight_decay: float = 1.0e-4
    gradient_clip_norm: float = 1.0

    def __post_init__(self) -> None:
        if self.schema_version != SUCCESSOR_TRAINING_CONFIG_SCHEMA_VERSION:
            raise ValueError(
                "SuccessorTrainingConfig schema_version must be "
                f"{SUCCESSOR_TRAINING_CONFIG_SCHEMA_VERSION}"
            )
        if type(self.enabled) is not bool:
            raise ValueError("enabled must be a boolean")
        for name in (
            "cosine_weight",
            "calibration_weight",
            "weight_decay",
        ):
            _finite_non_negative(getattr(self, name), name)
        for name in ("learning_rate", "gradient_clip_norm"):
            if _finite_non_negative(getattr(self, name), name) == 0.0:
                raise ValueError(f"{name} must be positive")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> SuccessorTrainingConfig:
        if not isinstance(value, Mapping):
            raise ValueError("Successor training config must be an object")
        known = {item.name for item in fields(cls)}
        unknown = sorted(set(value).difference(known))
        if unknown:
            raise ValueError(
                "Unknown successor training config fields: "
                + ", ".join(unknown)
            )
        return cls(**dict(value))

    def to_record(self) -> dict[str, Any]:
        return {item.name: getattr(self, item.name) for item in fields(self)}


class SuccessorModel(SuccessModeSelector):
    """The v2 selector under its v2.1 reachable-success training contract.

    This subclass deliberately adds no parameters.  Its state dictionary is
    therefore strictly interchangeable with ``SuccessModeSelector`` and the
    existing Flow-DiT deployment bundle remains byte-schema compatible.
    """

    def __init__(self, config: SuccessManifoldConfig) -> None:
        super().__init__(config)


@dataclass(frozen=True)
class SuccessorLosses:
    negative_log_likelihood: torch.Tensor
    cosine_alignment: torch.Tensor
    calibration: torch.Tensor
    total: torch.Tensor


def successor_losses(
    distribution: DiagonalGaussian,
    target_latents: torch.Tensor,
    *,
    config: SuccessorTrainingConfig,
) -> SuccessorLosses:
    """NLL, directional alignment, and standardized-residual calibration.

    A calibrated diagonal Gaussian has unit expected squared standardized
    residual per latent dimension.  The calibration term is computed per
    sample before averaging so over- and under-confident examples cannot
    cancel each other.
    """

    if not config.enabled:
        raise ValueError("Successor losses require an enabled training config")
    base = success_selector_losses(
        distribution,
        target_latents,
        contrastive_weight=0.0,
    )
    squared_standardized_residual = (
        (target_latents - distribution.mean).square()
        * torch.exp(-2.0 * distribution.log_std)
    )
    residual_energy = squared_standardized_residual.mean(dim=-1)
    calibration = (residual_energy - 1.0).square().mean()
    cosine_alignment = base.contrastive_alignment
    total = (
        base.negative_log_likelihood
        + float(config.cosine_weight) * cosine_alignment
        + float(config.calibration_weight) * calibration
    )
    return SuccessorLosses(
        negative_log_likelihood=base.negative_log_likelihood,
        cosine_alignment=cosine_alignment,
        calibration=calibration,
        total=total,
    )


__all__ = [
    "SUCCESSOR_TRAINING_CONFIG_SCHEMA_VERSION",
    "SuccessorLosses",
    "SuccessorModel",
    "SuccessorTrainingConfig",
    "successor_losses",
]
