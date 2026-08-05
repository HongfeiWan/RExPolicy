"""Fingerprintable train-split normalization for Stage 0 state/action data."""

from __future__ import annotations

import json
import math
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch

from rexpolicy.stage0.types import canonical_fingerprint


STAGE0_NORMALIZATION_SCHEMA_ID = "rexpolicy/stage0-normalization/v1"
STAGE0_NORMALIZATION_SCHEMA_VERSION = 1


def _vector(value: Any, *, name: str, dtype: torch.dtype) -> torch.Tensor:
    if not isinstance(value, torch.Tensor) or value.ndim != 1:
        raise ValueError(f"{name} must be a rank-one tensor")
    if value.numel() < 1 or value.dtype is not dtype or value.device.type != "cpu":
        raise ValueError(f"{name} must be non-empty {dtype} on CPU")
    if not value.is_contiguous():
        raise ValueError(f"{name} must be contiguous")
    if torch.is_floating_point(value) and not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} contains non-finite values")
    return value


def _positive_finite(value: Any, *, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) <= 0.0
    ):
        raise ValueError(f"{name} must be finite and positive")
    return float(value)


@dataclass(frozen=True)
class Stage0Normalization:
    """Immutable statistics fitted only on the training trajectory split.

    Non-effective action dimensions are explicitly mapped to zero.  Newton's
    task adapter restores its reset-state hold targets before execution, so
    policy capacity and loss are not spent reproducing irrelevant joints.
    """

    state_mean: torch.Tensor
    state_std: torch.Tensor
    action_mean: torch.Tensor
    action_std: torch.Tensor
    effective_action_mask: torch.Tensor
    state_sample_count: int
    action_sample_count: int
    epsilon: float = 1.0e-4

    def __post_init__(self) -> None:
        state_mean = _vector(self.state_mean, name="state_mean", dtype=torch.float32)
        state_std = _vector(self.state_std, name="state_std", dtype=torch.float32)
        action_mean = _vector(self.action_mean, name="action_mean", dtype=torch.float32)
        action_std = _vector(self.action_std, name="action_std", dtype=torch.float32)
        action_mask = _vector(
            self.effective_action_mask,
            name="effective_action_mask",
            dtype=torch.bool,
        )
        if state_mean.shape != state_std.shape:
            raise ValueError("state mean/std dimensions differ")
        if (
            action_mean.shape != action_std.shape
            or action_mean.shape != action_mask.shape
        ):
            raise ValueError("action mean/std/mask dimensions differ")
        if not bool(action_mask.any()):
            raise ValueError(
                "effective_action_mask must retain at least one action field"
            )
        epsilon = _positive_finite(self.epsilon, name="epsilon")
        if bool((state_std < epsilon).any()) or bool((action_std <= 0.0).any()):
            raise ValueError("normalization standard deviations violate epsilon")
        if type(self.state_sample_count) is not int or self.state_sample_count < 2:
            raise ValueError("state_sample_count must be an integer >= 2")
        if type(self.action_sample_count) is not int or self.action_sample_count < 2:
            raise ValueError("action_sample_count must be an integer >= 2")
        if bool(action_mean[~action_mask].count_nonzero()):
            raise ValueError("non-effective action means must be zero")
        if not torch.equal(
            action_std[~action_mask], torch.ones_like(action_std[~action_mask])
        ):
            raise ValueError("non-effective action standard deviations must be one")

    @property
    def state_dim(self) -> int:
        return int(self.state_mean.numel())

    @property
    def action_dim(self) -> int:
        return int(self.action_mean.numel())

    def to_record(self) -> dict[str, Any]:
        return {
            "action_mean": self.action_mean.tolist(),
            "action_sample_count": self.action_sample_count,
            "action_std": self.action_std.tolist(),
            "effective_action_mask": self.effective_action_mask.tolist(),
            "epsilon": self.epsilon,
            "schema_id": STAGE0_NORMALIZATION_SCHEMA_ID,
            "schema_version": STAGE0_NORMALIZATION_SCHEMA_VERSION,
            "state_mean": self.state_mean.tolist(),
            "state_sample_count": self.state_sample_count,
            "state_std": self.state_std.tolist(),
        }

    @property
    def sha256(self) -> str:
        return canonical_fingerprint(self.to_record())

    @classmethod
    def from_record(cls, value: Mapping[str, Any]) -> Stage0Normalization:
        if not isinstance(value, Mapping):
            raise ValueError("normalization record must be an object")
        expected = {
            "action_mean",
            "action_sample_count",
            "action_std",
            "effective_action_mask",
            "epsilon",
            "schema_id",
            "schema_version",
            "state_mean",
            "state_sample_count",
            "state_std",
        }
        if set(value) != expected:
            raise ValueError("normalization record fields changed")
        if value["schema_id"] != STAGE0_NORMALIZATION_SCHEMA_ID:
            raise ValueError("normalization schema_id mismatch")
        if value["schema_version"] != STAGE0_NORMALIZATION_SCHEMA_VERSION:
            raise ValueError("normalization schema_version mismatch")
        try:
            tensors = {
                "state_mean": torch.tensor(value["state_mean"], dtype=torch.float32),
                "state_std": torch.tensor(value["state_std"], dtype=torch.float32),
                "action_mean": torch.tensor(value["action_mean"], dtype=torch.float32),
                "action_std": torch.tensor(value["action_std"], dtype=torch.float32),
                "effective_action_mask": torch.tensor(
                    value["effective_action_mask"], dtype=torch.bool
                ),
            }
        except (TypeError, ValueError) as error:
            raise ValueError("normalization record contains invalid vectors") from error
        return cls(
            **tensors,
            state_sample_count=value["state_sample_count"],
            action_sample_count=value["action_sample_count"],
            epsilon=value["epsilon"],
        )

    def _stats(
        self, reference: torch.Tensor, *, action: bool
    ) -> tuple[torch.Tensor, ...]:
        expected = self.action_dim if action else self.state_dim
        if not isinstance(reference, torch.Tensor) or reference.shape[-1] != expected:
            kind = "action" if action else "state"
            raise ValueError(f"{kind} tensor must end in dimension {expected}")
        if not torch.is_floating_point(reference) or not bool(
            torch.isfinite(reference).all()
        ):
            raise ValueError("normalization input must be finite floating point")
        mean = self.action_mean if action else self.state_mean
        std = self.action_std if action else self.state_std
        return (
            mean.to(device=reference.device, dtype=reference.dtype),
            std.to(device=reference.device, dtype=reference.dtype),
        )

    def normalize_states(
        self,
        states: torch.Tensor,
        *,
        valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        mean, std = self._stats(states, action=False)
        result = (states - mean) / std
        return _mask_steps(result, valid_mask)

    def normalize_actions(
        self,
        actions: torch.Tensor,
        *,
        valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        mean, std = self._stats(actions, action=True)
        result = (actions - mean) / std
        effective = self.effective_action_mask.to(device=actions.device)
        result = result.masked_fill(~effective, 0.0)
        return _mask_steps(result, valid_mask)

    def denormalize_actions(
        self,
        actions: torch.Tensor,
        *,
        valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        mean, std = self._stats(actions, action=True)
        result = actions * std + mean
        effective = self.effective_action_mask.to(device=actions.device)
        result = result.masked_fill(~effective, 0.0)
        return _mask_steps(result, valid_mask)


def _mask_steps(values: torch.Tensor, valid_mask: torch.Tensor | None) -> torch.Tensor:
    if valid_mask is None:
        return values
    if (
        not isinstance(valid_mask, torch.Tensor)
        or valid_mask.dtype is not torch.bool
        or valid_mask.device != values.device
        or tuple(valid_mask.shape) != tuple(values.shape[:-1])
    ):
        raise ValueError("valid_mask must be bool and match all non-feature axes")
    return values.masked_fill(~valid_mask.unsqueeze(-1), 0.0)


def fit_stage0_normalization(
    states: torch.Tensor,
    actions: torch.Tensor,
    *,
    effective_action_mask: torch.Tensor,
    epsilon: float = 1.0e-4,
) -> Stage0Normalization:
    """Fit deterministic CPU statistics from train-split executed samples."""
    epsilon = _positive_finite(epsilon, name="epsilon")
    for name, values in (("states", states), ("actions", actions)):
        if (
            not isinstance(values, torch.Tensor)
            or values.ndim != 2
            or values.shape[0] < 2
            or values.shape[1] < 1
            or not torch.is_floating_point(values)
            or not bool(torch.isfinite(values).all())
        ):
            raise ValueError(f"{name} must be finite floating [samples, features]")
    if (
        not isinstance(effective_action_mask, torch.Tensor)
        or effective_action_mask.dtype is not torch.bool
        or effective_action_mask.ndim != 1
        or effective_action_mask.shape[0] != actions.shape[1]
        or not bool(effective_action_mask.any())
    ):
        raise ValueError("effective_action_mask must be a non-empty bool action vector")
    states64 = states.detach().to(device="cpu", dtype=torch.float64)
    actions64 = actions.detach().to(device="cpu", dtype=torch.float64)
    effective = effective_action_mask.detach().to(device="cpu").contiguous()
    state_mean = states64.mean(dim=0)
    state_std = states64.std(dim=0, unbiased=False).clamp_min(epsilon)
    action_mean = torch.zeros(actions.shape[1], dtype=torch.float64)
    action_std = torch.ones(actions.shape[1], dtype=torch.float64)
    action_mean[effective] = actions64[:, effective].mean(dim=0)
    action_std[effective] = (
        actions64[:, effective].std(dim=0, unbiased=False).clamp_min(epsilon)
    )
    return Stage0Normalization(
        state_mean=state_mean.to(torch.float32).contiguous(),
        state_std=state_std.to(torch.float32).contiguous(),
        action_mean=action_mean.to(torch.float32).contiguous(),
        action_std=action_std.to(torch.float32).contiguous(),
        effective_action_mask=effective,
        state_sample_count=int(states.shape[0]),
        action_sample_count=int(actions.shape[0]),
        epsilon=epsilon,
    )


def write_stage0_normalization(path: Path, normalization: Stage0Normalization) -> None:
    """Atomically persist a human-readable, hash-stable normalization artifact."""
    if not isinstance(path, Path):
        raise TypeError("path must be a pathlib.Path")
    if not isinstance(normalization, Stage0Normalization):
        raise TypeError("normalization must be Stage0Normalization")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial-{uuid.uuid4().hex}")
    payload = (
        json.dumps(
            normalization.to_record(),
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def read_stage0_normalization(path: Path) -> Stage0Normalization:
    if not isinstance(path, Path):
        raise TypeError("path must be a pathlib.Path")
    return Stage0Normalization.from_record(json.loads(path.read_text(encoding="utf-8")))


__all__ = [
    "STAGE0_NORMALIZATION_SCHEMA_ID",
    "STAGE0_NORMALIZATION_SCHEMA_VERSION",
    "Stage0Normalization",
    "fit_stage0_normalization",
    "read_stage0_normalization",
    "write_stage0_normalization",
]
