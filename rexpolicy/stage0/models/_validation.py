"""Shared validation helpers for the isolated Stage 0 neural modules."""

from __future__ import annotations

import math

import torch


def require_positive_int(value: int, *, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def require_non_negative_float(value: float, *, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0.0
    ):
        raise ValueError(f"{name} must be finite and non-negative")
    return float(value)


def require_probability(value: float, *, name: str) -> float:
    result = require_non_negative_float(value, name=name)
    if result > 1.0:
        raise ValueError(f"{name} must be between zero and one")
    return result


def require_float_tensor(
    value: torch.Tensor,
    *,
    name: str,
    ndim: int,
    last_dim: int | None = None,
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor) or value.ndim != ndim:
        shape = " x ".join("*" for _ in range(ndim))
        raise ValueError(f"{name} must be a rank-{ndim} tensor [{shape}]")
    if last_dim is not None and value.shape[-1] != last_dim:
        raise ValueError(f"{name} feature dimension must be {last_dim}")
    if not torch.is_floating_point(value):
        raise ValueError(f"{name} must be floating point")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must contain only finite values")
    return value


def require_bool_mask(
    value: torch.Tensor,
    *,
    name: str,
    shape: tuple[int, int],
    device: torch.device,
) -> torch.Tensor:
    if (
        not isinstance(value, torch.Tensor)
        or value.dtype != torch.bool
        or tuple(value.shape) != shape
        or value.device != device
    ):
        raise ValueError(
            f"{name} must be a bool tensor with shape {list(shape)} on the input device"
        )
    return value
