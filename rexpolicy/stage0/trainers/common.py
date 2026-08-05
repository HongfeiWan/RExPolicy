"""Shared optimizer safety checks for Stage 0 trainers."""

from __future__ import annotations

import math

import torch
from torch import nn


def positive_finite(value: float, *, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) <= 0.0
    ):
        raise ValueError(f"{name} must be finite and positive")
    return float(value)


def module_device_dtype(module: nn.Module) -> tuple[torch.device, torch.dtype]:
    parameters = tuple(module.parameters())
    if not parameters:
        raise ValueError("trainer module must have parameters")
    devices = {parameter.device for parameter in parameters}
    dtypes = {parameter.dtype for parameter in parameters}
    if len(devices) != 1 or len(dtypes) != 1:
        raise ValueError("trainer module parameters must share device and dtype")
    dtype = next(iter(dtypes))
    if not torch.is_floating_point(torch.empty((), dtype=dtype)):
        raise ValueError("trainer module parameters must be floating point")
    return next(iter(devices)), dtype


def finish_optimizer_step(
    *,
    loss: torch.Tensor,
    module: nn.Module,
    optimizer: torch.optim.Optimizer,
    gradient_clip_norm: float,
) -> float:
    """Backpropagate, reject missing/non-finite grads, clip, and update."""
    if not isinstance(loss, torch.Tensor) or loss.ndim != 0:
        raise ValueError("training loss must be a scalar tensor")
    if not bool(torch.isfinite(loss.detach())):
        raise FloatingPointError("training loss is non-finite")
    loss.backward()
    named = tuple(
        (name, parameter)
        for name, parameter in module.named_parameters()
        if parameter.requires_grad
    )
    missing = tuple(name for name, parameter in named if parameter.grad is None)
    if missing:
        optimizer.zero_grad(set_to_none=True)
        raise RuntimeError(
            "trainable parameters received no gradient; DDP must use "
            f"find_unused_parameters=False only with a complete graph: {missing}"
        )
    non_finite = tuple(
        name
        for name, parameter in named
        if not bool(torch.isfinite(parameter.grad).all())
    )
    if non_finite:
        optimizer.zero_grad(set_to_none=True)
        raise FloatingPointError(f"non-finite gradients: {non_finite}")
    parameters = tuple(parameter for _, parameter in named)
    norm = torch.nn.utils.clip_grad_norm_(
        parameters,
        max_norm=gradient_clip_norm,
        error_if_nonfinite=True,
    )
    gradient_norm = float(norm.detach())
    if not math.isfinite(gradient_norm):
        optimizer.zero_grad(set_to_none=True)
        raise FloatingPointError("gradient norm is non-finite")
    optimizer.step()
    for name, parameter in named:
        if not bool(torch.isfinite(parameter.detach()).all()):
            raise FloatingPointError(f"optimizer produced non-finite parameter: {name}")
    return gradient_norm
