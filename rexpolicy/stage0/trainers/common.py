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


def _non_finite_names(
    named_tensors: tuple[tuple[str, torch.Tensor], ...],
) -> tuple[str, ...]:
    """Check any number of tensors with one device-to-host synchronization."""

    if not named_tensors:
        return ()
    flags = torch.stack(
        tuple(torch.isfinite(value.detach()).all() for _, value in named_tensors)
    )
    if bool(flags.all()):
        return ()
    finite = flags.detach().cpu().tolist()
    return tuple(
        name for (name, _), is_finite in zip(named_tensors, finite) if not is_finite
    )


def materialize_finite_scalars(
    **values: torch.Tensor,
) -> dict[str, float]:
    """Materialize scalar metrics through one synchronized device transfer."""

    if not values:
        raise ValueError("at least one scalar metric is required")
    items = tuple(values.items())
    device = items[0][1].device if isinstance(items[0][1], torch.Tensor) else None
    if any(
        not isinstance(value, torch.Tensor)
        or value.ndim != 0
        or value.device != device
        for _, value in items
    ):
        raise ValueError("metrics must be scalar tensors on one device")
    materialized = torch.stack(tuple(value.detach().float() for _, value in items))
    result = materialized.cpu().tolist()
    non_finite = tuple(
        name for (name, _), value in zip(items, result) if not math.isfinite(value)
    )
    if non_finite:
        raise FloatingPointError(f"non-finite optimizer metrics: {non_finite}")
    return {name: float(value) for (name, _), value in zip(items, result)}


def finish_optimizer_step(
    *,
    loss: torch.Tensor,
    module: nn.Module,
    optimizer: torch.optim.Optimizer,
    gradient_clip_norm: float,
) -> torch.Tensor:
    """Backpropagate, reject missing/non-finite grads, clip, and update."""
    if not isinstance(loss, torch.Tensor) or loss.ndim != 0:
        raise ValueError("training loss must be a scalar tensor")
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
    non_finite = _non_finite_names(
        (("training loss", loss),)
        + tuple(
            (f"gradient {name}", parameter.grad) for name, parameter in named
        )
    )
    if non_finite:
        optimizer.zero_grad(set_to_none=True)
        raise FloatingPointError(f"non-finite loss or gradients: {non_finite}")
    parameters = tuple(parameter for _, parameter in named)
    norm = torch.nn.utils.clip_grad_norm_(
        parameters,
        max_norm=gradient_clip_norm,
        error_if_nonfinite=False,
    )
    if not bool(torch.isfinite(norm.detach())):
        optimizer.zero_grad(set_to_none=True)
        raise FloatingPointError("gradient norm is non-finite")
    optimizer.step()
    non_finite_parameters = _non_finite_names(
        tuple((name, parameter) for name, parameter in named)
    )
    if non_finite_parameters:
        raise FloatingPointError(
            "optimizer produced non-finite parameters: "
            f"{non_finite_parameters}"
        )
    return norm.detach()
