"""Held-out temporal-control diagnostics for Stage 0 flow policies.

The diagnostic deliberately evaluates normalized windows without consulting the
future encoder.  Every window from one trajectory is conditioned on the one
latent encoded at episode start, so later-window scores measure whether a policy
can keep using a fixed success intent as state changes.  Only the sampled
chunk's first action and the explicitly effective action dimensions are scored.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from rexpolicy.stage0.trainers.batch import Stage0WindowBatch


def _positive_int(value: Any, *, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _finite_non_negative(value: Any, *, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0.0
    ):
        raise ValueError(f"{name} must be finite and non-negative")
    return float(value)


@dataclass(frozen=True)
class FirstActionMSE:
    """One immutable aggregate over normalized effective action values."""

    sample_count: int
    effective_action_dimensions: int
    squared_error_sum: float
    first_action_mse: float

    def __post_init__(self) -> None:
        _positive_int(self.sample_count, name="sample_count")
        _positive_int(
            self.effective_action_dimensions,
            name="effective_action_dimensions",
        )
        squared_error_sum = _finite_non_negative(
            self.squared_error_sum,
            name="squared_error_sum",
        )
        first_action_mse = _finite_non_negative(
            self.first_action_mse,
            name="first_action_mse",
        )
        expected = squared_error_sum / (
            self.sample_count * self.effective_action_dimensions
        )
        if not math.isclose(first_action_mse, expected, rel_tol=1e-12, abs_tol=1e-15):
            raise ValueError("first_action_mse must equal the normalized error mean")
        object.__setattr__(self, "squared_error_sum", squared_error_sum)
        object.__setattr__(self, "first_action_mse", first_action_mse)

    @property
    def evaluated_action_values(self) -> int:
        return self.sample_count * self.effective_action_dimensions

    def to_record(self) -> dict[str, Any]:
        return {
            "effective_action_dimensions": self.effective_action_dimensions,
            "evaluated_action_values": self.evaluated_action_values,
            "first_action_mse": self.first_action_mse,
            "sample_count": self.sample_count,
            "squared_error_sum": self.squared_error_sum,
        }


@dataclass(frozen=True)
class ModeFirstActionMSE:
    """First-action error for one trajectory mode."""

    mode_id: str
    metrics: FirstActionMSE

    def __post_init__(self) -> None:
        if not isinstance(self.mode_id, str) or not self.mode_id.strip():
            raise ValueError("mode_id must be non-empty text")
        if not isinstance(self.metrics, FirstActionMSE):
            raise TypeError("metrics must be FirstActionMSE")

    def to_record(self) -> dict[str, Any]:
        return self.metrics.to_record()


@dataclass(frozen=True)
class StartBinFirstActionMSE:
    """First-action error for one half-open window-start interval."""

    start_inclusive: int
    start_exclusive: int
    metrics: FirstActionMSE

    def __post_init__(self) -> None:
        if type(self.start_inclusive) is not int or self.start_inclusive < 0:
            raise ValueError("start_inclusive must be a non-negative integer")
        if (
            type(self.start_exclusive) is not int
            or self.start_exclusive <= self.start_inclusive
        ):
            raise ValueError("start_exclusive must exceed start_inclusive")
        if not isinstance(self.metrics, FirstActionMSE):
            raise TypeError("metrics must be FirstActionMSE")

    @property
    def label(self) -> str:
        return f"[{self.start_inclusive},{self.start_exclusive})"

    def to_record(self) -> dict[str, Any]:
        return {
            "metrics": self.metrics.to_record(),
            "start_exclusive": self.start_exclusive,
            "start_inclusive": self.start_inclusive,
        }


@dataclass(frozen=True)
class TemporalControlMetrics:
    """Fixed-start-latent first-action MSE and its diagnostic slices."""

    overall: FirstActionMSE
    start_zero: FirstActionMSE
    nonzero_start: FirstActionMSE
    per_mode: tuple[ModeFirstActionMSE, ...]
    per_start_bin: tuple[StartBinFirstActionMSE, ...]
    flow_sample_steps: int
    start_bin_width: int
    sample_batch_size: int

    def __post_init__(self) -> None:
        for name in ("overall", "start_zero", "nonzero_start"):
            if not isinstance(getattr(self, name), FirstActionMSE):
                raise TypeError(f"{name} must be FirstActionMSE")
        if not isinstance(self.per_mode, tuple) or not self.per_mode:
            raise ValueError("per_mode must be a non-empty tuple")
        if any(not isinstance(value, ModeFirstActionMSE) for value in self.per_mode):
            raise TypeError("per_mode must contain ModeFirstActionMSE values")
        mode_ids = tuple(value.mode_id for value in self.per_mode)
        if mode_ids != tuple(sorted(set(mode_ids))):
            raise ValueError("per_mode must contain unique modes in sorted order")
        if not isinstance(self.per_start_bin, tuple) or not self.per_start_bin:
            raise ValueError("per_start_bin must be a non-empty tuple")
        if any(
            not isinstance(value, StartBinFirstActionMSE)
            for value in self.per_start_bin
        ):
            raise TypeError("per_start_bin must contain StartBinFirstActionMSE values")
        starts = tuple(value.start_inclusive for value in self.per_start_bin)
        if starts != tuple(sorted(set(starts))):
            raise ValueError("per_start_bin must be unique and sorted")
        _positive_int(self.flow_sample_steps, name="flow_sample_steps")
        _positive_int(self.start_bin_width, name="start_bin_width")
        _positive_int(self.sample_batch_size, name="sample_batch_size")
        if self.start_zero.sample_count + self.nonzero_start.sample_count != (
            self.overall.sample_count
        ):
            raise ValueError(
                "start-zero and nonzero-start slices must partition overall"
            )
        if sum(value.metrics.sample_count for value in self.per_mode) != (
            self.overall.sample_count
        ):
            raise ValueError("per-mode slices must partition overall")
        if sum(value.metrics.sample_count for value in self.per_start_bin) != (
            self.overall.sample_count
        ):
            raise ValueError("start-bin slices must partition overall")

    def to_record(self) -> dict[str, Any]:
        return {
            "flow_sample_steps": self.flow_sample_steps,
            "metric": "normalized_effective_first_action_mse",
            "nonzero_start": self.nonzero_start.to_record(),
            "overall": self.overall.to_record(),
            "per_mode": {value.mode_id: value.to_record() for value in self.per_mode},
            "per_start_bin": {
                value.label: value.to_record() for value in self.per_start_bin
            },
            "protocol": "held_out_fixed_episode_start_latent/v1",
            "sample_batch_size": self.sample_batch_size,
            "start_bin_width": self.start_bin_width,
            "start_zero": self.start_zero.to_record(),
        }


def _module_device_dtype(
    module: nn.Module,
    reference: torch.Tensor,
) -> tuple[torch.device, torch.dtype]:
    tensors = tuple(module.parameters()) + tuple(module.buffers())
    if not tensors:
        return reference.device, reference.dtype
    device = tensors[0].device
    if any(value.device != device for value in tensors):
        raise ValueError("policy parameters and buffers must share one device")
    dtype = next(
        (value.dtype for value in tensors if torch.is_floating_point(value)),
        reference.dtype,
    )
    if not torch.is_floating_point(torch.empty((), dtype=dtype)):
        raise ValueError("policy must expose a floating point dtype")
    return device, dtype


def _policy_contract(policy: nn.Module) -> tuple[int, int, int, int]:
    values: list[int] = []
    for name in ("state_dim", "latent_dim", "action_dim", "action_horizon"):
        value = getattr(policy, name, None)
        if type(value) is not int or value <= 0:
            raise TypeError(f"policy must expose a positive integer {name}")
        values.append(value)
    if not callable(getattr(policy, "sample", None)):
        raise TypeError("policy must provide sample()")
    return tuple(values)  # type: ignore[return-value]


def _validated_design(
    batch: Stage0WindowBatch,
    trajectory_mode_ids: Mapping[str, str],
) -> tuple[str, ...]:
    if not isinstance(trajectory_mode_ids, Mapping):
        raise TypeError("trajectory_mode_ids must be a mapping")
    keys = tuple(zip(batch.trajectory_ids, batch.starts))
    if len(set(keys)) != len(keys):
        raise ValueError("batch cannot repeat a trajectory/start window")

    modes: list[str] = []
    for trajectory_id in batch.trajectory_ids:
        try:
            mode_id = trajectory_mode_ids[trajectory_id]
        except KeyError as error:
            raise ValueError(
                f"trajectory_mode_ids is missing trajectory {trajectory_id!r}"
            ) from error
        if not isinstance(mode_id, str) or not mode_id.strip():
            raise ValueError("trajectory mode IDs must be non-empty text")
        modes.append(mode_id)

    if not any(start == 0 for start in batch.starts):
        raise ValueError("temporal-control evaluation requires start-zero windows")
    if not any(start > 0 for start in batch.starts):
        raise ValueError("temporal-control evaluation rejects a start-zero-only batch")
    mode_tuple = tuple(modes)
    for mode_id in sorted(set(mode_tuple)):
        if not any(
            mode == mode_id and start > 0
            for mode, start in zip(mode_tuple, batch.starts)
        ):
            raise ValueError(f"mode {mode_id!r} must contain a nonzero-start window")
    return mode_tuple


def _episode_latents(
    batch: Stage0WindowBatch,
    episode_start_latents: Mapping[str, torch.Tensor],
    *,
    latent_dim: int,
) -> torch.Tensor:
    if not isinstance(episode_start_latents, Mapping):
        raise TypeError("episode_start_latents must be a mapping")
    rows: list[torch.Tensor] = []
    validated: dict[str, torch.Tensor] = {}
    for trajectory_id in batch.trajectory_ids:
        if trajectory_id not in validated:
            try:
                latent = episode_start_latents[trajectory_id]
            except KeyError as error:
                raise ValueError(
                    f"episode_start_latents is missing trajectory {trajectory_id!r}"
                ) from error
            if (
                not isinstance(latent, torch.Tensor)
                or tuple(latent.shape) != (latent_dim,)
                or not torch.is_floating_point(latent)
                or not bool(torch.isfinite(latent).all())
            ):
                raise ValueError(
                    f"every episode-start latent must be finite floating [{latent_dim}]"
                )
            validated[trajectory_id] = latent.detach()
        rows.append(validated[trajectory_id])
    try:
        return torch.stack(rows)
    except RuntimeError as error:
        raise ValueError("episode-start latents must share device and dtype") from error


def _metric(
    per_sample_squared_error_sum: torch.Tensor,
    indices: tuple[int, ...],
    *,
    effective_action_dimensions: int,
) -> FirstActionMSE:
    if not indices:
        raise ValueError("temporal-control metric slices cannot be empty")
    squared_error_sum = math.fsum(
        float(per_sample_squared_error_sum[index]) for index in indices
    )
    sample_count = len(indices)
    return FirstActionMSE(
        sample_count=sample_count,
        effective_action_dimensions=effective_action_dimensions,
        squared_error_sum=squared_error_sum,
        first_action_mse=squared_error_sum
        / (sample_count * effective_action_dimensions),
    )


def evaluate_fixed_start_latent_first_action_mse(
    policy: nn.Module,
    batch: Stage0WindowBatch,
    *,
    episode_start_latents: Mapping[str, torch.Tensor],
    trajectory_mode_ids: Mapping[str, str],
    initial_noise: torch.Tensor,
    effective_action_mask: torch.Tensor,
    flow_sample_steps: int = 16,
    start_bin_width: int = 4,
    sample_batch_size: int = 256,
) -> TemporalControlMetrics:
    """Score held-out windows using one fixed start latent per trajectory.

    ``batch`` must already be normalized with the immutable train-split
    statistics.  The caller supplies deterministic ``initial_noise`` for every
    row.  The helper never re-encodes later futures and never substitutes a
    later-window latent.
    """

    if not isinstance(policy, nn.Module):
        raise TypeError("policy must be a Torch module")
    if not isinstance(batch, Stage0WindowBatch):
        raise TypeError("batch must be a Stage0WindowBatch")
    flow_sample_steps = _positive_int(
        flow_sample_steps,
        name="flow_sample_steps",
    )
    start_bin_width = _positive_int(start_bin_width, name="start_bin_width")
    sample_batch_size = _positive_int(sample_batch_size, name="sample_batch_size")
    state_dim, latent_dim, action_dim, action_horizon = _policy_contract(policy)
    if batch.current_states.shape[1] != state_dim:
        raise ValueError("policy and batch state dimensions differ")
    if tuple(batch.action_chunks.shape[1:]) != (action_horizon, action_dim):
        raise ValueError("policy and batch action contracts differ")
    if not bool(batch.action_mask[:, 0].all()):
        raise ValueError("every evaluated window must have a valid first action")
    if (
        not isinstance(effective_action_mask, torch.Tensor)
        or effective_action_mask.dtype is not torch.bool
        or tuple(effective_action_mask.shape) != (action_dim,)
        or not bool(effective_action_mask.any())
    ):
        raise ValueError(f"effective_action_mask must be non-empty bool [{action_dim}]")
    if (
        not isinstance(initial_noise, torch.Tensor)
        or tuple(initial_noise.shape) != (batch.batch_size, action_horizon, action_dim)
        or not torch.is_floating_point(initial_noise)
        or not bool(torch.isfinite(initial_noise).all())
    ):
        raise ValueError(
            "initial_noise must be finite floating "
            f"[{batch.batch_size}, {action_horizon}, {action_dim}]"
        )

    modes = _validated_design(batch, trajectory_mode_ids)
    latents = _episode_latents(
        batch,
        episode_start_latents,
        latent_dim=latent_dim,
    )
    device, dtype = _module_device_dtype(policy, batch.current_states)
    feature_mask = effective_action_mask.to(device=device)
    effective_action_dimensions = int(feature_mask.sum())
    error_rows: list[torch.Tensor] = []

    policy_training = policy.training
    policy.eval()
    try:
        with torch.inference_mode():
            for start in range(0, batch.batch_size, sample_batch_size):
                stop = min(start + sample_batch_size, batch.batch_size)
                current_states = batch.current_states[start:stop].to(
                    device=device,
                    dtype=dtype,
                )
                target_actions = batch.action_chunks[start:stop].to(
                    device=device,
                    dtype=dtype,
                )
                action_mask = batch.action_mask[start:stop].to(device=device)
                latent_slice = latents[start:stop].to(device=device, dtype=dtype)
                noise_slice = initial_noise[start:stop].to(
                    device=device,
                    dtype=dtype,
                )
                prediction = policy.sample(
                    current_states,
                    success_latent=latent_slice,
                    steps=flow_sample_steps,
                    action_mask=action_mask,
                    action_feature_mask=feature_mask,
                    initial_noise=noise_slice,
                )
                expected_shape = (stop - start, action_horizon, action_dim)
                if (
                    not isinstance(prediction, torch.Tensor)
                    or tuple(prediction.shape) != expected_shape
                    or prediction.device != device
                    or prediction.dtype != dtype
                    or not bool(torch.isfinite(prediction).all())
                ):
                    raise ValueError(
                        "policy sample must be finite and have shape "
                        f"{expected_shape} on the policy device/dtype"
                    )
                predicted_first = prediction[:, 0, feature_mask]
                target_first = target_actions[:, 0, feature_mask]
                squared_error_sum = (
                    (
                        predicted_first.detach().to(device="cpu", dtype=torch.float64)
                        - target_first.detach().to(device="cpu", dtype=torch.float64)
                    )
                    .square()
                    .sum(dim=1)
                )
                error_rows.append(squared_error_sum)
    finally:
        policy.train(policy_training)

    per_sample_error = torch.cat(error_rows)
    all_indices = tuple(range(batch.batch_size))
    start_zero_indices = tuple(
        index for index, start in enumerate(batch.starts) if start == 0
    )
    nonzero_indices = tuple(
        index for index, start in enumerate(batch.starts) if start > 0
    )
    per_mode = tuple(
        ModeFirstActionMSE(
            mode_id=mode_id,
            metrics=_metric(
                per_sample_error,
                tuple(index for index, mode in enumerate(modes) if mode == mode_id),
                effective_action_dimensions=effective_action_dimensions,
            ),
        )
        for mode_id in sorted(set(modes))
    )
    bin_indices = tuple(start // start_bin_width for start in batch.starts)
    per_start_bin = tuple(
        StartBinFirstActionMSE(
            start_inclusive=bin_index * start_bin_width,
            start_exclusive=(bin_index + 1) * start_bin_width,
            metrics=_metric(
                per_sample_error,
                tuple(
                    index
                    for index, value in enumerate(bin_indices)
                    if value == bin_index
                ),
                effective_action_dimensions=effective_action_dimensions,
            ),
        )
        for bin_index in sorted(set(bin_indices))
    )
    return TemporalControlMetrics(
        overall=_metric(
            per_sample_error,
            all_indices,
            effective_action_dimensions=effective_action_dimensions,
        ),
        start_zero=_metric(
            per_sample_error,
            start_zero_indices,
            effective_action_dimensions=effective_action_dimensions,
        ),
        nonzero_start=_metric(
            per_sample_error,
            nonzero_indices,
            effective_action_dimensions=effective_action_dimensions,
        ),
        per_mode=per_mode,
        per_start_bin=per_start_bin,
        flow_sample_steps=flow_sample_steps,
        start_bin_width=start_bin_width,
        sample_batch_size=sample_batch_size,
    )


__all__ = [
    "FirstActionMSE",
    "ModeFirstActionMSE",
    "StartBinFirstActionMSE",
    "TemporalControlMetrics",
    "evaluate_fixed_start_latent_first_action_mse",
]
