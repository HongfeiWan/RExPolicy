"""Stage 0 manifold objectives without VLA or image dependencies."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

import torch
import torch.nn.functional as functional

from rexpolicy.stage0.models.future_encoder import (
    FutureTrajectoryEncoding,
    masked_reconstruction_loss,
)


@dataclass(frozen=True)
class DiversityLosses:
    variance: torch.Tensor
    covariance: torch.Tensor


@dataclass(frozen=True)
class ManifoldLosses:
    reconstruction: torch.Tensor
    contrastive: torch.Tensor
    mode_alignment: torch.Tensor
    variance: torch.Tensor
    covariance: torch.Tensor
    total: torch.Tensor


def _metadata(
    value: Sequence[Any],
    *,
    expected: int,
    name: str,
) -> tuple[Any, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{name} must be a sequence")
    result = tuple(value)
    if len(result) != expected:
        raise ValueError(f"{name} length must match the embedding batch")
    return result


def _mode_metadata(
    value: Sequence[Any],
    *,
    expected: int,
) -> tuple[Any, ...]:
    result = _metadata(value, expected=expected, name="mode_ids")
    for item in result:
        if item is None or (isinstance(item, str) and not item):
            raise ValueError("mode_ids must contain non-empty labels")
        try:
            hash(item)
        except TypeError as exc:
            raise ValueError("mode_ids must contain hashable labels") from exc
    return result


def _validate_trajectory_modes(
    trajectory_ids: tuple[Any, ...],
    mode_ids: tuple[Any, ...],
) -> None:
    trajectory_modes: dict[Any, Any] = {}
    for trajectory_id, mode_id in zip(trajectory_ids, mode_ids, strict=True):
        try:
            previous = trajectory_modes.setdefault(trajectory_id, mode_id)
        except TypeError as exc:
            raise ValueError("trajectory_ids must contain hashable labels") from exc
        if previous != mode_id:
            raise ValueError("each trajectory_id must map to exactly one mode_id")


def _gather_for_contrastive(
    embeddings: torch.Tensor,
    trajectory_ids: tuple[Any, ...],
    reset_group_ids: tuple[Any, ...],
    starts: tuple[int, ...],
    mode_ids: tuple[Any, ...] | None,
    *,
    gather_distributed: bool,
    group: Any,
) -> tuple[
    torch.Tensor,
    tuple[Any, ...],
    tuple[Any, ...],
    tuple[int, ...],
    tuple[Any, ...] | None,
]:
    if not gather_distributed:
        return embeddings, trajectory_ids, reset_group_ids, starts, mode_ids
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        raise RuntimeError(
            "distributed contrastive gather requires an initialized group"
        )
    from torch.distributed.nn.functional import all_gather

    gathered_embeddings = torch.cat(all_gather(embeddings, group=group), dim=0)
    world_size = torch.distributed.get_world_size(group=group)
    metadata: list[Any] = [None for _ in range(world_size)]
    torch.distributed.all_gather_object(
        metadata,
        (trajectory_ids, reset_group_ids, starts, mode_ids),
        group=group,
    )
    trajectories = tuple(value for item in metadata for value in item[0])
    reset_groups = tuple(value for item in metadata for value in item[1])
    offsets = tuple(int(value) for item in metadata for value in item[2])
    gathered_mode_metadata = tuple(item[3] for item in metadata)
    has_modes = tuple(item is not None for item in gathered_mode_metadata)
    if any(has_modes) and not all(has_modes):
        raise RuntimeError(
            "distributed contrastive ranks must agree on whether mode_ids are supplied"
        )
    modes = (
        tuple(value for item in gathered_mode_metadata for value in item)
        if all(has_modes)
        else None
    )
    gathered_count = int(gathered_embeddings.shape[0])
    if not (len(trajectories) == len(reset_groups) == len(offsets) == gathered_count):
        raise RuntimeError("distributed contrastive batch metadata does not align")
    if modes is not None and len(modes) != gathered_count:
        raise RuntimeError("distributed mode metadata does not align")
    return gathered_embeddings, trajectories, reset_groups, offsets, modes


def _pairwise_equal(
    values: tuple[Any, ...],
    *,
    device: torch.device,
) -> torch.Tensor:
    count = len(values)
    try:
        entries = [left == right for left in values for right in values]
        if any(type(entry) is not bool for entry in entries):
            raise TypeError
    except (TypeError, ValueError) as exc:
        raise ValueError("metadata labels must support scalar equality") from exc
    return torch.tensor(entries, dtype=torch.bool, device=device).reshape(
        count,
        count,
    )


def temporal_group_contrastive_loss(
    embeddings: torch.Tensor,
    trajectory_ids: Sequence[Any],
    reset_group_ids: Sequence[Any],
    starts: Sequence[int],
    *,
    temporal_radius: int,
    temperature: float,
    mode_ids: Sequence[Any] | None = None,
    gather_distributed: bool = False,
    group: Any = None,
) -> torch.Tensor:
    """Multi-positive InfoNCE with explicit temporal and group supervision.

    Without ``mode_ids``, a positive is a nearby window from the same
    trajectory and a negative is a different successful trajectory from the
    exact same reset-state group. Cross-reset windows are excluded, preserving
    the original unsupervised behavior.

    With ``mode_ids``, nearby windows from the same mode and a different reset
    are additional positives, while every different-mode row is a negative.
    Each trajectory must map to exactly one mode, and each anchor must have a
    cross-reset same-mode positive and a different-mode negative.
    """
    if (
        not isinstance(embeddings, torch.Tensor)
        or embeddings.ndim != 2
        or embeddings.shape[0] < 1
        or embeddings.shape[1] < 1
        or not torch.is_floating_point(embeddings)
        or not bool(torch.isfinite(embeddings).all())
    ):
        raise ValueError("embeddings must be finite floating [batch, features]")
    if type(temporal_radius) is not int or temporal_radius < 1:
        raise ValueError("temporal_radius must be a positive integer")
    if (
        isinstance(temperature, bool)
        or not isinstance(temperature, (int, float))
        or not math.isfinite(float(temperature))
        or float(temperature) <= 0.0
    ):
        raise ValueError("temperature must be finite and positive")
    count = int(embeddings.shape[0])
    trajectories = _metadata(
        trajectory_ids,
        expected=count,
        name="trajectory_ids",
    )
    reset_groups = _metadata(
        reset_group_ids,
        expected=count,
        name="reset_group_ids",
    )
    offsets_raw = _metadata(starts, expected=count, name="starts")
    if any(type(value) is not int or value < 0 for value in offsets_raw):
        raise ValueError("starts must contain non-negative integers")
    offsets = tuple(int(value) for value in offsets_raw)
    modes = None if mode_ids is None else _mode_metadata(mode_ids, expected=count)
    if modes is not None:
        _validate_trajectory_modes(trajectories, modes)
    embeddings, trajectories, reset_groups, offsets, modes = _gather_for_contrastive(
        embeddings,
        trajectories,
        reset_groups,
        offsets,
        modes,
        gather_distributed=gather_distributed,
        group=group,
    )
    count = int(embeddings.shape[0])
    if modes is not None:
        _validate_trajectory_modes(trajectories, modes)
    diagonal = torch.eye(count, dtype=torch.bool, device=embeddings.device)
    same_trajectory = _pairwise_equal(trajectories, device=embeddings.device)
    same_reset_group = _pairwise_equal(reset_groups, device=embeddings.device)
    offset_tensor = torch.tensor(offsets, device=embeddings.device)
    nearby = (offset_tensor[:, None] - offset_tensor[None, :]).abs() <= temporal_radius
    trajectory_positive = same_trajectory & nearby & ~diagonal
    if modes is None:
        positive = trajectory_positive
        same_reset_cross_trajectory = same_reset_group & ~same_trajectory
        candidates = positive | same_reset_cross_trajectory
        negative = candidates & ~positive
    else:
        same_mode = _pairwise_equal(modes, device=embeddings.device)
        cross_reset_mode_positive = same_mode & ~same_reset_group & nearby & ~diagonal
        positive = trajectory_positive | cross_reset_mode_positive
        negative = ~same_mode & ~diagonal
        candidates = positive | negative
        if not bool(cross_reset_mode_positive.any(dim=1).all()):
            raise ValueError(
                "every mode-supervised anchor needs a temporally-near "
                "same-mode cross-reset positive"
            )
    if not bool(positive.any(dim=1).all()):
        raise ValueError(
            "every contrastive anchor needs a same-trajectory temporal positive"
        )
    if not bool(negative.any(dim=1).all()):
        if modes is None:
            raise ValueError(
                "every contrastive anchor needs a different-trajectory "
                "same-reset negative"
            )
        raise ValueError("every mode-supervised anchor needs a different-mode negative")

    normalized = functional.normalize(embeddings, dim=-1)
    logits = normalized @ normalized.transpose(0, 1)
    logits = logits / float(temperature)
    minimum = torch.finfo(logits.dtype).min
    denominator = torch.logsumexp(logits.masked_fill(~candidates, minimum), dim=1)
    numerator = torch.logsumexp(logits.masked_fill(~positive, minimum), dim=1)
    return (denominator - numerator).mean()


def mode_invariant_alignment_loss(
    latents: torch.Tensor,
    trajectory_ids: Sequence[Any],
    reset_group_ids: Sequence[Any],
    starts: Sequence[int],
    mode_ids: Sequence[Any],
    *,
    temporal_radius: int,
    gather_distributed: bool = False,
    group: Any = None,
) -> torch.Tensor:
    """Align every raw latent with its same-mode peer from another reset.

    Unlike projection-space contrastive learning, this direct squared-distance
    penalty constrains every latent coordinate, so a projection head cannot
    hide reset identity in an unused subspace.
    """
    if (
        not isinstance(latents, torch.Tensor)
        or latents.ndim != 2
        or latents.shape[0] < 1
        or latents.shape[1] < 1
        or not torch.is_floating_point(latents)
        or not bool(torch.isfinite(latents).all())
    ):
        raise ValueError("latents must be finite floating [batch, features]")
    if type(temporal_radius) is not int or temporal_radius < 1:
        raise ValueError("temporal_radius must be a positive integer")
    count = int(latents.shape[0])
    trajectories = _metadata(
        trajectory_ids,
        expected=count,
        name="trajectory_ids",
    )
    reset_groups = _metadata(
        reset_group_ids,
        expected=count,
        name="reset_group_ids",
    )
    offsets_raw = _metadata(starts, expected=count, name="starts")
    if any(type(value) is not int or value < 0 for value in offsets_raw):
        raise ValueError("starts must contain non-negative integers")
    offsets = tuple(int(value) for value in offsets_raw)
    modes = _mode_metadata(mode_ids, expected=count)
    _validate_trajectory_modes(trajectories, modes)
    latents, trajectories, reset_groups, offsets, gathered_modes = (
        _gather_for_contrastive(
            latents,
            trajectories,
            reset_groups,
            offsets,
            modes,
            gather_distributed=gather_distributed,
            group=group,
        )
    )
    if gathered_modes is None:
        raise RuntimeError("distributed mode alignment lost mode metadata")
    _validate_trajectory_modes(trajectories, gathered_modes)
    same_mode = _pairwise_equal(gathered_modes, device=latents.device)
    same_reset = _pairwise_equal(reset_groups, device=latents.device)
    offset_tensor = torch.tensor(offsets, device=latents.device)
    nearby = (offset_tensor[:, None] - offset_tensor[None, :]).abs() <= temporal_radius
    diagonal = torch.eye(
        latents.shape[0],
        dtype=torch.bool,
        device=latents.device,
    )
    positive = same_mode & ~same_reset & nearby & ~diagonal
    if not bool(positive.any(dim=1).all()):
        raise ValueError(
            "every mode-supervised anchor needs a temporally-near "
            "same-mode cross-reset latent peer"
        )
    squared_distance = (latents[:, None, :] - latents[None, :, :]).square().mean(dim=-1)
    return squared_distance[positive].mean()


def variance_covariance_losses(
    latents: torch.Tensor,
    *,
    target_std: float = 1.0,
    epsilon: float = 1.0e-4,
) -> DiversityLosses:
    """VICReg-style variance floor and off-diagonal covariance penalty."""
    if (
        not isinstance(latents, torch.Tensor)
        or latents.ndim != 2
        or latents.shape[0] < 1
        or latents.shape[1] < 1
        or not torch.is_floating_point(latents)
        or not bool(torch.isfinite(latents).all())
    ):
        raise ValueError("latents must be finite floating [batch, features]")
    if not math.isfinite(float(target_std)) or float(target_std) <= 0.0:
        raise ValueError("target_std must be finite and positive")
    if not math.isfinite(float(epsilon)) or float(epsilon) <= 0.0:
        raise ValueError("epsilon must be finite and positive")
    centered = latents - latents.mean(dim=0, keepdim=True)
    standard_deviation = torch.sqrt(centered.square().mean(dim=0) + float(epsilon))
    variance = functional.relu(float(target_std) - standard_deviation).mean()
    if latents.shape[0] < 2:
        covariance = latents.sum() * 0.0
    else:
        matrix = centered.transpose(0, 1) @ centered
        matrix = matrix / (latents.shape[0] - 1)
        diagonal = torch.diagonal(matrix)
        covariance = (matrix.square().sum() - diagonal.square().sum()) / latents.shape[
            1
        ]
    return DiversityLosses(variance=variance, covariance=covariance)


def manifold_losses(
    encoding: FutureTrajectoryEncoding,
    future_states: torch.Tensor,
    future_actions: torch.Tensor,
    trajectory_ids: Sequence[Any],
    reset_group_ids: Sequence[Any],
    starts: Sequence[int],
    *,
    mode_ids: Sequence[Any] | None = None,
    reconstruction_weight: float,
    contrastive_weight: float,
    mode_alignment_weight: float = 1.0,
    variance_weight: float,
    covariance_weight: float,
    temporal_radius: int,
    contrastive_temperature: float,
    variance_target_std: float,
    variance_epsilon: float,
    gather_distributed: bool,
    group: Any = None,
) -> ManifoldLosses:
    reconstruction = masked_reconstruction_loss(
        encoding,
        future_states,
        future_actions,
    )
    contrastive = temporal_group_contrastive_loss(
        encoding.projection,
        trajectory_ids,
        reset_group_ids,
        starts,
        temporal_radius=temporal_radius,
        temperature=contrastive_temperature,
        mode_ids=mode_ids,
        gather_distributed=gather_distributed,
        group=group,
    )
    if mode_ids is None:
        mode_alignment = encoding.latent.sum() * 0.0
    else:
        mode_alignment = mode_invariant_alignment_loss(
            encoding.latent,
            trajectory_ids,
            reset_group_ids,
            starts,
            mode_ids,
            temporal_radius=temporal_radius,
            gather_distributed=gather_distributed,
            group=group,
        )
    diversity = variance_covariance_losses(
        encoding.latent,
        target_std=variance_target_std,
        epsilon=variance_epsilon,
    )
    total = (
        reconstruction_weight * reconstruction
        + contrastive_weight * contrastive
        + variance_weight * diversity.variance
        + covariance_weight * diversity.covariance
    )
    if mode_ids is not None:
        total = total + mode_alignment_weight * mode_alignment
    return ManifoldLosses(
        reconstruction=reconstruction,
        contrastive=contrastive,
        mode_alignment=mode_alignment,
        variance=diversity.variance,
        covariance=diversity.covariance,
        total=total,
    )
