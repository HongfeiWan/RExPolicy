"""Held-out mode and reset-geometry diagnostics for Stage 0 latents.

The metrics in this module are diagnostic only.  They operate in the original
latent space and use reset-group holdouts so repeated windows from the same
reset cannot leak into their own nearest-centroid or linear-probe fit.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch


def _validated_latents(latents: torch.Tensor) -> torch.Tensor:
    if not isinstance(latents, torch.Tensor) or latents.ndim != 2:
        raise ValueError("latents must be a rank-2 tensor [samples, latent_dim]")
    if latents.shape[0] < 4:
        raise ValueError("latents must contain at least four samples")
    if latents.shape[1] < 1:
        raise ValueError("latents must have a non-empty latent dimension")
    if not torch.is_floating_point(latents):
        raise ValueError("latents must be floating point")
    if not bool(torch.isfinite(latents).all()):
        raise ValueError("latents must contain only finite values")
    return latents.detach().to(device="cpu", dtype=torch.float64)


def _validated_ids(
    values: Sequence[str],
    *,
    name: str,
    sample_count: int,
) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError(f"{name} must be a sequence of strings")
    result = tuple(values)
    if len(result) != sample_count:
        raise ValueError(f"{name} must contain one value per latent sample")
    for index, value in enumerate(result):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name}[{index}] must be non-empty text")
    return result


def _encoded_ids(
    values: tuple[str, ...],
) -> tuple[tuple[str, ...], torch.Tensor]:
    vocabulary = tuple(sorted(set(values)))
    lookup = {value: index for index, value in enumerate(vocabulary)}
    encoded = torch.tensor([lookup[value] for value in values], dtype=torch.long)
    return vocabulary, encoded


def _validated_design(
    mode_names: tuple[str, ...],
    mode_codes: torch.Tensor,
    reset_names: tuple[str, ...],
    reset_codes: torch.Tensor,
) -> None:
    if len(mode_names) < 2:
        raise ValueError("mode_ids must contain at least two unique modes")
    if len(reset_names) < 2:
        raise ValueError(
            "reset_group_ids must contain at least two unique reset groups"
        )

    for mode_index, mode_name in enumerate(mode_names):
        mask = mode_codes == mode_index
        if int(mask.sum()) < 2:
            raise ValueError(f"mode {mode_name!r} must contain at least two samples")
        represented_resets = torch.unique(reset_codes[mask]).numel()
        if represented_resets < 2:
            raise ValueError(
                f"mode {mode_name!r} must occur in at least two reset groups"
            )

    upper = torch.triu(
        torch.ones(
            (mode_codes.numel(), mode_codes.numel()),
            dtype=torch.bool,
        ),
        diagonal=1,
    )
    same_reset_between_mode = (
        upper
        & reset_codes[:, None].eq(reset_codes[None, :])
        & mode_codes[:, None].ne(mode_codes[None, :])
    )
    if not bool(same_reset_between_mode.any()):
        raise ValueError(
            "at least one reset group must contain samples from different modes"
        )


def _leave_one_reset_out_centroids(
    latents: torch.Tensor,
    mode_codes: torch.Tensor,
    reset_codes: torch.Tensor,
    *,
    mode_count: int,
    reset_count: int,
) -> torch.Tensor:
    predictions = torch.empty_like(mode_codes)
    for reset_index in range(reset_count):
        test_mask = reset_codes == reset_index
        train_mask = ~test_mask
        centroids: list[torch.Tensor] = []
        for mode_index in range(mode_count):
            examples = latents[train_mask & (mode_codes == mode_index)]
            if examples.shape[0] == 0:
                raise ValueError(
                    "leave-one-reset-out centroid fit has a mode with no "
                    "training samples"
                )
            centroids.append(examples.mean(dim=0))
        centroid_matrix = torch.stack(centroids)
        distances = torch.cdist(latents[test_mask], centroid_matrix, p=2.0)
        if not bool(torch.isfinite(distances).all()):
            raise ValueError("latent distances overflowed to non-finite values")
        # Mode names are sorted before encoding, so argmin also defines a
        # deterministic lexical tie-break.
        predictions[test_mask] = distances.argmin(dim=1)
    return predictions


def _macro_f1(
    targets: torch.Tensor,
    predictions: torch.Tensor,
    *,
    class_count: int,
) -> float:
    scores: list[float] = []
    for class_index in range(class_count):
        target = targets == class_index
        predicted = predictions == class_index
        true_positive = int((target & predicted).sum())
        false_positive = int((~target & predicted).sum())
        false_negative = int((target & ~predicted).sum())
        denominator = 2 * true_positive + false_positive + false_negative
        scores.append(0.0 if denominator == 0 else 2.0 * true_positive / denominator)
    return math.fsum(scores) / class_count


def _normalized_mutual_information(
    targets: torch.Tensor,
    predictions: torch.Tensor,
    *,
    class_count: int,
) -> float:
    confusion = torch.zeros(
        (class_count, class_count),
        dtype=torch.float64,
    )
    for target, prediction in zip(targets.tolist(), predictions.tolist()):
        confusion[target, prediction] += 1.0
    joint = confusion / confusion.sum()
    target_probability = joint.sum(dim=1)
    prediction_probability = joint.sum(dim=0)
    positive = joint > 0.0
    independent = target_probability[:, None] * prediction_probability[None, :]
    mutual_information = (
        joint[positive] * (joint[positive] / independent[positive]).log()
    ).sum()

    target_positive = target_probability > 0.0
    prediction_positive = prediction_probability > 0.0
    target_entropy = -(
        target_probability[target_positive] * target_probability[target_positive].log()
    ).sum()
    prediction_entropy = -(
        prediction_probability[prediction_positive]
        * prediction_probability[prediction_positive].log()
    ).sum()
    denominator = (target_entropy + prediction_entropy) / 2.0
    if float(denominator) <= torch.finfo(torch.float64).eps:
        return 0.0
    # Floating point roundoff can put a theoretically unit score a few ulps
    # outside the closed interval.
    return min(1.0, max(0.0, float(mutual_information / denominator)))


def _silhouette_and_distance_ratio(
    latents: torch.Tensor,
    mode_codes: torch.Tensor,
    reset_codes: torch.Tensor,
    *,
    mode_count: int,
) -> tuple[float, float, float, float]:
    distances = torch.cdist(latents, latents, p=2.0)
    if not bool(torch.isfinite(distances).all()):
        raise ValueError("latent distances overflowed to non-finite values")

    silhouettes: list[torch.Tensor] = []
    for sample_index in range(latents.shape[0]):
        own_mode = mode_codes == mode_codes[sample_index]
        own_mode[sample_index] = False
        within_distance = distances[sample_index, own_mode].mean()
        other_distances = [
            distances[sample_index, mode_codes == mode_index].mean()
            for mode_index in range(mode_count)
            if mode_index != int(mode_codes[sample_index])
        ]
        nearest_other_distance = torch.stack(other_distances).min()
        scale = torch.maximum(within_distance, nearest_other_distance)
        if float(scale) == 0.0:
            silhouettes.append(torch.zeros((), dtype=torch.float64))
        else:
            silhouettes.append((nearest_other_distance - within_distance) / scale)

    sample_count = latents.shape[0]
    upper = torch.triu(
        torch.ones((sample_count, sample_count), dtype=torch.bool),
        diagonal=1,
    )
    same_reset_between_mode = (
        upper
        & reset_codes[:, None].eq(reset_codes[None, :])
        & mode_codes[:, None].ne(mode_codes[None, :])
    )
    same_mode_cross_reset = (
        upper
        & mode_codes[:, None].eq(mode_codes[None, :])
        & reset_codes[:, None].ne(reset_codes[None, :])
    )
    if not bool(same_mode_cross_reset.any()):
        raise ValueError(
            "same-mode samples must span different reset groups for distance comparison"
        )
    between_mode_distance = float(distances[same_reset_between_mode].mean())
    cross_reset_distance = float(distances[same_mode_cross_reset].mean())
    denominator = max(cross_reset_distance, torch.finfo(torch.float64).eps)
    distance_ratio = between_mode_distance / denominator
    if not math.isfinite(distance_ratio):
        raise ValueError("mode-to-reset distance ratio is non-finite")
    return (
        float(torch.stack(silhouettes).mean()),
        between_mode_distance,
        cross_reset_distance,
        distance_ratio,
    )


def _validated_reset_xy(
    reset_xy: torch.Tensor,
    *,
    sample_count: int,
    reset_names: tuple[str, ...],
    reset_codes: torch.Tensor,
) -> torch.Tensor:
    if not isinstance(reset_xy, torch.Tensor) or reset_xy.ndim != 2:
        raise ValueError("reset_xy must be a rank-2 tensor [samples, 2]")
    if tuple(reset_xy.shape) != (sample_count, 2):
        raise ValueError("reset_xy must have shape [samples, 2]")
    if not torch.is_floating_point(reset_xy):
        raise ValueError("reset_xy must be floating point")
    if not bool(torch.isfinite(reset_xy).all()):
        raise ValueError("reset_xy must contain only finite values")
    values = reset_xy.detach().to(device="cpu", dtype=torch.float64)
    if len(reset_names) < 3:
        raise ValueError("reset_xy linear probe requires at least three reset groups")
    for reset_index, reset_name in enumerate(reset_names):
        group_values = values[reset_codes == reset_index]
        reference = group_values[0].expand_as(group_values)
        if not torch.allclose(group_values, reference, rtol=1e-5, atol=1e-7):
            raise ValueError(
                f"reset_xy must be invariant within reset group {reset_name!r}"
            )
    centered = values - values.mean(dim=0, keepdim=True)
    if float(centered.square().sum()) <= torch.finfo(torch.float64).eps:
        raise ValueError("reset_xy must vary across reset groups")
    return values


def _ridge_reset_probe(
    latents: torch.Tensor,
    reset_xy: torch.Tensor,
    reset_codes: torch.Tensor,
    *,
    reset_count: int,
    ridge_alpha: float,
) -> tuple[float, tuple[float | None, float | None]]:
    predictions = torch.empty_like(reset_xy)
    identity = torch.eye(latents.shape[1], dtype=torch.float64)
    for reset_index in range(reset_count):
        test_mask = reset_codes == reset_index
        train_mask = ~test_mask
        train_x = latents[train_mask]
        train_y = reset_xy[train_mask]
        x_mean = train_x.mean(dim=0, keepdim=True)
        y_mean = train_y.mean(dim=0, keepdim=True)
        centered_x = train_x - x_mean
        centered_y = train_y - y_mean
        coefficients = torch.linalg.solve(
            centered_x.mT @ centered_x + ridge_alpha * identity,
            centered_x.mT @ centered_y,
        )
        predictions[test_mask] = (latents[test_mask] - x_mean) @ coefficients + y_mean
    if not bool(torch.isfinite(predictions).all()):
        raise ValueError("reset_xy ridge predictions are non-finite")

    residual_sum_squares = (reset_xy - predictions).square().sum(dim=0)
    centered_targets = reset_xy - reset_xy.mean(dim=0, keepdim=True)
    total_sum_squares = centered_targets.square().sum(dim=0)
    overall = 1.0 - float(residual_sum_squares.sum() / total_sum_squares.sum())
    per_axis: list[float | None] = []
    tolerance = torch.finfo(torch.float64).eps
    for residual, total in zip(residual_sum_squares, total_sum_squares):
        if float(total) <= tolerance:
            per_axis.append(None)
        else:
            per_axis.append(1.0 - float(residual / total))
    if not math.isfinite(overall) or any(
        value is not None and not math.isfinite(value) for value in per_axis
    ):
        raise ValueError("reset_xy ridge R-squared is non-finite")
    return overall, (per_axis[0], per_axis[1])


@dataclass(frozen=True)
class LatentModeMetrics:
    """Deterministic held-out mode diagnostics in the original latent space."""

    sample_count: int
    latent_dim: int
    mode_ids: tuple[str, ...]
    reset_group_count: int
    nearest_centroid_predictions: tuple[str, ...]
    nearest_centroid_macro_f1: float
    nearest_centroid_nmi: float
    euclidean_silhouette: float
    same_reset_between_mode_distance: float
    same_mode_cross_reset_distance: float
    mode_to_reset_distance_ratio: float
    reset_xy_linear_probe_r2: float | None
    reset_xy_linear_probe_r2_by_axis: tuple[float | None, float | None] | None
    ridge_alpha: float

    def to_record(self) -> dict[str, Any]:
        """Return a standards-compliant JSON-ready diagnostic record."""

        return {
            "authority": "diagnostic_only",
            "geometry": {
                "euclidean_silhouette": self.euclidean_silhouette,
                "mode_to_reset_distance_ratio": (self.mode_to_reset_distance_ratio),
                "same_mode_cross_reset_distance": (self.same_mode_cross_reset_distance),
                "same_reset_between_mode_distance": (
                    self.same_reset_between_mode_distance
                ),
            },
            "latent_dim": self.latent_dim,
            "mode_ids": list(self.mode_ids),
            "nearest_centroid": {
                "macro_f1": self.nearest_centroid_macro_f1,
                "normalized_mutual_information": self.nearest_centroid_nmi,
                "predictions": list(self.nearest_centroid_predictions),
                "protocol": "leave_one_reset_out",
            },
            "reset_group_count": self.reset_group_count,
            "reset_geometry_linear_probe": (
                None
                if self.reset_xy_linear_probe_r2 is None
                else {
                    "interpretation": (
                        "diagnostic_only_causal_task_geometry_not_identity_leakage"
                    ),
                    "r2": self.reset_xy_linear_probe_r2,
                    "r2_by_axis": list(self.reset_xy_linear_probe_r2_by_axis or ()),
                    "ridge_alpha": self.ridge_alpha,
                    "protocol": "leave_one_reset_out",
                    "target": "object_position_xy_m",
                }
            ),
            "sample_count": self.sample_count,
            "schema_version": 2,
            "space": "original_latent",
        }


def analyze_latent_modes(
    latents: torch.Tensor,
    mode_ids: Sequence[str],
    reset_group_ids: Sequence[str],
    *,
    reset_xy: torch.Tensor | None = None,
    ridge_alpha: float = 1e-3,
) -> LatentModeMetrics:
    """Measure mode separability and reset leakage on held-out resets.

    ``mode_ids`` and ``reset_group_ids`` are aligned with rows of ``latents``.
    If supplied, ``reset_xy`` must be constant within each reset group.  The
    returned probe score is computed from leave-one-reset-out predictions, not
    from an in-sample regression fit.
    """

    values = _validated_latents(latents)
    sample_count, latent_dim = (int(axis) for axis in values.shape)
    modes = _validated_ids(
        mode_ids,
        name="mode_ids",
        sample_count=sample_count,
    )
    resets = _validated_ids(
        reset_group_ids,
        name="reset_group_ids",
        sample_count=sample_count,
    )
    mode_names, mode_codes = _encoded_ids(modes)
    reset_names, reset_codes = _encoded_ids(resets)
    _validated_design(mode_names, mode_codes, reset_names, reset_codes)

    if (
        isinstance(ridge_alpha, bool)
        or not isinstance(ridge_alpha, (int, float))
        or not math.isfinite(float(ridge_alpha))
        or float(ridge_alpha) <= 0.0
    ):
        raise ValueError("ridge_alpha must be finite and positive")
    ridge_alpha = float(ridge_alpha)

    prediction_codes = _leave_one_reset_out_centroids(
        values,
        mode_codes,
        reset_codes,
        mode_count=len(mode_names),
        reset_count=len(reset_names),
    )
    predictions = tuple(mode_names[index] for index in prediction_codes.tolist())
    macro_f1 = _macro_f1(
        mode_codes,
        prediction_codes,
        class_count=len(mode_names),
    )
    nmi = _normalized_mutual_information(
        mode_codes,
        prediction_codes,
        class_count=len(mode_names),
    )
    (
        silhouette,
        between_mode_distance,
        cross_reset_distance,
        distance_ratio,
    ) = _silhouette_and_distance_ratio(
        values,
        mode_codes,
        reset_codes,
        mode_count=len(mode_names),
    )

    probe_r2: float | None = None
    probe_r2_by_axis: tuple[float | None, float | None] | None = None
    if reset_xy is not None:
        reset_values = _validated_reset_xy(
            reset_xy,
            sample_count=sample_count,
            reset_names=reset_names,
            reset_codes=reset_codes,
        )
        probe_r2, probe_r2_by_axis = _ridge_reset_probe(
            values,
            reset_values,
            reset_codes,
            reset_count=len(reset_names),
            ridge_alpha=ridge_alpha,
        )

    return LatentModeMetrics(
        sample_count=sample_count,
        latent_dim=latent_dim,
        mode_ids=mode_names,
        reset_group_count=len(reset_names),
        nearest_centroid_predictions=predictions,
        nearest_centroid_macro_f1=macro_f1,
        nearest_centroid_nmi=nmi,
        euclidean_silhouette=silhouette,
        same_reset_between_mode_distance=between_mode_distance,
        same_mode_cross_reset_distance=cross_reset_distance,
        mode_to_reset_distance_ratio=distance_ratio,
        reset_xy_linear_probe_r2=probe_r2,
        reset_xy_linear_probe_r2_by_axis=probe_r2_by_axis,
        ridge_alpha=ridge_alpha,
    )
