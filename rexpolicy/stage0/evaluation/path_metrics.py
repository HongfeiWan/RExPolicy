"""Deterministic same-reset path-adherence diagnostics for Stage 0 rollouts.

The classifier in this module is deliberately geometric and non-learned.  An
episode path is compared only with the configured reference paths from its own
reset group.  Both polylines are resampled at uniform normalized arc-length
progress before their pointwise RMS distance is measured, which makes the
diagnostic insensitive to control frequency and most speed-profile changes.

Failed rollouts and successful rollouts without a path are explicit unscored
predictions.  They therefore contribute false negatives to the primary macro
F1 and adherence rate instead of silently improving a conditional score.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch

UNSCORED_PREDICTION = "__unscored__"


def _text_sequence(values: Sequence[str], *, name: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError(f"{name} must be a sequence of strings")
    result = tuple(values)
    for index, value in enumerate(result):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name}[{index}] must be non-empty text")
    return result


def _mode_vocabulary(mode_ids: Sequence[str]) -> tuple[str, ...]:
    result = _text_sequence(mode_ids, name="mode_ids")
    if len(result) < 2:
        raise ValueError("mode_ids must contain at least two modes")
    if len(set(result)) != len(result):
        raise ValueError("mode_ids must be unique")
    if UNSCORED_PREDICTION in result:
        raise ValueError(f"mode_ids cannot contain reserved id {UNSCORED_PREDICTION!r}")
    return result


def _success_flags(
    successful: Sequence[bool] | torch.Tensor,
    *,
    episode_count: int,
) -> tuple[bool, ...]:
    if isinstance(successful, torch.Tensor):
        if successful.dtype is not torch.bool or successful.ndim != 1:
            raise ValueError("successful must be a rank-1 boolean tensor")
        if successful.shape[0] != episode_count:
            raise ValueError("successful must contain one value per episode")
        return tuple(bool(value) for value in successful.detach().cpu().tolist())
    if isinstance(successful, (str, bytes)) or not isinstance(successful, Sequence):
        raise TypeError("successful must be a sequence of booleans")
    result = tuple(successful)
    if len(result) != episode_count:
        raise ValueError("successful must contain one value per episode")
    if any(type(value) is not bool for value in result):
        raise ValueError("successful must contain only booleans")
    return result


def _path_sequence(
    values: Sequence[torch.Tensor | None],
) -> tuple[torch.Tensor | None, ...]:
    if isinstance(values, (str, bytes, torch.Tensor)) or not isinstance(
        values, Sequence
    ):
        raise TypeError("learned_paths must be a sequence of tensors or None")
    result = tuple(values)
    if not result:
        raise ValueError("learned_paths must contain at least one episode")
    return result


def _polyline(value: torch.Tensor, *, name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a Torch tensor")
    if value.ndim != 2 or value.shape[1] != 3 or value.shape[0] < 1:
        raise ValueError(f"{name} must have shape [points >= 1, 3]")
    if not torch.is_floating_point(value):
        raise ValueError(f"{name} must be floating point")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must contain only finite values")
    return value.detach().to(device="cpu", dtype=torch.float64).contiguous()


def _resample_polyline(path: torch.Tensor, *, points: int) -> torch.Tensor:
    if path.shape[0] == 1:
        return path.expand(points, -1).clone()

    segment_lengths = torch.linalg.vector_norm(path[1:] - path[:-1], dim=1)
    if not bool(torch.isfinite(segment_lengths).all()):
        raise FloatingPointError("polyline arc lengths overflowed to non-finite values")
    cumulative = torch.cat(
        (torch.zeros(1, dtype=torch.float64), segment_lengths.cumsum(dim=0))
    )
    total_length = float(cumulative[-1])
    if not math.isfinite(total_length):
        raise FloatingPointError("polyline total arc length is non-finite")
    if total_length <= torch.finfo(torch.float64).eps:
        return path[:1].expand(points, -1).clone()

    progress = torch.linspace(0.0, total_length, points, dtype=torch.float64)
    right = torch.searchsorted(cumulative, progress, right=True)
    right.clamp_(min=1, max=path.shape[0] - 1)
    left = right - 1
    lower_progress = cumulative[left]
    upper_progress = cumulative[right]
    width = upper_progress - lower_progress
    interpolation = torch.where(
        width > torch.finfo(torch.float64).eps,
        (progress - lower_progress) / width.clamp_min(torch.finfo(torch.float64).eps),
        torch.zeros_like(progress),
    )
    result = path[left] + interpolation[:, None] * (path[right] - path[left])
    if not bool(torch.isfinite(result).all()):
        raise FloatingPointError("resampled polyline is non-finite")
    return result


def _rms_path_distance(first: torch.Tensor, second: torch.Tensor) -> float:
    distance = torch.sqrt((first - second).square().sum(dim=1).mean())
    result = float(distance)
    if not math.isfinite(result):
        raise FloatingPointError("path RMS distance is non-finite")
    return result


def _ratio(numerator: int, denominator: int) -> float:
    return 0.0 if denominator == 0 else numerator / denominator


def _mean(values: Sequence[float]) -> float | None:
    return None if not values else math.fsum(values) / len(values)


def _class_f1(
    expected: tuple[str, ...],
    predicted: tuple[str | None, ...],
    mode_id: str,
    *,
    selected: tuple[bool, ...] | None = None,
) -> tuple[int, int, int, float, float, float]:
    if selected is None:
        selected = (True,) * len(expected)
    true_positive = false_positive = false_negative = 0
    for target, prediction, include in zip(expected, predicted, selected):
        if not include:
            continue
        if target == mode_id and prediction == mode_id:
            true_positive += 1
        elif target != mode_id and prediction == mode_id:
            false_positive += 1
        elif target == mode_id and prediction != mode_id:
            false_negative += 1
    precision = _ratio(true_positive, true_positive + false_positive)
    recall = _ratio(true_positive, true_positive + false_negative)
    f1 = _ratio(
        2 * true_positive,
        2 * true_positive + false_positive + false_negative,
    )
    return true_positive, false_positive, false_negative, precision, recall, f1


@dataclass(frozen=True)
class ModePathAdherence:
    """One expected mode's confusion row and adherence measurements."""

    mode_id: str
    support: int
    successful: int
    scored: int
    correct: int
    missing_success_path: int
    predicted_mode_counts: tuple[tuple[str, int], ...]
    unscored: int
    precision: float
    recall: float
    f1: float
    adherence_rate: float
    conditional_adherence_rate: float
    mean_nearest_reference_rms_m: float | None
    mean_expected_reference_rms_m: float | None

    def to_record(self) -> dict[str, Any]:
        return {
            "adherence_rate": self.adherence_rate,
            "conditional_adherence_rate": self.conditional_adherence_rate,
            "correct": self.correct,
            "f1": self.f1,
            "mean_expected_reference_rms_m": (
                self.mean_expected_reference_rms_m
            ),
            "mean_nearest_reference_rms_m": self.mean_nearest_reference_rms_m,
            "missing_success_path": self.missing_success_path,
            "mode_id": self.mode_id,
            "precision": self.precision,
            "predicted_counts": {
                **dict(self.predicted_mode_counts),
                UNSCORED_PREDICTION: self.unscored,
            },
            "recall": self.recall,
            "scored": self.scored,
            "successful": self.successful,
            "support": self.support,
        }


@dataclass(frozen=True)
class PathAdherenceMetrics:
    """JSON-ready result of same-reset nearest-reference path matching."""

    mode_ids: tuple[str, ...]
    reset_group_ids: tuple[str, ...]
    expected_mode_ids: tuple[str, ...]
    successful: tuple[bool, ...]
    supplied_path: tuple[bool, ...]
    statuses: tuple[str, ...]
    predicted_mode_ids: tuple[str | None, ...]
    reference_rms_m: tuple[tuple[float, ...] | None, ...]
    nearest_reference_rms_m: tuple[float | None, ...]
    expected_reference_rms_m: tuple[float | None, ...]
    per_mode: tuple[ModePathAdherence, ...]
    resample_points: int
    successful_count: int
    scored_count: int
    correct_count: int
    missing_success_path_count: int
    macro_f1: float
    conditional_macro_f1: float
    adherence_rate: float
    conditional_adherence_rate: float
    mean_nearest_reference_rms_m: float | None
    mean_expected_reference_rms_m: float | None

    @property
    def episode_count(self) -> int:
        return len(self.expected_mode_ids)

    @property
    def success_rate(self) -> float:
        return self.successful_count / self.episode_count

    @property
    def scoring_rate(self) -> float:
        return self.scored_count / self.episode_count

    def to_record(self) -> dict[str, Any]:
        """Return a finite standards-compliant JSON diagnostic record."""

        episodes: list[dict[str, Any]] = []
        for index in range(self.episode_count):
            distances = self.reference_rms_m[index]
            prediction = self.predicted_mode_ids[index]
            episodes.append(
                {
                    "adheres": prediction == self.expected_mode_ids[index],
                    "episode_index": index,
                    "expected_mode_id": self.expected_mode_ids[index],
                    "expected_reference_rms_m": (
                        self.expected_reference_rms_m[index]
                    ),
                    "nearest_reference_rms_m": (
                        self.nearest_reference_rms_m[index]
                    ),
                    "predicted_mode_id": prediction,
                    "reference_rms_m": (
                        None
                        if distances is None
                        else dict(zip(self.mode_ids, distances))
                    ),
                    "reset_group_id": self.reset_group_ids[index],
                    "status": self.statuses[index],
                    "successful": self.successful[index],
                    "supplied_path": self.supplied_path[index],
                }
            )
        return {
            "adherence_rate": self.adherence_rate,
            "authority": "diagnostic_only",
            "conditional_adherence_rate": self.conditional_adherence_rate,
            "conditional_macro_f1": self.conditional_macro_f1,
            "confusion": {
                "column_labels": [*self.mode_ids, UNSCORED_PREDICTION],
                "matrix": [
                    [
                        *dict(metrics.predicted_mode_counts).values(),
                        metrics.unscored,
                    ]
                    for metrics in self.per_mode
                ],
                "row_labels": list(self.mode_ids),
            },
            "correct_count": self.correct_count,
            "episode_count": self.episode_count,
            "episodes": episodes,
            "expected_mode_semantics": "caller_supplied_episode_target",
            "macro_f1": self.macro_f1,
            "mean_expected_reference_rms_m": self.mean_expected_reference_rms_m,
            "mean_nearest_reference_rms_m": self.mean_nearest_reference_rms_m,
            "missing_success_path_count": self.missing_success_path_count,
            "mode_ids": list(self.mode_ids),
            "per_mode": {
                metrics.mode_id: metrics.to_record() for metrics in self.per_mode
            },
            "protocol": {
                "distance": "pointwise_euclidean_rms_m",
                "matching": "nearest_reference_within_same_reset",
                "progress": "uniform_normalized_arc_length",
                "resample_points": self.resample_points,
                "tie_break": "configured_mode_order",
                "unscored_policy": "count_as_false_negative",
            },
            "schema_version": 1,
            "scored_count": self.scored_count,
            "scoring_rate": self.scoring_rate,
            "success_rate": self.success_rate,
            "successful_count": self.successful_count,
            "supplied_path_count": sum(self.supplied_path),
            "unsuccessful_count": self.episode_count - self.successful_count,
        }


def analyze_path_adherence(
    learned_paths: Sequence[torch.Tensor | None],
    *,
    successful: Sequence[bool] | torch.Tensor,
    reset_group_ids: Sequence[str],
    expected_mode_ids: Sequence[str],
    reference_paths: Mapping[str, Mapping[str, torch.Tensor]],
    mode_ids: Sequence[str],
    resample_points: int = 64,
) -> PathAdherenceMetrics:
    """Classify rollout paths against same-reset references and score adherence.

    ``expected_mode_ids`` defines the causal target for every episode.  For a
    permuted-z control, callers should pass the mode IDs belonging to the
    permuted latent sources, not the pre-permutation episode ordering.

    Unsuccessful episodes are never classified even if they contain a partial
    path.  Successful episodes with ``None`` paths are recorded separately as
    ``missing_success_path``.  Both cases are false negatives in ``macro_f1``;
    ``conditional_macro_f1`` reports geometry only over scored paths.
    """

    paths = _path_sequence(learned_paths)
    episode_count = len(paths)
    modes = _mode_vocabulary(mode_ids)
    resets = _text_sequence(reset_group_ids, name="reset_group_ids")
    expected = _text_sequence(expected_mode_ids, name="expected_mode_ids")
    if len(resets) != episode_count:
        raise ValueError("reset_group_ids must contain one value per episode")
    if len(expected) != episode_count:
        raise ValueError("expected_mode_ids must contain one value per episode")
    unknown_expected = sorted(set(expected).difference(modes))
    if unknown_expected:
        raise ValueError(
            "expected_mode_ids contain unconfigured modes: "
            + ", ".join(repr(value) for value in unknown_expected)
        )
    flags = _success_flags(successful, episode_count=episode_count)
    if type(resample_points) is not int or resample_points < 2:
        raise ValueError("resample_points must be an integer of at least two")
    if not isinstance(reference_paths, Mapping):
        raise TypeError("reference_paths must map reset IDs to mode-path mappings")

    validated_paths: list[torch.Tensor | None] = []
    supplied_path: list[bool] = []
    for index, path in enumerate(paths):
        supplied_path.append(path is not None)
        validated_paths.append(
            None if path is None else _polyline(path, name=f"learned_paths[{index}]")
        )

    reference_cache: dict[tuple[str, str], torch.Tensor] = {}
    for reset_id in dict.fromkeys(resets):
        if reset_id not in reference_paths:
            raise ValueError(f"reference_paths is missing reset group {reset_id!r}")
        by_mode = reference_paths[reset_id]
        if not isinstance(by_mode, Mapping):
            raise TypeError(
                f"reference_paths[{reset_id!r}] must map mode IDs to paths"
            )
        for mode_id in modes:
            if mode_id not in by_mode:
                raise ValueError(
                    f"reference_paths[{reset_id!r}] is missing mode {mode_id!r}"
                )
            reference = _polyline(
                by_mode[mode_id],
                name=f"reference_paths[{reset_id!r}][{mode_id!r}]",
            )
            reference_cache[(reset_id, mode_id)] = _resample_polyline(
                reference,
                points=resample_points,
            )

    statuses: list[str] = []
    predictions: list[str | None] = []
    distance_rows: list[tuple[float, ...] | None] = []
    nearest_distances: list[float | None] = []
    expected_distances: list[float | None] = []
    for path, is_success, reset_id, expected_mode in zip(
        validated_paths, flags, resets, expected
    ):
        if not is_success:
            statuses.append("unsuccessful")
            predictions.append(None)
            distance_rows.append(None)
            nearest_distances.append(None)
            expected_distances.append(None)
            continue
        if path is None:
            statuses.append("missing_success_path")
            predictions.append(None)
            distance_rows.append(None)
            nearest_distances.append(None)
            expected_distances.append(None)
            continue

        learned = _resample_polyline(path, points=resample_points)
        distances = tuple(
            _rms_path_distance(learned, reference_cache[(reset_id, mode_id)])
            for mode_id in modes
        )
        prediction_index = min(range(len(modes)), key=distances.__getitem__)
        expected_index = modes.index(expected_mode)
        statuses.append("scored")
        predictions.append(modes[prediction_index])
        distance_rows.append(distances)
        nearest_distances.append(distances[prediction_index])
        expected_distances.append(distances[expected_index])

    prediction_tuple = tuple(predictions)
    scored_mask = tuple(status == "scored" for status in statuses)
    per_mode: list[ModePathAdherence] = []
    primary_f1: list[float] = []
    conditional_f1: list[float] = []
    for mode_id in modes:
        target_indices = [
            index for index, target in enumerate(expected) if target == mode_id
        ]
        support = len(target_indices)
        successful_count = sum(flags[index] for index in target_indices)
        scored_count = sum(scored_mask[index] for index in target_indices)
        correct_count = sum(
            prediction_tuple[index] == mode_id for index in target_indices
        )
        missing_count = sum(
            statuses[index] == "missing_success_path" for index in target_indices
        )
        predicted_counts = tuple(
            (
                predicted_mode,
                sum(
                    prediction_tuple[index] == predicted_mode
                    for index in target_indices
                ),
            )
            for predicted_mode in modes
        )
        unscored_count = sum(
            prediction_tuple[index] is None for index in target_indices
        )
        _, _, _, precision, recall, f1 = _class_f1(
            expected,
            prediction_tuple,
            mode_id,
        )
        _, _, _, _, _, scored_f1 = _class_f1(
            expected,
            prediction_tuple,
            mode_id,
            selected=scored_mask,
        )
        primary_f1.append(f1)
        conditional_f1.append(scored_f1)
        mode_nearest = [
            nearest_distances[index]
            for index in target_indices
            if nearest_distances[index] is not None
        ]
        mode_expected = [
            expected_distances[index]
            for index in target_indices
            if expected_distances[index] is not None
        ]
        per_mode.append(
            ModePathAdherence(
                mode_id=mode_id,
                support=support,
                successful=successful_count,
                scored=scored_count,
                correct=correct_count,
                missing_success_path=missing_count,
                predicted_mode_counts=predicted_counts,
                unscored=unscored_count,
                precision=precision,
                recall=recall,
                f1=f1,
                adherence_rate=_ratio(correct_count, support),
                conditional_adherence_rate=_ratio(correct_count, scored_count),
                mean_nearest_reference_rms_m=_mean(mode_nearest),
                mean_expected_reference_rms_m=_mean(mode_expected),
            )
        )

    successful_count = sum(flags)
    scored_count = sum(scored_mask)
    correct_count = sum(
        prediction == target for prediction, target in zip(prediction_tuple, expected)
    )
    finite_nearest = [value for value in nearest_distances if value is not None]
    finite_expected = [value for value in expected_distances if value is not None]
    return PathAdherenceMetrics(
        mode_ids=modes,
        reset_group_ids=resets,
        expected_mode_ids=expected,
        successful=flags,
        supplied_path=tuple(supplied_path),
        statuses=tuple(statuses),
        predicted_mode_ids=prediction_tuple,
        reference_rms_m=tuple(distance_rows),
        nearest_reference_rms_m=tuple(nearest_distances),
        expected_reference_rms_m=tuple(expected_distances),
        per_mode=tuple(per_mode),
        resample_points=resample_points,
        successful_count=successful_count,
        scored_count=scored_count,
        correct_count=correct_count,
        missing_success_path_count=sum(
            status == "missing_success_path" for status in statuses
        ),
        macro_f1=math.fsum(primary_f1) / len(primary_f1),
        conditional_macro_f1=math.fsum(conditional_f1) / len(conditional_f1),
        adherence_rate=correct_count / episode_count,
        conditional_adherence_rate=_ratio(correct_count, scored_count),
        mean_nearest_reference_rms_m=_mean(finite_nearest),
        mean_expected_reference_rms_m=_mean(finite_expected),
    )
