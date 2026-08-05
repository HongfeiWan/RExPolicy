"""Leakage-safe trajectory-group splitting before window extraction."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Iterable

from rexpolicy.stage0.data.trajectory import (
    Stage0Trajectory,
    trajectory_corpus_sha256,
)
from rexpolicy.stage0.data.windows import (
    Stage0SuccessWindow,
    Stage0WindowPolicy,
    extract_success_windows,
)


@dataclass(frozen=True)
class Stage0SplitPolicy:
    """Deterministic reset-group split policy."""

    train_fraction: float = 0.8
    validation_fraction: float = 0.1
    test_fraction: float = 0.1
    seed: int = 0

    def __post_init__(self) -> None:
        fractions = (
            self.train_fraction,
            self.validation_fraction,
            self.test_fraction,
        )
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) < 0.0
            for value in fractions
        ):
            raise ValueError("split fractions must be finite and non-negative")
        if not math.isclose(sum(fractions), 1.0, rel_tol=0.0, abs_tol=1.0e-12):
            raise ValueError("split fractions must sum to one")
        if self.train_fraction <= 0.0:
            raise ValueError("train_fraction must be positive")
        if (
            isinstance(self.seed, bool)
            or not isinstance(self.seed, int)
            or self.seed < 0
        ):
            raise ValueError("split seed must be a non-negative integer")

    @property
    def sha256(self) -> str:
        record = {
            "schema_id": "rexpolicy/stage0-reset-group-split/v1",
            "seed": self.seed,
            "test_fraction": self.test_fraction,
            "train_fraction": self.train_fraction,
            "validation_fraction": self.validation_fraction,
        }
        encoded = json.dumps(record, separators=(",", ":"), sort_keys=True).encode(
            "utf-8"
        )
        return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class Stage0TrajectorySplits:
    train: tuple[Stage0Trajectory, ...]
    validation: tuple[Stage0Trajectory, ...]
    test: tuple[Stage0Trajectory, ...]
    policy_sha256: str

    def __post_init__(self) -> None:
        trajectory_sets = [
            {item.provenance.trajectory_id for item in split}
            for split in (self.train, self.validation, self.test)
        ]
        group_sets = [
            {item.provenance.reset_group_id for item in split}
            for split in (self.train, self.validation, self.test)
        ]
        for index, left in enumerate(trajectory_sets):
            for right in trajectory_sets[index + 1 :]:
                if left & right:
                    raise ValueError("trajectory leakage across dataset splits")
        for index, left in enumerate(group_sets):
            for right in group_sets[index + 1 :]:
                if left & right:
                    raise ValueError("reset-group leakage across dataset splits")


@dataclass(frozen=True)
class Stage0WindowSplits:
    train: tuple[Stage0SuccessWindow, ...]
    validation: tuple[Stage0SuccessWindow, ...]
    test: tuple[Stage0SuccessWindow, ...]
    corpus_sha256: str
    split_policy_sha256: str
    window_policy_sha256: str


def _allocation_counts(group_count: int, fractions: tuple[float, ...]) -> list[int]:
    raw = [group_count * value for value in fractions]
    counts = [math.floor(value) for value in raw]
    remaining = group_count - sum(counts)
    priorities = sorted(
        range(len(fractions)),
        key=lambda index: (-(raw[index] - counts[index]), index),
    )
    for index in priorities[:remaining]:
        counts[index] += 1
    return counts


def split_trajectories(
    trajectories: Iterable[Stage0Trajectory],
    policy: Stage0SplitPolicy,
) -> Stage0TrajectorySplits:
    """Assign complete reset groups before any overlapping windows exist."""
    if not isinstance(policy, Stage0SplitPolicy):
        raise TypeError("policy must be Stage0SplitPolicy")
    items = tuple(trajectories)
    if not items:
        raise ValueError("cannot split an empty trajectory corpus")
    groups: dict[str, list[Stage0Trajectory]] = {}
    trajectory_ids: set[str] = set()
    for trajectory in items:
        if not isinstance(trajectory, Stage0Trajectory):
            raise TypeError("all corpus entries must be Stage0Trajectory")
        trajectory_id = trajectory.provenance.trajectory_id
        if trajectory_id in trajectory_ids:
            raise ValueError(f"duplicate trajectory_id: {trajectory_id}")
        trajectory_ids.add(trajectory_id)
        groups.setdefault(trajectory.provenance.reset_group_id, []).append(trajectory)
    ordered_groups = sorted(
        groups,
        key=lambda group_id: hashlib.sha256(
            f"{policy.seed}:{group_id}".encode("utf-8")
        ).digest(),
    )
    counts = _allocation_counts(
        len(ordered_groups),
        (
            float(policy.train_fraction),
            float(policy.validation_fraction),
            float(policy.test_fraction),
        ),
    )
    boundaries = (counts[0], counts[0] + counts[1])
    group_partitions = (
        ordered_groups[: boundaries[0]],
        ordered_groups[boundaries[0] : boundaries[1]],
        ordered_groups[boundaries[1] :],
    )

    def materialize(group_ids: list[str]) -> tuple[Stage0Trajectory, ...]:
        return tuple(
            sorted(
                (item for group_id in group_ids for item in groups[group_id]),
                key=lambda item: item.provenance.trajectory_id,
            )
        )

    return Stage0TrajectorySplits(
        train=materialize(group_partitions[0]),
        validation=materialize(group_partitions[1]),
        test=materialize(group_partitions[2]),
        policy_sha256=policy.sha256,
    )


def build_success_window_splits(
    trajectories: Iterable[Stage0Trajectory],
    *,
    split_policy: Stage0SplitPolicy,
    window_policy: Stage0WindowPolicy,
) -> Stage0WindowSplits:
    """Split trajectories first, then extract success-only windows."""
    items = tuple(trajectories)
    corpus_sha256 = trajectory_corpus_sha256(items)
    splits = split_trajectories(items, split_policy)

    def extract(split: tuple[Stage0Trajectory, ...]) -> tuple[Stage0SuccessWindow, ...]:
        return tuple(
            window
            for trajectory in split
            for window in extract_success_windows(
                trajectory,
                window_policy,
                corpus_sha256=corpus_sha256,
            )
        )

    return Stage0WindowSplits(
        train=extract(splits.train),
        validation=extract(splits.validation),
        test=extract(splits.test),
        corpus_sha256=corpus_sha256,
        split_policy_sha256=split_policy.sha256,
        window_policy_sha256=window_policy.sha256,
    )
