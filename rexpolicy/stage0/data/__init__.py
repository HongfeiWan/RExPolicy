"""Materialized, state-only data contracts for Stage 0.

This package intentionally stores full state/action trajectories separately
from the metadata-only v2 success archive.  Importing it does not import
PyTorch; tensor validation and persistence load PyTorch only when used.
"""

from rexpolicy.stage0.data.splits import (
    Stage0SplitPolicy,
    Stage0TrajectorySplits,
    Stage0WindowSplits,
    build_success_window_splits,
    split_trajectories,
)
from rexpolicy.stage0.data.store import (
    Stage0ShardError,
    Stage0ShardIntegrityError,
    load_trajectory_shard,
    write_trajectory_shard,
)
from rexpolicy.stage0.data.trajectory import (
    Stage0Trajectory,
    Stage0TrajectoryProvenance,
    trajectory_corpus_sha256,
    trajectory_sha256,
)
from rexpolicy.stage0.data.windows import (
    Stage0SuccessWindow,
    Stage0WindowPolicy,
    extract_success_windows,
)

__all__ = [
    "Stage0ShardError",
    "Stage0ShardIntegrityError",
    "Stage0SplitPolicy",
    "Stage0SuccessWindow",
    "Stage0Trajectory",
    "Stage0TrajectoryProvenance",
    "Stage0TrajectorySplits",
    "Stage0WindowPolicy",
    "Stage0WindowSplits",
    "build_success_window_splits",
    "extract_success_windows",
    "load_trajectory_shard",
    "split_trajectories",
    "trajectory_corpus_sha256",
    "trajectory_sha256",
    "write_trajectory_shard",
]
