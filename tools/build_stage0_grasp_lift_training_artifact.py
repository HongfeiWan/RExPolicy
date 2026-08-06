#!/usr/bin/env python3
"""Publish the immutable train-only artifact for the accepted Grasp-Lift pilot."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def _positive_integer(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-directory", required=True)
    parser.add_argument("--output-directory", required=True)
    parser.add_argument("--cpu-threads", type=_positive_integer, default=16)
    return parser.parse_args()


def _restrict_cpu_threads(cpu_threads: int) -> tuple[int, ...]:
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[name] = str(cpu_threads)
    if not hasattr(os, "sched_getaffinity") or not hasattr(os, "sched_setaffinity"):
        return ()
    available = tuple(sorted(os.sched_getaffinity(0)))
    if len(available) < cpu_threads:
        raise RuntimeError(
            f"requested {cpu_threads} CPU threads but only {len(available)} are available"
        )
    selected = available[:cpu_threads]
    os.sched_setaffinity(0, selected)
    return selected


def main() -> None:
    args = _parse_args()
    affinity = _restrict_cpu_threads(args.cpu_threads)

    import torch

    from rexpolicy.stage0.data.grasp_lift_training_artifact import (
        publish_grasp_lift_training_artifact,
    )

    torch.set_num_threads(args.cpu_threads)
    artifact = publish_grasp_lift_training_artifact(
        Path(args.output_directory),
        pilot_directory=Path(args.pilot_directory),
    )
    summary = {
        "artifact_sha256": artifact.artifact_sha256,
        "audit_dataset_sha256": artifact.audit_dataset_sha256,
        "cpu_affinity": list(affinity),
        "manifest_sha256": artifact.manifest_sha256,
        "normalization_sha256": artifact.normalization.sha256,
        "output_directory": str(artifact.root),
        "schema_id": "rexpolicy/stage0-grasp-lift-training-build-result/v1",
        "test_windows": len(artifact.data.window_splits.test),
        "train_content_sha256": artifact.train_content_sha256,
        "train_windows": len(artifact.data.window_splits.train),
        "validation_windows": len(artifact.data.window_splits.validation),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
