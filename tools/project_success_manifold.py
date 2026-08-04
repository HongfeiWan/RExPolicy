#!/usr/bin/env python3
"""Project a content-addressed success-latent dataset for inspection."""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

from rexpolicy.manifold.visualization import (
    LatentProjectionPolicy,
    SuccessLatentDataset,
    project_success_latents,
    projection_svg,
)
from rexpolicy.tasking.canonical import canonical_json


def _write_atomic(path: Path, payload: str) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.partial-", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="ascii") as file:
            file.write(payload)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--svg", type=Path)
    parser.add_argument("--method", choices=("pca", "umap", "tsne"), default="pca")
    parser.add_argument("--seed", type=int, default=20260804)
    parser.add_argument("--max-rows", type=int, default=4096)
    parser.add_argument("--umap-neighbors", type=int, default=15)
    parser.add_argument("--umap-min-dist", type=float, default=0.1)
    parser.add_argument("--tsne-perplexity", type=float, default=30.0)
    return parser


def main() -> None:
    args = create_parser().parse_args()
    dataset = SuccessLatentDataset.from_json(
        args.input.expanduser().resolve().read_text(encoding="ascii")
    )
    policy = LatentProjectionPolicy(
        method=args.method,
        seed=args.seed,
        max_rows=args.max_rows,
        umap_neighbors=args.umap_neighbors,
        umap_min_dist=args.umap_min_dist,
        tsne_perplexity=args.tsne_perplexity,
    )
    projection = project_success_latents(dataset, policy=policy)
    _write_atomic(args.output, canonical_json(projection) + "\n")
    if args.svg is not None:
        _write_atomic(args.svg, projection_svg(projection))


if __name__ == "__main__":
    main()
