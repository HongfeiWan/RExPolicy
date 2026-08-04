"""Deterministic, diagnostic-only projections of learned success latents.

The two-dimensional coordinates produced here are never an oracle, reward,
quality-diversity descriptor, replay eligibility signal, or promotion gate.
They are a rebuildable view used to inspect whether self-supervised latents
develop useful structure.
"""

from __future__ import annotations

import html
import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from rexpolicy.tasking.canonical import (
    canonical_fingerprint,
    canonical_json,
    strict_json_loads,
)

_METHODS = frozenset(("pca", "umap", "tsne"))


def _mapping(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{path} must be an object")
    return value


def _keys(record: Mapping[str, Any], expected: set[str], path: str) -> None:
    missing = sorted(expected.difference(record))
    unknown = sorted(set(record).difference(expected))
    if missing:
        raise ValueError(f"{path} is missing fields: {', '.join(missing)}")
    if unknown:
        raise ValueError(f"{path} has unknown fields: {', '.join(unknown)}")


def _finite(value: Any, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{path} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{path} must be finite")
    return number


def _positive_int(value: Any, path: str) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{path} must be a positive integer")
    return value


def _label(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value or len(value) > 512:
        raise ValueError(f"{path} must be a non-empty bounded string")
    return value


@dataclass(frozen=True)
class SuccessLatentRecord:
    """One learned latent and non-authoritative diagnostic labels."""

    latent_id: str
    trajectory_id: str
    vector: tuple[float, ...]

    @classmethod
    def from_record(cls, value: Any, path: str) -> SuccessLatentRecord:
        record = _mapping(value, path)
        _keys(record, {"latent_id", "trajectory_id", "vector"}, path)
        raw_vector = record["vector"]
        if not isinstance(raw_vector, list) or not raw_vector:
            raise ValueError(f"{path}.vector must be a non-empty array")
        return cls(
            latent_id=_label(record["latent_id"], f"{path}.latent_id"),
            trajectory_id=_label(
                record["trajectory_id"], f"{path}.trajectory_id"
            ),
            vector=tuple(
                _finite(item, f"{path}.vector[{index}]")
                for index, item in enumerate(raw_vector)
            ),
        )

    def to_record(self) -> dict[str, Any]:
        return {
            "latent_id": self.latent_id,
            "trajectory_id": self.trajectory_id,
            "vector": list(self.vector),
        }


@dataclass(frozen=True)
class SuccessLatentDataset:
    """Content-addressed encoder output accepted by projection tools."""

    schema_version: int
    encoder_checkpoint_sha256: str
    window_policy_sha256: str
    rows: tuple[SuccessLatentRecord, ...]
    artifact_type: str = "rexpolicy_success_latents"

    @classmethod
    def from_record(cls, value: Any) -> SuccessLatentDataset:
        path = "$success_latents"
        record = _mapping(value, path)
        _keys(
            record,
            {
                "schema_version",
                "artifact_type",
                "encoder_checkpoint_sha256",
                "window_policy_sha256",
                "rows",
            },
            path,
        )
        raw_rows = record["rows"]
        if not isinstance(raw_rows, list) or not raw_rows:
            raise ValueError("Success latent dataset must contain rows")
        dataset = cls(
            schema_version=record["schema_version"],
            artifact_type=record["artifact_type"],
            encoder_checkpoint_sha256=record["encoder_checkpoint_sha256"],
            window_policy_sha256=record["window_policy_sha256"],
            rows=tuple(
                SuccessLatentRecord.from_record(item, f"{path}.rows[{index}]")
                for index, item in enumerate(raw_rows)
            ),
        )
        dataset.validate()
        return dataset

    @classmethod
    def from_json(cls, text: str) -> SuccessLatentDataset:
        return cls.from_record(strict_json_loads(text))

    def validate(self) -> None:
        if self.schema_version != 1:
            raise ValueError("Success latent dataset schema_version must be 1")
        if self.artifact_type != "rexpolicy_success_latents":
            raise ValueError("Unsupported success latent artifact type")
        for name, digest in (
            ("encoder_checkpoint_sha256", self.encoder_checkpoint_sha256),
            ("window_policy_sha256", self.window_policy_sha256),
        ):
            if (
                not isinstance(digest, str)
                or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
            ):
                raise ValueError(f"{name} must be a lowercase SHA-256")
        rows = tuple(self.rows)
        if not rows:
            raise ValueError("Success latent dataset must contain rows")
        dimensions = {len(row.vector) for row in rows}
        if len(dimensions) != 1:
            raise ValueError("Every success latent must have the same dimension")
        ids = tuple(row.latent_id for row in rows)
        if ids != tuple(sorted(set(ids))):
            raise ValueError("Success latent rows must be sorted by unique latent_id")
        for row in rows:
            SuccessLatentRecord.from_record(row.to_record(), "$row")

    def to_record(self) -> dict[str, Any]:
        self.validate()
        return {
            "schema_version": self.schema_version,
            "artifact_type": self.artifact_type,
            "encoder_checkpoint_sha256": self.encoder_checkpoint_sha256,
            "window_policy_sha256": self.window_policy_sha256,
            "rows": [row.to_record() for row in self.rows],
        }

    def to_json(self) -> str:
        return canonical_json(self.to_record())

    @property
    def fingerprint(self) -> str:
        return canonical_fingerprint(self.to_record())


@dataclass(frozen=True)
class LatentProjectionPolicy:
    """Fully specified UMAP/t-SNE/PCA diagnostic policy."""

    method: str = "pca"
    seed: int = 20260804
    max_rows: int = 4096
    umap_neighbors: int = 15
    umap_min_dist: float = 0.1
    tsne_perplexity: float = 30.0
    schema_version: int = 1
    policy_id: str = "success_manifold_projection/v1"

    def validate(self) -> None:
        if self.schema_version != 1:
            raise ValueError("Projection policy schema_version must be 1")
        if self.policy_id != "success_manifold_projection/v1":
            raise ValueError("Unsupported projection policy_id")
        if self.method not in _METHODS:
            raise ValueError("Projection method must be pca, umap, or tsne")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("Projection seed must be a non-negative integer")
        _positive_int(self.max_rows, "max_rows")
        if self.max_rows > 1_000_000:
            raise ValueError("Projection max_rows is unbounded")
        if not 2 <= self.umap_neighbors <= 1024:
            raise ValueError("umap_neighbors must be in [2, 1024]")
        minimum_distance = _finite(self.umap_min_dist, "umap_min_dist")
        if not 0.0 <= minimum_distance <= 1.0:
            raise ValueError("umap_min_dist must be in [0, 1]")
        perplexity = _finite(self.tsne_perplexity, "tsne_perplexity")
        if perplexity <= 0.0:
            raise ValueError("tsne_perplexity must be positive")

    def to_record(self) -> dict[str, Any]:
        self.validate()
        return {
            "schema_version": self.schema_version,
            "policy_id": self.policy_id,
            "method": self.method,
            "seed": self.seed,
            "max_rows": self.max_rows,
            "umap_neighbors": self.umap_neighbors,
            "umap_min_dist": self.umap_min_dist,
            "tsne_perplexity": self.tsne_perplexity,
        }

    @property
    def fingerprint(self) -> str:
        return canonical_fingerprint(self.to_record())


def _pca_2d(matrix: Any) -> tuple[Any, str]:
    import numpy as np

    centered = matrix - matrix.mean(axis=0, keepdims=True)
    _, _, right = np.linalg.svd(centered, full_matrices=False)
    component_count = min(2, right.shape[0])
    components = right[:component_count].copy()
    for index in range(component_count):
        pivot = int(np.argmax(np.abs(components[index])))
        if components[index, pivot] < 0.0:
            components[index] *= -1.0
    coordinates = centered @ components.T
    if component_count < 2:
        coordinates = np.pad(coordinates, ((0, 0), (0, 2 - component_count)))
    return coordinates, f"numpy-{np.__version__}"


def _project(matrix: Any, policy: LatentProjectionPolicy) -> tuple[Any, str]:
    if policy.method == "pca":
        return _pca_2d(matrix)
    if policy.method == "umap":
        try:
            import umap
        except ImportError as error:
            raise RuntimeError(
                "UMAP projection requires the optional umap-learn package"
            ) from error
        if matrix.shape[0] < 3:
            raise ValueError("UMAP projection requires at least three rows")
        neighbors = min(policy.umap_neighbors, int(matrix.shape[0]) - 1)
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=neighbors,
            min_dist=policy.umap_min_dist,
            metric="cosine",
            random_state=policy.seed,
            transform_seed=policy.seed,
            n_jobs=1,
        )
        return reducer.fit_transform(matrix), f"umap-{umap.__version__}"
    try:
        import sklearn
        from sklearn.manifold import TSNE
    except ImportError as error:
        raise RuntimeError(
            "t-SNE projection requires the optional scikit-learn package"
        ) from error
    if matrix.shape[0] <= policy.tsne_perplexity:
        raise ValueError("t-SNE row count must exceed perplexity")
    reducer = TSNE(
        n_components=2,
        perplexity=policy.tsne_perplexity,
        metric="cosine",
        init="random",
        learning_rate="auto",
        random_state=policy.seed,
    )
    return reducer.fit_transform(matrix), f"scikit-learn-{sklearn.__version__}"


def project_success_latents(
    dataset: SuccessLatentDataset,
    *,
    policy: LatentProjectionPolicy,
) -> dict[str, Any]:
    """Project latents into 2-D without granting coordinates authority."""
    import numpy as np

    dataset.validate()
    policy.validate()
    if len(dataset.rows) > policy.max_rows:
        raise ValueError(
            f"Latent dataset has {len(dataset.rows)} rows, limit is {policy.max_rows}"
        )
    matrix = np.asarray([row.vector for row in dataset.rows], dtype=np.float64)
    if matrix.ndim != 2 or not np.isfinite(matrix).all():
        raise ValueError("Latent matrix must be finite and two-dimensional")
    coordinates, backend = _project(matrix, policy)
    coordinates = np.asarray(coordinates, dtype=np.float64)
    if coordinates.shape != (len(dataset.rows), 2) or not np.isfinite(
        coordinates
    ).all():
        raise RuntimeError("Projection backend returned invalid coordinates")
    return {
        "schema_version": 1,
        "artifact_type": "rexpolicy_success_latent_projection",
        "authority": "diagnostic_only",
        "input_latents_sha256": dataset.fingerprint,
        "projection_policy": policy.to_record(),
        "projection_policy_sha256": policy.fingerprint,
        "backend": backend,
        "rows": [
            {
                "latent_id": row.latent_id,
                "trajectory_id": row.trajectory_id,
                "coordinates": [float(value) for value in coordinate],
            }
            for row, coordinate in zip(dataset.rows, coordinates)
        ],
    }


def projection_svg(
    projection: Mapping[str, Any],
    *,
    width: int = 800,
    height: int = 600,
) -> str:
    """Render a dependency-free SVG scatter plot for an exact projection."""
    if min(width, height) < 100:
        raise ValueError("SVG dimensions are too small")
    rows = projection.get("rows")
    if not isinstance(rows, Sequence) or not rows:
        raise ValueError("Projection has no rows")
    points = []
    for index, raw in enumerate(rows):
        record = _mapping(raw, f"$projection.rows[{index}]")
        coordinates = record.get("coordinates")
        if not isinstance(coordinates, list) or len(coordinates) != 2:
            raise ValueError("Projection coordinate must have two dimensions")
        points.append(
            (
                _finite(coordinates[0], "x"),
                _finite(coordinates[1], "y"),
                _label(record.get("latent_id"), "latent_id"),
                _label(record.get("trajectory_id"), "trajectory_id"),
            )
        )
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    x_span = max(x_max - x_min, 1.0e-12)
    y_span = max(y_max - y_min, 1.0e-12)
    margin = 32.0
    palette = (
        "#2563eb",
        "#dc2626",
        "#059669",
        "#7c3aed",
        "#d97706",
        "#0891b2",
    )
    trajectories = sorted({point[3] for point in points})
    colors = {
        trajectory: palette[index % len(palette)]
        for index, trajectory in enumerate(trajectories)
    }
    circles = []
    for x, y, latent_id, trajectory_id in points:
        px = margin + (x - x_min) / x_span * (width - 2.0 * margin)
        py = height - margin - (y - y_min) / y_span * (height - 2.0 * margin)
        title = html.escape(f"{latent_id} · {trajectory_id}")
        circles.append(
            f'<circle cx="{px:.3f}" cy="{py:.3f}" r="4" '
            f'fill="{colors[trajectory_id]}"><title>{title}</title></circle>'
        )
    body = "".join(circles)
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="{height}" viewBox="0 0 {width} {height}">'
        '<rect width="100%" height="100%" fill="white"/>'
        f"{body}</svg>\n"
    )
