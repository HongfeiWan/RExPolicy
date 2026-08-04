"""Tests for diagnostic-only success-manifold projections."""

from __future__ import annotations

import unittest

from rexpolicy.manifold.visualization import (
    LatentProjectionPolicy,
    SuccessLatentDataset,
    project_success_latents,
    projection_svg,
)


def _dataset() -> SuccessLatentDataset:
    return SuccessLatentDataset.from_record(
        {
            "schema_version": 1,
            "artifact_type": "rexpolicy_success_latents",
            "encoder_checkpoint_sha256": "a" * 64,
            "window_policy_sha256": "b" * 64,
            "rows": [
                {
                    "latent_id": f"latent-{index}",
                    "trajectory_id": "trajectory-a" if index < 2 else "trajectory-b",
                    "vector": vector,
                }
                for index, vector in enumerate(
                    ([1.0, 0.0, 0.1], [0.9, 0.1, 0.0], [0.0, 1.0, 0.9])
                )
            ],
        }
    )


class TestSuccessManifoldVisualization(unittest.TestCase):
    def test_pca_projection_is_canonical_and_diagnostic_only(self) -> None:
        dataset = _dataset()
        policy = LatentProjectionPolicy(method="pca")
        first = project_success_latents(dataset, policy=policy)
        second = project_success_latents(dataset, policy=policy)
        self.assertEqual(first, second)
        self.assertEqual(first["authority"], "diagnostic_only")
        self.assertEqual(first["input_latents_sha256"], dataset.fingerprint)
        self.assertEqual(len(first["rows"]), 3)
        self.assertTrue(projection_svg(first).startswith("<svg"))

    def test_strict_dataset_rejects_dimension_and_order_drift(self) -> None:
        record = _dataset().to_record()
        record["rows"][1]["vector"] = [1.0]
        with self.assertRaisesRegex(ValueError, "same dimension"):
            SuccessLatentDataset.from_record(record)
        record = _dataset().to_record()
        record["rows"] = list(reversed(record["rows"]))
        with self.assertRaisesRegex(ValueError, "sorted"):
            SuccessLatentDataset.from_record(record)

    def test_projection_limits_and_tsne_preconditions_fail_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "limit"):
            project_success_latents(
                _dataset(), policy=LatentProjectionPolicy(max_rows=2)
            )
        with self.assertRaisesRegex(ValueError, "exceed perplexity"):
            project_success_latents(
                _dataset(),
                policy=LatentProjectionPolicy(method="tsne", tsne_perplexity=3.0),
            )


if __name__ == "__main__":
    unittest.main()
