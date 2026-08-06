"""Tests for deterministic success-latent occupancy and resume."""

from __future__ import annotations

import copy
import math
import unittest

from rexpolicy.manifold.occupancy import (
    OccupancySuccessRecord,
    SuccessOccupancyConfig,
    SuccessOccupancyIndex,
)


def _config(**overrides):
    values = {
        "enabled": True,
        "latent_dim": 2,
        "cluster_radius": 0.5,
        "density_bandwidth": 0.5,
        "coverage_target_modes": 4,
        "max_records": 8,
    }
    values.update(overrides)
    return SuccessOccupancyConfig(**values)


def _record(
    sample_id: str,
    latent: tuple[float, float],
    *,
    generation: int,
    rank: int = 0,
) -> OccupancySuccessRecord:
    return OccupancySuccessRecord.from_record(
        {
            "sample_id": sample_id,
            "latent": list(latent),
            "generation": generation,
            "rank": rank,
            "episode_id": f"episode-{generation}-{rank}",
            "source_sha256": "a" * 64,
        },
        latent_dim=2,
    )


class TestSuccessOccupancy(unittest.TestCase):
    def test_default_off_config_is_strict(self) -> None:
        self.assertFalse(SuccessOccupancyConfig().enabled)
        with self.assertRaisesRegex(ValueError, "Unknown success occupancy"):
            SuccessOccupancyConfig.from_mapping({"typo": True})
        with self.assertRaisesRegex(ValueError, "raw_l2"):
            SuccessOccupancyConfig(distance_metric="cosine")

    def test_empty_near_far_density_and_metrics(self) -> None:
        index = SuccessOccupancyIndex(_config())
        empty = index.query((0.0, 0.0))
        self.assertIsNone(empty.nearest_distance)
        self.assertIsNone(empty.log_density)
        self.assertTrue(empty.is_novel)
        self.assertEqual(index.snapshot().effective_mode_count, 0.0)

        first = index.observe_success(
            _record("sample-a", (0.0, 0.0), generation=1)
        )
        near = index.observe_success(
            _record("sample-b", (0.1, 0.0), generation=1)
        )
        far = index.observe_success(
            _record("sample-c", (2.0, 0.0), generation=2)
        )
        self.assertTrue(first.is_new_mode)
        self.assertFalse(near.is_new_mode)
        self.assertTrue(far.is_new_mode)
        self.assertTrue(math.isfinite(near.log_density_before))
        metrics = index.snapshot()
        self.assertEqual(metrics.successful_latent_count, 3)
        self.assertEqual(metrics.discovered_modes, 2)
        self.assertEqual(metrics.novel_successes, 2)
        self.assertAlmostEqual(metrics.target_mode_coverage, 0.5)
        generation = index.snapshot(1)
        self.assertEqual(generation.successful_latent_count, 2)
        self.assertEqual(generation.active_modes, 1)

    def test_canonical_merge_is_order_independent_and_checkpointable(self) -> None:
        records = [
            _record("sample-c", (2.0, 0.0), generation=2),
            _record("sample-b", (0.1, 0.0), generation=1, rank=1),
            _record("sample-a", (0.0, 0.0), generation=1),
        ]
        first = SuccessOccupancyIndex(_config())
        second = SuccessOccupancyIndex(_config())
        first.merge(records)
        second.merge(list(reversed(records)))
        self.assertEqual(first.state_dict(), second.state_dict())
        self.assertEqual(first.fingerprint, second.fingerprint)

        restored = SuccessOccupancyIndex.from_state_dict(
            _config(), first.state_dict()
        )
        self.assertEqual(restored.fingerprint, first.fingerprint)
        duplicate = restored.observe_success(records[0])
        self.assertFalse(duplicate.accepted)
        self.assertEqual(duplicate.reason, "duplicate_sample_id")

        corrupt = copy.deepcopy(first.state_dict())
        corrupt["records"][0]["latent"] = [float("nan"), 0.0]
        with self.assertRaisesRegex(ValueError, "finite"):
            restored.load_state_dict(corrupt)

    def test_conflicts_order_and_capacity_fail_closed(self) -> None:
        index = SuccessOccupancyIndex(_config(max_records=2))
        original = _record("sample-a", (0.0, 0.0), generation=1)
        index.observe_success(original)
        conflict = _record("sample-a", (1.0, 0.0), generation=1)
        with self.assertRaisesRegex(ValueError, "Conflicting"):
            index.observe_success(conflict)
        index.observe_success(
            _record("sample-b", (1.0, 0.0), generation=2)
        )
        with self.assertRaisesRegex(ValueError, "max_records"):
            index.observe_success(
                _record("sample-c", (2.0, 0.0), generation=3)
            )
        out_of_order = SuccessOccupancyIndex(_config())
        out_of_order.observe_success(
            _record("sample-b", (1.0, 0.0), generation=2)
        )
        with self.assertRaisesRegex(ValueError, "canonical order"):
            out_of_order.observe_success(original)


if __name__ == "__main__":
    unittest.main()
