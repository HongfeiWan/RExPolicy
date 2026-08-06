"""Tests for shadow-only exploration scoring and schedules."""

from __future__ import annotations

import json
import unittest

from rexpolicy.manifold.exploration import (
    ExplorationPolicy,
    score_shadow_exploration,
)
from rexpolicy.manifold.occupancy import OccupancyQuery


def _query(distance):
    return OccupancyQuery(
        nearest_mode_id=None if distance is None else "mode-a",
        nearest_distance=distance,
        log_density=None,
        is_novel=distance is None or distance >= 0.5,
        successful_latent_count=0 if distance is None else 7,
        discovered_modes=0 if distance is None else 3,
    )


class TestSuccessExploration(unittest.TestCase):
    def test_default_off_policy_is_strict_and_schedule_is_resume_stable(self) -> None:
        self.assertFalse(ExplorationPolicy().enabled)
        with self.assertRaisesRegex(ValueError, "Unknown exploration"):
            ExplorationPolicy.from_mapping({"typo": 1})
        policy = ExplorationPolicy(
            enabled=True,
            selector_temperature_start=2.0,
            selector_temperature_end=0.5,
            novelty_coefficient_start=1.0,
            novelty_coefficient_end=0.0,
            anneal_generations=10,
        )
        self.assertEqual(policy.at_generation(1).selector_temperature, 2.0)
        middle = policy.at_generation(6)
        self.assertAlmostEqual(middle.selector_temperature, 1.25)
        self.assertAlmostEqual(middle.novelty_coefficient, 0.5)
        self.assertEqual(policy.at_generation(11).selector_temperature, 0.5)
        self.assertEqual(
            ExplorationPolicy.from_mapping(policy.to_record()).fingerprint,
            policy.fingerprint,
        )

    def test_shadow_formula_does_not_emit_json_infinity(self) -> None:
        policy = ExplorationPolicy(
            enabled=True,
            novelty_coefficient_start=0.5,
            novelty_coefficient_end=0.5,
            novelty_scale=2.0,
            novelty_cap=3.0,
        )
        schedule = policy.at_generation(4)
        bootstrap = score_shadow_exploration(
            sample_id="sample-a",
            latent_sha256="a" * 64,
            latent_source="selector",
            task_success_score=2.0,
            occupancy=_query(None),
            schedule=schedule,
            policy=policy,
        )
        self.assertTrue(bootstrap.bootstrap)
        self.assertIsNone(bootstrap.novelty_distance)
        self.assertEqual(bootstrap.shadow_score, 3.5)
        payload = json.dumps(bootstrap.to_record(), allow_nan=False)
        self.assertNotIn("Infinity", payload)

        observed = score_shadow_exploration(
            sample_id="sample-b",
            latent_sha256="b" * 64,
            latent_source="latent_memory",
            task_success_score=2.0,
            occupancy=_query(1.0),
            schedule=schedule,
            policy=policy,
        )
        self.assertFalse(observed.bootstrap)
        self.assertEqual(observed.normalized_novelty, 0.5)
        self.assertEqual(observed.shadow_score, 2.25)
        self.assertEqual(observed.authority, "shadow_only")

    def test_policy_or_schedule_drift_fails_closed(self) -> None:
        disabled = ExplorationPolicy()
        with self.assertRaisesRegex(ValueError, "enabled policy"):
            score_shadow_exploration(
                sample_id="sample",
                latent_sha256="a" * 64,
                latent_source="selector",
                task_success_score=0.0,
                occupancy=_query(None),
                schedule=disabled.at_generation(1),
                policy=disabled,
            )
        policy = ExplorationPolicy(enabled=True)
        drifted = ExplorationPolicy(enabled=True, novelty_scale=2.0)
        with self.assertRaisesRegex(ValueError, "does not match"):
            score_shadow_exploration(
                sample_id="sample",
                latent_sha256="a" * 64,
                latent_source="selector",
                task_success_score=0.0,
                occupancy=_query(None),
                schedule=drifted.at_generation(1),
                policy=policy,
            )


if __name__ == "__main__":
    unittest.main()
