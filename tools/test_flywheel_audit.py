"""Tests for sealed audit-suite and formal claim policy contracts."""

from __future__ import annotations

import unittest

from rexpolicy.flywheel.audit import (
    CapabilityClaimPolicy,
    SealedAuditSuite,
    canonical_sha256,
)


def _suite(**overrides: object) -> SealedAuditSuite:
    values: dict[str, object] = {
        "schema_version": 1,
        "audit_suite_id": "reach_green_cap/audit.v1",
        "task_id": "reach_green_cap/v1",
        "task_spec_fingerprint": "a" * 64,
        "claim_spec_id": "paired_k1_mastery/v1",
        "audit_instruction_count": 8,
        "audit_instruction_sha256": "b" * 64,
        "episodes_per_training_seed": 256,
        "episode_recipe_sha256": "c" * 64,
        "independent_training_seeds": (11, 22, 33),
        "candidate_count": 1,
    }
    values.update(overrides)
    return SealedAuditSuite(**values)


class TestSealedAuditSuite(unittest.TestCase):
    def test_commitment_contains_no_instruction_text(self) -> None:
        suite = _suite()
        suite.validate_against(CapabilityClaimPolicy.production_v1())
        record = suite.to_record()

        self.assertEqual(len(suite.fingerprint), 64)
        self.assertNotIn("instructions", record)
        self.assertNotIn("instruction_text", record)
        self.assertEqual(
            SealedAuditSuite.from_record(record).fingerprint,
            suite.fingerprint,
        )

    def test_production_policy_requires_formal_paired_k1_evidence(self) -> None:
        policy = CapabilityClaimPolicy.production_v1()
        policy.validate()

        self.assertEqual(policy.candidate_count, 1)
        self.assertEqual(policy.min_independent_training_seeds, 3)
        self.assertEqual(policy.min_episodes_per_seed, 256)
        self.assertEqual(policy.max_new_safety_failures, 0)
        self.assertEqual(policy.max_invalid_actions, 0)

    def test_k_way_search_cannot_be_an_audit_suite(self) -> None:
        with self.assertRaisesRegex(ValueError, "exactly one candidate"):
            _suite(candidate_count=8).validate()

    def test_suite_cannot_weaken_seed_or_episode_floor(self) -> None:
        policy = CapabilityClaimPolicy.production_v1()
        for suite, message in (
            (_suite(independent_training_seeds=(11, 22)), "too few independent"),
            (_suite(episodes_per_training_seed=255), "too few episodes"),
        ):
            with self.assertRaisesRegex(ValueError, message):
                suite.validate_against(policy)

    def test_seed_order_and_hash_encoding_are_deterministic(self) -> None:
        with self.assertRaisesRegex(ValueError, "unique, and sorted"):
            _suite(independent_training_seeds=(22, 11, 22)).validate()
        left = canonical_sha256({"b": 2, "a": 1})
        right = canonical_sha256({"a": 1, "b": 2})
        self.assertEqual(left, right)

    def test_non_finite_policy_threshold_is_rejected(self) -> None:
        policy = CapabilityClaimPolicy(
            **{
                **CapabilityClaimPolicy.production_v1().__dict__,
                "min_success_lower_bound": float("nan"),
            }
        )
        with self.assertRaisesRegex(ValueError, "threshold"):
            policy.validate()


if __name__ == "__main__":
    unittest.main()
