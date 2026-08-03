"""Tests for sealed audit-suite and formal claim policy contracts."""

from __future__ import annotations

import unittest

from rexpolicy.flywheel.audit import (
    ClaimEpisodeResult,
    CapabilityClaimPolicy,
    SealedAuditSuite,
    canonical_sha256,
    evaluate_capability_claim,
    wilson_lower_bound,
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


def _test_policy() -> CapabilityClaimPolicy:
    return CapabilityClaimPolicy(
        schema_version=1,
        claim_spec_id="paired_k1_mastery/v1",
        candidate_count=1,
        min_independent_training_seeds=2,
        min_episodes_per_seed=4,
        min_success_lower_bound=0.20,
        max_new_safety_failures=0,
        max_invalid_actions=0,
    )


def _episodes() -> list[ClaimEpisodeResult]:
    return [
        ClaimEpisodeResult(
            training_seed=training_seed,
            reset_seed=100 + recipe,
            diffusion_seed=200 + recipe,
            success=True,
            safety_failure=False,
            baseline_safety_failure=False,
            invalid_action=False,
        )
        for training_seed in (11, 22)
        for recipe in range(4)
    ]


class TestCapabilityClaimReport(unittest.TestCase):
    def test_accepts_complete_paired_k1_evidence(self) -> None:
        suite = _suite(
            independent_training_seeds=(11, 22),
            episodes_per_training_seed=4,
        )
        report = evaluate_capability_claim(
            suite=suite,
            policy=_test_policy(),
            episodes=_episodes(),
        )

        self.assertTrue(report.accepted)
        self.assertEqual(report.total_episode_count, 8)
        self.assertEqual(len(report.seed_metrics), 2)
        self.assertGreater(report.seed_metrics[0].success_lower_bound, 0.20)
        self.assertEqual(len(report.fingerprint), 64)

    def test_report_hash_is_independent_of_input_order(self) -> None:
        suite = _suite(
            independent_training_seeds=(11, 22),
            episodes_per_training_seed=4,
        )
        forward = evaluate_capability_claim(
            suite=suite,
            policy=_test_policy(),
            episodes=_episodes(),
        )
        reverse = evaluate_capability_claim(
            suite=suite,
            policy=_test_policy(),
            episodes=list(reversed(_episodes())),
        )
        self.assertEqual(forward.fingerprint, reverse.fingerprint)

    def test_new_safety_and_invalid_actions_reject_claim(self) -> None:
        suite = _suite(
            independent_training_seeds=(11, 22),
            episodes_per_training_seed=4,
        )
        episodes = _episodes()
        episodes[0] = ClaimEpisodeResult(
            **{
                **episodes[0].__dict__,
                "safety_failure": True,
            }
        )
        episodes[1] = ClaimEpisodeResult(
            **{
                **episodes[1].__dict__,
                "invalid_action": True,
            }
        )
        report = evaluate_capability_claim(
            suite=suite,
            policy=_test_policy(),
            episodes=episodes,
        )
        self.assertFalse(report.accepted)
        self.assertTrue(any("safety" in reason for reason in report.reasons))
        self.assertTrue(any("invalid" in reason for reason in report.reasons))

    def test_incomplete_duplicate_or_unpaired_evidence_is_rejected(self) -> None:
        suite = _suite(
            independent_training_seeds=(11, 22),
            episodes_per_training_seed=4,
        )
        cases = (
            (_episodes()[:-1], "incomplete"),
            ([*_episodes(), _episodes()[0]], "duplicate"),
        )
        for episodes, message in cases:
            with self.assertRaisesRegex(ValueError, message):
                evaluate_capability_claim(
                    suite=suite,
                    policy=_test_policy(),
                    episodes=episodes,
                )

        unpaired = _episodes()
        unpaired[-1] = ClaimEpisodeResult(
            **{
                **unpaired[-1].__dict__,
                "reset_seed": 999,
            }
        )
        with self.assertRaisesRegex(ValueError, "not paired"):
            evaluate_capability_claim(
                suite=suite,
                policy=_test_policy(),
                episodes=unpaired,
            )

    def test_wilson_bound_is_conservative(self) -> None:
        self.assertLess(wilson_lower_bound(4, 4), 1.0)
        self.assertEqual(wilson_lower_bound(0, 4), 0.0)


if __name__ == "__main__":
    unittest.main()
