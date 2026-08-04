"""Tests for reward-free sealed-evidence curriculum scheduling."""

from __future__ import annotations

import hashlib
import unittest
from dataclasses import replace

from rexpolicy.tasking.curriculum import (
    FRONTIER,
    HARD,
    INSUFFICIENT_EVIDENCE,
    MASTERED,
    PREREQUISITE_BLOCKED,
    SAFETY_EXCLUDED,
    CurriculumPolicy,
    CurriculumTaskUnit,
    SealedEvaluationEvidence,
    assess_curriculum,
    build_curriculum_schedule,
    frontier_weight,
)


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _unit(
    name: str,
    *,
    prerequisites: tuple[str, ...] = (),
) -> CurriculumTaskUnit:
    return CurriculumTaskUnit(
        schema_version=1,
        unit_id=f"reach/{name}.v1",
        task_contract_id="rexpolicy_taskspec_v2/reach_green_cap/v2",
        task_oracle_sha256=_hash("reach-task-oracle"),
        instruction=f"Reach the green cap from the {name} initial state.",
        initial_state_id=f"reach_reset/{name}.v1",
        initial_state_sha256=_hash(f"initial:{name}"),
        prerequisite_unit_ids=prerequisites,
    )


_CANDIDATE = _hash("candidate-policy")


def _evidence(
    unit: CurriculumTaskUnit,
    *,
    episodes: int,
    successes: int,
    safety_failures: int = 0,
    invalid_actions: int = 0,
) -> SealedEvaluationEvidence:
    return SealedEvaluationEvidence(
        schema_version=1,
        evidence_id=f"sealed_eval/{unit.unit_id.replace('/', '_')}.v1",
        unit_id=unit.unit_id,
        unit_sha256=unit.fingerprint,
        candidate_policy_sha256=_CANDIDATE,
        audit_suite_sha256=_hash(f"audit:{unit.unit_id}"),
        evaluation_report_sha256=_hash(f"report:{unit.unit_id}"),
        candidate_count=1,
        episode_count=episodes,
        success_count=successes,
        safety_failure_count=safety_failures,
        invalid_action_count=invalid_actions,
    )


def _policy(**overrides: object) -> CurriculumPolicy:
    return replace(
        CurriculumPolicy.production_v1(),
        min_evaluation_episodes=20,
        **overrides,
    )


def _assess(
    units: tuple[CurriculumTaskUnit, ...],
    evidence: tuple[SealedEvaluationEvidence, ...],
    *,
    policy: CurriculumPolicy | None = None,
):
    return assess_curriculum(
        units=units,
        evidence=evidence,
        candidate_policy_sha256=_CANDIDATE,
        policy=_policy() if policy is None else policy,
    )


class TestFrontierWeight(unittest.TestCase):
    def test_true_frontier_weight_peaks_at_half_success(self) -> None:
        peak = frontier_weight(0.5)
        left = frontier_weight(0.3)
        right = frontier_weight(0.7)

        self.assertEqual(peak, 1.0)
        self.assertAlmostEqual(left, right)
        self.assertGreater(peak, left)
        self.assertGreater(left, frontier_weight(0.0))
        self.assertGreater(right, frontier_weight(1.0))


class TestCurriculumEvidenceGates(unittest.TestCase):
    def test_insufficient_sealed_sample_count_is_not_schedulable(self) -> None:
        unit = _unit("small_sample")
        assessment = _assess(
            (unit,),
            (_evidence(unit, episodes=8, successes=4),),
        )[0]

        self.assertEqual(assessment.status, INSUFFICIENT_EVIDENCE)
        self.assertFalse(assessment.eligible)
        self.assertEqual(assessment.selection_weight, 0.0)
        self.assertIsNotNone(assessment.confidence_lower_bound)

    def test_any_disallowed_safety_result_excludes_the_unit(self) -> None:
        unit = _unit("unsafe")
        assessment = _assess(
            (unit,),
            (
                _evidence(
                    unit,
                    episodes=100,
                    successes=50,
                    safety_failures=1,
                ),
            ),
        )[0]

        self.assertEqual(assessment.status, SAFETY_EXCLUDED)
        self.assertFalse(assessment.eligible)

    def test_prerequisite_requires_confident_mastery(self) -> None:
        prerequisite = _unit("prerequisite")
        child = _unit(
            "child",
            prerequisites=(prerequisite.unit_id,),
        )
        by_id = {
            item.unit_id: item
            for item in _assess(
                (child, prerequisite),
                (
                    _evidence(prerequisite, episodes=100, successes=50),
                    _evidence(child, episodes=100, successes=100),
                ),
            )
        }

        self.assertEqual(by_id[prerequisite.unit_id].status, FRONTIER)
        self.assertEqual(by_id[child.unit_id].status, PREREQUISITE_BLOCKED)
        self.assertEqual(
            by_id[child.unit_id].missing_prerequisite_ids,
            (prerequisite.unit_id,),
        )

        mastered = _evidence(prerequisite, episodes=100, successes=100)
        unlocked = {
            item.unit_id: item
            for item in _assess(
                (prerequisite, child),
                (mastered, _evidence(child, episodes=100, successes=50)),
            )
        }
        self.assertEqual(unlocked[prerequisite.unit_id].status, MASTERED)
        self.assertEqual(unlocked[child.unit_id].status, FRONTIER)

    def test_reward_fields_are_rejected_at_the_evidence_boundary(self) -> None:
        unit = _unit("reward_free")
        record = _evidence(unit, episodes=100, successes=50).to_record()
        record["training_reward"] = 123.0

        with self.assertRaisesRegex(ValueError, "unknown fields: training_reward"):
            SealedEvaluationEvidence.from_record(record)

        task_record = unit.to_record()
        task_record["reward"] = 1.0
        with self.assertRaisesRegex(ValueError, "unknown fields: reward"):
            CurriculumTaskUnit.from_record(task_record)

    def test_evidence_is_bound_to_exact_instruction_state_and_policy(self) -> None:
        unit = _unit("bound")
        evidence = _evidence(unit, episodes=100, successes=50)
        changed = replace(unit, instruction="Reach the green cap carefully.")

        with self.assertRaisesRegex(ValueError, "unit fingerprint mismatch"):
            _assess((changed,), (evidence,))

        with self.assertRaisesRegex(ValueError, "candidate policy mismatch"):
            assess_curriculum(
                units=(unit,),
                evidence=(evidence,),
                candidate_policy_sha256=_hash("different-policy"),
                policy=_policy(),
            )


class TestCurriculumSchedule(unittest.TestCase):
    def test_schedule_is_deterministic_and_preserves_lane_minimums(self) -> None:
        mastered = _unit("mastered")
        hard = _unit("hard")
        frontier_a = _unit("frontier_a")
        frontier_b = _unit("frontier_b")
        units = (mastered, hard, frontier_a, frontier_b)
        evidence = (
            _evidence(mastered, episodes=100, successes=100),
            _evidence(hard, episodes=100, successes=0),
            _evidence(frontier_a, episodes=100, successes=50),
            _evidence(frontier_b, episodes=100, successes=55),
        )

        first = build_curriculum_schedule(
            units=units,
            evidence=evidence,
            candidate_policy_sha256=_CANDIDATE,
            policy=_policy(),
            total_slots=12,
            seed=73,
        )
        second = build_curriculum_schedule(
            units=tuple(reversed(units)),
            evidence=tuple(reversed(evidence)),
            candidate_policy_sha256=_CANDIDATE,
            policy=_policy(),
            total_slots=12,
            seed=73,
        )

        self.assertEqual(first, second)
        self.assertEqual(first.fingerprint, second.fingerprint)
        self.assertEqual(first.slots[0].lane, "mastered_replay")
        self.assertEqual(first.slots[0].unit_id, mastered.unit_id)
        self.assertEqual(first.slots[1].lane, "hard_exposure")
        self.assertEqual(first.slots[1].unit_id, hard.unit_id)
        self.assertEqual(len(first.slots), 12)
        with self.assertRaises(AttributeError):
            first.slots = ()

    def test_hard_and_mastered_minimums_fail_closed(self) -> None:
        frontier = _unit("only_frontier")
        evidence = (_evidence(frontier, episodes=100, successes=50),)

        with self.assertRaisesRegex(ValueError, "Mastered replay minimum"):
            build_curriculum_schedule(
                units=(frontier,),
                evidence=evidence,
                candidate_policy_sha256=_CANDIDATE,
                policy=_policy(),
                total_slots=4,
            )

    def test_deterministic_ranking_without_a_seed(self) -> None:
        mastered = _unit("mastered_unseeded")
        hard = _unit("hard_unseeded")
        frontier = _unit("frontier_unseeded")
        units = (mastered, hard, frontier)
        evidence = (
            _evidence(mastered, episodes=100, successes=100),
            _evidence(hard, episodes=100, successes=0),
            _evidence(frontier, episodes=100, successes=50),
        )
        kwargs = {
            "units": units,
            "evidence": evidence,
            "candidate_policy_sha256": _CANDIDATE,
            "policy": _policy(),
            "total_slots": 5,
        }

        first = build_curriculum_schedule(**kwargs)
        second = build_curriculum_schedule(**kwargs)

        self.assertEqual(first, second)
        self.assertEqual(first.slots[2].lane, "frontier_mix")
        self.assertEqual(first.slots[2].unit_id, frontier.unit_id)


class TestCurriculumArtifacts(unittest.TestCase):
    def test_canonical_round_trips_have_stable_fingerprints(self) -> None:
        unit = _unit("round_trip")
        evidence = _evidence(unit, episodes=100, successes=50)
        policy = _policy()

        self.assertEqual(
            CurriculumTaskUnit.from_json(unit.to_json()).fingerprint,
            unit.fingerprint,
        )
        self.assertEqual(
            SealedEvaluationEvidence.from_json(evidence.to_json()).fingerprint,
            evidence.fingerprint,
        )
        self.assertEqual(
            CurriculumPolicy.from_json(policy.to_json()).fingerprint,
            policy.fingerprint,
        )

    def test_confidence_bounds_define_hard_and_mastered_groups(self) -> None:
        hard = _unit("classified_hard")
        mastered = _unit("classified_mastered")
        by_id = {
            item.unit_id: item
            for item in _assess(
                (hard, mastered),
                (
                    _evidence(hard, episodes=100, successes=0),
                    _evidence(mastered, episodes=100, successes=100),
                ),
            )
        }

        self.assertEqual(by_id[hard.unit_id].status, HARD)
        self.assertLessEqual(by_id[hard.unit_id].confidence_upper_bound, 0.2)
        self.assertEqual(by_id[mastered.unit_id].status, MASTERED)
        self.assertGreaterEqual(
            by_id[mastered.unit_id].confidence_lower_bound,
            0.8,
        )


if __name__ == "__main__":
    unittest.main()
