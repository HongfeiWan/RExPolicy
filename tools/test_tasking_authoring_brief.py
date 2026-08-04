"""Tests for declassified curriculum-to-authoring briefs."""

from __future__ import annotations

import hashlib
import unittest
from dataclasses import replace

from rexpolicy.tasking.authoring_brief import (
    PublicAuthoringBrief,
    derive_public_authoring_brief,
)
from rexpolicy.tasking.canonical import canonical_json
from rexpolicy.tasking.curriculum import (
    FRONTIER,
    CurriculumPolicy,
    CurriculumSnapshot,
    CurriculumTaskUnit,
    SealedEvaluationEvidence,
)
from rexpolicy.tasking.repository import load_production_reach_artifacts


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


_CANDIDATE = _hash("authoring-brief-candidate")


def _snapshot(
    names_and_successes: tuple[tuple[str, int], ...],
) -> CurriculumSnapshot:
    parent = load_production_reach_artifacts()
    units = tuple(
        CurriculumTaskUnit(
            schema_version=1,
            unit_id=f"reach_green_cap/{name}/v1",
            task_contract_id=parent.contract.task_contract_id,
            task_oracle_sha256=parent.contract.oracle_fingerprint,
            instruction=f"Reach the green cap from public state {name}.",
            initial_state_id=f"reach_reset/{name}/v1",
            initial_state_sha256=_hash(f"initial:{name}"),
            prerequisite_unit_ids=(),
        )
        for name, _ in names_and_successes
    )
    evidence = tuple(
        SealedEvaluationEvidence(
            schema_version=1,
            evidence_id=f"sealed_eval/{name}/v1",
            unit_id=unit.unit_id,
            unit_sha256=unit.fingerprint,
            candidate_policy_sha256=_CANDIDATE,
            audit_suite_sha256=_hash(f"audit:{name}"),
            evaluation_report_sha256=_hash(f"report:{name}"),
            candidate_count=1,
            episode_count=100,
            success_count=successes,
            safety_failure_count=0,
            invalid_action_count=0,
        )
        for unit, (name, successes) in zip(units, names_and_successes)
    )
    return CurriculumSnapshot.create(
        candidate_policy_sha256=_CANDIDATE,
        curriculum_policy=replace(
            CurriculumPolicy.production_v1(),
            min_evaluation_episodes=20,
        ),
        units=units,
        evidence=evidence,
    )


class TestPublicAuthoringBrief(unittest.TestCase):
    def test_curriculum_declassification_exposes_only_categories_and_hashes(self) -> None:
        parent = load_production_reach_artifacts()
        snapshot = _snapshot((("mastered", 100), ("frontier", 51)))
        frontier = next(
            item for item in snapshot.assessments if item.status == FRONTIER
        )
        brief = derive_public_authoring_brief(
            brief_id="reach_green_cap/v3/brief/v1",
            task_id="reach_green_cap/v3",
            family_id="reach_green_cap",
            objective="Author a safe task variant for the public curriculum gap.",
            curriculum_snapshot=snapshot,
            gap_codes_by_unit={
                frontier.unit_id: "instruction_robustness",
            },
            parent_task=parent.task,
            parent_contract=parent.contract,
            allowed_change_paths=("$.task_id", "$.version"),
            required_curriculum_prerequisite_ids=(),
            public_constraints=("Keep reward profiles fixed.",),
        )

        restored = PublicAuthoringBrief.from_record(brief.to_record())
        self.assertEqual(restored, brief)
        self.assertEqual(
            brief.source_curriculum_snapshot_fingerprint,
            snapshot.fingerprint,
        )
        self.assertEqual(len(brief.gap_targets), 1)
        self.assertEqual(brief.gap_targets[0].status, FRONTIER)
        self.assertEqual(
            brief.gap_targets[0].assessment_fingerprint,
            frontier.fingerprint,
        )
        serialized = canonical_json(brief.to_record())
        for forbidden in (
            "episode_count",
            "success_count",
            "success_probability",
            "confidence_lower_bound",
            "evidence_sha256",
        ):
            self.assertNotIn(forbidden, serialized)

    def test_non_gap_assessments_and_partial_gap_codes_fail_closed(self) -> None:
        parent = load_production_reach_artifacts()
        arguments = {
            "brief_id": "reach_green_cap/v3/brief/v1",
            "task_id": "reach_green_cap/v3",
            "family_id": "reach_green_cap",
            "objective": "Author a safe task variant.",
            "parent_task": parent.task,
            "parent_contract": parent.contract,
            "allowed_change_paths": ("$.task_id", "$.version"),
            "required_curriculum_prerequisite_ids": (),
            "public_constraints": ("Keep reward profiles fixed.",),
        }
        with self.assertRaisesRegex(ValueError, "No authorable"):
            derive_public_authoring_brief(
                curriculum_snapshot=_snapshot((("mastered", 100),)),
                gap_codes_by_unit={},
                **arguments,
            )

        snapshot = _snapshot((("frontier", 51),))
        with self.assertRaisesRegex(ValueError, "exactly cover"):
            derive_public_authoring_brief(
                curriculum_snapshot=snapshot,
                gap_codes_by_unit={},
                **arguments,
            )

    def test_brief_cannot_pair_an_unrelated_snapshot_hash_and_assessments(self) -> None:
        parent = load_production_reach_artifacts()
        snapshot = _snapshot((("frontier", 51),))
        unit_id = snapshot.assessments[0].unit_id
        brief = derive_public_authoring_brief(
            brief_id="reach_green_cap/v3/brief/v1",
            task_id="reach_green_cap/v3",
            family_id="reach_green_cap",
            objective="Author a safe task variant.",
            curriculum_snapshot=snapshot,
            gap_codes_by_unit={unit_id: "instruction_robustness"},
            parent_task=parent.task,
            parent_contract=parent.contract,
            allowed_change_paths=("$.task_id", "$.version"),
            required_curriculum_prerequisite_ids=(),
            public_constraints=("Keep reward profiles fixed.",),
        )

        self.assertEqual(
            brief.source_curriculum_snapshot_fingerprint,
            snapshot.fingerprint,
        )
        self.assertEqual(
            brief.gap_targets[0].assessment_fingerprint,
            snapshot.assessments[0].fingerprint,
        )
        with self.assertRaises(TypeError):
            derive_public_authoring_brief(
                brief_id="reach_green_cap/v3/brief/v1",
                task_id="reach_green_cap/v3",
                family_id="reach_green_cap",
                objective="Author a safe task variant.",
                source_curriculum_snapshot_fingerprint="3" * 64,
                assessments=snapshot.assessments,
                gap_codes_by_unit={unit_id: "instruction_robustness"},
                parent_task=parent.task,
                parent_contract=parent.contract,
                allowed_change_paths=("$.task_id", "$.version"),
                required_curriculum_prerequisite_ids=(),
                public_constraints=("Keep reward profiles fixed.",),
            )

    def test_parentless_brief_requires_explicit_root_authority(self) -> None:
        record = {
            "schema_version": 1,
            "brief_id": "new_family/v1/brief/v1",
            "task_id": "new_family/v1",
            "family_id": "new_family",
            "objective": "Author a new task family.",
            "source_curriculum_snapshot_fingerprint": "3" * 64,
            "gap_targets": [
                {
                    "unit_id": "new_family/gap/v1",
                    "assessment_fingerprint": "4" * 64,
                    "status": "hard",
                    "gap_code": "new_capability",
                }
            ],
            "parent_task_spec_fingerprint": None,
            "parent_task_oracle_fingerprint": None,
            "allowed_change_paths": ["$.goal"],
            "required_curriculum_prerequisite_ids": [],
            "public_constraints": ["Use only approved capabilities."],
        }
        with self.assertRaisesRegex(ValueError, "root path"):
            PublicAuthoringBrief.from_record(record)

        record["allowed_change_paths"] = ["$"]
        self.assertEqual(
            PublicAuthoringBrief.from_record(record).allowed_change_paths,
            ("$",),
        )


if __name__ == "__main__":
    unittest.main()
