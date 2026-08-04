"""Tests for declassified curriculum-to-authoring briefs."""

from __future__ import annotations

import unittest

from rexpolicy.tasking.authoring_brief import (
    PublicAuthoringBrief,
    derive_public_authoring_brief,
)
from rexpolicy.tasking.canonical import canonical_json
from rexpolicy.tasking.curriculum import (
    FRONTIER,
    MASTERED,
    CurriculumAssessment,
)
from rexpolicy.tasking.repository import load_production_reach_artifacts


def _assessment(unit_id: str, status: str, success_count: int) -> CurriculumAssessment:
    return CurriculumAssessment(
        unit_id=unit_id,
        unit_sha256="1" * 64,
        evidence_sha256="2" * 64,
        episode_count=100,
        success_count=success_count,
        safety_failure_count=0,
        invalid_action_count=0,
        success_probability=success_count / 100,
        confidence_lower_bound=0.4,
        confidence_upper_bound=0.6,
        status=status,
        missing_prerequisite_ids=(),
        frontier_mass=1.0,
        confidence_mass=0.8,
        selection_weight=0.8,
    )


class TestPublicAuthoringBrief(unittest.TestCase):
    def test_curriculum_declassification_exposes_only_categories_and_hashes(self) -> None:
        parent = load_production_reach_artifacts()
        frontier = _assessment("reach_green_cap/frontier/v1", FRONTIER, 51)
        mastered = _assessment("reach_green_cap/mastered/v1", MASTERED, 99)
        brief = derive_public_authoring_brief(
            brief_id="reach_green_cap/v3/brief/v1",
            task_id="reach_green_cap/v3",
            family_id="reach_green_cap",
            objective="Author a safe task variant for the public curriculum gap.",
            source_curriculum_snapshot_fingerprint="3" * 64,
            assessments=(mastered, frontier),
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
        mastered = _assessment("reach_green_cap/mastered/v1", MASTERED, 99)
        arguments = {
            "brief_id": "reach_green_cap/v3/brief/v1",
            "task_id": "reach_green_cap/v3",
            "family_id": "reach_green_cap",
            "objective": "Author a safe task variant.",
            "source_curriculum_snapshot_fingerprint": "3" * 64,
            "parent_task": parent.task,
            "parent_contract": parent.contract,
            "allowed_change_paths": ("$.task_id", "$.version"),
            "required_curriculum_prerequisite_ids": (),
            "public_constraints": ("Keep reward profiles fixed.",),
        }
        with self.assertRaisesRegex(ValueError, "No authorable"):
            derive_public_authoring_brief(
                assessments=(mastered,),
                gap_codes_by_unit={},
                **arguments,
            )

        frontier = _assessment("reach_green_cap/frontier/v1", FRONTIER, 51)
        with self.assertRaisesRegex(ValueError, "exactly cover"):
            derive_public_authoring_brief(
                assessments=(frontier,),
                gap_codes_by_unit={},
                **arguments,
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
