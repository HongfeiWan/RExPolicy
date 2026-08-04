"""Tests for the automatic-authoring static quarantine boundary."""

from __future__ import annotations

import copy
import json
import unittest
from dataclasses import replace
from unittest.mock import patch

from rexpolicy.tasking.authoring import (
    DEFAULT_AUTHORING_POLICY,
    AuthoringIntent,
    ProposalAttempt,
)
from rexpolicy.tasking.authoring_candidate import (
    StaticCandidateBundle,
    TrustedAuthoringBindingError,
    validate_static_candidate,
)
from rexpolicy.tasking.authoring_brief import PublicAuthoringBrief
from rexpolicy.tasking.capabilities import CapabilityCatalog
from rexpolicy.tasking.contract import DEFAULT_TASK_COMPILER_POLICY
from rexpolicy.tasking.property_validation import (
    DEFAULT_PROPERTY_VALIDATION_POLICY,
)
from rexpolicy.tasking.process import ProcessSpecV1
from rexpolicy.tasking.process_contract import DEFAULT_PROCESS_COMPILER_POLICY
from rexpolicy.tasking.repository import (
    PRODUCTION_CAPABILITIES_PATH,
    PRODUCTION_REACH_PROCESS_PATH,
    PRODUCTION_REACH_TASK_PATH,
    load_production_reach_artifacts,
)


def candidate_record() -> dict:
    record = json.loads(PRODUCTION_REACH_TASK_PATH.read_text(encoding="utf-8"))
    record["task_id"] = "reach_green_cap/v3"
    record["version"] = 3
    return record


def capability_catalog() -> CapabilityCatalog:
    return CapabilityCatalog.from_json(
        PRODUCTION_CAPABILITIES_PATH.read_text(encoding="utf-8")
    )


def process_specs(*, task_id: str = "reach_green_cap/v3") -> dict[str, ProcessSpecV1]:
    record = json.loads(PRODUCTION_REACH_PROCESS_PATH.read_text(encoding="utf-8"))
    record["task_id"] = task_id
    process_spec = ProcessSpecV1.from_record(record)
    return {process_spec.process_spec_id: process_spec}


def authoring_brief(record: dict | None = None) -> PublicAuthoringBrief:
    proposal = record or candidate_record()
    parent = load_production_reach_artifacts()
    return PublicAuthoringBrief.from_record(
        {
            "schema_version": 1,
            "brief_id": "reach_green_cap/v3/brief/v1",
            "task_id": proposal["task_id"],
            "family_id": proposal["family_id"],
            "objective": "Author the next public reach curriculum variant.",
            "source_curriculum_snapshot_fingerprint": "8" * 64,
            "gap_targets": [
                {
                    "unit_id": "reach_green_cap/frontier/v1",
                    "assessment_fingerprint": "7" * 64,
                    "status": "frontier",
                    "gap_code": "instruction_robustness",
                }
            ],
            "parent_task_spec_fingerprint": parent.task.fingerprint,
            "parent_task_oracle_fingerprint": parent.contract.oracle_fingerprint,
            "allowed_change_paths": [
                "$.task_id",
                "$.validation_examples",
                "$.version",
            ],
            "required_curriculum_prerequisite_ids": [],
            "public_constraints": ["Keep reward profiles fixed."],
        }
    )


def authoring_intent(
    record: dict | None = None,
    *,
    catalog: CapabilityCatalog | None = None,
    process_registry: dict[str, ProcessSpecV1] | None = None,
    brief: PublicAuthoringBrief | None = None,
) -> AuthoringIntent:
    proposal = record or candidate_record()
    capabilities = catalog or capability_catalog()
    bound_process_specs = process_registry or process_specs()
    bound_brief = brief or authoring_brief(proposal)
    return AuthoringIntent.from_record(
        {
            "schema_version": 2,
            "task_id": proposal["task_id"],
            "task_version": proposal["version"],
            "family_id": proposal["family_id"],
            "audit_commitment": proposal["instructions"]["audit"],
            "allowed_adapter_ids": [proposal["environment"]["adapter_id"]],
            "allowed_event_schema_ids": [proposal["event_schema_id"]],
            "allowed_process_spec_ids": [proposal["process_spec_id"]],
            "capability_catalog_fingerprint": capabilities.fingerprint,
            "compiler_policy_id": DEFAULT_TASK_COMPILER_POLICY.policy_id,
            "compiler_policy_fingerprint": (
                DEFAULT_TASK_COMPILER_POLICY.fingerprint
            ),
            "property_policy_id": DEFAULT_PROPERTY_VALIDATION_POLICY.policy_id,
            "property_policy_fingerprint": (
                DEFAULT_PROPERTY_VALIDATION_POLICY.fingerprint
            ),
            "authoring_policy_id": DEFAULT_AUTHORING_POLICY.policy_id,
            "authoring_policy_fingerprint": DEFAULT_AUTHORING_POLICY.fingerprint,
            "authoring_brief_id": bound_brief.brief_id,
            "authoring_brief_fingerprint": bound_brief.fingerprint,
            "process_spec_fingerprints": {
                process_spec_id: process_spec.fingerprint
                for process_spec_id, process_spec in bound_process_specs.items()
            },
            "process_compiler_policy_id": (
                DEFAULT_PROCESS_COMPILER_POLICY.policy_id
            ),
            "process_compiler_policy_fingerprint": (
                DEFAULT_PROCESS_COMPILER_POLICY.fingerprint
            ),
            "preauthoring_audit_plan_fingerprint": "3" * 64,
        }
    )


def response_text(record: dict | None = None) -> str:
    return json.dumps(
        record or candidate_record(),
        sort_keys=True,
        separators=(",", ":"),
    )


def candidate_attempt(raw: str, intent: AuthoringIntent) -> ProposalAttempt:
    return ProposalAttempt.record_response(
        raw,
        intent=intent,
        proposer_id="openai/task_proposer/v1",
        model_id="openai/gpt-5.4",
        template_fingerprint="c" * 64,
        request_fingerprint="d" * 64,
        status="candidate",
    )


class TestStaticCandidateValidation(unittest.TestCase):
    def setUp(self) -> None:
        self.catalog = capability_catalog()
        self.record = candidate_record()
        self.parent = load_production_reach_artifacts()
        self.brief = authoring_brief(self.record)
        self.intent = authoring_intent(
            self.record,
            catalog=self.catalog,
            brief=self.brief,
        )
        self.process_specs = process_specs()

    def validate(self, record: dict | None = None):
        return validate_static_candidate(
            response_text(record or self.record),
            intent=self.intent,
            brief=self.brief,
            parent_task=self.parent.task,
            parent_contract=self.parent.contract,
            catalog=self.catalog,
            process_specs=self.process_specs,
        )

    def test_valid_candidate_compiles_and_passes_properties(self) -> None:
        assessment = self.validate()

        self.assertTrue(assessment.accepted)
        self.assertEqual(assessment.issues, ())
        self.assertEqual(assessment.task_spec.task_id, self.intent.task_id)
        self.assertTrue(assessment.property_report.accepted)
        self.assertEqual(assessment.process_spec.task_id, self.intent.task_id)
        self.assertEqual(
            assessment.process_contract.task_contract_fingerprint,
            assessment.task_contract.fingerprint,
        )
        self.assertEqual(len(assessment.task_contract.oracle_fingerprint), 64)

    def test_process_spec_must_compile_against_exact_candidate_task(self) -> None:
        v2_process_specs = process_specs(task_id="reach_green_cap/v2")
        v2_bound_intent = authoring_intent(
            self.record,
            catalog=self.catalog,
            process_registry=v2_process_specs,
            brief=self.brief,
        )
        mismatched = validate_static_candidate(
            response_text(self.record),
            intent=v2_bound_intent,
            brief=self.brief,
            parent_task=self.parent.task,
            parent_contract=self.parent.contract,
            catalog=self.catalog,
            process_specs=v2_process_specs,
        )
        exact = self.validate()

        self.assertEqual(
            [(item.code, item.path) for item in mismatched.issues],
            [("compile_rejected", "$.environment")],
        )
        self.assertFalse(mismatched.accepted)
        self.assertTrue(exact.accepted)
        self.assertEqual(exact.process_spec.task_id, "reach_green_cap/v3")
        self.assertEqual(
            exact.process_contract.process_compiler_policy_fingerprint,
            DEFAULT_PROCESS_COMPILER_POLICY.fingerprint,
        )

    def test_json_and_schema_failures_are_public_and_deterministic(self) -> None:
        malformed = validate_static_candidate(
            '{"task_id":"x/v1",',
            intent=self.intent,
            brief=self.brief,
            parent_task=self.parent.task,
            parent_contract=self.parent.contract,
            catalog=self.catalog,
            process_specs=self.process_specs,
        )
        duplicate = validate_static_candidate(
            '{"schema_version":2,"schema_version":2}',
            intent=self.intent,
            brief=self.brief,
            parent_task=self.parent.task,
            parent_contract=self.parent.contract,
            catalog=self.catalog,
            process_specs=self.process_specs,
        )
        schema = validate_static_candidate(
            '{"schema_version":2}',
            intent=self.intent,
            brief=self.brief,
            parent_task=self.parent.task,
            parent_contract=self.parent.contract,
            catalog=self.catalog,
            process_specs=self.process_specs,
        )

        self.assertEqual(
            [(item.code, item.path) for item in malformed.issues],
            [("invalid_json", "$")],
        )
        self.assertEqual(
            [(item.code, item.path) for item in duplicate.issues],
            [("invalid_json", "$")],
        )
        self.assertEqual(
            [(item.code, item.path) for item in schema.issues],
            [("schema_invalid", "$")],
        )
        self.assertFalse(malformed.accepted)
        self.assertIsNone(malformed.task_contract)

    def test_identity_and_audit_are_locked_before_compilation(self) -> None:
        changed = copy.deepcopy(self.record)
        changed["task_id"] = "reach_green_cap/v4"
        changed["version"] = 4
        changed["family_id"] = "different_family"
        changed["instructions"]["audit"] = {
            "count": 99,
            "sha256": "a" * 64,
        }

        assessment = self.validate(changed)

        self.assertEqual(
            [(item.code, item.path) for item in assessment.issues],
            [
                ("audit_commitment_locked", "$.instructions.audit"),
                ("identity_locked", "$.family_id"),
                ("identity_locked", "$.task_id"),
                ("identity_locked", "$.version"),
            ],
        )

    def test_adapter_event_and_process_allowlists_are_enforced(self) -> None:
        changed = copy.deepcopy(self.record)
        changed["environment"]["adapter_id"] = "untrusted/adapter/v1"
        changed["event_schema_id"] = "untrusted/events/v1"
        changed["process_spec_id"] = "untrusted/process/v1"

        assessment = self.validate(changed)

        self.assertEqual(
            [(item.code, item.path) for item in assessment.issues],
            [
                ("adapter_not_allowed", "$.environment.adapter_id"),
                ("event_schema_not_allowed", "$.event_schema_id"),
                ("process_spec_not_allowed", "$.process_spec_id"),
            ],
        )

    def test_public_brief_prevents_unauthorized_reward_and_prerequisite_changes(self) -> None:
        changed = copy.deepcopy(self.record)
        changed["reward_profiles"][0]["terms"][0]["weight"] = 0.5
        changed["curriculum_prerequisite_ids"] = ["reach_green_cap/v2"]

        assessment = self.validate(changed)

        self.assertEqual(
            [(item.code, item.path) for item in assessment.issues],
            [("brief_violation", "$")],
        )
        self.assertNotIn("reward", assessment.issues[0].message.lower())

    def test_compile_rejection_never_exposes_internal_exception_text(self) -> None:
        secret = "sealed-internal-compiler-detail"
        with patch(
            "rexpolicy.tasking.authoring_candidate.compile_task_contract",
            side_effect=RuntimeError(secret),
        ):
            assessment = self.validate()

        self.assertEqual(
            [(item.code, item.path) for item in assessment.issues],
            [("compile_rejected", "$.environment")],
        )
        self.assertNotIn(secret, repr(assessment))
        self.assertNotIn(secret, assessment.issues[0].message)

    def test_property_rejection_is_collapsed_to_one_safe_issue(self) -> None:
        changed = copy.deepcopy(self.record)
        changed["validation_examples"] = [changed["validation_examples"][0]]

        assessment = self.validate(changed)

        self.assertEqual(
            [(item.code, item.path) for item in assessment.issues],
            [("property_rejected", "$.validation_examples")],
        )
        self.assertIsNone(assessment.property_report)

    def test_all_trusted_dependency_fingerprints_are_required(self) -> None:
        drifted_catalog_record = self.catalog.to_record()
        drifted_catalog_record["adapters"][0]["max_episode_control_steps"] += 1
        drifted_catalog = CapabilityCatalog.from_record(drifted_catalog_record)
        drifted_compiler = replace(
            DEFAULT_TASK_COMPILER_POLICY,
            max_goal_hold_steps=(
                DEFAULT_TASK_COMPILER_POLICY.max_goal_hold_steps - 1
            ),
        )
        drifted_property = replace(
            DEFAULT_PROPERTY_VALIDATION_POLICY,
            max_examples=DEFAULT_PROPERTY_VALIDATION_POLICY.max_examples - 1,
        )
        drifted_authoring = replace(
            DEFAULT_AUTHORING_POLICY,
            max_response_bytes=DEFAULT_AUTHORING_POLICY.max_response_bytes - 1,
        )

        invocations = (
            {"catalog": drifted_catalog},
            {"catalog": self.catalog, "compiler_policy": drifted_compiler},
            {"catalog": self.catalog, "property_policy": drifted_property},
            {"catalog": self.catalog, "authoring_policy": drifted_authoring},
        )
        for overrides in invocations:
            with self.subTest(overrides=tuple(overrides)):
                with self.assertRaisesRegex(
                    TrustedAuthoringBindingError,
                    "^Trusted authoring dependency binding mismatch$",
                ):
                    validate_static_candidate(
                        response_text(self.record),
                        intent=self.intent,
                        brief=self.brief,
                        parent_task=self.parent.task,
                        parent_contract=self.parent.contract,
                        process_specs=self.process_specs,
                        **overrides,
                    )


class TestStaticCandidateBundle(unittest.TestCase):
    def setUp(self) -> None:
        self.catalog = capability_catalog()
        self.record = candidate_record()
        self.raw = response_text(self.record)
        self.parent = load_production_reach_artifacts()
        self.brief = authoring_brief(self.record)
        self.intent = authoring_intent(
            self.record,
            catalog=self.catalog,
            brief=self.brief,
        )
        self.process_specs = process_specs()
        self.assessment = validate_static_candidate(
            self.raw,
            intent=self.intent,
            brief=self.brief,
            parent_task=self.parent.task,
            parent_contract=self.parent.contract,
            catalog=self.catalog,
            process_specs=self.process_specs,
        )
        self.attempt = candidate_attempt(self.raw, self.intent)
        self.bundle = self.assessment.to_bundle(self.attempt)

    def test_bundle_is_deterministic_immutable_and_explicitly_unadmitted(self) -> None:
        record = self.bundle.to_record()
        restored = StaticCandidateBundle.from_record(
            record,
            expected_fingerprint=self.bundle.fingerprint,
            intent=self.intent,
            brief=self.brief,
            parent_task=self.parent.task,
            parent_contract=self.parent.contract,
            attempt=self.attempt,
            catalog=self.catalog,
            process_specs=self.process_specs,
        )

        self.assertEqual(restored, self.bundle)
        self.assertEqual(restored.fingerprint, self.bundle.fingerprint)
        self.assertEqual(record["intent_fingerprint"], self.intent.fingerprint)
        self.assertEqual(record["attempt_fingerprint"], self.attempt.fingerprint)
        self.assertEqual(record["candidate_state"], "quarantined_static")
        self.assertEqual(record["admission_status"], "not_admitted")
        self.assertNotIn("admitted", record)
        json.loads(self.bundle.to_json())
        with self.assertRaises(TypeError):
            self.bundle.task_spec["task_id"] = "evil/v1"

    def test_attempt_must_bind_exact_raw_response_and_intent(self) -> None:
        wrong = candidate_attempt(self.raw + " ", self.intent)
        with self.assertRaisesRegex(ValueError, "attempt binding mismatch"):
            StaticCandidateBundle.from_record(
                self.bundle.to_record(),
                attempt=wrong,
            )
        with self.assertRaisesRegex(ValueError, "does not match"):
            self.assessment.to_bundle(wrong)

    def test_nested_artifact_and_outer_fingerprint_tampering_is_rejected(self) -> None:
        nested = self.bundle.to_record()
        nested["task_spec"]["family_id"] = "tampered_family"
        with self.assertRaisesRegex(ValueError, "TaskSpec fingerprint mismatch"):
            StaticCandidateBundle.from_record(nested)

        outer = self.bundle.to_record()
        outer["attempt_fingerprint"] = "f" * 64
        with self.assertRaisesRegex(ValueError, "bundle fingerprint mismatch"):
            StaticCandidateBundle.from_record(
                outer,
                expected_fingerprint=self.bundle.fingerprint,
            )

        state = self.bundle.to_record()
        state["candidate_state"] = "active"
        with self.assertRaisesRegex(ValueError, "remain quarantined_static"):
            StaticCandidateBundle.from_record(state)

    def test_semantic_rebuild_rejects_coordinated_artifact_tampering(self) -> None:
        record = self.bundle.to_record()
        record["task_contract"]["required_metrics"] = []
        from rexpolicy.tasking.canonical import canonical_fingerprint

        record["task_contract_fingerprint"] = canonical_fingerprint(
            record["task_contract"]
        )
        record["property_report"]["contract_fingerprint"] = record[
            "task_contract_fingerprint"
        ]
        record["property_report_fingerprint"] = canonical_fingerprint(
            record["property_report"]
        )
        record["process_contract"]["task_contract_fingerprint"] = record[
            "task_contract_fingerprint"
        ]
        record["process_contract_fingerprint"] = canonical_fingerprint(
            record["process_contract"]
        )

        with self.assertRaisesRegex(ValueError, "rebuilt artifacts mismatch"):
            StaticCandidateBundle.from_record(
                record,
                intent=self.intent,
                brief=self.brief,
                parent_task=self.parent.task,
                parent_contract=self.parent.contract,
                attempt=self.attempt,
                catalog=self.catalog,
                process_specs=self.process_specs,
            )


if __name__ == "__main__":
    unittest.main()
