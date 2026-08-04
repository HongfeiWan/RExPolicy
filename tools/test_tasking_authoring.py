"""Tests for the bounded automatic-authoring provenance layer."""

from __future__ import annotations

import copy
import hashlib
import json
import unittest

from rexpolicy.tasking.authoring import (
    DEFAULT_AUTHORING_POLICY,
    AuthoringIntent,
    AuthoringPolicy,
    ProposalAttempt,
    normalize_public_issue,
    normalize_public_issues,
)
from rexpolicy.tasking.contract import DEFAULT_TASK_COMPILER_POLICY
from rexpolicy.tasking.property_validation import (
    DEFAULT_PROPERTY_VALIDATION_POLICY,
)


def intent_record() -> dict:
    return {
        "schema_version": 1,
        "task_id": "reach_green_cap/v3",
        "task_version": 3,
        "family_id": "reach_green_cap",
        "audit_commitment": {"count": 19, "sha256": "a" * 64},
        "allowed_adapter_ids": ["groot_newton/reach/v1"],
        "allowed_event_schema_ids": ["groot_newton/reach_events/v1"],
        "allowed_process_spec_ids": ["reach_pregrasp/v1"],
        "capability_catalog_fingerprint": "b" * 64,
        "compiler_policy_id": DEFAULT_TASK_COMPILER_POLICY.policy_id,
        "compiler_policy_fingerprint": DEFAULT_TASK_COMPILER_POLICY.fingerprint,
        "property_policy_id": DEFAULT_PROPERTY_VALIDATION_POLICY.policy_id,
        "property_policy_fingerprint": (
            DEFAULT_PROPERTY_VALIDATION_POLICY.fingerprint
        ),
        "authoring_policy_id": DEFAULT_AUTHORING_POLICY.policy_id,
        "authoring_policy_fingerprint": DEFAULT_AUTHORING_POLICY.fingerprint,
    }


def proposal_record(*, intent: AuthoringIntent | None = None) -> dict:
    bound_intent = intent or AuthoringIntent.from_record(intent_record())
    issue = normalize_public_issue(code="invalid_json", path="$")
    return {
        "schema_version": 1,
        "intent_fingerprint": bound_intent.fingerprint,
        "ordinal": 1,
        "parent_attempt_fingerprint": None,
        "proposer_id": "openai/task_proposer/v1",
        "model_id": "openai/gpt-5.4",
        "template_fingerprint": "c" * 64,
        "request_fingerprint": "d" * 64,
        "raw_response_sha256": "e" * 64,
        "raw_response_bytes": 91,
        "status": "repair_requested",
        "issues": [issue.to_record()],
    }


class TestAuthoringPolicy(unittest.TestCase):
    def test_policy_is_canonical_bounded_and_deeply_detached(self) -> None:
        source = copy.deepcopy(DEFAULT_AUTHORING_POLICY.to_record())
        policy = AuthoringPolicy.from_record(source)
        restored = AuthoringPolicy.from_json(policy.to_json())

        self.assertEqual(restored, policy)
        self.assertEqual(restored.fingerprint, policy.fingerprint)
        self.assertEqual(policy.max_attempts, 3)
        json.loads(policy.to_json())

        source["issue_rules"][0]["allowed_paths"][0] = "$"
        self.assertEqual(
            policy.issue_rules[0].allowed_paths,
            ("$.environment.adapter_id",),
        )
        with self.assertRaises(TypeError):
            policy.issue_rules[0].allowed_paths[0] = "$"  # type: ignore[index]

        excessive = policy.to_record()
        excessive["max_attempts"] = 9
        with self.assertRaisesRegex(ValueError, "trusted maximum 8"):
            AuthoringPolicy.from_record(excessive)

    def test_policy_rules_must_be_sorted_and_public_safe(self) -> None:
        unsorted = DEFAULT_AUTHORING_POLICY.to_record()
        unsorted["issue_rules"][0], unsorted["issue_rules"][1] = (
            unsorted["issue_rules"][1],
            unsorted["issue_rules"][0],
        )
        with self.assertRaisesRegex(ValueError, "codes must be sorted and unique"):
            AuthoringPolicy.from_record(unsorted)

        internal_path = DEFAULT_AUTHORING_POLICY.to_record()
        internal_path["issue_rules"][0]["allowed_paths"] = [
            "$.environment.adapter_id[0]"
        ]
        with self.assertRaisesRegex(ValueError, "normalized public JSON path"):
            AuthoringPolicy.from_record(internal_path)

        diagnostic = DEFAULT_AUTHORING_POLICY.to_record()
        diagnostic["issue_rules"][0]["message"] = "traceback:\nsecret"
        with self.assertRaisesRegex(ValueError, "control characters"):
            AuthoringPolicy.from_record(diagnostic)


class TestAuthoringIntent(unittest.TestCase):
    def test_intent_locks_identity_audit_and_all_policy_fingerprints(self) -> None:
        source = intent_record()
        intent = AuthoringIntent.from_record(source)
        restored = AuthoringIntent.from_json(intent.to_json())

        self.assertEqual(restored, intent)
        self.assertEqual(restored.fingerprint, intent.fingerprint)
        self.assertEqual(intent.task_id, "reach_green_cap/v3")
        self.assertEqual(intent.task_version, 3)
        self.assertEqual(intent.audit_commitment.count, 19)
        self.assertEqual(
            intent.compiler_policy_fingerprint,
            DEFAULT_TASK_COMPILER_POLICY.fingerprint,
        )
        self.assertEqual(
            intent.property_policy_fingerprint,
            DEFAULT_PROPERTY_VALIDATION_POLICY.fingerprint,
        )
        self.assertEqual(
            intent.authoring_policy_fingerprint,
            DEFAULT_AUTHORING_POLICY.fingerprint,
        )

        source["allowed_adapter_ids"][0] = "malicious/adapter/v1"
        source["audit_commitment"]["sha256"] = "f" * 64
        self.assertEqual(
            intent.allowed_adapter_ids,
            ("groot_newton/reach/v1",),
        )
        self.assertEqual(intent.audit_commitment.sha256, "a" * 64)

    def test_intent_rejects_policy_drift_and_identity_drift(self) -> None:
        policy_drift = intent_record()
        policy_drift["authoring_policy_fingerprint"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "policy fingerprint mismatch"):
            AuthoringIntent.from_record(policy_drift)

        wrong_version = intent_record()
        wrong_version["task_version"] = 4
        with self.assertRaisesRegex(ValueError, "suffix must match"):
            AuthoringIntent.from_record(wrong_version)

        unordered = intent_record()
        unordered["allowed_process_spec_ids"] = ["z/v1", "a/v1"]
        with self.assertRaisesRegex(ValueError, "sorted and unique"):
            AuthoringIntent.from_record(unordered)

    def test_intent_cannot_contain_audit_content_or_reward_results(self) -> None:
        leaked_audit = intent_record()
        leaked_audit["sealed_audit_examples"] = [{"secret": True}]
        with self.assertRaisesRegex(
            ValueError,
            "unknown fields: sealed_audit_examples",
        ):
            AuthoringIntent.from_record(leaked_audit)

        reward_result = intent_record()
        reward_result["reward_result"] = 1.0
        with self.assertRaisesRegex(ValueError, "unknown fields: reward_result"):
            AuthoringIntent.from_record(reward_result)

    def test_intent_duplicate_json_keys_are_rejected(self) -> None:
        payload = json.dumps(intent_record())
        duplicate = payload.replace(
            '"schema_version": 1',
            '"schema_version": 1, "schema_version": 1',
            1,
        )
        with self.assertRaisesRegex(ValueError, "Duplicate JSON key"):
            AuthoringIntent.from_json(duplicate)


class TestPublicIssues(unittest.TestCase):
    def test_public_issues_use_only_policy_owned_messages(self) -> None:
        issue = normalize_public_issue(
            code="identity_locked",
            path="$.task_id",
        )
        self.assertEqual(
            issue.message,
            "Keep the task identity fixed by the authoring intent.",
        )

        with self.assertRaisesRegex(ValueError, "code is outside"):
            normalize_public_issue(code="internal_traceback", path="$")
        with self.assertRaisesRegex(ValueError, "path is outside"):
            normalize_public_issue(
                code="identity_locked",
                path="$.validation_examples",
            )

        tampered = issue.to_record()
        tampered["message"] = "The sealed score was 0.51"
        with self.assertRaisesRegex(ValueError, "policy-owned public message"):
            type(issue).from_record(tampered, "$issue")

    def test_public_issues_are_deduplicated_and_canonically_sorted(self) -> None:
        issues = normalize_public_issues(
            [
                ("schema_invalid", "$"),
                ("identity_locked", "$.version"),
                ("schema_invalid", "$"),
            ]
        )
        self.assertEqual(
            tuple((item.code, item.path) for item in issues),
            (("identity_locked", "$.version"), ("schema_invalid", "$")),
        )


class TestProposalAttempt(unittest.TestCase):
    def setUp(self) -> None:
        self.intent = AuthoringIntent.from_record(intent_record())
        self.template_fingerprint = "c" * 64
        self.request_fingerprint = "d" * 64

    def record(
        self,
        response: str | bytes,
        *,
        status: str,
        issues=(),
        parent: ProposalAttempt | None = None,
    ) -> ProposalAttempt:
        return ProposalAttempt.record_response(
            response,
            intent=self.intent,
            proposer_id="openai/task_proposer/v1",
            model_id="openai/gpt-5.4",
            template_fingerprint=self.template_fingerprint,
            request_fingerprint=self.request_fingerprint,
            status=status,
            issues=issues,
            parent=parent,
        )

    def test_attempt_records_hash_only_provenance(self) -> None:
        raw = '{"task_id":"reach_green_cap/v3","note":"你好"}'
        attempt = self.record(raw, status="candidate")
        expected = raw.encode("utf-8")

        self.assertEqual(attempt.raw_response_bytes, len(expected))
        self.assertEqual(
            attempt.raw_response_sha256,
            hashlib.sha256(expected).hexdigest(),
        )
        self.assertEqual(attempt.ordinal, 1)
        self.assertIsNone(attempt.parent_attempt_fingerprint)
        self.assertNotIn("raw_response", attempt.to_record())
        self.assertNotIn("audit", attempt.to_json())
        self.assertNotIn("reward_result", attempt.to_json())

        restored = ProposalAttempt.from_json(attempt.to_json())
        self.assertEqual(restored, attempt)
        self.assertEqual(restored.fingerprint, attempt.fingerprint)

    def test_repair_lineage_is_hash_linked_and_bounded(self) -> None:
        first = self.record(
            "not json",
            status="repair_requested",
            issues=(("invalid_json", "$"),),
        )
        second = self.record(
            "{}",
            status="repair_requested",
            issues=(("schema_invalid", "$"),),
            parent=first,
        )
        third = self.record("{\"schema_version\":2}", status="candidate", parent=second)

        self.assertEqual(second.ordinal, 2)
        self.assertEqual(second.parent_attempt_fingerprint, first.fingerprint)
        self.assertEqual(third.ordinal, 3)
        self.assertEqual(third.parent_attempt_fingerprint, second.fingerprint)
        with self.assertRaisesRegex(ValueError, "Only a repair-requested"):
            self.record("{}", status="candidate", parent=third)

        at_limit = second.to_record()
        at_limit["ordinal"] = 3
        at_limit["parent_attempt_fingerprint"] = first.fingerprint
        with self.assertRaisesRegex(ValueError, "cannot request repair"):
            ProposalAttempt.from_record(at_limit)

    def test_lineage_cannot_cross_intents(self) -> None:
        first = self.record(
            "bad",
            status="repair_requested",
            issues=(("invalid_json", "$"),),
        )
        changed = intent_record()
        changed["task_id"] = "reach_green_cap/v4"
        changed["task_version"] = 4
        other_intent = AuthoringIntent.from_record(changed)
        with self.assertRaisesRegex(ValueError, "different authoring intent"):
            ProposalAttempt.record_response(
                "{}",
                intent=other_intent,
                proposer_id="openai/task_proposer/v1",
                model_id="openai/gpt-5.4",
                template_fingerprint=self.template_fingerprint,
                request_fingerprint=self.request_fingerprint,
                status="candidate",
                parent=first,
            )

    def test_attempt_status_issue_and_size_invariants_fail_closed(self) -> None:
        candidate_with_issue = proposal_record(intent=self.intent)
        candidate_with_issue["status"] = "candidate"
        with self.assertRaisesRegex(ValueError, "Only candidate attempts"):
            ProposalAttempt.from_record(candidate_with_issue)

        with self.assertRaisesRegex(TypeError, "text or bytes"):
            self.record(17, status="candidate")  # type: ignore[arg-type]

        oversized = b"x" * (DEFAULT_AUTHORING_POLICY.max_response_bytes + 1)
        rejected = self.record(
            oversized,
            status="rejected",
            issues=(("response_too_large", "$"),),
        )
        self.assertGreater(
            rejected.raw_response_bytes,
            DEFAULT_AUTHORING_POLICY.max_response_bytes,
        )

        missing_size_issue = rejected.to_record()
        missing_size_issue["issues"] = [
            normalize_public_issue(code="invalid_json", path="$").to_record()
        ]
        with self.assertRaisesRegex(ValueError, "response size"):
            ProposalAttempt.from_record(missing_size_issue)

    def test_attempt_rejects_extra_results_and_duplicate_json_keys(self) -> None:
        extra = proposal_record(intent=self.intent)
        extra["sealed_audit_score"] = 0.9
        extra["reward_result"] = 1.0
        with self.assertRaisesRegex(
            ValueError,
            "unknown fields: reward_result, sealed_audit_score",
        ):
            ProposalAttempt.from_record(extra)

        payload = json.dumps(proposal_record(intent=self.intent))
        duplicate = payload.replace(
            '"ordinal": 1',
            '"ordinal": 1, "ordinal": 2',
            1,
        )
        with self.assertRaisesRegex(ValueError, "Duplicate JSON key"):
            ProposalAttempt.from_json(duplicate)


if __name__ == "__main__":
    unittest.main()
