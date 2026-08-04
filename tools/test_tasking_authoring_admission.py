"""Tests for authored-task certification, admission, and activation."""

from __future__ import annotations

import copy
import importlib.util
import json
import sys
import unittest
from dataclasses import replace
from pathlib import Path

_AUDIT_PATH = Path(__file__).parents[1] / "rexpolicy" / "flywheel" / "audit.py"
_AUDIT_SPEC = importlib.util.spec_from_file_location(
    "rexpolicy_standalone_audit_contracts",
    _AUDIT_PATH,
)
if _AUDIT_SPEC is None or _AUDIT_SPEC.loader is None:
    raise RuntimeError("Unable to load standalone audit contracts")
_AUDIT = importlib.util.module_from_spec(_AUDIT_SPEC)
sys.modules[_AUDIT_SPEC.name] = _AUDIT
_AUDIT_SPEC.loader.exec_module(_AUDIT)
CapabilityClaimPolicy = _AUDIT.CapabilityClaimPolicy
ClaimEpisodeResult = _AUDIT.ClaimEpisodeResult
SealedAuditSuite = _AUDIT.SealedAuditSuite
evaluate_capability_claim = _AUDIT.evaluate_capability_claim
from rexpolicy.tasking.authoring import (
    DEFAULT_AUTHORING_POLICY,
    AuthoringIntent,
    ProposalAttempt,
)
from rexpolicy.tasking.authoring_admission import (
    ACTIVE,
    ADMISSION_READY,
    ADMITTED_DORMANT,
    QUARANTINED_STATIC,
    GenerationBoundaryBinding,
    LifecycleAuthority,
    LifecycleEvent,
    PreAuthoringAuditPlan,
    RuntimeCertification,
    SealedAuditCertification,
    StaticCandidateVerificationContext,
    transition_candidate_lifecycle,
    validate_audit_suite_for_candidate,
    validate_lifecycle_chain,
)
from rexpolicy.tasking.authoring_candidate import validate_static_candidate
from rexpolicy.tasking.authoring_brief import PublicAuthoringBrief
from rexpolicy.tasking.capabilities import CapabilityCatalog
from rexpolicy.tasking.contract import DEFAULT_TASK_COMPILER_POLICY
from rexpolicy.tasking.process import ProcessSpecV1
from rexpolicy.tasking.process_contract import DEFAULT_PROCESS_COMPILER_POLICY
from rexpolicy.tasking.property_validation import (
    DEFAULT_PROPERTY_VALIDATION_POLICY,
)
from rexpolicy.tasking.repository import (
    PRODUCTION_CAPABILITIES_PATH,
    PRODUCTION_REACH_PROCESS_PATH,
    PRODUCTION_REACH_TASK_PATH,
    load_production_reach_artifacts,
)


def _candidate_bundle(preauthoring_audit_plan_fingerprint: str):
    proposal = json.loads(PRODUCTION_REACH_TASK_PATH.read_text(encoding="utf-8"))
    proposal["task_id"] = "reach_green_cap/v3"
    proposal["version"] = 3
    catalog = CapabilityCatalog.from_json(
        PRODUCTION_CAPABILITIES_PATH.read_text(encoding="utf-8")
    )
    process_record = json.loads(
        PRODUCTION_REACH_PROCESS_PATH.read_text(encoding="utf-8")
    )
    process_record["task_id"] = proposal["task_id"]
    process_spec = ProcessSpecV1.from_record(process_record)
    process_specs = {process_spec.process_spec_id: process_spec}
    parent = load_production_reach_artifacts()
    brief = PublicAuthoringBrief.from_record(
        {
            "schema_version": 1,
            "brief_id": "reach_green_cap/v3/brief/v1",
            "task_id": proposal["task_id"],
            "family_id": proposal["family_id"],
            "objective": "Author the next safe reach task from a public gap.",
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
            "allowed_change_paths": ["$.task_id", "$.version"],
            "required_curriculum_prerequisite_ids": [],
            "public_constraints": [
                "Keep success and safety independent from shaping reward."
            ],
        }
    )
    intent = AuthoringIntent.from_record(
        {
            "schema_version": 2,
            "task_id": proposal["task_id"],
            "task_version": proposal["version"],
            "family_id": proposal["family_id"],
            "audit_commitment": proposal["instructions"]["audit"],
            "allowed_adapter_ids": [proposal["environment"]["adapter_id"]],
            "allowed_event_schema_ids": [proposal["event_schema_id"]],
            "allowed_process_spec_ids": [proposal["process_spec_id"]],
            "capability_catalog_fingerprint": catalog.fingerprint,
            "compiler_policy_id": DEFAULT_TASK_COMPILER_POLICY.policy_id,
            "compiler_policy_fingerprint": DEFAULT_TASK_COMPILER_POLICY.fingerprint,
            "property_policy_id": DEFAULT_PROPERTY_VALIDATION_POLICY.policy_id,
            "property_policy_fingerprint": (
                DEFAULT_PROPERTY_VALIDATION_POLICY.fingerprint
            ),
            "authoring_policy_id": DEFAULT_AUTHORING_POLICY.policy_id,
            "authoring_policy_fingerprint": DEFAULT_AUTHORING_POLICY.fingerprint,
            "authoring_brief_id": "reach_green_cap/v3/brief/v1",
            "authoring_brief_fingerprint": brief.fingerprint,
            "process_spec_fingerprints": {
                process_spec.process_spec_id: process_spec.fingerprint
            },
            "process_compiler_policy_id": DEFAULT_PROCESS_COMPILER_POLICY.policy_id,
            "process_compiler_policy_fingerprint": (
                DEFAULT_PROCESS_COMPILER_POLICY.fingerprint
            ),
            "preauthoring_audit_plan_fingerprint": (
                preauthoring_audit_plan_fingerprint
            ),
        }
    )
    raw = json.dumps(proposal, sort_keys=True, separators=(",", ":"))
    assessment = validate_static_candidate(
        raw,
        intent=intent,
        brief=brief,
        parent_task=parent.task,
        parent_contract=parent.contract,
        catalog=catalog,
        process_specs=process_specs,
    )
    attempt = ProposalAttempt.record_response(
        raw,
        intent=intent,
        proposer_id="openai/task_proposer/v1",
        model_id="openai/gpt-5.4",
        template_fingerprint="c" * 64,
        request_fingerprint="d" * 64,
        status="candidate",
    )
    context = StaticCandidateVerificationContext(
        attempt=attempt,
        brief=brief,
        parent_task=parent.task,
        parent_contract=parent.contract,
        catalog=catalog,
        process_specs=process_specs,
    )
    return assessment.to_bundle(attempt), intent, context


class AdmissionFixture(unittest.TestCase):
    def setUp(self) -> None:
        self.policy = CapabilityClaimPolicy.production_v1()
        task = json.loads(PRODUCTION_REACH_TASK_PATH.read_text(encoding="utf-8"))
        task["task_id"] = "reach_green_cap/v3"
        task["version"] = 3
        audit = task["instructions"]["audit"]
        self.plan = PreAuthoringAuditPlan.from_record(
            {
                "schema_version": 1,
                "audit_plan_id": "reach_green_cap/v3/audit_plan/v1",
                "task_id": task["task_id"],
                "audit_instruction_count": audit["count"],
                "audit_instruction_sha256": audit["sha256"],
                "episode_recipe_sha256": "e" * 64,
                "independent_training_seeds": [11, 22, 33],
                "episodes_per_training_seed": 256,
                "candidate_count": 1,
                "claim_policy_id": self.policy.claim_spec_id,
                "claim_policy_fingerprint": self.policy.fingerprint,
            },
            policy=self.policy,
        )
        self.candidate, self.intent, self.verification_context = _candidate_bundle(
            self.plan.fingerprint
        )
        self.suite = SealedAuditSuite(
            schema_version=1,
            audit_suite_id="reach_green_cap/v3/audit/v1",
            task_id=task["task_id"],
            task_spec_fingerprint=self.candidate.task_spec_fingerprint,
            claim_spec_id=self.policy.claim_spec_id,
            audit_instruction_count=audit["count"],
            audit_instruction_sha256=audit["sha256"],
            episodes_per_training_seed=256,
            episode_recipe_sha256="e" * 64,
            independent_training_seeds=(11, 22, 33),
            candidate_count=1,
        )
        self.episodes = tuple(
            ClaimEpisodeResult(
                training_seed=training_seed,
                reset_seed=episode,
                diffusion_seed=10_000 + episode,
                success=True,
                safety_failure=False,
                baseline_safety_failure=False,
                invalid_action=False,
            )
            for training_seed in self.suite.independent_training_seeds
            for episode in range(self.suite.episodes_per_training_seed)
        )
        self.report = evaluate_capability_claim(
            suite=self.suite,
            policy=self.policy,
            episodes=list(self.episodes),
        )
        self.runtime = RuntimeCertification.issue(
            certification_id="reach_green_cap/v3/runtime_cert/v1",
            candidate=self.candidate,
            runtime_policy_id="isaac/runtime_cert/v1",
            runtime_policy_fingerprint="2" * 64,
            runtime_evidence_sha256="3" * 64,
            accepted=True,
            reasons=(),
            intent=self.intent,
            verification_context=self.verification_context,
        )
        self.audit_cert = SealedAuditCertification.issue(
            certification_id="reach_green_cap/v3/audit_cert/v1",
            candidate=self.candidate,
            plan=self.plan,
            suite=self.suite,
            policy=self.policy,
            report=self.report,
            intent=self.intent,
            verification_context=self.verification_context,
            episode_results=self.episodes,
        )

    def evidence(self) -> dict:
        return {
            "runtime_certification": self.runtime,
            "sealed_audit_certification": self.audit_cert,
            "runtime_policy_id": self.runtime.runtime_policy_id,
            "runtime_policy_fingerprint": self.runtime.runtime_policy_fingerprint,
            "suite": self.suite,
            "policy": self.policy,
            "report": self.report,
            "episode_results": self.episodes,
        }

    def lifecycle(self):
        quarantined = transition_candidate_lifecycle(
            candidate=self.candidate,
            plan=self.plan,
            intent=self.intent,
            verification_context=self.verification_context,
            target_state=QUARANTINED_STATIC,
        )
        ready = transition_candidate_lifecycle(
            candidate=self.candidate,
            plan=self.plan,
            intent=self.intent,
            verification_context=self.verification_context,
            history=(quarantined,),
            target_state=ADMISSION_READY,
            **self.evidence(),
        )
        admission_authority = LifecycleAuthority.issue(
            authority_id="operator/admission/v1",
            action="admit",
            candidate=self.candidate,
            predecessor_event=ready,
            decision_sha256="4" * 64,
        )
        dormant = transition_candidate_lifecycle(
            candidate=self.candidate,
            plan=self.plan,
            intent=self.intent,
            verification_context=self.verification_context,
            history=(quarantined, ready),
            target_state=ADMITTED_DORMANT,
            authority=admission_authority,
            **self.evidence(),
        )
        activation_authority = LifecycleAuthority.issue(
            authority_id="operator/activation/v1",
            action="activate",
            candidate=self.candidate,
            predecessor_event=dormant,
            decision_sha256="5" * 64,
        )
        boundary = GenerationBoundaryBinding.issue(
            boundary_id="generation/gen_0002",
            candidate=self.candidate,
            admitted_event=dormant,
            generation_manifest_sha256="6" * 64,
        )
        active = transition_candidate_lifecycle(
            candidate=self.candidate,
            plan=self.plan,
            intent=self.intent,
            verification_context=self.verification_context,
            history=(quarantined, ready, dormant),
            target_state=ACTIVE,
            authority=activation_authority,
            generation_boundary=boundary,
            **self.evidence(),
        )
        return quarantined, ready, dormant, active


class TestPreAuthoringAuditPlan(AdmissionFixture):
    def test_plan_is_complete_policy_bound_and_round_trips(self) -> None:
        restored = PreAuthoringAuditPlan.from_record(
            self.plan.to_record(),
            policy=self.policy,
            expected_fingerprint=self.plan.fingerprint,
        )

        self.assertEqual(restored, self.plan)
        self.assertEqual(restored.candidate_count, 1)
        self.assertEqual(restored.independent_training_seeds, (11, 22, 33))
        self.assertEqual(restored.claim_policy_fingerprint, self.policy.fingerprint)

    def test_plan_rejects_policy_drift_search_and_tampering(self) -> None:
        search = self.plan.to_record()
        search["candidate_count"] = 2
        with self.assertRaisesRegex(ValueError, "K=1"):
            PreAuthoringAuditPlan.from_record(search, policy=self.policy)

        weak = self.plan.to_record()
        weak["independent_training_seeds"] = [11]
        with self.assertRaisesRegex(ValueError, "too few"):
            PreAuthoringAuditPlan.from_record(weak, policy=self.policy)

        drifted_policy = replace(self.policy, min_episodes_per_seed=257)
        with self.assertRaisesRegex(ValueError, "exact claim policy"):
            PreAuthoringAuditPlan.from_record(
                self.plan.to_record(),
                policy=drifted_policy,
            )

        changed = self.plan.to_record()
        changed["episode_recipe_sha256"] = "f" * 64
        with self.assertRaisesRegex(ValueError, "fingerprint mismatch"):
            PreAuthoringAuditPlan.from_record(
                changed,
                expected_fingerprint=self.plan.fingerprint,
            )

        unknown = self.plan.to_record()
        unknown["audit_instructions"] = ["secret"]
        with self.assertRaisesRegex(ValueError, "unknown fields"):
            PreAuthoringAuditPlan.from_record(unknown)


class TestSealedSuiteCandidateBinding(AdmissionFixture):
    def test_suite_matches_candidate_plan_and_policy_exactly(self) -> None:
        binding = validate_audit_suite_for_candidate(
            suite=self.suite,
            candidate=self.candidate,
            plan=self.plan,
            policy=self.policy,
            intent=self.intent,
            verification_context=self.verification_context,
        )

        self.assertEqual(
            binding.candidate.candidate_fingerprint,
            self.candidate.fingerprint,
        )
        self.assertEqual(binding.audit_plan_fingerprint, self.plan.fingerprint)
        self.assertEqual(binding.audit_suite_fingerprint, self.suite.fingerprint)
        self.assertEqual(binding.claim_policy_fingerprint, self.policy.fingerprint)

    def test_every_sealed_suite_axis_is_cross_checked(self) -> None:
        mismatches = (
            replace(self.suite, task_id="other_task/v1"),
            replace(self.suite, task_spec_fingerprint="f" * 64),
            replace(
                self.suite,
                audit_instruction_count=self.suite.audit_instruction_count + 1,
            ),
            replace(self.suite, audit_instruction_sha256="f" * 64),
            replace(self.suite, episode_recipe_sha256="f" * 64),
            replace(self.suite, independent_training_seeds=(11, 22, 44)),
            replace(self.suite, episodes_per_training_seed=257),
        )
        for suite in mismatches:
            with self.subTest(field=suite):
                with self.assertRaises(ValueError):
                    validate_audit_suite_for_candidate(
                        suite=suite,
                        candidate=self.candidate,
                        plan=self.plan,
                        policy=self.policy,
                        intent=self.intent,
                        verification_context=self.verification_context,
                    )

        wrong_intent_record = self.intent.to_record()
        wrong_intent_record["preauthoring_audit_plan_fingerprint"] = "f" * 64
        wrong_intent = AuthoringIntent.from_record(wrong_intent_record)
        with self.assertRaisesRegex(ValueError, "binding mismatch|does not bind"):
            validate_audit_suite_for_candidate(
                suite=self.suite,
                candidate=self.candidate,
                plan=self.plan,
                policy=self.policy,
                intent=wrong_intent,
                verification_context=self.verification_context,
            )

        incomplete_candidate = self.candidate.to_record()
        del incomplete_candidate["property_report"]
        with self.assertRaisesRegex(ValueError, "missing fields"):
            validate_audit_suite_for_candidate(
                suite=self.suite,
                candidate=incomplete_candidate,
                plan=self.plan,
                policy=self.policy,
                intent=self.intent,
                verification_context=self.verification_context,
            )


class TestCandidateCertifications(AdmissionFixture):
    def test_certificates_are_exact_candidate_bound_and_strict(self) -> None:
        runtime = RuntimeCertification.from_record(
            self.runtime.to_record(),
            candidate=self.candidate,
            expected_fingerprint=self.runtime.fingerprint,
        )
        audit = SealedAuditCertification.from_record(
            self.audit_cert.to_record(),
            candidate=self.candidate,
            plan=self.plan,
            suite=self.suite,
            policy=self.policy,
            report=self.report,
            intent=self.intent,
            verification_context=self.verification_context,
            episode_results=self.episodes,
            expected_fingerprint=self.audit_cert.fingerprint,
        )

        self.assertEqual(runtime, self.runtime)
        self.assertEqual(audit, self.audit_cert)
        self.assertTrue(runtime.accepted)
        self.assertTrue(audit.accepted)

        other = self.candidate.to_record()
        other["attempt_fingerprint"] = "f" * 64
        with self.assertRaisesRegex(ValueError, "candidate binding mismatch"):
            RuntimeCertification.from_record(self.runtime.to_record(), candidate=other)
        with self.assertRaisesRegex(ValueError, "candidate binding mismatch"):
            SealedAuditCertification.from_record(
                self.audit_cert.to_record(),
                candidate=other,
                plan=self.plan,
                intent=self.intent,
            )

    def test_acceptance_and_reasons_cannot_disagree(self) -> None:
        invalid_runtime = self.runtime.to_record()
        invalid_runtime["reasons"] = ["runtime failed"]
        with self.assertRaisesRegex(ValueError, "inconsistent"):
            RuntimeCertification.from_record(invalid_runtime)

        invalid_audit = self.audit_cert.to_record()
        invalid_audit["accepted"] = False
        with self.assertRaisesRegex(ValueError, "inconsistent"):
            SealedAuditCertification.from_record(invalid_audit)

        rejected_episodes = tuple(
            replace(episode, success=False) for episode in self.episodes
        )
        rejected_report = evaluate_capability_claim(
            suite=self.suite,
            policy=self.policy,
            episodes=list(rejected_episodes),
        )
        rejected = SealedAuditCertification.issue(
            certification_id="reach_green_cap/v3/audit_cert/rejected/v1",
            candidate=self.candidate,
            plan=self.plan,
            suite=self.suite,
            policy=self.policy,
            report=rejected_report,
            intent=self.intent,
            verification_context=self.verification_context,
            episode_results=rejected_episodes,
        )
        self.assertFalse(rejected.accepted)
        self.assertEqual(rejected.reasons, rejected_report.reasons)

    def test_report_or_certificate_tampering_is_rejected(self) -> None:
        wrong_report = replace(self.report, task_spec_fingerprint="f" * 64)
        with self.assertRaisesRegex(ValueError, "report binding mismatch"):
            SealedAuditCertification.issue(
                certification_id="reach_green_cap/v3/audit_cert/v2",
                candidate=self.candidate,
                plan=self.plan,
                suite=self.suite,
                policy=self.policy,
                report=wrong_report,
                intent=self.intent,
                verification_context=self.verification_context,
                episode_results=self.episodes,
            )

        falsified_metrics = replace(
            self.report,
            seed_metrics=tuple(
                replace(metrics, success_lower_bound=0.0)
                for metrics in self.report.seed_metrics
            ),
        )
        with self.assertRaisesRegex(ValueError, "report binding mismatch"):
            SealedAuditCertification.issue(
                certification_id="reach_green_cap/v3/audit_cert/falsified/v1",
                candidate=self.candidate,
                plan=self.plan,
                suite=self.suite,
                policy=self.policy,
                report=falsified_metrics,
                intent=self.intent,
                verification_context=self.verification_context,
                episode_results=self.episodes,
            )

        changed = self.audit_cert.to_record()
        changed["capability_claim_report_fingerprint"] = "f" * 64
        with self.assertRaisesRegex(ValueError, "fingerprint mismatch"):
            SealedAuditCertification.from_record(
                changed,
                expected_fingerprint=self.audit_cert.fingerprint,
            )


class TestLifecycle(AdmissionFixture):
    def test_only_complete_authorized_chain_reaches_active(self) -> None:
        events = self.lifecycle()

        self.assertEqual(
            tuple(event.state for event in events),
            (QUARANTINED_STATIC, ADMISSION_READY, ADMITTED_DORMANT, ACTIVE),
        )
        self.assertEqual(tuple(event.sequence for event in events), (0, 1, 2, 3))
        self.assertEqual(events[1].previous_event_sha256, events[0].fingerprint)
        self.assertEqual(
            events[3].generation_boundary.admitted_event_sha256,
            events[2].fingerprint,
        )
        self.assertNotEqual(events[2].authority, events[3].authority)
        self.assertEqual(
            validate_lifecycle_chain(
                events,
                candidate=self.candidate,
                plan=self.plan,
                intent=self.intent,
                verification_context=self.verification_context,
                **self.evidence(),
            ),
            events,
        )

    def test_skips_missing_certificates_and_implicit_authority_fail(self) -> None:
        with self.assertRaisesRegex(ValueError, "skips"):
            transition_candidate_lifecycle(
                candidate=self.candidate,
                plan=self.plan,
                intent=self.intent,
                verification_context=self.verification_context,
                target_state=ADMISSION_READY,
                **self.evidence(),
            )

        quarantined = transition_candidate_lifecycle(
            candidate=self.candidate,
            plan=self.plan,
            intent=self.intent,
            verification_context=self.verification_context,
            target_state=QUARANTINED_STATIC,
        )
        with self.assertRaisesRegex(ValueError, "complete evidence"):
            transition_candidate_lifecycle(
                candidate=self.candidate,
                plan=self.plan,
                intent=self.intent,
                verification_context=self.verification_context,
                history=(quarantined,),
                target_state=ADMISSION_READY,
            )
        ready = transition_candidate_lifecycle(
            candidate=self.candidate,
            plan=self.plan,
            intent=self.intent,
            verification_context=self.verification_context,
            history=(quarantined,),
            target_state=ADMISSION_READY,
            **self.evidence(),
        )
        with self.assertRaisesRegex(ValueError, "explicit authority"):
            transition_candidate_lifecycle(
                candidate=self.candidate,
                plan=self.plan,
                intent=self.intent,
                verification_context=self.verification_context,
                history=(quarantined, ready),
                target_state=ADMITTED_DORMANT,
                **self.evidence(),
            )
        admission_authority = LifecycleAuthority.issue(
            authority_id="operator/admission/v1",
            action="admit",
            candidate=self.candidate,
            predecessor_event=ready,
            decision_sha256="4" * 64,
        )
        dormant = transition_candidate_lifecycle(
            candidate=self.candidate,
            plan=self.plan,
            intent=self.intent,
            verification_context=self.verification_context,
            history=(quarantined, ready),
            target_state=ADMITTED_DORMANT,
            authority=admission_authority,
            **self.evidence(),
        )
        activation_authority = LifecycleAuthority.issue(
            authority_id="operator/activation/v1",
            action="activate",
            candidate=self.candidate,
            predecessor_event=dormant,
            decision_sha256="5" * 64,
        )
        with self.assertRaisesRegex(ValueError, "generation-boundary"):
            transition_candidate_lifecycle(
                candidate=self.candidate,
                plan=self.plan,
                intent=self.intent,
                verification_context=self.verification_context,
                history=(quarantined, ready, dormant),
                target_state=ACTIVE,
                authority=activation_authority,
                **self.evidence(),
            )

    def test_rejected_or_replaced_certificates_cannot_advance(self) -> None:
        quarantined = transition_candidate_lifecycle(
            candidate=self.candidate,
            plan=self.plan,
            intent=self.intent,
            verification_context=self.verification_context,
            target_state=QUARANTINED_STATIC,
        )
        rejected_runtime = RuntimeCertification.issue(
            certification_id="reach_green_cap/v3/runtime_cert/rejected/v1",
            candidate=self.candidate,
            runtime_policy_id="isaac/runtime_cert/v1",
            runtime_policy_fingerprint="2" * 64,
            runtime_evidence_sha256="7" * 64,
            accepted=False,
            reasons=("runtime smoke test failed",),
            intent=self.intent,
            verification_context=self.verification_context,
        )
        rejected_evidence = self.evidence()
        rejected_evidence["runtime_certification"] = rejected_runtime
        with self.assertRaisesRegex(ValueError, "accepted certificates"):
            transition_candidate_lifecycle(
                candidate=self.candidate,
                plan=self.plan,
                intent=self.intent,
                verification_context=self.verification_context,
                history=(quarantined,),
                target_state=ADMISSION_READY,
                **rejected_evidence,
            )

        valid_chain = self.lifecycle()
        valid_quarantined, ready = valid_chain[:2]
        replacement = RuntimeCertification.issue(
            certification_id="reach_green_cap/v3/runtime_cert/v2",
            candidate=self.candidate,
            runtime_policy_id="isaac/runtime_cert/v1",
            runtime_policy_fingerprint="2" * 64,
            runtime_evidence_sha256="8" * 64,
            accepted=True,
            reasons=(),
            intent=self.intent,
            verification_context=self.verification_context,
        )
        replacement_evidence = self.evidence()
        replacement_evidence["runtime_certification"] = replacement
        authority = LifecycleAuthority.issue(
            authority_id="operator/admission/v1",
            action="admit",
            candidate=self.candidate,
            predecessor_event=ready,
            decision_sha256="4" * 64,
        )
        with self.assertRaisesRegex(
            ValueError,
            "replace readiness|certification fingerprint mismatch",
        ):
            transition_candidate_lifecycle(
                candidate=self.candidate,
                plan=self.plan,
                intent=self.intent,
                verification_context=self.verification_context,
                history=(valid_quarantined, ready),
                target_state=ADMITTED_DORMANT,
                authority=authority,
                **replacement_evidence,
            )

    def test_hash_chain_and_event_record_tampering_are_rejected(self) -> None:
        quarantined, ready, _, _ = self.lifecycle()
        changed = ready.to_record()
        changed["previous_event_sha256"] = "f" * 64
        with self.assertRaisesRegex(ValueError, "predecessor mismatch"):
            LifecycleEvent.from_record(
                changed,
                candidate=self.candidate,
                plan=self.plan,
                intent=self.intent,
                verification_context=self.verification_context,
                history=(quarantined,),
                **self.evidence(),
            )

    def test_hand_constructed_predecessor_cannot_skip_the_chain(self) -> None:
        forged = LifecycleEvent(
            schema_version=1,
            candidate_fingerprint=self.candidate.fingerprint,
            audit_plan_fingerprint=self.plan.fingerprint,
            sequence=77,
            previous_event_sha256="f" * 64,
            previous_state=ADMISSION_READY,
            state=ADMITTED_DORMANT,
            runtime_certification_fingerprint=self.runtime.fingerprint,
            sealed_audit_certification_fingerprint=self.audit_cert.fingerprint,
            authority=None,
            generation_boundary=None,
        )
        with self.assertRaisesRegex(ValueError, "begin at QUARANTINED_STATIC"):
            transition_candidate_lifecycle(
                candidate=self.candidate,
                plan=self.plan,
                intent=self.intent,
                verification_context=self.verification_context,
                history=(forged,),
                target_state=ACTIVE,
                **self.evidence(),
            )

        quarantined, ready, _, _ = self.lifecycle()
        changed = ready.to_record()
        changed["state"] = ACTIVE
        with self.assertRaisesRegex(ValueError, "fingerprint mismatch"):
            LifecycleEvent.from_record(
                changed,
                candidate=self.candidate,
                plan=self.plan,
                intent=self.intent,
                verification_context=self.verification_context,
                history=(quarantined,),
                **self.evidence(),
                expected_fingerprint=ready.fingerprint,
            )

        unknown = copy.deepcopy(ready.to_record())
        unknown["mutable_status"] = True
        with self.assertRaisesRegex(ValueError, "unknown fields"):
            LifecycleEvent.from_record(
                unknown,
                candidate=self.candidate,
                plan=self.plan,
                intent=self.intent,
                verification_context=self.verification_context,
                history=(quarantined,),
                **self.evidence(),
            )


if __name__ == "__main__":
    unittest.main()
