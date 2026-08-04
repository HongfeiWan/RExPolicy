"""Integration tests for bounded automatic TaskSpec authoring sessions."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

from rexpolicy.tasking.authoring import (
    DEFAULT_AUTHORING_POLICY,
    AuthoringIntent,
)
from rexpolicy.tasking.authoring_orchestrator import (
    PublicAuthoringBrief,
    run_automatic_authoring,
)
from rexpolicy.tasking.authoring_runner import ProposerCommand
from rexpolicy.tasking.canonical import canonical_json
from rexpolicy.tasking.contract import DEFAULT_TASK_COMPILER_POLICY
from rexpolicy.tasking.process import ProcessSpecV1
from rexpolicy.tasking.process_contract import DEFAULT_PROCESS_COMPILER_POLICY
from rexpolicy.tasking.property_validation import (
    DEFAULT_PROPERTY_VALIDATION_POLICY,
)
from rexpolicy.tasking.repository import (
    PRODUCTION_CAPABILITIES_PATH,
    PRODUCTION_REACH_TASK_PATH,
    load_capability_catalog,
    load_production_reach_artifacts,
)
from tools.test_tasking_authoring_brief import _snapshot

_PROCESS_V2_PATH = (
    PRODUCTION_REACH_TASK_PATH.parents[1]
    / "process_specs"
    / "reach_pregrasp.v2.json"
)
_SECRET_SENTINEL = "SEALED_AUDIT_TEXT_MUST_NEVER_REENTER_REQUEST"


def _candidate_record() -> dict:
    record = json.loads(PRODUCTION_REACH_TASK_PATH.read_text(encoding="utf-8"))
    record["task_id"] = "reach_green_cap/v3"
    record["version"] = 3
    record["process_spec_id"] = "reach_pregrasp/v2"
    return record


def _brief_record(parent, snapshot) -> dict:
    frontier = snapshot.assessments[0]
    return {
        "schema_version": 1,
        "brief_id": "reach_green_cap/v3/brief/v1",
        "task_id": "reach_green_cap/v3",
        "family_id": "reach_green_cap",
        "objective": "Author the next safe reach curriculum task from the public gap.",
        "source_curriculum_snapshot_fingerprint": snapshot.fingerprint,
        "gap_targets": [
            {
                "unit_id": "reach_green_cap/frontier/v1",
                "assessment_fingerprint": frontier.fingerprint,
                "status": frontier.status,
                "gap_code": "instruction_robustness",
            }
        ],
        "parent_task_spec_fingerprint": parent.task.fingerprint,
        "parent_task_oracle_fingerprint": parent.contract.oracle_fingerprint,
        "allowed_change_paths": [
            "$.instructions",
            "$.process_spec_id",
            "$.task_id",
            "$.validation_examples",
            "$.version",
        ],
        "required_curriculum_prerequisite_ids": [],
        "public_constraints": [
            "Keep success and safety independent from shaping reward.",
            "Use the pre-approved reach process contract.",
        ],
    }


def _write_script(directory: Path, body: str) -> Path:
    path = directory / "fake_authoring_proposer.py"
    path.write_text(body, encoding="utf-8")
    return path


def _command(script: Path, request_log: Path) -> ProposerCommand:
    return ProposerCommand.bind(
        command_id="rexpolicy/fake_task_proposer/v1",
        argv=(sys.executable, str(script), str(request_log)),
    )


def _script_body(valid_payload: str, first_mode: str) -> str:
    return f"""\
import json
import sys

raw = sys.stdin.buffer.read()
with open(sys.argv[1], "ab") as output:
    output.write(raw + b"\\n")
request = json.loads(raw)
ordinal = request["attempt"]["ordinal"]
valid = {valid_payload!r}
if ordinal == 1 and {first_mode!r} == "invalid":
    sys.stderr.write({_SECRET_SENTINEL!r})
    sys.stdout.write("not-json")
elif ordinal == 1 and {first_mode!r} == "oversized":
    sys.stdout.write("x" * ({DEFAULT_AUTHORING_POLICY.max_response_bytes + 4096}))
elif ordinal == 1 and {first_mode!r} == "nonzero":
    sys.stdout.write(valid)
    sys.stderr.write({_SECRET_SENTINEL!r})
    raise SystemExit(7)
elif {first_mode!r} == "always_invalid":
    sys.stdout.write("not-json")
else:
    sys.stdout.write(valid)
"""


class TestAutomaticAuthoringOrchestrator(unittest.TestCase):
    def setUp(self) -> None:
        self.catalog = load_capability_catalog(PRODUCTION_CAPABILITIES_PATH)
        self.parent = load_production_reach_artifacts()
        self.candidate = _candidate_record()
        self.process_spec = ProcessSpecV1.from_json(
            _PROCESS_V2_PATH.read_text(encoding="utf-8")
        )
        self.process_specs = {self.process_spec.process_spec_id: self.process_spec}
        self.curriculum_snapshot = _snapshot((("frontier", 51),))
        self.brief = PublicAuthoringBrief.from_record(
            _brief_record(self.parent, self.curriculum_snapshot)
        )
        self.intent = AuthoringIntent.from_record(
            {
                "schema_version": 2,
                "task_id": self.candidate["task_id"],
                "task_version": self.candidate["version"],
                "family_id": self.candidate["family_id"],
                "audit_commitment": self.candidate["instructions"]["audit"],
                "allowed_adapter_ids": [
                    self.candidate["environment"]["adapter_id"]
                ],
                "allowed_event_schema_ids": [self.candidate["event_schema_id"]],
                "allowed_process_spec_ids": [self.candidate["process_spec_id"]],
                "capability_catalog_fingerprint": self.catalog.fingerprint,
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
                "authoring_brief_id": self.brief.brief_id,
                "authoring_brief_fingerprint": self.brief.fingerprint,
                "process_spec_fingerprints": {
                    self.process_spec.process_spec_id: self.process_spec.fingerprint
                },
                "process_compiler_policy_id": (
                    DEFAULT_PROCESS_COMPILER_POLICY.policy_id
                ),
                "process_compiler_policy_fingerprint": (
                    DEFAULT_PROCESS_COMPILER_POLICY.fingerprint
                ),
                "preauthoring_audit_plan_fingerprint": "9" * 64,
            }
        )

    def _run(self, directory: Path, first_mode: str):
        request_log = directory / "requests.jsonl"
        script = _write_script(
            directory,
            _script_body(canonical_json(self.candidate), first_mode),
        )
        run = run_automatic_authoring(
            intent=self.intent,
            brief=self.brief,
            curriculum_snapshot=self.curriculum_snapshot,
            catalog=self.catalog,
            process_specs=self.process_specs,
            parent_task=self.parent.task,
            parent_contract=self.parent.contract,
            command=_command(script, request_log),
            model_id="openai/fake-task-model/v1",
            quarantine_dir=directory / "run",
        )
        requests = [
            json.loads(line)
            for line in request_log.read_text(encoding="utf-8").splitlines()
        ]
        return run, requests

    def test_invalid_then_valid_is_repaired_and_stops_in_static_quarantine(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            run, requests = self._run(Path(raw_directory), "invalid")

        self.assertEqual(run.result.final_state, "quarantined_static")
        self.assertEqual(len(run.attempts), 2)
        self.assertEqual(run.attempts[0].status, "repair_requested")
        self.assertEqual(run.attempts[0].issues[0].code, "invalid_json")
        self.assertEqual(run.attempts[1].status, "candidate")
        self.assertEqual(
            run.attempts[1].parent_attempt_fingerprint,
            run.attempts[0].fingerprint,
        )
        self.assertEqual(
            {attempt.session_fingerprint for attempt in run.attempts},
            {run.result.session_fingerprint},
        )
        self.assertIsNotNone(run.static_candidate_bundle)
        assert run.static_candidate_bundle is not None
        self.assertEqual(
            run.static_candidate_bundle.process_spec_fingerprint,
            self.process_spec.fingerprint,
        )
        self.assertEqual(run.static_candidate_bundle.admission_status, "not_admitted")

        self.assertEqual(len(requests), 2)
        self.assertEqual(requests[0]["attempt"]["public_issues"], [])
        self.assertEqual(
            requests[1]["attempt"]["public_issues"][0]["code"],
            "invalid_json",
        )
        serialized = canonical_json(requests)
        for forbidden in (
            _SECRET_SENTINEL,
            "success_count",
            "confidence_lower_bound",
            "stderr",
            "traceback",
        ):
            self.assertNotIn(forbidden, serialized)
        self.assertNotIn("raw_response", serialized)
        self.assertFalse(
            requests[1]["output_contract"]["automatic_admission"]
        )
        self.assertFalse(
            requests[1]["output_contract"]["automatic_activation"]
        )

    def test_attempt_limit_returns_exhausted_with_complete_lineage(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            run, requests = self._run(Path(raw_directory), "always_invalid")

        self.assertEqual(run.result.final_state, "exhausted")
        self.assertIsNone(run.static_candidate_bundle)
        self.assertEqual(len(run.attempts), DEFAULT_AUTHORING_POLICY.max_attempts)
        self.assertEqual(len(requests), DEFAULT_AUTHORING_POLICY.max_attempts)
        self.assertEqual(run.attempts[-1].status, "rejected")
        for ordinal, attempt in enumerate(run.attempts, start=1):
            self.assertEqual(attempt.ordinal, ordinal)
            if ordinal > 1:
                self.assertEqual(
                    attempt.parent_attempt_fingerprint,
                    run.attempts[ordinal - 2].fingerprint,
                )

    def test_oversized_capture_is_explicitly_incomplete_and_repairable(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            run, requests = self._run(Path(raw_directory), "oversized")

        self.assertEqual(run.result.final_state, "quarantined_static")
        self.assertEqual(len(run.attempts), 2)
        first = run.attempts[0]
        self.assertEqual(first.issues[0].code, "response_too_large")
        self.assertEqual(first.status, "repair_requested")
        self.assertFalse(first.response_complete)
        self.assertGreater(
            first.raw_response_bytes,
            DEFAULT_AUTHORING_POLICY.max_response_bytes,
        )
        self.assertEqual(
            requests[1]["attempt"]["public_issues"][0]["code"],
            "response_too_large",
        )

    def test_nonzero_exit_cannot_smuggle_a_valid_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            run, requests = self._run(Path(raw_directory), "nonzero")

        self.assertEqual(len(run.attempts), 2)
        self.assertEqual(run.attempts[0].issues[0].code, "proposer_failed")
        self.assertEqual(run.attempts[0].status, "repair_requested")
        self.assertEqual(run.result.final_state, "quarantined_static")
        self.assertNotIn(_SECRET_SENTINEL, canonical_json(requests[1]))

    def test_v1_intent_and_brief_drift_are_rejected_before_launch(self) -> None:
        # A v1 record has no brief or process-content bindings.
        from tools.test_tasking_authoring import intent_record

        legacy = AuthoringIntent.from_record(intent_record())
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            script = _write_script(directory, "print('{}')\n")
            with self.assertRaisesRegex(ValueError, "AuthoringIntent v2"):
                run_automatic_authoring(
                    intent=legacy,
                    brief=self.brief,
                    curriculum_snapshot=self.curriculum_snapshot,
                    catalog=self.catalog,
                    process_specs=self.process_specs,
                    parent_task=self.parent.task,
                    parent_contract=self.parent.contract,
                    command=_command(script, directory / "requests.jsonl"),
                    model_id="openai/fake-task-model/v1",
                    quarantine_dir=directory / "run",
                )

        changed = self.brief.to_record()
        changed["objective"] = "A different unauthorized objective."
        drifted = PublicAuthoringBrief.from_record(changed)
        with self.assertRaisesRegex(ValueError, "fingerprint binding drifted"):
            drifted.validate_against(self.intent)


if __name__ == "__main__":
    unittest.main()
