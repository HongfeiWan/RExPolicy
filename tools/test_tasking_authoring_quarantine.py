"""Tests for immutable automatic-authoring quarantine storage."""

from __future__ import annotations

import hashlib
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from rexpolicy.tasking.authoring import AuthoringIntent, ProposalAttempt
from rexpolicy.tasking.authoring_quarantine import (
    load_authoring_quarantine,
    write_authoring_quarantine,
)
from rexpolicy.tasking.authoring_runner import ProposerInvocation
from tools.test_tasking_authoring import intent_record


def _artifacts(raw: bytes = b'{"schema_version":2,"task_id":"reach_green_cap/v3"}'):
    intent = AuthoringIntent.from_record(intent_record())
    request = b'{"attempt":1}'
    request_sha256 = hashlib.sha256(request).hexdigest()
    attempt = ProposalAttempt.record_response(
        raw,
        intent=intent,
        proposer_id="openai/task_proposer/v1",
        model_id="openai/gpt-5.4",
        template_fingerprint="c" * 64,
        request_fingerprint=request_sha256,
        status="candidate",
    )
    invocation = ProposerInvocation(
        schema_version=1,
        command_id="rexpolicy/fake_proposer/v1",
        command_sha256="1" * 64,
        execution_policy_id="rexpolicy/proposer_execution/v1",
        execution_policy_sha256="2" * 64,
        request_sha256=request_sha256,
        request_bytes=len(request),
        status="completed",
        exit_code=0,
        stdout_sha256=hashlib.sha256(raw).hexdigest(),
        stdout_bytes=len(raw),
        stdout_truncated=False,
        stderr_sha256=hashlib.sha256(b"").hexdigest(),
        stderr_bytes=0,
        stderr_truncated=False,
    )
    candidate = {
        "schema_version": 1,
        "intent_fingerprint": intent.fingerprint,
        "proposal_attempt_fingerprint": attempt.fingerprint,
        "raw_response_sha256": attempt.raw_response_sha256,
        "state": "quarantined_static",
    }
    return raw, attempt, invocation, candidate


class TestAuthoringQuarantine(unittest.TestCase):
    def test_round_trip_verifies_the_complete_hash_chain(self) -> None:
        raw, attempt, invocation, candidate = _artifacts()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write = write_authoring_quarantine(
                run_dir=root,
                raw_response=raw,
                attempt=attempt,
                invocation=invocation,
                static_candidate_bundle=candidate,
            )
            loaded = load_authoring_quarantine(
                run_dir=root,
                intent_fingerprint=attempt.intent_fingerprint,
                ordinal=attempt.ordinal,
                attempt_fingerprint=attempt.fingerprint,
            )

            self.assertTrue(write.created)
            self.assertEqual(write.descriptor_sha256, loaded.descriptor.fingerprint)
            self.assertEqual(loaded.raw_response, raw)
            self.assertEqual(loaded.attempt, attempt)
            self.assertEqual(loaded.invocation, invocation)
            self.assertEqual(loaded.static_candidate_bundle, candidate)
            entry = root / write.relative_path
            self.assertEqual(
                set(item.name for item in entry.iterdir()),
                {
                    "descriptor.json",
                    "raw-response.bin",
                    "proposal-attempt.json",
                    "proposer-invocation.json",
                    "static-candidate-bundle.json",
                },
            )
            attempt_text = (entry / "proposal-attempt.json").read_text(
                encoding="ascii"
            )
            self.assertNotIn(raw.decode("ascii"), attempt_text)
            self.assertNotIn('"raw_response":', attempt_text)

    def test_identical_retry_is_idempotent(self) -> None:
        raw, attempt, invocation, candidate = _artifacts()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = write_authoring_quarantine(
                run_dir=root,
                raw_response=raw,
                attempt=attempt,
                invocation=invocation,
                static_candidate_bundle=candidate,
            )
            second = write_authoring_quarantine(
                run_dir=root,
                raw_response=raw,
                attempt=attempt,
                invocation=invocation,
                static_candidate_bundle=candidate,
            )

            self.assertTrue(first.created)
            self.assertFalse(second.created)
            self.assertEqual(first.relative_path, second.relative_path)
            self.assertEqual(first.descriptor_sha256, second.descriptor_sha256)

    def test_tampering_and_conflicting_overwrite_are_rejected(self) -> None:
        raw, attempt, invocation, candidate = _artifacts()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write = write_authoring_quarantine(
                run_dir=root,
                raw_response=raw,
                attempt=attempt,
                invocation=invocation,
                static_candidate_bundle=candidate,
            )
            entry = root / write.relative_path
            (entry / "raw-response.bin").write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "size drifted|hash drifted"):
                load_authoring_quarantine(
                    run_dir=root,
                    intent_fingerprint=attempt.intent_fingerprint,
                    ordinal=attempt.ordinal,
                    attempt_fingerprint=attempt.fingerprint,
                )
            with self.assertRaisesRegex(FileExistsError, "immutable verification"):
                write_authoring_quarantine(
                    run_dir=root,
                    raw_response=raw,
                    attempt=attempt,
                    invocation=invocation,
                    static_candidate_bundle=candidate,
                )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_authoring_quarantine(
                run_dir=root,
                raw_response=raw,
                attempt=attempt,
                invocation=invocation,
                static_candidate_bundle=candidate,
            )
            conflicting = replace(invocation, stderr_sha256="f" * 64)
            with self.assertRaisesRegex(FileExistsError, "different content"):
                write_authoring_quarantine(
                    run_dir=root,
                    raw_response=raw,
                    attempt=attempt,
                    invocation=conflicting,
                    static_candidate_bundle=candidate,
                )

    def test_fault_before_descriptor_never_publishes_a_complete_entry(self) -> None:
        raw, attempt, invocation, candidate = _artifacts()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            def fail(checkpoint: str) -> None:
                if checkpoint == "before:descriptor.json":
                    raise RuntimeError("injected crash")

            with self.assertRaisesRegex(RuntimeError, "injected crash"):
                write_authoring_quarantine(
                    run_dir=root,
                    raw_response=raw,
                    attempt=attempt,
                    invocation=invocation,
                    static_candidate_bundle=candidate,
                    _fault_injector=fail,
                )
            descriptors = list(root.rglob("descriptor.json"))
            self.assertEqual(descriptors, [])
            with self.assertRaisesRegex(ValueError, "descriptor is absent"):
                load_authoring_quarantine(
                    run_dir=root,
                    intent_fingerprint=attempt.intent_fingerprint,
                    ordinal=attempt.ordinal,
                    attempt_fingerprint=attempt.fingerprint,
                )

            entry = next(root.rglob(attempt.fingerprint))
            (entry / (".raw-response.bin.partial-" + "a" * 32)).write_bytes(
                b"orphaned private staging bytes"
            )
            resumed = write_authoring_quarantine(
                run_dir=root,
                raw_response=raw,
                attempt=attempt,
                invocation=invocation,
                static_candidate_bundle=candidate,
            )
            self.assertTrue(resumed.created)

    def test_bundle_and_invocation_must_bind_the_same_attempt(self) -> None:
        raw, attempt, invocation, candidate = _artifacts()
        with tempfile.TemporaryDirectory() as directory:
            wrong_candidate = dict(candidate)
            wrong_candidate["proposal_attempt_fingerprint"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "does not bind"):
                write_authoring_quarantine(
                    run_dir=Path(directory),
                    raw_response=raw,
                    attempt=attempt,
                    invocation=invocation,
                    static_candidate_bundle=wrong_candidate,
                )

            wrong_invocation = replace(invocation, request_sha256="0" * 64)
            with self.assertRaisesRegex(ValueError, "request does not bind"):
                write_authoring_quarantine(
                    run_dir=Path(directory),
                    raw_response=raw,
                    attempt=attempt,
                    invocation=wrong_invocation,
                    static_candidate_bundle=candidate,
                )

    def test_internal_symlink_cannot_escape_the_archive_root(self) -> None:
        raw, attempt, invocation, candidate = _artifacts()
        with tempfile.TemporaryDirectory() as directory, tempfile.TemporaryDirectory() as outside:
            root = Path(directory)
            archive = root / "archive" / "authoring"
            archive.mkdir(parents=True)
            (archive / "quarantine").symlink_to(Path(outside), target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "not a real directory"):
                write_authoring_quarantine(
                    run_dir=root,
                    raw_response=raw,
                    attempt=attempt,
                    invocation=invocation,
                    static_candidate_bundle=candidate,
                )
            self.assertEqual(list(Path(outside).iterdir()), [])


if __name__ == "__main__":
    unittest.main()
