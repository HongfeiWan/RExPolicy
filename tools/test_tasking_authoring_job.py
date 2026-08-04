"""Durability and exactly-one-chain tests for automatic authoring jobs."""

from __future__ import annotations

import json
import multiprocessing
import sys
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from unittest import mock

from rexpolicy.tasking.authoring import DEFAULT_AUTHORING_POLICY, AuthoringIntent
from rexpolicy.tasking.authoring_job import (
    AmbiguousAuthoringAttemptError,
    AuthoringJobDriftError,
)
from rexpolicy.tasking.authoring_orchestrator import run_automatic_authoring
from rexpolicy.tasking.authoring_runner import (
    DEFAULT_BUBBLEWRAP_SANDBOX_PROFILE,
    DEFAULT_PROPOSER_EXECUTION_POLICY,
    BubblewrapProposerCommand,
    ProposerCommand,
)
from rexpolicy.tasking.canonical import canonical_json
from rexpolicy.tasking.contract import DEFAULT_TASK_COMPILER_POLICY
from rexpolicy.tasking.process import ProcessSpecV1
from rexpolicy.tasking.process_contract import DEFAULT_PROCESS_COMPILER_POLICY
from rexpolicy.tasking.property_validation import DEFAULT_PROPERTY_VALIDATION_POLICY
from rexpolicy.tasking.repository import (
    PRODUCTION_CAPABILITIES_PATH,
    PRODUCTION_REACH_TASK_PATH,
    load_capability_catalog,
    load_production_reach_artifacts,
)
from tools.test_tasking_authoring_orchestrator import (
    _brief_record,
    _candidate_record,
)
from tools.test_tasking_authoring_brief import _snapshot
from rexpolicy.tasking.authoring_brief import PublicAuthoringBrief

_PROCESS_V2_PATH = (
    PRODUCTION_REACH_TASK_PATH.parents[1]
    / "process_specs"
    / "reach_pregrasp.v2.json"
)
_BUBBLEWRAP_AVAILABLE = sys.platform.startswith("linux") and Path(
    "/usr/bin/bwrap"
).is_file()


class DurableAuthoringFixture(unittest.TestCase):
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
                "compiler_policy_fingerprint": DEFAULT_TASK_COMPILER_POLICY.fingerprint,
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
                "process_compiler_policy_id": DEFAULT_PROCESS_COMPILER_POLICY.policy_id,
                "process_compiler_policy_fingerprint": (
                    DEFAULT_PROCESS_COMPILER_POLICY.fingerprint
                ),
                "preauthoring_audit_plan_fingerprint": "9" * 64,
            }
        )

    def _prepare(
        self,
        directory: Path,
        *,
        first_invalid: bool = False,
        always_invalid: bool = False,
        delay: float = 0.0,
    ):
        request_log = directory / "requests.jsonl"
        script = directory / "durable_proposer.py"
        script.write_text(
            "import json\n"
            "import sys\n"
            "import time\n"
            "raw = sys.stdin.buffer.read()\n"
            "with open(sys.argv[1], 'ab') as output:\n"
            "    output.write(raw + b'\\n')\n"
            f"time.sleep({delay!r})\n"
            "request = json.loads(raw)\n"
            f"first_invalid = {first_invalid!r}\n"
            f"always_invalid = {always_invalid!r}\n"
            "if always_invalid or (first_invalid and request['attempt']['ordinal'] == 1):\n"
            "    sys.stdout.write('not-json')\n"
            "else:\n"
            f"    sys.stdout.write({canonical_json(self.candidate)!r})\n",
            encoding="utf-8",
        )
        command = ProposerCommand.bind(
            command_id="rexpolicy/durable_fake_proposer/v1",
            argv=(sys.executable, str(script), str(request_log)),
        )
        return command, request_log

    def _run(
        self,
        directory: Path,
        command: ProposerCommand | BubblewrapProposerCommand,
        *,
        model_id: str = "openai/fake-task-model/v1",
        environment=None,
        execution_policy=DEFAULT_PROPOSER_EXECUTION_POLICY,
    ):
        return run_automatic_authoring(
            intent=self.intent,
            brief=self.brief,
            curriculum_snapshot=self.curriculum_snapshot,
            catalog=self.catalog,
            process_specs=self.process_specs,
            parent_task=self.parent.task,
            parent_contract=self.parent.contract,
            command=command,
            model_id=model_id,
            quarantine_dir=directory / "run",
            environment=environment,
            execution_policy=execution_policy,
        )


class TestDurableAuthoringJob(DurableAuthoringFixture):
    def test_terminal_rerun_is_idempotent_and_does_not_reinvoke(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            command, request_log = self._prepare(directory)
            first = self._run(directory, command)
            second = self._run(directory, command)

            self.assertEqual(first.result.fingerprint, second.result.fingerprint)
            self.assertEqual(first.result.final_state, "quarantined_static")
            self.assertEqual(len(request_log.read_text(encoding="utf-8").splitlines()), 1)
            manifest = directory / "run" / "authoring-job" / "manifest.json"
            self.assertTrue(manifest.is_file())
            manifest_record = json.loads(manifest.read_text(encoding="ascii"))
            self.assertEqual(
                manifest_record["source_curriculum_snapshot_fingerprint"],
                self.brief.source_curriculum_snapshot_fingerprint,
            )
            self.assertEqual(
                manifest_record["curriculum_snapshot"],
                self.curriculum_snapshot.to_record(),
            )
            self.assertEqual(manifest_record["public_brief"], self.brief.to_record())

    @unittest.skipUnless(
        _BUBBLEWRAP_AVAILABLE,
        "Durable sandbox test requires Linux and /usr/bin/bwrap",
    )
    def test_durable_sandbox_cannot_read_its_sealed_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            manifest_path = (
                directory / "run" / "authoring-job" / "manifest.json"
            )
            script = directory / "sandboxed_durable_proposer.py"
            script.write_text(
                "import os\n"
                "import sys\n"
                "manifest = os.environ['SEALED_MANIFEST_PATH']\n"
                "try:\n"
                "    leaked = open(manifest, 'rb').read()\n"
                "except OSError:\n"
                f"    sys.stdout.write({canonical_json(self.candidate)!r})\n"
                "else:\n"
                "    sys.stderr.buffer.write(b'MANIFEST_LEAK:' + leaked)\n"
                "    sys.stdout.write('not-json')\n",
                encoding="utf-8",
            )
            trusted_wrapper = ProposerCommand.bind(
                command_id="rexpolicy/sandboxed_durable_fake_proposer/v1",
                argv=(sys.executable, str(script)),
            )
            command = BubblewrapProposerCommand.bind(
                proposer_command=trusted_wrapper,
                sandbox_profile=DEFAULT_BUBBLEWRAP_SANDBOX_PROFILE,
            )
            execution_policy = replace(
                DEFAULT_PROPOSER_EXECUTION_POLICY,
                allowed_environment_names=("SEALED_MANIFEST_PATH",),
            )
            run = self._run(
                directory,
                command,
                environment={"SEALED_MANIFEST_PATH": str(manifest_path)},
                execution_policy=execution_policy,
            )

            self.assertEqual(run.result.final_state, "quarantined_static")
            self.assertEqual(run.invocations[0].status, "completed")
            self.assertEqual(
                run.invocations[0].command_sha256,
                command.fingerprint,
            )
            manifest_record = json.loads(manifest_path.read_text(encoding="ascii"))
            self.assertEqual(
                manifest_record["curriculum_snapshot"],
                self.curriculum_snapshot.to_record(),
            )
            self.assertEqual(
                manifest_record["source_curriculum_snapshot_fingerprint"],
                self.curriculum_snapshot.fingerprint,
            )
            self.assertEqual(manifest_record["command"], command.to_record())
            self.assertEqual(
                manifest_record["command_fingerprint"],
                command.fingerprint,
            )
            self.assertEqual(
                manifest_record["proposer_binding"]["command_fingerprint"],
                command.fingerprint,
            )
            self.assertEqual(
                manifest_record["command"]["sandbox_profile_sha256"],
                command.sandbox_profile.fingerprint,
            )
            self.assertEqual(
                manifest_record["command"]["sandbox_profile"]["network_policy"],
                "deny",
            )
            self.assertNotIn(str(manifest_path), canonical_json(manifest_record))

    def test_invalid_environment_fails_before_any_journal_record(self) -> None:
        cases = (
            ({"WRONG_ENV": "value"}, "allowlist"),
            ({"EXPECTED_ENV": "invalid\x00value"}, "value is invalid"),
        )
        for environment, message in cases:
            with self.subTest(message=message), tempfile.TemporaryDirectory() as raw:
                directory = Path(raw)
                command, _ = self._prepare(directory)
                execution_policy = replace(
                    DEFAULT_PROPOSER_EXECUTION_POLICY,
                    allowed_environment_names=("EXPECTED_ENV",),
                )
                with self.assertRaisesRegex(ValueError, message):
                    self._run(
                        directory,
                        command,
                        environment=environment,
                        execution_policy=execution_policy,
                    )
                run_dir = directory / "run"
                self.assertFalse((run_dir / "authoring-job" / "manifest.json").exists())
                self.assertEqual(list(run_dir.rglob("launch.json")), [])

    def test_oversized_request_fails_before_launch_marker(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            command, _ = self._prepare(directory)
            execution_policy = replace(
                DEFAULT_PROPOSER_EXECUTION_POLICY,
                max_request_bytes=1,
            )
            for _ in range(2):
                with self.assertRaisesRegex(ValueError, "request exceeds"):
                    self._run(
                        directory,
                        command,
                        execution_policy=execution_policy,
                    )
            run_dir = directory / "run"
            self.assertTrue(
                (run_dir / "authoring-job" / "manifest.json").is_file()
            )
            self.assertEqual(list(run_dir.rglob("launch.json")), [])

    def test_unrelated_curriculum_snapshot_is_rejected_before_launch(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            command, request_log = self._prepare(directory)
            unrelated = _snapshot((("hard", 0),))
            with self.assertRaisesRegex(
                AuthoringJobDriftError,
                "exact curriculum snapshot",
            ):
                run_automatic_authoring(
                    intent=self.intent,
                    brief=self.brief,
                    curriculum_snapshot=unrelated,
                    catalog=self.catalog,
                    process_specs=self.process_specs,
                    parent_task=self.parent.task,
                    parent_contract=self.parent.contract,
                    command=command,
                    model_id="openai/fake-task-model/v1",
                    quarantine_dir=directory / "run",
                )
            self.assertFalse(request_log.exists())

    def test_cross_thread_lock_allows_only_one_proposer_launch(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            command, request_log = self._prepare(directory, delay=0.2)
            barrier = threading.Barrier(2)

            def invoke():
                barrier.wait()
                return self._run(directory, command)

            with ThreadPoolExecutor(max_workers=2) as executor:
                results = list(executor.map(lambda _: invoke(), range(2)))

            self.assertEqual(results[0].result.fingerprint, results[1].result.fingerprint)
            self.assertEqual(len(request_log.read_text(encoding="utf-8").splitlines()), 1)

    def test_cross_process_lock_allows_only_one_proposer_launch(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            command, request_log = self._prepare(directory, delay=0.2)
            context = multiprocessing.get_context("fork")
            start = context.Event()
            results = context.Queue()

            def invoke() -> None:
                start.wait()
                try:
                    run = self._run(directory, command)
                    results.put(("ok", run.result.fingerprint))
                except BaseException as error:  # pragma: no cover - child diagnostic
                    results.put(("error", repr(error)))

            processes = [context.Process(target=invoke) for _ in range(2)]
            for process in processes:
                process.start()
            start.set()
            for process in processes:
                process.join(10)
                self.assertEqual(process.exitcode, 0)
            observed = [results.get(timeout=2) for _ in processes]
            self.assertEqual({status for status, _ in observed}, {"ok"})
            self.assertEqual(len({fingerprint for _, fingerprint in observed}), 1)
            self.assertEqual(len(request_log.read_text(encoding="utf-8").splitlines()), 1)

    def test_manifest_and_runtime_binding_drift_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            command, request_log = self._prepare(directory)
            self._run(directory, command)
            manifest_path = directory / "run" / "authoring-job" / "manifest.json"
            record = json.loads(manifest_path.read_text(encoding="ascii"))
            record["model_id"] = "openai/tampered/v1"
            manifest_path.write_text(canonical_json(record), encoding="ascii")

            with self.assertRaisesRegex(AuthoringJobDriftError, "manifest"):
                self._run(directory, command)
            self.assertEqual(len(request_log.read_text(encoding="utf-8").splitlines()), 1)

        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            command, request_log = self._prepare(directory)
            self._run(directory, command)
            with self.assertRaisesRegex(AuthoringJobDriftError, "manifest"):
                self._run(directory, command, model_id="openai/different-model/v1")
            self.assertEqual(len(request_log.read_text(encoding="utf-8").splitlines()), 1)

    def test_request_or_completion_tampering_fails_closed(self) -> None:
        for filename in ("request.json", "completion.json"):
            with self.subTest(filename=filename), tempfile.TemporaryDirectory() as raw_directory:
                directory = Path(raw_directory)
                command, request_log = self._prepare(directory)
                self._run(directory, command)
                path = (
                    directory / "run" / "authoring-job" / "attempts"
                    / "attempt-001" / filename
                )
                record = json.loads(path.read_text(encoding="ascii"))
                record["tampered"] = True
                path.write_text(canonical_json(record), encoding="ascii")
                with self.assertRaises(AuthoringJobDriftError):
                    self._run(directory, command)
                self.assertEqual(
                    len(request_log.read_text(encoding="utf-8").splitlines()), 1
                )

    def test_completed_repair_prefix_is_resumed_at_next_ordinal(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            command, request_log = self._prepare(directory, first_invalid=True)
            from rexpolicy.tasking import authoring_job

            original = authoring_job._publish_once
            injected = False

            def publish_then_crash(path, record):
                nonlocal injected
                created = original(path, record)
                if path.name == "completion.json" and not injected:
                    injected = True
                    raise RuntimeError("simulated crash after durable completion")
                return created

            with mock.patch.object(authoring_job, "_publish_once", publish_then_crash):
                with self.assertRaisesRegex(RuntimeError, "simulated crash"):
                    self._run(directory, command)

            resumed = self._run(directory, command)
            self.assertEqual(resumed.result.final_state, "quarantined_static")
            self.assertEqual([item.ordinal for item in resumed.attempts], [1, 2])
            requests = [
                json.loads(line)
                for line in request_log.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual([item["attempt"]["ordinal"] for item in requests], [1, 2])
            self.assertEqual(
                requests[1]["attempt"]["parent_attempt_fingerprint"],
                resumed.attempts[0].fingerprint,
            )

    def test_unresolved_launch_marker_is_ambiguous_and_never_retried(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            command, request_log = self._prepare(directory)
            from rexpolicy.tasking import authoring_job

            with mock.patch.object(
                authoring_job,
                "run_proposer",
                side_effect=RuntimeError("simulated crash after launch marker"),
            ):
                with self.assertRaisesRegex(RuntimeError, "simulated crash"):
                    self._run(directory, command)

            with self.assertRaises(AmbiguousAuthoringAttemptError):
                self._run(directory, command)
            self.assertFalse(request_log.exists())

    def test_attempt_budget_is_global_across_restarts(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            command, request_log = self._prepare(directory, always_invalid=True)
            from rexpolicy.tasking import authoring_job

            original = authoring_job._publish_once
            for expected_ordinal in (1, 2):
                injected = False

                def publish_then_crash(path, record):
                    nonlocal injected
                    created = original(path, record)
                    if path.name == "completion.json" and not injected:
                        injected = True
                        raise RuntimeError("simulated restart boundary")
                    return created

                with mock.patch.object(
                    authoring_job,
                    "_publish_once",
                    publish_then_crash,
                ):
                    with self.assertRaisesRegex(RuntimeError, "restart boundary"):
                        self._run(directory, command)
                requests = request_log.read_text(encoding="utf-8").splitlines()
                self.assertEqual(len(requests), expected_ordinal)

            terminal = self._run(directory, command)
            repeated = self._run(directory, command)
            self.assertEqual(terminal.result.final_state, "exhausted")
            self.assertEqual(terminal.result.fingerprint, repeated.result.fingerprint)
            self.assertEqual(len(terminal.attempts), DEFAULT_AUTHORING_POLICY.max_attempts)
            requests = [
                json.loads(line)
                for line in request_log.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(
                [item["attempt"]["ordinal"] for item in requests],
                [1, 2, 3],
            )


if __name__ == "__main__":
    unittest.main()
