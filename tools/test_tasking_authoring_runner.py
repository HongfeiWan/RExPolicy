"""Tests for the bounded shell-free authoring proposer protocol."""

from __future__ import annotations

import hashlib
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from rexpolicy.tasking.authoring import DEFAULT_AUTHORING_POLICY
from rexpolicy.tasking.authoring_runner import (
    DEFAULT_PROPOSER_EXECUTION_POLICY,
    ProposerCommand,
    run_proposer,
)
from rexpolicy.tasking.canonical import canonical_json


def _script(directory: str, body: str) -> Path:
    path = Path(directory) / "fake_proposer.py"
    path.write_text(body, encoding="utf-8")
    return path


def _command(script: Path, *arguments: str) -> ProposerCommand:
    return ProposerCommand.bind(
        command_id="rexpolicy/fake_proposer/v1",
        argv=(sys.executable, str(script), *arguments),
    )


class TestAuthoringRunner(unittest.TestCase):
    def test_json_request_and_response_are_hash_bound(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            script = _script(
                directory,
                "import sys\ndata=sys.stdin.buffer.read()\nsys.stdout.buffer.write(data)\n",
            )
            request = {"intent": {"task_id": "reach/v3"}, "attempt": 1}
            run = run_proposer(command=_command(script), request=request)

        expected = canonical_json(request).encode("ascii")
        self.assertEqual(run.invocation.status, "completed")
        self.assertEqual(run.invocation.exit_code, 0)
        self.assertEqual(run.stdout, expected)
        self.assertEqual(
            run.invocation.request_sha256,
            hashlib.sha256(expected).hexdigest(),
        )
        self.assertEqual(run.invocation.stdout_sha256, run.invocation.request_sha256)
        self.assertNotIn("stdout", run.invocation.to_record())

    def test_nonzero_stderr_is_hash_only_and_never_returned_as_prompt_text(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            script = _script(
                directory,
                "import sys\nsys.stderr.write('secret traceback')\nsys.exit(7)\n",
            )
            run = run_proposer(command=_command(script), request={"attempt": 1})

        self.assertEqual(run.invocation.status, "nonzero_exit")
        self.assertEqual(run.invocation.exit_code, 7)
        self.assertEqual(run.stdout, b"")
        self.assertGreater(run.invocation.stderr_bytes, 0)
        self.assertNotIn("secret", canonical_json(run.invocation.to_record()))

    def test_timeout_kills_the_proposer_process_group(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            script = _script(directory, "import time\ntime.sleep(10)\n")
            policy = replace(
                DEFAULT_PROPOSER_EXECUTION_POLICY,
                timeout_milliseconds=30,
            )
            run = run_proposer(
                command=_command(script),
                request={"attempt": 1},
                execution_policy=policy,
            )

        self.assertEqual(run.invocation.status, "timeout")
        self.assertIsNone(run.invocation.exit_code)

    def test_stdout_and_stderr_are_bounded_before_capture(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            script = _script(
                directory,
                "import sys\nsys.stdout.write('x' * 10000)\nsys.stdout.flush()\n",
            )
            authoring = replace(DEFAULT_AUTHORING_POLICY, max_response_bytes=64)
            execution = replace(
                DEFAULT_PROPOSER_EXECUTION_POLICY,
                max_stdout_bytes=64,
            )
            run = run_proposer(
                command=_command(script),
                request={"attempt": 1},
                execution_policy=execution,
                authoring_policy=authoring,
            )

        self.assertEqual(run.invocation.status, "stdout_limit")
        self.assertTrue(run.invocation.stdout_truncated)
        self.assertEqual(len(run.stdout), 65)

    def test_executable_drift_fails_before_launch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            script = _script(directory, "print('{}')\n")
            command = replace(_command(script), executable_sha256="0" * 64)
            with self.assertRaisesRegex(ValueError, "fingerprint drifted"):
                run_proposer(command=command, request={"attempt": 1})

    def test_environment_is_exactly_allowlisted_and_not_inherited(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            script = _script(
                directory,
                "import os\nprint(os.environ.get('AUTHORIZED_TOKEN', 'missing'))\n",
            )
            policy = replace(
                DEFAULT_PROPOSER_EXECUTION_POLICY,
                allowed_environment_names=("AUTHORIZED_TOKEN",),
            )
            run = run_proposer(
                command=_command(script),
                request={"attempt": 1},
                environment={"AUTHORIZED_TOKEN": "present"},
                execution_policy=policy,
            )
            self.assertEqual(run.stdout.strip(), b"present")
            with self.assertRaisesRegex(ValueError, "allowlist"):
                run_proposer(
                    command=_command(script),
                    request={"attempt": 1},
                    environment={"UNAUTHORIZED": "value"},
                    execution_policy=policy,
                )

    def test_shell_metacharacters_remain_literal_arguments(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            marker = Path(directory) / "should_not_exist"
            argument = f";touch {marker}"
            script = _script(
                directory,
                "import sys\nsys.stdout.write(sys.argv[1])\n",
            )
            run = run_proposer(
                command=_command(script, argument),
                request={"attempt": 1},
            )

            self.assertEqual(run.stdout.decode(), argument)
            self.assertFalse(marker.exists())


if __name__ == "__main__":
    unittest.main()
