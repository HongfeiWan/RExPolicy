"""Tests for the bounded shell-free authoring proposer protocol."""

from __future__ import annotations

import hashlib
import stat
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from rexpolicy.tasking.authoring import DEFAULT_AUTHORING_POLICY
from rexpolicy.tasking.authoring_runner import (
    DEFAULT_BUBBLEWRAP_SANDBOX_PROFILE,
    DEFAULT_PROPOSER_EXECUTION_POLICY,
    BubblewrapProposerCommand,
    BubblewrapReadOnlyFile,
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


def _resolved_bubblewrap_test_python() -> str | None:
    """Return one real interpreter file already covered by the /usr mount."""

    try:
        resolved = Path("/usr/bin/python3").resolve(strict=True)
        mode = resolved.lstat().st_mode
        resolved.relative_to("/usr")
    except (OSError, ValueError):
        return None
    if not stat.S_ISREG(mode):
        return None
    return str(resolved)


BUBBLEWRAP_TEST_PYTHON = _resolved_bubblewrap_test_python()


def _sandbox_command(
    script: Path,
    *arguments: str,
    profile=DEFAULT_BUBBLEWRAP_SANDBOX_PROFILE,
) -> BubblewrapProposerCommand:
    if BUBBLEWRAP_TEST_PYTHON is None:
        raise RuntimeError("Bubblewrap test interpreter is unavailable")
    return BubblewrapProposerCommand.bind(
        proposer_command=ProposerCommand.bind(
            command_id="rexpolicy/fake_proposer/v1",
            argv=(BUBBLEWRAP_TEST_PYTHON, str(script), *arguments),
        ),
        sandbox_profile=profile,
    )


_BUBBLEWRAP_AVAILABLE = sys.platform.startswith("linux") and Path(
    "/usr/bin/bwrap"
).is_file() and BUBBLEWRAP_TEST_PYTHON is not None


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

    def test_interpreted_proposer_program_drift_is_hash_bound(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            script = _script(directory, "print('{}')\n")
            command = _command(script)
            script.write_text("print('{\"changed\":true}')\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "artifact fingerprint drifted"):
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


@unittest.skipUnless(
    _BUBBLEWRAP_AVAILABLE,
    "Bubblewrap isolation tests require Linux and /usr/bin/bwrap",
)
class TestBubblewrapAuthoringRunner(unittest.TestCase):
    def test_host_sentinel_is_not_visible_and_cwd_is_tmpfs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            sentinel = Path(directory) / "host_secret.txt"
            sentinel.write_text("sealed evidence", encoding="utf-8")
            script = _script(
                directory,
                "import os\n"
                f"path = {str(sentinel)!r}\n"
                "try:\n"
                "    open(path, 'rb').read()\n"
                "except OSError:\n"
                "    print('denied:' + os.getcwd())\n"
                "else:\n"
                "    print('LEAKED')\n",
            )
            run = run_proposer(
                command=_sandbox_command(script),
                request={"attempt": 1},
            )

        self.assertEqual(run.invocation.status, "completed")
        self.assertEqual(run.stdout.strip(), b"denied:/rexpolicy/work")

    def test_explicit_hash_bound_file_is_readable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            public = Path(directory) / "public_input.txt"
            public.write_text("declassified", encoding="utf-8")
            mounted = BubblewrapReadOnlyFile.bind(
                source_path=str(public),
                sandbox_path="/rexpolicy/inputs/public_input",
            )
            profile = replace(
                DEFAULT_BUBBLEWRAP_SANDBOX_PROFILE,
                readonly_files=(mounted,),
            )
            script = _script(
                directory,
                "from pathlib import Path\n"
                "print(Path('/rexpolicy/inputs/public_input').read_text())\n",
            )
            run = run_proposer(
                command=_sandbox_command(script, profile=profile),
                request={"attempt": 1},
            )

        self.assertEqual(run.invocation.status, "completed")
        self.assertEqual(run.stdout.strip(), b"declassified")

    def test_argv_artifact_is_rewritten_to_its_isolated_mount(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            public = Path(directory) / "public_argument.txt"
            public.write_text("bound argument", encoding="utf-8")
            script = _script(
                directory,
                "from pathlib import Path\nimport sys\n"
                "print(Path(sys.argv[1]).read_text())\n",
            )
            run = run_proposer(
                command=_sandbox_command(script, str(public)),
                request={"attempt": 1},
            )

        self.assertEqual(run.invocation.status, "completed")
        self.assertEqual(run.stdout.strip(), b"bound argument")

    def test_timeout_still_kills_sandboxed_process_group(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            script = _script(directory, "import time\ntime.sleep(10)\n")
            policy = replace(
                DEFAULT_PROPOSER_EXECUTION_POLICY,
                timeout_milliseconds=100,
            )
            run = run_proposer(
                command=_sandbox_command(script),
                request={"attempt": 1},
                execution_policy=policy,
            )

        self.assertEqual(run.invocation.status, "timeout")
        self.assertIsNone(run.invocation.exit_code)

    def test_output_limit_still_bounds_sandboxed_process(self) -> None:
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
                command=_sandbox_command(script),
                request={"attempt": 1},
                execution_policy=execution,
                authoring_policy=authoring,
            )

        self.assertEqual(run.invocation.status, "stdout_limit")
        self.assertTrue(run.invocation.stdout_truncated)
        self.assertEqual(len(run.stdout), 65)

    def test_profile_and_bubblewrap_binary_are_pinned(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            script = _script(directory, "print('{}')\n")
            command = _sandbox_command(script)
            changed_profile = replace(
                command.sandbox_profile,
                network_policy="inherit",
            )
            drifted_profile = replace(command, sandbox_profile=changed_profile)
            with self.assertRaisesRegex(ValueError, "profile fingerprint drifted"):
                run_proposer(command=drifted_profile, request={"attempt": 1})

            drifted_bwrap = replace(
                command,
                bubblewrap_artifact=replace(
                    command.bubblewrap_artifact,
                    sha256="0" * 64,
                ),
            )
            with self.assertRaisesRegex(ValueError, "executable fingerprint drifted"):
                run_proposer(command=drifted_bwrap, request={"attempt": 1})

    def test_sandboxed_script_drift_is_rejected_before_launch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            script = _script(directory, "print('{}')\n")
            command = _sandbox_command(script)
            script.write_text("print('changed')\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "artifact fingerprint drifted"):
                run_proposer(command=command, request={"attempt": 1})

    def test_outside_runtime_symlinked_executable_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            script = _script(directory, "print('{}')\n")
            executable = Path(directory) / "python"
            executable.symlink_to(BUBBLEWRAP_TEST_PYTHON)
            proposer = ProposerCommand.bind(
                command_id="rexpolicy/symlinked_fake_proposer/v1",
                argv=(str(executable), str(script)),
            )
            command = BubblewrapProposerCommand.bind(
                proposer_command=proposer,
            )

            with self.assertRaisesRegex(
                ValueError,
                "Sandboxed proposer executable must be a regular file",
            ):
                run_proposer(command=command, request={"attempt": 1})


class TestBubblewrapProfileValidation(unittest.TestCase):
    def test_network_policy_is_explicit_and_fingerprinted(self) -> None:
        inherited = replace(
            DEFAULT_BUBBLEWRAP_SANDBOX_PROFILE,
            network_policy="inherit",
        )
        self.assertNotEqual(
            inherited.fingerprint,
            DEFAULT_BUBBLEWRAP_SANDBOX_PROFILE.fingerprint,
        )
        with self.assertRaisesRegex(ValueError, "network policy"):
            replace(
                DEFAULT_BUBBLEWRAP_SANDBOX_PROFILE,
                network_policy="unspecified",
            ).canonical()

    def test_dangerous_or_directory_mounts_are_rejected(self) -> None:
        digest = "0" * 64
        for source in ("/", "/home", "/root", "relative/file"):
            with self.subTest(source=source):
                with self.assertRaisesRegex(ValueError, "mount|absolute"):
                    BubblewrapReadOnlyFile(
                        source_path=source,
                        sandbox_path="/rexpolicy/inputs/file",
                        sha256=digest,
                    ).canonical()
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "regular file"):
                BubblewrapReadOnlyFile.bind(
                    source_path=directory,
                    sandbox_path="/rexpolicy/inputs/directory",
                )
        with self.assertRaisesRegex(ValueError, "trusted allowlist"):
            replace(
                DEFAULT_BUBBLEWRAP_SANDBOX_PROFILE,
                runtime_readonly_paths=("/",),
            ).canonical()

    def test_mount_targets_and_sensitive_sources_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "public.txt"
            source.write_text("public", encoding="utf-8")
            for target in ("relative", "/", "/home", "/root/file"):
                with self.subTest(target=target):
                    with self.assertRaisesRegex(ValueError, "target"):
                        BubblewrapReadOnlyFile.bind(
                            source_path=str(source),
                            sandbox_path=target,
                        )
            quarantine = Path(directory) / "quarantine"
            quarantine.mkdir()
            secret = quarantine / "evidence.json"
            secret.write_text("secret", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Quarantine and sealed"):
                BubblewrapReadOnlyFile.bind(
                    source_path=str(secret),
                    sandbox_path="/rexpolicy/inputs/evidence",
                )


if __name__ == "__main__":
    unittest.main()
