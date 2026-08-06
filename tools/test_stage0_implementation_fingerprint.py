"""CPU-only tests for the Stage 0 scientific implementation fingerprint."""

from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path

from rexpolicy.stage0.implementation_fingerprint import (
    STAGE0_IMPLEMENTATION_DISTRIBUTIONS,
    STAGE0_IMPLEMENTATION_FINGERPRINT_SCHEMA_ID,
    Stage0ImplementationFingerprintError,
    capture_stage0_implementation_fingerprint,
)
from rexpolicy.stage0.types import canonical_fingerprint


_HEAD = "a" * 40


class _GitRunner:
    def __init__(self, status: str = "") -> None:
        self.status = status
        self.calls: list[tuple[tuple[str, ...], Path]] = []

    def __call__(self, arguments: tuple[str, ...], cwd: Path) -> str:
        self.calls.append((arguments, cwd))
        if arguments == ("git", "rev-parse", "--show-toplevel"):
            return "/fixture/repository\n"
        if arguments == ("git", "rev-parse", "--verify", "HEAD^{commit}"):
            return _HEAD + "\n"
        if arguments == (
            "git",
            "--no-optional-locks",
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
            "--ignore-submodules=none",
        ):
            return self.status
        raise AssertionError(f"unexpected command: {arguments!r}")


def _runtime() -> dict[str, str]:
    return {"implementation": "CPython", "version": "3.12.9"}


class Stage0ImplementationFingerprintTest(unittest.TestCase):
    def test_clean_capture_is_canonical_and_records_missing_distributions(self) -> None:
        runner = _GitRunner()
        installed = {
            "torch": "2.11.0",
            "warp-lang": "1.10.0",
            "newton": None,
            "mujoco-warp": None,
        }
        queries: list[str] = []

        def version(distribution: str) -> str | None:
            queries.append(distribution)
            return installed[distribution]

        first = capture_stage0_implementation_fingerprint(
            "/requested/checkout",
            command_runner=runner,
            distribution_version=version,
            python_runtime=_runtime,
        )
        second = capture_stage0_implementation_fingerprint(
            "/another/checkout",
            command_runner=_GitRunner(),
            distribution_version=installed.get,
            python_runtime=_runtime,
        )

        self.assertEqual(first, second)
        record = first.to_record()
        self.assertEqual(
            record["schema_id"], STAGE0_IMPLEMENTATION_FINGERPRINT_SCHEMA_ID
        )
        self.assertEqual(record["git"]["head"], _HEAD)
        self.assertEqual(record["git"]["worktree_status"], "clean")
        self.assertEqual(record["python"], _runtime())
        self.assertEqual(record["distributions"]["torch"]["status"], "installed")
        self.assertEqual(record["distributions"]["newton"]["status"], "missing")
        self.assertIsNone(record["distributions"]["newton"]["version"])
        self.assertEqual(first.sha256, canonical_fingerprint(record))
        self.assertEqual(
            queries,
            [distribution for _, distribution in STAGE0_IMPLEMENTATION_DISTRIBUTIONS],
        )
        self.assertNotIn("/fixture/repository", str(record))
        self.assertNotIn("/requested/checkout", str(record))

    def test_tracked_or_untracked_change_fails_closed(self) -> None:
        for status in (
            " M rexpolicy/stage0/models/flow_dit.py\n",
            "?? rexpolicy/stage0/experimental.py\n",
        ):
            with self.subTest(status=status):
                with self.assertRaisesRegex(
                    Stage0ImplementationFingerprintError,
                    "clean Git worktree",
                ):
                    capture_stage0_implementation_fingerprint(
                        "/repository",
                        command_runner=_GitRunner(status),
                        distribution_version=lambda _name: None,
                        python_runtime=_runtime,
                    )

    def test_invalid_git_identity_and_metadata_errors_fail_closed(self) -> None:
        class InvalidHeadRunner(_GitRunner):
            def __call__(self, arguments: tuple[str, ...], cwd: Path) -> str:
                if arguments == (
                    "git",
                    "rev-parse",
                    "--verify",
                    "HEAD^{commit}",
                ):
                    return "main\n"
                return super().__call__(arguments, cwd)

        with self.assertRaisesRegex(
            Stage0ImplementationFingerprintError,
            "invalid full object ID",
        ):
            capture_stage0_implementation_fingerprint(
                "/repository",
                command_runner=InvalidHeadRunner(),
                distribution_version=lambda _name: None,
                python_runtime=_runtime,
            )

        def broken_metadata(_distribution: str) -> str | None:
            raise RuntimeError("metadata backend failed")

        with self.assertRaisesRegex(
            Stage0ImplementationFingerprintError,
            "cannot inspect distribution metadata",
        ):
            capture_stage0_implementation_fingerprint(
                "/repository",
                command_runner=_GitRunner(),
                distribution_version=broken_metadata,
                python_runtime=_runtime,
            )

        with self.assertRaisesRegex(
            Stage0ImplementationFingerprintError,
            "invalid distribution metadata",
        ):
            capture_stage0_implementation_fingerprint(
                "/repository",
                command_runner=_GitRunner(),
                distribution_version=lambda _name: 7,  # type: ignore[return-value]
                python_runtime=_runtime,
            )

    def test_gitignored_outputs_do_not_dirty_a_real_worktree(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            subprocess.run(("git", "init", "-q"), cwd=root, check=True)
            (root / ".gitignore").write_text("outputs/\n", encoding="ascii")
            (root / "implementation.py").write_text("VALUE = 1\n", encoding="ascii")
            subprocess.run(
                ("git", "add", ".gitignore", "implementation.py"),
                cwd=root,
                check=True,
            )
            subprocess.run(
                (
                    "git",
                    "-c",
                    "user.name=Stage0 Test",
                    "-c",
                    "user.email=stage0@example.invalid",
                    "commit",
                    "-qm",
                    "fixture",
                ),
                cwd=root,
                check=True,
            )
            (root / "outputs").mkdir()
            (root / "outputs" / "run.json").write_text("{}\n", encoding="ascii")

            fingerprint = capture_stage0_implementation_fingerprint(
                root,
                distribution_version=lambda _name: None,
                python_runtime=_runtime,
            )
            self.assertEqual(fingerprint.to_record()["git"]["worktree_status"], "clean")

            (root / "untracked.py").write_text("VALUE = 2\n", encoding="ascii")
            with self.assertRaisesRegex(
                Stage0ImplementationFingerprintError,
                "untracked.py",
            ):
                capture_stage0_implementation_fingerprint(
                    root,
                    distribution_version=lambda _name: None,
                    python_runtime=_runtime,
                )


if __name__ == "__main__":
    unittest.main()
