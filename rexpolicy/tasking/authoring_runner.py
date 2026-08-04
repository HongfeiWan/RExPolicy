"""Bounded shell-free subprocess protocol for automatic TaskSpec proposers.

``ProposerCommand`` is the compatibility path for a *trusted* local wrapper.  It
does not isolate that wrapper from the host filesystem; only the model response
handled by the wrapper is considered untrusted.  Use
``BubblewrapProposerCommand`` when the proposer process itself needs a filesystem
boundary.  The sandbox deliberately exposes individual, hash-bound files rather
than a repository or home directory.
"""

from __future__ import annotations

import hashlib
import os
import selectors
import signal
import stat
import subprocess
import tempfile
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .authoring import DEFAULT_AUTHORING_POLICY, AuthoringPolicy
from .canonical import canonical_fingerprint, canonical_json

_BASE_ENVIRONMENT = {
    "PYTHONHASHSEED": "0",
    "PYTHONIOENCODING": "utf-8",
}
_STATUSES = frozenset(
    ("completed", "nonzero_exit", "timeout", "stdout_limit", "stderr_limit")
)
_INTERPRETER_PREFIXES = ("bash", "node", "perl", "python", "ruby", "sh", "zsh")
_BUBBLEWRAP_NETWORK_POLICIES = frozenset(("deny", "inherit"))
_BUBBLEWRAP_RUNTIME_PATHS = frozenset(("/bin", "/lib", "/lib64", "/usr"))
_BUBBLEWRAP_WORK_DIRECTORY = "/rexpolicy/work"
_BUBBLEWRAP_INPUT_DIRECTORY = "/rexpolicy/inputs"
_SENSITIVE_HOST_PATH_PARTS = frozenset(("quarantine", "sealed"))


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while True:
            chunk = file.read(1024 * 1024)
            if not chunk:
                return digest.hexdigest()
            digest.update(chunk)


def _bounded_positive(value: Any, path: str, maximum: int) -> int:
    if type(value) is not int or not 1 <= value <= maximum:
        raise ValueError(f"{path} must be between 1 and {maximum}")
    return value


def _canonical_absolute_path(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise ValueError(f"{path} must be an absolute normalized path")
    candidate = Path(value)
    if not candidate.is_absolute() or os.path.normpath(value) != value:
        raise ValueError(f"{path} must be an absolute normalized path")
    return value


def _has_sensitive_host_component(path: str) -> bool:
    return bool(
        _SENSITIVE_HOST_PATH_PARTS.intersection(
            part.lower() for part in Path(path).parts
        )
    )


def _is_beneath(path: str, parent: str) -> bool:
    try:
        Path(path).relative_to(parent)
    except ValueError:
        return False
    return path != parent


@dataclass(frozen=True)
class ProposerExecutionPolicy:
    policy_id: str
    timeout_milliseconds: int
    max_request_bytes: int
    max_stdout_bytes: int
    max_stderr_bytes: int
    allowed_environment_names: tuple[str, ...]

    def validate(
        self,
        *,
        authoring_policy: AuthoringPolicy = DEFAULT_AUTHORING_POLICY,
    ) -> None:
        if self.policy_id != "rexpolicy/proposer_execution/v1":
            raise ValueError("Unsupported proposer execution policy")
        _bounded_positive(self.timeout_milliseconds, "timeout_milliseconds", 300_000)
        _bounded_positive(self.max_request_bytes, "max_request_bytes", 1_000_000)
        stdout_limit = _bounded_positive(
            self.max_stdout_bytes,
            "max_stdout_bytes",
            authoring_policy.max_response_bytes,
        )
        if stdout_limit != authoring_policy.max_response_bytes:
            raise ValueError(
                "Proposer stdout limit must equal the authoring response limit"
            )
        _bounded_positive(self.max_stderr_bytes, "max_stderr_bytes", 262_144)
        names = tuple(self.allowed_environment_names)
        if names != tuple(sorted(set(names))):
            raise ValueError("Allowed proposer environment names must be sorted")
        for name in names:
            if (
                not isinstance(name, str)
                or not name
                or not name.replace("_", "A").isalnum()
                or not name[0].isalpha()
                or name.upper() != name
            ):
                raise ValueError("Allowed proposer environment name is invalid")
            if name in _BASE_ENVIRONMENT:
                raise ValueError("Base proposer environment cannot be overridden")

    def to_record(self) -> dict[str, Any]:
        return {
            "policy_id": self.policy_id,
            "timeout_milliseconds": self.timeout_milliseconds,
            "max_request_bytes": self.max_request_bytes,
            "max_stdout_bytes": self.max_stdout_bytes,
            "max_stderr_bytes": self.max_stderr_bytes,
            "allowed_environment_names": list(self.allowed_environment_names),
        }

    @property
    def fingerprint(self) -> str:
        return canonical_fingerprint(self.to_record())


DEFAULT_PROPOSER_EXECUTION_POLICY = ProposerExecutionPolicy(
    policy_id="rexpolicy/proposer_execution/v1",
    timeout_milliseconds=120_000,
    max_request_bytes=262_144,
    max_stdout_bytes=DEFAULT_AUTHORING_POLICY.max_response_bytes,
    max_stderr_bytes=65_536,
    allowed_environment_names=(),
)


@dataclass(frozen=True)
class ProposerArtifact:
    path: str
    sha256: str

    def canonical(self) -> ProposerArtifact:
        path = Path(self.path)
        if not path.is_absolute() or "\x00" in self.path:
            raise ValueError("Proposer artifact path must be absolute")
        if (
            not isinstance(self.sha256, str)
            or len(self.sha256) != 64
            or any(character not in "0123456789abcdef" for character in self.sha256)
        ):
            raise ValueError("Proposer artifact fingerprint is invalid")
        return ProposerArtifact(path=str(path), sha256=self.sha256)

    def to_record(self) -> dict[str, str]:
        return {"path": self.path, "sha256": self.sha256}


@dataclass(frozen=True)
class ProposerCommand:
    command_id: str
    argv: tuple[str, ...]
    executable_sha256: str
    artifacts: tuple[ProposerArtifact, ...]

    @classmethod
    def bind(cls, *, command_id: str, argv: tuple[str, ...]) -> ProposerCommand:
        if not isinstance(argv, tuple) or not argv:
            raise ValueError("Proposer argv must be a non-empty tuple")
        executable = Path(argv[0])
        try:
            mode = executable.stat().st_mode
        except OSError as error:
            raise ValueError("Proposer executable is unavailable") from error
        if not executable.is_absolute() or not stat.S_ISREG(mode):
            raise ValueError("Proposer executable must be an absolute regular file")
        artifacts = []
        for argument in argv[1:]:
            candidate = Path(argument)
            try:
                candidate_mode = candidate.stat().st_mode
            except OSError:
                continue
            if candidate.is_absolute() and stat.S_ISREG(candidate_mode):
                artifacts.append(
                    ProposerArtifact(
                        path=str(candidate),
                        sha256=_file_sha256(candidate),
                    )
                )
        return cls(
            command_id=command_id,
            argv=argv,
            executable_sha256=_file_sha256(executable),
            artifacts=tuple(sorted(artifacts, key=lambda item: item.path)),
        ).canonical()

    def canonical(self) -> ProposerCommand:
        if (
            not isinstance(self.command_id, str)
            or not self.command_id.startswith("rexpolicy/")
            or any(character.isspace() for character in self.command_id)
        ):
            raise ValueError("Proposer command ID is invalid")
        if not isinstance(self.argv, tuple) or not 1 <= len(self.argv) <= 64:
            raise ValueError("Proposer argv is invalid")
        for argument in self.argv:
            if (
                not isinstance(argument, str)
                or not argument
                or len(argument) > 4096
                or "\x00" in argument
            ):
                raise ValueError("Proposer argument is invalid")
        executable = Path(self.argv[0])
        if not executable.is_absolute():
            raise ValueError("Proposer executable path must be absolute")
        if (
            not isinstance(self.executable_sha256, str)
            or len(self.executable_sha256) != 64
            or any(character not in "0123456789abcdef" for character in self.executable_sha256)
        ):
            raise ValueError("Proposer executable fingerprint is invalid")
        artifacts = tuple(item.canonical() for item in self.artifacts)
        artifact_paths = tuple(item.path for item in artifacts)
        if artifact_paths != tuple(sorted(set(artifact_paths))):
            raise ValueError("Proposer artifacts must be sorted and unique")
        executable_name = executable.name.lower()
        if executable_name.startswith(_INTERPRETER_PREFIXES):
            script_arguments = [
                argument
                for argument in self.argv[1:]
                if not argument.startswith("-") and Path(argument).is_absolute()
            ]
            if script_arguments and script_arguments[0] not in set(artifact_paths):
                raise ValueError("Interpreted proposer program must be hash-bound")
        return ProposerCommand(
            command_id=self.command_id,
            argv=tuple(self.argv),
            executable_sha256=self.executable_sha256,
            artifacts=artifacts,
        )

    def verify_executable(self) -> None:
        executable = Path(self.argv[0])
        try:
            actual = _file_sha256(executable)
        except OSError as error:
            raise ValueError("Proposer executable is unavailable") from error
        if actual != self.executable_sha256:
            raise ValueError("Proposer executable fingerprint drifted")
        for artifact in self.artifacts:
            try:
                actual = _file_sha256(Path(artifact.path))
            except OSError as error:
                raise ValueError("Proposer artifact is unavailable") from error
            if actual != artifact.sha256:
                raise ValueError("Proposer artifact fingerprint drifted")

    def to_record(self) -> dict[str, Any]:
        return {
            "command_id": self.command_id,
            "argv": list(self.argv),
            "executable_sha256": self.executable_sha256,
            "artifacts": [artifact.to_record() for artifact in self.artifacts],
        }

    @property
    def fingerprint(self) -> str:
        return canonical_fingerprint(self.to_record())


@dataclass(frozen=True)
class BubblewrapReadOnlyFile:
    """One explicitly declassified file exposed at a fixed sandbox path."""

    source_path: str
    sandbox_path: str
    sha256: str

    @classmethod
    def bind(
        cls,
        *,
        source_path: str,
        sandbox_path: str,
    ) -> BubblewrapReadOnlyFile:
        canonical_source = _canonical_absolute_path(
            source_path, "Bubblewrap mount source"
        )
        source = Path(canonical_source)
        try:
            source_mode = source.lstat().st_mode
        except OSError as error:
            raise ValueError("Bubblewrap mount source is unavailable") from error
        if not stat.S_ISREG(source_mode):
            raise ValueError("Bubblewrap mount source must be a regular file")
        return cls(
            source_path=canonical_source,
            sandbox_path=sandbox_path,
            sha256=_file_sha256(source),
        ).canonical()

    def canonical(self) -> BubblewrapReadOnlyFile:
        source_path = _canonical_absolute_path(
            self.source_path, "Bubblewrap mount source"
        )
        sandbox_path = _canonical_absolute_path(
            self.sandbox_path, "Bubblewrap mount target"
        )
        if source_path in {"/", "/home", "/root"}:
            raise ValueError("Dangerous Bubblewrap host mount is forbidden")
        if _has_sensitive_host_component(source_path):
            raise ValueError("Quarantine and sealed files cannot be sandbox mounts")
        if not _is_beneath(sandbox_path, _BUBBLEWRAP_INPUT_DIRECTORY):
            raise ValueError(
                "Bubblewrap file targets must be beneath /rexpolicy/inputs"
            )
        if Path(sandbox_path).parent != Path(_BUBBLEWRAP_INPUT_DIRECTORY):
            raise ValueError("Bubblewrap file targets must be direct input files")
        if (
            not isinstance(self.sha256, str)
            or len(self.sha256) != 64
            or any(character not in "0123456789abcdef" for character in self.sha256)
        ):
            raise ValueError("Bubblewrap mount fingerprint is invalid")
        return BubblewrapReadOnlyFile(
            source_path=source_path,
            sandbox_path=sandbox_path,
            sha256=self.sha256,
        )

    def verify(self) -> None:
        canonical = self.canonical()
        source = Path(canonical.source_path)
        try:
            source_mode = source.lstat().st_mode
            actual = _file_sha256(source)
        except OSError as error:
            raise ValueError("Bubblewrap mount source is unavailable") from error
        if not stat.S_ISREG(source_mode):
            raise ValueError("Bubblewrap mount source must be a regular file")
        if actual != canonical.sha256:
            raise ValueError("Bubblewrap mount fingerprint drifted")

    def to_record(self) -> dict[str, str]:
        canonical = self.canonical()
        return {
            "source_path": canonical.source_path,
            "sandbox_path": canonical.sandbox_path,
            "sha256": canonical.sha256,
        }


@dataclass(frozen=True)
class BubblewrapSandboxProfile:
    """Hashable least-privilege filesystem and network policy for bwrap."""

    schema_version: int
    profile_id: str
    network_policy: str
    runtime_readonly_paths: tuple[str, ...]
    readonly_files: tuple[BubblewrapReadOnlyFile, ...]
    working_directory: str = _BUBBLEWRAP_WORK_DIRECTORY

    def canonical(self) -> BubblewrapSandboxProfile:
        if self.schema_version != 1:
            raise ValueError("Bubblewrap sandbox profile schema_version must be 1")
        if self.profile_id != "rexpolicy/proposer_sandbox/bubblewrap/v1":
            raise ValueError("Unsupported Bubblewrap sandbox profile")
        if self.network_policy not in _BUBBLEWRAP_NETWORK_POLICIES:
            raise ValueError("Bubblewrap network policy must be deny or inherit")
        runtime_paths = tuple(
            _canonical_absolute_path(path, "Bubblewrap runtime path")
            for path in self.runtime_readonly_paths
        )
        if runtime_paths != tuple(sorted(set(runtime_paths))):
            raise ValueError("Bubblewrap runtime paths must be sorted and unique")
        if not runtime_paths or any(
            path not in _BUBBLEWRAP_RUNTIME_PATHS for path in runtime_paths
        ):
            raise ValueError("Bubblewrap runtime path is not in the trusted allowlist")
        readonly_files = tuple(file.canonical() for file in self.readonly_files)
        sources = tuple(file.source_path for file in readonly_files)
        targets = tuple(file.sandbox_path for file in readonly_files)
        if sources != tuple(sorted(set(sources))):
            raise ValueError("Bubblewrap mount sources must be sorted and unique")
        if len(targets) != len(set(targets)):
            raise ValueError("Bubblewrap mount targets must be unique")
        if self.working_directory != _BUBBLEWRAP_WORK_DIRECTORY:
            raise ValueError("Bubblewrap working directory is fixed and ephemeral")
        return BubblewrapSandboxProfile(
            schema_version=1,
            profile_id=self.profile_id,
            network_policy=self.network_policy,
            runtime_readonly_paths=runtime_paths,
            readonly_files=readonly_files,
            working_directory=self.working_directory,
        )

    def verify(self) -> None:
        canonical = self.canonical()
        for runtime_path in canonical.runtime_readonly_paths:
            try:
                mode = Path(runtime_path).stat().st_mode
            except OSError as error:
                raise ValueError("Bubblewrap runtime path is unavailable") from error
            if not stat.S_ISDIR(mode):
                raise ValueError("Bubblewrap runtime path must be a directory")
        for readonly_file in canonical.readonly_files:
            readonly_file.verify()

    def to_record(self) -> dict[str, Any]:
        canonical = self.canonical()
        return {
            "schema_version": canonical.schema_version,
            "profile_id": canonical.profile_id,
            "network_policy": canonical.network_policy,
            "runtime_readonly_paths": list(canonical.runtime_readonly_paths),
            "readonly_files": [
                file.to_record() for file in canonical.readonly_files
            ],
            "working_directory": canonical.working_directory,
        }

    @property
    def fingerprint(self) -> str:
        return canonical_fingerprint(self.to_record())


DEFAULT_BUBBLEWRAP_SANDBOX_PROFILE = BubblewrapSandboxProfile(
    schema_version=1,
    profile_id="rexpolicy/proposer_sandbox/bubblewrap/v1",
    network_policy="deny",
    runtime_readonly_paths=("/lib", "/lib64", "/usr"),
    readonly_files=(),
)


def _path_in_runtime(path: str, runtime_paths: tuple[str, ...]) -> bool:
    return any(path == root or _is_beneath(path, root) for root in runtime_paths)


def _verify_mounted_regular_file(path: str, description: str) -> None:
    candidate = Path(path)
    try:
        mode = candidate.lstat().st_mode
    except OSError as error:
        raise ValueError(f"{description} is unavailable") from error
    if not stat.S_ISREG(mode):
        raise ValueError(f"{description} must be a regular file")


@dataclass(frozen=True)
class BubblewrapProposerCommand:
    """A proposer command plus a pinned Bubblewrap binary and sandbox profile."""

    schema_version: int
    proposer_command: ProposerCommand
    bubblewrap_artifact: ProposerArtifact
    sandbox_profile: BubblewrapSandboxProfile
    sandbox_profile_sha256: str

    @classmethod
    def bind(
        cls,
        *,
        proposer_command: ProposerCommand,
        sandbox_profile: BubblewrapSandboxProfile = DEFAULT_BUBBLEWRAP_SANDBOX_PROFILE,
        bubblewrap_path: str = "/usr/bin/bwrap",
    ) -> BubblewrapProposerCommand:
        path = Path(
            _canonical_absolute_path(bubblewrap_path, "Bubblewrap executable path")
        )
        try:
            mode = path.stat().st_mode
        except OSError as error:
            raise ValueError("Bubblewrap executable is unavailable") from error
        if not stat.S_ISREG(mode) or path.name != "bwrap":
            raise ValueError("Bubblewrap executable must be a regular bwrap file")
        canonical_profile = sandbox_profile.canonical()
        return cls(
            schema_version=1,
            proposer_command=proposer_command.canonical(),
            bubblewrap_artifact=ProposerArtifact(
                path=str(path), sha256=_file_sha256(path)
            ),
            sandbox_profile=canonical_profile,
            sandbox_profile_sha256=canonical_profile.fingerprint,
        ).canonical()

    @property
    def command_id(self) -> str:
        return self.proposer_command.command_id

    def canonical(self) -> BubblewrapProposerCommand:
        if self.schema_version != 1:
            raise ValueError("Bubblewrap proposer command schema_version must be 1")
        proposer_command = self.proposer_command.canonical()
        bubblewrap_artifact = self.bubblewrap_artifact.canonical()
        if _canonical_absolute_path(
            bubblewrap_artifact.path, "Bubblewrap executable path"
        ) != bubblewrap_artifact.path:
            raise ValueError("Bubblewrap executable path is not canonical")
        if Path(bubblewrap_artifact.path).name != "bwrap":
            raise ValueError("Bubblewrap executable path must name bwrap")
        sandbox_profile = self.sandbox_profile.canonical()
        if self.sandbox_profile_sha256 != sandbox_profile.fingerprint:
            raise ValueError("Bubblewrap sandbox profile fingerprint drifted")
        exposed_paths = (
            proposer_command.argv[0],
            *(artifact.path for artifact in proposer_command.artifacts),
        )
        if any(_has_sensitive_host_component(path) for path in exposed_paths):
            raise ValueError("Quarantine and sealed proposer artifacts are forbidden")
        return BubblewrapProposerCommand(
            schema_version=1,
            proposer_command=proposer_command,
            bubblewrap_artifact=bubblewrap_artifact,
            sandbox_profile=sandbox_profile,
            sandbox_profile_sha256=self.sandbox_profile_sha256,
        )

    def verify_executable(self) -> None:
        canonical = self.canonical()
        _verify_mounted_regular_file(
            canonical.bubblewrap_artifact.path, "Bubblewrap executable"
        )
        try:
            actual_bubblewrap = _file_sha256(
                Path(canonical.bubblewrap_artifact.path)
            )
        except OSError as error:
            raise ValueError("Bubblewrap executable is unavailable") from error
        if actual_bubblewrap != canonical.bubblewrap_artifact.sha256:
            raise ValueError("Bubblewrap executable fingerprint drifted")
        canonical.proposer_command.verify_executable()
        runtime_paths = canonical.sandbox_profile.runtime_readonly_paths
        executable_path = canonical.proposer_command.argv[0]
        if not _path_in_runtime(executable_path, runtime_paths):
            _verify_mounted_regular_file(executable_path, "Sandboxed proposer executable")
        for artifact in canonical.proposer_command.artifacts:
            _verify_mounted_regular_file(artifact.path, "Sandboxed proposer artifact")
        canonical.sandbox_profile.verify()

    def launch_argv(self) -> tuple[str, ...]:
        """Build a shell-free bwrap argv after re-verifying all pinned inputs."""
        canonical = self.canonical()
        canonical.verify_executable()
        profile = canonical.sandbox_profile
        argv = [
            canonical.bubblewrap_artifact.path,
            "--unshare-all",
            "--unshare-net" if profile.network_policy == "deny" else "--share-net",
            "--die-with-parent",
            "--new-session",
            "--cap-drop",
            "ALL",
        ]
        for runtime_path in profile.runtime_readonly_paths:
            argv.extend(("--ro-bind", runtime_path, runtime_path))
        argv.extend(
            (
                "--proc",
                "/proc",
                "--dev",
                "/dev",
                "--tmpfs",
                "/tmp",
                "--dir",
                "/rexpolicy",
                "--dir",
                "/rexpolicy/proposer",
                "--dir",
                _BUBBLEWRAP_INPUT_DIRECTORY,
                "--tmpfs",
                profile.working_directory,
                "--chdir",
                profile.working_directory,
            )
        )
        replacements: dict[str, str] = {}
        command = canonical.proposer_command
        if not _path_in_runtime(command.argv[0], profile.runtime_readonly_paths):
            target = "/rexpolicy/proposer/executable"
            argv.extend(("--ro-bind", command.argv[0], target))
            replacements[command.argv[0]] = target
        for ordinal, artifact in enumerate(command.artifacts):
            target = f"/rexpolicy/proposer/artifact-{ordinal:06d}"
            argv.extend(("--ro-bind", artifact.path, target))
            replacements[artifact.path] = target
        for readonly_file in profile.readonly_files:
            argv.extend(
                ("--ro-bind", readonly_file.source_path, readonly_file.sandbox_path)
            )
        argv.append("--")
        argv.extend(replacements.get(argument, argument) for argument in command.argv)
        return tuple(argv)

    def to_record(self) -> dict[str, Any]:
        canonical = self.canonical()
        return {
            "schema_version": canonical.schema_version,
            "proposer_command": canonical.proposer_command.to_record(),
            "bubblewrap_artifact": canonical.bubblewrap_artifact.to_record(),
            "sandbox_profile": canonical.sandbox_profile.to_record(),
            "sandbox_profile_sha256": canonical.sandbox_profile_sha256,
        }

    @property
    def fingerprint(self) -> str:
        return canonical_fingerprint(self.to_record())


@dataclass(frozen=True)
class ProposerInvocation:
    schema_version: int
    command_id: str
    command_sha256: str
    execution_policy_id: str
    execution_policy_sha256: str
    request_sha256: str
    request_bytes: int
    status: str
    exit_code: int | None
    stdout_sha256: str
    stdout_bytes: int
    stdout_truncated: bool
    stderr_sha256: str
    stderr_bytes: int
    stderr_truncated: bool

    def to_record(self) -> dict[str, Any]:
        if self.status not in _STATUSES:
            raise ValueError("Proposer invocation status is invalid")
        return {
            "schema_version": self.schema_version,
            "command_id": self.command_id,
            "command_sha256": self.command_sha256,
            "execution_policy_id": self.execution_policy_id,
            "execution_policy_sha256": self.execution_policy_sha256,
            "request_sha256": self.request_sha256,
            "request_bytes": self.request_bytes,
            "status": self.status,
            "exit_code": self.exit_code,
            "stdout_sha256": self.stdout_sha256,
            "stdout_bytes": self.stdout_bytes,
            "stdout_truncated": self.stdout_truncated,
            "stderr_sha256": self.stderr_sha256,
            "stderr_bytes": self.stderr_bytes,
            "stderr_truncated": self.stderr_truncated,
        }

    @property
    def fingerprint(self) -> str:
        return canonical_fingerprint(self.to_record())


@dataclass(frozen=True)
class ProposerRun:
    invocation: ProposerInvocation
    stdout: bytes


def _terminate(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()


def _read_bounded_process(
    process: subprocess.Popen[bytes],
    *,
    timeout_seconds: float,
    stdout_limit: int,
    stderr_limit: int,
) -> tuple[str, bytes, bytes]:
    assert process.stdout is not None and process.stderr is not None
    selector = selectors.DefaultSelector()
    selector.register(process.stdout, selectors.EVENT_READ, "stdout")
    selector.register(process.stderr, selectors.EVENT_READ, "stderr")
    buffers = {"stdout": bytearray(), "stderr": bytearray()}
    limits = {"stdout": stdout_limit, "stderr": stderr_limit}
    deadline = time.monotonic() + timeout_seconds
    status: str | None = None
    try:
        while selector.get_map():
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                status = "timeout"
                break
            events = selector.select(remaining)
            if not events:
                status = "timeout"
                break
            for key, _ in events:
                name = key.data
                chunk = os.read(key.fileobj.fileno(), 65_536)
                if not chunk:
                    selector.unregister(key.fileobj)
                    continue
                buffer = buffers[name]
                remaining_capacity = limits[name] + 1 - len(buffer)
                if remaining_capacity > 0:
                    buffer.extend(chunk[:remaining_capacity])
                if len(buffer) > limits[name]:
                    status = f"{name}_limit"
                    break
            if status is not None:
                break
        if status is not None:
            _terminate(process)
        else:
            process.wait()
            status = "completed" if process.returncode == 0 else "nonzero_exit"
        return status, bytes(buffers["stdout"]), bytes(buffers["stderr"])
    finally:
        selector.close()
        if process.poll() is None:
            _terminate(process)
        process.stdout.close()
        process.stderr.close()


def run_proposer(
    *,
    command: ProposerCommand | BubblewrapProposerCommand,
    request: Mapping[str, Any],
    environment: Mapping[str, str] | None = None,
    execution_policy: ProposerExecutionPolicy = DEFAULT_PROPOSER_EXECUTION_POLICY,
    authoring_policy: AuthoringPolicy = DEFAULT_AUTHORING_POLICY,
) -> ProposerRun:
    """Run one fixed proposer command via JSON stdin and bounded stdout.

    A plain ``ProposerCommand`` is intentionally a trusted-wrapper compatibility
    path and has host filesystem access.  Passing a ``BubblewrapProposerCommand``
    applies its pinned isolation profile before the wrapper starts.
    """
    canonical_command = command.canonical()
    canonical_command.verify_executable()
    if isinstance(canonical_command, BubblewrapProposerCommand):
        launch_argv = canonical_command.launch_argv()
    else:
        launch_argv = canonical_command.argv
    execution_policy.validate(authoring_policy=authoring_policy)
    request_bytes = canonical_json(dict(request)).encode("ascii")
    if len(request_bytes) > execution_policy.max_request_bytes:
        raise ValueError("Proposer request exceeds the trusted size limit")
    supplied_environment = dict(environment or {})
    if set(supplied_environment) != set(execution_policy.allowed_environment_names):
        raise ValueError("Proposer environment does not match the trusted allowlist")
    if any(not isinstance(value, str) or "\x00" in value for value in supplied_environment.values()):
        raise ValueError("Proposer environment value is invalid")
    child_environment = {**_BASE_ENVIRONMENT, **supplied_environment}

    with tempfile.TemporaryFile() as stdin, tempfile.TemporaryDirectory() as cwd:
        stdin.write(request_bytes)
        stdin.seek(0)
        process = subprocess.Popen(
            launch_argv,
            stdin=stdin,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=cwd,
            env=child_environment,
            shell=False,
            start_new_session=True,
        )
        status, stdout, stderr = _read_bounded_process(
            process,
            timeout_seconds=execution_policy.timeout_milliseconds / 1000.0,
            stdout_limit=execution_policy.max_stdout_bytes,
            stderr_limit=execution_policy.max_stderr_bytes,
        )
        exit_code = process.returncode if status in {"completed", "nonzero_exit"} else None

    invocation = ProposerInvocation(
        schema_version=1,
        command_id=canonical_command.command_id,
        command_sha256=canonical_command.fingerprint,
        execution_policy_id=execution_policy.policy_id,
        execution_policy_sha256=execution_policy.fingerprint,
        request_sha256=hashlib.sha256(request_bytes).hexdigest(),
        request_bytes=len(request_bytes),
        status=status,
        exit_code=exit_code,
        stdout_sha256=hashlib.sha256(stdout).hexdigest(),
        stdout_bytes=len(stdout),
        stdout_truncated=status == "stdout_limit",
        stderr_sha256=hashlib.sha256(stderr).hexdigest(),
        stderr_bytes=len(stderr),
        stderr_truncated=status == "stderr_limit",
    )
    canonical_fingerprint(invocation.to_record())
    return ProposerRun(invocation=invocation, stdout=stdout)
