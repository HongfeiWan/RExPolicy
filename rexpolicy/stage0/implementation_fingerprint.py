"""Fail-closed provenance for one Stage 0 scientific implementation.

The module deliberately avoids importing simulator or tensor packages.  Their
installed versions are read from distribution metadata so recording provenance
cannot initialize a GPU runtime as a side effect.
"""

from __future__ import annotations

import importlib.metadata
import platform
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .types import canonical_fingerprint


STAGE0_IMPLEMENTATION_FINGERPRINT_SCHEMA_ID = (
    "rexpolicy/stage0-scientific-implementation/v1"
)
STAGE0_IMPLEMENTATION_FINGERPRINT_SCHEMA_VERSION = 1
STAGE0_IMPLEMENTATION_DISTRIBUTIONS = (
    ("torch", "torch"),
    ("warp", "warp-lang"),
    ("newton", "newton"),
    ("mujoco-warp", "mujoco-warp"),
)

_GIT_HEAD = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})\Z")
_SAFE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")
_WORKTREE_POLICY = "tracked-and-untracked-excluding-gitignored/v1"

CommandRunner = Callable[[Sequence[str], Path], str]
DistributionVersionProvider = Callable[[str], str | None]
PythonRuntimeProvider = Callable[[], Mapping[str, str]]


class Stage0ImplementationFingerprintError(RuntimeError):
    """Raised when scientific implementation provenance cannot be trusted."""


@dataclass(frozen=True)
class Stage0DistributionVersion:
    """One exact distribution query and its installed-version outcome."""

    component: str
    distribution: str
    version: str | None

    def __post_init__(self) -> None:
        for value, name in (
            (self.component, "component"),
            (self.distribution, "distribution"),
        ):
            if not isinstance(value, str) or _SAFE_NAME.fullmatch(value) is None:
                raise ValueError(f"{name} must be a safe package identifier")
        if self.version is not None and (
            not isinstance(self.version, str)
            or not self.version
            or self.version != self.version.strip()
        ):
            raise ValueError("distribution version must be clean non-empty text")

    def to_record(self) -> dict[str, Any]:
        return {
            "distribution": self.distribution,
            "status": "missing" if self.version is None else "installed",
            "version": self.version,
        }


@dataclass(frozen=True)
class Stage0ImplementationFingerprint:
    """Canonical, clean-worktree scientific implementation identity."""

    git_head: str
    python_implementation: str
    python_version: str
    distributions: tuple[Stage0DistributionVersion, ...]

    def __post_init__(self) -> None:
        if (
            not isinstance(self.git_head, str)
            or _GIT_HEAD.fullmatch(self.git_head) is None
        ):
            raise ValueError("git_head must be a full lowercase Git object ID")
        for value, name in (
            (self.python_implementation, "python_implementation"),
            (self.python_version, "python_version"),
        ):
            if not isinstance(value, str) or not value or value != value.strip():
                raise ValueError(f"{name} must be clean non-empty text")
        if not self.distributions:
            raise ValueError("distributions cannot be empty")
        components = tuple(item.component for item in self.distributions)
        if len(set(components)) != len(components):
            raise ValueError("distribution components must be unique")

    def to_record(self) -> dict[str, Any]:
        """Return the canonical record; absolute checkout paths are excluded."""

        return {
            "distributions": {
                item.component: item.to_record() for item in self.distributions
            },
            "git": {
                "head": self.git_head,
                "worktree_policy": _WORKTREE_POLICY,
                "worktree_status": "clean",
            },
            "python": {
                "implementation": self.python_implementation,
                "version": self.python_version,
            },
            "schema_id": STAGE0_IMPLEMENTATION_FINGERPRINT_SCHEMA_ID,
            "schema_version": STAGE0_IMPLEMENTATION_FINGERPRINT_SCHEMA_VERSION,
            "version_source": "importlib.metadata.version/v1",
        }

    @property
    def sha256(self) -> str:
        """Canonical SHA-256 of :meth:`to_record`."""

        return canonical_fingerprint(self.to_record())


def _run_command(arguments: Sequence[str], cwd: Path) -> str:
    completed = subprocess.run(
        tuple(arguments),
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout


def _distribution_version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def _python_runtime() -> Mapping[str, str]:
    return {
        "implementation": platform.python_implementation(),
        "version": platform.python_version(),
    }


def _command_output(
    runner: CommandRunner,
    arguments: Sequence[str],
    cwd: Path,
    *,
    purpose: str,
) -> str:
    try:
        output = runner(tuple(arguments), cwd)
    except Exception as error:
        raise Stage0ImplementationFingerprintError(
            f"cannot {purpose}: {type(error).__name__}"
        ) from error
    if not isinstance(output, str):
        raise Stage0ImplementationFingerprintError(
            f"cannot {purpose}: command output is not text"
        )
    return output


def capture_stage0_implementation_fingerprint(
    repository: str | Path | None = None,
    *,
    command_runner: CommandRunner | None = None,
    distribution_version: DistributionVersionProvider | None = None,
    python_runtime: PythonRuntimeProvider | None = None,
) -> Stage0ImplementationFingerprint:
    """Capture a clean Git revision and runtime versions without GPU imports.

    Tracked changes and non-ignored untracked files fail closed.  Git-ignored
    artifacts, such as output directories and caches, are intentionally absent
    because porcelain status excludes them by default.  Missing distributions
    remain explicit records; no import-name aliases or fallback guesses are
    attempted.
    """

    runner = command_runner or _run_command
    version_provider = distribution_version or _distribution_version
    runtime_provider = python_runtime or _python_runtime
    requested = (
        Path(__file__).resolve().parents[2] if repository is None else Path(repository)
    )
    requested = requested.expanduser().resolve()

    root_output = _command_output(
        runner,
        ("git", "rev-parse", "--show-toplevel"),
        requested,
        purpose="locate the Git worktree",
    ).strip()
    if not root_output:
        raise Stage0ImplementationFingerprintError(
            "cannot locate the Git worktree: empty repository root"
        )
    root = Path(root_output).expanduser().resolve()

    head = _command_output(
        runner,
        ("git", "rev-parse", "--verify", "HEAD^{commit}"),
        root,
        purpose="resolve the Git HEAD commit",
    ).strip()
    if _GIT_HEAD.fullmatch(head) is None:
        raise Stage0ImplementationFingerprintError(
            "cannot resolve the Git HEAD commit: invalid full object ID"
        )

    status = _command_output(
        runner,
        (
            "git",
            "--no-optional-locks",
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
            "--ignore-submodules=none",
        ),
        root,
        purpose="verify the Git worktree",
    )
    if status.strip():
        changed = tuple(line for line in status.splitlines() if line.strip())
        preview = "; ".join(changed[:8])
        if len(changed) > 8:
            preview += f"; ... ({len(changed)} entries total)"
        raise Stage0ImplementationFingerprintError(
            "scientific implementation requires a clean Git worktree: " + preview
        )

    try:
        runtime = dict(runtime_provider())
    except Exception as error:
        raise Stage0ImplementationFingerprintError(
            f"cannot inspect the Python runtime: {type(error).__name__}"
        ) from error
    if set(runtime) != {"implementation", "version"}:
        raise Stage0ImplementationFingerprintError(
            "Python runtime metadata must contain exactly implementation and version"
        )

    distributions = []
    for component, distribution in STAGE0_IMPLEMENTATION_DISTRIBUTIONS:
        try:
            version = version_provider(distribution)
        except Exception as error:
            raise Stage0ImplementationFingerprintError(
                f"cannot inspect distribution metadata for {distribution}: "
                f"{type(error).__name__}"
            ) from error
        try:
            item = Stage0DistributionVersion(
                component=component,
                distribution=distribution,
                version=version,
            )
        except (TypeError, ValueError) as error:
            raise Stage0ImplementationFingerprintError(
                f"invalid distribution metadata for {distribution}: {error}"
            ) from error
        distributions.append(item)
    try:
        return Stage0ImplementationFingerprint(
            git_head=head,
            python_implementation=runtime["implementation"],
            python_version=runtime["version"],
            distributions=tuple(distributions),
        )
    except (TypeError, ValueError) as error:
        raise Stage0ImplementationFingerprintError(
            f"invalid scientific implementation metadata: {error}"
        ) from error


__all__ = [
    "STAGE0_IMPLEMENTATION_DISTRIBUTIONS",
    "STAGE0_IMPLEMENTATION_FINGERPRINT_SCHEMA_ID",
    "STAGE0_IMPLEMENTATION_FINGERPRINT_SCHEMA_VERSION",
    "Stage0DistributionVersion",
    "Stage0ImplementationFingerprint",
    "Stage0ImplementationFingerprintError",
    "capture_stage0_implementation_fingerprint",
]
