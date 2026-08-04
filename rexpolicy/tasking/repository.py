"""Loading and validation of production task-authoring artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .canonical import MAX_JSON_BYTES, strict_json_loads
from .capabilities import CapabilityCatalog
from .contract import (
    DEFAULT_TASK_COMPILER_POLICY,
    CompiledTaskContract,
    TaskCompilerPolicy,
    compile_task_contract,
)
from .model import TaskSpecV2
from .process import ProcessSpecV1
from .process_contract import (
    DEFAULT_PROCESS_COMPILER_POLICY,
    CompiledProcessContract,
    ProcessCompilerPolicy,
    compile_process_contract,
)
from .property_validation import (
    DEFAULT_PROPERTY_VALIDATION_POLICY,
    PropertyValidationPolicy,
    TaskPropertyReport,
    validate_task_properties,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
PRODUCTION_CAPABILITIES_PATH = (
    REPOSITORY_ROOT / "configs" / "task_authoring" / "capabilities.v1.json"
)
PRODUCTION_REACH_TASK_PATH = (
    REPOSITORY_ROOT / "configs" / "tasks" / "reach_green_cap.v2.json"
)
PRODUCTION_REACH_PROCESS_PATH = (
    REPOSITORY_ROOT
    / "configs"
    / "process_specs"
    / "reach_pregrasp.v1.json"
)


@dataclass(frozen=True)
class TaskArtifacts:
    capabilities: CapabilityCatalog
    task: TaskSpecV2
    contract: CompiledTaskContract
    property_report: TaskPropertyReport


@dataclass(frozen=True)
class ProcessArtifacts:
    task_artifacts: TaskArtifacts
    process_spec: ProcessSpecV1
    contract: CompiledProcessContract


def _read_json(path: Path) -> object:
    resolved = path.expanduser().resolve()
    try:
        payload = resolved.read_bytes()
    except OSError as error:
        raise ValueError(f"Cannot read task artifact {resolved}") from error
    if len(payload) > MAX_JSON_BYTES:
        raise ValueError(f"Task artifact {resolved} exceeds the size limit")
    try:
        text = payload.decode("utf-8")
    except UnicodeError as error:
        raise ValueError(f"Task artifact {resolved} is not UTF-8") from error
    return strict_json_loads(text)


def load_capability_catalog(path: Path) -> CapabilityCatalog:
    """Load one strict trusted capability catalog."""
    return CapabilityCatalog.from_record(_read_json(path))


def load_task_spec(path: Path) -> TaskSpecV2:
    """Load one strict untrusted TaskSpec proposal/artifact."""
    return TaskSpecV2.from_record(_read_json(path))


def load_process_spec(path: Path) -> ProcessSpecV1:
    """Load one strict untrusted ProcessSpec proposal/artifact."""
    return ProcessSpecV1.from_record(_read_json(path))


def load_task_artifacts(
    *,
    capabilities_path: Path,
    task_path: Path,
    compiler_policy: TaskCompilerPolicy = DEFAULT_TASK_COMPILER_POLICY,
    property_policy: PropertyValidationPolicy = (
        DEFAULT_PROPERTY_VALIDATION_POLICY
    ),
) -> TaskArtifacts:
    """Load, compile, and require property acceptance for one task."""
    capabilities = load_capability_catalog(capabilities_path)
    task = load_task_spec(task_path)
    contract = compile_task_contract(
        task,
        catalog=capabilities,
        policy=compiler_policy,
    )
    report = validate_task_properties(
        task,
        contract,
        catalog=capabilities,
        policy=property_policy,
        compiler_policy=compiler_policy,
    )
    if not report.accepted:
        codes = ", ".join(issue.code for issue in report.issues)
        raise ValueError(f"Task property validation failed: {codes}")
    return TaskArtifacts(
        capabilities=capabilities,
        task=task,
        contract=contract,
        property_report=report,
    )


def load_production_reach_artifacts() -> TaskArtifacts:
    """Return the production Reach v2 contract without test-only fixtures."""
    return load_task_artifacts(
        capabilities_path=PRODUCTION_CAPABILITIES_PATH,
        task_path=PRODUCTION_REACH_TASK_PATH,
    )


def load_process_artifacts(
    *,
    process_path: Path,
    task_artifacts: TaskArtifacts,
    process_policy: ProcessCompilerPolicy = DEFAULT_PROCESS_COMPILER_POLICY,
    task_policy: TaskCompilerPolicy = DEFAULT_TASK_COMPILER_POLICY,
) -> ProcessArtifacts:
    """Load and compile one process contract against exact task artifacts."""
    process_spec = load_process_spec(process_path)
    contract = compile_process_contract(
        process_spec,
        task=task_artifacts.task,
        task_contract=task_artifacts.contract,
        catalog=task_artifacts.capabilities,
        process_policy=process_policy,
        task_policy=task_policy,
    )
    return ProcessArtifacts(
        task_artifacts=task_artifacts,
        process_spec=process_spec,
        contract=contract,
    )


def load_production_reach_process_artifacts(
    task_artifacts: TaskArtifacts | None = None,
) -> ProcessArtifacts:
    """Return the production Reach process contract from its trusted path."""
    return load_process_artifacts(
        process_path=PRODUCTION_REACH_PROCESS_PATH,
        task_artifacts=task_artifacts or load_production_reach_artifacts(),
    )
