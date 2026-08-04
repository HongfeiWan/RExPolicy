"""Canonical future-window views over successful Event Ledger paths.

The episode archive and Event Ledger remain the sources of truth.  This module
stores only content bindings and Event Ledger transition locators; observations,
images, frozen features, actions, rewards, and scores never enter the graph.
They can be reconstructed at runtime from the bound, validated ledger.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from rexpolicy.flywheel.experience import (
    DIRECT_SUCCESS_ROLE,
    SUCCESS_PATH_ROLE,
    SUCCESS_ROLES,
    format_sample_id,
    validate_candidate_archive_record,
)
from rexpolicy.flywheel.success_archive import SuccessReference
from rexpolicy.tasking.canonical import (
    canonical_fingerprint,
    canonical_json,
    strict_json_loads,
)
from rexpolicy.tasking.event_ledger import (
    DecisionClosedEvent,
    DecisionRootEvent,
    EpisodeEventLedger,
    SignalFrame,
    TransitionEvent,
)

_IDENTIFIER = re.compile(r"^[a-z][a-z0-9_.-]*(?:/[a-z0-9_.-]+)*$")
_EPISODE_ID = re.compile(r"^g[0-9]{6}-r[0-9]{5}-e[0-9]{5}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_MAX_HORIZON = 4096
_EPISODE_FIELDS = {
    "schema_version",
    "generation",
    "data_generation",
    "sampling_policy_generation",
    "policy_version",
    "rank",
    "episode",
    "task_id",
    "reward_profile_id",
    "simulator_fingerprint",
    "reset_recipe",
    "instruction",
    "selected_path",
    "decisions",
    "success_samples",
}
_DECISION_FIELDS = {
    "decision_index",
    "start_control_step",
    "physical_replay_gate",
    "advantage_baseline",
    "selection_mode",
    "success_constraint_active",
    "safe_candidate_count",
    "selected_chunk_count",
    "normalized_action_pairwise_rms_mean",
    "normalized_action_pairwise_rms_max",
    "candidates",
}
_CANDIDATE_FIELDS = {
    "sample_id",
    "success_roles",
    "world",
    "score",
    "success",
    "failure",
    "safety_violation",
    "failure_reasons",
    "terminated",
    "truncated",
    "valid_steps",
    "rewards",
    "advantage",
    "sample_weight",
    "selected_for_training",
    "chosen_for_continuation",
    "action",
}
_SUCCESS_SAMPLE_FIELDS = {
    "sample_id",
    "generation",
    "rank",
    "episode",
    "decision",
    "world",
    "roles",
    "weight",
}
_FORBIDDEN_PERSISTED_FIELDS = frozenset(
    {
        "action",
        "action_chunk",
        "current_state",
        "feature",
        "features",
        "future_state",
        "future_state_sequence",
        "image",
        "images",
        "reward",
        "rewards",
        "score",
        "success_score",
        "trajectory_return",
        "visual_feature",
    }
)


def _mapping(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{path} must be an object")
    return value


def _keys(record: Mapping[str, Any], expected: set[str], path: str) -> None:
    missing = sorted(expected.difference(record))
    unknown = sorted(set(record).difference(expected))
    if missing:
        raise ValueError(f"{path} is missing fields: {', '.join(missing)}")
    if unknown:
        raise ValueError(f"{path} has unknown fields: {', '.join(unknown)}")


def _integer(value: Any, path: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{path} must be an integer >= {minimum}")
    return value


def _finite(value: Any, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{path} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{path} must be finite")
    return number


def _identifier(value: Any, path: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise ValueError(f"{path} must be a versioned identifier")
    return value


def _sha256(value: Any, path: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{path} must be a lowercase SHA-256")
    return value


def _episode_id(value: Any, path: str) -> str:
    if not isinstance(value, str) or _EPISODE_ID.fullmatch(value) is None:
        raise ValueError(f"{path} must be a canonical episode ID")
    return value


def _relative_path(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{path} must be a non-empty relative path")
    candidate = Path(value)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError(f"{path} must be run-relative")
    if candidate.as_posix() != value:
        raise ValueError(f"{path} must use canonical POSIX separators")
    return value


def _roles(value: Any, path: str, *, require_nonempty: bool) -> tuple[str, ...]:
    if not isinstance(value, (tuple, list)):
        raise ValueError(f"{path} must be an array")
    roles = tuple(value)
    if any(not isinstance(role, str) or role not in SUCCESS_ROLES for role in roles):
        raise ValueError(f"{path} contains an unsupported success role")
    if require_nonempty and not roles:
        raise ValueError(f"{path} must be non-empty")
    if roles != tuple(sorted(set(roles))):
        raise ValueError(f"{path} must be sorted and unique")
    return roles


def _assert_metadata_only(value: Any, path: str = "$success_experience_graph") -> None:
    if isinstance(value, dict):
        forbidden = _FORBIDDEN_PERSISTED_FIELDS.intersection(value)
        if forbidden:
            raise ValueError(
                f"{path} contains materialized fields: {', '.join(sorted(forbidden))}"
            )
        for key, child in value.items():
            _assert_metadata_only(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _assert_metadata_only(child, f"{path}[{index}]")


@dataclass(frozen=True)
class FutureWindowPolicy:
    """Deterministic policy for slicing every successful transition path."""

    action_horizon: int
    future_horizon: int
    stride: int = 1
    schema_version: int = 1
    policy_id: str = "success_future_window/v1"

    @classmethod
    def from_record(cls, value: Any) -> FutureWindowPolicy:
        path = "$future_window_policy"
        record = _mapping(value, path)
        _keys(
            record,
            {
                "schema_version",
                "policy_id",
                "action_horizon",
                "future_horizon",
                "stride",
            },
            path,
        )
        policy = cls(
            action_horizon=record["action_horizon"],
            future_horizon=record["future_horizon"],
            stride=record["stride"],
            schema_version=record["schema_version"],
            policy_id=record["policy_id"],
        )
        policy.validate()
        return policy

    def validate(self) -> None:
        if self.schema_version != 1:
            raise ValueError("FutureWindowPolicy schema_version must be 1")
        if self.policy_id != "success_future_window/v1":
            raise ValueError("Unsupported future-window policy ID")
        for name, value in (
            ("action_horizon", self.action_horizon),
            ("future_horizon", self.future_horizon),
            ("stride", self.stride),
        ):
            _integer(value, name, minimum=1)
            if value > _MAX_HORIZON:
                raise ValueError(f"{name} exceeds {_MAX_HORIZON}")
        if self.action_horizon > self.future_horizon:
            raise ValueError("action_horizon cannot exceed future_horizon")

    def to_record(self) -> dict[str, Any]:
        self.validate()
        return {
            "schema_version": self.schema_version,
            "policy_id": self.policy_id,
            "action_horizon": self.action_horizon,
            "future_horizon": self.future_horizon,
            "stride": self.stride,
        }

    @property
    def fingerprint(self) -> str:
        return canonical_fingerprint(self.to_record())

    def offsets(self, transition_count: int) -> tuple[int, ...]:
        _integer(transition_count, "transition_count", minimum=1)
        return tuple(range(0, transition_count, self.stride))


@dataclass(frozen=True)
class FutureWindowLocator:
    """A compact locator for one future window in a bound Event Ledger."""

    schema_version: int
    episode_id: str
    event_ledger_sha256: str
    terminal_sample_id: str
    transition_sequences: tuple[int, ...]
    action_steps: int

    @classmethod
    def from_record(cls, value: Any) -> FutureWindowLocator:
        path = "$future_window_locator"
        record = _mapping(value, path)
        _keys(
            record,
            {
                "schema_version",
                "episode_id",
                "event_ledger_sha256",
                "terminal_sample_id",
                "transition_sequences",
                "action_steps",
            },
            path,
        )
        raw_sequences = record["transition_sequences"]
        if not isinstance(raw_sequences, list):
            raise ValueError(f"{path}.transition_sequences must be an array")
        locator = cls(
            schema_version=record["schema_version"],
            episode_id=record["episode_id"],
            event_ledger_sha256=record["event_ledger_sha256"],
            terminal_sample_id=record["terminal_sample_id"],
            transition_sequences=tuple(raw_sequences),
            action_steps=record["action_steps"],
        )
        locator.validate()
        return locator

    def validate(self) -> None:
        if self.schema_version != 1:
            raise ValueError("FutureWindowLocator schema_version must be 1")
        _episode_id(self.episode_id, "episode_id")
        _sha256(self.event_ledger_sha256, "event_ledger_sha256")
        _identifier(self.terminal_sample_id, "terminal_sample_id")
        sequences = tuple(self.transition_sequences)
        if not sequences:
            raise ValueError("Future-window transition sequences cannot be empty")
        for sequence in sequences:
            _integer(sequence, "transition sequence")
        if sequences != tuple(sorted(set(sequences))):
            raise ValueError("Future-window transition sequences must increase")
        _integer(self.action_steps, "action_steps", minimum=1)
        if self.action_steps > len(sequences):
            raise ValueError("Future-window action span exceeds its future span")

    def to_record(self) -> dict[str, Any]:
        self.validate()
        return {
            "schema_version": self.schema_version,
            "episode_id": self.episode_id,
            "event_ledger_sha256": self.event_ledger_sha256,
            "terminal_sample_id": self.terminal_sample_id,
            "transition_sequences": list(self.transition_sequences),
            "action_steps": self.action_steps,
        }

    @property
    def current_transition_sequence(self) -> int:
        return self.transition_sequences[0]


@dataclass(frozen=True)
class StateWitness:
    """Transient, reward-free evidence for one state on a ledger path."""

    control_step: int
    source_event_sequence: int
    dynamics_digest_sha256: str
    signals: SignalFrame


@dataclass(frozen=True)
class MaterializedSuccessExperience:
    """Runtime-only tensors' source values recovered from an Event Ledger."""

    current_witness: StateWitness
    action_chunk: tuple[tuple[float, ...], ...]
    future_witnesses: tuple[StateWitness, ...]

    @property
    def future_witness(self) -> StateWitness:
        """Return the final state witness in the configured future window."""
        return self.future_witnesses[-1]


@dataclass(frozen=True)
class TerminalSuccessPath:
    """One terminal success branch plus its shared continuation prefix."""

    schema_version: int
    trajectory_id: str
    terminal_sample_id: str
    terminal_success_reference_sha256: str
    transition_sequences: tuple[int, ...]

    @classmethod
    def from_record(cls, value: Any) -> TerminalSuccessPath:
        path = "$terminal_success_path"
        record = _mapping(value, path)
        _keys(
            record,
            {
                "schema_version",
                "trajectory_id",
                "terminal_sample_id",
                "terminal_success_reference_sha256",
                "transition_sequences",
            },
            path,
        )
        raw_sequences = record["transition_sequences"]
        if not isinstance(raw_sequences, list):
            raise ValueError(f"{path}.transition_sequences must be an array")
        result = cls(
            schema_version=record["schema_version"],
            trajectory_id=record["trajectory_id"],
            terminal_sample_id=record["terminal_sample_id"],
            terminal_success_reference_sha256=record[
                "terminal_success_reference_sha256"
            ],
            transition_sequences=tuple(raw_sequences),
        )
        result.validate()
        return result

    def validate(self) -> None:
        if self.schema_version != 1:
            raise ValueError("TerminalSuccessPath schema_version must be 1")
        _sha256(self.trajectory_id, "trajectory_id")
        _identifier(self.terminal_sample_id, "terminal_sample_id")
        _sha256(
            self.terminal_success_reference_sha256,
            "terminal_success_reference_sha256",
        )
        sequences = tuple(self.transition_sequences)
        if not sequences:
            raise ValueError("Terminal success paths cannot be empty")
        for sequence in sequences:
            _integer(sequence, "transition sequence")
        if sequences != tuple(sorted(set(sequences))):
            raise ValueError("Terminal success path sequences must increase")

    def to_record(self) -> dict[str, Any]:
        self.validate()
        return {
            "schema_version": self.schema_version,
            "trajectory_id": self.trajectory_id,
            "terminal_sample_id": self.terminal_sample_id,
            "terminal_success_reference_sha256": (
                self.terminal_success_reference_sha256
            ),
            "transition_sequences": list(self.transition_sequences),
        }


@dataclass(frozen=True)
class SuccessExperience:
    """One metadata-only future-conditioned training example."""

    schema_version: int
    experience_id: str
    trajectory_id: str
    environment_seed: int
    locator: FutureWindowLocator

    @classmethod
    def from_record(cls, value: Any) -> SuccessExperience:
        path = "$success_experience"
        record = _mapping(value, path)
        _keys(
            record,
            {
                "schema_version",
                "experience_id",
                "trajectory_id",
                "environment_seed",
                "locator",
            },
            path,
        )
        result = cls(
            schema_version=record["schema_version"],
            experience_id=record["experience_id"],
            trajectory_id=record["trajectory_id"],
            environment_seed=record["environment_seed"],
            locator=FutureWindowLocator.from_record(record["locator"]),
        )
        result.validate()
        return result

    def validate(self) -> None:
        if self.schema_version != 1:
            raise ValueError("SuccessExperience schema_version must be 1")
        _sha256(self.experience_id, "experience_id")
        _sha256(self.trajectory_id, "trajectory_id")
        _integer(self.environment_seed, "environment_seed")
        canonical_locator = FutureWindowLocator.from_record(self.locator.to_record())
        if canonical_locator != self.locator:
            raise ValueError("SuccessExperience locator is not canonical")

    def to_record(self) -> dict[str, Any]:
        self.validate()
        return {
            "schema_version": self.schema_version,
            "experience_id": self.experience_id,
            "trajectory_id": self.trajectory_id,
            "environment_seed": self.environment_seed,
            "locator": self.locator.to_record(),
        }

    def materialize(
        self,
        event_ledger: EpisodeEventLedger,
    ) -> MaterializedSuccessExperience:
        """Recover current evidence, actions, and future evidence at runtime."""
        self.validate()
        if not isinstance(event_ledger, EpisodeEventLedger):
            raise TypeError("materialize requires an EpisodeEventLedger")
        if event_ledger.fingerprint != self.locator.event_ledger_sha256:
            raise ValueError("SuccessExperience Event Ledger fingerprint mismatch")
        if event_ledger.episode_id != self.locator.episode_id:
            raise ValueError("SuccessExperience Event Ledger episode mismatch")
        if event_ledger.reset_recipe.seed != self.environment_seed:
            raise ValueError("SuccessExperience reset seed mismatch")
        path = _terminal_path_sequences(
            event_ledger,
            self.locator.terminal_sample_id,
        )
        first = self.locator.current_transition_sequence
        try:
            offset = path.index(first)
        except ValueError as error:
            raise ValueError(
                "Future-window transition is not on its success path"
            ) from error
        located = self.locator.transition_sequences
        if path[offset : offset + len(located)] != located:
            raise ValueError(
                "Future-window transitions are not a contiguous path slice"
            )

        blocks = _ledger_decisions(event_ledger)
        by_sequence = {event.sequence: event for event in event_ledger.events}
        current_transition = by_sequence[first]
        if not isinstance(current_transition, TransitionEvent):
            raise ValueError("Future-window locator does not identify a transition")
        block = blocks[current_transition.decision_index]
        candidate = block.transitions[current_transition.sample_id]
        if current_transition.branch_step == 0:
            current_witness = _root_witness(block.root)
        else:
            current_witness = _post_witness(
                block.root,
                candidate[current_transition.branch_step - 1],
            )

        transitions = []
        for sequence in located:
            event = by_sequence.get(sequence)
            if not isinstance(event, TransitionEvent):
                raise ValueError(
                    "Future-window locator contains a non-transition event"
                )
            transitions.append(event)
        actions = tuple(
            event.effective_action_19d
            for event in transitions[: self.locator.action_steps]
        )
        future = tuple(
            _post_witness(blocks[event.decision_index].root, event)
            for event in transitions
        )
        return MaterializedSuccessExperience(
            current_witness=current_witness,
            action_chunk=actions,
            future_witnesses=future,
        )


@dataclass(frozen=True)
class SuccessExperienceGraph:
    """A content-addressed, rebuildable graph of successful future windows."""

    schema_version: int
    artifact_type: str
    episode_id: str
    task_id: str
    task_contract_id: str
    task_contract_sha256: str
    simulator_fingerprint: str
    source_episode_path: str
    source_episode_sha256: str
    source_episode_record_index: int
    source_success_set_sha256: str
    event_ledger_sha256: str
    environment_adapter_id: str
    environment_seed: int
    window_policy: FutureWindowPolicy
    window_policy_sha256: str
    terminal_paths: tuple[TerminalSuccessPath, ...]
    experiences: tuple[SuccessExperience, ...]

    @classmethod
    def compile(
        cls,
        *,
        episode_record: Mapping[str, Any],
        event_ledger: EpisodeEventLedger,
        success_references: Iterable[SuccessReference],
        window_policy: FutureWindowPolicy,
    ) -> SuccessExperienceGraph:
        """Compile the graph from one successful committed archive record."""
        return compile_success_experience_graph(
            episode_record=episode_record,
            event_ledger=event_ledger,
            success_references=success_references,
            window_policy=window_policy,
        )

    @classmethod
    def from_record(cls, value: Any) -> SuccessExperienceGraph:
        path = "$success_experience_graph"
        record = _mapping(value, path)
        _keys(
            record,
            {
                "schema_version",
                "artifact_type",
                "episode_id",
                "task_id",
                "task_contract_id",
                "task_contract_sha256",
                "simulator_fingerprint",
                "source_episode_path",
                "source_episode_sha256",
                "source_episode_record_index",
                "source_success_set_sha256",
                "event_ledger_sha256",
                "environment_adapter_id",
                "environment_seed",
                "window_policy",
                "window_policy_sha256",
                "terminal_paths",
                "experiences",
            },
            path,
        )
        raw_paths = record["terminal_paths"]
        raw_experiences = record["experiences"]
        if not isinstance(raw_paths, list):
            raise ValueError(f"{path}.terminal_paths must be an array")
        if not isinstance(raw_experiences, list):
            raise ValueError(f"{path}.experiences must be an array")
        graph = cls(
            schema_version=record["schema_version"],
            artifact_type=record["artifact_type"],
            episode_id=record["episode_id"],
            task_id=record["task_id"],
            task_contract_id=record["task_contract_id"],
            task_contract_sha256=record["task_contract_sha256"],
            simulator_fingerprint=record["simulator_fingerprint"],
            source_episode_path=record["source_episode_path"],
            source_episode_sha256=record["source_episode_sha256"],
            source_episode_record_index=record["source_episode_record_index"],
            source_success_set_sha256=record["source_success_set_sha256"],
            event_ledger_sha256=record["event_ledger_sha256"],
            environment_adapter_id=record["environment_adapter_id"],
            environment_seed=record["environment_seed"],
            window_policy=FutureWindowPolicy.from_record(record["window_policy"]),
            window_policy_sha256=record["window_policy_sha256"],
            terminal_paths=tuple(
                TerminalSuccessPath.from_record(item) for item in raw_paths
            ),
            experiences=tuple(
                SuccessExperience.from_record(item) for item in raw_experiences
            ),
        )
        graph.validate()
        if graph.to_record() != record:
            raise ValueError("SuccessExperienceGraph is not exactly canonical")
        return graph

    @classmethod
    def from_json(cls, text: str) -> SuccessExperienceGraph:
        return cls.from_record(strict_json_loads(text))

    def validate(self) -> None:
        if self.schema_version != 1:
            raise ValueError("SuccessExperienceGraph schema_version must be 1")
        if self.artifact_type != "rexpolicy_success_experience_graph":
            raise ValueError("Unsupported success-experience graph artifact type")
        _episode_id(self.episode_id, "episode_id")
        _identifier(self.task_id, "task_id")
        _identifier(self.task_contract_id, "task_contract_id")
        _sha256(self.task_contract_sha256, "task_contract_sha256")
        _sha256(self.simulator_fingerprint, "simulator_fingerprint")
        _relative_path(self.source_episode_path, "source_episode_path")
        _sha256(self.source_episode_sha256, "source_episode_sha256")
        _integer(self.source_episode_record_index, "source_episode_record_index")
        _sha256(self.source_success_set_sha256, "source_success_set_sha256")
        _sha256(self.event_ledger_sha256, "event_ledger_sha256")
        _identifier(self.environment_adapter_id, "environment_adapter_id")
        _integer(self.environment_seed, "environment_seed")
        canonical_policy = FutureWindowPolicy.from_record(
            self.window_policy.to_record()
        )
        if canonical_policy != self.window_policy:
            raise ValueError("Future-window policy is not canonical")
        if self.window_policy_sha256 != self.window_policy.fingerprint:
            raise ValueError("Future-window policy fingerprint mismatch")

        paths = tuple(self.terminal_paths)
        if not paths:
            raise ValueError("SuccessExperienceGraph requires a terminal success path")
        for item in paths:
            if TerminalSuccessPath.from_record(item.to_record()) != item:
                raise ValueError("Terminal success path is not canonical")
        if paths != tuple(sorted(paths, key=lambda item: item.terminal_sample_id)):
            raise ValueError("Terminal success paths must be sorted by sample ID")
        for item in paths:
            expected_id = _trajectory_id(
                episode_id=self.episode_id,
                event_ledger_sha256=self.event_ledger_sha256,
                terminal_reference_sha256=(
                    item.terminal_success_reference_sha256
                ),
                transition_sequences=item.transition_sequences,
            )
            if item.trajectory_id != expected_id:
                raise ValueError("Terminal success trajectory identity mismatch")
        terminal_ids = tuple(item.terminal_sample_id for item in paths)
        trajectory_ids = tuple(item.trajectory_id for item in paths)
        if len(set(terminal_ids)) != len(terminal_ids):
            raise ValueError("Terminal success sample IDs must be unique")
        if len(set(trajectory_ids)) != len(trajectory_ids):
            raise ValueError("Success trajectory IDs must be unique")

        experiences = tuple(self.experiences)
        if not experiences:
            raise ValueError("SuccessExperienceGraph requires future windows")
        for item in experiences:
            if SuccessExperience.from_record(item.to_record()) != item:
                raise ValueError("SuccessExperience is not canonical")
        expected_order = tuple(
            sorted(
                experiences,
                key=lambda item: (
                    item.trajectory_id,
                    item.locator.current_transition_sequence,
                ),
            )
        )
        if experiences != expected_order:
            raise ValueError("Success experiences are not canonically sorted")
        experience_ids = tuple(item.experience_id for item in experiences)
        if len(set(experience_ids)) != len(experience_ids):
            raise ValueError("Success experience IDs must be unique")

        by_trajectory = {item.trajectory_id: item for item in paths}
        actual_windows: dict[str, list[SuccessExperience]] = {
            trajectory_id: [] for trajectory_id in by_trajectory
        }
        for item in experiences:
            path = by_trajectory.get(item.trajectory_id)
            if path is None:
                raise ValueError("SuccessExperience references an unknown trajectory")
            locator = item.locator
            if item.environment_seed != self.environment_seed:
                raise ValueError("SuccessExperience environment seed mismatch")
            if (
                locator.episode_id != self.episode_id
                or locator.event_ledger_sha256 != self.event_ledger_sha256
                or locator.terminal_sample_id != path.terminal_sample_id
            ):
                raise ValueError("SuccessExperience locator source mismatch")
            expected_id = _experience_id(
                trajectory_id=item.trajectory_id,
                window_policy_sha256=self.window_policy_sha256,
                environment_seed=item.environment_seed,
                locator=locator,
            )
            if item.experience_id != expected_id:
                raise ValueError("SuccessExperience identity mismatch")
            actual_windows[item.trajectory_id].append(item)

        for trajectory_id, path in by_trajectory.items():
            expected = []
            for offset in self.window_policy.offsets(len(path.transition_sequences)):
                sequences = path.transition_sequences[
                    offset : offset + self.window_policy.future_horizon
                ]
                action_steps = min(
                    self.window_policy.action_horizon,
                    len(path.transition_sequences) - offset,
                )
                expected.append((sequences, action_steps))
            actual = [
                (item.locator.transition_sequences, item.locator.action_steps)
                for item in actual_windows[trajectory_id]
            ]
            if actual != expected:
                raise ValueError("SuccessExperience windows do not match their policy")
        _assert_metadata_only(self.to_record())

    def to_record(self) -> dict[str, Any]:
        record = {
            "schema_version": self.schema_version,
            "artifact_type": self.artifact_type,
            "episode_id": self.episode_id,
            "task_id": self.task_id,
            "task_contract_id": self.task_contract_id,
            "task_contract_sha256": self.task_contract_sha256,
            "simulator_fingerprint": self.simulator_fingerprint,
            "source_episode_path": self.source_episode_path,
            "source_episode_sha256": self.source_episode_sha256,
            "source_episode_record_index": self.source_episode_record_index,
            "source_success_set_sha256": self.source_success_set_sha256,
            "event_ledger_sha256": self.event_ledger_sha256,
            "environment_adapter_id": self.environment_adapter_id,
            "environment_seed": self.environment_seed,
            "window_policy": self.window_policy.to_record(),
            "window_policy_sha256": self.window_policy_sha256,
            "terminal_paths": [item.to_record() for item in self.terminal_paths],
            "experiences": [item.to_record() for item in self.experiences],
        }
        _assert_metadata_only(record)
        return record

    def to_json(self) -> str:
        self.validate()
        return canonical_json(self.to_record())

    @property
    def fingerprint(self) -> str:
        return canonical_fingerprint(self.to_record())

    def materialize(
        self,
        experience_id: str,
        event_ledger: EpisodeEventLedger,
    ) -> MaterializedSuccessExperience:
        """Materialize one graph member by its content-derived identity."""
        for experience in self.experiences:
            if experience.experience_id == experience_id:
                return experience.materialize(event_ledger)
        raise KeyError(f"Unknown SuccessExperience {experience_id}")


@dataclass(frozen=True)
class _LedgerDecision:
    root: DecisionRootEvent
    transitions: dict[str, tuple[TransitionEvent, ...]]
    closed: DecisionClosedEvent


@dataclass(frozen=True)
class _ValidatedEpisode:
    generation: int
    rank: int
    episode: int
    task_id: str
    reward_profile_id: str
    simulator_fingerprint: str
    reset_seed: int
    candidates: dict[str, dict[str, Any]]
    candidate_decisions: dict[str, int]
    success_roles: dict[str, tuple[str, ...]]
    success_weights: dict[str, float]


def _ledger_decisions(event_ledger: EpisodeEventLedger) -> tuple[_LedgerDecision, ...]:
    decisions = []
    cursor = 0
    events = event_ledger.events
    while cursor < len(events):
        root = events[cursor]
        if not isinstance(root, DecisionRootEvent):
            raise ValueError("Event Ledger decision is missing its root")
        cursor += 1
        candidates: dict[str, list[TransitionEvent]] = {}
        closed = None
        while cursor < len(events):
            event = events[cursor]
            cursor += 1
            if isinstance(event, TransitionEvent):
                candidates.setdefault(event.sample_id, []).append(event)
                continue
            if isinstance(event, DecisionClosedEvent):
                closed = event
                break
            raise ValueError("Event Ledger decision has invalid event ordering")
        if closed is None:
            raise ValueError("Event Ledger decision is not closed")
        decisions.append(
            _LedgerDecision(
                root=root,
                transitions={key: tuple(value) for key, value in candidates.items()},
                closed=closed,
            )
        )
    return tuple(decisions)


def _terminal_path_sequences(
    event_ledger: EpisodeEventLedger,
    terminal_sample_id: str,
) -> tuple[int, ...]:
    blocks = _ledger_decisions(event_ledger)
    terminal_decision = None
    for index, block in enumerate(blocks):
        if terminal_sample_id in block.transitions:
            terminal_decision = index
            break
    if terminal_decision is None:
        raise ValueError("Terminal success reference is absent from the Event Ledger")
    if terminal_decision != len(blocks) - 1:
        raise ValueError("Terminal success branch must belong to the final decision")
    path: list[int] = []
    for index, block in enumerate(blocks[: terminal_decision + 1]):
        sample_id = (
            terminal_sample_id
            if index == terminal_decision
            else block.closed.continued_sample_id
        )
        if sample_id is None or sample_id not in block.transitions:
            raise ValueError("Terminal success path has a missing continuation")
        path.extend(event.sequence for event in block.transitions[sample_id])
    return tuple(path)


def _root_witness(root: DecisionRootEvent) -> StateWitness:
    return StateWitness(
        control_step=root.root_control_step,
        source_event_sequence=root.sequence,
        dynamics_digest_sha256=root.dynamics_digest_sha256,
        signals=root.signals,
    )


def _post_witness(
    root: DecisionRootEvent,
    transition: TransitionEvent,
) -> StateWitness:
    return StateWitness(
        control_step=root.root_control_step + transition.branch_step + 1,
        source_event_sequence=transition.sequence,
        dynamics_digest_sha256=transition.post_dynamics_digest_sha256,
        signals=transition.post_signals,
    )


def _validate_episode_schema5(value: Any) -> _ValidatedEpisode:
    path = "$episode"
    record = _mapping(value, path)
    _keys(record, _EPISODE_FIELDS, path)
    if record["schema_version"] != 5:
        raise ValueError("SuccessExperience extraction requires episode schema 5")
    generation = _integer(record["generation"], f"{path}.generation", minimum=1)
    if record["data_generation"] != generation:
        raise ValueError("Episode generation aliases do not match")
    sampling_generation = _integer(
        record["sampling_policy_generation"],
        f"{path}.sampling_policy_generation",
    )
    if sampling_generation != generation - 1:
        raise ValueError("Episode sampling policy must precede data generation")
    if record["policy_version"] != f"generation-{sampling_generation:06d}":
        raise ValueError("Episode policy version is not canonical")
    rank = _integer(record["rank"], f"{path}.rank")
    episode = _integer(record["episode"], f"{path}.episode")
    task_id = _identifier(record["task_id"], f"{path}.task_id")
    reward_profile_id = _identifier(
        record["reward_profile_id"], f"{path}.reward_profile_id"
    )
    simulator_fingerprint = _sha256(
        record["simulator_fingerprint"], f"{path}.simulator_fingerprint"
    )
    if not isinstance(record["instruction"], str) or not record["instruction"].strip():
        raise ValueError("Successful episode instruction cannot be empty")

    reset = _mapping(record["reset_recipe"], f"{path}.reset_recipe")
    _keys(reset, {"environment", "seed", "initial_state"}, f"{path}.reset_recipe")
    if not isinstance(reset["environment"], str) or not reset["environment"]:
        raise ValueError("Episode reset environment cannot be empty")
    reset_seed = _integer(reset["seed"], f"{path}.reset_recipe.seed")
    _mapping(reset["initial_state"], f"{path}.reset_recipe.initial_state")

    selected = _mapping(record["selected_path"], f"{path}.selected_path")
    _keys(selected, {"score", "success", "rewards", "actions"}, f"{path}.selected_path")
    if selected["success"] is not True:
        raise ValueError("SuccessExperience extraction requires a successful episode")
    _finite(selected["score"], f"{path}.selected_path.score")
    if not isinstance(selected["rewards"], list):
        raise ValueError("Episode selected rewards must be an array")
    for index, reward in enumerate(selected["rewards"]):
        _finite(reward, f"{path}.selected_path.rewards[{index}]")
    if not isinstance(selected["actions"], list):
        raise ValueError("Episode selected actions must be an array")

    raw_decisions = record["decisions"]
    if not isinstance(raw_decisions, list) or not raw_decisions:
        raise ValueError("Successful episode decisions must be a non-empty array")
    candidates: dict[str, dict[str, Any]] = {}
    candidate_decisions = {}
    for decision_index, raw_decision in enumerate(raw_decisions):
        decision_path = f"{path}.decisions[{decision_index}]"
        decision = _mapping(raw_decision, decision_path)
        _keys(decision, _DECISION_FIELDS, decision_path)
        if decision["decision_index"] != decision_index:
            raise ValueError("Episode decision indexes must be contiguous")
        _integer(decision["start_control_step"], f"{decision_path}.start_control_step")
        raw_candidates = decision["candidates"]
        if not isinstance(raw_candidates, list) or not raw_candidates:
            raise ValueError(f"{decision_path}.candidates must be non-empty")
        for candidate_index, raw_candidate in enumerate(raw_candidates):
            candidate_path = f"{decision_path}.candidates[{candidate_index}]"
            candidate = _mapping(raw_candidate, candidate_path)
            _keys(candidate, _CANDIDATE_FIELDS, candidate_path)
            validate_candidate_archive_record(candidate)
            world = _integer(candidate["world"], f"{candidate_path}.world")
            expected_id = format_sample_id(
                generation,
                rank,
                episode,
                decision_index,
                world,
            )
            if candidate["sample_id"] != expected_id:
                raise ValueError("Episode candidate sample identity is inconsistent")
            _roles(
                candidate["success_roles"],
                f"{candidate_path}.success_roles",
                require_nonempty=False,
            )
            if not isinstance(candidate["action"], dict):
                raise ValueError(f"{candidate_path}.action must be an object")
            if expected_id in candidates:
                raise ValueError("Episode candidate sample IDs must be unique")
            candidates[expected_id] = candidate
            candidate_decisions[expected_id] = decision_index

    raw_success_samples = record["success_samples"]
    if not isinstance(raw_success_samples, list) or not raw_success_samples:
        raise ValueError("Successful episode must name its success samples")
    success_roles = {}
    success_weights = {}
    for index, raw_sample in enumerate(raw_success_samples):
        sample_path = f"{path}.success_samples[{index}]"
        sample = _mapping(raw_sample, sample_path)
        _keys(sample, _SUCCESS_SAMPLE_FIELDS, sample_path)
        provenance = (
            sample["generation"],
            sample["rank"],
            sample["episode"],
            sample["decision"],
            sample["world"],
        )
        if provenance[:3] != (generation, rank, episode):
            raise ValueError("Episode success-sample provenance mismatch")
        if any(type(item) is not int or item < 0 for item in provenance):
            raise ValueError("Episode success-sample provenance is invalid")
        expected_id = format_sample_id(*provenance)
        if sample["sample_id"] != expected_id or expected_id not in candidates:
            raise ValueError("Episode success sample is not an archived candidate")
        roles = _roles(sample["roles"], f"{sample_path}.roles", require_nonempty=True)
        candidate_roles = _roles(
            candidates[expected_id]["success_roles"],
            f"{sample_path}.candidate_roles",
            require_nonempty=True,
        )
        if roles != candidate_roles:
            raise ValueError("Episode success roles disagree with its candidate")
        if DIRECT_SUCCESS_ROLE in roles and not candidates[expected_id]["success"]:
            raise ValueError("Direct-success role requires a successful candidate")
        if expected_id in success_roles:
            raise ValueError("Episode success sample IDs must be unique")
        weight = _finite(sample["weight"], f"{sample_path}.weight")
        if weight <= 0.0:
            raise ValueError("Episode success-sample weight must be positive")
        success_roles[expected_id] = roles
        success_weights[expected_id] = weight
    role_bearing_candidates = {
        sample_id
        for sample_id, candidate in candidates.items()
        if candidate["success_roles"]
    }
    if role_bearing_candidates != set(success_roles):
        raise ValueError("Episode success_samples do not cover every success role")
    for sample_id, candidate in candidates.items():
        direct = DIRECT_SUCCESS_ROLE in success_roles.get(sample_id, ())
        if bool(candidate["success"]) != direct:
            raise ValueError("Episode direct-success roles do not match outcomes")
    # This checks that arbitrary initial-state/action metadata is finite JSON, but
    # none of it is copied into the graph.
    canonical_json(record)
    return _ValidatedEpisode(
        generation=generation,
        rank=rank,
        episode=episode,
        task_id=task_id,
        reward_profile_id=reward_profile_id,
        simulator_fingerprint=simulator_fingerprint,
        reset_seed=reset_seed,
        candidates=candidates,
        candidate_decisions=candidate_decisions,
        success_roles=success_roles,
        success_weights=success_weights,
    )


def _canonical_references(
    references: Iterable[SuccessReference],
    episode: _ValidatedEpisode,
) -> tuple[SuccessReference, ...]:
    canonical = []
    for index, reference in enumerate(references):
        if not isinstance(reference, SuccessReference):
            raise ValueError(f"success_references[{index}] must be a SuccessReference")
        restored = SuccessReference.from_record(reference.to_record())
        if restored != reference:
            raise ValueError("Success reference is not canonical")
        if tuple(reference.success_roles) != tuple(
            sorted(set(reference.success_roles))
        ):
            raise ValueError("Success reference roles must be sorted and unique")
        if (
            reference.source_generation != episode.generation
            or reference.source_rank != episode.rank
            or reference.source_episode != episode.episode
        ):
            raise ValueError("Success reference belongs to another episode")
        expected_sample_id = format_sample_id(
            reference.source_generation,
            reference.source_rank,
            reference.source_episode,
            reference.decision,
            reference.world,
        )
        if reference.sample_id != expected_sample_id:
            raise ValueError("Success reference sample provenance is inconsistent")
        if reference.task_id != episode.task_id:
            raise ValueError("Success reference task does not match its episode")
        if reference.reward_profile_id != episode.reward_profile_id:
            raise ValueError(
                "Success reference reward profile does not match its episode"
            )
        if reference.simulator_fingerprint != episode.simulator_fingerprint:
            raise ValueError("Success reference simulator does not match its episode")
        roles = episode.success_roles.get(reference.sample_id)
        if roles is None or tuple(reference.success_roles) != roles:
            raise ValueError("Success reference is absent from episode success_samples")
        if reference.archived_weight != episode.success_weights[reference.sample_id]:
            raise ValueError("Success reference weight does not match its episode")
        canonical.append(restored)
    canonical.sort(key=lambda item: item.sample_id)
    if not canonical:
        raise ValueError("SuccessExperience extraction requires committed references")
    sample_ids = tuple(item.sample_id for item in canonical)
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("Committed success references must be unique")
    if set(sample_ids) != set(episode.success_roles):
        raise ValueError("Committed references must exactly cover episode successes")
    source_locations = {
        (
            item.episode_path,
            item.episode_sha256,
            item.episode_record_index,
        )
        for item in canonical
    }
    if len(source_locations) != 1:
        raise ValueError("Success references disagree on their source episode")
    return tuple(canonical)


def _trajectory_id(
    *,
    episode_id: str,
    event_ledger_sha256: str,
    terminal_reference_sha256: str,
    transition_sequences: tuple[int, ...],
) -> str:
    return canonical_fingerprint(
        {
            "schema_version": 1,
            "episode_id": episode_id,
            "event_ledger_sha256": event_ledger_sha256,
            "terminal_success_reference_sha256": terminal_reference_sha256,
            "transition_sequences": list(transition_sequences),
        }
    )


def _experience_id(
    *,
    trajectory_id: str,
    window_policy_sha256: str,
    environment_seed: int,
    locator: FutureWindowLocator,
) -> str:
    return canonical_fingerprint(
        {
            "schema_version": 1,
            "trajectory_id": trajectory_id,
            "window_policy_sha256": window_policy_sha256,
            "environment_seed": environment_seed,
            "locator": locator.to_record(),
        }
    )


def compile_success_experience_graph(
    *,
    episode_record: Mapping[str, Any],
    event_ledger: EpisodeEventLedger,
    success_references: Iterable[SuccessReference],
    window_policy: FutureWindowPolicy,
) -> SuccessExperienceGraph:
    """Compile all terminal success branches in one committed archive pair."""
    episode = _validate_episode_schema5(episode_record)
    if not isinstance(event_ledger, EpisodeEventLedger):
        raise TypeError("event_ledger must be an EpisodeEventLedger")
    event_ledger.assert_reward_agnostic()
    policy = FutureWindowPolicy.from_record(window_policy.to_record())
    references = _canonical_references(success_references, episode)
    expected_episode_id = (
        f"g{episode.generation:06d}-r{episode.rank:05d}-e{episode.episode:05d}"
    )
    if event_ledger.episode_id != expected_episode_id:
        raise ValueError("Episode archive/Event Ledger provenance mismatch")
    if event_ledger.collection.data_generation != episode.generation:
        raise ValueError("Episode archive/Event Ledger generation mismatch")
    if event_ledger.reset_recipe.seed != episode.reset_seed:
        raise ValueError("Episode archive/Event Ledger reset seed mismatch")

    blocks = _ledger_decisions(event_ledger)
    if len(blocks) != len(episode_record["decisions"]):
        raise ValueError("Episode archive/Event Ledger decision count mismatch")
    for decision_index, block in enumerate(blocks):
        ledger_ids = set(block.transitions)
        episode_ids = {
            sample_id
            for sample_id, source_decision in episode.candidate_decisions.items()
            if source_decision == decision_index
        }
        if ledger_ids != episode_ids:
            raise ValueError("Episode archive/Event Ledger candidate set mismatch")
        for sample_id, transitions in block.transitions.items():
            if len(transitions) != episode.candidates[sample_id]["valid_steps"]:
                raise ValueError("Episode archive/Event Ledger candidate span mismatch")

    by_id = {item.sample_id: item for item in references}
    for index, block in enumerate(blocks[:-1]):
        continued = block.closed.continued_sample_id
        reference = None if continued is None else by_id.get(continued)
        if reference is None or SUCCESS_PATH_ROLE not in reference.success_roles:
            raise ValueError("Success path continuation lacks a committed reference")

    terminal_references = tuple(
        item for item in references if DIRECT_SUCCESS_ROLE in item.success_roles
    )
    if not terminal_references:
        raise ValueError("Successful episode has no committed direct-success branch")
    terminal_decision = len(blocks) - 1
    for reference in terminal_references:
        if reference.decision != terminal_decision:
            raise ValueError(
                "Direct-success references must belong to the final decision"
            )
        candidate = episode.candidates[reference.sample_id]
        if candidate["success"] is not True:
            raise ValueError("Terminal reference does not identify a successful branch")

    ledger_sha256 = event_ledger.fingerprint
    terminal_paths = []
    experiences = []
    reference_hashes = {
        item.sample_id: canonical_fingerprint(item.to_record()) for item in references
    }
    for reference in sorted(terminal_references, key=lambda item: item.sample_id):
        sequences = _terminal_path_sequences(event_ledger, reference.sample_id)
        reference_sha256 = reference_hashes[reference.sample_id]
        trajectory_id = _trajectory_id(
            episode_id=event_ledger.episode_id,
            event_ledger_sha256=ledger_sha256,
            terminal_reference_sha256=reference_sha256,
            transition_sequences=sequences,
        )
        terminal_paths.append(
            TerminalSuccessPath(
                schema_version=1,
                trajectory_id=trajectory_id,
                terminal_sample_id=reference.sample_id,
                terminal_success_reference_sha256=reference_sha256,
                transition_sequences=sequences,
            )
        )
        for offset in policy.offsets(len(sequences)):
            located = sequences[offset : offset + policy.future_horizon]
            locator = FutureWindowLocator(
                schema_version=1,
                episode_id=event_ledger.episode_id,
                event_ledger_sha256=ledger_sha256,
                terminal_sample_id=reference.sample_id,
                transition_sequences=located,
                action_steps=min(policy.action_horizon, len(sequences) - offset),
            )
            experiences.append(
                SuccessExperience(
                    schema_version=1,
                    experience_id=_experience_id(
                        trajectory_id=trajectory_id,
                        window_policy_sha256=policy.fingerprint,
                        environment_seed=event_ledger.reset_recipe.seed,
                        locator=locator,
                    ),
                    trajectory_id=trajectory_id,
                    environment_seed=event_ledger.reset_recipe.seed,
                    locator=locator,
                )
            )

    source_path, source_sha256, source_index = (
        references[0].episode_path,
        references[0].episode_sha256,
        references[0].episode_record_index,
    )
    graph = SuccessExperienceGraph(
        schema_version=1,
        artifact_type="rexpolicy_success_experience_graph",
        episode_id=event_ledger.episode_id,
        task_id=episode.task_id,
        task_contract_id=event_ledger.bindings.task_contract_id,
        task_contract_sha256=event_ledger.bindings.task_contract_sha256,
        simulator_fingerprint=episode.simulator_fingerprint,
        source_episode_path=source_path,
        source_episode_sha256=source_sha256,
        source_episode_record_index=source_index,
        source_success_set_sha256=canonical_fingerprint(
            [item.to_record() for item in references]
        ),
        event_ledger_sha256=ledger_sha256,
        environment_adapter_id=event_ledger.reset_recipe.environment_adapter_id,
        environment_seed=event_ledger.reset_recipe.seed,
        window_policy=policy,
        window_policy_sha256=policy.fingerprint,
        terminal_paths=tuple(terminal_paths),
        experiences=tuple(
            sorted(
                experiences,
                key=lambda item: (
                    item.trajectory_id,
                    item.locator.current_transition_sequence,
                ),
            )
        ),
    )
    graph.validate()
    return graph
