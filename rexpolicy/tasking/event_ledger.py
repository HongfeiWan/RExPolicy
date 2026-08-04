"""Immutable reward-agnostic episode event-ledger contracts."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Union

from .canonical import canonical_fingerprint, canonical_json
from .capabilities import EventSchema, validate_event_values

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_VERSIONED_ID = re.compile(r"^[a-z][a-z0-9_.-]*(?:/[a-z0-9_.-]+)*$")
_EPISODE_ID = re.compile(r"^g[0-9]{6}-r[0-9]{5}-e[0-9]{5}$")
_SAMPLE_ID = re.compile(
    r"^g[0-9]{6}-r[0-9]{5}-e[0-9]{5}-d[0-9]{5}-w[0-9]{5}$"
)
_FORBIDDEN_DERIVED_FIELDS = frozenset(
    (
        "reward",
        "rewards",
        "reward_profile_id",
        "score",
        "advantage",
        "sample_weight",
        "selected_for_training",
        "chosen_for_continuation",
    )
)


def _mapping(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{path} must be an object")
    return value


def _keys(record: dict[str, Any], expected: set[str], path: str) -> None:
    missing = sorted(expected.difference(record))
    unknown = sorted(set(record).difference(expected))
    if missing:
        raise ValueError(f"{path} is missing fields: {', '.join(missing)}")
    if unknown:
        raise ValueError(f"{path} has unknown fields: {', '.join(unknown)}")


def _sha256(value: Any, path: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{path} must be a lowercase SHA-256")
    return value


def _identifier(value: Any, path: str) -> str:
    if not isinstance(value, str) or _VERSIONED_ID.fullmatch(value) is None:
        raise ValueError(f"{path} must be a versioned identifier")
    return value


def _integer(value: Any, path: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{path} must be an integer >= {minimum}")
    return value


SignalFrame = tuple[tuple[str, Any], ...]


def _signals(value: Any, path: str, schema: EventSchema) -> SignalFrame:
    signals = dict(_mapping(value, path))
    validate_event_values(signals, schema=schema, at="current")
    return tuple(sorted(signals.items()))


@dataclass(frozen=True)
class LedgerBindings:
    task_contract_id: str
    task_contract_sha256: str
    dynamics_contract_sha256: str
    observation_contract_sha256: str
    event_schema_id: str
    event_schema_sha256: str

    @classmethod
    def from_record(cls, value: Any) -> LedgerBindings:
        path = "$.bindings"
        record = _mapping(value, path)
        _keys(
            record,
            {
                "task_contract_id",
                "task_contract_sha256",
                "dynamics_contract_sha256",
                "observation_contract_sha256",
                "event_schema_id",
                "event_schema_sha256",
            },
            path,
        )
        return cls(
            task_contract_id=_identifier(
                record["task_contract_id"],
                f"{path}.task_contract_id",
            ),
            task_contract_sha256=_sha256(
                record["task_contract_sha256"],
                f"{path}.task_contract_sha256",
            ),
            dynamics_contract_sha256=_sha256(
                record["dynamics_contract_sha256"],
                f"{path}.dynamics_contract_sha256",
            ),
            observation_contract_sha256=_sha256(
                record["observation_contract_sha256"],
                f"{path}.observation_contract_sha256",
            ),
            event_schema_id=_identifier(
                record["event_schema_id"],
                f"{path}.event_schema_id",
            ),
            event_schema_sha256=_sha256(
                record["event_schema_sha256"],
                f"{path}.event_schema_sha256",
            ),
        )

    def to_record(self) -> dict[str, Any]:
        return {
            "task_contract_id": self.task_contract_id,
            "task_contract_sha256": self.task_contract_sha256,
            "dynamics_contract_sha256": self.dynamics_contract_sha256,
            "observation_contract_sha256": self.observation_contract_sha256,
            "event_schema_id": self.event_schema_id,
            "event_schema_sha256": self.event_schema_sha256,
        }


@dataclass(frozen=True)
class CollectionProvenance:
    data_generation: int
    sampling_policy_generation: int
    rank: int
    episode: int
    instruction_variant_id: str

    @classmethod
    def from_record(cls, value: Any) -> CollectionProvenance:
        path = "$.collection"
        record = _mapping(value, path)
        _keys(
            record,
            {
                "data_generation",
                "sampling_policy_generation",
                "rank",
                "episode",
                "instruction_variant_id",
            },
            path,
        )
        provenance = cls(
            data_generation=_integer(
                record["data_generation"],
                f"{path}.data_generation",
                minimum=1,
            ),
            sampling_policy_generation=_integer(
                record["sampling_policy_generation"],
                f"{path}.sampling_policy_generation",
            ),
            rank=_integer(record["rank"], f"{path}.rank"),
            episode=_integer(record["episode"], f"{path}.episode"),
            instruction_variant_id=_identifier(
                record["instruction_variant_id"],
                f"{path}.instruction_variant_id",
            ),
        )
        if provenance.sampling_policy_generation != provenance.data_generation - 1:
            raise ValueError("Collection policy generation must precede data generation")
        return provenance

    @property
    def episode_id(self) -> str:
        return (
            f"g{self.data_generation:06d}-r{self.rank:05d}"
            f"-e{self.episode:05d}"
        )

    def to_record(self) -> dict[str, Any]:
        return {
            "data_generation": self.data_generation,
            "sampling_policy_generation": self.sampling_policy_generation,
            "rank": self.rank,
            "episode": self.episode,
            "instruction_variant_id": self.instruction_variant_id,
        }


@dataclass(frozen=True)
class ResetRecipe:
    environment_adapter_id: str
    seed: int

    @classmethod
    def from_record(cls, value: Any) -> ResetRecipe:
        path = "$.reset_recipe"
        record = _mapping(value, path)
        _keys(record, {"environment_adapter_id", "seed"}, path)
        return cls(
            environment_adapter_id=_identifier(
                record["environment_adapter_id"],
                f"{path}.environment_adapter_id",
            ),
            seed=_integer(record["seed"], f"{path}.seed"),
        )

    def to_record(self) -> dict[str, Any]:
        return {
            "environment_adapter_id": self.environment_adapter_id,
            "seed": self.seed,
        }


@dataclass(frozen=True)
class DecisionRootEvent:
    sequence: int
    decision_index: int
    root_control_step: int
    dynamics_digest_sha256: str
    signals: SignalFrame
    kind: str = "decision_root"

    def to_record(self) -> dict[str, Any]:
        return {
            "sequence": self.sequence,
            "kind": self.kind,
            "decision_index": self.decision_index,
            "root_control_step": self.root_control_step,
            "dynamics_digest_sha256": self.dynamics_digest_sha256,
            "signals": dict(self.signals),
        }


@dataclass(frozen=True)
class TransitionEvent:
    sequence: int
    sample_id: str
    decision_index: int
    world: int
    branch_step: int
    effective_action_19d: tuple[float, ...]
    post_dynamics_digest_sha256: str
    post_signals: SignalFrame
    terminated: bool
    truncated: bool
    kind: str = "transition"

    def to_record(self) -> dict[str, Any]:
        return {
            "sequence": self.sequence,
            "kind": self.kind,
            "sample_id": self.sample_id,
            "decision_index": self.decision_index,
            "world": self.world,
            "branch_step": self.branch_step,
            "effective_action_19d": list(self.effective_action_19d),
            "post_dynamics_digest_sha256": self.post_dynamics_digest_sha256,
            "post_signals": dict(self.post_signals),
            "terminated": self.terminated,
            "truncated": self.truncated,
        }


@dataclass(frozen=True)
class DecisionClosedEvent:
    sequence: int
    decision_index: int
    continued_sample_id: str | None
    kind: str = "decision_closed"

    def to_record(self) -> dict[str, Any]:
        return {
            "sequence": self.sequence,
            "kind": self.kind,
            "decision_index": self.decision_index,
            "continued_sample_id": self.continued_sample_id,
        }


LedgerEvent = Union[DecisionRootEvent, TransitionEvent, DecisionClosedEvent]


def _event_from_record(
    value: Any,
    *,
    path: str,
    schema: EventSchema,
) -> LedgerEvent:
    record = _mapping(value, path)
    kind = record.get("kind")
    if kind == "decision_root":
        expected = {
            "sequence",
            "kind",
            "decision_index",
            "root_control_step",
            "dynamics_digest_sha256",
            "signals",
        }
        _keys(record, expected, path)
        return DecisionRootEvent(
            sequence=_integer(record["sequence"], f"{path}.sequence"),
            decision_index=_integer(
                record["decision_index"],
                f"{path}.decision_index",
            ),
            root_control_step=_integer(
                record["root_control_step"],
                f"{path}.root_control_step",
            ),
            dynamics_digest_sha256=_sha256(
                record["dynamics_digest_sha256"],
                f"{path}.dynamics_digest_sha256",
            ),
            signals=_signals(record["signals"], f"{path}.signals", schema),
        )
    if kind == "transition":
        expected = {
            "sequence",
            "kind",
            "sample_id",
            "decision_index",
            "world",
            "branch_step",
            "effective_action_19d",
            "post_dynamics_digest_sha256",
            "post_signals",
            "terminated",
            "truncated",
        }
        _keys(record, expected, path)
        sample_id = record["sample_id"]
        if not isinstance(sample_id, str) or _SAMPLE_ID.fullmatch(sample_id) is None:
            raise ValueError(f"{path}.sample_id is invalid")
        raw_action = record["effective_action_19d"]
        if not isinstance(raw_action, list) or len(raw_action) != 19:
            raise ValueError(f"{path}.effective_action_19d must have length 19")
        action = []
        for index, value in enumerate(raw_action):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{path}.effective_action_19d[{index}] is not numeric")
            number = float(value)
            if not math.isfinite(number):
                raise ValueError(f"{path}.effective_action_19d[{index}] is non-finite")
            action.append(number)
        for flag in ("terminated", "truncated"):
            if type(record[flag]) is not bool:
                raise ValueError(f"{path}.{flag} must be Boolean")
        return TransitionEvent(
            sequence=_integer(record["sequence"], f"{path}.sequence"),
            sample_id=sample_id,
            decision_index=_integer(
                record["decision_index"],
                f"{path}.decision_index",
            ),
            world=_integer(record["world"], f"{path}.world"),
            branch_step=_integer(
                record["branch_step"],
                f"{path}.branch_step",
            ),
            effective_action_19d=tuple(action),
            post_dynamics_digest_sha256=_sha256(
                record["post_dynamics_digest_sha256"],
                f"{path}.post_dynamics_digest_sha256",
            ),
            post_signals=_signals(
                record["post_signals"],
                f"{path}.post_signals",
                schema,
            ),
            terminated=record["terminated"],
            truncated=record["truncated"],
        )
    if kind == "decision_closed":
        expected = {
            "sequence",
            "kind",
            "decision_index",
            "continued_sample_id",
        }
        _keys(record, expected, path)
        continued = record["continued_sample_id"]
        if continued is not None and (
            not isinstance(continued, str)
            or _SAMPLE_ID.fullmatch(continued) is None
        ):
            raise ValueError(f"{path}.continued_sample_id is invalid")
        return DecisionClosedEvent(
            sequence=_integer(record["sequence"], f"{path}.sequence"),
            decision_index=_integer(
                record["decision_index"],
                f"{path}.decision_index",
            ),
            continued_sample_id=continued,
        )
    raise ValueError(f"{path}.kind is unsupported")


@dataclass(frozen=True)
class EpisodeEventLedger:
    schema_version: int
    episode_id: str
    bindings: LedgerBindings
    collection: CollectionProvenance
    reset_recipe: ResetRecipe
    events: tuple[LedgerEvent, ...]

    @classmethod
    def from_record(
        cls,
        value: Any,
        *,
        event_schema: EventSchema,
    ) -> EpisodeEventLedger:
        path = "$"
        record = _mapping(value, path)
        _keys(
            record,
            {
                "schema_version",
                "episode_id",
                "bindings",
                "collection",
                "reset_recipe",
                "events",
            },
            path,
        )
        if record["schema_version"] != 1:
            raise ValueError("Event Ledger schema_version must be 1")
        bindings = LedgerBindings.from_record(record["bindings"])
        collection = CollectionProvenance.from_record(record["collection"])
        raw_events = record["events"]
        if not isinstance(raw_events, list) or not raw_events:
            raise ValueError("$.events must be a non-empty array")
        ledger = cls(
            schema_version=1,
            episode_id=record["episode_id"],
            bindings=bindings,
            collection=collection,
            reset_recipe=ResetRecipe.from_record(record["reset_recipe"]),
            events=tuple(
                _event_from_record(
                    item,
                    path=f"$.events[{index}]",
                    schema=event_schema,
                )
                for index, item in enumerate(raw_events)
            ),
        )
        ledger.validate(event_schema=event_schema)
        return ledger

    def validate(self, *, event_schema: EventSchema) -> None:
        """Enforce event ordering, candidate spans, and actual-path continuity."""
        if self.schema_version != 1:
            raise ValueError("Event Ledger schema_version must be 1")
        if not isinstance(self.episode_id, str) or _EPISODE_ID.fullmatch(
            self.episode_id
        ) is None:
            raise ValueError("Event Ledger episode_id is invalid")
        if self.episode_id != self.collection.episode_id:
            raise ValueError("Event Ledger episode_id/provenance mismatch")
        if (
            self.bindings.event_schema_id != event_schema.event_schema_id
            or self.bindings.event_schema_sha256 != event_schema.fingerprint
        ):
            raise ValueError("Event Ledger EventSchema fingerprint mismatch")
        if self.reset_recipe.environment_adapter_id == "":
            raise ValueError("Event Ledger reset adapter cannot be empty")
        if not self.events:
            raise ValueError("Event Ledger requires events")
        if LedgerBindings.from_record(self.bindings.to_record()) != self.bindings:
            raise ValueError("Event Ledger bindings are not canonical")
        if CollectionProvenance.from_record(
            self.collection.to_record()
        ) != self.collection:
            raise ValueError("Event Ledger collection provenance is not canonical")
        if ResetRecipe.from_record(self.reset_recipe.to_record()) != self.reset_recipe:
            raise ValueError("Event Ledger reset recipe is not canonical")
        for index, event in enumerate(self.events):
            restored = _event_from_record(
                event.to_record(),
                path=f"$.events[{index}]",
                schema=event_schema,
            )
            if restored != event:
                raise ValueError("Event Ledger event is not canonical")
        if [event.sequence for event in self.events] != list(range(len(self.events))):
            raise ValueError("Event Ledger event sequence must be contiguous")

        expected_decision = 0
        expected_root_step = 0
        expected_root_digest: str | None = None
        expected_root_signals: SignalFrame | None = None
        cursor = 0
        terminal_decision_seen = False
        while cursor < len(self.events):
            if terminal_decision_seen:
                raise ValueError("Event Ledger continues after a closed terminal path")
            root = self.events[cursor]
            if not isinstance(root, DecisionRootEvent):
                raise ValueError("Each decision must start with a decision_root")
            if root.decision_index != expected_decision:
                raise ValueError("Event Ledger decision indexes must be contiguous")
            if root.root_control_step != expected_root_step:
                raise ValueError("Event Ledger root control step is inconsistent")
            if (
                expected_root_digest is not None
                and root.dynamics_digest_sha256 != expected_root_digest
            ):
                raise ValueError("Decision root digest does not match continuation")
            if expected_root_signals is not None and root.signals != expected_root_signals:
                raise ValueError("Decision root signals do not match continuation")
            validate_event_values(
                dict(root.signals),
                schema=event_schema,
                at="current",
            )
            cursor += 1
            transitions: dict[str, list[TransitionEvent]] = {}
            world_by_sample: dict[str, int] = {}
            sample_by_world: dict[int, str] = {}
            closed: DecisionClosedEvent | None = None
            while cursor < len(self.events):
                event = self.events[cursor]
                if isinstance(event, DecisionRootEvent):
                    break
                if event.decision_index != expected_decision:
                    raise ValueError("Event Ledger event has the wrong decision index")
                if isinstance(event, DecisionClosedEvent):
                    closed = event
                    cursor += 1
                    break
                if not isinstance(event, TransitionEvent):
                    raise AssertionError(type(event))
                expected_sample_id = (
                    f"{self.episode_id}-d{event.decision_index:05d}"
                    f"-w{event.world:05d}"
                )
                if event.sample_id != expected_sample_id:
                    raise ValueError("Transition sample_id/provenance mismatch")
                sample_events = transitions.setdefault(event.sample_id, [])
                if sample_events and (
                    sample_events[-1].terminated or sample_events[-1].truncated
                ):
                    raise ValueError("Candidate has transitions after termination")
                if event.branch_step != len(sample_events):
                    raise ValueError("Candidate branch steps must be contiguous")
                previous_world = world_by_sample.setdefault(
                    event.sample_id,
                    event.world,
                )
                if previous_world != event.world:
                    raise ValueError("Candidate sample_id changes world")
                previous_sample = sample_by_world.setdefault(
                    event.world,
                    event.sample_id,
                )
                if previous_sample != event.sample_id:
                    raise ValueError("Decision world maps to multiple sample IDs")
                sample_events.append(event)
                validate_event_values(
                    dict(event.post_signals),
                    schema=event_schema,
                    at="current",
                )
                cursor += 1
            if closed is None:
                raise ValueError("Each decision must end with decision_closed")
            if not transitions:
                raise ValueError("Each decision requires candidate transitions")
            if closed.continued_sample_id is None:
                terminal_decision_seen = True
            else:
                if closed.continued_sample_id not in transitions:
                    raise ValueError("Decision continuation references no candidate")
                selected_events = transitions[closed.continued_sample_id]
                expected_root_step += len(selected_events)
                expected_root_digest = selected_events[-1].post_dynamics_digest_sha256
                expected_root_signals = selected_events[-1].post_signals
                terminal_decision_seen = bool(
                    selected_events[-1].terminated
                    or selected_events[-1].truncated
                )
            expected_decision += 1
        if not isinstance(self.events[-1], DecisionClosedEvent):
            raise ValueError("Event Ledger must end with decision_closed")
        self.assert_reward_agnostic()

    def to_record(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "episode_id": self.episode_id,
            "bindings": self.bindings.to_record(),
            "collection": self.collection.to_record(),
            "reset_recipe": self.reset_recipe.to_record(),
            "events": [event.to_record() for event in self.events],
        }

    def to_json(self) -> str:
        return canonical_json(self.to_record())

    @property
    def fingerprint(self) -> str:
        return canonical_fingerprint(self.to_record())

    def assert_reward_agnostic(self) -> None:
        """Defend against derived training fields entering the fact ledger."""

        def visit(value: Any, path: str) -> None:
            if isinstance(value, dict):
                forbidden = _FORBIDDEN_DERIVED_FIELDS.intersection(value)
                if forbidden:
                    raise ValueError(
                        f"{path} contains derived fields: "
                        + ", ".join(sorted(forbidden))
                    )
                for key, child in value.items():
                    visit(child, f"{path}.{key}")
            elif isinstance(value, list):
                for index, child in enumerate(value):
                    visit(child, f"{path}[{index}]")

        visit(self.to_record(), "$")
