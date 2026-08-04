"""State-machine builder for deterministic reward-agnostic Event Ledgers."""

from __future__ import annotations

from typing import Any, Mapping

from rexpolicy.tasking.capabilities import EventSchema, validate_event_values
from rexpolicy.tasking.dynamics_contract import (
    DYNAMICS_CONTRACT_V1_SHA256,
)
from rexpolicy.tasking.event_ledger import (
    CollectionProvenance,
    EpisodeEventLedger,
    LedgerBindings,
    ResetRecipe,
)

from .experience import format_sample_id


def production_ledger_bindings(
    *,
    task_contract_id: str,
    task_contract_sha256: str,
    observation_contract_sha256: str,
    event_schema: EventSchema,
) -> LedgerBindings:
    """Bind separate task/dynamics/observation/EventSchema identities."""
    return LedgerBindings.from_record(
        {
            "task_contract_id": task_contract_id,
            "task_contract_sha256": task_contract_sha256,
            "dynamics_contract_sha256": DYNAMICS_CONTRACT_V1_SHA256,
            "observation_contract_sha256": observation_contract_sha256,
            "event_schema_id": event_schema.event_schema_id,
            "event_schema_sha256": event_schema.fingerprint,
        }
    )


class EpisodeEventLedgerBuilder:
    """Buffer parallel transitions and emit one canonical immutable ledger."""

    def __init__(
        self,
        *,
        bindings: LedgerBindings,
        collection: CollectionProvenance,
        reset_recipe: ResetRecipe,
        event_schema: EventSchema,
    ) -> None:
        self._bindings = bindings
        self._collection = collection
        self._reset_recipe = reset_recipe
        self._event_schema = event_schema
        self._events: list[dict[str, Any]] = []
        self._open_root: dict[str, Any] | None = None
        self._transitions: dict[int, list[dict[str, Any]]] = {}
        self._next_decision = 0
        self._closed_terminal = False
        self._finished = False

    @property
    def episode_id(self) -> str:
        return self._collection.episode_id

    def _require_writable(self) -> None:
        if self._finished:
            raise ValueError("Event Ledger builder is already finished")
        if self._closed_terminal:
            raise ValueError("Event Ledger builder is closed on a terminal path")

    def open_decision(
        self,
        *,
        decision_index: int,
        root_control_step: int,
        dynamics_digest_sha256: str,
        current_metrics: Mapping[str, Any],
    ) -> None:
        self._require_writable()
        if self._open_root is not None:
            raise ValueError("Event Ledger decision is already open")
        if decision_index != self._next_decision:
            raise ValueError("Event Ledger decision index is not contiguous")
        metrics = dict(current_metrics)
        validate_event_values(
            metrics,
            schema=self._event_schema,
            at="current",
        )
        self._open_root = {
            "kind": "decision_root",
            "decision_index": decision_index,
            "root_control_step": root_control_step,
            "dynamics_digest_sha256": dynamics_digest_sha256,
            "signals": metrics,
        }
        self._transitions = {}

    def append_transition(
        self,
        *,
        decision_index: int,
        world: int,
        branch_step: int,
        effective_action_19d: Any,
        post_dynamics_digest_sha256: str,
        current_metrics: Mapping[str, Any],
        terminated: bool,
        truncated: bool,
    ) -> str:
        self._require_writable()
        if self._open_root is None:
            raise ValueError("Event Ledger transition requires an open decision")
        if decision_index != self._next_decision:
            raise ValueError("Event Ledger transition decision index mismatch")
        if type(world) is not int or world < 0:
            raise ValueError("Event Ledger transition world is invalid")
        world_events = self._transitions.setdefault(world, [])
        if branch_step != len(world_events):
            raise ValueError("Event Ledger branch steps must be contiguous")
        if world_events and (
            world_events[-1]["terminated"] or world_events[-1]["truncated"]
        ):
            raise ValueError("Event Ledger candidate continues after termination")
        if type(terminated) is not bool or type(truncated) is not bool:
            raise ValueError("Event Ledger transition flags must be Boolean")
        metrics = dict(current_metrics)
        validate_event_values(
            metrics,
            schema=self._event_schema,
            at="current",
        )
        if hasattr(effective_action_19d, "tolist"):
            effective_action_19d = effective_action_19d.tolist()
        action = list(effective_action_19d)
        sample_id = format_sample_id(
            self._collection.data_generation,
            self._collection.rank,
            self._collection.episode,
            decision_index,
            world,
        )
        world_events.append(
            {
                "kind": "transition",
                "sample_id": sample_id,
                "decision_index": decision_index,
                "world": world,
                "branch_step": branch_step,
                "effective_action_19d": action,
                "post_dynamics_digest_sha256": (
                    post_dynamics_digest_sha256
                ),
                "post_signals": metrics,
                "terminated": terminated,
                "truncated": truncated,
            }
        )
        return sample_id

    def close_decision(
        self,
        *,
        decision_index: int,
        root_continuation_sample_id: str | None,
    ) -> None:
        self._require_writable()
        if self._open_root is None:
            raise ValueError("Event Ledger close requires an open decision")
        if decision_index != self._next_decision:
            raise ValueError("Event Ledger close decision index mismatch")
        if not self._transitions:
            raise ValueError("Event Ledger decision has no transitions")
        candidate_ids = {
            events[0]["sample_id"] for events in self._transitions.values()
        }
        if (
            root_continuation_sample_id is not None
            and root_continuation_sample_id not in candidate_ids
        ):
            raise ValueError("Event Ledger continuation is not a candidate")
        self._append(self._open_root)
        for world in sorted(self._transitions):
            for transition in self._transitions[world]:
                self._append(transition)
        self._append(
            {
                "kind": "decision_closed",
                "decision_index": decision_index,
                "continued_sample_id": root_continuation_sample_id,
            }
        )
        if root_continuation_sample_id is None:
            self._closed_terminal = True
        else:
            selected = next(
                events[-1]
                for events in self._transitions.values()
                if events[0]["sample_id"] == root_continuation_sample_id
            )
            self._closed_terminal = bool(
                selected["terminated"] or selected["truncated"]
            )
        self._open_root = None
        self._transitions = {}
        self._next_decision += 1

    def _append(self, record: dict[str, Any]) -> None:
        self._events.append({"sequence": len(self._events), **record})

    def finish(self) -> EpisodeEventLedger:
        if self._finished:
            raise ValueError("Event Ledger builder is already finished")
        if self._open_root is not None:
            raise ValueError("Event Ledger builder has an open decision")
        if not self._events:
            raise ValueError("Event Ledger builder has no events")
        ledger = EpisodeEventLedger.from_record(
            {
                "schema_version": 1,
                "episode_id": self.episode_id,
                "bindings": self._bindings.to_record(),
                "collection": self._collection.to_record(),
                "reset_recipe": self._reset_recipe.to_record(),
                "events": self._events,
            },
            event_schema=self._event_schema,
        )
        self._finished = True
        return ledger
