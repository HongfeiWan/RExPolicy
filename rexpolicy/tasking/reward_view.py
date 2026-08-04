"""Versioned reward sidecars materialized from reward-agnostic Event Ledgers."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from .canonical import canonical_fingerprint, canonical_json, strict_json_loads
from .capabilities import CapabilityCatalog, EventSchema
from .contract import (
    DEFAULT_TASK_COMPILER_POLICY,
    CompiledTaskContract,
    TaskCompilerPolicy,
    validate_compiled_task_contract,
)
from .event_ledger import (
    DecisionClosedEvent,
    DecisionRootEvent,
    EpisodeEventLedger,
    TransitionEvent,
)
from .expression import MetricFrame
from .runtime import (
    ACTIVE,
    FAILURE,
    SUCCESS,
    TRUNCATED,
    RewardTermResult,
    TaskRuntimeState,
    evaluate_task_step,
    initial_runtime_state,
)


@dataclass(frozen=True)
class RewardViewPolicy:
    policy_id: str
    algorithm: str
    require_terminal_flag_parity: bool
    max_transitions: int

    def validate(self) -> None:
        if self.policy_id != "rexpolicy/reward_view/v1":
            raise ValueError("Unsupported RewardView policy")
        if self.algorithm != "branch_replay_failure_precedence/v1":
            raise ValueError("Unsupported RewardView algorithm")
        if self.require_terminal_flag_parity is not True:
            raise ValueError("RewardView terminal parity cannot be disabled")
        if type(self.max_transitions) is not int or self.max_transitions < 1:
            raise ValueError("RewardView transition limit must be positive")

    def to_record(self) -> dict[str, Any]:
        return {
            "policy_id": self.policy_id,
            "algorithm": self.algorithm,
            "require_terminal_flag_parity": self.require_terminal_flag_parity,
            "max_transitions": self.max_transitions,
        }

    @property
    def fingerprint(self) -> str:
        return canonical_fingerprint(self.to_record())


DEFAULT_REWARD_VIEW_POLICY = RewardViewPolicy(
    policy_id="rexpolicy/reward_view/v1",
    algorithm="branch_replay_failure_precedence/v1",
    require_terminal_flag_parity=True,
    max_transitions=1_000_000,
)


@dataclass(frozen=True)
class OracleState:
    control_step: int
    goal_streak: int
    status: str

    def to_record(self) -> dict[str, Any]:
        return {
            "control_step": self.control_step,
            "goal_streak": self.goal_streak,
            "status": self.status,
        }

    @property
    def fingerprint(self) -> str:
        return canonical_fingerprint(self.to_record())


@dataclass(frozen=True)
class RewardTransition:
    source_sequence: int
    sample_id: str
    decision_index: int
    world: int
    branch_step: int
    status: str
    goal_predicate: bool
    failure_codes: tuple[str, ...]
    safety_codes: tuple[str, ...]
    oracle_state_after: OracleState
    oracle_state_after_sha256: str
    reward: float
    shaping_unclipped: float | None
    shaping_clipped: float | None
    reward_terms: tuple[RewardTermResult, ...]

    def to_record(self) -> dict[str, Any]:
        return {
            "source_sequence": self.source_sequence,
            "sample_id": self.sample_id,
            "decision_index": self.decision_index,
            "world": self.world,
            "branch_step": self.branch_step,
            "status": self.status,
            "goal_predicate": self.goal_predicate,
            "failure_codes": list(self.failure_codes),
            "safety_codes": list(self.safety_codes),
            "oracle_state_after": self.oracle_state_after.to_record(),
            "oracle_state_after_sha256": self.oracle_state_after_sha256,
            "reward": self.reward,
            "shaping_unclipped": self.shaping_unclipped,
            "shaping_clipped": self.shaping_clipped,
            "reward_terms": [term.to_record() for term in self.reward_terms],
        }


@dataclass(frozen=True)
class RewardCandidateSummary:
    sample_id: str
    decision_index: int
    world: int
    transition_count: int
    final_status: str
    final_oracle_state_sha256: str
    total_reward: float
    continued_from_root: bool

    def to_record(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "decision_index": self.decision_index,
            "world": self.world,
            "transition_count": self.transition_count,
            "final_status": self.final_status,
            "final_oracle_state_sha256": self.final_oracle_state_sha256,
            "total_reward": self.total_reward,
            "continued_from_root": self.continued_from_root,
        }


@dataclass(frozen=True)
class RewardView:
    artifact_type: str
    schema_version: int
    episode_id: str
    event_ledger_sha256: str
    source_bindings_sha256: str
    source_task_contract_id: str
    source_task_contract_sha256: str
    task_oracle_sha256: str
    scoring_task_contract_id: str
    scoring_task_contract_sha256: str
    reward_profile_id: str
    reward_profile_sha256: str
    reward_view_policy_id: str
    reward_view_policy_sha256: str
    transitions: tuple[RewardTransition, ...]
    candidates: tuple[RewardCandidateSummary, ...]

    def to_record(self) -> dict[str, Any]:
        return {
            "artifact_type": self.artifact_type,
            "schema_version": self.schema_version,
            "episode_id": self.episode_id,
            "event_ledger_sha256": self.event_ledger_sha256,
            "source_bindings_sha256": self.source_bindings_sha256,
            "source_task_contract_id": self.source_task_contract_id,
            "source_task_contract_sha256": self.source_task_contract_sha256,
            "task_oracle_sha256": self.task_oracle_sha256,
            "scoring_task_contract_id": self.scoring_task_contract_id,
            "scoring_task_contract_sha256": self.scoring_task_contract_sha256,
            "reward_profile_id": self.reward_profile_id,
            "reward_profile_sha256": self.reward_profile_sha256,
            "reward_view_policy_id": self.reward_view_policy_id,
            "reward_view_policy_sha256": self.reward_view_policy_sha256,
            "transitions": [item.to_record() for item in self.transitions],
            "candidates": [item.to_record() for item in self.candidates],
        }

    def to_json(self) -> str:
        return canonical_json(self.to_record())

    @property
    def fingerprint(self) -> str:
        return canonical_fingerprint(self.to_record())

    @classmethod
    def from_record(
        cls,
        value: Any,
        *,
        ledger: EpisodeEventLedger,
        scoring_contract: CompiledTaskContract,
        reward_profile_id: str,
        catalog: CapabilityCatalog,
        reward_view_policy: RewardViewPolicy = DEFAULT_REWARD_VIEW_POLICY,
        task_policy: TaskCompilerPolicy = DEFAULT_TASK_COMPILER_POLICY,
    ) -> RewardView:
        """Accept only an exact replay of the bound fact ledger."""
        if not isinstance(value, dict):
            raise ValueError("RewardView must be an object")
        expected = materialize_reward_view(
            ledger,
            scoring_contract=scoring_contract,
            reward_profile_id=reward_profile_id,
            catalog=catalog,
            reward_view_policy=reward_view_policy,
            task_policy=task_policy,
        )
        try:
            supplied = canonical_json(value)
        except (TypeError, ValueError) as error:
            raise ValueError("RewardView is not canonical JSON") from error
        if supplied != expected.to_json():
            raise ValueError("RewardView does not match deterministic replay")
        return expected

    @classmethod
    def from_json(
        cls,
        text: str,
        **bindings: Any,
    ) -> RewardView:
        return cls.from_record(strict_json_loads(text), **bindings)


def _schema(
    contract: CompiledTaskContract,
    catalog: CapabilityCatalog,
) -> EventSchema:
    try:
        return next(
            item
            for item in catalog.event_schemas
            if item.event_schema_id == contract.event_schema_id
        )
    except StopIteration as error:
        raise ValueError("RewardView EventSchema is unavailable") from error


def _frame(
    schema: EventSchema,
    *,
    previous: dict[str, Any],
    current: dict[str, Any],
) -> MetricFrame:
    return MetricFrame(
        event_schema_id=schema.event_schema_id,
        event_schema_fingerprint=schema.fingerprint,
        previous={
            metric.name: previous[metric.name]
            for metric in schema.metrics
            if "previous" in metric.available_at
        },
        current={
            metric.name: current[metric.name]
            for metric in schema.metrics
            if "current" in metric.available_at
        },
    )


def _oracle_state(state: TaskRuntimeState) -> OracleState:
    return OracleState(
        control_step=state.control_step,
        goal_streak=state.goal_streak,
        status=state.status,
    )


def _require_terminal_parity(
    transition: TransitionEvent,
    *,
    status: str,
) -> None:
    expected_terminated = status in {SUCCESS, FAILURE}
    expected_truncated = status == TRUNCATED
    if (
        transition.terminated != expected_terminated
        or transition.truncated != expected_truncated
    ):
        raise ValueError(
            "RewardView task oracle disagrees with source terminal flags"
        )


def materialize_reward_view(
    ledger: EpisodeEventLedger,
    *,
    scoring_contract: CompiledTaskContract,
    reward_profile_id: str,
    catalog: CapabilityCatalog,
    reward_view_policy: RewardViewPolicy = DEFAULT_REWARD_VIEW_POLICY,
    task_policy: TaskCompilerPolicy = DEFAULT_TASK_COMPILER_POLICY,
) -> RewardView:
    """Replay every branch with one versioned reward profile."""
    reward_view_policy.validate()
    canonical_catalog = CapabilityCatalog.from_record(catalog.to_record())
    validate_compiled_task_contract(
        scoring_contract,
        catalog=canonical_catalog,
        policy=task_policy,
    )
    schema = _schema(scoring_contract, canonical_catalog)
    ledger.validate(event_schema=schema)
    if ledger.bindings.task_contract_id != scoring_contract.task_contract_id:
        raise ValueError("RewardView task contract identity mismatch")
    if ledger.bindings.task_oracle_sha256 != scoring_contract.oracle_fingerprint:
        raise ValueError("RewardView task oracle fingerprint mismatch")
    if (
        ledger.bindings.event_schema_id != scoring_contract.event_schema_id
        or ledger.bindings.event_schema_sha256
        != scoring_contract.event_schema_fingerprint
    ):
        raise ValueError("RewardView EventSchema binding mismatch")
    profiles = {
        profile.reward_profile_id: profile
        for profile in scoring_contract.reward_profiles
    }
    try:
        profile = profiles[reward_profile_id]
    except KeyError as error:
        raise ValueError("RewardView reward profile is unavailable") from error
    transition_count = sum(
        isinstance(event, TransitionEvent) for event in ledger.events
    )
    if transition_count > reward_view_policy.max_transitions:
        raise ValueError("RewardView source exceeds the transition limit")

    root_state = initial_runtime_state(
        scoring_contract,
        reward_profile_id=reward_profile_id,
        catalog=canonical_catalog,
        policy=task_policy,
    )
    results: list[RewardTransition] = []
    summaries: list[RewardCandidateSummary] = []
    cursor = 0
    while cursor < len(ledger.events):
        root = ledger.events[cursor]
        if not isinstance(root, DecisionRootEvent):
            raise ValueError("RewardView decision root is missing")
        if root.root_control_step != root_state.control_step:
            raise ValueError("RewardView runtime/root control-step mismatch")
        root_signals = dict(root.signals)
        cursor += 1
        candidates: dict[str, list[TransitionEvent]] = {}
        closed: DecisionClosedEvent | None = None
        while cursor < len(ledger.events):
            event = ledger.events[cursor]
            cursor += 1
            if isinstance(event, DecisionClosedEvent):
                closed = event
                break
            if not isinstance(event, TransitionEvent):
                raise ValueError("RewardView decision contains an invalid event")
            candidates.setdefault(event.sample_id, []).append(event)
        if closed is None:
            raise ValueError("RewardView decision is not closed")

        final_states: dict[str, TaskRuntimeState] = {}
        for sample_id in sorted(candidates):
            state = root_state
            previous = root_signals
            candidate_rewards = []
            candidate_results = []
            transitions = candidates[sample_id]
            for transition in transitions:
                current = dict(transition.post_signals)
                step = evaluate_task_step(
                    scoring_contract,
                    state=state,
                    frame=_frame(
                        schema,
                        previous=previous,
                        current=current,
                    ),
                    catalog=canonical_catalog,
                    policy=task_policy,
                )
                if reward_view_policy.require_terminal_flag_parity:
                    _require_terminal_parity(transition, status=step.status)
                oracle_after = _oracle_state(step.next_state)
                candidate_results.append(
                    RewardTransition(
                        source_sequence=transition.sequence,
                        sample_id=transition.sample_id,
                        decision_index=transition.decision_index,
                        world=transition.world,
                        branch_step=transition.branch_step,
                        status=step.status,
                        goal_predicate=step.goal_predicate,
                        failure_codes=step.failure_codes,
                        safety_codes=step.safety_codes,
                        oracle_state_after=oracle_after,
                        oracle_state_after_sha256=oracle_after.fingerprint,
                        reward=step.reward,
                        shaping_unclipped=step.shaping_unclipped,
                        shaping_clipped=step.shaping_clipped,
                        reward_terms=step.reward_terms,
                    )
                )
                candidate_rewards.append(step.reward)
                state = step.next_state
                previous = current
            final_states[sample_id] = state
            results.extend(candidate_results)
            final_oracle = _oracle_state(state)
            total_reward = math.fsum(candidate_rewards)
            if not math.isfinite(total_reward):
                raise ValueError("RewardView candidate return is non-finite")
            summaries.append(
                RewardCandidateSummary(
                    sample_id=sample_id,
                    decision_index=transitions[0].decision_index,
                    world=transitions[0].world,
                    transition_count=len(transitions),
                    final_status=state.status,
                    final_oracle_state_sha256=final_oracle.fingerprint,
                    total_reward=total_reward,
                    continued_from_root=(
                        sample_id == closed.continued_sample_id
                    ),
                )
            )
        if closed.continued_sample_id is not None:
            root_state = final_states[closed.continued_sample_id]

    results.sort(key=lambda item: item.source_sequence)
    summaries.sort(
        key=lambda item: (item.decision_index, item.world, item.sample_id)
    )
    if len(results) != transition_count:
        raise ValueError("RewardView does not cover every source transition")
    view = RewardView(
        artifact_type="rexpolicy_reward_view",
        schema_version=1,
        episode_id=ledger.episode_id,
        event_ledger_sha256=ledger.fingerprint,
        source_bindings_sha256=canonical_fingerprint(
            ledger.bindings.to_record()
        ),
        source_task_contract_id=ledger.bindings.task_contract_id,
        source_task_contract_sha256=ledger.bindings.task_contract_sha256,
        task_oracle_sha256=scoring_contract.oracle_fingerprint,
        scoring_task_contract_id=scoring_contract.task_contract_id,
        scoring_task_contract_sha256=scoring_contract.fingerprint,
        reward_profile_id=profile.reward_profile_id,
        reward_profile_sha256=profile.fingerprint,
        reward_view_policy_id=reward_view_policy.policy_id,
        reward_view_policy_sha256=reward_view_policy.fingerprint,
        transitions=tuple(results),
        candidates=tuple(summaries),
    )
    canonical_fingerprint(view.to_record())
    return view
