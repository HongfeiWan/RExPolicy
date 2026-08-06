"""Transition-aligned future windows over successful Stage 0 rollouts."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from rexpolicy.stage0.data.trajectory import (
    Stage0Trajectory,
    trajectory_corpus_sha256,
    trajectory_sha256,
)


WINDOW_SCHEMA_VERSION = 1
WINDOW_SCHEMA_ID = "rexpolicy/stage0-success-window/v1"


def _positive_integer(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True)
class Stage0WindowPolicy:
    """Fingerprintable extraction policy shared by every dataset rank."""

    action_horizon: int
    future_horizon: int
    stride: int = 1

    def __post_init__(self) -> None:
        _positive_integer(self.action_horizon, "action_horizon")
        _positive_integer(self.future_horizon, "future_horizon")
        _positive_integer(self.stride, "stride")
        if self.action_horizon > self.future_horizon:
            raise ValueError("action_horizon must be <= future_horizon")

    def to_record(self) -> dict[str, Any]:
        return {
            "action_horizon": self.action_horizon,
            "future_horizon": self.future_horizon,
            "padding": "zero-right/v1",
            "schema_id": WINDOW_SCHEMA_ID,
            "schema_version": WINDOW_SCHEMA_VERSION,
            "stride": self.stride,
            "transition_alignment": "a[t+i]-to-s[t+i+1]/v1",
        }

    @property
    def sha256(self) -> str:
        encoded = json.dumps(
            self.to_record(),
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class Stage0SuccessWindow:
    """One padded, transition-aligned training example."""

    trajectory_id: str
    reset_group_id: str
    start: int
    current_state: Any
    action_chunk: Any
    action_mask: Any
    future_actions: Any
    future_states: Any
    future_mask: Any
    trajectory_sha256: str
    corpus_sha256: str
    window_policy_sha256: str

    def __post_init__(self) -> None:
        import torch

        for name in ("trajectory_id", "reset_group_id"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(f"{name} must be non-empty")
        if (
            isinstance(self.start, bool)
            or not isinstance(self.start, int)
            or self.start < 0
        ):
            raise ValueError("start must be a non-negative integer")
        for name in (
            "trajectory_sha256",
            "corpus_sha256",
            "window_policy_sha256",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or len(value) != 64:
                raise ValueError(f"{name} must be a SHA-256 digest")
        float_tensors = {
            "current_state": self.current_state,
            "action_chunk": self.action_chunk,
            "future_actions": self.future_actions,
            "future_states": self.future_states,
        }
        for name, tensor in float_tensors.items():
            if (
                not isinstance(tensor, torch.Tensor)
                or tensor.dtype is not torch.float32
            ):
                raise TypeError(f"{name} must be a float32 torch.Tensor")
            if tensor.device.type != "cpu" or not tensor.is_contiguous():
                raise ValueError(f"{name} must be contiguous on CPU")
            if not bool(torch.isfinite(tensor).all().item()):
                raise ValueError(f"{name} contains non-finite values")
        for name in ("action_mask", "future_mask"):
            mask = getattr(self, name)
            if not isinstance(mask, torch.Tensor) or mask.dtype is not torch.bool:
                raise TypeError(f"{name} must be a bool torch.Tensor")
            if mask.ndim != 1 or mask.device.type != "cpu":
                raise ValueError(f"{name} must be a one-dimensional CPU mask")
        if self.current_state.ndim != 1:
            raise ValueError("current_state must have shape [state_dim]")
        if self.action_chunk.ndim != 2 or self.future_actions.ndim != 2:
            raise ValueError("action tensors must be rank two")
        if self.future_states.ndim != 2:
            raise ValueError("future_states must be rank two")
        if self.action_chunk.shape[0] != self.action_mask.shape[0]:
            raise ValueError("action_chunk and action_mask horizons differ")
        if self.future_actions.shape[0] != self.future_mask.shape[0]:
            raise ValueError("future_actions and future_mask horizons differ")
        if self.future_states.shape[0] != self.future_mask.shape[0]:
            raise ValueError("future_states and future_mask horizons differ")
        if self.action_chunk.shape[1] != self.future_actions.shape[1]:
            raise ValueError("action dimensions differ")
        if self.current_state.shape[0] != self.future_states.shape[1]:
            raise ValueError("state dimensions differ")
        if self.action_mask.shape[0] > self.future_mask.shape[0]:
            raise ValueError("action horizon must not exceed future horizon")
        if not torch.equal(
            self.action_mask,
            self.future_mask[: self.action_mask.shape[0]],
        ):
            raise ValueError("action_mask must share the future mask prefix")
        if not torch.equal(
            self.action_chunk,
            self.future_actions[: self.action_chunk.shape[0]],
        ):
            raise ValueError("action_chunk must share the future action prefix")
        valid = int(self.future_mask.sum().item())
        if valid < 1:
            raise ValueError("a success window must contain a valid transition")
        expected_mask = torch.arange(self.future_mask.shape[0]) < valid
        if not torch.equal(self.future_mask, expected_mask):
            raise ValueError("future_mask must be a contiguous valid prefix")
        if bool(self.future_actions[~self.future_mask].count_nonzero().item()):
            raise ValueError("padded future_actions must be zero")
        if bool(self.future_states[~self.future_mask].count_nonzero().item()):
            raise ValueError("padded future_states must be zero")


def extract_success_windows(
    trajectory: Stage0Trajectory,
    policy: Stage0WindowPolicy,
    *,
    corpus_sha256: str | None = None,
) -> tuple[Stage0SuccessWindow, ...]:
    """Extract windows only from a simulator-verified successful trajectory.

    Alignment is fixed as ``current_state=s_t``, ``future_actions=a[t:t+K]``
    and ``future_states=s[t+1:t+K+1]``.  Failure and timeout trajectories are
    deliberately retained by the corpus but return no training windows.
    """
    import torch

    if not isinstance(trajectory, Stage0Trajectory):
        raise TypeError("trajectory must be Stage0Trajectory")
    if not isinstance(policy, Stage0WindowPolicy):
        raise TypeError("policy must be Stage0WindowPolicy")
    if not trajectory.is_success:
        return ()
    source_sha256 = trajectory_sha256(trajectory)
    if corpus_sha256 is None:
        corpus_sha256 = trajectory_corpus_sha256((trajectory,))
    if not isinstance(corpus_sha256, str) or len(corpus_sha256) != 64:
        raise ValueError("corpus_sha256 must be a SHA-256 digest")
    result: list[Stage0SuccessWindow] = []
    transition_count = trajectory.transition_count
    for start in range(0, transition_count, policy.stride):
        valid_future = min(policy.future_horizon, transition_count - start)
        future_actions = torch.zeros(
            (policy.future_horizon, trajectory.action_dim),
            dtype=torch.float32,
        )
        future_states = torch.zeros(
            (policy.future_horizon, trajectory.state_dim),
            dtype=torch.float32,
        )
        future_mask = torch.zeros(policy.future_horizon, dtype=torch.bool)
        future_actions[:valid_future].copy_(
            trajectory.actions[start : start + valid_future]
        )
        future_states[:valid_future].copy_(
            trajectory.states[start + 1 : start + valid_future + 1]
        )
        future_mask[:valid_future] = True
        action_chunk = future_actions[: policy.action_horizon].clone()
        action_mask = future_mask[: policy.action_horizon].clone()
        result.append(
            Stage0SuccessWindow(
                trajectory_id=trajectory.provenance.trajectory_id,
                reset_group_id=trajectory.provenance.reset_group_id,
                start=start,
                current_state=trajectory.states[start].clone(),
                action_chunk=action_chunk,
                action_mask=action_mask,
                future_actions=future_actions,
                future_states=future_states,
                future_mask=future_mask,
                trajectory_sha256=source_sha256,
                corpus_sha256=corpus_sha256,
                window_policy_sha256=policy.sha256,
            )
        )
    return tuple(result)
