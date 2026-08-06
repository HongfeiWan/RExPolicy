"""Strict collation for transition-aligned Stage 0 success windows."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Sequence

import torch

from rexpolicy.stage0.data.windows import Stage0SuccessWindow


@dataclass(frozen=True)
class Stage0WindowBatch:
    """One homogeneous training batch.

    Both masks use the persisted-window convention: ``True`` means valid.
    ``future_padding_mask`` is the only supported conversion to the encoder's
    PyTorch padding convention, where ``True`` instead means padding.
    """

    current_states: torch.Tensor
    action_chunks: torch.Tensor
    action_mask: torch.Tensor
    future_actions: torch.Tensor
    future_states: torch.Tensor
    future_mask: torch.Tensor
    trajectory_ids: tuple[str, ...]
    reset_group_ids: tuple[str, ...]
    starts: tuple[int, ...]
    corpus_sha256: str
    window_policy_sha256: str

    def __post_init__(self) -> None:
        batch = len(self.trajectory_ids)
        if batch < 1:
            raise ValueError("a Stage 0 window batch cannot be empty")
        if len(self.reset_group_ids) != batch or len(self.starts) != batch:
            raise ValueError("window metadata lengths must match the batch")
        if any(not value for value in self.trajectory_ids):
            raise ValueError("trajectory IDs must be non-empty")
        if any(not value for value in self.reset_group_ids):
            raise ValueError("reset-group IDs must be non-empty")
        if any(type(value) is not int or value < 0 for value in self.starts):
            raise ValueError("window starts must be non-negative integers")
        for name in ("corpus_sha256", "window_policy_sha256"):
            value = getattr(self, name)
            if not isinstance(value, str) or len(value) != 64:
                raise ValueError(f"{name} must be a SHA-256 digest")

        floats = {
            "current_states": (self.current_states, 2),
            "action_chunks": (self.action_chunks, 3),
            "future_actions": (self.future_actions, 3),
            "future_states": (self.future_states, 3),
        }
        device: torch.device | None = None
        dtype: torch.dtype | None = None
        for name, (value, rank) in floats.items():
            if not isinstance(value, torch.Tensor) or value.ndim != rank:
                raise ValueError(f"{name} must be rank {rank}")
            if not torch.is_floating_point(value) or not bool(
                torch.isfinite(value).all()
            ):
                raise ValueError(f"{name} must be finite floating point")
            if value.shape[0] != batch:
                raise ValueError(f"{name} batch dimension is inconsistent")
            if device is None:
                device, dtype = value.device, value.dtype
            elif value.device != device or value.dtype != dtype:
                raise ValueError("all floating tensors must share device and dtype")
        if self.current_states.shape[-1] != self.future_states.shape[-1]:
            raise ValueError("current and future state dimensions differ")
        if self.action_chunks.shape[-1] != self.future_actions.shape[-1]:
            raise ValueError("action chunk and future action dimensions differ")

        masks = {
            "action_mask": (self.action_mask, self.action_chunks.shape[:2]),
            "future_mask": (self.future_mask, self.future_actions.shape[:2]),
        }
        for name, (value, shape) in masks.items():
            if (
                not isinstance(value, torch.Tensor)
                or value.dtype != torch.bool
                or tuple(value.shape) != tuple(shape)
                or value.device != device
            ):
                raise ValueError(
                    f"{name} must be bool {list(shape)} on the batch device"
                )
            if bool((~value).all(dim=1).any()):
                raise ValueError(f"every row in {name} must contain a valid step")
        if self.action_mask.shape[1] > self.future_mask.shape[1]:
            raise ValueError("action horizon must not exceed future horizon")
        if not torch.equal(
            self.action_mask,
            self.future_mask[:, : self.action_mask.shape[1]],
        ):
            raise ValueError("action_mask must equal the future-mask prefix")

    @property
    def batch_size(self) -> int:
        return int(self.current_states.shape[0])

    @property
    def future_padding_mask(self) -> torch.Tensor:
        """Return encoder padding mask (``True`` means padding)."""
        return ~self.future_mask

    def to(
        self,
        device: torch.device | str,
        *,
        dtype: torch.dtype | None = None,
    ) -> Stage0WindowBatch:
        def move(value: torch.Tensor) -> torch.Tensor:
            if torch.is_floating_point(value):
                return value.to(device=device, dtype=dtype)
            return value.to(device=device)

        return replace(
            self,
            current_states=move(self.current_states),
            action_chunks=move(self.action_chunks),
            action_mask=move(self.action_mask),
            future_actions=move(self.future_actions),
            future_states=move(self.future_states),
            future_mask=move(self.future_mask),
        )


def collate_success_windows(
    windows: Sequence[Stage0SuccessWindow],
) -> Stage0WindowBatch:
    """Stack one corpus/policy-homogeneous sequence without changing masks."""
    if isinstance(windows, (str, bytes)) or not isinstance(windows, Sequence):
        raise TypeError("windows must be a sequence of Stage0SuccessWindow")
    items = tuple(windows)
    if not items:
        raise ValueError("cannot collate an empty window sequence")
    if any(not isinstance(item, Stage0SuccessWindow) for item in items):
        raise TypeError("every item must be a Stage0SuccessWindow")
    corpus_hashes = {item.corpus_sha256 for item in items}
    policy_hashes = {item.window_policy_sha256 for item in items}
    if len(corpus_hashes) != 1 or len(policy_hashes) != 1:
        raise ValueError("a batch cannot mix corpus or window-policy hashes")
    try:
        return Stage0WindowBatch(
            current_states=torch.stack(tuple(item.current_state for item in items)),
            action_chunks=torch.stack(tuple(item.action_chunk for item in items)),
            action_mask=torch.stack(tuple(item.action_mask for item in items)),
            future_actions=torch.stack(tuple(item.future_actions for item in items)),
            future_states=torch.stack(tuple(item.future_states for item in items)),
            future_mask=torch.stack(tuple(item.future_mask for item in items)),
            trajectory_ids=tuple(item.trajectory_id for item in items),
            reset_group_ids=tuple(item.reset_group_id for item in items),
            starts=tuple(item.start for item in items),
            corpus_sha256=items[0].corpus_sha256,
            window_policy_sha256=items[0].window_policy_sha256,
        )
    except RuntimeError as exc:
        raise ValueError("windows in one batch must have identical shapes") from exc
