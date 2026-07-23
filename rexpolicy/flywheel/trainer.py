"""DDP-synchronized, globally weighted Flow-Matching updates."""

from __future__ import annotations

import contextlib
import math
import random
import time
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Sequence

from .distributed import DistributedContext
from .experience import (
    DIRECT_SUCCESS_ROLE,
    SUCCESS_PATH_ROLE,
    TrainingSample,
    collate_training_samples,
)


@dataclass(frozen=True)
class ScheduledSample:
    """One real or zero-weight padding item in an update schedule."""

    sample: TrainingSample
    weight: float
    category: str
    is_dummy: bool = False


@dataclass(frozen=True)
class UpdateSchedule:
    """A rank-local schedule with a DDP-identical optimizer-step count."""

    steps: tuple[tuple[tuple[ScheduledSample, ...], ...], ...]
    requested_optimizer_steps: int
    optimizer_steps: int
    local_current_samples: int
    local_current_success_samples: int
    local_historical_samples: int
    local_real_slots: int
    local_dummy_slots: int
    local_current_exposed: int
    local_current_success_exposed: int

    @property
    def current_coverage(self) -> float:
        """Return the fraction of current-generation samples scheduled."""
        if self.local_current_samples == 0:
            return 1.0
        return self.local_current_exposed / self.local_current_samples


@dataclass(frozen=True)
class UpdateMetrics:
    """Summary of one generation's synchronized update."""

    mean_loss: float
    optimizer_steps: int
    local_samples: int
    global_samples: int
    mean_selected_advantage: float
    parameter_sync_max_abs: float
    elapsed_seconds: float
    requested_optimizer_steps: int = 0
    global_optimizer_step: int = 0
    local_current_samples: int = 0
    global_current_samples: int = 0
    local_current_success_samples: int = 0
    global_current_success_samples: int = 0
    local_historical_samples: int = 0
    global_historical_samples: int = 0
    local_real_slots: int = 0
    global_real_slots: int = 0
    local_dummy_slots: int = 0
    global_dummy_slots: int = 0
    local_current_exposed: int = 0
    global_current_exposed: int = 0
    current_coverage: float = 0.0
    local_current_success_exposed: int = 0
    global_current_success_exposed: int = 0
    current_success_coverage: float = 0.0
    global_weight: float = 0.0
    gradient_norm: float = 0.0
    gradient_norm_max: float = 0.0
    gradient_norm_rank_spread: float = 0.0
    update_rms: float = 0.0
    update_max_abs: float = 0.0
    changed_fraction: float = 0.0
    update_rms_rank_spread: float = 0.0
    update_max_abs_rank_spread: float = 0.0
    changed_fraction_rank_spread: float = 0.0
    parameter_probe_size: int = 0
    optimizer_restore_seconds: float = 0.0
    optimizer_offload_seconds: float = 0.0
    optimizer_state_device: str = ""


def _is_historical_sample(sample: TrainingSample) -> bool:
    source = str(getattr(sample, "source", "")).lower()
    return source in {"archive", "historical", "history", "success_archive"}


def _is_success_sample(sample: TrainingSample) -> bool:
    source = str(getattr(sample, "source", "")).lower()
    success_roles = set(getattr(sample, "success_roles", ()))
    return bool(
        DIRECT_SUCCESS_ROLE in success_roles
        or SUCCESS_PATH_ROLE in success_roles
        or getattr(sample, "direct_success", False)
        or getattr(sample, "selected_success_path", False)
        or getattr(sample, "success", False)
        or source in {"direct_success", "selected_success_path", "success_path"}
    )


def _sample_identity(sample: TrainingSample) -> tuple[str, Any]:
    sample_id = getattr(sample, "sample_id", None)
    if sample_id not in (None, ""):
        return ("sample_id", str(sample_id))
    return ("object", id(sample))


def _shuffle_group(
    samples: Iterable[TrainingSample],
    *,
    rng: random.Random,
) -> list[TrainingSample]:
    shuffled = list(samples)
    rng.shuffle(shuffled)
    return shuffled


def build_update_schedule(
    current_samples: Sequence[TrainingSample],
    *,
    historical_samples: Sequence[TrainingSample] = (),
    requested_optimizer_steps: int,
    batch_size: int,
    gradient_accumulation: int,
    seed: int,
    rank: int,
    synchronized_item_count: int | None = None,
    dummy_sample: TrainingSample | None = None,
) -> UpdateSchedule:
    """Build success-first, without-replacement coverage followed by replay.

    ``synchronized_item_count`` is the maximum number of real items on any
    rank. Padding up to that count is zero-weight, so a short rank is not
    over-represented merely because DDP requires the same number of backwards.
    Requested steps beyond the coverage epoch use seeded replacement sampling.
    """
    if requested_optimizer_steps < 0:
        raise ValueError("requested_optimizer_steps must be non-negative")
    if batch_size < 1 or gradient_accumulation < 1:
        raise ValueError("batch_size and gradient_accumulation must be positive")

    embedded_history = [
        sample for sample in current_samples if _is_historical_sample(sample)
    ]
    current = [
        sample for sample in current_samples if not _is_historical_sample(sample)
    ]
    history = [*embedded_history, *historical_samples]
    all_samples = [*current, *history]

    explicit_ids = [
        str(getattr(sample, "sample_id"))
        for sample in all_samples
        if getattr(sample, "sample_id", None) not in (None, "")
    ]
    if len(explicit_ids) != len(set(explicit_ids)):
        raise RuntimeError("Duplicate TrainingSample.sample_id in update schedule")
    for sample in all_samples:
        weight = float(getattr(sample, "sample_weight", 1.0))
        if not math.isfinite(weight) or weight <= 0.0:
            raise ValueError(
                "Every real training sample needs a finite positive weight"
            )
        advantage = float(getattr(sample, "advantage", 0.0))
        if not math.isfinite(advantage):
            raise ValueError("Every training sample needs a finite advantage")

    rng = random.Random(int(seed) + int(rank) * 1_000_003)
    success = _shuffle_group(
        (sample for sample in current if _is_success_sample(sample)),
        rng=rng,
    )
    other_current = _shuffle_group(
        (sample for sample in current if not _is_success_sample(sample)),
        rng=rng,
    )
    history = _shuffle_group(history, rng=rng)
    ordered: list[tuple[TrainingSample, str]] = [
        *((sample, "success") for sample in success),
        *((sample, "current") for sample in other_current),
        *((sample, "history") for sample in history),
    ]

    local_item_count = len(ordered)
    if synchronized_item_count is None:
        synchronized_item_count = local_item_count
    synchronized_item_count = int(synchronized_item_count)
    if synchronized_item_count < local_item_count:
        raise ValueError(
            "synchronized_item_count cannot be smaller than the local item count"
        )
    if synchronized_item_count > 0 and not ordered and dummy_sample is None:
        raise RuntimeError(
            "A rank with no real samples needs dummy_sample for synchronized DDP"
        )
    if synchronized_item_count == 0:
        raise RuntimeError("The global update schedule has no training samples")

    slots_per_step = batch_size * gradient_accumulation
    coverage_steps = math.ceil(synchronized_item_count / slots_per_step)
    optimizer_steps = max(int(requested_optimizer_steps), coverage_steps)
    coverage_capacity = coverage_steps * slots_per_step

    flat: list[ScheduledSample] = []
    for sample, category in ordered:
        flat.append(
            ScheduledSample(
                sample=sample,
                weight=float(getattr(sample, "sample_weight", 1.0)),
                category=category,
            )
        )
    padding_sample = ordered[0][0] if ordered else dummy_sample
    assert padding_sample is not None
    while len(flat) < coverage_capacity:
        flat.append(
            ScheduledSample(
                sample=padding_sample,
                weight=0.0,
                category="dummy",
                is_dummy=True,
            )
        )

    extra_slots = (optimizer_steps - coverage_steps) * slots_per_step
    if ordered:
        for _ in range(extra_slots):
            sample, category = rng.choice(ordered)
            flat.append(
                ScheduledSample(
                    sample=sample,
                    weight=float(getattr(sample, "sample_weight", 1.0)),
                    category=f"{category}_replacement",
                )
            )
    else:
        for _ in range(extra_slots):
            flat.append(
                ScheduledSample(
                    sample=padding_sample,
                    weight=0.0,
                    category="dummy",
                    is_dummy=True,
                )
            )

    steps: list[tuple[tuple[ScheduledSample, ...], ...]] = []
    offset = 0
    for _ in range(optimizer_steps):
        microbatches = []
        for _ in range(gradient_accumulation):
            microbatch = tuple(flat[offset : offset + batch_size])
            if len(microbatch) != batch_size:
                raise AssertionError("Internal update schedule is incomplete")
            microbatches.append(microbatch)
            offset += batch_size
        steps.append(tuple(microbatches))

    current_ids = {_sample_identity(sample) for sample in current}
    current_success_ids = {
        _sample_identity(sample) for sample in current if _is_success_sample(sample)
    }
    exposed_current_ids = {
        _sample_identity(item.sample)
        for item in flat
        if not item.is_dummy and _sample_identity(item.sample) in current_ids
    }
    exposed_success_ids = exposed_current_ids & current_success_ids
    return UpdateSchedule(
        steps=tuple(steps),
        requested_optimizer_steps=int(requested_optimizer_steps),
        optimizer_steps=optimizer_steps,
        local_current_samples=len(current),
        local_current_success_samples=len(current_success_ids),
        local_historical_samples=len(history),
        local_real_slots=sum(not item.is_dummy for item in flat),
        local_dummy_slots=sum(item.is_dummy for item in flat),
        local_current_exposed=len(exposed_current_ids),
        local_current_success_exposed=len(exposed_success_ids),
    )


def _move_nested_tensors(value: Any, device: Any) -> Any:
    """Recursively move tensors while preserving optimizer-state structure."""
    import torch

    if isinstance(value, torch.Tensor):
        return value.to(device=device, non_blocking=False)
    if isinstance(value, dict):
        return {
            key: _move_nested_tensors(item, device) for key, item in value.items()
        }
    if isinstance(value, list):
        return [_move_nested_tensors(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_move_nested_tensors(item, device) for item in value)
    return value


class _ParameterProbe:
    """A deterministic, layer-balanced view of trainable FP32 parameters."""

    def __init__(
        self,
        named_parameters: Sequence[tuple[str, Any]],
        *,
        sample_size: int,
    ) -> None:
        import torch

        if sample_size < 1:
            raise ValueError("parameter_probe_size must be positive")
        parameters = [
            (name, parameter)
            for name, parameter in named_parameters
            if parameter.requires_grad and parameter.numel() > 0
        ]
        if not parameters:
            raise RuntimeError("Flow-DiT has no trainable parameters to probe")

        target = min(
            int(sample_size),
            sum(int(parameter.numel()) for _, parameter in parameters),
        )
        allocations = [0] * len(parameters)
        remaining = target
        active = set(range(len(parameters)))
        while remaining and active:
            share = max(1, remaining // len(active))
            for index in tuple(sorted(active)):
                capacity = int(parameters[index][1].numel()) - allocations[index]
                take = min(capacity, share, remaining)
                allocations[index] += take
                remaining -= take
                if allocations[index] == int(parameters[index][1].numel()):
                    active.remove(index)
                if remaining == 0:
                    break

        self.entries: list[tuple[str, Any, Any]] = []
        for (name, parameter), count in zip(parameters, allocations):
            if count == 0:
                continue
            if count == 1:
                indices = torch.tensor(
                    [parameter.numel() // 2],
                    dtype=torch.int64,
                    device=parameter.device,
                )
            else:
                indices = (
                    torch.arange(count, dtype=torch.int64, device=parameter.device)
                    * (parameter.numel() - 1)
                    // (count - 1)
                )
            self.entries.append((name, parameter, indices))
        self.sample_size = sum(int(indices.numel()) for _, _, indices in self.entries)

    def values(self) -> Any:
        """Gather the probe as one FP32 device tensor."""
        import torch

        return torch.cat(
            [
                parameter.detach().reshape(-1).index_select(0, indices).float()
                for _, parameter, indices in self.entries
            ]
        )


class DitDdpTrainer:
    """Wrap only the action head so DDP synchronizes trainable Flow-DiT."""

    STATE_SCHEMA_VERSION = 1

    def __init__(
        self,
        *,
        action_head: Any,
        context: DistributedContext,
        learning_rate: float,
        weight_decay: float,
        gradient_clip_norm: float,
        optimizer_state_offload: bool = False,
        parameter_probe_size: int = 65_536,
        parameter_sync_tolerance: float = 0.0,
        collate_fn: Callable[..., tuple[Any, Any]] = collate_training_samples,
    ) -> None:
        import torch

        self.torch = torch
        self.context = context
        self.action_head = action_head
        self.gradient_clip_norm = float(gradient_clip_norm)
        self.optimizer_state_offload = bool(optimizer_state_offload)
        self.parameter_sync_tolerance = float(parameter_sync_tolerance)
        self.collate_fn = collate_fn
        self.global_optimizer_step = 0
        self._optimizer_state_device = context.device

        self._named_trainable_parameters = [
            (name, parameter)
            for name, parameter in action_head.named_parameters()
            if parameter.requires_grad
        ]
        if not self._named_trainable_parameters:
            raise RuntimeError("Flow-DiT has no trainable parameters")
        self.validate_fp32_contract()

        if context.initialized:
            from torch.nn.parallel import DistributedDataParallel

            ddp_kwargs: dict[str, Any] = {
                "broadcast_buffers": False,
                "find_unused_parameters": False,
                "gradient_as_bucket_view": True,
            }
            if context.device.type == "cuda":
                ddp_kwargs.update(
                    device_ids=[context.local_rank],
                    output_device=context.local_rank,
                )
            self.module = DistributedDataParallel(action_head, **ddp_kwargs)
        else:
            self.module = action_head

        parameters = [
            parameter for _, parameter in self._named_trainable_parameters
        ]
        optimizer_kwargs: dict[str, Any] = {
            "lr": float(learning_rate),
            "weight_decay": float(weight_decay),
        }
        if context.device.type == "cuda":
            optimizer_kwargs["fused"] = True
        self.optimizer = torch.optim.AdamW(parameters, **optimizer_kwargs)
        self._probe = _ParameterProbe(
            self._named_trainable_parameters,
            sample_size=int(parameter_probe_size),
        )

    @property
    def trainable_parameters(self) -> list[Any]:
        """Return trainable parameters in their stable optimizer order."""
        return [parameter for _, parameter in self._named_trainable_parameters]

    def validate_fp32_contract(self) -> None:
        """Require every trainable parameter to be FP32 on this rank's device."""
        torch = self.torch
        bad_dtype = [
            f"{name}:{parameter.dtype}"
            for name, parameter in self._named_trainable_parameters
            if parameter.dtype != torch.float32
        ]
        bad_device = [
            f"{name}:{parameter.device}"
            for name, parameter in self._named_trainable_parameters
            if parameter.device != self.context.device
        ]
        if bad_dtype or bad_device:
            details = []
            if bad_dtype:
                details.append("non-FP32=" + ", ".join(bad_dtype[:8]))
            if bad_device:
                details.append("wrong-device=" + ", ".join(bad_device[:8]))
            raise RuntimeError(
                "Trainable Flow-DiT contract violation: " + "; ".join(details)
            )

    def move_optimizer_state(self, device: Any) -> None:
        """Recursively move all Adam state tensors to ``device``."""
        target = self.torch.device(device)
        for parameter, state in list(self.optimizer.state.items()):
            self.optimizer.state[parameter] = _move_nested_tensors(state, target)
        self._optimizer_state_device = target
        if target.type == "cpu" and self.context.device.type == "cuda":
            self.torch.cuda.empty_cache()

    def offload_optimizer_state(self) -> None:
        """Move Adam moments to ordinary CPU memory between update phases."""
        self.move_optimizer_state(self.torch.device("cpu"))

    def restore_optimizer_state(self) -> None:
        """Move Adam moments back to the trainable parameter device."""
        self.move_optimizer_state(self.context.device)

    def prepare_update(self) -> None:
        """Restore optimizer state and enter the trainable update phase."""
        self.restore_optimizer_state()
        self.validate_fp32_contract()
        self.module.train()

    def finish_update(self) -> None:
        """Clear gradients, leave train mode, and optionally offload Adam state."""
        self.optimizer.zero_grad(set_to_none=True)
        self.module.eval()
        if self.optimizer_state_offload:
            self.offload_optimizer_state()

    def state_dict(self) -> dict[str, Any]:
        """Return CPU optimizer state and trainer counters for checkpointing."""
        return {
            "schema_version": self.STATE_SCHEMA_VERSION,
            "optimizer": _move_nested_tensors(
                self.optimizer.state_dict(),
                self.torch.device("cpu"),
            ),
            "global_optimizer_step": self.global_optimizer_step,
            "optimizer_state_offload": self.optimizer_state_offload,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore optimizer state and counters from a trainer checkpoint."""
        if int(state.get("schema_version", -1)) != self.STATE_SCHEMA_VERSION:
            raise RuntimeError(
                f"Unsupported trainer state schema {state.get('schema_version')}"
            )
        self.optimizer.load_state_dict(state["optimizer"])
        self.global_optimizer_step = int(state["global_optimizer_step"])
        target = (
            self.torch.device("cpu")
            if self.optimizer_state_offload
            else self.context.device
        )
        self.move_optimizer_state(target)
        self._validate_adam_state()

    def _validate_adam_state(self) -> None:
        for state in self.optimizer.state.values():
            for key in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
                value = state.get(key)
                if value is not None and value.dtype != self.torch.float32:
                    raise RuntimeError(
                        f"Adam state {key} must remain FP32, got {value.dtype}"
                    )

    def _all_reduce_float(self, value: float, *, operation: str = "sum") -> float:
        if not self.context.initialized:
            return float(value)
        import torch.distributed as dist

        tensor = self.torch.tensor(
            float(value),
            dtype=self.torch.float64,
            device=self.context.device,
        )
        op = {
            "sum": dist.ReduceOp.SUM,
            "min": dist.ReduceOp.MIN,
            "max": dist.ReduceOp.MAX,
        }[operation]
        dist.all_reduce(tensor, op=op)
        return float(tensor.item())

    def _all_reduce_int(self, value: int, *, operation: str = "sum") -> int:
        if not self.context.initialized:
            return int(value)
        import torch.distributed as dist

        tensor = self.torch.tensor(
            int(value),
            dtype=self.torch.int64,
            device=self.context.device,
        )
        op = {
            "sum": dist.ReduceOp.SUM,
            "min": dist.ReduceOp.MIN,
            "max": dist.ReduceOp.MAX,
        }[operation]
        dist.all_reduce(tensor, op=op)
        return int(tensor.item())

    def _require_distributed_equal(self, value: int, *, name: str) -> None:
        minimum = self._all_reduce_int(value, operation="min")
        maximum = self._all_reduce_int(value, operation="max")
        if minimum != maximum:
            raise RuntimeError(
                f"All ranks need the same {name}, got range [{minimum}, {maximum}]"
            )

    def _require_all_ranks(self, condition: bool, *, message: str) -> None:
        passed = self._all_reduce_int(int(condition), operation="min")
        if not passed:
            raise FloatingPointError(message)

    def _parameter_sync_error(self) -> float:
        local = self._probe.values()
        if not self.context.initialized:
            return 0.0
        import torch.distributed as dist

        minimum = local.clone()
        maximum = local.clone()
        dist.all_reduce(minimum, op=dist.ReduceOp.MIN)
        dist.all_reduce(maximum, op=dist.ReduceOp.MAX)
        return float((maximum - minimum).abs().max().item())

    def _autocast(self, dtype: Any) -> Any:
        enabled = (
            self.context.device.type in {"cpu", "cuda"}
            and dtype in {self.torch.bfloat16, self.torch.float16}
        )
        if not enabled:
            return contextlib.nullcontext()
        return self.torch.autocast(
            device_type=self.context.device.type,
            dtype=dtype,
            enabled=True,
        )

    def update(
        self,
        samples: list[TrainingSample],
        *,
        optimizer_steps: int,
        batch_size: int,
        gradient_accumulation: int,
        seed: int,
        dtype: Any,
        historical_samples: Sequence[TrainingSample] = (),
        dummy_sample: TrainingSample | None = None,
    ) -> UpdateMetrics:
        """Update FP32 DiT using the exact global weighted objective.

        DDP averages gradients, so every local microbatch contributes
        ``world_size * local_weighted_loss / global_window_weight``. The
        denominator covers the complete accumulation window across all ranks.
        """
        if optimizer_steps < 0 or batch_size < 1 or gradient_accumulation < 1:
            raise ValueError("Invalid update schedule")
        self.validate_fp32_contract()
        for value, name in (
            (optimizer_steps, "optimizer_steps"),
            (batch_size, "batch_size"),
            (gradient_accumulation, "gradient_accumulation"),
        ):
            self._require_distributed_equal(int(value), name=name)

        embedded_history = [
            sample for sample in samples if _is_historical_sample(sample)
        ]
        local_current_count = len(samples) - len(embedded_history)
        local_history_count = len(embedded_history) + len(historical_samples)
        local_item_count = local_current_count + local_history_count
        global_samples = self._all_reduce_int(local_item_count)
        if global_samples == 0:
            raise RuntimeError("No selected chunk samples exist on any rank")
        synchronized_item_count = self._all_reduce_int(
            local_item_count,
            operation="max",
        )
        schedule = build_update_schedule(
            samples,
            historical_samples=historical_samples,
            requested_optimizer_steps=optimizer_steps,
            batch_size=batch_size,
            gradient_accumulation=gradient_accumulation,
            seed=seed,
            rank=self.context.rank,
            synchronized_item_count=synchronized_item_count,
            dummy_sample=dummy_sample,
        )
        self._require_distributed_equal(
            schedule.optimizer_steps,
            name="effective optimizer_steps",
        )

        all_unique_samples = [
            *[sample for sample in samples if not _is_historical_sample(sample)],
            *embedded_history,
            *historical_samples,
        ]
        local_advantage_sum = sum(
            float(getattr(sample, "advantage", 0.0))
            for sample in all_unique_samples
        )
        global_advantage_sum = self._all_reduce_float(local_advantage_sum)
        mean_advantage = global_advantage_sum / global_samples

        optimizer_restore_started = time.perf_counter()
        self.prepare_update()
        optimizer_restore_seconds = (
            time.perf_counter() - optimizer_restore_started
        )
        started = time.perf_counter()
        local_numerator_total = 0.0
        global_weight_total = 0.0
        gradient_norms: list[float] = []
        update_squared_sum = 0.0
        update_max_abs = 0.0
        changed_count = 0
        probed_count = 0
        optimizer_offload_seconds = 0.0

        try:
            for step in schedule.steps:
                self.optimizer.zero_grad(set_to_none=True)
                local_window_weight = sum(
                    item.weight
                    for microbatch in step
                    for item in microbatch
                    if not item.is_dummy
                )
                global_window_weight = self._all_reduce_float(local_window_weight)
                if not math.isfinite(global_window_weight) or global_window_weight <= 0:
                    raise FloatingPointError(
                        "Global accumulation-window sample weight must be finite "
                        "and positive"
                    )
                global_weight_total += global_window_weight
                before = self._probe.values()

                for accumulation_index, scheduled_batch in enumerate(step):
                    batch = [item.sample for item in scheduled_batch]
                    backbone_output, action_input = self.collate_fn(
                        batch,
                        device=self.context.device,
                        dtype=dtype,
                    )
                    synchronize_now = accumulation_index + 1 == len(step)
                    sync_context = (
                        contextlib.nullcontext()
                        if synchronize_now or not hasattr(self.module, "no_sync")
                        else self.module.no_sync()
                    )
                    with sync_context, self._autocast(dtype):
                        output = self.module(backbone_output, action_input)
                        action_loss = output["action_loss"].float()
                        action_mask = output["weighted_action_mask"].float()
                        per_sample_loss = action_loss.sum(dim=(1, 2)) / (
                            action_mask.sum(dim=(1, 2)) + 1.0e-6
                        )
                        self._require_all_ranks(
                            bool(self.torch.isfinite(per_sample_loss).all()),
                            message="Non-finite per-sample Flow-Matching loss "
                            "on one or more ranks",
                        )
                        sample_weight = self.torch.tensor(
                            [
                                0.0 if item.is_dummy else item.weight
                                for item in scheduled_batch
                            ],
                            dtype=self.torch.float32,
                            device=self.context.device,
                        )
                        local_numerator = (per_sample_loss * sample_weight).sum()
                        loss = local_numerator * (
                            self.context.world_size / global_window_weight
                        )
                        self._require_all_ranks(
                            bool(self.torch.isfinite(loss)),
                            message="Non-finite globally normalized "
                            "Flow-Matching loss on one or more ranks",
                        )
                        loss.backward()
                    local_numerator_total += float(local_numerator.detach().item())

                clip_limit = (
                    self.gradient_clip_norm
                    if self.gradient_clip_norm > 0.0
                    else float("inf")
                )
                gradient_norm = self.torch.nn.utils.clip_grad_norm_(
                    self.trainable_parameters,
                    clip_limit,
                    error_if_nonfinite=True,
                )
                gradient_norm_value = float(gradient_norm.item())
                if (
                    not math.isfinite(gradient_norm_value)
                    or gradient_norm_value <= 0.0
                ):
                    raise FloatingPointError(
                        "Flow-DiT gradient norm must be finite and non-zero, got "
                        f"{gradient_norm_value}"
                    )
                gradient_norms.append(gradient_norm_value)

                self.optimizer.step()
                self.global_optimizer_step += 1
                self._validate_adam_state()
                after = self._probe.values()
                delta = after - before
                if not bool(self.torch.isfinite(delta).all()):
                    raise FloatingPointError("Non-finite Flow-DiT parameter update")
                step_max = float(delta.abs().max().item())
                if step_max == 0.0:
                    raise RuntimeError(
                        "Optimizer step produced no effective FP32 parameter update"
                    )
                step_squared_sum = float(delta.double().square().sum().item())
                if not math.isfinite(step_squared_sum):
                    raise FloatingPointError(
                        "Non-finite Flow-DiT parameter update magnitude"
                    )
                update_squared_sum += step_squared_sum
                update_max_abs = max(update_max_abs, step_max)
                changed_count += int((delta != 0.0).sum().item())
                probed_count += int(delta.numel())

            parameter_sync_max_abs = self._parameter_sync_error()
            if (
                not math.isfinite(parameter_sync_max_abs)
                or parameter_sync_max_abs > self.parameter_sync_tolerance
            ):
                raise RuntimeError(
                    "Flow-DiT parameters diverged across ranks: "
                    f"max_abs={parameter_sync_max_abs:.9g}, "
                    f"tolerance={self.parameter_sync_tolerance:.9g}"
                )
        finally:
            optimizer_offload_started = time.perf_counter()
            self.finish_update()
            optimizer_offload_seconds = (
                time.perf_counter() - optimizer_offload_started
            )

        if self.context.device.type == "cuda":
            self.torch.cuda.synchronize(self.context.device)
        elapsed = time.perf_counter() - started
        global_numerator_total = self._all_reduce_float(local_numerator_total)
        global_real_slots = self._all_reduce_int(schedule.local_real_slots)
        global_dummy_slots = self._all_reduce_int(schedule.local_dummy_slots)
        global_current_samples = self._all_reduce_int(schedule.local_current_samples)
        global_current_success_samples = self._all_reduce_int(
            schedule.local_current_success_samples
        )
        global_historical_samples = self._all_reduce_int(
            schedule.local_historical_samples
        )
        global_current_exposed = self._all_reduce_int(
            schedule.local_current_exposed
        )
        global_current_success_exposed = self._all_reduce_int(
            schedule.local_current_success_exposed
        )
        current_coverage = (
            global_current_exposed / global_current_samples
            if global_current_samples
            else 1.0
        )
        if current_coverage != 1.0:
            raise RuntimeError(
                f"Current-generation sample coverage is incomplete: {current_coverage}"
            )
        current_success_coverage = (
            global_current_success_exposed / global_current_success_samples
            if global_current_success_samples
            else 1.0
        )
        if current_success_coverage != 1.0:
            raise RuntimeError(
                "Current-generation success coverage is incomplete: "
                f"{current_success_coverage}"
            )

        mean_gradient_norm = sum(gradient_norms) / len(gradient_norms)
        local_update_rms = math.sqrt(update_squared_sum / probed_count)
        local_changed_fraction = changed_count / probed_count

        def rank_spread(value: float) -> float:
            return self._all_reduce_float(
                value,
                operation="max",
            ) - self._all_reduce_float(value, operation="min")

        return UpdateMetrics(
            mean_loss=global_numerator_total / global_weight_total,
            optimizer_steps=schedule.optimizer_steps,
            local_samples=local_item_count,
            global_samples=global_samples,
            mean_selected_advantage=mean_advantage,
            parameter_sync_max_abs=parameter_sync_max_abs,
            elapsed_seconds=elapsed,
            requested_optimizer_steps=schedule.requested_optimizer_steps,
            global_optimizer_step=self.global_optimizer_step,
            local_current_samples=schedule.local_current_samples,
            global_current_samples=global_current_samples,
            local_current_success_samples=(
                schedule.local_current_success_samples
            ),
            global_current_success_samples=global_current_success_samples,
            local_historical_samples=schedule.local_historical_samples,
            global_historical_samples=global_historical_samples,
            local_real_slots=schedule.local_real_slots,
            global_real_slots=global_real_slots,
            local_dummy_slots=schedule.local_dummy_slots,
            global_dummy_slots=global_dummy_slots,
            local_current_exposed=schedule.local_current_exposed,
            global_current_exposed=global_current_exposed,
            current_coverage=current_coverage,
            local_current_success_exposed=(
                schedule.local_current_success_exposed
            ),
            global_current_success_exposed=global_current_success_exposed,
            current_success_coverage=current_success_coverage,
            global_weight=global_weight_total,
            gradient_norm=mean_gradient_norm,
            gradient_norm_max=max(gradient_norms),
            gradient_norm_rank_spread=rank_spread(mean_gradient_norm),
            update_rms=local_update_rms,
            update_max_abs=update_max_abs,
            changed_fraction=local_changed_fraction,
            update_rms_rank_spread=rank_spread(local_update_rms),
            update_max_abs_rank_spread=rank_spread(update_max_abs),
            changed_fraction_rank_spread=rank_spread(
                local_changed_fraction
            ),
            parameter_probe_size=self._probe.sample_size,
            optimizer_restore_seconds=optimizer_restore_seconds,
            optimizer_offload_seconds=optimizer_offload_seconds,
            optimizer_state_device=str(self._optimizer_state_device),
        )
