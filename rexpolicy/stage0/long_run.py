"""Configuration and durable control primitives for Stage 0 long runs.

This module is intentionally simulator- and Torch-free.  It can validate a
long-run contract, recover the exact training cursor from a checkpoint, and
commit monitoring records before any CUDA or Newton dependency is imported.
"""

from __future__ import annotations

import json
import math
import os
import re
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from rexpolicy.stage0.config import Stage0Config
from rexpolicy.stage0.types import canonical_fingerprint


STAGE0_LONG_RUN_SCHEMA_VERSION = 2
STAGE0_LONG_RUN_SCHEMA_ID = "rexpolicy/stage0-long-run/v2"
_SAFE_MODE_ID = re.compile(r"^[a-z0-9][a-z0-9_./-]{0,127}$")


def _exact_mapping(value: Any, keys: set[str], context: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must be an object")
    actual = set(value)
    if actual != keys:
        raise ValueError(
            f"{context} keys changed: missing={sorted(keys - actual)}, "
            f"unexpected={sorted(actual - keys)}"
        )
    return dict(value)


def _integer(
    value: Any,
    name: str,
    *,
    minimum: int,
    maximum: int | None = None,
) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    if maximum is not None and value > maximum:
        raise ValueError(f"{name} must be <= {maximum}")
    return value


def _finite_number(value: Any, name: str, *, minimum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < minimum:
        raise ValueError(f"{name} must be finite and >= {minimum}")
    return result


def _mode_ids(value: Any) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError("corpus.mode_ids must be an array")
    result = tuple(value)
    if len(result) < 2:
        raise ValueError("corpus.mode_ids must contain at least two modes")
    for mode_id in result:
        if (
            not isinstance(mode_id, str)
            or _SAFE_MODE_ID.fullmatch(mode_id) is None
            or mode_id.startswith("/")
            or mode_id.endswith("/")
            or ".." in mode_id
        ):
            raise ValueError(
                "corpus.mode_ids entries must be safe versioned identifiers"
            )
    if len(set(result)) != len(result):
        raise ValueError("corpus.mode_ids must be unique")
    return result


@dataclass(frozen=True)
class LongRunCorpusConfig:
    """Contract for a same-reset, multiple-success-mode trajectory corpus."""

    reset_groups: int
    mode_ids: tuple[str, ...]
    minimum_success_modes_per_reset: int

    def __post_init__(self) -> None:
        _integer(self.reset_groups, "corpus.reset_groups", minimum=3)
        modes = _mode_ids(self.mode_ids)
        object.__setattr__(self, "mode_ids", modes)
        _integer(
            self.minimum_success_modes_per_reset,
            "corpus.minimum_success_modes_per_reset",
            minimum=2,
            maximum=len(modes),
        )

    @classmethod
    def from_mapping(cls, value: Any) -> LongRunCorpusConfig:
        record = _exact_mapping(
            value,
            {"minimum_success_modes_per_reset", "mode_ids", "reset_groups"},
            "long-run corpus config",
        )
        record["mode_ids"] = _mode_ids(record["mode_ids"])
        return cls(**record)

    def to_record(self) -> dict[str, Any]:
        return {
            "minimum_success_modes_per_reset": self.minimum_success_modes_per_reset,
            "mode_ids": list(self.mode_ids),
            "reset_groups": self.reset_groups,
        }


@dataclass(frozen=True)
class LongRunTrainingConfig:
    """Step budgets and persistence cadence for all independent components."""

    batch_size: int
    cuda_fused_adamw: bool
    learning_rate: float
    weight_decay: float
    future_encoder_steps: int
    selector_steps: int
    conditional_policy_steps: int
    no_z_policy_steps: int
    checkpoint_every_steps: int
    eval_every_steps: int
    log_every_steps: int

    def __post_init__(self) -> None:
        for name in (
            "batch_size",
            "future_encoder_steps",
            "selector_steps",
            "conditional_policy_steps",
            "no_z_policy_steps",
            "checkpoint_every_steps",
            "eval_every_steps",
            "log_every_steps",
        ):
            _integer(getattr(self, name), f"training.{name}", minimum=1)
        _finite_number(
            self.learning_rate,
            "training.learning_rate",
            minimum=math.nextafter(0.0, 1.0),
        )
        _finite_number(self.weight_decay, "training.weight_decay", minimum=0.0)
        if not isinstance(self.cuda_fused_adamw, bool):
            raise ValueError("training.cuda_fused_adamw must be boolean")
        if self.batch_size < 8 or self.batch_size % 8:
            raise ValueError(
                "training.batch_size must be a multiple of eight for paired "
                "same-mode cross-reset contrastive groups"
            )
        if self.conditional_policy_steps != self.no_z_policy_steps:
            raise ValueError(
                "conditional_policy_steps and no_z_policy_steps must match for "
                "an equal-budget baseline"
            )

    @classmethod
    def from_mapping(cls, value: Any) -> LongRunTrainingConfig:
        keys = {
            "batch_size",
            "checkpoint_every_steps",
            "conditional_policy_steps",
            "cuda_fused_adamw",
            "eval_every_steps",
            "future_encoder_steps",
            "learning_rate",
            "log_every_steps",
            "no_z_policy_steps",
            "selector_steps",
            "weight_decay",
        }
        return cls(**_exact_mapping(value, keys, "long-run training config"))

    def to_record(self) -> dict[str, Any]:
        return {
            "batch_size": self.batch_size,
            "checkpoint_every_steps": self.checkpoint_every_steps,
            "conditional_policy_steps": self.conditional_policy_steps,
            "cuda_fused_adamw": self.cuda_fused_adamw,
            "eval_every_steps": self.eval_every_steps,
            "future_encoder_steps": self.future_encoder_steps,
            "learning_rate": self.learning_rate,
            "log_every_steps": self.log_every_steps,
            "no_z_policy_steps": self.no_z_policy_steps,
            "selector_steps": self.selector_steps,
            "weight_decay": self.weight_decay,
        }


@dataclass(frozen=True)
class LongRunRuntimeConfig:
    """Process limits and deterministic evaluation settings."""

    cpu_threads: int
    max_wall_seconds: float
    seed: int
    eval_window_count: int
    flow_sample_steps: int
    rollout_capture_graph: bool

    def __post_init__(self) -> None:
        _integer(self.cpu_threads, "runtime.cpu_threads", minimum=1)
        _finite_number(
            self.max_wall_seconds,
            "runtime.max_wall_seconds",
            minimum=math.nextafter(0.0, 1.0),
        )
        _integer(self.seed, "runtime.seed", minimum=0, maximum=(1 << 63) - 1)
        _integer(self.eval_window_count, "runtime.eval_window_count", minimum=2)
        _integer(self.flow_sample_steps, "runtime.flow_sample_steps", minimum=1)
        if not isinstance(self.rollout_capture_graph, bool):
            raise ValueError("runtime.rollout_capture_graph must be boolean")

    @classmethod
    def from_mapping(cls, value: Any) -> LongRunRuntimeConfig:
        keys = {
            "cpu_threads",
            "eval_window_count",
            "flow_sample_steps",
            "max_wall_seconds",
            "rollout_capture_graph",
            "seed",
        }
        return cls(**_exact_mapping(value, keys, "long-run runtime config"))

    def to_record(self) -> dict[str, Any]:
        return {
            "cpu_threads": self.cpu_threads,
            "eval_window_count": self.eval_window_count,
            "flow_sample_steps": self.flow_sample_steps,
            "max_wall_seconds": self.max_wall_seconds,
            "rollout_capture_graph": self.rollout_capture_graph,
            "seed": self.seed,
        }


@dataclass(frozen=True)
class Stage0LongRunConfig:
    """Complete fingerprinted contract for one resumable Stage 0 long run."""

    stage0: Stage0Config
    corpus: LongRunCorpusConfig
    training: LongRunTrainingConfig
    runtime: LongRunRuntimeConfig
    schema_version: int = STAGE0_LONG_RUN_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != STAGE0_LONG_RUN_SCHEMA_VERSION:
            raise ValueError(
                f"long-run schema_version must be {STAGE0_LONG_RUN_SCHEMA_VERSION}"
            )
        if not isinstance(self.stage0, Stage0Config):
            raise TypeError("stage0 must be Stage0Config")
        self.stage0.require_enabled_stage0()
        if self.stage0.task.value != "reach":
            raise ValueError("Stage 0 long-run v1 supports Reach only")
        if self.stage0.window_stride != 1:
            raise ValueError(
                "Stage 0 long-run grouped temporal positives require window_stride=1"
            )
        if not isinstance(self.corpus, LongRunCorpusConfig):
            raise TypeError("corpus must be LongRunCorpusConfig")
        if not isinstance(self.training, LongRunTrainingConfig):
            raise TypeError("training must be LongRunTrainingConfig")
        if not isinstance(self.runtime, LongRunRuntimeConfig):
            raise TypeError("runtime must be LongRunRuntimeConfig")
        if self.stage0.num_envs > self.corpus.reset_groups:
            raise ValueError("stage0.num_envs cannot exceed corpus.reset_groups")
        if self.corpus.reset_groups % self.stage0.num_envs:
            raise ValueError("corpus.reset_groups must be divisible by stage0.num_envs")

    @classmethod
    def from_mapping(cls, value: Any) -> Stage0LongRunConfig:
        record = _exact_mapping(
            value,
            {"corpus", "runtime", "schema_version", "stage0", "training"},
            "Stage 0 long-run config",
        )
        return cls(
            schema_version=record["schema_version"],
            stage0=Stage0Config.from_mapping(record["stage0"]),
            corpus=LongRunCorpusConfig.from_mapping(record["corpus"]),
            training=LongRunTrainingConfig.from_mapping(record["training"]),
            runtime=LongRunRuntimeConfig.from_mapping(record["runtime"]),
        )

    def to_record(self) -> dict[str, Any]:
        return {
            "corpus": self.corpus.to_record(),
            "runtime": self.runtime.to_record(),
            "schema_version": self.schema_version,
            "stage0": self.stage0.to_record(),
            "training": self.training.to_record(),
        }

    @property
    def fingerprint(self) -> str:
        return canonical_fingerprint(
            {"schema_id": STAGE0_LONG_RUN_SCHEMA_ID, **self.to_record()}
        )


def load_long_run_config(path: str | Path) -> Stage0LongRunConfig:
    source = Path(path)
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(
            f"cannot read Stage 0 long-run config {source}: {error}"
        ) from error
    return Stage0LongRunConfig.from_mapping(value)


class LongRunPhase(str, Enum):
    FUTURE_ENCODER = "future_encoder"
    SELECTOR = "selector"
    CONDITIONAL_POLICY = "conditional_policy"
    NO_Z_POLICY = "no_z_policy"
    COMPLETE = "complete"


_TRAINING_PHASES = tuple(
    phase for phase in LongRunPhase if phase is not LongRunPhase.COMPLETE
)


@dataclass(frozen=True)
class LongRunCursor:
    """Monotonic component steps restored from checkpoint metadata."""

    future_encoder: int = 0
    selector: int = 0
    conditional_policy: int = 0
    no_z_policy: int = 0

    def __post_init__(self) -> None:
        for phase in _TRAINING_PHASES:
            _integer(getattr(self, phase.value), phase.value, minimum=0)

    @property
    def global_step(self) -> int:
        return sum(getattr(self, phase.value) for phase in _TRAINING_PHASES)

    def to_steps(self) -> dict[str, int]:
        result = {phase.value: getattr(self, phase.value) for phase in _TRAINING_PHASES}
        result["global"] = self.global_step
        return result

    @classmethod
    def from_steps(cls, value: Any) -> LongRunCursor:
        keys = {phase.value for phase in _TRAINING_PHASES} | {"global"}
        record = _exact_mapping(value, keys, "long-run checkpoint steps")
        cursor = cls(**{phase.value: record[phase.value] for phase in _TRAINING_PHASES})
        if record["global"] != cursor.global_step:
            raise ValueError("long-run checkpoint global step is inconsistent")
        return cursor

    def target(self, config: LongRunTrainingConfig, phase: LongRunPhase) -> int:
        if phase is LongRunPhase.COMPLETE:
            return self.global_step
        field = {
            LongRunPhase.FUTURE_ENCODER: "future_encoder_steps",
            LongRunPhase.SELECTOR: "selector_steps",
            LongRunPhase.CONDITIONAL_POLICY: "conditional_policy_steps",
            LongRunPhase.NO_Z_POLICY: "no_z_policy_steps",
        }[phase]
        return int(getattr(config, field))

    def next_phase(self, config: LongRunTrainingConfig) -> LongRunPhase:
        if not isinstance(config, LongRunTrainingConfig):
            raise TypeError("config must be LongRunTrainingConfig")
        for phase in _TRAINING_PHASES:
            completed = getattr(self, phase.value)
            target = self.target(config, phase)
            if completed > target:
                raise ValueError(f"checkpoint exceeds configured {phase.value} budget")
            if completed < target:
                return phase
        return LongRunPhase.COMPLETE

    def advance(
        self, config: LongRunTrainingConfig, phase: LongRunPhase
    ) -> LongRunCursor:
        if phase is LongRunPhase.COMPLETE:
            raise ValueError("cannot advance the complete phase")
        expected = self.next_phase(config)
        if phase is not expected:
            raise ValueError(
                f"cannot advance {phase.value}; next phase is {expected.value}"
            )
        values = {item.value: getattr(self, item.value) for item in _TRAINING_PHASES}
        values[phase.value] += 1
        return LongRunCursor(**values)


class WallClockBudget:
    """Monotonic, injectable wall-clock stop condition for graceful checkpoints."""

    def __init__(
        self,
        max_wall_seconds: float,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.max_wall_seconds = _finite_number(
            max_wall_seconds,
            "max_wall_seconds",
            minimum=math.nextafter(0.0, 1.0),
        )
        if not callable(clock):
            raise TypeError("clock must be callable")
        self._clock = clock
        self._started = float(clock())

    @property
    def elapsed_seconds(self) -> float:
        return max(0.0, float(self._clock()) - self._started)

    @property
    def remaining_seconds(self) -> float:
        return max(0.0, self.max_wall_seconds - self.elapsed_seconds)

    @property
    def expired(self) -> bool:
        return self.elapsed_seconds >= self.max_wall_seconds


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _canonical_json_line(value: Mapping[str, Any]) -> bytes:
    try:
        return (
            json.dumps(
                dict(value),
                ensure_ascii=True,
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n"
        ).encode("ascii")
    except (TypeError, ValueError) as error:
        raise ValueError("monitoring records must be finite JSON objects") from error


def atomic_write_json(path: str | Path, value: Mapping[str, Any]) -> Path:
    """Replace one JSON record only after its bytes and parent are durable."""

    if not isinstance(value, Mapping):
        raise TypeError("atomic JSON value must be a mapping")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(
            dict(value),
            ensure_ascii=True,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        ).encode("ascii")
        + b"\n"
    )
    temporary = target.with_name(f".{target.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}")
    try:
        with temporary.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
        _fsync_directory(target.parent)
    finally:
        temporary.unlink(missing_ok=True)
    return target


class AtomicJsonlLog:
    """Crash-consistent JSONL heartbeat using write-fsync-rename per event.

    Long runs emit heartbeat records infrequently.  Replacing the small log is
    deliberately preferred over accepting a torn trailing line after a host
    failure; readers therefore never need a repair mode.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def records(self) -> tuple[dict[str, Any], ...]:
        if not self.path.exists():
            return ()
        try:
            payload = self.path.read_bytes()
        except OSError as error:
            raise RuntimeError(f"cannot read heartbeat log: {error}") from error
        if payload and not payload.endswith(b"\n"):
            raise RuntimeError("heartbeat log has a torn trailing record")
        result: list[dict[str, Any]] = []
        for index, line in enumerate(payload.splitlines(), start=1):
            try:
                value = json.loads(line.decode("ascii"))
            except (UnicodeError, json.JSONDecodeError) as error:
                raise RuntimeError(
                    f"heartbeat log record {index} is invalid"
                ) from error
            if not isinstance(value, dict):
                raise RuntimeError(f"heartbeat log record {index} is not an object")
            result.append(value)
        return tuple(result)

    def append(self, value: Mapping[str, Any]) -> None:
        if not isinstance(value, Mapping):
            raise TypeError("heartbeat record must be a mapping")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        previous = self.path.read_bytes() if self.path.exists() else b""
        if previous and not previous.endswith(b"\n"):
            raise RuntimeError("heartbeat log has a torn trailing record")
        line = _canonical_json_line(value)
        temporary = self.path.with_name(
            f".{self.path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}"
        )
        try:
            with temporary.open("xb") as stream:
                stream.write(previous)
                stream.write(line)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.path)
            _fsync_directory(self.path.parent)
        finally:
            temporary.unlink(missing_ok=True)


__all__ = [
    "AtomicJsonlLog",
    "LongRunCorpusConfig",
    "LongRunCursor",
    "LongRunPhase",
    "LongRunRuntimeConfig",
    "LongRunTrainingConfig",
    "STAGE0_LONG_RUN_SCHEMA_ID",
    "STAGE0_LONG_RUN_SCHEMA_VERSION",
    "Stage0LongRunConfig",
    "WallClockBudget",
    "atomic_write_json",
    "load_long_run_config",
]
