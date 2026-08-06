"""Pre-registered state-only no-z training contract for Grasp-Lift."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import torch

from rexpolicy.stage0.data.grasp_lift_training import (
    GRASP_LIFT_MODEL_ACTION_VIEW_SHA256,
    GraspLiftTrainingData,
)
from rexpolicy.stage0.envs.grasp_lift_action import (
    GRASP_LIFT_EFFECTIVE_ACTION_MASK,
)
from rexpolicy.stage0.envs.grasp_lift_state_view import (
    DEFAULT_GRASP_LIFT_STATE_VIEW,
    GRASP_LIFT_FIXED_PHASE_ONE_HOT,
)
from rexpolicy.stage0.normalization import Stage0Normalization
from rexpolicy.stage0.runtime import Stage0FlowPolicy
from rexpolicy.stage0.seed import derive_stage0_seed
from rexpolicy.stage0.trainers.batch import Stage0WindowBatch
from rexpolicy.stage0.types import canonical_fingerprint

GRASP_LIFT_NO_Z_CONFIG_SCHEMA_ID = "rexpolicy/stage0-grasp-lift-no-z-config/v1"
GRASP_LIFT_NO_Z_CONFIG_SCHEMA_VERSION = 1
GRASP_LIFT_NO_Z_TRAINING_SEEDS = (31001, 31002, 31003)
GRASP_LIFT_NO_Z_SAMPLER_ID = (
    "rexpolicy/global-no-replacement-cross-epoch-step-indexed/v1"
)
GRASP_LIFT_NO_Z_NORMALIZED_BATCH_SCHEMA_ID = (
    "rexpolicy/stage0-grasp-lift-normalized-model-batch/v1"
)


def _exact_mapping(value: Any, keys: set[str], context: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != keys:
        raise ValueError(f"{context} fields changed")
    return dict(value)


def _positive_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _non_negative_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _positive_finite(value: Any, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) <= 0.0
    ):
        raise ValueError(f"{name} must be finite and positive")
    return float(value)


@dataclass(frozen=True)
class GraspLiftNoZConfig:
    """One immutable three-seed, equal-budget no-z training protocol."""

    training_seeds: tuple[int, ...]
    cpu_threads: int
    interop_threads: int
    require_cuda: bool
    dtype: str
    deterministic_algorithms: bool
    cublas_workspace_config: str
    nvidia_tf32_override: str
    torch_allow_tf32_cublas_override: str
    cudnn_benchmark: bool
    allow_tf32: bool
    torch_compile: bool
    state_dim: int
    latent_dim: int
    action_dim: int
    action_horizon: int
    model_dim: int
    transformer_layers: int
    attention_heads: int
    feedforward_dim: int
    dropout: float
    global_batch_size: int
    total_steps: int
    checkpoint_every_steps: int
    log_every_steps: int
    learning_rate: float
    weight_decay: float
    gradient_clip_norm: float
    cuda_fused_adamw: bool
    expected_train_trajectories: int
    expected_train_windows: int
    sampler_id: str
    schema_version: int = GRASP_LIFT_NO_Z_CONFIG_SCHEMA_VERSION
    schema_id: str = GRASP_LIFT_NO_Z_CONFIG_SCHEMA_ID

    def __post_init__(self) -> None:
        if self.schema_id != GRASP_LIFT_NO_Z_CONFIG_SCHEMA_ID:
            raise ValueError("Grasp-Lift no-z config schema_id changed")
        if (
            type(self.schema_version) is not int
            or self.schema_version != GRASP_LIFT_NO_Z_CONFIG_SCHEMA_VERSION
        ):
            raise ValueError("Grasp-Lift no-z config schema_version changed")
        if self.training_seeds != GRASP_LIFT_NO_Z_TRAINING_SEEDS:
            raise ValueError("Grasp-Lift no-z training seeds changed")
        if len(set(self.training_seeds)) != 3:
            raise ValueError("Grasp-Lift no-z requires three unique seeds")
        for name in (
            "cpu_threads",
            "interop_threads",
            "state_dim",
            "latent_dim",
            "action_dim",
            "action_horizon",
            "model_dim",
            "transformer_layers",
            "attention_heads",
            "feedforward_dim",
            "global_batch_size",
            "total_steps",
            "checkpoint_every_steps",
            "log_every_steps",
            "expected_train_trajectories",
            "expected_train_windows",
        ):
            _positive_integer(getattr(self, name), name)
        for name in (
            "require_cuda",
            "deterministic_algorithms",
            "cudnn_benchmark",
            "allow_tf32",
            "torch_compile",
            "cuda_fused_adamw",
        ):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be boolean")
        for name in (
            "learning_rate",
            "weight_decay",
            "gradient_clip_norm",
        ):
            _positive_finite(getattr(self, name), name)
        if (
            isinstance(self.dropout, bool)
            or not isinstance(self.dropout, (int, float))
            or not math.isfinite(float(self.dropout))
            or not 0.0 <= float(self.dropout) < 1.0
        ):
            raise ValueError("dropout must be finite in [0, 1)")
        expected = {
            "allow_tf32": False,
            "attention_heads": 4,
            "cpu_threads": 16,
            "cuda_fused_adamw": True,
            "cublas_workspace_config": ":4096:8",
            "cudnn_benchmark": False,
            "deterministic_algorithms": True,
            "dropout": 0.0,
            "dtype": "float32",
            "expected_train_trajectories": 24,
            "expected_train_windows": 1030,
            "feedforward_dim": 256,
            "global_batch_size": 2048,
            "gradient_clip_norm": 1.0,
            "interop_threads": 4,
            "latent_dim": 32,
            "learning_rate": 3.0e-4,
            "log_every_steps": 50,
            "checkpoint_every_steps": 500,
            "model_dim": 128,
            "nvidia_tf32_override": "0",
            "action_dim": 19,
            "action_horizon": 8,
            "require_cuda": True,
            "sampler_id": GRASP_LIFT_NO_Z_SAMPLER_ID,
            "state_dim": 79,
            "torch_compile": False,
            "torch_allow_tf32_cublas_override": "0",
            "total_steps": 10000,
            "transformer_layers": 3,
            "weight_decay": 1.0e-4,
        }
        for name, value in expected.items():
            if getattr(self, name) != value:
                raise ValueError(f"pre-registered Grasp-Lift no-z {name} changed")
        if self.total_steps % self.checkpoint_every_steps:
            raise ValueError("total_steps must be divisible by checkpoint interval")
        if self.checkpoint_every_steps % self.log_every_steps:
            raise ValueError("checkpoint interval must be divisible by log interval")

    def to_record(self) -> dict[str, Any]:
        return {
            "data": {
                "expected_train_trajectories": self.expected_train_trajectories,
                "expected_train_windows": self.expected_train_windows,
                "sampler_id": self.sampler_id,
            },
            "model": {
                "action_dim": self.action_dim,
                "action_horizon": self.action_horizon,
                "attention_heads": self.attention_heads,
                "dropout": self.dropout,
                "feedforward_dim": self.feedforward_dim,
                "latent_dim": self.latent_dim,
                "model_dim": self.model_dim,
                "state_dim": self.state_dim,
                "transformer_layers": self.transformer_layers,
            },
            "optimization": {
                "checkpoint_every_steps": self.checkpoint_every_steps,
                "cuda_fused_adamw": self.cuda_fused_adamw,
                "global_batch_size": self.global_batch_size,
                "gradient_clip_norm": self.gradient_clip_norm,
                "learning_rate": self.learning_rate,
                "log_every_steps": self.log_every_steps,
                "total_steps": self.total_steps,
                "weight_decay": self.weight_decay,
            },
            "runtime": {
                "allow_tf32": self.allow_tf32,
                "cpu_threads": self.cpu_threads,
                "cublas_workspace_config": self.cublas_workspace_config,
                "cudnn_benchmark": self.cudnn_benchmark,
                "deterministic_algorithms": self.deterministic_algorithms,
                "dtype": self.dtype,
                "interop_threads": self.interop_threads,
                "nvidia_tf32_override": self.nvidia_tf32_override,
                "require_cuda": self.require_cuda,
                "torch_compile": self.torch_compile,
                "torch_allow_tf32_cublas_override": (
                    self.torch_allow_tf32_cublas_override
                ),
            },
            "schema_id": self.schema_id,
            "schema_version": self.schema_version,
            "training_seeds": list(self.training_seeds),
        }

    @property
    def sha256(self) -> str:
        return canonical_fingerprint(self.to_record())

    def run_sha256(self, training_seed: int, *, world_size: int) -> str:
        self.require_training_seed(training_seed)
        _positive_integer(world_size, "world_size")
        if self.global_batch_size % world_size:
            raise ValueError("global_batch_size must be divisible by world_size")
        return canonical_fingerprint(
            {
                "config_sha256": self.sha256,
                "schema_id": "rexpolicy/stage0-grasp-lift-no-z-run/v1",
                "training_seed": training_seed,
                "world_size": world_size,
            }
        )

    def require_training_seed(self, training_seed: int) -> None:
        if training_seed not in self.training_seeds:
            raise ValueError(
                "training_seed is not one of the three pre-registered seeds"
            )

    @classmethod
    def from_record(cls, value: Mapping[str, Any]) -> GraspLiftNoZConfig:
        root = _exact_mapping(
            value,
            {
                "data",
                "model",
                "optimization",
                "runtime",
                "schema_id",
                "schema_version",
                "training_seeds",
            },
            "Grasp-Lift no-z config",
        )
        data = _exact_mapping(
            root["data"],
            {
                "expected_train_trajectories",
                "expected_train_windows",
                "sampler_id",
            },
            "Grasp-Lift no-z data config",
        )
        model = _exact_mapping(
            root["model"],
            {
                "action_dim",
                "action_horizon",
                "attention_heads",
                "dropout",
                "feedforward_dim",
                "latent_dim",
                "model_dim",
                "state_dim",
                "transformer_layers",
            },
            "Grasp-Lift no-z model config",
        )
        optimization = _exact_mapping(
            root["optimization"],
            {
                "checkpoint_every_steps",
                "cuda_fused_adamw",
                "global_batch_size",
                "gradient_clip_norm",
                "learning_rate",
                "log_every_steps",
                "total_steps",
                "weight_decay",
            },
            "Grasp-Lift no-z optimization config",
        )
        runtime = _exact_mapping(
            root["runtime"],
            {
                "allow_tf32",
                "cpu_threads",
                "cublas_workspace_config",
                "cudnn_benchmark",
                "deterministic_algorithms",
                "dtype",
                "interop_threads",
                "nvidia_tf32_override",
                "require_cuda",
                "torch_compile",
                "torch_allow_tf32_cublas_override",
            },
            "Grasp-Lift no-z runtime config",
        )
        seeds = root["training_seeds"]
        if not isinstance(seeds, list):
            raise TypeError("training_seeds must be a JSON list")
        return cls(
            **data,
            **model,
            **optimization,
            **runtime,
            training_seeds=tuple(seeds),
            schema_id=root["schema_id"],
            schema_version=root["schema_version"],
        )


def load_grasp_lift_no_z_config(path: str | Path) -> GraspLiftNoZConfig:
    requested = Path(path).expanduser()
    if requested.is_symlink() or not requested.is_file():
        raise ValueError("Grasp-Lift no-z config is missing or a symlink")

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    try:
        value = json.loads(
            requested.read_text(encoding="utf-8"),
            object_pairs_hook=reject_duplicates,
            parse_constant=lambda constant: (_ for _ in ()).throw(
                ValueError(f"invalid JSON constant: {constant}")
            ),
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"cannot read Grasp-Lift no-z config: {error}") from error
    if not isinstance(value, Mapping):
        raise TypeError("Grasp-Lift no-z config must contain an object")
    return GraspLiftNoZConfig.from_record(value)


def freeze_grasp_lift_no_z_latent_branch(policy: Stage0FlowPolicy) -> dict[str, Any]:
    """Freeze the branch omitted by ``success_latent=None`` before optimization."""
    if not isinstance(policy, Stage0FlowPolicy):
        raise TypeError("policy must be Stage0FlowPolicy")
    for parameter in policy.condition_encoder.latent_projector.parameters():
        parameter.requires_grad_(False)
    policy.condition_encoder.latent_token_type.requires_grad_(False)
    frozen = tuple(
        name
        for name, parameter in policy.named_parameters()
        if not parameter.requires_grad
    )
    expected = tuple(
        name
        for name, _ in policy.named_parameters()
        if name == "condition_encoder.latent_token_type"
        or name.startswith("condition_encoder.latent_projector.")
    )
    if frozen != expected or not frozen:
        raise RuntimeError("Grasp-Lift no-z frozen parameter set changed")
    trainable_parameters = sum(
        parameter.numel()
        for parameter in policy.parameters()
        if parameter.requires_grad
    )
    total_parameters = sum(parameter.numel() for parameter in policy.parameters())
    return {
        "frozen_parameter_names": list(frozen),
        "schema_id": "rexpolicy/stage0-grasp-lift-no-z-freeze/v1",
        "total_parameters": total_parameters,
        "trainable_parameters": trainable_parameters,
    }


def build_grasp_lift_no_z_policy(
    config: GraspLiftNoZConfig,
    *,
    device: torch.device | str,
) -> tuple[Stage0FlowPolicy, dict[str, Any]]:
    if not isinstance(config, GraspLiftNoZConfig):
        raise TypeError("config must be GraspLiftNoZConfig")
    policy = Stage0FlowPolicy(
        state_dim=config.state_dim,
        latent_dim=config.latent_dim,
        action_dim=config.action_dim,
        action_horizon=config.action_horizon,
        model_dim=config.model_dim,
        transformer_layers=config.transformer_layers,
        attention_heads=config.attention_heads,
        feedforward_dim=config.feedforward_dim,
        dropout=config.dropout,
    )
    freeze_record = freeze_grasp_lift_no_z_latent_branch(policy)
    return policy.to(device=device, dtype=torch.float32), freeze_record


def validate_grasp_lift_no_z_training_data(
    data: GraspLiftTrainingData,
    config: GraspLiftNoZConfig,
) -> None:
    if not isinstance(data, GraspLiftTrainingData):
        raise TypeError("data must be GraspLiftTrainingData")
    if not isinstance(config, GraspLiftNoZConfig):
        raise TypeError("config must be GraspLiftNoZConfig")
    if len(data.train_trajectories) != config.expected_train_trajectories:
        raise ValueError("Grasp-Lift no-z train trajectory count changed")
    if len(data.window_splits.train) != config.expected_train_windows:
        raise ValueError("Grasp-Lift no-z train window count changed")
    if data.window_splits.validation or data.window_splits.test:
        raise ValueError("Grasp-Lift no-z training materialized held-out windows")
    expected_mask = torch.tensor(GRASP_LIFT_EFFECTIVE_ACTION_MASK, dtype=torch.bool)
    if not torch.equal(data.normalization.effective_action_mask, expected_mask):
        raise ValueError("Grasp-Lift no-z effective action mask changed")


def normalized_grasp_lift_corpus_sha256(
    source_corpus_sha256: str,
    normalization: Stage0Normalization,
) -> str:
    if (
        not isinstance(source_corpus_sha256, str)
        or len(source_corpus_sha256) != 64
        or any(
            character not in "0123456789abcdef" for character in source_corpus_sha256
        )
    ):
        raise ValueError("source_corpus_sha256 must be a SHA-256 digest")
    if not isinstance(normalization, Stage0Normalization):
        raise TypeError("normalization must be Stage0Normalization")
    return canonical_fingerprint(
        {
            "model_action_view_sha256": GRASP_LIFT_MODEL_ACTION_VIEW_SHA256,
            "normalization_sha256": normalization.sha256,
            "schema_id": GRASP_LIFT_NO_Z_NORMALIZED_BATCH_SCHEMA_ID,
            "source_corpus_sha256": source_corpus_sha256,
            "state_view_sha256": DEFAULT_GRASP_LIFT_STATE_VIEW.sha256,
        }
    )


def normalize_grasp_lift_no_z_batch_once(
    batch: Stage0WindowBatch,
    normalization: Stage0Normalization,
    *,
    source_corpus_sha256: str,
) -> Stage0WindowBatch:
    """Normalize an already-hybrid batch without applying XYZ delta again."""
    if not isinstance(batch, Stage0WindowBatch):
        raise TypeError("batch must be Stage0WindowBatch")
    if not isinstance(normalization, Stage0Normalization):
        raise TypeError("normalization must be Stage0Normalization")
    if batch.corpus_sha256 != source_corpus_sha256:
        raise ValueError(
            "batch is not the raw model-view corpus or was normalized twice"
        )
    expected_mask = torch.tensor(
        GRASP_LIFT_EFFECTIVE_ACTION_MASK,
        dtype=torch.bool,
        device=normalization.effective_action_mask.device,
    )
    if not torch.equal(normalization.effective_action_mask, expected_mask):
        raise ValueError("normalization does not use the Grasp-Lift 13D mask")
    if bool(batch.action_chunks[..., 3:9].count_nonzero()) or bool(
        batch.future_actions[..., 3:9].count_nonzero()
    ):
        raise ValueError("batch contains physical or non-zero rotation actions")
    phase_slice = DEFAULT_GRASP_LIFT_STATE_VIEW.to_record()["phase_slice"]
    phase = batch.current_states[:, phase_slice[0] : phase_slice[1]]
    expected_phase = phase.new_tensor(GRASP_LIFT_FIXED_PHASE_ONE_HOT).expand_as(phase)
    if not torch.equal(phase, expected_phase):
        raise ValueError("batch does not use the fixed Grasp-Lift state view")
    return replace(
        batch,
        current_states=normalization.normalize_states(batch.current_states),
        action_chunks=normalization.normalize_actions(
            batch.action_chunks,
            valid_mask=batch.action_mask,
        ),
        future_actions=normalization.normalize_actions(
            batch.future_actions,
            valid_mask=batch.future_mask,
        ),
        future_states=normalization.normalize_states(
            batch.future_states,
            valid_mask=batch.future_mask,
        ),
        corpus_sha256=normalized_grasp_lift_corpus_sha256(
            source_corpus_sha256,
            normalization,
        ),
    )


@dataclass(frozen=True)
class GraspLiftNoZSamplerCursor:
    optimizer_step: int
    total_samples: int
    epoch: int
    offset: int

    def to_record(self) -> dict[str, int | str]:
        return {
            "epoch": self.epoch,
            "offset": self.offset,
            "optimizer_step": self.optimizer_step,
            "schema_id": "rexpolicy/stage0-grasp-lift-no-z-sampler-cursor/v1",
            "total_samples": self.total_samples,
        }


class GraspLiftNoZSampler:
    """Stateless global permutations addressed only by optimizer step."""

    def __init__(
        self,
        *,
        training_seed: int,
        dataset_size: int = 1030,
        global_batch_size: int = 2048,
    ) -> None:
        if training_seed not in GRASP_LIFT_NO_Z_TRAINING_SEEDS:
            raise ValueError("sampler seed is not pre-registered")
        self.training_seed = training_seed
        self.dataset_size = _positive_integer(dataset_size, "dataset_size")
        self.global_batch_size = _positive_integer(
            global_batch_size,
            "global_batch_size",
        )
        if self.dataset_size != 1030 or self.global_batch_size != 2048:
            raise ValueError("pre-registered Grasp-Lift sampler shape changed")
        self._cache: dict[int, torch.Tensor] = {}

    def _permutation(self, epoch: int) -> torch.Tensor:
        _non_negative_integer(epoch, "epoch")
        cached = self._cache.get(epoch)
        if cached is not None:
            return cached
        seed = derive_stage0_seed(
            self.training_seed,
            "sampler",
            "epoch",
            epoch,
            namespace="stage0-grasp-lift-no-z",
        )
        result = torch.randperm(
            self.dataset_size,
            generator=torch.Generator(device="cpu").manual_seed(seed),
        )
        self._cache = {epoch: result}
        return result

    def cursor(self, optimizer_step: int) -> GraspLiftNoZSamplerCursor:
        step = _non_negative_integer(optimizer_step, "optimizer_step")
        total = step * self.global_batch_size
        return GraspLiftNoZSamplerCursor(
            optimizer_step=step,
            total_samples=total,
            epoch=total // self.dataset_size,
            offset=total % self.dataset_size,
        )

    def global_indices(self, optimizer_step: int) -> torch.Tensor:
        cursor = self.cursor(optimizer_step)
        remaining = self.global_batch_size
        epoch = cursor.epoch
        offset = cursor.offset
        chunks: list[torch.Tensor] = []
        while remaining:
            permutation = self._permutation(epoch)
            count = min(remaining, self.dataset_size - offset)
            chunks.append(permutation[offset : offset + count])
            remaining -= count
            epoch += 1
            offset = 0
        return torch.cat(chunks).contiguous()

    def rank_indices(
        self,
        optimizer_step: int,
        *,
        rank: int,
        world_size: int,
    ) -> torch.Tensor:
        world = _positive_integer(world_size, "world_size")
        if isinstance(rank, bool) or not isinstance(rank, int) or not 0 <= rank < world:
            raise ValueError("rank must lie in [0, world_size)")
        if self.global_batch_size % world:
            raise ValueError("global_batch_size must be divisible by world_size")
        per_rank = self.global_batch_size // world
        start = rank * per_rank
        return self.global_indices(optimizer_step)[start : start + per_rank].clone()

    def to_record(self) -> dict[str, Any]:
        return {
            "dataset_size": self.dataset_size,
            "global_batch_size": self.global_batch_size,
            "sampler_id": GRASP_LIFT_NO_Z_SAMPLER_ID,
            "training_seed": self.training_seed,
        }


def select_grasp_lift_no_z_batch(
    full_batch: Stage0WindowBatch,
    indices: torch.Tensor,
) -> Stage0WindowBatch:
    """Index a pre-normalized full train batch without reopening any shard."""
    if not isinstance(full_batch, Stage0WindowBatch):
        raise TypeError("full_batch must be Stage0WindowBatch")
    if (
        not isinstance(indices, torch.Tensor)
        or indices.dtype is not torch.int64
        or indices.ndim != 1
        or indices.device.type != "cpu"
        or indices.numel() < 1
    ):
        raise ValueError("indices must be a non-empty one-dimensional CPU int64 tensor")
    if bool((indices < 0).any()) or bool((indices >= full_batch.batch_size).any()):
        raise ValueError("batch index is outside the full train batch")
    rows = tuple(int(index) for index in indices.tolist())
    device_indices = indices.to(device=full_batch.current_states.device)

    def select(value: torch.Tensor) -> torch.Tensor:
        return value.index_select(0, device_indices)

    return replace(
        full_batch,
        current_states=select(full_batch.current_states),
        action_chunks=select(full_batch.action_chunks),
        action_mask=select(full_batch.action_mask),
        future_actions=select(full_batch.future_actions),
        future_states=select(full_batch.future_states),
        future_mask=select(full_batch.future_mask),
        trajectory_ids=tuple(full_batch.trajectory_ids[index] for index in rows),
        reset_group_ids=tuple(full_batch.reset_group_ids[index] for index in rows),
        starts=tuple(full_batch.starts[index] for index in rows),
    )


__all__ = [
    "GRASP_LIFT_NO_Z_CONFIG_SCHEMA_ID",
    "GRASP_LIFT_NO_Z_CONFIG_SCHEMA_VERSION",
    "GRASP_LIFT_NO_Z_NORMALIZED_BATCH_SCHEMA_ID",
    "GRASP_LIFT_NO_Z_SAMPLER_ID",
    "GRASP_LIFT_NO_Z_TRAINING_SEEDS",
    "GraspLiftNoZConfig",
    "GraspLiftNoZSampler",
    "GraspLiftNoZSamplerCursor",
    "build_grasp_lift_no_z_policy",
    "freeze_grasp_lift_no_z_latent_branch",
    "load_grasp_lift_no_z_config",
    "normalize_grasp_lift_no_z_batch_once",
    "normalized_grasp_lift_corpus_sha256",
    "select_grasp_lift_no_z_batch",
    "validate_grasp_lift_no_z_training_data",
]
