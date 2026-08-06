"""Standalone DDP training and checkpoints for the v2.1 successor objective."""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch

from rexpolicy.flywheel.checkpoint import file_sha256
from rexpolicy.flywheel.distributed import DistributedContext
from rexpolicy.tasking.canonical import canonical_fingerprint

from .config import SuccessManifoldConfig
from .encoder import FutureTrajectoryEncoder
from .runtime import save_success_manifold_bundle
from .successor import (
    SuccessorModel,
    SuccessorTrainingConfig,
    successor_losses,
)


SUCCESSOR_CHECKPOINT_SCHEMA_VERSION = 1


def _cpu(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu(item) for item in value)
    return value


@dataclass(frozen=True)
class SuccessorUpdateMetrics:
    loss: float
    negative_log_likelihood: float
    cosine_alignment: float
    calibration: float
    gradient_norm: float
    optimizer_step: int
    local_batch_size: int
    global_batch_size: int
    samples_seen: int


class SuccessorDdpTrainer:
    """Train one selector-compatible successor with exact resume state."""

    STATE_SCHEMA_VERSION = 1

    def __init__(
        self,
        *,
        manifold_config: SuccessManifoldConfig,
        training_config: SuccessorTrainingConfig,
        context: DistributedContext,
        model: SuccessorModel | None = None,
    ) -> None:
        if not manifold_config.enabled:
            raise ValueError("Successor trainer requires an enabled manifold config")
        if not training_config.enabled:
            raise ValueError("Successor trainer requires an enabled training config")
        self.manifold_config = manifold_config
        self.training_config = training_config
        self.context = context
        self.model = (model or SuccessorModel(manifold_config)).to(context.device)
        self.module: Any = self.model
        if context.initialized:
            from torch.nn.parallel import DistributedDataParallel

            kwargs: dict[str, Any] = {"broadcast_buffers": False}
            if context.device.type == "cuda":
                kwargs.update(
                    device_ids=[context.local_rank],
                    output_device=context.local_rank,
                )
            self.module = DistributedDataParallel(
                self.model,
                find_unused_parameters=False,
                **kwargs,
            )
        optimizer_kwargs: dict[str, Any] = {
            "weight_decay": float(training_config.weight_decay)
        }
        if context.device.type == "cuda":
            optimizer_kwargs["fused"] = True
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=float(training_config.learning_rate),
            **optimizer_kwargs,
        )
        self.step_count = 0
        self.samples_seen = 0

    def _batch_sizes(self, local_batch_size: int) -> tuple[int, int]:
        if type(local_batch_size) is not int or local_batch_size < 1:
            raise ValueError("Successor batches must be non-empty")
        sizes = self.context.all_gather_objects(local_batch_size)
        if any(type(item) is not int or item < 1 for item in sizes):
            raise ValueError("Every successor DDP rank must have a non-empty batch")
        if len(set(sizes)) != 1:
            raise ValueError(
                "Successor DDP requires equal local batch sizes on every rank"
            )
        return local_batch_size, sum(sizes)

    def train_batch(
        self,
        *,
        current_states: torch.Tensor,
        target_latents: torch.Tensor,
    ) -> SuccessorUpdateMetrics:
        if (
            not isinstance(current_states, torch.Tensor)
            or current_states.ndim != 2
            or current_states.shape[1] != self.manifold_config.state_dim
            or not torch.is_floating_point(current_states)
        ):
            raise ValueError("current_states must be floating [batch, state_dim]")
        if (
            not isinstance(target_latents, torch.Tensor)
            or target_latents.ndim != 2
            or target_latents.shape
            != (current_states.shape[0], self.manifold_config.latent_dim)
            or not torch.is_floating_point(target_latents)
        ):
            raise ValueError("target_latents must be floating [batch, latent_dim]")
        local_batch_size, global_batch_size = self._batch_sizes(
            int(current_states.shape[0])
        )
        self.module.train()
        current_states = current_states.to(self.context.device)
        target_latents = target_latents.to(self.context.device).detach()
        self.optimizer.zero_grad(set_to_none=True)
        losses = successor_losses(
            self.module(current_states),
            target_latents,
            config=self.training_config,
        )
        if not bool(torch.isfinite(losses.total)):
            raise FloatingPointError("Successor loss is non-finite")
        losses.total.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            list(self.model.parameters()),
            float(self.training_config.gradient_clip_norm),
            error_if_nonfinite=True,
        )
        gradient_norm_value = float(gradient_norm.item())
        if gradient_norm_value <= 0.0:
            raise FloatingPointError(
                "Successor gradient norm must be finite and positive"
            )
        self.optimizer.step()
        self.step_count += 1
        self.samples_seen += global_batch_size
        return SuccessorUpdateMetrics(
            loss=float(losses.total.detach()),
            negative_log_likelihood=float(
                losses.negative_log_likelihood.detach()
            ),
            cosine_alignment=float(losses.cosine_alignment.detach()),
            calibration=float(losses.calibration.detach()),
            gradient_norm=gradient_norm_value,
            optimizer_step=self.step_count,
            local_batch_size=local_batch_size,
            global_batch_size=global_batch_size,
            samples_seen=self.samples_seen,
        )

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.STATE_SCHEMA_VERSION,
            "manifold_config_sha256": canonical_fingerprint(
                self.manifold_config.to_record()
            ),
            "training_config_sha256": canonical_fingerprint(
                self.training_config.to_record()
            ),
            "model": _cpu(self.model.state_dict()),
            "optimizer": _cpu(self.optimizer.state_dict()),
            "step_count": self.step_count,
            "samples_seen": self.samples_seen,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        expected = {
            "schema_version",
            "manifold_config_sha256",
            "training_config_sha256",
            "model",
            "optimizer",
            "step_count",
            "samples_seen",
        }
        if not isinstance(state, Mapping) or set(state) != expected:
            raise ValueError("Successor trainer state inventory is invalid")
        if state["schema_version"] != self.STATE_SCHEMA_VERSION:
            raise ValueError("Unsupported successor trainer state schema")
        if state["manifold_config_sha256"] != canonical_fingerprint(
            self.manifold_config.to_record()
        ):
            raise ValueError("Successor manifold config changed on resume")
        if state["training_config_sha256"] != canonical_fingerprint(
            self.training_config.to_record()
        ):
            raise ValueError("Successor training config changed on resume")
        step_count = state["step_count"]
        samples_seen = state["samples_seen"]
        if type(step_count) is not int or step_count < 0:
            raise ValueError("Successor step_count is invalid")
        if type(samples_seen) is not int or samples_seen < 0:
            raise ValueError("Successor samples_seen is invalid")
        self.model.load_state_dict(state["model"], strict=True)
        self.optimizer.load_state_dict(state["optimizer"])
        self.step_count = step_count
        self.samples_seen = samples_seen

    def checkpoint_payload(self) -> dict[str, Any]:
        return {
            "schema_version": SUCCESSOR_CHECKPOINT_SCHEMA_VERSION,
            "artifact_type": "rexpolicy_successor_training_checkpoint",
            "manifold_config": self.manifold_config.to_record(),
            "manifold_config_sha256": canonical_fingerprint(
                self.manifold_config.to_record()
            ),
            "training_config": self.training_config.to_record(),
            "training_config_sha256": canonical_fingerprint(
                self.training_config.to_record()
            ),
            "trainer_state": self.state_dict(),
        }

    def save_checkpoint(self, path: Path) -> str:
        """Atomically save training state and return its file SHA-256."""
        path = Path(path).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.partial-", dir=path.parent
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as file:
                torch.save(self.checkpoint_payload(), file)
                file.flush()
                os.fsync(file.fileno())
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
        return file_sha256(path)

    @classmethod
    def load_checkpoint(
        cls,
        path: Path,
        *,
        expected_sha256: str,
        context: DistributedContext,
    ) -> SuccessorDdpTrainer:
        path = Path(path).expanduser().resolve()
        actual_sha256 = file_sha256(path)
        if actual_sha256 != expected_sha256:
            raise ValueError(
                "Successor checkpoint checksum mismatch: "
                f"expected={expected_sha256}, actual={actual_sha256}"
            )
        payload = torch.load(path, map_location="cpu", weights_only=True)
        expected = {
            "schema_version",
            "artifact_type",
            "manifold_config",
            "manifold_config_sha256",
            "training_config",
            "training_config_sha256",
            "trainer_state",
        }
        if not isinstance(payload, Mapping) or set(payload) != expected:
            raise ValueError("Successor checkpoint inventory is invalid")
        if (
            payload["schema_version"] != SUCCESSOR_CHECKPOINT_SCHEMA_VERSION
            or payload["artifact_type"]
            != "rexpolicy_successor_training_checkpoint"
        ):
            raise ValueError("Unsupported successor checkpoint schema")
        manifold_config = SuccessManifoldConfig.from_mapping(
            payload["manifold_config"]
        )
        training_config = SuccessorTrainingConfig.from_mapping(
            payload["training_config"]
        )
        if payload["manifold_config_sha256"] != canonical_fingerprint(
            manifold_config.to_record()
        ):
            raise ValueError("Successor checkpoint manifold config is corrupt")
        if payload["training_config_sha256"] != canonical_fingerprint(
            training_config.to_record()
        ):
            raise ValueError("Successor checkpoint training config is corrupt")
        trainer = cls(
            manifold_config=manifold_config,
            training_config=training_config,
            context=context,
        )
        trainer.load_state_dict(payload["trainer_state"])
        return trainer

    def save_deployment_bundle(
        self,
        path: Path,
        *,
        encoder: FutureTrajectoryEncoder,
    ) -> str:
        """Write the existing v2 bundle schema with successor weights in selector."""
        return save_success_manifold_bundle(
            path,
            config=self.manifold_config,
            encoder=encoder,
            selector=self.model,
        )


__all__ = [
    "SUCCESSOR_CHECKPOINT_SCHEMA_VERSION",
    "SuccessorDdpTrainer",
    "SuccessorUpdateMetrics",
]
