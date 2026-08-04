"""Stage-wise DDP training for the success encoder and mode selector."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch

from rexpolicy.flywheel.distributed import DistributedContext
from rexpolicy.tasking.canonical import canonical_fingerprint

from .config import SuccessManifoldConfig
from .encoder import FutureTrajectoryEncoder
from .losses import success_manifold_losses, success_selector_losses
from .runtime import save_success_manifold_bundle
from .selector import SuccessModeSelector


@dataclass(frozen=True)
class ManifoldUpdateMetrics:
    stage: str
    loss: float
    gradient_norm: float
    optimizer_step: int
    reconstruction: float = 0.0
    info_nce: float = 0.0
    variance: float = 0.0
    covariance: float = 0.0
    negative_log_likelihood: float = 0.0
    contrastive_alignment: float = 0.0


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


class SuccessManifoldDdpTrainer:
    """Train representation first, then selector, with resumable optimizers."""

    STATE_SCHEMA_VERSION = 1

    def __init__(
        self,
        *,
        config: SuccessManifoldConfig,
        context: DistributedContext,
        encoder_learning_rate: float = 1.0e-4,
        selector_learning_rate: float = 1.0e-4,
        weight_decay: float = 1.0e-4,
        gradient_clip_norm: float = 1.0,
        encoder: FutureTrajectoryEncoder | None = None,
        selector: SuccessModeSelector | None = None,
    ) -> None:
        if not config.enabled:
            raise ValueError("Manifold trainer requires enabled config")
        if min(encoder_learning_rate, selector_learning_rate) <= 0.0:
            raise ValueError("Manifold learning rates must be positive")
        if weight_decay < 0.0 or gradient_clip_norm <= 0.0:
            raise ValueError("Invalid manifold optimizer configuration")
        self.config = config
        self.context = context
        self.gradient_clip_norm = float(gradient_clip_norm)
        self.encoder = (encoder or FutureTrajectoryEncoder(config)).to(
            context.device
        )
        self.selector = (selector or SuccessModeSelector(config)).to(
            context.device
        )
        self.encoder_module: Any = self.encoder
        self.selector_module: Any = self.selector
        if context.initialized:
            from torch.nn.parallel import DistributedDataParallel

            kwargs: dict[str, Any] = {"broadcast_buffers": False}
            if context.device.type == "cuda":
                kwargs.update(
                    device_ids=[context.local_rank],
                    output_device=context.local_rank,
                )
            self.encoder_module = DistributedDataParallel(
                self.encoder,
                find_unused_parameters=True,
                **kwargs,
            )
            self.selector_module = DistributedDataParallel(
                self.selector,
                find_unused_parameters=False,
                **kwargs,
            )
        optimizer_kwargs = {"weight_decay": float(weight_decay)}
        if context.device.type == "cuda":
            optimizer_kwargs["fused"] = True
        self.encoder_optimizer = torch.optim.AdamW(
            self.encoder.parameters(),
            lr=float(encoder_learning_rate),
            **optimizer_kwargs,
        )
        self.selector_optimizer = torch.optim.AdamW(
            self.selector.parameters(),
            lr=float(selector_learning_rate),
            **optimizer_kwargs,
        )
        self.encoder_step_count = 0
        self.selector_step_count = 0

    def _finish_step(
        self,
        *,
        loss: torch.Tensor,
        parameters: Sequence[torch.nn.Parameter],
        optimizer: torch.optim.Optimizer,
    ) -> float:
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("Success-manifold loss is non-finite")
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            parameters,
            self.gradient_clip_norm,
            error_if_nonfinite=True,
        )
        value = float(gradient_norm.item())
        if value <= 0.0:
            raise FloatingPointError(
                "Success-manifold gradient norm must be finite and positive"
            )
        optimizer.step()
        return value

    def train_encoder_batch(
        self,
        *,
        future_states: torch.Tensor,
        future_actions: torch.Tensor,
        trajectory_ids: Sequence[Any] | torch.Tensor,
        visual_features: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> ManifoldUpdateMetrics:
        """Run one future reconstruction/contrastive/diversity update."""
        self.encoder_module.train()
        future_states = future_states.to(self.context.device)
        future_actions = future_actions.to(self.context.device)
        if visual_features is not None:
            visual_features = visual_features.to(self.context.device)
        if padding_mask is None:
            padding_mask = torch.zeros(
                future_states.shape[:2],
                dtype=torch.bool,
                device=self.context.device,
            )
        else:
            padding_mask = padding_mask.to(
                self.context.device, dtype=torch.bool
            )
        reconstruction_mask = self.encoder.sample_reconstruction_mask(
            padding_mask,
            generator=generator,
        )
        self.encoder_optimizer.zero_grad(set_to_none=True)
        output = self.encoder_module(
            future_states,
            future_actions,
            visual_features=visual_features,
            padding_mask=padding_mask,
            reconstruction_mask=reconstruction_mask,
        )
        losses = success_manifold_losses(
            output,
            future_states,
            future_actions,
            trajectory_ids,
            config=self.config,
            visual_features=visual_features,
            gather_distributed=self.context.initialized,
        )
        gradient_norm = self._finish_step(
            loss=losses.total,
            parameters=list(self.encoder.parameters()),
            optimizer=self.encoder_optimizer,
        )
        self.encoder_step_count += 1
        return ManifoldUpdateMetrics(
            stage="encoder",
            loss=float(losses.total.detach()),
            gradient_norm=gradient_norm,
            optimizer_step=self.encoder_step_count,
            reconstruction=float(losses.reconstruction.detach()),
            info_nce=float(losses.info_nce.detach()),
            variance=float(losses.variance.detach()),
            covariance=float(losses.covariance.detach()),
        )

    def train_selector_batch(
        self,
        *,
        current_states: torch.Tensor,
        target_latents: torch.Tensor,
        contrastive_weight: float = 1.0,
    ) -> ManifoldUpdateMetrics:
        """Run one current-state -> future-success-latent update."""
        self.selector_module.train()
        current_states = current_states.to(self.context.device)
        target_latents = target_latents.to(self.context.device).detach()
        self.selector_optimizer.zero_grad(set_to_none=True)
        distribution = self.selector_module(current_states)
        losses = success_selector_losses(
            distribution,
            target_latents,
            contrastive_weight=contrastive_weight,
        )
        gradient_norm = self._finish_step(
            loss=losses.total,
            parameters=list(self.selector.parameters()),
            optimizer=self.selector_optimizer,
        )
        self.selector_step_count += 1
        return ManifoldUpdateMetrics(
            stage="selector",
            loss=float(losses.total.detach()),
            gradient_norm=gradient_norm,
            optimizer_step=self.selector_step_count,
            negative_log_likelihood=float(
                losses.negative_log_likelihood.detach()
            ),
            contrastive_alignment=float(
                losses.contrastive_alignment.detach()
            ),
        )

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.STATE_SCHEMA_VERSION,
            "config_sha256": canonical_fingerprint(self.config.to_record()),
            "encoder": _cpu(self.encoder.state_dict()),
            "selector": _cpu(self.selector.state_dict()),
            "encoder_optimizer": _cpu(self.encoder_optimizer.state_dict()),
            "selector_optimizer": _cpu(self.selector_optimizer.state_dict()),
            "encoder_step_count": self.encoder_step_count,
            "selector_step_count": self.selector_step_count,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        expected = {
            "schema_version",
            "config_sha256",
            "encoder",
            "selector",
            "encoder_optimizer",
            "selector_optimizer",
            "encoder_step_count",
            "selector_step_count",
        }
        if not isinstance(state, dict) or set(state) != expected:
            raise ValueError("Manifold trainer state inventory is invalid")
        if state["schema_version"] != self.STATE_SCHEMA_VERSION:
            raise ValueError("Unsupported manifold trainer state schema")
        if state["config_sha256"] != canonical_fingerprint(
            self.config.to_record()
        ):
            raise ValueError("Manifold trainer config changed on resume")
        self.encoder.load_state_dict(state["encoder"], strict=True)
        self.selector.load_state_dict(state["selector"], strict=True)
        self.encoder_optimizer.load_state_dict(state["encoder_optimizer"])
        self.selector_optimizer.load_state_dict(state["selector_optimizer"])
        self.encoder_step_count = int(state["encoder_step_count"])
        self.selector_step_count = int(state["selector_step_count"])
        if min(self.encoder_step_count, self.selector_step_count) < 0:
            raise ValueError("Manifold optimizer steps cannot be negative")

    def save_deployment_bundle(self, path: Any) -> str:
        """Export frozen encoder/selector weights for the online runner."""
        return save_success_manifold_bundle(
            path,
            config=self.config,
            encoder=self.encoder,
            selector=self.selector,
        )
