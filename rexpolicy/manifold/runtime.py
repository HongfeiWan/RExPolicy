"""Frozen success-manifold bundle and bounded rollout conditioning runtime."""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from rexpolicy.flywheel.checkpoint import file_sha256
from rexpolicy.flywheel.groot_policy import CachedCondition, GrootFlowDitPolicy
from rexpolicy.tasking.canonical import canonical_fingerprint

from .config import SuccessManifoldConfig
from .encoder import FutureTrajectoryEncoder
from .memory import LatentAdmission, LatentMemory
from .selector import SuccessModeSelector

_BUNDLE_SCHEMA_VERSION = 1
_RUNTIME_STATE_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class ConditionedLatentBatch:
    """Selector outputs and the exact cached conditions sent to Flow-DiT."""

    conditions: tuple[CachedCondition, ...]
    latents: Any
    condition_tokens: Any
    sources: tuple[str, ...]


def _bundle_payload(
    *,
    config: SuccessManifoldConfig,
    encoder: FutureTrajectoryEncoder,
    selector: SuccessModeSelector,
) -> dict[str, Any]:
    if not config.enabled:
        raise ValueError("Cannot save a disabled success-manifold bundle")
    return {
        "schema_version": _BUNDLE_SCHEMA_VERSION,
        "config": config.to_record(),
        "config_sha256": canonical_fingerprint(config.to_record()),
        "encoder": {
            key: value.detach().cpu().contiguous()
            for key, value in encoder.state_dict().items()
        },
        "selector": {
            key: value.detach().cpu().contiguous()
            for key, value in selector.state_dict().items()
        },
    }


def save_success_manifold_bundle(
    path: Path,
    *,
    config: SuccessManifoldConfig,
    encoder: FutureTrajectoryEncoder,
    selector: SuccessModeSelector,
) -> str:
    """Atomically save inference weights and return the file SHA-256."""
    import torch

    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.partial-", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as file:
            torch.save(
                _bundle_payload(config=config, encoder=encoder, selector=selector),
                file,
            )
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return file_sha256(path)


class SuccessManifoldRuntime:
    """Use frozen selector/encoder heads while checkpointing only latent memory."""

    def __init__(
        self,
        *,
        config: SuccessManifoldConfig,
        encoder: FutureTrajectoryEncoder,
        selector: SuccessModeSelector,
        bundle_sha256: str,
        device: Any,
    ) -> None:
        import torch

        if not config.enabled:
            raise ValueError("SuccessManifoldRuntime requires enabled config")
        if (
            not isinstance(bundle_sha256, str)
            or len(bundle_sha256) != 64
            or any(character not in "0123456789abcdef" for character in bundle_sha256)
        ):
            raise ValueError("bundle_sha256 must be a lowercase SHA-256")
        self.torch = torch
        self.config = config
        self.device = torch.device(device)
        self.bundle_sha256 = bundle_sha256
        self.encoder = encoder.to(self.device).eval().requires_grad_(False)
        self.selector = selector.to(self.device).eval().requires_grad_(False)
        self.memory = LatentMemory(
            latent_dim=config.latent_dim,
            capacity=config.memory_capacity,
            novelty_threshold=config.novelty_threshold,
            encoder_fingerprint=bundle_sha256,
        )

    @classmethod
    def load(
        cls,
        path: Path,
        *,
        expected_sha256: str,
        device: Any,
    ) -> SuccessManifoldRuntime:
        """Load only a checksum-pinned, weights-only deployment bundle."""
        import torch

        path = Path(path).expanduser().resolve()
        actual_sha256 = file_sha256(path)
        if actual_sha256 != expected_sha256:
            raise ValueError(
                "Success-manifold bundle checksum mismatch: "
                f"expected={expected_sha256}, actual={actual_sha256}"
            )
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(payload, dict) or set(payload) != {
            "schema_version",
            "config",
            "config_sha256",
            "encoder",
            "selector",
        }:
            raise ValueError("Success-manifold bundle has an invalid inventory")
        if payload["schema_version"] != _BUNDLE_SCHEMA_VERSION:
            raise ValueError("Unsupported success-manifold bundle schema")
        config = SuccessManifoldConfig.from_mapping(payload["config"])
        if canonical_fingerprint(config.to_record()) != payload["config_sha256"]:
            raise ValueError("Success-manifold config fingerprint mismatch")
        encoder = FutureTrajectoryEncoder(config)
        selector = SuccessModeSelector(config)
        encoder.load_state_dict(payload["encoder"], strict=True)
        selector.load_state_dict(payload["selector"], strict=True)
        return cls(
            config=config,
            encoder=encoder,
            selector=selector,
            bundle_sha256=actual_sha256,
            device=device,
        )

    def condition(
        self,
        *,
        policy: GrootFlowDitPolicy,
        conditions: Sequence[CachedCondition],
        memory_candidates: int = 0,
        deterministic_selector: bool = False,
        selector_temperature: float = 1.0,
        generator: Any | None = None,
    ) -> ConditionedLatentBatch:
        """Sample reachable modes, optionally reserving bounded memory slots."""
        if not conditions:
            raise ValueError("At least one cached condition is required")
        if type(memory_candidates) is not int or memory_candidates < 0:
            raise ValueError("memory_candidates must be a non-negative integer")
        states = self.torch.stack(
            [self.torch.as_tensor(condition.state) for condition in conditions]
        ).to(device=self.device, dtype=self.torch.float32)
        states = states.flatten(start_dim=1)
        if int(states.shape[1]) != self.config.state_dim:
            raise ValueError(
                "Selector state dimension does not match cached GR00T state: "
                f"{int(states.shape[1])} != {self.config.state_dim}"
            )
        with self.torch.inference_mode():
            latents = self.selector.sample(
                states,
                deterministic=deterministic_selector,
                temperature=selector_temperature,
                generator=generator,
            )
            sources = ["selector"] * len(conditions)
            memory_entries = self.memory.sample(
                min(memory_candidates, len(conditions))
            )
            for index, entry in enumerate(memory_entries):
                latents[index].copy_(
                    self.torch.tensor(
                        entry.latent,
                        dtype=latents.dtype,
                        device=latents.device,
                    )
                )
                sources[index] = "latent_memory"
            tokens = self.encoder.condition_head(latents)
        if int(tokens.shape[1]) != policy.success_token_dimension:
            raise ValueError(
                "Manifold condition_dim does not match GR00T backbone width: "
                f"{int(tokens.shape[1])} != {policy.success_token_dimension}"
            )
        if not bool(self.torch.isfinite(latents).all()) or not bool(
            self.torch.isfinite(tokens).all()
        ):
            raise FloatingPointError("Selector produced non-finite manifold values")
        conditioned = policy.attach_success_latent_tokens(
            list(conditions), tokens, latents=latents
        )
        return ConditionedLatentBatch(
            conditions=tuple(conditioned),
            latents=latents.detach().cpu(),
            condition_tokens=tokens.detach().cpu(),
            sources=tuple(sources),
        )

    def record_successes(
        self,
        *,
        sample_ids: Sequence[str],
        latents: Any,
        metadata: Sequence[Mapping[str, Any] | None] | None = None,
    ) -> tuple[LatentAdmission, ...]:
        """Expand memory from simulator-verified successes only."""
        values = self.torch.as_tensor(latents).detach().cpu().float()
        if values.ndim != 2 or values.shape != (
            len(sample_ids),
            self.config.latent_dim,
        ):
            raise ValueError("latents must have shape [sample_ids, latent_dim]")
        if not bool(self.torch.isfinite(values).all()):
            raise ValueError("Success latents must be finite")
        if metadata is None:
            metadata = [None] * len(sample_ids)
        if len(metadata) != len(sample_ids):
            raise ValueError("metadata length must match sample_ids")
        return tuple(
            self.memory.add(
                sample_id,
                values[index].tolist(),
                metadata=metadata[index],
            )
            for index, sample_id in enumerate(sample_ids)
        )

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema_version": _RUNTIME_STATE_SCHEMA_VERSION,
            "bundle_sha256": self.bundle_sha256,
            "config_sha256": canonical_fingerprint(self.config.to_record()),
            "memory": self.memory.state_dict(),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if not isinstance(state, Mapping) or set(state) != {
            "schema_version",
            "bundle_sha256",
            "config_sha256",
            "memory",
        }:
            raise ValueError("Success-manifold runtime state inventory is invalid")
        if state["schema_version"] != _RUNTIME_STATE_SCHEMA_VERSION:
            raise ValueError("Unsupported success-manifold runtime state schema")
        if state["bundle_sha256"] != self.bundle_sha256:
            raise ValueError("Success-manifold runtime bundle changed on resume")
        expected_config = canonical_fingerprint(self.config.to_record())
        if state["config_sha256"] != expected_config:
            raise ValueError("Success-manifold runtime config changed on resume")
        self.memory.load_state_dict(state["memory"])
