"""Deterministic diagnostics and evidence conversion for no-z validation."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import torch
from torch import nn

from rexpolicy.stage0.envs.grasp_lift_action import (
    GRASP_LIFT_ACTION_DIM,
    GRASP_LIFT_EFFECTIVE_ACTION_MASK,
)
from rexpolicy.stage0.evaluation.grasp_lift_rollout import (
    GraspLiftRolloutResult,
)
from rexpolicy.stage0.grasp_lift_no_z_selection import (
    GRASP_LIFT_NO_Z_POLICY_NOISE_INDICES,
    GRASP_LIFT_NO_Z_VALIDATION_RESET_SEEDS,
    GraspLiftNoZSeedCheckpointEvidence,
    GraspLiftNoZValidationRollout,
)
from rexpolicy.stage0.normalization import Stage0Normalization
from rexpolicy.stage0.trainers.batch import Stage0WindowBatch
from rexpolicy.stage0.types import canonical_fingerprint


GRASP_LIFT_OFFLINE_DIAGNOSTIC_NOISE_NAMESPACE = (
    "rexpolicy/stage0-grasp-lift-offline-diagnostic-noise/20260807-v1"
)
GRASP_LIFT_OFFLINE_DIAGNOSTIC_FLOW_STEPS = 16


def _require_sha256(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _module_device_dtype(
    module: nn.Module,
    reference: torch.Tensor,
) -> tuple[torch.device, torch.dtype]:
    tensors = tuple(module.parameters()) + tuple(module.buffers())
    if not tensors:
        return reference.device, reference.dtype
    device = tensors[0].device
    dtype = next(
        (value.dtype for value in tensors if torch.is_floating_point(value)),
        reference.dtype,
    )
    if any(value.device != device for value in tensors):
        raise ValueError("policy parameters and buffers must share one device")
    return device, dtype


def _diagnostic_noise_seed(validation_data_sha256: str) -> int:
    _require_sha256(validation_data_sha256, "validation_data_sha256")
    digest = canonical_fingerprint(
        {
            "namespace": GRASP_LIFT_OFFLINE_DIAGNOSTIC_NOISE_NAMESPACE,
            "validation_data_sha256": validation_data_sha256,
        }
    )
    return int(digest[:16], 16) % ((1 << 63) - 1)


def _fixed_initial_noise(
    batch: Stage0WindowBatch,
    *,
    action_horizon: int,
    action_dim: int,
    validation_data_sha256: str,
) -> torch.Tensor:
    generator = torch.Generator(device=batch.current_states.device)
    generator.manual_seed(_diagnostic_noise_seed(validation_data_sha256))
    return torch.randn(
        (batch.batch_size, action_horizon, action_dim),
        dtype=batch.current_states.dtype,
        device=batch.current_states.device,
        generator=generator,
    )


def _finite_non_negative(value: float, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise FloatingPointError(f"{name} is not finite and non-negative")
    return result


def evaluate_grasp_lift_offline_action_diagnostic(
    policy: nn.Module,
    batch: Stage0WindowBatch,
    normalization: Stage0Normalization,
    *,
    validation_data_sha256: str,
) -> dict[str, float]:
    """Sample one paired flow chunk per validation window and record MSE only.

    These values are descriptive diagnostics.  The fixed checkpoint selector
    never receives a threshold or ranking rule involving them.
    """

    if not isinstance(policy, nn.Module):
        raise TypeError("policy must be a Torch module")
    if not isinstance(batch, Stage0WindowBatch):
        raise TypeError("batch must be Stage0WindowBatch")
    if not isinstance(normalization, Stage0Normalization):
        raise TypeError("normalization must be Stage0Normalization")
    state_dim = getattr(policy, "state_dim", None)
    action_dim = getattr(policy, "action_dim", None)
    action_horizon = getattr(policy, "action_horizon", None)
    if (
        type(state_dim) is not int
        or type(action_dim) is not int
        or type(action_horizon) is not int
        or state_dim != normalization.state_dim
        or action_dim != normalization.action_dim
        or action_dim != GRASP_LIFT_ACTION_DIM
        or action_horizon != batch.action_chunks.shape[1]
        or batch.current_states.shape[1] != state_dim
    ):
        raise ValueError("policy, batch, and normalization dimensions differ")
    if not callable(getattr(policy, "sample", None)):
        raise TypeError("policy must expose sample()")
    expected_mask = torch.tensor(GRASP_LIFT_EFFECTIVE_ACTION_MASK, dtype=torch.bool)
    if not torch.equal(normalization.effective_action_mask, expected_mask):
        raise ValueError("offline diagnostic requires the fixed 13D action mask")
    device, dtype = _module_device_dtype(policy, batch.current_states)
    if batch.current_states.device != device or batch.current_states.dtype != dtype:
        raise ValueError("policy and validation batch must share device and dtype")
    _require_sha256(validation_data_sha256, "validation_data_sha256")

    initial_noise = _fixed_initial_noise(
        batch,
        action_horizon=action_horizon,
        action_dim=action_dim,
        validation_data_sha256=validation_data_sha256,
    )
    feature_mask = normalization.effective_action_mask.to(device=device)
    policy_training = policy.training
    policy.eval()
    try:
        with torch.inference_mode():
            predicted = policy.sample(
                batch.current_states,
                success_latent=None,
                steps=GRASP_LIFT_OFFLINE_DIAGNOSTIC_FLOW_STEPS,
                action_mask=batch.action_mask,
                action_feature_mask=feature_mask,
                initial_noise=initial_noise,
            )
    finally:
        policy.train(policy_training)
    if (
        not isinstance(predicted, torch.Tensor)
        or predicted.shape != batch.action_chunks.shape
        or predicted.device != device
        or predicted.dtype != dtype
        or not bool(torch.isfinite(predicted).all())
    ):
        raise ValueError(
            "offline policy sample changed shape, device, dtype, or finiteness"
        )

    valid_steps = batch.action_mask
    effective = feature_mask[None, None, :]
    valid_features = valid_steps[:, :, None] & effective
    feature_count = int(valid_features.sum())
    valid_step_count = int(valid_steps.sum())
    if feature_count < 1 or valid_step_count < 1:
        raise ValueError("offline diagnostic contains no valid effective actions")
    target = batch.action_chunks
    normalized_error = (predicted - target).square()
    normalized_mse = normalized_error[valid_features].mean()
    train_zero_mse = target.square()[valid_features].mean()
    if not bool(train_zero_mse > 0.0):
        raise ValueError("train-zero validation baseline is degenerate")

    physical_predicted = normalization.denormalize_actions(
        predicted,
        valid_mask=valid_steps,
    )
    physical_target = normalization.denormalize_actions(
        target,
        valid_mask=valid_steps,
    )
    physical_error = (physical_predicted - physical_target).square()
    xyz_valid = valid_steps[:, :, None].expand(-1, -1, 3)
    hand_valid = valid_steps[:, :, None].expand(-1, -1, 10)
    xyz_rmse = torch.sqrt(physical_error[:, :, :3][xyz_valid].mean())
    hand_rmse = torch.sqrt(physical_error[:, :, 9:19][hand_valid].mean())
    ratio = normalized_mse / train_zero_mse
    result = {
        "action_horizon": float(action_horizon),
        "effective_action_dimensions": float(feature_mask.sum()),
        "flow_sample_steps": float(GRASP_LIFT_OFFLINE_DIAGNOSTIC_FLOW_STEPS),
        "hand_physical_rmse_rad": _finite_non_negative(
            float(hand_rmse),
            "hand_physical_rmse_rad",
        ),
        "normalized_effective_action_mse": _finite_non_negative(
            float(normalized_mse),
            "normalized_effective_action_mse",
        ),
        "normalized_mse_to_train_zero_ratio": _finite_non_negative(
            float(ratio),
            "normalized_mse_to_train_zero_ratio",
        ),
        "train_zero_normalized_effective_action_mse": _finite_non_negative(
            float(train_zero_mse),
            "train_zero_normalized_effective_action_mse",
        ),
        "valid_action_steps": float(valid_step_count),
        "validation_windows": float(batch.batch_size),
        "xyz_physical_rmse_m": _finite_non_negative(
            float(xyz_rmse),
            "xyz_physical_rmse_m",
        ),
    }
    expected_feature_count = valid_step_count * sum(
        GRASP_LIFT_EFFECTIVE_ACTION_MASK
    )
    if feature_count != expected_feature_count:
        raise RuntimeError("offline effective feature count changed")
    return result


def grasp_lift_rollout_to_selection_evidence(
    result: GraspLiftRolloutResult,
    *,
    checkpoint_step: int,
    training_seed: int,
    model_sha256: str,
    offline_mse_diagnostic: Mapping[str, Any],
) -> GraspLiftNoZSeedCheckpointEvidence:
    """Convert all 24 formal rollout cells without collapsing failure facts."""

    if not isinstance(result, GraspLiftRolloutResult):
        raise TypeError("result must be GraspLiftRolloutResult")
    expected_cells = tuple(
        (reset_seed, noise_index)
        for reset_seed in GRASP_LIFT_NO_Z_VALIDATION_RESET_SEEDS
        for noise_index in GRASP_LIFT_NO_Z_POLICY_NOISE_INDICES
    )
    observed_cells = tuple(
        zip(
            result.budget.reset_seeds,
            result.budget.noise_variants,
            strict=True,
        )
    )
    if observed_cells != expected_cells:
        raise ValueError("rollout result is not the canonical 6-reset by 4-noise grid")
    rollouts = tuple(
        GraspLiftNoZValidationRollout(
            reset_seed=reset_seed,
            policy_noise_index=noise_index,
            success=bool(result.success[index]),
            safety_violation=bool(result.safety_violation[index]),
            integrity_violation=bool(result.execution_integrity_violation[index]),
            oracle_disagreement=bool(result.oracle_input_disagreement[index]),
        )
        for index, (reset_seed, noise_index) in enumerate(expected_cells)
    )
    return GraspLiftNoZSeedCheckpointEvidence(
        checkpoint_step=checkpoint_step,
        training_seed=training_seed,
        model_sha256=model_sha256,
        rollouts=rollouts,
        offline_mse_diagnostic=offline_mse_diagnostic,
    )


__all__ = [
    "GRASP_LIFT_OFFLINE_DIAGNOSTIC_FLOW_STEPS",
    "GRASP_LIFT_OFFLINE_DIAGNOSTIC_NOISE_NAMESPACE",
    "evaluate_grasp_lift_offline_action_diagnostic",
    "grasp_lift_rollout_to_selection_evidence",
]
