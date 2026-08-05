"""Fixed-budget learned-policy rollouts for the Stage 0 Newton Reach task.

This module intentionally owns no Newton construction.  It consumes the
state-only adapter contract, a trained :class:`Stage0FlowPolicy`, immutable
train-split normalization, and an independent Reach oracle.  Keeping that
boundary narrow makes the scientific controls CPU-testable while the real
runner remains responsible for simulator configuration and checkpoint IO.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any

import torch
from torch import nn

from rexpolicy.stage0.envs.oracles import ReachSuccessOracle
from rexpolicy.stage0.envs.state_schema import DEFAULT_STAGE0_STATE_SCHEMA
from rexpolicy.stage0.normalization import Stage0Normalization
from rexpolicy.stage0.types import Stage0Outcome, canonical_fingerprint

MAX_REACH_TRANSLATION_STEP_M = 0.030


class RolloutConditioningMode(str, Enum):
    """Scientific controls supported by the learned-policy rollout path."""

    NO_Z = "no-z"
    ORACLE_Z = "oracle-z"
    SELECTOR_Z = "selector-z"
    PERMUTED_Z = "permuted-z"


def _conditioning_mode(
    value: RolloutConditioningMode | str,
) -> RolloutConditioningMode:
    try:
        return RolloutConditioningMode(value)
    except (TypeError, ValueError) as error:
        choices = ", ".join(mode.value for mode in RolloutConditioningMode)
        raise ValueError(f"conditioning mode must be one of: {choices}") from error


def _positive_int(value: Any, *, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _non_negative_seed(value: Any, *, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


@dataclass(frozen=True)
class ReachRolloutBudget:
    """Reset and per-episode flow-noise allocation shared by every control."""

    reset_group_ids: tuple[str, ...]
    reset_seeds: tuple[int, ...]
    noise_seeds: tuple[int, ...]
    max_steps_per_episode: int

    def __post_init__(self) -> None:
        if not isinstance(self.reset_group_ids, tuple) or not self.reset_group_ids:
            raise ValueError("reset_group_ids must be a non-empty tuple")
        if not isinstance(self.reset_seeds, tuple):
            raise TypeError("reset_seeds must be a tuple")
        if not isinstance(self.noise_seeds, tuple):
            raise TypeError("noise_seeds must be a tuple")
        if not (
            len(self.reset_group_ids) == len(self.reset_seeds) == len(self.noise_seeds)
        ):
            raise ValueError("reset groups, reset seeds, and noise seeds must align")
        allocations: set[tuple[str, int, int]] = set()
        for index, (group_id, reset_seed, noise_seed) in enumerate(
            zip(self.reset_group_ids, self.reset_seeds, self.noise_seeds)
        ):
            if not isinstance(group_id, str) or not group_id.strip():
                raise ValueError(f"reset_group_ids[{index}] must be non-empty text")
            _non_negative_seed(reset_seed, name=f"reset_seeds[{index}]")
            _non_negative_seed(noise_seed, name=f"noise_seeds[{index}]")
            allocation = (group_id, reset_seed, noise_seed)
            if allocation in allocations:
                raise ValueError("rollout budget allocations must be unique")
            allocations.add(allocation)
        _positive_int(
            self.max_steps_per_episode,
            name="max_steps_per_episode",
        )

    @property
    def episode_count(self) -> int:
        return len(self.reset_group_ids)

    @property
    def allocated_world_steps(self) -> int:
        return self.episode_count * self.max_steps_per_episode

    @property
    def fingerprint(self) -> str:
        return canonical_fingerprint(self.to_record())

    def to_record(self) -> dict[str, Any]:
        return {
            "allocated_world_steps": self.allocated_world_steps,
            "max_steps_per_episode": self.max_steps_per_episode,
            "noise_seeds": list(self.noise_seeds),
            "reset_group_ids": list(self.reset_group_ids),
            "reset_seeds": list(self.reset_seeds),
        }


def _cpu_tensor(
    value: torch.Tensor,
    *,
    name: str,
    dtype: torch.dtype,
    shape: tuple[int, ...],
) -> torch.Tensor:
    if (
        not isinstance(value, torch.Tensor)
        or value.dtype is not dtype
        or value.device.type != "cpu"
        or tuple(value.shape) != shape
    ):
        raise ValueError(f"{name} must be CPU {dtype} with shape {shape}")
    return value.detach().contiguous()


@dataclass(frozen=True)
class ReachRolloutResult:
    """Per-world outcomes plus the path witness needed for mode adherence."""

    mode: RolloutConditioningMode
    budget: ReachRolloutBudget
    evaluated_mask: torch.Tensor
    success: torch.Tensor
    failure: torch.Tensor
    timeout: torch.Tensor
    completed_steps: torch.Tensor
    contact_violation: torch.Tensor
    maximum_object_displacement_m: torch.Tensor
    final_goal_distance_m: torch.Tensor
    eef_trace_m: torch.Tensor
    trace_valid_mask: torch.Tensor
    episode_latents: torch.Tensor | None = None
    latent_permutation: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "mode", _conditioning_mode(self.mode))
        if not isinstance(self.budget, ReachRolloutBudget):
            raise TypeError("budget must be a ReachRolloutBudget")
        episodes = self.budget.episode_count
        horizon = self.budget.max_steps_per_episode
        for name in (
            "evaluated_mask",
            "success",
            "failure",
            "timeout",
            "contact_violation",
        ):
            object.__setattr__(
                self,
                name,
                _cpu_tensor(
                    getattr(self, name),
                    name=name,
                    dtype=torch.bool,
                    shape=(episodes,),
                ),
            )
        object.__setattr__(
            self,
            "completed_steps",
            _cpu_tensor(
                self.completed_steps,
                name="completed_steps",
                dtype=torch.int64,
                shape=(episodes,),
            ),
        )
        for name in (
            "maximum_object_displacement_m",
            "final_goal_distance_m",
        ):
            tensor = _cpu_tensor(
                getattr(self, name),
                name=name,
                dtype=torch.float32,
                shape=(episodes,),
            )
            if not bool(torch.isfinite(tensor).all()) or bool((tensor < 0.0).any()):
                raise ValueError(f"{name} must be finite and non-negative")
            object.__setattr__(self, name, tensor)
        trace = _cpu_tensor(
            self.eef_trace_m,
            name="eef_trace_m",
            dtype=torch.float32,
            shape=(episodes, horizon + 1, 3),
        )
        valid = _cpu_tensor(
            self.trace_valid_mask,
            name="trace_valid_mask",
            dtype=torch.bool,
            shape=(episodes, horizon + 1),
        )
        if not bool(torch.isfinite(trace[valid]).all()):
            raise ValueError("valid EEF trace entries must be finite")
        if bool(torch.isfinite(trace[~valid]).any()):
            raise ValueError("invalid EEF trace entries must be NaN padded")
        object.__setattr__(self, "eef_trace_m", trace)
        object.__setattr__(self, "trace_valid_mask", valid)

        terminal_count = (
            self.success.to(torch.int8)
            + self.failure.to(torch.int8)
            + self.timeout.to(torch.int8)
        )
        if not torch.equal(terminal_count, self.evaluated_mask.to(torch.int8)):
            raise ValueError(
                "success/failure/timeout must partition exactly the evaluated worlds"
            )
        expected_trace_lengths = self.completed_steps + self.evaluated_mask.to(
            torch.int64
        )
        if not torch.equal(valid.sum(dim=1), expected_trace_lengths):
            raise ValueError("trace lengths must equal completed_steps + initial state")
        for world in range(episodes):
            length = int(expected_trace_lengths[world])
            if length and not bool(valid[world, :length].all()):
                raise ValueError("trace_valid_mask must be prefix-contiguous")
            if bool(valid[world, length:].any()):
                raise ValueError("trace_valid_mask must be prefix-contiguous")
        if bool((self.completed_steps[~self.evaluated_mask] != 0).any()):
            raise ValueError("unevaluated worlds must execute zero steps")
        evaluated_steps = self.completed_steps[self.evaluated_mask]
        if bool((evaluated_steps < 1).any()) or bool((evaluated_steps > horizon).any()):
            raise ValueError("evaluated completed_steps must be inside the budget")

        if self.episode_latents is not None:
            latents = self.episode_latents
            if (
                not isinstance(latents, torch.Tensor)
                or latents.device.type != "cpu"
                or latents.ndim != 2
                or latents.shape[0] != episodes
                or not torch.is_floating_point(latents)
                or not bool(torch.isfinite(latents).all())
            ):
                raise ValueError(
                    "episode_latents must be finite floating CPU [episodes, latent_dim]"
                )
            object.__setattr__(self, "episode_latents", latents.detach().contiguous())
        if self.mode is RolloutConditioningMode.NO_Z:
            if self.episode_latents is not None:
                raise ValueError("no-z rollouts cannot carry episode latents")
        elif self.episode_latents is None:
            raise ValueError(f"{self.mode.value} rollouts require episode latents")

        if self.latent_permutation is not None:
            permutation = _validate_permutation(self.latent_permutation, episodes)
            object.__setattr__(self, "latent_permutation", permutation)
        if self.mode is RolloutConditioningMode.PERMUTED_Z:
            if self.latent_permutation is None:
                raise ValueError("permuted-z rollouts require latent_permutation")
        elif self.latent_permutation is not None:
            raise ValueError("latent_permutation is only valid for permuted-z")

    @property
    def executed_world_steps(self) -> int:
        return int(self.completed_steps.sum())

    @property
    def evaluated_episode_count(self) -> int:
        return int(self.evaluated_mask.sum())

    @property
    def success_rate(self) -> float:
        if self.evaluated_episode_count == 0:
            return math.nan
        return int(self.success.sum()) / self.evaluated_episode_count

    @property
    def outcomes(self) -> tuple[Stage0Outcome | None, ...]:
        values: list[Stage0Outcome | None] = []
        for world in range(self.budget.episode_count):
            if bool(self.success[world]):
                values.append(Stage0Outcome.SUCCESS)
            elif bool(self.failure[world]):
                values.append(Stage0Outcome.FAILURE)
            elif bool(self.timeout[world]):
                values.append(Stage0Outcome.TIMEOUT)
            else:
                values.append(None)
        return tuple(values)

    def eef_paths(self) -> tuple[torch.Tensor | None, ...]:
        """Return unpadded per-world paths for geometric mode diagnostics."""
        paths: list[torch.Tensor | None] = []
        for world in range(self.budget.episode_count):
            valid = self.trace_valid_mask[world]
            paths.append(
                self.eef_trace_m[world, valid].clone() if bool(valid.any()) else None
            )
        return tuple(paths)

    def to_record(self, *, include_eef_trace: bool = True) -> dict[str, Any]:
        record: dict[str, Any] = {
            "budget_sha256": self.budget.fingerprint,
            "completed_steps": self.completed_steps.tolist(),
            "contact_violation": self.contact_violation.tolist(),
            "evaluated_mask": self.evaluated_mask.tolist(),
            "executed_world_steps": self.executed_world_steps,
            "failure": self.failure.tolist(),
            "final_goal_distance_m": self.final_goal_distance_m.tolist(),
            "latent_permutation": None
            if self.latent_permutation is None
            else list(self.latent_permutation),
            "maximum_object_displacement_m": (
                self.maximum_object_displacement_m.tolist()
            ),
            "mode": self.mode.value,
            "outcomes": [
                None if value is None else value.value for value in self.outcomes
            ],
            "success": self.success.tolist(),
            "success_rate": self.success_rate,
            "timeout": self.timeout.tolist(),
            "trace_lengths": self.trace_valid_mask.sum(dim=1).tolist(),
        }
        if include_eef_trace:
            record["eef_trace_m"] = [
                None if path is None else path.tolist() for path in self.eef_paths()
            ]
        return record


def _validate_permutation(
    value: Sequence[int],
    episode_count: int,
) -> tuple[int, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError("latent_permutation must be a sequence")
    result = tuple(value)
    if any(type(index) is not int for index in result):
        raise ValueError("latent_permutation must contain integers")
    if len(result) != episode_count or set(result) != set(range(episode_count)):
        raise ValueError("latent_permutation must be a bijection over all episodes")
    if any(world == source for world, source in enumerate(result)):
        raise ValueError("latent_permutation must not contain fixed points")
    return result


def _default_permutation(episode_count: int) -> tuple[int, ...]:
    if episode_count < 2:
        raise ValueError("permuted-z requires at least two episodes")
    return (*range(1, episode_count), 0)


def _policy_contract(policy: nn.Module) -> tuple[int, int, int, int]:
    values: list[int] = []
    for name in ("state_dim", "latent_dim", "action_dim", "action_horizon"):
        value = getattr(policy, name, None)
        if type(value) is not int or value <= 0:
            raise TypeError(f"policy must expose a positive integer {name}")
        values.append(value)
    if not callable(getattr(policy, "sample", None)):
        raise TypeError("policy must provide sample()")
    return tuple(values)  # type: ignore[return-value]


def _module_device_dtype(
    module: nn.Module,
    reference: torch.Tensor,
) -> tuple[torch.device, torch.dtype]:
    parameters = tuple(module.parameters())
    buffers = tuple(module.buffers())
    tensors = parameters + buffers
    if not tensors:
        return reference.device, reference.dtype
    device = tensors[0].device
    dtype = next(
        (value.dtype for value in tensors if torch.is_floating_point(value)),
        reference.dtype,
    )
    if any(value.device != device for value in tensors):
        raise ValueError("module parameters and buffers must share one device")
    if not torch.is_floating_point(torch.empty((), dtype=dtype)):
        raise ValueError("module must expose a floating point dtype")
    return device, dtype


def _validate_state(state: torch.Tensor, *, episodes: int) -> None:
    expected = DEFAULT_STAGE0_STATE_SCHEMA.dimension
    if (
        not isinstance(state, torch.Tensor)
        or tuple(state.shape) != (episodes, expected)
        or not torch.is_floating_point(state)
        or not bool(torch.isfinite(state).all())
    ):
        raise ValueError(
            f"environment state must be finite floating [{episodes}, {expected}]"
        )


def _step_noise_seed(base_seed: int, control_step: int) -> int:
    digest = canonical_fingerprint(
        {
            "control_step": control_step,
            "noise_seed": base_seed,
            "schema_id": "rexpolicy/stage0-rollout-noise/v1",
        }
    )
    return int(digest[:16], 16) % ((1 << 63) - 1)


def _initial_noise(
    budget: ReachRolloutBudget,
    *,
    control_step: int,
    horizon: int,
    action_dim: int,
    reference: torch.Tensor,
) -> torch.Tensor:
    rows: list[torch.Tensor] = []
    for base_seed in budget.noise_seeds:
        generator = torch.Generator(device=reference.device)
        generator.manual_seed(_step_noise_seed(base_seed, control_step))
        rows.append(
            torch.randn(
                (horizon, action_dim),
                device=reference.device,
                dtype=reference.dtype,
                generator=generator,
            )
        )
    return torch.stack(rows)


def _project_effective_action(env: Any, action: torch.Tensor) -> torch.Tensor:
    candidates = (env, getattr(env, "env", None))
    for candidate in candidates:
        if candidate is None:
            continue
        method = getattr(candidate, "project_effective_action_torch", None)
        if method is None:
            method = getattr(candidate, "project_effective_action", None)
        if method is not None:
            projected = method(action)
            if not isinstance(projected, torch.Tensor):
                raise TypeError("project_effective_action must return a Torch tensor")
            if projected.shape != action.shape or projected.device != action.device:
                raise ValueError("projected action must preserve shape and device")
            if not bool(torch.isfinite(projected).all()):
                raise FloatingPointError("projected action contains non-finite values")
            return projected
    raise TypeError(
        "environment or wrapped environment must expose project_effective_action"
    )


def _optional_info_vector(
    info: Mapping[str, Any],
    names: Sequence[str],
    *,
    reference: torch.Tensor,
    dtype: torch.dtype,
) -> torch.Tensor | None:
    for name in names:
        if name not in info:
            continue
        value = info[name]
        if not isinstance(value, torch.Tensor) or value.shape != reference.shape:
            raise ValueError(
                f"transition info {name!r} must have shape {tuple(reference.shape)}"
            )
        return value.to(device=reference.device, dtype=dtype)
    return None


def _episode_latent(
    mode: RolloutConditioningMode,
    *,
    policy: nn.Module,
    normalized_initial_state: torch.Tensor,
    oracle_latents: torch.Tensor | None,
    selector: nn.Module | None,
    selector_generator: torch.Generator | None,
    latent_permutation: Sequence[int] | None,
) -> tuple[torch.Tensor | None, tuple[int, ...] | None]:
    episodes = normalized_initial_state.shape[0]
    latent_dim = int(policy.latent_dim)

    def checked(value: torch.Tensor | None, name: str) -> torch.Tensor:
        if (
            not isinstance(value, torch.Tensor)
            or tuple(value.shape) != (episodes, latent_dim)
            or not torch.is_floating_point(value)
            or not bool(torch.isfinite(value).all())
        ):
            raise ValueError(
                f"{name} must be finite floating [{episodes}, {latent_dim}]"
            )
        return value.to(
            device=normalized_initial_state.device,
            dtype=normalized_initial_state.dtype,
        )

    if mode is RolloutConditioningMode.NO_Z:
        return normalized_initial_state.new_zeros((episodes, latent_dim)), None
    if mode is RolloutConditioningMode.ORACLE_Z:
        return checked(oracle_latents, "oracle_latents"), None
    if mode is RolloutConditioningMode.SELECTOR_Z:
        if not isinstance(selector, nn.Module):
            raise TypeError("selector-z requires a Torch selector module")
        sampler = getattr(selector, "sample", None)
        predictor = getattr(selector, "predict", None)
        if sampler is not None:
            selected = sampler(
                normalized_initial_state,
                generator=selector_generator,
            )
        elif predictor is not None:
            selected = predictor(normalized_initial_state)
        else:
            distribution = selector(normalized_initial_state)
            selected = getattr(distribution, "mode", None)
        return checked(selected, "selector latent"), None

    oracle = checked(oracle_latents, "oracle_latents")
    permutation = (
        _default_permutation(episodes)
        if latent_permutation is None
        else _validate_permutation(latent_permutation, episodes)
    )
    indices = torch.tensor(permutation, device=oracle.device, dtype=torch.long)
    return oracle.index_select(0, indices), permutation


def rollout_reach_policy(
    env: Any,
    policy: nn.Module,
    normalization: Stage0Normalization,
    budget: ReachRolloutBudget,
    *,
    mode: RolloutConditioningMode | str,
    oracle: ReachSuccessOracle | None = None,
    oracle_latents: torch.Tensor | None = None,
    selector: nn.Module | None = None,
    selector_generator: torch.Generator | None = None,
    latent_permutation: Sequence[int] | None = None,
    initial_active_mask: torch.Tensor | None = None,
    flow_sample_steps: int = 16,
    max_translation_step_m: float = MAX_REACH_TRANSLATION_STEP_M,
) -> ReachRolloutResult:
    """Execute one receding-horizon Reach evaluation under a fixed budget.

    At every control step the current state is normalized and a complete flow
    chunk is sampled with per-episode deterministic noise.  Only the first
    normalized XYZ delta is denormalized.  Its Euclidean norm is bounded by
    3 cm, added to the current EEF position, and projected through Newton's
    task-specific effective-action contract before execution.  The latent is
    selected exactly once from the reset state and remains fixed for the whole
    episode.
    """

    if not isinstance(policy, nn.Module):
        raise TypeError("policy must be a Torch module")
    if not isinstance(normalization, Stage0Normalization):
        raise TypeError("normalization must be a Stage0Normalization")
    if not isinstance(budget, ReachRolloutBudget):
        raise TypeError("budget must be a ReachRolloutBudget")
    mode = _conditioning_mode(mode)
    flow_sample_steps = _positive_int(flow_sample_steps, name="flow_sample_steps")
    if (
        isinstance(max_translation_step_m, bool)
        or not isinstance(max_translation_step_m, (int, float))
        or not math.isfinite(float(max_translation_step_m))
        or float(max_translation_step_m) <= 0.0
        or float(max_translation_step_m) > MAX_REACH_TRANSLATION_STEP_M
    ):
        raise ValueError(
            f"max_translation_step_m must be in (0, {MAX_REACH_TRANSLATION_STEP_M}]"
        )
    episodes = budget.episode_count
    if getattr(env, "num_envs", None) != episodes:
        raise ValueError("environment world count must equal rollout budget episodes")
    state_dim, _, action_dim, action_horizon = _policy_contract(policy)
    if state_dim != normalization.state_dim:
        raise ValueError("policy and normalization state dimensions differ")
    if action_dim != normalization.action_dim:
        raise ValueError("policy and normalization action dimensions differ")
    if action_dim != 19:
        raise ValueError("Stage 0 Reach rollout requires the 19D action contract")
    effective_mask = normalization.effective_action_mask
    if not bool(effective_mask[:3].all()) or bool(effective_mask[3:].any()):
        raise ValueError("Reach rollout requires XYZ-only effective actions")

    reset_result = env.reset(seed=list(budget.reset_seeds))
    if not isinstance(reset_result, tuple) or len(reset_result) != 2:
        raise TypeError("environment reset must return (state, info)")
    state = reset_result[0]
    _validate_state(state, episodes=episodes)
    device, dtype = _module_device_dtype(policy, state)
    if state.device != device:
        raise ValueError("policy and environment state must share one device")
    if state.dtype != dtype:
        raise ValueError("policy and environment state must share one dtype")

    if initial_active_mask is None:
        evaluated = torch.ones(episodes, dtype=torch.bool, device=device)
    else:
        if (
            not isinstance(initial_active_mask, torch.Tensor)
            or initial_active_mask.dtype is not torch.bool
            or tuple(initial_active_mask.shape) != (episodes,)
        ):
            raise ValueError("initial_active_mask must be bool with shape [episodes]")
        evaluated = initial_active_mask.to(device=device).clone()
    if not bool(evaluated.any()):
        raise ValueError("initial_active_mask must evaluate at least one world")

    if oracle is None:
        oracle = ReachSuccessOracle(episodes)
    if not isinstance(oracle, ReachSuccessOracle) or oracle.num_envs != episodes:
        raise ValueError("oracle must be a ReachSuccessOracle for every budget episode")
    oracle.reset(state)

    policy_training = policy.training
    selector_training = selector.training if isinstance(selector, nn.Module) else None
    policy.eval()
    if isinstance(selector, nn.Module):
        selector.eval()

    schema = DEFAULT_STAGE0_STATE_SCHEMA
    eef_slice = schema.slice("eef_position")
    goal_slice = schema.slice("goal_position")
    trace = torch.full(
        (episodes, budget.max_steps_per_episode + 1, 3),
        math.nan,
        dtype=torch.float32,
        device="cpu",
    )
    trace_valid = torch.zeros(
        (episodes, budget.max_steps_per_episode + 1),
        dtype=torch.bool,
        device="cpu",
    )
    evaluated_cpu = evaluated.detach().cpu()
    trace[evaluated_cpu, 0] = state[evaluated, eef_slice].detach().float().cpu()
    trace_valid[evaluated_cpu, 0] = True

    success = torch.zeros(episodes, dtype=torch.bool, device=device)
    failure = torch.zeros_like(success)
    timeout = torch.zeros_like(success)
    contact_violation = torch.zeros_like(success)
    completed_steps = torch.zeros(episodes, dtype=torch.int64, device=device)
    maximum_displacement = torch.zeros(episodes, dtype=dtype, device=device)
    final_distance = torch.linalg.vector_norm(
        state[:, goal_slice] - state[:, eef_slice], dim=-1
    )
    active = evaluated.clone()

    try:
        with torch.inference_mode():
            normalized_initial = normalization.normalize_states(state)
            episode_latent, permutation = _episode_latent(
                mode,
                policy=policy,
                normalized_initial_state=normalized_initial,
                oracle_latents=oracle_latents,
                selector=selector,
                selector_generator=selector_generator,
                latent_permutation=latent_permutation,
            )
            feature_mask = normalization.effective_action_mask.to(device=device)

            for control_step in range(budget.max_steps_per_episode):
                if not bool(active.any()):
                    break
                active_before = active.clone()
                normalized_state = normalization.normalize_states(state)
                initial_noise = _initial_noise(
                    budget,
                    control_step=control_step,
                    horizon=action_horizon,
                    action_dim=action_dim,
                    reference=normalized_state,
                )
                normalized_chunk = policy.sample(
                    normalized_state,
                    success_latent=episode_latent,
                    steps=flow_sample_steps,
                    action_feature_mask=feature_mask,
                    initial_noise=initial_noise,
                )
                expected_shape = (episodes, action_horizon, action_dim)
                if (
                    not isinstance(normalized_chunk, torch.Tensor)
                    or tuple(normalized_chunk.shape) != expected_shape
                    or normalized_chunk.device != device
                    or normalized_chunk.dtype != dtype
                    or not bool(torch.isfinite(normalized_chunk).all())
                ):
                    raise ValueError(
                        "policy sample must be finite and have shape "
                        f"{expected_shape} on the policy device/dtype"
                    )

                physical_first = normalization.denormalize_actions(
                    normalized_chunk[:, 0, :]
                )
                delta = physical_first[:, :3]
                delta_norm = torch.linalg.vector_norm(delta, dim=-1, keepdim=True)
                scale = torch.clamp(
                    float(max_translation_step_m)
                    / torch.clamp_min(delta_norm, torch.finfo(dtype).eps),
                    max=1.0,
                )
                bounded_delta = delta * scale
                current_eef = state[:, eef_slice]
                proposed = torch.zeros(
                    (episodes, action_dim),
                    dtype=dtype,
                    device=device,
                )
                proposed[:, :3] = current_eef + torch.where(
                    active_before[:, None],
                    bounded_delta,
                    torch.zeros_like(bounded_delta),
                )
                effective_action = _project_effective_action(env, proposed)
                transition = env.step(effective_action)
                next_state = getattr(transition, "next_state", None)
                _validate_state(next_state, episodes=episodes)
                if next_state.device != device or next_state.dtype != dtype:
                    raise ValueError(
                        "environment next_state must preserve policy device and dtype"
                    )
                terminated = getattr(transition, "terminated", None)
                truncated = getattr(transition, "truncated", None)
                if (
                    not isinstance(terminated, torch.Tensor)
                    or terminated.dtype is not torch.bool
                    or tuple(terminated.shape) != (episodes,)
                    or terminated.device != device
                    or not isinstance(truncated, torch.Tensor)
                    or truncated.dtype is not torch.bool
                    or tuple(truncated.shape) != (episodes,)
                    or truncated.device != device
                ):
                    raise ValueError(
                        "environment termination flags must be device bool [episodes]"
                    )
                info = getattr(transition, "info", None)
                if not isinstance(info, Mapping):
                    raise TypeError("environment transition info must be a mapping")

                completed_steps.add_(active_before.to(torch.int64))
                active_before_cpu = active_before.detach().cpu()
                trace[active_before_cpu, control_step + 1] = (
                    next_state[active_before, eef_slice].detach().float().cpu()
                )
                trace_valid[active_before_cpu, control_step + 1] = True

                result = oracle.evaluate(next_state)
                info_contact = _optional_info_vector(
                    info,
                    (
                        "had_hand_contact_this_control_step",
                        "contact_violation",
                    ),
                    reference=active,
                    dtype=torch.bool,
                )
                step_contact = result.contact_violation
                if info_contact is not None:
                    step_contact = step_contact | info_contact
                info_displacement = _optional_info_vector(
                    info,
                    (
                        "reach_max_bottle_displacement",
                        "object_displacement",
                    ),
                    reference=active,
                    dtype=dtype,
                )
                step_displacement = result.object_displacement
                if info_displacement is not None:
                    step_displacement = torch.maximum(
                        step_displacement,
                        info_displacement,
                    )
                contact_violation |= step_contact & active_before
                maximum_displacement = torch.where(
                    active_before,
                    torch.maximum(maximum_displacement, step_displacement),
                    maximum_displacement,
                )
                final_distance = torch.where(
                    active_before,
                    result.distance,
                    final_distance,
                )

                info_success = _optional_info_vector(
                    info,
                    ("success",),
                    reference=active,
                    dtype=torch.bool,
                )
                info_failure = _optional_info_vector(
                    info,
                    ("fail", "failure"),
                    reference=active,
                    dtype=torch.bool,
                )
                success_candidate = (
                    result.success if info_success is None else info_success
                )
                failure_candidate = (
                    result.failure if info_failure is None else info_failure
                )

                step_failure = (
                    failure_candidate
                    | step_contact
                    | (step_displacement > oracle.displacement_limit_m)
                ) & active_before
                step_success = success_candidate & ~step_failure & active_before
                unexplained_termination = (
                    terminated & ~step_success & ~step_failure & active_before
                )
                step_failure |= unexplained_termination
                step_timeout = truncated & ~step_success & ~step_failure & active_before
                success |= step_success
                failure |= step_failure
                timeout |= step_timeout
                active &= ~(step_success | step_failure | step_timeout)
                state = next_state

            timeout |= active
            active.zero_()
    finally:
        policy.train(policy_training)
        if isinstance(selector, nn.Module) and selector_training is not None:
            selector.train(selector_training)

    return ReachRolloutResult(
        mode=mode,
        budget=budget,
        evaluated_mask=evaluated.detach().cpu(),
        success=success.detach().cpu(),
        failure=failure.detach().cpu(),
        timeout=timeout.detach().cpu(),
        completed_steps=completed_steps.detach().cpu(),
        contact_violation=contact_violation.detach().cpu(),
        maximum_object_displacement_m=maximum_displacement.detach().float().cpu(),
        final_goal_distance_m=final_distance.detach().float().cpu(),
        eef_trace_m=trace,
        trace_valid_mask=trace_valid,
        episode_latents=(
            None
            if mode is RolloutConditioningMode.NO_Z
            else episode_latent.detach().float().cpu()
        ),
        latent_permutation=permutation,
    )


__all__ = [
    "MAX_REACH_TRANSLATION_STEP_M",
    "ReachRolloutBudget",
    "ReachRolloutResult",
    "RolloutConditioningMode",
    "rollout_reach_policy",
]
