# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""ManiSkill-style vector wrapper for PPO rollouts."""

from __future__ import annotations

from typing import Any

from .groot_newton_env import GrootNewtonEnv


class GrootNewtonVectorEnv:
    """Add GPU auto-reset and final-transition data to :class:`GrootNewtonEnv`.

    The wrapper mirrors the subset of ``ManiSkillVectorEnv`` consumed by the
    official PPO baseline. Final observation buffers are allocated lazily, so
    the base and Diffusion Policy environments do not pay this memory cost.
    """

    def __init__(
        self,
        env: GrootNewtonEnv,
        *,
        auto_reset: bool = True,
        ignore_terminations: bool = False,
        record_metrics: bool = True,
    ):
        self.env = env
        self.num_envs = env.num_envs
        self.device = env.device
        self.auto_reset = auto_reset
        self.ignore_terminations = ignore_terminations
        self.record_metrics = record_metrics
        self.action_space = env.action_space
        self.single_action_space = env.single_action_space
        self.observation_space = env.observation_space
        self.single_observation_space = env.single_observation_space
        self._final_observation: Any = None
        self._final_episode: dict[str, Any] | None = None

    @property
    def unwrapped(self) -> GrootNewtonEnv:
        """Return the underlying Newton environment."""
        return self.env

    @staticmethod
    def _empty_like_tree(value: Any) -> Any:
        import torch

        if isinstance(value, dict):
            return {key: GrootNewtonVectorEnv._empty_like_tree(child) for key, child in value.items()}
        return torch.empty_like(value)

    @staticmethod
    def _copy_masked(destination: Any, source: Any, mask: Any) -> None:
        if isinstance(source, dict):
            for key, child in source.items():
                GrootNewtonVectorEnv._copy_masked(destination[key], child, mask)
            return
        destination[mask] = source[mask]

    def reset(
        self,
        *,
        seed: int | list[int] | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[Any, dict[str, Any]]:
        """Reset the wrapped environment."""
        return self.env.reset(seed=seed, options=options)

    def step(self, action: Any) -> tuple[Any, Any, Any, Any, dict[str, Any]]:
        """Step once and optionally auto-reset completed CUDA worlds."""
        import torch

        observation, reward, terminated, truncated, info = self.env.step(action)
        if not self.auto_reset:
            return observation, reward, terminated, truncated, info

        reward_out = reward.clone()
        terminated_out = terminated.clone()
        truncated_out = truncated.clone()
        if self.ignore_terminations:
            terminated_out.zero_()
        done = terminated_out | truncated_out

        if self._final_observation is None:
            self._final_observation = self._empty_like_tree(observation)
        self._copy_masked(self._final_observation, observation, done)

        episode = info["episode"]
        if self._final_episode is None:
            self._final_episode = {key: torch.empty_like(value) for key, value in episode.items()}
        for key, value in episode.items():
            self._final_episode[key][done] = value[done]

        observation, reset_info = self.env.reset(world_mask=done)
        output_info = {
            **reset_info,
            "_final_info": done,
            "final_observation": self._final_observation,
            "final_info": {"episode": self._final_episode},
        }
        return observation, reward_out, terminated_out, truncated_out, output_info

    def close(self) -> None:
        """Close the underlying environment."""
        self.env.close()
