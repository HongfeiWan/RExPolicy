"""K=1 held-out evaluation metrics and non-regression gate."""

from __future__ import annotations

import hashlib
import statistics
from dataclasses import asdict, dataclass
from typing import Any

TRAIN_RESET_SEED_DOMAIN = 1
HELD_OUT_RESET_SEED_DOMAIN = 2
HELD_OUT_DIFFUSION_SEED_DOMAIN = 3
_SEED_PAYLOAD_BITS = 60
_SEED_PAYLOAD_MASK = (1 << _SEED_PAYLOAD_BITS) - 1


def domain_seed(
    base_seed: int,
    domain: int,
    *indices: int,
) -> int:
    """Derive a stable positive seed in a mathematically disjoint domain."""
    if domain < 1 or domain > 7:
        raise ValueError("seed domain must be in [1, 7]")
    payload = ":".join(
        str(int(value)) for value in (base_seed, domain, *indices)
    ).encode("ascii")
    digest = hashlib.sha256(payload).digest()
    lower = int.from_bytes(digest[:8], "big") & _SEED_PAYLOAD_MASK
    return (int(domain) << _SEED_PAYLOAD_BITS) | lower


@dataclass(frozen=True)
class HeldOutEpisodeResult:
    """One fixed reset-recipe and diffusion-seed rollout."""

    reset_seed: int
    diffusion_seed: int
    success: bool
    safety_failure: bool
    invalid_action: bool
    final_distance: float
    episode_return: float
    object_displacement: float
    control_steps: int


@dataclass(frozen=True)
class HeldOutMetrics:
    """Aggregate paired K=1 metrics for one policy generation."""

    generation: int
    episode_count: int
    success_count: int
    safety_failure_count: int
    invalid_action_count: int
    median_final_distance: float
    mean_episode_return: float
    max_object_displacement: float
    episodes: tuple[HeldOutEpisodeResult, ...]

    @classmethod
    def from_episodes(
        cls,
        *,
        generation: int,
        episodes: list[HeldOutEpisodeResult],
    ) -> HeldOutMetrics:
        """Build deterministic aggregate metrics from the complete suite."""
        if not episodes:
            raise ValueError("Held-out evaluation requires at least one episode")
        for episode in episodes:
            values = (
                episode.final_distance,
                episode.episode_return,
                episode.object_displacement,
            )
            if not all(_is_finite(value) for value in values):
                raise ValueError("Held-out metrics contain non-finite values")
        return cls(
            generation=int(generation),
            episode_count=len(episodes),
            success_count=sum(episode.success for episode in episodes),
            safety_failure_count=sum(
                episode.safety_failure for episode in episodes
            ),
            invalid_action_count=sum(episode.invalid_action for episode in episodes),
            median_final_distance=float(
                statistics.median(episode.final_distance for episode in episodes)
            ),
            mean_episode_return=float(
                statistics.fmean(episode.episode_return for episode in episodes)
            ),
            max_object_displacement=max(
                episode.object_displacement for episode in episodes
            ),
            episodes=tuple(episodes),
        )

    def to_record(self) -> dict[str, Any]:
        """Return a JSON-ready metrics record."""
        record = asdict(self)
        record["episodes"] = [asdict(episode) for episode in self.episodes]
        return record

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> HeldOutMetrics:
        """Restore metrics from checkpoint state."""
        values = dict(record)
        values["episodes"] = tuple(
            HeldOutEpisodeResult(**episode) for episode in values["episodes"]
        )
        return cls(**values)


@dataclass(frozen=True)
class GateDecision:
    """Promotion decision against the last accepted K=1 policy."""

    accepted: bool
    reasons: tuple[str, ...]


def evaluate_non_regression(
    *,
    last_good: HeldOutMetrics,
    candidate: HeldOutMetrics,
    success_rate_drop_tolerance: float = 1.0 / 32.0,
    distance_tolerance_m: float = 0.005,
) -> GateDecision:
    """Reject safety regressions and material K=1 task regressions."""
    if candidate.episode_count != last_good.episode_count:
        raise ValueError("Held-out suites must have identical episode counts")
    if not 0.0 <= success_rate_drop_tolerance <= 1.0:
        raise ValueError("Success-rate drop tolerance must be in [0, 1]")
    last_by_recipe = {
        (episode.reset_seed, episode.diffusion_seed): episode
        for episode in last_good.episodes
    }
    candidate_by_recipe = {
        (episode.reset_seed, episode.diffusion_seed): episode
        for episode in candidate.episodes
    }
    if (
        len(last_by_recipe) != last_good.episode_count
        or len(candidate_by_recipe) != candidate.episode_count
        or set(last_by_recipe) != set(candidate_by_recipe)
    ):
        raise ValueError(
            "Held-out evaluations must contain the same unique reset/diffusion pairs"
        )
    reasons = []
    if candidate.invalid_action_count > 0:
        reasons.append(
            f"candidate has {candidate.invalid_action_count} invalid actions"
        )
    new_safety_failures = sorted(
        recipe
        for recipe, episode in candidate_by_recipe.items()
        if episode.safety_failure and not last_by_recipe[recipe].safety_failure
    )
    if new_safety_failures:
        reasons.append(
            f"candidate introduced {len(new_safety_failures)} new safety failures"
        )
    success_rate_drop = (
        last_good.success_count - candidate.success_count
    ) / last_good.episode_count
    if success_rate_drop > success_rate_drop_tolerance:
        reasons.append(
            "success rate regressed "
            f"{last_good.success_count}/{last_good.episode_count}->"
            f"{candidate.success_count}/{candidate.episode_count}"
        )
    success_increased = candidate.success_count > last_good.success_count
    if (
        not success_increased
        and candidate.median_final_distance
        > last_good.median_final_distance + distance_tolerance_m
    ):
        reasons.append(
            "median final distance regressed "
            f"{last_good.median_final_distance:.6f}->"
            f"{candidate.median_final_distance:.6f}"
        )
    return GateDecision(accepted=not reasons, reasons=tuple(reasons))


def fixed_evaluation_suite(
    *,
    base_seed: int,
    reset_recipes: int = 8,
    diffusion_seeds_per_recipe: int = 4,
) -> tuple[tuple[int, int], ...]:
    """Return the sealed 8x4 reset/diffusion seed pairs."""
    if reset_recipes < 1 or diffusion_seeds_per_recipe < 1:
        raise ValueError("Evaluation suite dimensions must be positive")
    pairs = []
    for recipe_index in range(reset_recipes):
        reset_seed = domain_seed(
            base_seed,
            HELD_OUT_RESET_SEED_DOMAIN,
            recipe_index,
        )
        for diffusion_index in range(diffusion_seeds_per_recipe):
            diffusion_seed = domain_seed(
                base_seed,
                HELD_OUT_DIFFUSION_SEED_DOMAIN,
                recipe_index,
                diffusion_index,
            )
            pairs.append((reset_seed, diffusion_seed))
    return tuple(pairs)


def _is_finite(value: float) -> bool:
    import math

    return math.isfinite(float(value))
