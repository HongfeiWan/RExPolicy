"""Deterministic per-rank and per-world reset recipe generation."""

from __future__ import annotations

from dataclasses import dataclass, field


_UINT64_MASK = (1 << 64) - 1


def _splitmix64(value: int) -> int:
    value = (value + 0x9E3779B97F4A7C15) & _UINT64_MASK
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & _UINT64_MASK
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & _UINT64_MASK
    return (value ^ (value >> 31)) & _UINT64_MASK


@dataclass
class Stage0ResetSampler:
    """Generate stable reset seeds without sharing streams across DDP ranks."""

    base_seed: int
    num_envs: int
    rank: int = 0
    _episodes: list[int] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.num_envs < 1:
            raise ValueError("num_envs must be positive")
        if self.rank < 0:
            raise ValueError("rank cannot be negative")
        self._episodes = [0] * self.num_envs

    def next(self, worlds: list[int] | None = None) -> list[int]:
        selected = list(range(self.num_envs)) if worlds is None else list(worlds)
        if len(set(selected)) != len(selected):
            raise ValueError("worlds must not contain duplicates")
        seeds = [0] * self.num_envs
        for world in range(self.num_envs):
            episode = self._episodes[world]
            mixed = _splitmix64(int(self.base_seed) & _UINT64_MASK)
            mixed = _splitmix64(mixed ^ int(self.rank))
            mixed = _splitmix64(mixed ^ int(world))
            mixed = _splitmix64(mixed ^ int(episode))
            seeds[world] = int(mixed & 0x7FFFFFFF)
        for world in selected:
            if world < 0 or world >= self.num_envs:
                raise IndexError(f"world index {world} outside [0, {self.num_envs})")
            self._episodes[world] += 1
        return seeds

    def state_dict(self) -> dict[str, object]:
        return {
            "base_seed": int(self.base_seed),
            "num_envs": self.num_envs,
            "rank": self.rank,
            "episodes": list(self._episodes),
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        expected = {"base_seed", "num_envs", "rank", "episodes"}
        if set(state) != expected:
            raise ValueError(f"Reset sampler state keys must equal {sorted(expected)}")
        if (
            int(state["base_seed"]) != int(self.base_seed)
            or int(state["num_envs"]) != self.num_envs
        ):
            raise ValueError("Reset sampler identity mismatch")
        if int(state["rank"]) != self.rank:
            raise ValueError("Reset sampler rank mismatch")
        episodes = state["episodes"]
        if not isinstance(episodes, list) or len(episodes) != self.num_envs:
            raise ValueError("Reset sampler episodes must match num_envs")
        if any(not isinstance(value, int) or value < 0 for value in episodes):
            raise ValueError(
                "Reset sampler episode counters must be nonnegative integers"
            )
        self._episodes = list(episodes)


__all__ = ["Stage0ResetSampler"]
