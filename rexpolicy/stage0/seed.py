"""Deterministic, stateless seed derivation for Stage 0 workers and worlds."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .types import canonical_fingerprint


STAGE0_SEED_SCHEMA_VERSION = 1
MAX_STAGE0_SEED = (1 << 31) - 1


def _seed_integer(value: Any, name: str) -> int:
    if type(value) is not int or value < 0 or value > (1 << 63) - 1:
        raise ValueError(f"{name} must be an integer in [0, {(1 << 63) - 1}]")
    return value


def _coordinate(value: Any, index: int) -> int | str:
    if type(value) is int:
        if value < 0:
            raise ValueError(f"seed coordinate {index} cannot be negative")
        return value
    if isinstance(value, str) and value:
        return value
    raise ValueError(f"seed coordinate {index} must be a non-negative int or text")


def derive_stage0_seed(
    base_seed: int,
    *coordinates: int | str,
    namespace: str = "stage0",
) -> int:
    """Derive a portable 31-bit seed without mutable RNG state."""
    _seed_integer(base_seed, "base_seed")
    if not isinstance(namespace, str) or not namespace:
        raise ValueError("namespace must be non-empty text")
    normalized = [_coordinate(value, index) for index, value in enumerate(coordinates)]
    digest = canonical_fingerprint(
        {
            "schema_version": STAGE0_SEED_SCHEMA_VERSION,
            "namespace": namespace,
            "base_seed": base_seed,
            "coordinates": normalized,
        }
    )
    return int(digest[:16], 16) % (MAX_STAGE0_SEED + 1)


@dataclass(frozen=True)
class Stage0SeedSequence:
    """A replayable hierarchy for rank, world, and episode seeds."""

    base_seed: int
    namespace: str = "stage0"

    def __post_init__(self) -> None:
        _seed_integer(self.base_seed, "base_seed")
        if not isinstance(self.namespace, str) or not self.namespace:
            raise ValueError("namespace must be non-empty text")

    def derive(self, *coordinates: int | str) -> int:
        return derive_stage0_seed(
            self.base_seed,
            *coordinates,
            namespace=self.namespace,
        )

    def rank(self, rank: int) -> int:
        return self.derive("rank", rank)

    def world(self, rank: int, world: int) -> int:
        return self.derive("rank", rank, "world", world)

    def episode(self, rank: int, world: int, episode: int) -> int:
        return self.derive(
            "rank",
            rank,
            "world",
            world,
            "episode",
            episode,
        )

    def to_record(self) -> dict[str, Any]:
        return {
            "schema_version": STAGE0_SEED_SCHEMA_VERSION,
            "base_seed": self.base_seed,
            "namespace": self.namespace,
        }

    @property
    def fingerprint(self) -> str:
        return canonical_fingerprint(self.to_record())


__all__ = [
    "MAX_STAGE0_SEED",
    "STAGE0_SEED_SCHEMA_VERSION",
    "Stage0SeedSequence",
    "derive_stage0_seed",
]
