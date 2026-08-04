"""Deterministic, checkpointable latent replay memory.

This module intentionally uses only the Python standard library.  The memory
is small control-plane state; tensor batching remains the caller's concern.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence


LATENT_MEMORY_SCHEMA_VERSION = 1


def _vector(
    value: Sequence[float],
    *,
    latent_dim: int,
    path: str,
) -> tuple[float, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{path} must be a numeric sequence")
    if len(value) != latent_dim:
        raise ValueError(f"{path} must contain exactly {latent_dim} values")
    result: list[float] = []
    for index, item in enumerate(value):
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise ValueError(f"{path}[{index}] must be numeric")
        number = float(item)
        if not math.isfinite(number):
            raise ValueError(f"{path}[{index}] must be finite")
        result.append(number)
    return tuple(result)


def _distance(left: Sequence[float], right: Sequence[float]) -> float:
    return math.sqrt(math.fsum((a - b) ** 2 for a, b in zip(left, right)))


@dataclass(frozen=True)
class LatentMemoryEntry:
    sample_id: str
    latent: tuple[float, ...]
    metadata: dict[str, Any]


@dataclass(frozen=True)
class LatentNeighbor:
    entry: LatentMemoryEntry
    distance: float


@dataclass(frozen=True)
class LatentAdmission:
    accepted: bool
    reason: str
    nearest_sample_id: str | None
    nearest_distance: float | None
    evicted_sample_id: str | None = None


class LatentMemory:
    """FIFO-capacity memory with deterministic novelty and replay order."""

    def __init__(
        self,
        *,
        latent_dim: int,
        capacity: int,
        novelty_threshold: float,
        encoder_fingerprint: str = "unbound",
    ) -> None:
        if type(latent_dim) is not int or latent_dim <= 0:
            raise ValueError("latent_dim must be a positive integer")
        if type(capacity) is not int or capacity <= 0:
            raise ValueError("capacity must be a positive integer")
        if (
            isinstance(novelty_threshold, bool)
            or not isinstance(novelty_threshold, (int, float))
            or not math.isfinite(float(novelty_threshold))
            or float(novelty_threshold) < 0.0
        ):
            raise ValueError("novelty_threshold must be finite and non-negative")
        if not isinstance(encoder_fingerprint, str) or not encoder_fingerprint:
            raise ValueError("encoder_fingerprint must be a non-empty string")
        self.latent_dim = latent_dim
        self.capacity = capacity
        self.novelty_threshold = float(novelty_threshold)
        self.encoder_fingerprint = encoder_fingerprint
        self.rebase_revision = 0
        self._sample_cursor = 0
        self._entries: list[LatentMemoryEntry] = []

    def __len__(self) -> int:
        return len(self._entries)

    @property
    def entries(self) -> tuple[LatentMemoryEntry, ...]:
        return tuple(self._copy_entry(item) for item in self._entries)

    @property
    def sample_cursor(self) -> int:
        return self._sample_cursor

    @property
    def rebase_state(self) -> dict[str, Any]:
        return {
            "encoder_fingerprint": self.encoder_fingerprint,
            "rebase_revision": self.rebase_revision,
        }

    @staticmethod
    def _copy_entry(entry: LatentMemoryEntry) -> LatentMemoryEntry:
        return LatentMemoryEntry(
            sample_id=entry.sample_id,
            latent=tuple(entry.latent),
            metadata=copy.deepcopy(entry.metadata),
        )

    def nearest(
        self,
        latent: Sequence[float],
        *,
        k: int = 1,
    ) -> tuple[LatentNeighbor, ...]:
        query = _vector(latent, latent_dim=self.latent_dim, path="latent")
        if type(k) is not int or k <= 0:
            raise ValueError("k must be a positive integer")
        ranked = sorted(
            (
                (_distance(query, item.latent), item.sample_id, item)
                for item in self._entries
            ),
            key=lambda candidate: (candidate[0], candidate[1]),
        )
        return tuple(
            LatentNeighbor(
                entry=self._copy_entry(item),
                distance=distance,
            )
            for distance, _sample_id, item in ranked[:k]
        )

    def novelty(self, latent: Sequence[float]) -> float:
        neighbors = self.nearest(latent)
        return neighbors[0].distance if neighbors else math.inf

    def add(
        self,
        sample_id: str,
        latent: Sequence[float],
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> LatentAdmission:
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError("sample_id must be a non-empty string")
        if any(item.sample_id == sample_id for item in self._entries):
            return LatentAdmission(
                accepted=False,
                reason="duplicate_sample_id",
                nearest_sample_id=sample_id,
                nearest_distance=0.0,
            )
        encoded = _vector(latent, latent_dim=self.latent_dim, path="latent")
        neighbors = self.nearest(encoded)
        nearest = neighbors[0] if neighbors else None
        if nearest is not None and nearest.distance < self.novelty_threshold:
            return LatentAdmission(
                accepted=False,
                reason="below_novelty_threshold",
                nearest_sample_id=nearest.entry.sample_id,
                nearest_distance=nearest.distance,
            )
        if metadata is not None and not isinstance(metadata, Mapping):
            raise ValueError("metadata must be an object")
        entry = LatentMemoryEntry(
            sample_id=sample_id,
            latent=encoded,
            metadata=copy.deepcopy(dict(metadata or {})),
        )
        evicted: str | None = None
        if len(self._entries) == self.capacity:
            evicted = self._entries.pop(0).sample_id
            if self._sample_cursor > 0:
                self._sample_cursor -= 1
        self._entries.append(entry)
        if self._entries:
            self._sample_cursor %= len(self._entries)
        return LatentAdmission(
            accepted=True,
            reason="admitted",
            nearest_sample_id=None if nearest is None else nearest.entry.sample_id,
            nearest_distance=None if nearest is None else nearest.distance,
            evicted_sample_id=evicted,
        )

    def sample(self, count: int) -> tuple[LatentMemoryEntry, ...]:
        """Return a deterministic round-robin sample without replacement."""
        if type(count) is not int or count < 0:
            raise ValueError("count must be a non-negative integer")
        if not self._entries or count == 0:
            return ()
        actual = min(count, len(self._entries))
        indices = [
            (self._sample_cursor + offset) % len(self._entries)
            for offset in range(actual)
        ]
        self._sample_cursor = (self._sample_cursor + actual) % len(self._entries)
        return tuple(self._copy_entry(self._entries[index]) for index in indices)

    def rebase(
        self,
        latents_by_sample_id: Mapping[str, Sequence[float]],
        *,
        encoder_fingerprint: str,
    ) -> None:
        """Atomically replace every latent after an encoder checkpoint change."""
        if not isinstance(latents_by_sample_id, Mapping):
            raise ValueError("latents_by_sample_id must be an object")
        if not isinstance(encoder_fingerprint, str) or not encoder_fingerprint:
            raise ValueError("encoder_fingerprint must be a non-empty string")
        expected = {item.sample_id for item in self._entries}
        actual = set(latents_by_sample_id)
        if actual != expected:
            missing = sorted(expected.difference(actual))
            unknown = sorted(actual.difference(expected))
            raise ValueError(
                "Rebase sample ids do not match memory: "
                f"missing={missing}, unknown={unknown}"
            )
        # Validate the complete replacement before mutating live state.
        replacements = {
            sample_id: _vector(
                latents_by_sample_id[sample_id],
                latent_dim=self.latent_dim,
                path=f"latents_by_sample_id[{sample_id!r}]",
            )
            for sample_id in expected
        }
        self._entries = [
            LatentMemoryEntry(
                sample_id=item.sample_id,
                latent=replacements[item.sample_id],
                metadata=copy.deepcopy(item.metadata),
            )
            for item in self._entries
        ]
        self.encoder_fingerprint = encoder_fingerprint
        self.rebase_revision += 1

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema_version": LATENT_MEMORY_SCHEMA_VERSION,
            "latent_dim": self.latent_dim,
            "capacity": self.capacity,
            "novelty_threshold": self.novelty_threshold,
            "encoder_fingerprint": self.encoder_fingerprint,
            "rebase_revision": self.rebase_revision,
            "sample_cursor": self._sample_cursor,
            "entries": [
                {
                    "sample_id": item.sample_id,
                    "latent": list(item.latent),
                    "metadata": copy.deepcopy(item.metadata),
                }
                for item in self._entries
            ],
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if not isinstance(state, Mapping):
            raise ValueError("Latent memory state must be an object")
        expected_keys = {
            "schema_version",
            "latent_dim",
            "capacity",
            "novelty_threshold",
            "encoder_fingerprint",
            "rebase_revision",
            "sample_cursor",
            "entries",
        }
        if set(state) != expected_keys:
            raise ValueError("Latent memory state has missing or unknown fields")
        if state["schema_version"] != LATENT_MEMORY_SCHEMA_VERSION:
            raise ValueError("Unsupported latent memory state schema")
        if state["latent_dim"] != self.latent_dim:
            raise ValueError("Latent memory state latent_dim mismatch")
        if state["capacity"] != self.capacity:
            raise ValueError("Latent memory state capacity mismatch")
        if float(state["novelty_threshold"]) != self.novelty_threshold:
            raise ValueError("Latent memory state novelty_threshold mismatch")
        raw_entries = state["entries"]
        if not isinstance(raw_entries, list) or len(raw_entries) > self.capacity:
            raise ValueError("Latent memory state entries are invalid")
        restored: list[LatentMemoryEntry] = []
        seen: set[str] = set()
        for index, raw in enumerate(raw_entries):
            if not isinstance(raw, Mapping) or set(raw) != {
                "sample_id",
                "latent",
                "metadata",
            }:
                raise ValueError(f"entries[{index}] is invalid")
            sample_id = raw["sample_id"]
            if not isinstance(sample_id, str) or not sample_id or sample_id in seen:
                raise ValueError(f"entries[{index}].sample_id is invalid")
            if not isinstance(raw["metadata"], Mapping):
                raise ValueError(f"entries[{index}].metadata must be an object")
            seen.add(sample_id)
            restored.append(
                LatentMemoryEntry(
                    sample_id=sample_id,
                    latent=_vector(
                        raw["latent"],
                        latent_dim=self.latent_dim,
                        path=f"entries[{index}].latent",
                    ),
                    metadata=copy.deepcopy(dict(raw["metadata"])),
                )
            )
        revision = state["rebase_revision"]
        cursor = state["sample_cursor"]
        if type(revision) is not int or revision < 0:
            raise ValueError("rebase_revision must be a non-negative integer")
        if type(cursor) is not int or cursor < 0:
            raise ValueError("sample_cursor must be a non-negative integer")
        if restored and cursor >= len(restored):
            raise ValueError("sample_cursor is outside the restored memory")
        if not restored and cursor != 0:
            raise ValueError("Empty latent memory must have sample_cursor=0")
        fingerprint = state["encoder_fingerprint"]
        if not isinstance(fingerprint, str) or not fingerprint:
            raise ValueError("encoder_fingerprint must be a non-empty string")

        self._entries = restored
        self.encoder_fingerprint = fingerprint
        self.rebase_revision = revision
        self._sample_cursor = cursor

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> LatentMemory:
        memory = cls(
            latent_dim=state.get("latent_dim"),
            capacity=state.get("capacity"),
            novelty_threshold=state.get("novelty_threshold"),
            encoder_fingerprint=state.get("encoder_fingerprint", "unbound"),
        )
        memory.load_state_dict(state)
        return memory
