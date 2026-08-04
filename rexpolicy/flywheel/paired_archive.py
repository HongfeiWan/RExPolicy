"""Atomic-commit descriptor for aligned legacy episode/Event Ledger shards."""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rexpolicy.tasking.event_ledger import EpisodeEventLedger

from .checkpoint import atomic_write_json, file_sha256
from .experience import EpisodeExperience


@dataclass(frozen=True)
class PairedArchiveWrite:
    episode_path: str
    episode_sha256: str
    event_path: str
    event_sha256: str
    pair_descriptor_path: str
    record_count: int


@dataclass(frozen=True)
class CollectedEpisode:
    experience: EpisodeExperience
    event_ledger: EpisodeEventLedger


def _atomic_write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite archive shard {path}")
    temporary = path.with_name(f".{path.name}.partial-{uuid.uuid4().hex}")
    try:
        with temporary.open("x", encoding="utf-8") as file:
            for record in records:
                file.write(
                    json.dumps(
                        record,
                        separators=(",", ":"),
                        sort_keys=True,
                        allow_nan=False,
                    )
                    + "\n"
                )
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, path)
        descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        temporary.unlink(missing_ok=True)


def _validate_alignment(
    episodes: list[EpisodeExperience],
    ledgers: list[EpisodeEventLedger],
) -> None:
    if not episodes or len(episodes) != len(ledgers):
        raise ValueError("Episode/Event Ledger shard counts must match and be nonzero")
    for index, (episode, ledger) in enumerate(zip(episodes, ledgers)):
        collection = ledger.collection
        if (
            episode.generation != collection.data_generation
            or episode.sampling_policy_generation
            != collection.sampling_policy_generation
            or episode.rank != collection.rank
            or episode.episode != collection.episode
        ):
            raise ValueError(
                f"Episode/Event Ledger provenance mismatch at record {index}"
            )


def write_generation_archive_pair(
    *,
    run_dir: Path,
    attempt_id: str,
    generation: int,
    rank: int,
    episodes: list[EpisodeExperience],
    event_ledgers: list[EpisodeEventLedger],
) -> PairedArchiveWrite:
    """Write both shards, then atomically publish their commit descriptor."""
    _validate_alignment(episodes, event_ledgers)
    base = Path("archive") / "attempts" / attempt_id
    base /= f"generation-{generation:06d}"
    episode_relative = base / f"rank-{rank:05d}.episodes.jsonl"
    event_relative = base / f"rank-{rank:05d}.events.jsonl"
    pair_relative = base / f"rank-{rank:05d}.pair.json"
    episode_path = run_dir / episode_relative
    event_path = run_dir / event_relative
    pair_path = run_dir / pair_relative
    if any(path.exists() for path in (episode_path, event_path, pair_path)):
        raise FileExistsError("Refusing to overwrite an archive pair")

    _atomic_write_jsonl(
        episode_path,
        [episode.archive_record() for episode in episodes],
    )
    _atomic_write_jsonl(
        event_path,
        [ledger.to_record() for ledger in event_ledgers],
    )
    episode_sha256 = file_sha256(episode_path)
    event_sha256 = file_sha256(event_path)
    count = len(episodes)
    atomic_write_json(
        pair_path,
        {
            "schema_version": 1,
            "episode_shard": {
                "path": episode_relative.as_posix(),
                "sha256": episode_sha256,
                "record_count": count,
            },
            "event_shard": {
                "path": event_relative.as_posix(),
                "sha256": event_sha256,
                "record_count": count,
            },
        },
    )
    return PairedArchiveWrite(
        episode_path=episode_relative.as_posix(),
        episode_sha256=episode_sha256,
        event_path=event_relative.as_posix(),
        event_sha256=event_sha256,
        pair_descriptor_path=pair_relative.as_posix(),
        record_count=count,
    )
