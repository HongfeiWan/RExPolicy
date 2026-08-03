"""Persistent metadata-only references to successful executed chunks."""

from __future__ import annotations

import json
import os
import re
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from .checkpoint import file_sha256
from .experience import SUCCESS_ROLES, TrainingSample

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_REASON_CODE = re.compile(r"^[a-z][a-z0-9_]*$")


@dataclass(frozen=True)
class QuarantineRecord:
    """Why one immutable success reference is no longer replay eligible."""

    schema_version: int
    sample_id: str
    reason_code: str
    detail: str
    detected_generation: int

    def validate(self) -> None:
        """Reject incomplete or unsupported quarantine records."""
        if self.schema_version != 1:
            raise ValueError(
                f"Unsupported archive quarantine schema {self.schema_version}"
            )
        if self.detected_generation < 0:
            raise ValueError("Quarantine generation cannot be negative")
        if not self.sample_id:
            raise ValueError("Quarantine sample_id cannot be empty")
        if _REASON_CODE.fullmatch(self.reason_code) is None:
            raise ValueError(f"Invalid quarantine reason code {self.reason_code!r}")
        if not self.detail.strip():
            raise ValueError("Quarantine records require detail")

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> QuarantineRecord:
        """Validate and decode one checkpointed quarantine record."""
        quarantine = cls(**record)
        quarantine.validate()
        return quarantine


@dataclass(frozen=True)
class SuccessReference:
    """A replay recipe for one direct or successful-path action chunk."""

    schema_version: int
    sample_id: str
    source_generation: int
    source_rank: int
    source_episode: int
    decision: int
    world: int
    success_roles: tuple[str, ...]
    episode_path: str
    episode_sha256: str
    episode_record_index: int
    archived_weight: float
    task_id: str
    reward_profile_id: str
    simulator_fingerprint: str

    @classmethod
    def from_sample(
        cls,
        sample: TrainingSample,
        *,
        episode_path: str,
        episode_sha256: str,
        episode_record_index: int,
        simulator_fingerprint: str,
    ) -> SuccessReference:
        """Build an immutable reference from a transient training sample."""
        if not sample.sample_id or not sample.success_roles:
            raise ValueError("Success references require an identified success sample")
        return cls(
            schema_version=2,
            sample_id=sample.sample_id,
            source_generation=sample.generation,
            source_rank=sample.rank,
            source_episode=sample.episode,
            decision=sample.decision,
            world=sample.world,
            success_roles=tuple(sample.success_roles),
            episode_path=episode_path,
            episode_sha256=str(episode_sha256),
            episode_record_index=int(episode_record_index),
            archived_weight=float(sample.sample_weight),
            task_id=sample.task_id,
            reward_profile_id=sample.reward_profile_id,
            simulator_fingerprint=simulator_fingerprint,
        )

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> SuccessReference:
        """Validate and decode one JSONL reference."""
        values = dict(record)
        values["success_roles"] = tuple(values.get("success_roles", ()))
        reference = cls(**values)
        reference.validate()
        return reference

    def validate(self) -> None:
        """Reject corrupt or semantically incomplete references."""
        if self.schema_version != 2:
            raise ValueError(
                f"Unsupported Success Archive schema {self.schema_version}"
            )
        if not self.sample_id:
            raise ValueError("Success reference sample_id cannot be empty")
        if self.source_generation < 0 or min(
            self.source_rank,
            self.source_episode,
            self.decision,
            self.world,
            self.episode_record_index,
        ) < 0:
            raise ValueError(f"Invalid success provenance for {self.sample_id}")
        if not self.success_roles or any(
            role not in SUCCESS_ROLES for role in self.success_roles
        ):
            raise ValueError(
                f"Invalid success roles for {self.sample_id}: {self.success_roles}"
            )
        if self.archived_weight <= 0.0:
            raise ValueError(
                f"Success reference {self.sample_id} has non-positive weight"
            )
        if Path(self.episode_path).is_absolute():
            raise ValueError("Success reference episode paths must be run-relative")
        if _SHA256.fullmatch(self.episode_sha256) is None:
            raise ValueError(
                f"Success reference {self.sample_id} has an invalid episode hash"
            )
        if not self.task_id or not self.reward_profile_id:
            raise ValueError("Success references require task and reward profile IDs")

    def to_record(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        record = asdict(self)
        record["success_roles"] = list(self.success_roles)
        return record


def _atomic_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    """Create one immutable JSONL shard on the same filesystem."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite archive shard {path}")
    temporary = path.with_name(f".{path.name}.partial-{uuid.uuid4().hex}")
    try:
        with temporary.open("x", encoding="utf-8") as file:
            for record in records:
                file.write(json.dumps(record, separators=(",", ":")) + "\n")
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary.exists():
            temporary.unlink()


class SuccessArchive:
    """Per-rank append-only archive with deterministic round-robin replay."""

    def __init__(
        self,
        *,
        run_dir: Path,
        rank: int,
        attempt_id: str,
        state: dict[str, Any] | None = None,
    ) -> None:
        self.run_dir = run_dir.expanduser().resolve()
        self.rank = int(rank)
        self.attempt_id = str(attempt_id)
        self._shards: list[dict[str, Any]] = []
        self._references: list[SuccessReference] = []
        self._reference_ids: set[str] = set()
        self._quarantined: dict[str, QuarantineRecord] = {}
        self.cursor = 0
        if state is not None:
            self.load_state_dict(state)

    @property
    def size(self) -> int:
        """Number of unique successful chunks retained by this rank."""
        return len(self._references)

    @property
    def shards(self) -> tuple[str, ...]:
        """Committed immutable shard paths, relative to the run directory."""
        return tuple(str(shard["path"]) for shard in self._shards)

    @property
    def eligible_size(self) -> int:
        """Number of retained references currently eligible for replay."""
        return self.size - self.quarantined_size

    @property
    def quarantined_size(self) -> int:
        """Number of retained references excluded after replay mismatch."""
        return len(self._quarantined)

    @property
    def quarantined(self) -> tuple[QuarantineRecord, ...]:
        """Stable quarantine audit records ordered by sample identity."""
        return tuple(self._quarantined[key] for key in sorted(self._quarantined))

    def quarantine(
        self,
        reference: SuccessReference,
        *,
        reason_code: str,
        detail: str,
        detected_generation: int,
    ) -> bool:
        """Exclude a replay mismatch without altering its source shard."""
        if reference.sample_id not in self._reference_ids:
            raise KeyError(f"Unknown success sample ID {reference.sample_id}")
        retained = next(
            item
            for item in self._references
            if item.sample_id == reference.sample_id
        )
        if retained != reference:
            raise ValueError(
                f"Quarantine reference changed for {reference.sample_id}"
            )
        quarantine = QuarantineRecord(
            schema_version=1,
            sample_id=reference.sample_id,
            reason_code=str(reason_code),
            detail=str(detail),
            detected_generation=int(detected_generation),
        )
        quarantine.validate()
        if quarantine.detected_generation < reference.source_generation:
            raise ValueError(
                "Quarantine generation precedes the source generation"
            )
        existing = self._quarantined.get(reference.sample_id)
        if existing is not None:
            if existing == quarantine:
                return False
            raise ValueError(
                f"Conflicting quarantine for {reference.sample_id}"
            )
        self._quarantined[reference.sample_id] = quarantine
        return True

    def write_generation(
        self,
        *,
        generation: int,
        references: list[SuccessReference],
    ) -> str | None:
        """Write a pending immutable shard without making it replay eligible."""
        if not references:
            return None
        seen = set()
        for reference in references:
            reference.validate()
            if reference.source_rank != self.rank:
                raise ValueError(
                    f"Rank {self.rank} cannot own rank {reference.source_rank} success"
                )
            if reference.source_generation != generation:
                raise ValueError(
                    f"Reference generation mismatch for {reference.sample_id}"
                )
            if reference.sample_id in seen or reference.sample_id in self._reference_ids:
                raise ValueError(f"Duplicate success sample ID {reference.sample_id}")
            seen.add(reference.sample_id)
        relative = Path("archive") / "attempts" / self.attempt_id
        relative /= f"generation-{generation:06d}"
        relative /= f"rank-{self.rank:05d}.success.jsonl"
        _atomic_jsonl(
            self.run_dir / relative,
            (reference.to_record() for reference in references),
        )
        return relative.as_posix()

    def commit_shard(self, relative_path: str) -> None:
        """Make a successfully updated generation eligible for later replay."""
        if relative_path in self.shards:
            raise ValueError(f"Success shard is already committed: {relative_path}")
        references = self._read_shard(relative_path)
        if not references:
            raise ValueError(f"Success Archive shard is unexpectedly empty: {relative_path}")
        for reference in references:
            if reference.sample_id in self._reference_ids:
                raise ValueError(f"Duplicate success sample ID {reference.sample_id}")
        absolute = self.run_dir / relative_path
        self._shards.append(
            {
                "path": relative_path,
                "record_count": len(references),
                "sha256": file_sha256(absolute),
            }
        )
        self._references.extend(references)
        self._reference_ids.update(reference.sample_id for reference in references)

    def plan(
        self,
        *,
        replay_per_rank: int,
        max_generation_exclusive: int,
    ) -> list[SuccessReference]:
        """Return unique historical references from a circular append-only list."""
        if replay_per_rank < 0:
            raise ValueError("replay_per_rank cannot be negative")
        eligible = sorted(
            (
                reference
                for reference in self._references
                if reference.source_generation < max_generation_exclusive
                and reference.sample_id not in self._quarantined
            ),
            key=lambda reference: (
                reference.source_generation,
                reference.source_rank,
                reference.source_episode,
                reference.decision,
                reference.world,
            ),
        )
        if not eligible or replay_per_rank == 0:
            return []
        count = min(replay_per_rank, len(eligible))
        start = self.cursor % len(eligible)
        chosen = [eligible[(start + offset) % len(eligible)] for offset in range(count)]
        self.cursor = (start + count) % len(eligible)
        return chosen

    def state_dict(self) -> dict[str, Any]:
        """Return the restart state; shards remain the source of truth."""
        return {
            "schema_version": 3,
            "rank": self.rank,
            "cursor": self.cursor,
            "shards": [dict(shard) for shard in self._shards],
            "quarantined": [asdict(record) for record in self.quarantined],
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore and strictly validate every committed shard."""
        schema_version = int(state.get("schema_version", -1))
        if schema_version not in (2, 3):
            raise ValueError("Unsupported Success Archive state schema")
        if schema_version == 3 and "quarantined" not in state:
            raise ValueError("Success Archive v3 state is missing quarantined records")
        if int(state.get("rank", -1)) != self.rank:
            raise ValueError("Success Archive rank does not match resume rank")
        shards = [dict(shard) for shard in state.get("shards", ())]
        references: list[SuccessReference] = []
        seen: set[str] = set()
        for shard in shards:
            if set(shard) != {"path", "record_count", "sha256"}:
                raise ValueError("Invalid Success Archive shard descriptor")
            path = str(shard["path"])
            absolute = self.run_dir / path
            if not absolute.is_file():
                raise FileNotFoundError(
                    f"Missing Success Archive shard {absolute}"
                )
            actual_sha256 = file_sha256(absolute)
            if actual_sha256 != shard["sha256"]:
                raise ValueError(
                    f"Success Archive checksum mismatch for {absolute}"
                )
            shard_references = self._read_shard(path)
            if len(shard_references) != int(shard["record_count"]):
                raise ValueError(
                    f"Success Archive record count mismatch for {absolute}"
                )
            if not shard_references:
                raise ValueError(
                    f"Success Archive shard is unexpectedly empty: {absolute}"
                )
            for reference in shard_references:
                if reference.sample_id in seen:
                    raise ValueError(
                        f"Duplicate success sample ID {reference.sample_id}"
                    )
                seen.add(reference.sample_id)
                references.append(reference)
        self._shards = shards
        self._references = references
        self._reference_ids = seen
        self._quarantined = {}
        quarantine_records = (
            state.get("quarantined", ()) if schema_version >= 3 else ()
        )
        previous_sample_id = None
        for record in quarantine_records:
            quarantine = QuarantineRecord.from_record(dict(record))
            if quarantine.sample_id not in self._reference_ids:
                raise ValueError(
                    "Archive quarantine references unknown sample "
                    f"{quarantine.sample_id}"
                )
            if (
                previous_sample_id is not None
                and quarantine.sample_id <= previous_sample_id
            ):
                raise ValueError(
                    "Archive quarantine records must be uniquely sorted"
                )
            reference = next(
                item
                for item in references
                if item.sample_id == quarantine.sample_id
            )
            if quarantine.detected_generation < reference.source_generation:
                raise ValueError(
                    "Quarantine generation precedes the source generation"
                )
            self._quarantined[quarantine.sample_id] = quarantine
            previous_sample_id = quarantine.sample_id
        self.cursor = int(state.get("cursor", 0))
        if self.cursor < 0:
            raise ValueError("Success Archive cursor cannot be negative")
        for reference in references:
            load_episode_record(self.run_dir, reference)

    def _read_shard(self, relative_path: str) -> list[SuccessReference]:
        path = Path(relative_path)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(f"Unsafe Success Archive path {relative_path!r}")
        absolute = self.run_dir / path
        if not absolute.is_file():
            raise FileNotFoundError(f"Missing Success Archive shard {absolute}")
        references = []
        with absolute.open(encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                try:
                    record = json.loads(line)
                    references.append(SuccessReference.from_record(record))
                except Exception as error:
                    raise ValueError(
                        f"Corrupt Success Archive record {absolute}:{line_number}"
                    ) from error
        return references


def load_episode_record(
    run_dir: Path,
    reference: SuccessReference,
) -> dict[str, Any]:
    """Load the source episode selected by one success reference."""
    relative = Path(reference.episode_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Unsafe episode path {reference.episode_path!r}")
    path = run_dir.expanduser().resolve() / relative
    if not path.is_file():
        raise FileNotFoundError(f"Missing archived episode shard {path}")
    actual_sha256 = file_sha256(path)
    if actual_sha256 != reference.episode_sha256:
        raise ValueError(
            f"Episode archive checksum mismatch for {reference.sample_id}"
        )
    with path.open(encoding="utf-8") as file:
        for index, line in enumerate(file):
            if index == reference.episode_record_index:
                record = json.loads(line)
                if (
                    int(record.get("generation", -1))
                    != reference.source_generation
                    or int(record.get("rank", -1)) != reference.source_rank
                    or int(record.get("episode", -1)) != reference.source_episode
                ):
                    raise ValueError(
                        f"Episode provenance mismatch for {reference.sample_id}"
                    )
                return record
    raise IndexError(
        f"Episode record {reference.episode_record_index} is missing from {path}"
    )
