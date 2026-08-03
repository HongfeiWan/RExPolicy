"""Typed admission boundary for replayed Success Archive candidates."""

from __future__ import annotations

from dataclasses import dataclass

from .replay import (
    CandidateReplayMismatch,
    CandidateReplayResult,
    validate_candidate_replay,
)
from .success_archive import SuccessArchive, SuccessReference

REPLAY_MISMATCH_REASON = "candidate_replay_mismatch"


@dataclass(frozen=True)
class ReplayAdmission:
    """Whether one historical candidate remains eligible for training."""

    accepted: bool
    quarantine_created: bool
    reason_code: str | None = None
    detail: str | None = None


def admit_candidate_replay(
    *,
    archive: SuccessArchive,
    reference: SuccessReference,
    archived: dict,
    replayed: CandidateReplayResult,
    detected_generation: int,
    action_tolerance: float,
    reward_tolerance: float,
) -> ReplayAdmission:
    """Quarantine only a typed candidate mismatch; propagate all hard errors."""
    try:
        validate_candidate_replay(
            archived,
            replayed,
            action_tolerance=action_tolerance,
            reward_tolerance=reward_tolerance,
        )
    except CandidateReplayMismatch as error:
        detail = str(error)
        created = archive.quarantine(
            reference,
            reason_code=REPLAY_MISMATCH_REASON,
            detail=detail,
            detected_generation=detected_generation,
        )
        return ReplayAdmission(
            accepted=False,
            quarantine_created=created,
            reason_code=REPLAY_MISMATCH_REASON,
            detail=detail,
        )
    return ReplayAdmission(accepted=True, quarantine_created=False)
