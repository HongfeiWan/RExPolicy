"""Immutable content-addressed storage for rebuildable flywheel sidecars."""

from __future__ import annotations

import os
import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Union

from rexpolicy.tasking.process_label_view import ProcessLabelView
from rexpolicy.tasking.process_reward import ProcessRewardShadowView
from rexpolicy.tasking.reward_view import RewardView

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_ARTIFACT_DIRECTORIES = {
    "rexpolicy_process_label_view": "process-labels",
    "rexpolicy_process_reward_shadow_view": "process-reward-shadow",
    "rexpolicy_reward_view": "rewards",
}

DerivedView = Union[ProcessLabelView, ProcessRewardShadowView, RewardView]


@dataclass(frozen=True)
class DerivedViewWrite:
    artifact_type: str
    artifact_sha256: str
    event_ledger_sha256: str
    relative_path: str
    created: bool


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _publish_immutable(path: Path, payload: bytes) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise FileExistsError(
                f"Content-addressed derived artifact does not match {path}"
            )
        return False
    temporary = path.with_name(f".{path.name}.partial-{uuid.uuid4().hex}")
    try:
        with temporary.open("xb") as file:
            file.write(payload)
            file.flush()
            os.fsync(file.fileno())
        try:
            os.link(temporary, path)
            created = True
        except FileExistsError:
            if path.read_bytes() != payload:
                raise FileExistsError(
                    "Concurrent derived artifact has different content"
                )
            created = False
        _fsync_directory(path.parent)
        return created
    finally:
        temporary.unlink(missing_ok=True)


def write_derived_view(
    *,
    run_dir: Path,
    view: DerivedView,
) -> DerivedViewWrite:
    """Publish one derived view without changing the source archive pair."""
    try:
        directory = _ARTIFACT_DIRECTORIES[view.artifact_type]
    except KeyError as error:
        raise ValueError("Unsupported derived view artifact type") from error
    artifact_sha256 = view.fingerprint
    if _SHA256.fullmatch(artifact_sha256) is None:
        raise ValueError("Derived view fingerprint is invalid")
    if _SHA256.fullmatch(view.event_ledger_sha256) is None:
        raise ValueError("Derived view Event Ledger fingerprint is invalid")
    relative = (
        Path("archive")
        / "derived"
        / directory
        / artifact_sha256[:2]
        / f"{artifact_sha256}.json"
    )
    payload = (view.to_json() + "\n").encode("ascii")
    created = _publish_immutable(Path(run_dir) / relative, payload)
    return DerivedViewWrite(
        artifact_type=view.artifact_type,
        artifact_sha256=artifact_sha256,
        event_ledger_sha256=view.event_ledger_sha256,
        relative_path=relative.as_posix(),
        created=created,
    )
