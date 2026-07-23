"""Stable task contracts used by collection, replay, and evaluation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class FlywheelTaskSpec:
    """Instruction, reward identity, and trainable action dimensions."""

    task_id: str
    reward_profile_id: str
    environment_task_mode: str
    instruction: str
    action_dimension_masks: dict[str, tuple[bool, ...]]

    @property
    def fingerprint(self) -> str:
        """Return a stable hash embedded in archives and checkpoints."""
        payload = json.dumps(
            asdict(self),
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        return hashlib.sha256(payload).hexdigest()


REACH_GREEN_CAP_V1 = FlywheelTaskSpec(
    task_id="reach_green_cap/v1",
    reward_profile_id="reach_progress/v1",
    environment_task_mode="reach_green_cap",
    instruction=(
        "move the open right hand to the safe pre-grasp position beside the "
        "green bottle cap without moving the bottle"
    ),
    action_dimension_masks={
        "eef_9d": (True, True, True, False, False, False, False, False, False),
        "hand_joint_target": (False,) * 10,
        "arm_joint_target": (False,) * 7,
    },
)


TASK_SPECS = {
    REACH_GREEN_CAP_V1.task_id: REACH_GREEN_CAP_V1,
}


def get_task_spec(task_id: str) -> FlywheelTaskSpec:
    """Resolve one versioned task contract."""
    try:
        return TASK_SPECS[task_id]
    except KeyError as error:
        raise ValueError(
            f"Unsupported flywheel task {task_id!r}; expected one of "
            f"{sorted(TASK_SPECS)}"
        ) from error
