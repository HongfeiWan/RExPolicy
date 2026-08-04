"""Versioned reward-free simulator-state contract for Event Ledger v1."""

from __future__ import annotations

from typing import Any

from .canonical import canonical_fingerprint

DYNAMICS_CONTRACT_V1_ID = "groot_newton/reach_dynamics/v1"
DYNAMICS_CONTRACT_V1_TASK_FIELDS = (
    "obj_pose",
    "initial_obj_pose",
    "tcp_pose",
    "goal_pos_world",
    "reach_goal_pos_base",
    "reach_reset_xy_offset",
    "reach_eef_pos_world",
    "reach_eef_pos_base",
    "reach_distance",
    "reach_bottle_displacement",
    "reach_max_bottle_displacement",
)


def dynamics_contract_v1_record() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "dynamics_contract_id": DYNAMICS_CONTRACT_V1_ID,
        "included_groups": ["joint", "body", "control", "task"],
        "task_fields": list(DYNAMICS_CONTRACT_V1_TASK_FIELDS),
        "explicitly_excluded_semantics": [
            "reward",
            "episode_return",
            "success",
            "failure",
            "task_phase",
            "success_hold_count",
            "observation_images",
        ],
    }


DYNAMICS_CONTRACT_V1_SHA256 = canonical_fingerprint(
    dynamics_contract_v1_record()
)
