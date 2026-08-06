"""CPU tests for the reward-free Grasp-Lift authoring controller."""

from __future__ import annotations

import unittest

import torch

from rexpolicy.stage0.envs.grasp_lift_modes import (
    DEFAULT_GRASP_LIFT_AUTHORING_MODE,
    GRASP_LIFT_AUTHORING_MODE_CLOSE4,
    GraspLiftAuthoringController,
    GraspLiftAuthoringModeSpec,
    GraspLiftAuthoringPhase,
)
from rexpolicy.stage0.envs.grasp_lift_oracle import (
    GraspLiftOracleResult,
    GraspLiftTransientEvents,
)
from rexpolicy.stage0.envs.state_schema import DEFAULT_STAGE0_STATE_SCHEMA


SCHEMA = DEFAULT_STAGE0_STATE_SCHEMA


def _states(batch_size: int = 1) -> torch.Tensor:
    state = torch.zeros(batch_size, SCHEMA.dimension, dtype=torch.float32)
    state[:, SCHEMA.slice("eef_position")] = torch.tensor((0.10, -0.20, 0.30))
    state[:, SCHEMA.slice("object_position")] = torch.tensor((0.40, 0.00, 0.10))
    state[:, SCHEMA.slice("goal_position")] = torch.tensor((0.40, 0.00, 0.18))
    state[:, SCHEMA.slice("object_to_goal")] = torch.tensor((0.00, 0.00, 0.08))
    state[:, SCHEMA.slice("object_rotation_6d")] = torch.tensor(
        (1.0, 0.0, 0.0, 0.0, 1.0, 0.0)
    )
    state[:, SCHEMA.slice("hand_joint_position")] = torch.linspace(
        0.01, 0.10, 10
    )
    return state


def _reference(state: torch.Tensor) -> torch.Tensor:
    action = torch.zeros(state.shape[0], 19, dtype=state.dtype)
    action[:, :3] = state[:, SCHEMA.slice("eef_position")]
    action[:, 3:9] = torch.tensor((1.0, 0.0, 0.0, 0.0, 1.0, 0.0))
    action[:, 9:19] = state[:, SCHEMA.slice("hand_joint_position")]
    return action


def _events(
    batch_size: int,
    *,
    streak: int | tuple[int, ...] = 0,
    forbidden: bool = False,
    overflow: bool = False,
) -> GraspLiftTransientEvents:
    if isinstance(streak, tuple):
        streak_tensor = torch.tensor(streak, dtype=torch.int32)
    else:
        streak_tensor = torch.full((batch_size,), streak, dtype=torch.int32)
    return GraspLiftTransientEvents(
        forbidden_hand_contact=torch.full(
            (batch_size,), forbidden, dtype=torch.bool
        ),
        opposed_grasp_max_consecutive_physics_frames=streak_tensor,
        had_hand_contact=streak_tensor > 0,
        collision_buffer_overflow=torch.full(
            (batch_size,), overflow, dtype=torch.bool
        ),
    )


def _result(
    batch_size: int,
    *,
    opposed: bool | tuple[bool, ...] = False,
    success: bool | tuple[bool, ...] = False,
    failure: bool | tuple[bool, ...] = False,
) -> GraspLiftOracleResult:
    def booleans(value: bool | tuple[bool, ...]) -> torch.Tensor:
        if isinstance(value, tuple):
            return torch.tensor(value, dtype=torch.bool)
        return torch.full((batch_size,), value, dtype=torch.bool)

    opposed_tensor = booleans(opposed)
    success_tensor = booleans(success)
    failure_tensor = booleans(failure)
    zeros = torch.zeros(batch_size, dtype=torch.float32)
    false = torch.zeros(batch_size, dtype=torch.bool)
    counts = torch.zeros(batch_size, dtype=torch.int64)
    return GraspLiftOracleResult(
        success=success_tensor,
        failure=failure_tensor,
        grasp_confirmed=opposed_tensor.clone(),
        current_opposed_grasp=opposed_tensor,
        lift_height=zeros.clone(),
        lateral_displacement=zeros.clone(),
        bottle_tilt=zeros.clone(),
        success_hold_steps=counts.clone(),
        drop_gap_steps=counts.clone(),
        forbidden_contact_violation=false.clone(),
        lateral_displacement_violation=false.clone(),
        bottle_tilt_violation=false.clone(),
        dropped_grasp_violation=false.clone(),
        collision_buffer_overflow_violation=false.clone(),
    )


def _fast_spec(**overrides) -> GraspLiftAuthoringModeSpec:
    values = {
        "approach_control_steps": 1,
        "close_schedule_control_steps": 4,
        "minimum_tentative_close_step": 2,
        "maximum_close_control_steps": 6,
        "stable_grasp_control_steps": 2,
        "full_opposed_physics_frames": 6,
        "confirm_hold_control_steps": 2,
        "lift_step_m": 0.01,
        "lift_target_m": 0.03,
        "maximum_lift_control_steps": 4,
    }
    values.update(overrides)
    return GraspLiftAuthoringModeSpec(**values)


def _observe(
    controller: GraspLiftAuthoringController,
    state: torch.Tensor,
    proposal: torch.Tensor,
    *,
    opposed: bool | tuple[bool, ...] = False,
    streak: int | tuple[int, ...] = 0,
    success: bool | tuple[bool, ...] = False,
    failure: bool | tuple[bool, ...] = False,
    static: bool = True,
) -> None:
    batch_size = state.shape[0]
    controller.observe(
        executed_action=proposal,
        next_state=state,
        oracle_result=_result(
            batch_size,
            opposed=opposed,
            success=success,
            failure=failure,
        ),
        events=_events(batch_size, streak=streak),
        is_obj_static=torch.full((batch_size,), static, dtype=torch.bool),
        is_robot_static=torch.full((batch_size,), static, dtype=torch.bool),
    )


class GraspLiftAuthoringModesTest(unittest.TestCase):
    def test_default_mode_is_hash_bound_and_calibrated(self) -> None:
        mode = DEFAULT_GRASP_LIFT_AUTHORING_MODE
        self.assertEqual(mode.pregrasp_offset_base_m, (-0.03, 0.19, 0.07))
        self.assertEqual(mode.close_fraction, 0.65)
        self.assertEqual(mode.minimum_tentative_close_step, 7)
        self.assertEqual(
            mode.mode_id,
            "stage0/grasp_lift/thumb_index_side_close7/v1",
        )
        self.assertEqual(
            GRASP_LIFT_AUTHORING_MODE_CLOSE4.minimum_tentative_close_step,
            4,
        )
        self.assertNotEqual(mode.sha256, GRASP_LIFT_AUTHORING_MODE_CLOSE4.sha256)
        self.assertEqual(len(mode.sha256), 64)
        self.assertEqual(mode.sha256, GraspLiftAuthoringModeSpec().sha256)
        self.assertEqual(len(mode.to_record()["action_schema_sha256"]), 64)

    def test_reset_freezes_pregrasp_and_lift_axis(self) -> None:
        state = _states()
        controller = GraspLiftAuthoringController(_fast_spec())
        controller.reset(state, _reference(state))

        changed = state.clone()
        changed[:, SCHEMA.slice("object_position")] += torch.tensor(
            (0.7, -0.4, 0.2)
        )
        proposal = controller.propose(changed)

        expected = torch.tensor((0.37, 0.19, 0.17))
        self.assertTrue(torch.allclose(proposal[0, :3], expected))

    def test_close7_cannot_capture_a_step6_grasp(self) -> None:
        state = _states()
        controller = GraspLiftAuthoringController(
            _fast_spec(
                close_schedule_control_steps=8,
                minimum_tentative_close_step=7,
                maximum_close_control_steps=10,
            )
        )
        controller.reset(state, _reference(state))
        approach = controller.propose(state)
        _observe(controller, state, approach)

        for _ in range(6):
            close = controller.propose(state)
            _observe(controller, state, close, opposed=True, streak=6)
            self.assertFalse(
                controller.snapshot().has_tentative_target.item()
            )

        close7 = controller.propose(state)
        _observe(controller, state, close7, opposed=True, streak=6)
        snapshot = controller.snapshot()
        self.assertTrue(snapshot.has_tentative_target.item())
        self.assertEqual(snapshot.close_control_steps.item(), 7)

    def test_freezes_executed_target_only_after_same_target_verification(self) -> None:
        state = _states()
        controller = GraspLiftAuthoringController(_fast_spec())
        controller.reset(state, _reference(state))

        proposal = controller.propose(state)
        _observe(controller, state, proposal)
        self.assertEqual(
            controller.phase.item(), int(GraspLiftAuthoringPhase.CLOSE)
        )

        proposal = controller.propose(state)
        _observe(controller, state, proposal)
        proposal = controller.propose(state)
        executed = proposal.clone()
        executed[:, 9:19] = torch.linspace(0.20, 0.29, 10)
        controller.observe(
            executed_action=executed,
            next_state=state,
            oracle_result=_result(1, opposed=True),
            events=_events(1, streak=6),
            is_obj_static=torch.ones(1, dtype=torch.bool),
            is_robot_static=torch.ones(1, dtype=torch.bool),
        )
        snapshot = controller.snapshot()
        self.assertTrue(snapshot.has_tentative_target.item())
        self.assertFalse(snapshot.has_frozen_target.item())

        verification = controller.propose(state)
        self.assertTrue(torch.equal(verification[:, 9:19], executed[:, 9:19]))
        _observe(controller, state, verification, opposed=True, streak=6)

        snapshot = controller.snapshot()
        self.assertTrue(snapshot.has_frozen_target.item())
        self.assertEqual(
            snapshot.phase.item(), int(GraspLiftAuthoringPhase.CONFIRM_HOLD)
        )

    def test_projection_drift_restarts_same_target_verification(self) -> None:
        state = _states()
        controller = GraspLiftAuthoringController(
            _fast_spec(minimum_tentative_close_step=1)
        )
        controller.reset(state, _reference(state))
        approach = controller.propose(state)
        _observe(controller, state, approach)

        close = controller.propose(state)
        first_executed = close.clone()
        first_executed[:, 9:19] = torch.linspace(0.20, 0.29, 10)
        controller.observe(
            executed_action=first_executed,
            next_state=state,
            oracle_result=_result(1, opposed=True),
            events=_events(1, streak=6),
            is_obj_static=torch.ones(1, dtype=torch.bool),
            is_robot_static=torch.ones(1, dtype=torch.bool),
        )

        verification = controller.propose(state)
        drifted = verification.clone()
        drifted[:, 9:19] += 0.01
        controller.observe(
            executed_action=drifted,
            next_state=state,
            oracle_result=_result(1, opposed=True),
            events=_events(1, streak=6),
            is_obj_static=torch.ones(1, dtype=torch.bool),
            is_robot_static=torch.ones(1, dtype=torch.bool),
        )
        self.assertFalse(controller.snapshot().has_frozen_target.item())
        repeated = controller.propose(state)
        self.assertTrue(torch.equal(repeated[:, 9:19], drifted[:, 9:19]))
        _observe(controller, state, repeated, opposed=True, streak=6)
        self.assertTrue(controller.snapshot().has_frozen_target.item())

    def test_non_static_tentative_verification_has_a_bounded_horizon(self) -> None:
        state = _states()
        controller = GraspLiftAuthoringController(
            _fast_spec(
                close_schedule_control_steps=1,
                minimum_tentative_close_step=1,
                maximum_close_control_steps=2,
            )
        )
        controller.reset(state, _reference(state))
        approach = controller.propose(state)
        _observe(controller, state, approach)
        close = controller.propose(state)
        _observe(controller, state, close, opposed=True, streak=7)
        verification = controller.propose(state)
        _observe(
            controller,
            state,
            verification,
            opposed=True,
            streak=7,
            static=False,
        )
        self.assertEqual(
            controller.phase.item(), int(GraspLiftAuthoringPhase.EXHAUSTED)
        )

    def test_confirm_loss_resumes_next_close_step(self) -> None:
        state = _states()
        spec = _fast_spec(minimum_tentative_close_step=1)
        controller = GraspLiftAuthoringController(spec)
        controller.reset(state, _reference(state))
        proposal = controller.propose(state)
        _observe(controller, state, proposal)

        first_close = controller.propose(state)
        _observe(controller, state, first_close, opposed=True, streak=6)
        verification = controller.propose(state)
        _observe(controller, state, verification, opposed=True, streak=6)
        self.assertEqual(
            controller.phase.item(), int(GraspLiftAuthoringPhase.CONFIRM_HOLD)
        )

        hold = controller.propose(state)
        _observe(controller, state, hold, opposed=False, streak=0)
        self.assertEqual(
            controller.phase.item(), int(GraspLiftAuthoringPhase.CLOSE)
        )
        resumed = controller.propose(state)
        self.assertFalse(torch.equal(resumed[:, 9:19], verification[:, 9:19]))

    def test_lift_uses_reset_axis_and_frozen_hand_until_oracle_success(self) -> None:
        state = _states()
        controller = GraspLiftAuthoringController(
            _fast_spec(minimum_tentative_close_step=1)
        )
        controller.reset(state, _reference(state))
        approach = controller.propose(state)
        _observe(controller, state, approach)
        close = controller.propose(state)
        _observe(controller, state, close, opposed=True, streak=6)
        verification = controller.propose(state)
        _observe(controller, state, verification, opposed=True, streak=6)
        frozen_hand = verification[:, 9:19].clone()
        for _ in range(2):
            hold = controller.propose(state)
            _observe(controller, state, hold, opposed=True, streak=6)
        self.assertEqual(controller.phase.item(), int(GraspLiftAuthoringPhase.LIFT))

        changed = state.clone()
        changed[:, SCHEMA.slice("object_to_goal")] = torch.tensor((0.08, 0.0, 0.0))
        lift = controller.propose(changed)
        lift_start = state[:, SCHEMA.slice("eef_position")]
        self.assertTrue(
            torch.allclose(lift[:, :3] - lift_start, torch.tensor(((0.0, 0.0, 0.01),)))
        )
        self.assertTrue(torch.equal(lift[:, 9:19], frozen_hand))
        _observe(
            controller,
            changed,
            lift,
            opposed=True,
            streak=6,
            success=True,
        )
        self.assertEqual(
            controller.phase.item(), int(GraspLiftAuthoringPhase.SUCCEEDED)
        )

    def test_last_lift_step_enters_bounded_success_hold(self) -> None:
        state = _states()
        controller = GraspLiftAuthoringController(
            _fast_spec(
                minimum_tentative_close_step=1,
                maximum_lift_control_steps=3,
                maximum_success_hold_control_steps=2,
            )
        )
        controller.reset(state, _reference(state))
        approach = controller.propose(state)
        _observe(controller, state, approach)
        close = controller.propose(state)
        _observe(controller, state, close, opposed=True, streak=6)
        verification = controller.propose(state)
        _observe(controller, state, verification, opposed=True, streak=6)
        for _ in range(2):
            hold = controller.propose(state)
            _observe(controller, state, hold, opposed=True, streak=6)
        for _ in range(3):
            lift = controller.propose(state)
            _observe(controller, state, lift, opposed=True, streak=6)
        self.assertEqual(
            controller.phase.item(), int(GraspLiftAuthoringPhase.SUCCESS_HOLD)
        )
        success_hold = controller.propose(state)
        _observe(
            controller,
            state,
            success_hold,
            opposed=True,
            streak=6,
            success=True,
        )
        self.assertEqual(
            controller.phase.item(), int(GraspLiftAuthoringPhase.SUCCEEDED)
        )

    def test_worlds_advance_independently_and_oracle_failure_is_terminal(self) -> None:
        state = _states(2)
        controller = GraspLiftAuthoringController(_fast_spec())
        controller.reset(state, _reference(state))
        proposal = controller.propose(state)
        _observe(controller, state, proposal, failure=(False, True))
        self.assertTrue(
            torch.equal(
                controller.phase,
                torch.tensor(
                    (
                        int(GraspLiftAuthoringPhase.CLOSE),
                        int(GraspLiftAuthoringPhase.ORACLE_FAILED),
                    )
                ),
            )
        )

    def test_proposal_observation_pairing_and_oracle_exclusivity_fail_closed(self) -> None:
        state = _states()
        controller = GraspLiftAuthoringController(_fast_spec())
        controller.reset(state, _reference(state))
        with self.assertRaisesRegex(RuntimeError, "propose"):
            controller.observe(
                executed_action=_reference(state),
                next_state=state,
                oracle_result=_result(1),
                events=_events(1),
                is_obj_static=torch.ones(1, dtype=torch.bool),
                is_robot_static=torch.ones(1, dtype=torch.bool),
            )

        proposal = controller.propose(state)
        with self.assertRaisesRegex(RuntimeError, "pending"):
            controller.propose(state)
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            controller.observe(
                executed_action=proposal,
                next_state=state,
                oracle_result=_result(1, success=True, failure=True),
                events=_events(1),
                is_obj_static=torch.ones(1, dtype=torch.bool),
                is_robot_static=torch.ones(1, dtype=torch.bool),
            )


if __name__ == "__main__":
    unittest.main()
