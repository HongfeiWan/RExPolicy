from __future__ import annotations

import unittest

import torch

from rexpolicy.stage0.envs.oracles import ReachSuccessOracle
from rexpolicy.stage0.envs.reset_sampler import Stage0ResetSampler
from rexpolicy.stage0.envs.state_schema import DEFAULT_STAGE0_STATE_SCHEMA


def _state(
    *, distance: float, displacement: float = 0.0, contact: bool = False
) -> torch.Tensor:
    state = torch.zeros(1, DEFAULT_STAGE0_STATE_SCHEMA.dimension)
    state[:, DEFAULT_STAGE0_STATE_SCHEMA.slice("eef_position")] = torch.tensor(
        [[distance, 0.0, 0.0]]
    )
    state[:, DEFAULT_STAGE0_STATE_SCHEMA.slice("object_position")] = torch.tensor(
        [[displacement, 0.0, 0.0]]
    )
    state[:, DEFAULT_STAGE0_STATE_SCHEMA.slice("has_hand_contact")] = float(contact)
    return state


class ReachSuccessOracleTest(unittest.TestCase):
    def test_requires_hold_without_contact_or_displacement(self) -> None:
        oracle = ReachSuccessOracle(1, success_threshold_m=0.04, success_hold_steps=2)
        initial = _state(distance=0.2)
        oracle.reset(initial)
        first = oracle.evaluate(_state(distance=0.03))
        second = oracle.evaluate(_state(distance=0.02))
        self.assertFalse(first.success.item())
        self.assertTrue(second.success.item())
        self.assertFalse(second.failure.item())

    def test_contact_and_displacement_fail_closed(self) -> None:
        oracle = ReachSuccessOracle(1, success_hold_steps=1, displacement_limit_m=0.01)
        oracle.reset(_state(distance=0.2))
        contact = oracle.evaluate(_state(distance=0.01, contact=True))
        self.assertTrue(contact.failure.item())
        self.assertFalse(contact.success.item())
        oracle.reset(_state(distance=0.2))
        moved = oracle.evaluate(_state(distance=0.01, displacement=0.02))
        self.assertTrue(moved.displacement_violation.item())
        self.assertFalse(moved.success.item())


class Stage0ResetSamplerTest(unittest.TestCase):
    def test_deterministic_resume_and_rank_separation(self) -> None:
        sampler = Stage0ResetSampler(base_seed=7, num_envs=3, rank=1)
        first = sampler.next()
        checkpoint = sampler.state_dict()
        expected = sampler.next([1])
        restored = Stage0ResetSampler(base_seed=7, num_envs=3, rank=1)
        restored.load_state_dict(checkpoint)
        self.assertEqual(restored.next([1]), expected)
        self.assertNotEqual(
            first, Stage0ResetSampler(base_seed=7, num_envs=3, rank=2).next()
        )


if __name__ == "__main__":
    unittest.main()
