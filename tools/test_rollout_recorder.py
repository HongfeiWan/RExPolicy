"""CPU-only tests for the read-only rollout recorder."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools.run_rollout_recorder import (
    OverlayDescriptor,
    RuntimeState,
    WorldTelemetry,
    _freeze_policy_for_inference,
    create_parser,
    execute_rollout,
    resolve_dit_overlay,
    video_paths,
)


def _status(
    step: int,
    world: int,
    *,
    terminated: bool = False,
    safety_failure: bool = False,
) -> WorldTelemetry:
    return WorldTelemetry(
        step=step,
        distance_m=0.20 - 0.01 * step - 0.001 * world,
        success=terminated and not safety_failure,
        safety_failure=safety_failure,
        terminated=terminated,
        truncated=False,
    )


class _FakeRuntime:
    num_worlds = 2

    def __init__(self) -> None:
        self.step = 0
        self.sample_calls = 0
        self.executed_active: list[tuple[bool, ...]] = []
        self.closed = False
        self.update_calls = 0
        self.archive_calls = 0

    def reset(self, reset_seed: int) -> RuntimeState:
        if reset_seed != 17:
            raise AssertionError("unexpected reset seed")
        return RuntimeState(
            observation={"frame": 0},
            telemetry=(_status(0, 0), _status(0, 1)),
        )

    def sample_action_chunk(self, observation: object) -> tuple[object, ...]:
        del observation
        self.sample_calls += 1
        return tuple(object() for _ in range(8))

    def execute_action(
        self,
        action_batch: object,
        active: tuple[bool, ...],
    ) -> RuntimeState:
        del action_batch
        self.executed_active.append(active)
        self.step += 1
        return RuntimeState(
            observation={"frame": self.step},
            telemetry=(
                _status(self.step, 0, terminated=self.step >= 3),
                _status(
                    self.step,
                    1,
                    terminated=self.step >= 5,
                    safety_failure=self.step == 4,
                ),
            ),
        )

    def update(self) -> None:
        self.update_calls += 1
        raise AssertionError("read-only execution attempted an update")

    def write_archive(self) -> None:
        self.archive_calls += 1
        raise AssertionError("read-only execution attempted an archive write")

    def close(self) -> None:
        self.closed = True


class _FakeSink:
    def __init__(self) -> None:
        self.frames: list[tuple[object, tuple[WorldTelemetry, ...]]] = []
        self.closed = False

    def write(
        self,
        observation: object,
        telemetry: tuple[WorldTelemetry, ...],
    ) -> None:
        self.frames.append((observation, telemetry))

    def close(self) -> None:
        self.closed = True


class _FakeModel:
    def __init__(self) -> None:
        self.requires_grad_value = True
        self.training = True

    def requires_grad_(self, value: bool) -> "_FakeModel":
        self.requires_grad_value = value
        return self

    def eval(self) -> "_FakeModel":
        self.training = False
        return self


class TestRolloutExecution(unittest.TestCase):
    def test_cli_exposes_no_training_or_archive_switches(self) -> None:
        parser = create_parser()
        option_strings = {
            option
            for action in parser._actions
            for option in action.option_strings
        }
        forbidden = {
            "--run-dir",
            "--save",
            "--resume-latest",
            "--resume-from",
            "--train-steps-per-generation",
            "--learning-rate",
            "--success-replay-per-rank",
        }
        self.assertFalse(option_strings & forbidden)

    def test_horizon_two_stops_when_all_worlds_are_done(self) -> None:
        runtime = _FakeRuntime()
        sink = _FakeSink()

        result = execute_rollout(
            runtime=runtime,
            frame_sink=sink,
            reset_seed=17,
            episode_control_steps=8,
            execution_horizon=2,
        )

        self.assertEqual(result.control_steps, 5)
        self.assertEqual(result.policy_decisions, 3)
        self.assertEqual(runtime.sample_calls, 3)
        self.assertEqual(len(sink.frames), 6)  # reset frame + five steps
        self.assertEqual(runtime.executed_active[3], (False, True))
        self.assertTrue(result.final_telemetry[0].success)
        self.assertTrue(result.final_telemetry[1].success)
        self.assertTrue(result.final_telemetry[1].safety_failure)
        self.assertEqual(runtime.update_calls, 0)
        self.assertEqual(runtime.archive_calls, 0)

    def test_inference_freeze_disables_all_model_gradients(self) -> None:
        model = _FakeModel()
        policy = type("Policy", (), {"model": model})()

        _freeze_policy_for_inference(policy)

        self.assertFalse(model.requires_grad_value)
        self.assertFalse(model.training)

    def test_only_supported_execution_horizons_are_accepted(self) -> None:
        runtime = _FakeRuntime()
        sink = _FakeSink()
        with self.assertRaisesRegex(ValueError, "one of 1, 2, 4, or 8"):
            execute_rollout(
                runtime=runtime,
                frame_sink=sink,
                reset_seed=17,
                episode_control_steps=8,
                execution_horizon=3,
            )


class TestRolloutArtifacts(unittest.TestCase):
    def test_video_layout_is_flat_for_k1_and_partitioned_for_small_k(self) -> None:
        output = Path("/tmp/rollout")
        single = video_paths(output, 1)
        self.assertEqual(single[0]["ego"], output / "ego.mp4")
        self.assertEqual(single[0]["wrist"], output / "wrist.mp4")
        self.assertEqual(
            single[0]["side_by_side"],
            output / "side_by_side.mp4",
        )

        multiple = video_paths(output, 2)
        self.assertEqual(
            multiple[1]["side_by_side"],
            output / "world-001" / "side_by_side.mp4",
        )

    def test_standalone_overlay_is_read_and_fingerprinted(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            overlay = Path(temporary)
            (overlay / "model.safetensors").write_bytes(b"read-only-weights")

            descriptor = resolve_dit_overlay(overlay)

        self.assertIsInstance(descriptor, OverlayDescriptor)
        self.assertEqual(descriptor.kind, "standalone_dit_overlay")
        self.assertEqual(descriptor.model_file, "model.safetensors")
        self.assertIsNone(descriptor.completed_generation)

    def test_generation_overlay_uses_full_checkpoint_verification(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            generation = (
                Path(temporary)
                / "checkpoints"
                / "generation-000007"
            )
            generation.mkdir(parents=True)
            (generation / "model.safetensors").write_bytes(b"weights")
            verified = {"completed_generation": 7, "accepted": False}
            with patch(
                "rexpolicy.flywheel.checkpoint.CheckpointManager.verify",
                return_value=verified,
            ) as verify:
                descriptor = resolve_dit_overlay(generation)

        verify.assert_called_once_with(generation.resolve())
        self.assertEqual(descriptor.kind, "verified_flywheel_generation")
        self.assertEqual(descriptor.completed_generation, 7)
        self.assertFalse(descriptor.accepted)

    def test_partial_generation_is_rejected_before_model_load(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            partial = Path(temporary) / ".partial-generation-000007"
            partial.mkdir()
            (partial / "model.safetensors").write_bytes(b"weights")
            with self.assertRaisesRegex(ValueError, "partial checkpoint"):
                resolve_dit_overlay(partial)


if __name__ == "__main__":
    unittest.main()
