#!/usr/bin/env python3
"""Record deterministic, read-only GR00T + Newton rollout videos.

This entry point intentionally owns no trainer, optimizer, archive, or
checkpoint writer.  It loads an immutable base policy plus an optional
Flow-DiT overlay, executes one fixed-seed rollout, and writes only videos and
one JSON summary under a new output directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Protocol

from rexpolicy.scene_asset import validate_scene_glb


REPO_ROOT = Path(__file__).resolve().parents[1]
_GENERATION_DIRECTORY = re.compile(r"^generation-(\d{6,})$")
_MAX_RECORDED_WORLDS = 8


def _first_existing(*paths: Path) -> Path:
    for path in paths:
        if path.exists():
            return path
    return paths[0]


DEFAULT_GROOT_ROOT = _first_existing(
    REPO_ROOT / "external" / "Isaac-GR00T",
    REPO_ROOT.parent / "Isaac-GR00T",
)
DEFAULT_CHECKPOINT = REPO_ROOT / "checkpoints" / "groot" / "checkpoint-200000"
DEFAULT_VLM = REPO_ROOT / "checkpoints" / "nvidia" / "Cosmos-Reason2-2B"


def _environment_path(name: str, fallback: Path) -> Path:
    value = os.environ.get(name)
    return Path(value).expanduser() if value else fallback


@dataclass(frozen=True)
class WorldTelemetry:
    """One world's visible status after a reset or control step."""

    step: int
    distance_m: float
    success: bool
    safety_failure: bool
    terminated: bool
    truncated: bool

    @property
    def done(self) -> bool:
        return self.terminated or self.truncated


@dataclass(frozen=True)
class RuntimeState:
    """Observation plus one telemetry row per world."""

    observation: Any
    telemetry: tuple[WorldTelemetry, ...]


@dataclass(frozen=True)
class RolloutResult:
    """Small, JSON-safe execution result retained beside the videos."""

    control_steps: int
    policy_decisions: int
    telemetry_history: tuple[tuple[WorldTelemetry, ...], ...]

    @property
    def final_telemetry(self) -> tuple[WorldTelemetry, ...]:
        return self.telemetry_history[-1]


@dataclass(frozen=True)
class OverlayDescriptor:
    """Verified read-only identity of an optional DiT overlay."""

    path: str
    kind: str
    model_file: str
    model_sha256: str
    completed_generation: int | None
    accepted: bool | None


class ReadOnlyRolloutRuntime(Protocol):
    """Narrow interface that cannot expose training or persistence calls."""

    num_worlds: int

    def reset(self, reset_seed: int) -> RuntimeState: ...

    def sample_action_chunk(self, observation: Any) -> tuple[Any, ...]: ...

    def execute_action(
        self,
        action_batch: Any,
        active: tuple[bool, ...],
    ) -> RuntimeState: ...

    def close(self) -> None: ...


class RolloutFrameSink(Protocol):
    """Consumer for reset and post-control-step frames."""

    def write(
        self,
        observation: Any,
        telemetry: tuple[WorldTelemetry, ...],
    ) -> None: ...

    def close(self) -> None: ...


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--isaac-groot-root",
        type=Path,
        default=_environment_path("ISAAC_GROOT_ROOT", DEFAULT_GROOT_ROOT),
    )
    parser.add_argument(
        "--policy-checkpoint",
        type=Path,
        default=_environment_path("GROOT_POLICY_CHECKPOINT", DEFAULT_CHECKPOINT),
    )
    parser.add_argument(
        "--vlm-model",
        type=Path,
        default=_environment_path("GROOT_VLM_MODEL", DEFAULT_VLM),
    )
    parser.add_argument(
        "--dit-overlay",
        type=Path,
        help=(
            "Optional completed flywheel generation directory or standalone "
            "DiT overlay directory."
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--task-id", default="reach_green_cap/v1")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--reset-seed", type=int, default=20260722)
    parser.add_argument("--diffusion-seed", type=int, default=20260723)
    parser.add_argument(
        "--num-envs",
        "--candidates-per-state",
        dest="num_envs",
        type=int,
        default=1,
    )
    parser.add_argument("--episode-control-steps", type=int, default=8)
    parser.add_argument(
        "--execution-horizon",
        type=int,
        choices=(1, 2, 4, 8),
        default=2,
    )
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--side-by-side-height", type=int, default=360)

    parser.add_argument(
        "--camera-textures",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--scene-visuals",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--scene-glb",
        type=Path,
        default=REPO_ROOT / "scene" / "scene.glb",
        help="Binary glTF room asset used when --scene-visuals is enabled.",
    )
    parser.add_argument(
        "--capture-graph",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--hydroelastic",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--bottle-settle-frames", type=int, default=60)
    parser.add_argument("--substeps-per-frame", type=int, default=16)
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if not 1 <= args.num_envs <= _MAX_RECORDED_WORLDS:
        raise ValueError(
            f"num-envs must be in [1, {_MAX_RECORDED_WORLDS}] for recording"
        )
    if args.episode_control_steps < 1:
        raise ValueError("episode-control-steps must be positive")
    if args.execution_horizon not in (1, 2, 4, 8):
        raise ValueError("execution-horizon must be one of 1, 2, 4, or 8")
    if not math.isfinite(args.fps) or args.fps <= 0.0:
        raise ValueError("fps must be finite and positive")
    if args.side_by_side_height < 64 or args.side_by_side_height % 2:
        raise ValueError("side-by-side-height must be an even integer >= 64")
    if args.substeps_per_frame != 16 or args.bottle_settle_frames != 60:
        raise ValueError(
            "reach_green_cap/v1 requires --substeps-per-frame 16 and "
            "--bottle-settle-frames 60"
        )
    if args.scene_visuals:
        args.scene_glb = validate_scene_glb(args.scene_glb)
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(
            "Refusing to overwrite an existing rollout output directory: "
            f"{output_dir}"
        )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_dit_overlay(path: Path | None) -> OverlayDescriptor | None:
    """Verify a generation checkpoint or identify a standalone overlay."""
    if path is None:
        return None
    directory = path.expanduser().resolve()
    if not directory.is_dir():
        raise FileNotFoundError(f"DiT overlay directory is missing: {directory}")
    if directory.name.startswith(".partial-"):
        raise ValueError(f"Refusing to load a partial checkpoint: {directory}")

    model_candidates = (
        directory / "diffusion_pytorch_model.safetensors",
        directory / "model.safetensors",
    )
    model_path = next((item for item in model_candidates if item.is_file()), None)
    if model_path is None:
        raise FileNotFoundError(f"No DiT safetensors file found in {directory}")

    match = _GENERATION_DIRECTORY.fullmatch(directory.name)
    completed_generation = None
    accepted = None
    kind = "standalone_dit_overlay"
    if match is not None:
        # CheckpointManager.verify is read-only and checks COMPLETE plus every
        # recorded checksum, including the optimizer and per-rank RNG state.
        from rexpolicy.flywheel.checkpoint import CheckpointManager

        manager = CheckpointManager(directory.parent.parent)
        state = manager.verify(directory)
        completed_generation = int(state["completed_generation"])
        accepted = bool(state["accepted"])
        kind = "verified_flywheel_generation"

    return OverlayDescriptor(
        path=str(directory),
        kind=kind,
        model_file=model_path.name,
        model_sha256=_file_sha256(model_path),
        completed_generation=completed_generation,
        accepted=accepted,
    )


def _freeze_policy_for_inference(policy: Any) -> None:
    """Make accidental gradient creation impossible after overlay loading."""
    policy.model.requires_grad_(False)
    policy.model.eval()


def _latch_telemetry(
    previous: WorldTelemetry,
    current: WorldTelemetry,
) -> WorldTelemetry:
    if previous.done:
        return replace(previous, step=current.step)
    return replace(
        current,
        success=previous.success or current.success,
        safety_failure=previous.safety_failure or current.safety_failure,
        terminated=previous.terminated or current.terminated,
        truncated=previous.truncated or current.truncated,
    )


def execute_rollout(
    *,
    runtime: ReadOnlyRolloutRuntime,
    frame_sink: RolloutFrameSink,
    reset_seed: int,
    episode_control_steps: int,
    execution_horizon: int,
) -> RolloutResult:
    """Execute inference only; this function has no persistence/update hooks."""
    if episode_control_steps < 1:
        raise ValueError("episode_control_steps must be positive")
    if execution_horizon not in (1, 2, 4, 8):
        raise ValueError("execution_horizon must be one of 1, 2, 4, or 8")
    if runtime.num_worlds < 1:
        raise ValueError("runtime must contain at least one world")

    state = runtime.reset(reset_seed)
    if len(state.telemetry) != runtime.num_worlds:
        raise RuntimeError("reset telemetry count does not match runtime worlds")
    frame_sink.write(state.observation, state.telemetry)
    history = [state.telemetry]
    active = tuple(not item.done for item in state.telemetry)
    control_steps = 0
    policy_decisions = 0

    while control_steps < episode_control_steps and any(active):
        action_chunk = runtime.sample_action_chunk(state.observation)
        policy_decisions += 1
        if not action_chunk:
            raise RuntimeError("policy returned an empty action chunk")
        horizon = min(
            execution_horizon,
            episode_control_steps - control_steps,
            len(action_chunk),
        )
        for action_batch in action_chunk[:horizon]:
            next_state = runtime.execute_action(action_batch, active)
            if len(next_state.telemetry) != runtime.num_worlds:
                raise RuntimeError(
                    "step telemetry count does not match runtime worlds"
                )
            latched = tuple(
                _latch_telemetry(previous, current)
                for previous, current in zip(
                    state.telemetry,
                    next_state.telemetry,
                    strict=True,
                )
            )
            state = RuntimeState(
                observation=next_state.observation,
                telemetry=latched,
            )
            control_steps += 1
            active = tuple(not item.done for item in latched)
            history.append(latched)
            frame_sink.write(state.observation, latched)
            if not any(active):
                break

    return RolloutResult(
        control_steps=control_steps,
        policy_decisions=policy_decisions,
        telemetry_history=tuple(history),
    )


def video_paths(output_dir: Path, num_worlds: int) -> tuple[dict[str, Path], ...]:
    """Return deterministic per-world locations for all three video views."""
    if num_worlds < 1:
        raise ValueError("num_worlds must be positive")
    paths = []
    for world in range(num_worlds):
        root = output_dir if num_worlds == 1 else output_dir / f"world-{world:03d}"
        paths.append(
            {
                "ego": root / "ego.mp4",
                "wrist": root / "wrist.mp4",
                "side_by_side": root / "side_by_side.mp4",
            }
        )
    return tuple(paths)


class OpenCvRolloutFrameSink:
    """Write RGB observations to three annotated MP4 files per world."""

    def __init__(
        self,
        *,
        output_dir: Path,
        num_worlds: int,
        episode_control_steps: int,
        fps: float,
        side_by_side_height: int,
    ) -> None:
        try:
            import cv2
            import numpy as np
        except ImportError as error:
            raise RuntimeError(
                "Rollout recording requires OpenCV (cv2) and NumPy"
            ) from error
        self.cv2 = cv2
        self.np = np
        self.output_dir = output_dir
        self.num_worlds = num_worlds
        self.episode_control_steps = episode_control_steps
        self.fps = float(fps)
        self.side_by_side_height = side_by_side_height
        self.paths = video_paths(output_dir, num_worlds)
        self._writers: dict[tuple[int, str], Any] = {}
        self.frame_count = 0

    def _host_rgb_batch(self, value: Any, name: str) -> Any:
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        array = self.np.asarray(value)
        if array.ndim != 4 or array.shape[0] != self.num_worlds:
            raise ValueError(
                f"{name} must have shape [world, height, width, channel], "
                f"got {array.shape}"
            )
        if array.shape[-1] != 3:
            raise ValueError(f"{name} must contain RGB channels, got {array.shape}")
        if array.dtype != self.np.uint8:
            array = self.np.clip(array, 0, 255).astype(self.np.uint8)
        return self.np.ascontiguousarray(array)

    def _annotate(
        self,
        frame: Any,
        telemetry: WorldTelemetry,
        label: str,
    ) -> Any:
        cv2 = self.cv2
        image = self.np.ascontiguousarray(frame.copy())
        height, width = image.shape[:2]
        bar_height = min(72, height)
        overlay = image.copy()
        cv2.rectangle(overlay, (0, 0), (width, bar_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.72, image, 0.28, 0.0, image)
        distance = (
            f"{telemetry.distance_m:.4f}m"
            if math.isfinite(telemetry.distance_m)
            else "n/a"
        )
        line_one = (
            f"{label}  step {telemetry.step:03d}/"
            f"{self.episode_control_steps:03d}  distance {distance}"
        )
        line_two = (
            f"success={'yes' if telemetry.success else 'no'}  "
            f"safety={'FAIL' if telemetry.safety_failure else 'ok'}"
        )
        line_three = (
            f"terminated={'yes' if telemetry.terminated else 'no'}  "
            f"truncated={'yes' if telemetry.truncated else 'no'}"
        )
        status_color = (
            (255, 80, 80)
            if telemetry.safety_failure
            else ((80, 255, 80) if telemetry.success else (255, 255, 255))
        )
        maximum_scale = 0.42 if width < 480 else 0.50
        for text, y, color in (
            (line_one, 18, (255, 255, 255)),
            (line_two, 40, status_color),
            (line_three, 62, status_color),
        ):
            measured_width = cv2.getTextSize(
                text,
                cv2.FONT_HERSHEY_SIMPLEX,
                maximum_scale,
                1,
            )[0][0]
            scale = min(
                maximum_scale,
                maximum_scale * max(width - 12, 1) / max(measured_width, 1),
            )
            cv2.putText(
                image,
                text,
                (6, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                scale,
                color,
                1,
                cv2.LINE_AA,
            )
        return image

    def _resize_to_height(self, frame: Any, height: int) -> Any:
        source_height, source_width = frame.shape[:2]
        width = int(round(source_width * height / source_height))
        if width % 2:
            width += 1
        return self.cv2.resize(
            frame,
            (width, height),
            interpolation=self.cv2.INTER_AREA,
        )

    def _writer(self, world: int, view: str, frame: Any) -> Any:
        key = (world, view)
        if key in self._writers:
            return self._writers[key]
        path = self.paths[world][view]
        path.parent.mkdir(parents=True, exist_ok=True)
        height, width = frame.shape[:2]
        if width % 2 or height % 2:
            raise ValueError(
                f"MP4 frame dimensions must be even, got {width}x{height}"
            )
        writer = None
        for codec in ("mp4v", "avc1"):
            candidate = self.cv2.VideoWriter(
                str(path),
                self.cv2.VideoWriter_fourcc(*codec),
                self.fps,
                (width, height),
            )
            if candidate.isOpened():
                writer = candidate
                break
            candidate.release()
        if writer is None:
            raise RuntimeError(f"No OpenCV MP4 encoder could open {path}")
        self._writers[key] = writer
        return writer

    def _write_rgb(self, world: int, view: str, frame: Any) -> None:
        writer = self._writer(world, view, frame)
        writer.write(self.np.ascontiguousarray(frame[..., ::-1]))

    def write(
        self,
        observation: Any,
        telemetry: tuple[WorldTelemetry, ...],
    ) -> None:
        if len(telemetry) != self.num_worlds:
            raise ValueError("telemetry count does not match recorded worlds")
        sensors = observation["sensor_data"]
        ego = self._host_rgb_batch(sensors["ego_view"]["rgb"], "ego RGB")
        wrist = self._host_rgb_batch(
            sensors["wrist_view"]["rgb"],
            "wrist RGB",
        )
        for world, status in enumerate(telemetry):
            ego_frame = self._annotate(ego[world], status, "ego")
            wrist_frame = self._annotate(wrist[world], status, "wrist")
            ego_panel = self._resize_to_height(
                ego[world],
                self.side_by_side_height,
            )
            wrist_panel = self._resize_to_height(
                wrist[world],
                self.side_by_side_height,
            )
            side_by_side = self.np.concatenate((ego_panel, wrist_panel), axis=1)
            side_by_side = self._annotate(side_by_side, status, "ego | wrist")
            self._write_rgb(world, "ego", ego_frame)
            self._write_rgb(world, "wrist", wrist_frame)
            self._write_rgb(world, "side_by_side", side_by_side)
        self.frame_count += 1

    def close(self) -> None:
        for writer in self._writers.values():
            writer.release()
        self._writers.clear()


def _host_values(value: Any, *, count: int, name: str) -> list[Any]:
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, list):
        value = [value]
    if len(value) != count:
        raise RuntimeError(f"{name} returned {len(value)} rows, expected {count}")
    return value


class NewtonPolicyRuntime:
    """Production adapter around the validated flywheel environment/policy."""

    def __init__(self, args: argparse.Namespace) -> None:
        import numpy as np
        import torch

        from rexpolicy.flywheel.groot_policy import GrootFlowDitPolicy
        from rexpolicy.flywheel.task_spec import get_task_spec
        from tools.run_flywheel_ddp import _create_environment, _raw_observations

        self.np = np
        self.torch = torch
        self.num_worlds = int(args.num_envs)
        self.task = get_task_spec(args.task_id)
        self.device = torch.device(args.device)
        if self.device.type != "cuda":
            raise ValueError("Newton rollout recording requires a CUDA device")
        torch.cuda.set_device(self.device)
        context = SimpleNamespace(device=self.device)
        self.env = _create_environment(
            args,
            context,
            self.task,
            num_envs=self.num_worlds,
        )
        self.policy = None
        try:
            self.policy = GrootFlowDitPolicy(
                isaac_groot_root=args.isaac_groot_root,
                checkpoint=args.policy_checkpoint,
                vlm_model=args.vlm_model,
                device=self.device,
                dit_overlay=args.dit_overlay,
            )
            _freeze_policy_for_inference(self.policy)
        except Exception:
            self.env.close()
            raise
        self._raw_observations = _raw_observations
        self._step = 0

    def seed_diffusion(self, seed: int) -> None:
        random.seed(seed)
        self.np.random.seed(seed % (2**32))
        self.torch.manual_seed(seed)
        self.torch.cuda.manual_seed(seed)

    def _telemetry(
        self,
        evaluation: dict[str, Any],
        *,
        terminated: Any | None = None,
        truncated: Any | None = None,
    ) -> tuple[WorldTelemetry, ...]:
        count = self.num_worlds
        distances = _host_values(
            evaluation["reach_distance"],
            count=count,
            name="reach_distance",
        )
        successes = _host_values(
            evaluation["success"],
            count=count,
            name="success",
        )
        failures = _host_values(
            evaluation["fail"],
            count=count,
            name="fail",
        )
        contacts = _host_values(
            evaluation["reach_contact_violation"],
            count=count,
            name="reach_contact_violation",
        )
        displacements = _host_values(
            evaluation["reach_displacement_violation"],
            count=count,
            name="reach_displacement_violation",
        )
        terminated_values = (
            [False] * count
            if terminated is None
            else _host_values(terminated, count=count, name="terminated")
        )
        truncated_values = (
            [False] * count
            if truncated is None
            else _host_values(truncated, count=count, name="truncated")
        )
        return tuple(
            WorldTelemetry(
                step=self._step,
                distance_m=float(distances[world]),
                success=bool(successes[world]),
                safety_failure=bool(
                    failures[world] or contacts[world] or displacements[world]
                ),
                terminated=bool(terminated_values[world]),
                truncated=bool(truncated_values[world]),
            )
            for world in range(count)
        )

    def reset(self, reset_seed: int) -> RuntimeState:
        self._step = 0
        observation, _ = self.env.reset(seed=reset_seed)
        return RuntimeState(
            observation=observation,
            telemetry=self._telemetry(self.env.evaluate()),
        )

    def sample_action_chunk(self, observation: Any) -> tuple[Any, ...]:
        raw = self._raw_observations(
            observation,
            instruction=self.task.instruction,
        )
        sampled = self.policy.sample(raw)
        decoded = sampled.decoded_action
        action_steps = min(
            int(decoded["eef_9d"].shape[1]),
            int(decoded["hand_joint_target"].shape[1]),
        )
        return tuple(
            self.np.concatenate(
                (
                    decoded["eef_9d"][:, step],
                    decoded["hand_joint_target"][:, step],
                ),
                axis=-1,
            ).astype(self.np.float32)
            for step in range(action_steps)
        )

    def execute_action(
        self,
        action_batch: Any,
        active: tuple[bool, ...],
    ) -> RuntimeState:
        proposed_np = self.np.asarray(action_batch, dtype=self.np.float32)
        if proposed_np.shape != (self.num_worlds, 19):
            raise ValueError(
                "policy action must have shape "
                f"({self.num_worlds}, 19), got {proposed_np.shape}"
            )
        if not self.np.isfinite(proposed_np).all():
            raise FloatingPointError("DiT sampled a non-finite action")
        proposed = self.torch.as_tensor(
            proposed_np,
            dtype=self.torch.float32,
            device=self.device,
        )
        active_tensor = self.torch.tensor(
            active,
            dtype=self.torch.bool,
            device=self.device,
        )
        hold = self.env.hold_action_torch().clone()
        mixed = self.torch.where(active_tensor[:, None], proposed, hold)
        effective = self.env.project_effective_action_torch(mixed)
        if not bool(self.torch.isfinite(effective).all()):
            raise FloatingPointError("projected action is non-finite")
        observation, _, terminated, truncated, info = self.env.step(effective)
        self._step += 1
        return RuntimeState(
            observation=observation,
            telemetry=self._telemetry(
                info,
                terminated=terminated,
                truncated=truncated,
            ),
        )

    def close(self) -> None:
        self.env.close()


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    try:
        with temporary.open("w", encoding="utf-8") as file:
            file.write(encoded)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _summary_payload(
    *,
    args: argparse.Namespace,
    overlay: OverlayDescriptor | None,
    result: RolloutResult,
    sink: OpenCvRolloutFrameSink,
) -> dict[str, Any]:
    videos = []
    for world, paths in enumerate(sink.paths):
        videos.append(
            {
                "world": world,
                **{
                    name: str(path.relative_to(args.output_dir.resolve()))
                    for name, path in paths.items()
                },
            }
        )
    return {
        "schema_version": 1,
        "mode": "read_only_rollout_recorder",
        "training_updates": 0,
        "archive_writes": 0,
        "policy_checkpoint": str(args.policy_checkpoint.resolve()),
        "vlm_model": str(args.vlm_model.resolve()),
        "scene_glb": (
            {
                "path": str(args.scene_glb),
                "size": args.scene_glb.stat().st_size,
                "sha256": _file_sha256(args.scene_glb),
            }
            if args.scene_visuals
            else None
        ),
        "dit_overlay": None if overlay is None else asdict(overlay),
        "task_id": args.task_id,
        "device": args.device,
        "reset_seed": args.reset_seed,
        "diffusion_seed": args.diffusion_seed,
        "num_envs": args.num_envs,
        "execution_horizon": args.execution_horizon,
        "requested_control_steps": args.episode_control_steps,
        "completed_control_steps": result.control_steps,
        "policy_decisions": result.policy_decisions,
        "frame_count": sink.frame_count,
        "videos": videos,
        "telemetry": [
            [asdict(item) for item in step_rows]
            for step_rows in result.telemetry_history
        ],
        "final": [asdict(item) for item in result.final_telemetry],
    }


def main(argv: list[str] | None = None) -> None:
    args = create_parser().parse_args(argv)
    validate_args(args)
    args.isaac_groot_root = args.isaac_groot_root.expanduser().resolve()
    args.policy_checkpoint = args.policy_checkpoint.expanduser().resolve()
    args.vlm_model = args.vlm_model.expanduser().resolve()
    args.scene_glb = args.scene_glb.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    if args.dit_overlay is not None:
        args.dit_overlay = args.dit_overlay.expanduser().resolve()
    overlay = resolve_dit_overlay(args.dit_overlay)

    args.output_dir.mkdir(parents=True)
    incomplete = args.output_dir / "INCOMPLETE"
    incomplete.write_text(
        "Rollout recording did not complete. This is not a training archive.\n",
        encoding="utf-8",
    )
    runtime = None
    sink = None
    try:
        runtime = NewtonPolicyRuntime(args)
        runtime.seed_diffusion(args.diffusion_seed)
        sink = OpenCvRolloutFrameSink(
            output_dir=args.output_dir,
            num_worlds=args.num_envs,
            episode_control_steps=args.episode_control_steps,
            fps=args.fps,
            side_by_side_height=args.side_by_side_height,
        )
        with runtime.torch.inference_mode():
            result = execute_rollout(
                runtime=runtime,
                frame_sink=sink,
                reset_seed=args.reset_seed,
                episode_control_steps=args.episode_control_steps,
                execution_horizon=args.execution_horizon,
            )
        sink.close()
        payload = _summary_payload(
            args=args,
            overlay=overlay,
            result=result,
            sink=sink,
        )
        _atomic_write_json(args.output_dir / "summary.json", payload)
        incomplete.unlink()
        print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
    finally:
        if sink is not None:
            sink.close()
        if runtime is not None:
            runtime.close()


if __name__ == "__main__":
    main()
