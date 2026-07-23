"""Atomic, resumable checkpoints for distributed flywheel training.

The module deliberately has no dependency on the GR00T policy wrapper.  A runner
passes the trainable DiT module and optimizer directly, which keeps checkpoint
tests small and makes the persistence contract usable by other policies.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import shutil
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


CHECKPOINT_SCHEMA_VERSION = 1
DEFAULT_MUTABLE_MANIFEST_PATHS = frozenset(
    {
        "max_generations",
        "heartbeat_interval_seconds",
        "gpu_monitor_interval_seconds",
        "config.max_generations",
        "config.heartbeat_interval_seconds",
        "config.gpu_monitor_interval_seconds",
    }
)
_MAX_GENERATION_PATHS = frozenset(
    {
        "max_generations",
        "config.max_generations",
    }
)
_GENERATION_DIRECTORY = re.compile(r"^generation-(\d{6,})$")


class CheckpointError(RuntimeError):
    """Base class for checkpoint failures."""


class CheckpointIntegrityError(CheckpointError):
    """Raised when a checkpoint is incomplete or has changed on disk."""


class ManifestMismatchError(CheckpointError):
    """Raised when a run tries to resume with incompatible semantics."""


@dataclass(frozen=True)
class ResumeState:
    """State restored from one completed checkpoint."""

    checkpoint_path: Path
    completed_generation: int
    next_generation: int
    global_optimizer_step: int
    replay_cursor: Any
    rank_state: dict[str, Any]
    run_manifest: dict[str, Any]
    extra_state: dict[str, Any]


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        default=_json_default,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}"
    )
    try:
        with temporary.open("wb") as file:
            file.write(payload)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write JSON through a same-directory temporary file and atomic replace."""
    encoded = (
        json.dumps(
            payload,
            default=_json_default,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    _atomic_write_bytes(path, encoded)


def _atomic_torch_save(path: Path, payload: Any) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}"
    )
    try:
        with temporary.open("wb") as file:
            torch.save(payload, file)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def file_sha256(path: Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    """Return the SHA-256 digest of a file without loading it all into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def parameter_order_hash(module: Any) -> str:
    """Hash parameter names, shapes, and dtypes in optimizer-visible order."""
    entries = [
        {
            "name": name,
            "shape": list(parameter.shape),
            "dtype": str(parameter.dtype),
            "requires_grad": bool(parameter.requires_grad),
        }
        for name, parameter in _unwrap_module(module).named_parameters()
    ]
    return hashlib.sha256(_canonical_json(entries).encode("utf-8")).hexdigest()


def manifest_hash(manifest: Mapping[str, Any]) -> str:
    """Return a stable hash for a JSON-compatible run manifest."""
    return hashlib.sha256(_canonical_json(manifest).encode("utf-8")).hexdigest()


def _flatten_manifest(value: Any, prefix: str = "") -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {prefix: value}
    flattened: dict[str, Any] = {}
    for key, item in value.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(item, Mapping):
            flattened.update(_flatten_manifest(item, path))
        else:
            flattened[path] = item
    return flattened


def validate_manifest_compatible(
    saved: Mapping[str, Any],
    current: Mapping[str, Any],
    *,
    allowed_differences: Sequence[str] = tuple(DEFAULT_MUTABLE_MANIFEST_PATHS),
) -> None:
    """Require exact manifest equality except for explicit operational fields."""
    # Manifests are persisted as JSON, which converts tuples to lists and NumPy
    # scalars to Python scalars. Compare the same representation that is stored
    # on disk so an otherwise identical in-memory task contract can resume.
    saved_flat = _flatten_manifest(json.loads(_canonical_json(saved)))
    current_flat = _flatten_manifest(json.loads(_canonical_json(current)))
    allowed = set(allowed_differences)
    differences: list[str] = []
    for path in sorted(set(saved_flat) | set(current_flat)):
        if path in allowed:
            if (
                path in _MAX_GENERATION_PATHS
                and path in saved_flat
                and path in current_flat
                and int(current_flat[path]) < int(saved_flat[path])
            ):
                differences.append(
                    f"{path}: cannot decrease from checkpoint="
                    f"{saved_flat[path]!r} to current={current_flat[path]!r}"
                )
            continue
        if path not in saved_flat:
            differences.append(f"{path}: missing in checkpoint")
        elif path not in current_flat:
            differences.append(f"{path}: missing in current run")
        elif saved_flat[path] != current_flat[path]:
            differences.append(
                f"{path}: checkpoint={saved_flat[path]!r}, current={current_flat[path]!r}"
            )
    if differences:
        detail = "; ".join(differences[:20])
        if len(differences) > 20:
            detail += f"; ... {len(differences) - 20} more"
        raise ManifestMismatchError(f"Incompatible resume manifest: {detail}")


def _unwrap_module(module: Any) -> Any:
    return getattr(module, "module", module)


def validate_fp32_module(module: Any) -> None:
    """Require every floating-point DiT parameter to use FP32 storage."""
    import torch

    bad = [
        f"{name}={parameter.dtype}"
        for name, parameter in _unwrap_module(module).named_parameters()
        if parameter.is_floating_point() and parameter.dtype != torch.float32
    ]
    if bad:
        preview = ", ".join(bad[:10])
        raise CheckpointError(f"DiT parameters must be FP32 before save: {preview}")


def validate_fp32_optimizer_state(optimizer: Any) -> None:
    """Require floating optimizer moments to use FP32 storage."""
    import torch

    bad: list[str] = []
    for parameter_index, state in enumerate(optimizer.state.values()):
        for key, value in state.items():
            if (
                torch.is_tensor(value)
                and value.is_floating_point()
                and value.dtype != torch.float32
            ):
                bad.append(f"state[{parameter_index}].{key}={value.dtype}")
    if bad:
        raise CheckpointError(
            "Optimizer floating state must be FP32 before save: "
            + ", ".join(bad[:10])
        )


def save_fp32_dit_safetensors(module: Any, path: Path) -> None:
    """Save a DiT state dictionary as contiguous CPU FP32 safetensors."""
    import torch
    from safetensors.torch import save_file

    module = _unwrap_module(module)
    validate_fp32_module(module)
    state: dict[str, Any] = {}
    for name, value in module.state_dict().items():
        if not torch.is_tensor(value):
            raise CheckpointError(f"DiT state entry {name!r} is not a tensor")
        tensor = value.detach().cpu()
        if tensor.is_floating_point():
            tensor = tensor.float()
        state[name] = tensor.contiguous()
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        state,
        str(path),
        metadata={"format": "pt", "component": "gr00t_flow_dit", "dtype": "float32"},
    )
    with path.open("rb") as file:
        os.fsync(file.fileno())


def load_dit_safetensors(module: Any, path: Path) -> None:
    """Strictly load a DiT-only safetensors file."""
    from safetensors.torch import load_file

    state = load_file(str(path), device="cpu")
    non_fp32 = [
        f"{name}={tensor.dtype}"
        for name, tensor in state.items()
        if tensor.is_floating_point() and str(tensor.dtype) != "torch.float32"
    ]
    if non_fp32:
        raise CheckpointIntegrityError(
            "Checkpoint DiT tensors are not FP32: " + ", ".join(non_fp32[:10])
        )
    _unwrap_module(module).load_state_dict(state, strict=True)
    validate_fp32_module(module)


def _clone_to_cpu(value: Any) -> Any:
    import torch

    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _clone_to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_to_cpu(item) for item in value)
    return value


def optimizer_state_dict_cpu(optimizer: Any) -> dict[str, Any]:
    """Copy an optimizer state dictionary without mutating live state."""
    validate_fp32_optimizer_state(optimizer)
    return _clone_to_cpu(optimizer.state_dict())


def move_optimizer_state_(optimizer: Any, device: Any) -> None:
    """Move all tensors in an optimizer state to one device in place."""
    import torch

    def move(value: Any) -> Any:
        if torch.is_tensor(value):
            return value.to(device=device)
        if isinstance(value, dict):
            return {key: move(item) for key, item in value.items()}
        if isinstance(value, list):
            return [move(item) for item in value]
        if isinstance(value, tuple):
            return tuple(move(item) for item in value)
        return value

    for parameter, state in list(optimizer.state.items()):
        optimizer.state[parameter] = move(state)


def capture_rng_state(*, cuda_device: Any | None = None) -> dict[str, Any]:
    """Capture Python, NumPy, Torch CPU, and this rank's CUDA RNG state."""
    import torch

    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": None,
    }
    if torch.cuda.is_available():
        device = torch.cuda.current_device() if cuda_device is None else cuda_device
        state["torch_cuda"] = torch.cuda.get_rng_state(device).cpu()
    return state


def restore_rng_state(
    state: Mapping[str, Any], *, cuda_device: Any | None = None
) -> None:
    """Restore a state returned by :func:`capture_rng_state`."""
    import torch

    required = {"python", "numpy", "torch_cpu", "torch_cuda"}
    missing = required - set(state)
    if missing:
        raise CheckpointIntegrityError(f"RNG state is missing keys: {sorted(missing)}")
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if state["torch_cuda"] is not None:
        if not torch.cuda.is_available():
            raise CheckpointIntegrityError(
                "Checkpoint contains CUDA RNG state but CUDA is unavailable"
            )
        device = torch.cuda.current_device() if cuda_device is None else cuda_device
        torch.cuda.set_rng_state(state["torch_cuda"], device=device)


class CheckpointManager:
    """Coordinate generation-boundary checkpoint commits across DDP ranks.

    Every rank calls :meth:`save`.  Rank zero writes the shared DiT and optimizer,
    while each rank writes its own RNG and replay cursor.  ``barrier`` must have
    the same semantics as :meth:`DistributedContext.barrier`.
    """

    def __init__(
        self,
        run_dir: Path,
        *,
        rank: int = 0,
        world_size: int = 1,
        barrier: Callable[[], None] | None = None,
    ) -> None:
        if rank < 0 or world_size < 1 or rank >= world_size:
            raise ValueError(f"Invalid rank/world_size: {rank}/{world_size}")
        self.run_dir = Path(run_dir).expanduser().resolve()
        self.checkpoints_dir = self.run_dir / "checkpoints"
        self.rank = rank
        self.world_size = world_size
        self._barrier = barrier or (lambda: None)

    def _sync(self) -> None:
        self._barrier()

    @staticmethod
    def _generation_name(completed_generation: int) -> str:
        if completed_generation < 0:
            raise ValueError("completed_generation cannot be negative")
        return f"generation-{completed_generation:06d}"

    def completed_checkpoints(self) -> list[Path]:
        """List verified-looking completed directories, ignoring partial writes."""
        if not self.checkpoints_dir.is_dir():
            return []
        candidates: list[tuple[int, Path]] = []
        for path in self.checkpoints_dir.iterdir():
            match = _GENERATION_DIRECTORY.fullmatch(path.name)
            if match and path.is_dir() and (path / "COMPLETE").is_file():
                candidates.append((int(match.group(1)), path))
        return [path for _, path in sorted(candidates)]

    def latest_checkpoint(self) -> Path | None:
        """Resolve the highest verified accepted checkpoint.

        A corrupt completed generation is never silently skipped. Valid
        rejected candidates are skipped because they must not replace the
        preceding last-good optimizer state.
        """
        for checkpoint in reversed(self.completed_checkpoints()):
            state = self.verify(checkpoint)
            if state["accepted"]:
                return checkpoint
        return None

    def save(
        self,
        *,
        dit: Any | None,
        optimizer: Any | None,
        completed_generation: int,
        global_optimizer_step: int,
        run_manifest: Mapping[str, Any],
        replay_cursor: Any = None,
        extra_rank_state: Mapping[str, Any] | None = None,
        extra_state: Mapping[str, Any] | None = None,
        cuda_device: Any | None = None,
        rng_state: Mapping[str, Any] | None = None,
        dit_saver: Callable[[Any, Path], None] = save_fp32_dit_safetensors,
        update_latest: bool = True,
        accepted: bool = True,
    ) -> Path:
        """Atomically save one completed generation.

        A directory is visible to resume only after ``COMPLETE`` is written and
        the partial directory is atomically renamed.  Existing completed
        generations are never overwritten.
        """
        if global_optimizer_step < 0:
            raise ValueError("global_optimizer_step cannot be negative")
        if update_latest and not accepted:
            raise ValueError("A rejected checkpoint cannot replace latest.json")
        name = self._generation_name(completed_generation)
        final = self.checkpoints_dir / name
        partial = self.checkpoints_dir / f".partial-{name}"
        error_path = partial / "ERROR.json"

        initialization_error: str | None = None
        if self.rank == 0:
            try:
                self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
                if final.exists():
                    raise CheckpointError(
                        f"Refusing to overwrite checkpoint {final}"
                    )
                if partial.exists():
                    shutil.rmtree(partial)
                (partial / "ranks").mkdir(parents=True)
                _fsync_directory(self.checkpoints_dir)
            except Exception as error:
                initialization_error = f"{type(error).__name__}: {error}"
        self._sync()
        if self.rank != 0:
            initialization_error = None
        initialization_error = self._broadcast_rank_zero_error(
            initialization_error
        )
        if initialization_error is not None:
            raise CheckpointError(
                "Checkpoint initialization failed: " + initialization_error
            )

        rank_error: str | None = None
        try:
            rank_payload = {
                "schema_version": CHECKPOINT_SCHEMA_VERSION,
                "rank": self.rank,
                "rng": dict(rng_state or capture_rng_state(cuda_device=cuda_device)),
                "replay_cursor": replay_cursor,
                "extra": dict(extra_rank_state or {}),
            }
            _atomic_torch_save(
                partial / "ranks" / f"rank-{self.rank:05d}.pt", rank_payload
            )
        except Exception as error:  # Coordinate an orderly failure across ranks.
            rank_error = f"{type(error).__name__}: {error}"
            atomic_write_json(
                partial / f"ERROR-rank-{self.rank:05d}.json",
                {"rank": self.rank, "error": rank_error},
            )
        self._sync()

        if self.rank == 0:
            rank_errors = sorted(partial.glob("ERROR-rank-*.json"))
            try:
                if rank_errors:
                    raise CheckpointError(
                        "Rank checkpoint state failed: "
                        + ", ".join(path.name for path in rank_errors)
                    )
                missing_ranks = [
                    rank
                    for rank in range(self.world_size)
                    if not (partial / "ranks" / f"rank-{rank:05d}.pt").is_file()
                ]
                if missing_ranks:
                    raise CheckpointError(
                        f"Missing checkpoint state for ranks {missing_ranks}"
                    )
                if dit is None or optimizer is None:
                    raise ValueError("Rank zero requires both dit and optimizer")

                validate_fp32_module(dit)
                validate_fp32_optimizer_state(optimizer)
                dit_saver(dit, partial / "model.safetensors")
                _atomic_torch_save(
                    partial / "optimizer.pt", optimizer_state_dict_cpu(optimizer)
                )
                state_payload = {
                    "schema_version": CHECKPOINT_SCHEMA_VERSION,
                    "completed_generation": completed_generation,
                    "next_generation": completed_generation + 1,
                    "global_optimizer_step": global_optimizer_step,
                    "world_size": self.world_size,
                    "accepted": bool(accepted),
                    "parameter_order_hash": parameter_order_hash(dit),
                    "run_manifest_hash": manifest_hash(run_manifest),
                    "run_manifest": dict(run_manifest),
                    "extra": dict(extra_state or {}),
                }
                atomic_write_json(partial / "checkpoint_state.json", state_payload)
            except Exception as error:  # Let waiting ranks observe the failure.
                atomic_write_json(
                    error_path,
                    {"rank": 0, "error": f"{type(error).__name__}: {error}"},
                )
        self._sync()

        if self.rank == 0 and not error_path.exists():
            try:
                tracked_files = [
                    partial / "model.safetensors",
                    partial / "optimizer.pt",
                    partial / "checkpoint_state.json",
                    *sorted((partial / "ranks").glob("rank-*.pt")),
                ]
                checksums = {
                    str(path.relative_to(partial)): file_sha256(path)
                    for path in tracked_files
                }
                atomic_write_json(partial / "checksums.json", checksums)
                atomic_write_json(
                    partial / "COMPLETE",
                    {
                        "schema_version": CHECKPOINT_SCHEMA_VERSION,
                        "completed_generation": completed_generation,
                        "world_size": self.world_size,
                    },
                )
                _fsync_directory(partial / "ranks")
                _fsync_directory(partial)
                os.replace(partial, final)
                _fsync_directory(self.checkpoints_dir)
                if update_latest:
                    atomic_write_json(
                        self.checkpoints_dir / "latest.json",
                        {
                            "checkpoint": final.name,
                            "completed_generation": completed_generation,
                            "checkpoint_state_sha256": checksums[
                                "checkpoint_state.json"
                            ],
                        },
                    )
            except Exception as error:
                if partial.exists():
                    atomic_write_json(
                        error_path,
                        {"rank": 0, "error": f"{type(error).__name__}: {error}"},
                    )
        self._sync()

        if not final.is_dir() or not (final / "COMPLETE").is_file():
            errors = []
            if partial.is_dir():
                for path in sorted(partial.glob("ERROR*.json")):
                    try:
                        errors.append(json.loads(path.read_text())["error"])
                    except Exception:
                        errors.append(path.name)
            detail = "; ".join(errors) or rank_error or "checkpoint commit incomplete"
            raise CheckpointError(f"Failed to save generation {completed_generation}: {detail}")
        return final

    def _broadcast_rank_zero_error(self, error: str | None) -> str | None:
        """Share a rank-zero setup failure without stranding other ranks."""
        if self.world_size == 1:
            return error
        import torch.distributed as dist

        if not dist.is_available() or not dist.is_initialized():
            # Test barriers may be thread based and cannot broadcast objects.
            # Their rank-zero setup path is expected to succeed.
            return error
        payload = [error if self.rank == 0 else None]
        dist.broadcast_object_list(payload, src=0)
        return payload[0]

    def verify(self, checkpoint: Path) -> dict[str, Any]:
        """Verify marker, schema, rank files, and all recorded checksums."""
        checkpoint = Path(checkpoint).expanduser().resolve()
        if checkpoint.name.startswith(".partial-") or not (checkpoint / "COMPLETE").is_file():
            raise CheckpointIntegrityError(f"Checkpoint is incomplete: {checkpoint}")
        try:
            complete = json.loads((checkpoint / "COMPLETE").read_text(encoding="utf-8"))
            state = json.loads(
                (checkpoint / "checkpoint_state.json").read_text(encoding="utf-8")
            )
            checksums = json.loads(
                (checkpoint / "checksums.json").read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError) as error:
            raise CheckpointIntegrityError(
                f"Unreadable checkpoint metadata at {checkpoint}: {error}"
            ) from error
        if not all(isinstance(item, dict) for item in (complete, state, checksums)):
            raise CheckpointIntegrityError("Checkpoint metadata must contain JSON objects")
        if complete.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
            raise CheckpointIntegrityError(
                f"Unsupported checkpoint schema: {complete.get('schema_version')}"
            )
        if state.get("world_size") != complete.get("world_size"):
            raise CheckpointIntegrityError("Checkpoint world_size metadata disagrees")
        if state.get("completed_generation") != complete.get("completed_generation"):
            raise CheckpointIntegrityError(
                "Checkpoint generation metadata disagrees"
            )
        match = _GENERATION_DIRECTORY.fullmatch(checkpoint.name)
        if match is None or int(match.group(1)) != state.get("completed_generation"):
            raise CheckpointIntegrityError(
                "Checkpoint directory name does not match its generation"
            )
        if state.get("next_generation") != state.get("completed_generation") + 1:
            raise CheckpointIntegrityError(
                "Checkpoint next_generation is not the next integer generation"
            )
        if not isinstance(state.get("accepted"), bool):
            raise CheckpointIntegrityError(
                "Checkpoint accepted status is missing or invalid"
            )
        run_manifest = state.get("run_manifest")
        if not isinstance(run_manifest, dict):
            raise CheckpointIntegrityError("Checkpoint run_manifest is not an object")
        if state.get("run_manifest_hash") != manifest_hash(run_manifest):
            raise CheckpointIntegrityError("Checkpoint run_manifest hash disagrees")
        world_size = state.get("world_size")
        if not isinstance(world_size, int) or world_size < 1:
            raise CheckpointIntegrityError(
                f"Invalid checkpoint world_size: {world_size!r}"
            )
        required_files = {
            "model.safetensors",
            "optimizer.pt",
            "checkpoint_state.json",
            *{f"ranks/rank-{rank:05d}.pt" for rank in range(world_size)},
        }
        if set(checksums) != required_files:
            missing = sorted(required_files - set(checksums))
            unexpected = sorted(set(checksums) - required_files)
            raise CheckpointIntegrityError(
                f"Checkpoint checksum inventory mismatch: missing={missing}, "
                f"unexpected={unexpected}"
            )
        for relative, expected in checksums.items():
            path = checkpoint / relative
            if not path.is_file():
                raise CheckpointIntegrityError(f"Checkpoint file is missing: {relative}")
            actual = file_sha256(path)
            if actual != expected:
                raise CheckpointIntegrityError(
                    f"Checksum mismatch for {relative}: expected={expected}, actual={actual}"
                )
        return state

    def load(
        self,
        *,
        dit: Any,
        optimizer: Any,
        current_manifest: Mapping[str, Any],
        checkpoint: Path | None = None,
        allowed_manifest_differences: Sequence[str] = tuple(
            DEFAULT_MUTABLE_MANIFEST_PATHS
        ),
        restore_rng: bool = True,
        cuda_device: Any | None = None,
        optimizer_device: Any = "cpu",
        dit_loader: Callable[[Any, Path], None] = load_dit_safetensors,
        allow_rejected: bool = False,
    ) -> ResumeState:
        """Strictly restore model, optimizer, this rank's RNG, and replay cursor."""
        import torch

        selected = (
            Path(checkpoint).expanduser().resolve()
            if checkpoint is not None
            else self.latest_checkpoint()
        )
        if selected is None:
            raise FileNotFoundError("No completed checkpoint is available")
        state = self.verify(selected)
        if not state["accepted"] and not allow_rejected:
            raise CheckpointIntegrityError(
                f"Refusing to resume rejected candidate checkpoint {selected}"
            )
        if state["world_size"] != self.world_size:
            raise ManifestMismatchError(
                f"world_size changed: checkpoint={state['world_size']}, "
                f"current={self.world_size}"
            )
        validate_manifest_compatible(
            state["run_manifest"],
            current_manifest,
            allowed_differences=allowed_manifest_differences,
        )
        if parameter_order_hash(dit) != state["parameter_order_hash"]:
            raise ManifestMismatchError("DiT parameter order/schema has changed")

        dit_loader(dit, selected / "model.safetensors")
        optimizer_payload = torch.load(
            selected / "optimizer.pt", map_location="cpu", weights_only=False
        )
        optimizer.load_state_dict(optimizer_payload)
        move_optimizer_state_(optimizer, optimizer_device)
        validate_fp32_optimizer_state(optimizer)

        rank_path = selected / "ranks" / f"rank-{self.rank:05d}.pt"
        if not rank_path.is_file():
            raise CheckpointIntegrityError(
                f"Checkpoint has no state for rank {self.rank}"
            )
        rank_state = torch.load(rank_path, map_location="cpu", weights_only=False)
        if rank_state.get("rank") != self.rank:
            raise CheckpointIntegrityError(
                f"Rank state mismatch in {rank_path}: {rank_state.get('rank')}"
            )
        if restore_rng:
            restore_rng_state(rank_state["rng"], cuda_device=cuda_device)
        return ResumeState(
            checkpoint_path=selected,
            completed_generation=int(state["completed_generation"]),
            next_generation=int(state["next_generation"]),
            global_optimizer_step=int(state["global_optimizer_step"]),
            replay_cursor=rank_state.get("replay_cursor"),
            rank_state=dict(rank_state.get("extra", {})),
            run_manifest=dict(state["run_manifest"]),
            extra_state=dict(state.get("extra", {})),
        )
