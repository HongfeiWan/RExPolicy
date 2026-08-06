"""Independent, atomic checkpoints for the Stage 0 training path."""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np


STAGE0_CHECKPOINT_SCHEMA_ID = "rexpolicy/stage0-checkpoint/v1"
STAGE0_CHECKPOINT_SCHEMA_VERSION = 1
_CHECKPOINT_NAME = re.compile(r"^checkpoint-(\d{12})$")
_SAFE_NAME = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_.-]{0,63}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class Stage0CheckpointError(RuntimeError):
    """Base class for Stage 0 checkpoint failures."""


class Stage0CheckpointIntegrityError(Stage0CheckpointError):
    """A checkpoint is incomplete, malformed, or changed on disk."""


class Stage0CheckpointMismatchError(Stage0CheckpointError):
    """The checkpoint belongs to a different experiment contract."""


def _require_sha256(value: Any, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


@dataclass(frozen=True)
class Stage0CheckpointHashes:
    """Semantic hashes that must match exactly before any state is restored."""

    config_sha256: str
    state_schema_sha256: str
    dataset_sha256: str
    split_sha256: str

    def __post_init__(self) -> None:
        for name, value in self.to_record().items():
            _require_sha256(value, name)

    def to_record(self) -> dict[str, str]:
        return {
            "config_sha256": self.config_sha256,
            "dataset_sha256": self.dataset_sha256,
            "split_sha256": self.split_sha256,
            "state_schema_sha256": self.state_schema_sha256,
        }

    @classmethod
    def from_record(cls, value: Any) -> Stage0CheckpointHashes:
        record = _exact_mapping(
            value,
            {
                "config_sha256",
                "dataset_sha256",
                "split_sha256",
                "state_schema_sha256",
            },
            "checkpoint hashes",
        )
        return cls(**record)


@dataclass(frozen=True)
class Stage0ResumeState:
    checkpoint_path: Path
    sequence: int
    phase: str
    steps: dict[str, int]
    hashes: Stage0CheckpointHashes
    rank_state: dict[str, Any]
    extra: dict[str, Any]


def _exact_mapping(value: Any, keys: set[str], context: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise Stage0CheckpointIntegrityError(f"{context} must be an object")
    actual = set(value)
    if actual != keys:
        raise Stage0CheckpointIntegrityError(
            f"{context} keys changed: missing={sorted(keys - actual)}, "
            f"unexpected={sorted(actual - keys)}"
        )
    return dict(value)


def _canonical_json(value: Any, *, pretty: bool = False) -> bytes:
    try:
        return (
            json.dumps(
                value,
                ensure_ascii=True,
                allow_nan=False,
                sort_keys=True,
                indent=2 if pretty else None,
                separators=None if pretty else (",", ":"),
            )
            + ("\n" if pretty else "")
        ).encode("ascii")
    except (TypeError, ValueError) as error:
        raise ValueError("checkpoint metadata must be finite canonical JSON") from error


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_bytes(path: Path, payload: bytes) -> None:
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    _write_bytes(path, _canonical_json(value, pretty=True))


def _write_torch(path: Path, value: Any) -> None:
    import torch

    with path.open("xb") as stream:
        torch.save(value, stream)
        stream.flush()
        os.fsync(stream.fileno())


def _safe_name(value: Any, context: str) -> str:
    if not isinstance(value, str) or _SAFE_NAME.fullmatch(value) is None:
        raise ValueError(f"{context} must be a safe non-empty identifier")
    return value


def _steps_record(value: Mapping[str, int]) -> dict[str, int]:
    if not isinstance(value, Mapping) or not value:
        raise ValueError("steps must be a non-empty mapping")
    result: dict[str, int] = {}
    for raw_name, raw_step in value.items():
        name = _safe_name(raw_name, "step name")
        if type(raw_step) is not int or raw_step < 0:
            raise ValueError(f"steps.{name} must be a non-negative integer")
        result[name] = raw_step
    return dict(sorted(result.items()))


def _named_components(value: Mapping[str, Any] | None, context: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must be a mapping")
    result = {_safe_name(name, f"{context} name"): item for name, item in value.items()}
    if len(result) != len(value):
        raise ValueError(f"{context} names must be unique")
    return dict(sorted(result.items()))


def _qualified_type(value: Any) -> str:
    kind = type(value)
    return f"{kind.__module__}.{kind.__qualname__}"


def _unwrap_module(module: Any) -> Any:
    try:
        from torch.nn.parallel import DistributedDataParallel

        if isinstance(module, DistributedDataParallel):
            return module.module
    except ImportError:  # pragma: no cover - torch is required by callers
        pass
    return module


def _model_schema_sha256(module: Any) -> str:
    import torch

    entries = []
    for name, value in _unwrap_module(module).state_dict().items():
        if not torch.is_tensor(value):
            raise ValueError(f"model state {name!r} is not a tensor")
        entries.append(
            {"dtype": str(value.dtype), "name": name, "shape": list(value.shape)}
        )
    return _sha256_bytes(_canonical_json(entries))


def _cpu_tree(value: Any) -> Any:
    import torch

    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_tree(item) for item in value)
    return value


def capture_rng_state(*, cuda_device: Any | None = None) -> dict[str, Any]:
    """Capture Python, NumPy, Torch CPU, and the selected CUDA stream RNG."""
    import torch

    cuda_index: int | None = None
    cuda_state = None
    if torch.cuda.is_available():
        device = torch.cuda.current_device() if cuda_device is None else cuda_device
        parsed_index = (
            torch.device(device).index if not isinstance(device, int) else device
        )
        cuda_index = (
            torch.cuda.current_device() if parsed_index is None else int(parsed_index)
        )
        cuda_state = torch.cuda.get_rng_state(device).cpu().clone()
    return {
        "schema_version": STAGE0_CHECKPOINT_SCHEMA_VERSION,
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state().cpu().clone(),
        "torch_cuda_device": cuda_index,
        "torch_cuda": cuda_state,
    }


def restore_rng_state(
    state: Mapping[str, Any], *, cuda_device: Any | None = None
) -> None:
    """Strictly restore a state returned by :func:`capture_rng_state`."""
    import torch

    record = _exact_mapping(
        state,
        {
            "schema_version",
            "python",
            "numpy",
            "torch_cpu",
            "torch_cuda_device",
            "torch_cuda",
        },
        "RNG state",
    )
    if record["schema_version"] != STAGE0_CHECKPOINT_SCHEMA_VERSION:
        raise Stage0CheckpointIntegrityError("unsupported RNG state version")
    try:
        random.setstate(record["python"])
        np.random.set_state(record["numpy"])
        torch.set_rng_state(record["torch_cpu"])
        if record["torch_cuda"] is not None:
            if not torch.cuda.is_available():
                raise Stage0CheckpointIntegrityError(
                    "checkpoint contains CUDA RNG but CUDA is unavailable"
                )
            device = record["torch_cuda_device"] if cuda_device is None else cuda_device
            torch.cuda.set_rng_state(record["torch_cuda"], device=device)
    except Stage0CheckpointIntegrityError:
        raise
    except Exception as error:
        raise Stage0CheckpointIntegrityError(f"invalid RNG state: {error}") from error


class Stage0CheckpointManager:
    """Save and restore all Stage 0 trainable components as one commit."""

    def __init__(
        self,
        root: str | Path,
        *,
        rank: int = 0,
        world_size: int = 1,
        barrier: Callable[[], None] | None = None,
        broadcast_rank0: Callable[[Any], Any] | None = None,
    ) -> None:
        if type(rank) is not int or type(world_size) is not int:
            raise ValueError("rank and world_size must be integers")
        if world_size < 1 or rank < 0 or rank >= world_size:
            raise ValueError(f"invalid rank/world_size: {rank}/{world_size}")
        if world_size > 1 and (barrier is None or broadcast_rank0 is None):
            raise ValueError(
                "multi-rank checkpoints require barrier and broadcast_rank0"
            )
        self.root = Path(root).expanduser().resolve()
        self.checkpoints_dir = self.root / "checkpoints"
        self.rank = rank
        self.world_size = world_size
        self._barrier = barrier or (lambda: None)
        self._broadcast_rank0 = broadcast_rank0 or (lambda value: value)

    @staticmethod
    def checkpoint_name(sequence: int) -> str:
        if type(sequence) is not int or sequence < 0:
            raise ValueError("sequence must be a non-negative integer")
        return f"checkpoint-{sequence:012d}"

    def _sync(self) -> None:
        self._barrier()

    def _broadcast_error(self, error: str | None) -> str | None:
        return self._broadcast_rank0(error if self.rank == 0 else None)

    def completed_checkpoints(self) -> list[Path]:
        if not self.checkpoints_dir.is_dir():
            return []
        paths = [
            path
            for path in self.checkpoints_dir.iterdir()
            if path.is_dir()
            and _CHECKPOINT_NAME.fullmatch(path.name)
            and (path / "COMPLETE").is_file()
        ]
        return sorted(paths, key=lambda path: path.name)

    def latest_checkpoint(self) -> Path | None:
        paths = self.completed_checkpoints()
        if not paths:
            return None
        self.verify(paths[-1])
        return paths[-1]

    def save(
        self,
        *,
        sequence: int,
        phase: str,
        steps: Mapping[str, int],
        hashes: Stage0CheckpointHashes,
        models: Mapping[str, Any] | None,
        optimizers: Mapping[str, Any] | None,
        rank_state: Mapping[str, Any] | None = None,
        extra: Mapping[str, Any] | None = None,
        cuda_device: Any | None = None,
        rng_state: Mapping[str, Any] | None = None,
    ) -> Path:
        name = self.checkpoint_name(sequence)
        phase = _safe_name(phase, "phase")
        steps_record = _steps_record(steps)
        if not isinstance(hashes, Stage0CheckpointHashes):
            raise TypeError("hashes must be Stage0CheckpointHashes")
        model_map = _named_components(models, "models")
        optimizer_map = _named_components(optimizers, "optimizers")
        if self.rank == 0 and not model_map:
            raise ValueError("rank zero must provide at least one model")
        rank_extra = dict(rank_state or {})
        manifest_extra = dict(extra or {})
        _canonical_json(manifest_extra)

        final = self.checkpoints_dir / name
        partial = self.checkpoints_dir / f".partial-{name}-{uuid.uuid4().hex}"
        setup_error: str | None = None
        if self.rank == 0:
            try:
                self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
                if final.exists():
                    raise Stage0CheckpointError(f"refusing to overwrite {final}")
                partial.mkdir()
                for child in ("models", "optimizers", "ranks"):
                    (partial / child).mkdir()
                _fsync_directory(self.checkpoints_dir)
            except Exception as error:
                setup_error = f"{type(error).__name__}: {error}"
        setup_error = self._broadcast_error(setup_error)
        if setup_error is not None:
            raise Stage0CheckpointError("checkpoint setup failed: " + setup_error)
        if self.world_size > 1:
            partial_text = self._broadcast_rank0(
                str(partial) if self.rank == 0 else None
            )
            partial = Path(partial_text)
        self._sync()

        rank_error: str | None = None
        try:
            _write_torch(
                partial / "ranks" / f"rank-{self.rank:05d}.pt",
                {
                    "schema_version": STAGE0_CHECKPOINT_SCHEMA_VERSION,
                    "rank": self.rank,
                    "world_size": self.world_size,
                    "rng": dict(
                        rng_state or capture_rng_state(cuda_device=cuda_device)
                    ),
                    "state": rank_extra,
                },
            )
        except Exception as error:
            rank_error = f"rank {self.rank}: {type(error).__name__}: {error}"
            _write_json(
                partial / f"ERROR-rank-{self.rank:05d}.json", {"error": rank_error}
            )
        self._sync()

        commit_error: str | None = None
        if self.rank == 0:
            try:
                errors = sorted(partial.glob("ERROR-rank-*.json"))
                if errors:
                    raise Stage0CheckpointError(
                        "rank state writes failed: "
                        + ", ".join(path.name for path in errors)
                    )
                missing = [
                    rank
                    for rank in range(self.world_size)
                    if not (partial / "ranks" / f"rank-{rank:05d}.pt").is_file()
                ]
                if missing:
                    raise Stage0CheckpointError(f"missing rank states: {missing}")

                model_records: dict[str, Any] = {}
                for component_name, model in model_map.items():
                    module = _unwrap_module(model)
                    schema_sha256 = _model_schema_sha256(module)
                    relative = f"models/{component_name}.pt"
                    _write_torch(
                        partial / relative,
                        {
                            "schema_version": STAGE0_CHECKPOINT_SCHEMA_VERSION,
                            "name": component_name,
                            "schema_sha256": schema_sha256,
                            "state_dict": _cpu_tree(module.state_dict()),
                        },
                    )
                    model_records[component_name] = {
                        "file": relative,
                        "schema_sha256": schema_sha256,
                        "type": _qualified_type(module),
                    }
                optimizer_records: dict[str, Any] = {}
                for component_name, optimizer in optimizer_map.items():
                    relative = f"optimizers/{component_name}.pt"
                    _write_torch(
                        partial / relative,
                        {
                            "schema_version": STAGE0_CHECKPOINT_SCHEMA_VERSION,
                            "name": component_name,
                            "type": _qualified_type(optimizer),
                            "state_dict": _cpu_tree(optimizer.state_dict()),
                        },
                    )
                    optimizer_records[component_name] = {
                        "file": relative,
                        "type": _qualified_type(optimizer),
                    }
                rank_records = {
                    str(rank): {"file": f"ranks/rank-{rank:05d}.pt"}
                    for rank in range(self.world_size)
                }
                manifest = {
                    "schema_id": STAGE0_CHECKPOINT_SCHEMA_ID,
                    "schema_version": STAGE0_CHECKPOINT_SCHEMA_VERSION,
                    "checkpoint_name": name,
                    "sequence": sequence,
                    "phase": phase,
                    "steps": steps_record,
                    "world_size": self.world_size,
                    "hashes": hashes.to_record(),
                    "models": model_records,
                    "optimizers": optimizer_records,
                    "ranks": rank_records,
                    "extra": manifest_extra,
                }
                manifest_bytes = _canonical_json(manifest, pretty=True)
                _write_bytes(partial / "manifest.json", manifest_bytes)
                payload_files = [
                    *(partial / item["file"] for item in model_records.values()),
                    *(partial / item["file"] for item in optimizer_records.values()),
                    *(partial / item["file"] for item in rank_records.values()),
                ]
                checksums = {
                    "schema_id": STAGE0_CHECKPOINT_SCHEMA_ID,
                    "schema_version": STAGE0_CHECKPOINT_SCHEMA_VERSION,
                    "files": {
                        str(path.relative_to(partial)): file_sha256(path)
                        for path in sorted(payload_files)
                    },
                }
                checksums_bytes = _canonical_json(checksums, pretty=True)
                _write_bytes(partial / "checksums.json", checksums_bytes)
                _write_json(
                    partial / "COMPLETE",
                    {
                        "schema_id": STAGE0_CHECKPOINT_SCHEMA_ID,
                        "schema_version": STAGE0_CHECKPOINT_SCHEMA_VERSION,
                        "checkpoint_name": name,
                        "sequence": sequence,
                        "manifest_sha256": _sha256_bytes(manifest_bytes),
                        "checksums_sha256": _sha256_bytes(checksums_bytes),
                    },
                )
                for child in ("models", "optimizers", "ranks", partial):
                    _fsync_directory(
                        partial / child if isinstance(child, str) else child
                    )
                os.rename(partial, final)
                _fsync_directory(self.checkpoints_dir)
            except Exception as error:
                commit_error = f"{type(error).__name__}: {error}"
                if partial.is_dir() and not (partial / "ERROR.json").exists():
                    _write_json(partial / "ERROR.json", {"error": commit_error})
        commit_error = self._broadcast_error(commit_error)
        self._sync()
        if commit_error is not None or not final.is_dir():
            raise Stage0CheckpointError(
                "checkpoint commit failed: "
                + (commit_error or rank_error or "incomplete")
            )
        self.verify(final)
        return final

    def verify(self, checkpoint: str | Path) -> dict[str, Any]:
        path = Path(checkpoint).expanduser().resolve()
        match = _CHECKPOINT_NAME.fullmatch(path.name)
        if match is None or path.name.startswith(".partial-"):
            raise Stage0CheckpointIntegrityError("partial or invalid checkpoint path")
        if not path.is_dir() or path.is_symlink() or not (path / "COMPLETE").is_file():
            raise Stage0CheckpointIntegrityError(
                "checkpoint is not completely committed"
            )

        complete_path = path / "COMPLETE"
        manifest_path = path / "manifest.json"
        checksums_path = path / "checksums.json"
        try:
            complete = json.loads(complete_path.read_text(encoding="ascii"))
            manifest = json.loads(manifest_path.read_text(encoding="ascii"))
            checksums = json.loads(checksums_path.read_text(encoding="ascii"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise Stage0CheckpointIntegrityError(
                f"cannot parse checkpoint: {error}"
            ) from error
        complete = _exact_mapping(
            complete,
            {
                "schema_id",
                "schema_version",
                "checkpoint_name",
                "sequence",
                "manifest_sha256",
                "checksums_sha256",
            },
            "COMPLETE",
        )
        if (
            complete["schema_id"] != STAGE0_CHECKPOINT_SCHEMA_ID
            or complete["schema_version"] != STAGE0_CHECKPOINT_SCHEMA_VERSION
        ):
            raise Stage0CheckpointIntegrityError("unsupported COMPLETE schema")
        sequence = int(match.group(1))
        if complete["checkpoint_name"] != path.name or complete["sequence"] != sequence:
            raise Stage0CheckpointIntegrityError("COMPLETE checkpoint identity changed")
        if file_sha256(manifest_path) != complete["manifest_sha256"]:
            raise Stage0CheckpointIntegrityError("manifest SHA-256 mismatch")
        if file_sha256(checksums_path) != complete["checksums_sha256"]:
            raise Stage0CheckpointIntegrityError("checksums SHA-256 mismatch")

        manifest = _exact_mapping(
            manifest,
            {
                "schema_id",
                "schema_version",
                "checkpoint_name",
                "sequence",
                "phase",
                "steps",
                "world_size",
                "hashes",
                "models",
                "optimizers",
                "ranks",
                "extra",
            },
            "manifest",
        )
        if (
            manifest["schema_id"] != STAGE0_CHECKPOINT_SCHEMA_ID
            or manifest["schema_version"] != STAGE0_CHECKPOINT_SCHEMA_VERSION
        ):
            raise Stage0CheckpointIntegrityError("unsupported manifest schema")
        if manifest["checkpoint_name"] != path.name or manifest["sequence"] != sequence:
            raise Stage0CheckpointIntegrityError("manifest checkpoint identity changed")
        try:
            _safe_name(manifest["phase"], "phase")
            _steps_record(manifest["steps"])
            Stage0CheckpointHashes.from_record(manifest["hashes"])
        except ValueError as error:
            raise Stage0CheckpointIntegrityError(str(error)) from error
        world_size = manifest["world_size"]
        if type(world_size) is not int or world_size < 1:
            raise Stage0CheckpointIntegrityError("manifest world_size is invalid")

        expected_files: set[str] = set()
        for group, prefix, fields in (
            ("models", "models", {"file", "schema_sha256", "type"}),
            ("optimizers", "optimizers", {"file", "type"}),
        ):
            records = manifest[group]
            if not isinstance(records, Mapping):
                raise Stage0CheckpointIntegrityError(
                    f"manifest {group} must be an object"
                )
            for component_name, raw_record in records.items():
                try:
                    _safe_name(component_name, f"{group} name")
                except ValueError as error:
                    raise Stage0CheckpointIntegrityError(str(error)) from error
                record = _exact_mapping(raw_record, fields, f"{group}.{component_name}")
                expected = f"{prefix}/{component_name}.pt"
                if record["file"] != expected:
                    raise Stage0CheckpointIntegrityError(
                        f"invalid component path {record['file']!r}"
                    )
                if group == "models":
                    try:
                        _require_sha256(record["schema_sha256"], "model schema")
                    except ValueError as error:
                        raise Stage0CheckpointIntegrityError(str(error)) from error
                if not isinstance(record["type"], str) or not record["type"]:
                    raise Stage0CheckpointIntegrityError("component type is invalid")
                expected_files.add(expected)
        ranks = manifest["ranks"]
        if not isinstance(ranks, Mapping) or set(ranks) != {
            str(i) for i in range(world_size)
        }:
            raise Stage0CheckpointIntegrityError(
                "rank inventory does not match world_size"
            )
        for rank in range(world_size):
            record = _exact_mapping(ranks[str(rank)], {"file"}, f"rank {rank}")
            expected = f"ranks/rank-{rank:05d}.pt"
            if record["file"] != expected:
                raise Stage0CheckpointIntegrityError("rank state path changed")
            expected_files.add(expected)

        checksum_record = _exact_mapping(
            checksums, {"schema_id", "schema_version", "files"}, "checksums"
        )
        if (
            checksum_record["schema_id"] != STAGE0_CHECKPOINT_SCHEMA_ID
            or checksum_record["schema_version"] != STAGE0_CHECKPOINT_SCHEMA_VERSION
        ):
            raise Stage0CheckpointIntegrityError("unsupported checksums schema")
        files = checksum_record["files"]
        if not isinstance(files, Mapping) or set(files) != expected_files:
            raise Stage0CheckpointIntegrityError("checksum inventory changed")
        for relative, expected_hash in files.items():
            pure = PurePosixPath(relative)
            if pure.is_absolute() or ".." in pure.parts:
                raise Stage0CheckpointIntegrityError("unsafe checksum path")
            try:
                _require_sha256(expected_hash, f"checksum {relative}")
            except ValueError as error:
                raise Stage0CheckpointIntegrityError(str(error)) from error
            payload_path = path / relative
            if not payload_path.is_file() or payload_path.is_symlink():
                raise Stage0CheckpointIntegrityError(
                    f"checkpoint file missing: {relative}"
                )
            if file_sha256(payload_path) != expected_hash:
                raise Stage0CheckpointIntegrityError(f"checksum mismatch: {relative}")
        actual_files = {
            str(item.relative_to(path)) for item in path.rglob("*") if item.is_file()
        }
        if actual_files != expected_files | {
            "COMPLETE",
            "manifest.json",
            "checksums.json",
        }:
            raise Stage0CheckpointIntegrityError(
                "checkpoint contains missing or untracked files"
            )
        return manifest

    def load(
        self,
        *,
        hashes: Stage0CheckpointHashes,
        models: Mapping[str, Any],
        optimizers: Mapping[str, Any],
        checkpoint: str | Path | None = None,
        restore_rng: bool = True,
        cuda_device: Any | None = None,
    ) -> Stage0ResumeState:
        import torch

        selected = (
            Path(checkpoint).expanduser().resolve()
            if checkpoint is not None
            else self.latest_checkpoint()
        )
        if selected is None:
            raise FileNotFoundError("no completed Stage 0 checkpoint")
        manifest = self.verify(selected)
        saved_hashes = Stage0CheckpointHashes.from_record(manifest["hashes"])
        if saved_hashes != hashes:
            changed = [
                name
                for name in saved_hashes.to_record()
                if saved_hashes.to_record()[name] != hashes.to_record()[name]
            ]
            raise Stage0CheckpointMismatchError(
                "checkpoint hash mismatch: " + ", ".join(changed)
            )
        if manifest["world_size"] != self.world_size:
            raise Stage0CheckpointMismatchError(
                "world_size mismatch: "
                f"checkpoint={manifest['world_size']}, current={self.world_size}"
            )
        model_map = _named_components(models, "models")
        optimizer_map = _named_components(optimizers, "optimizers")
        if set(model_map) != set(manifest["models"]):
            raise Stage0CheckpointMismatchError("model component inventory changed")
        if set(optimizer_map) != set(manifest["optimizers"]):
            raise Stage0CheckpointMismatchError("optimizer component inventory changed")

        model_payloads: dict[str, Any] = {}
        for name, model in model_map.items():
            expected_schema = manifest["models"][name]["schema_sha256"]
            if _model_schema_sha256(model) != expected_schema:
                raise Stage0CheckpointMismatchError(f"model schema changed: {name}")
            payload = torch.load(
                selected / f"models/{name}.pt",
                map_location="cpu",
                weights_only=False,
            )
            payload = _exact_mapping(
                payload,
                {"schema_version", "name", "schema_sha256", "state_dict"},
                f"model payload {name}",
            )
            if (
                payload["schema_version"] != STAGE0_CHECKPOINT_SCHEMA_VERSION
                or payload["name"] != name
                or payload["schema_sha256"] != expected_schema
            ):
                raise Stage0CheckpointIntegrityError(
                    f"model payload identity changed: {name}"
                )
            model_payloads[name] = payload["state_dict"]
        optimizer_payloads: dict[str, Any] = {}
        for name, optimizer in optimizer_map.items():
            expected_type = manifest["optimizers"][name]["type"]
            if _qualified_type(optimizer) != expected_type:
                raise Stage0CheckpointMismatchError(f"optimizer type changed: {name}")
            payload = torch.load(
                selected / f"optimizers/{name}.pt",
                map_location="cpu",
                weights_only=False,
            )
            payload = _exact_mapping(
                payload,
                {"schema_version", "name", "type", "state_dict"},
                f"optimizer payload {name}",
            )
            if (
                payload["schema_version"] != STAGE0_CHECKPOINT_SCHEMA_VERSION
                or payload["name"] != name
                or payload["type"] != expected_type
            ):
                raise Stage0CheckpointIntegrityError(
                    f"optimizer payload identity changed: {name}"
                )
            optimizer_payloads[name] = payload["state_dict"]
        rank_payload = torch.load(
            selected / f"ranks/rank-{self.rank:05d}.pt",
            map_location="cpu",
            weights_only=False,
        )
        rank_payload = _exact_mapping(
            rank_payload,
            {"schema_version", "rank", "world_size", "rng", "state"},
            "rank payload",
        )
        if (
            rank_payload["schema_version"] != STAGE0_CHECKPOINT_SCHEMA_VERSION
            or rank_payload["rank"] != self.rank
            or rank_payload["world_size"] != self.world_size
        ):
            raise Stage0CheckpointIntegrityError("rank payload identity changed")

        for name, model in model_map.items():
            _unwrap_module(model).load_state_dict(model_payloads[name], strict=True)
        for name, optimizer in optimizer_map.items():
            optimizer.load_state_dict(optimizer_payloads[name])
        if restore_rng:
            restore_rng_state(rank_payload["rng"], cuda_device=cuda_device)
        rank_state = rank_payload["state"]
        if not isinstance(rank_state, Mapping):
            raise Stage0CheckpointIntegrityError("rank state must be an object")
        return Stage0ResumeState(
            checkpoint_path=selected,
            sequence=manifest["sequence"],
            phase=manifest["phase"],
            steps=dict(manifest["steps"]),
            hashes=saved_hashes,
            rank_state=dict(rank_state),
            extra=dict(manifest["extra"]),
        )

    def load_model_component(
        self,
        *,
        hashes: Stage0CheckpointHashes,
        name: str,
        model: Any,
        checkpoint: str | Path | None = None,
    ) -> dict[str, Any]:
        """Load one verified model without touching optimizers or RNG state.

        This narrow operation supports held-out model selection after the full
        equal-budget training schedule.  Integrity, experiment hashes, and the
        destination model schema are checked exactly as they are for a full
        resume, but unrelated components remain unchanged.
        """

        import torch

        selected = (
            Path(checkpoint).expanduser().resolve()
            if checkpoint is not None
            else self.latest_checkpoint()
        )
        if selected is None:
            raise FileNotFoundError("no completed Stage 0 checkpoint")
        manifest = self.verify(selected)
        saved_hashes = Stage0CheckpointHashes.from_record(manifest["hashes"])
        if saved_hashes != hashes:
            changed = [
                field
                for field in saved_hashes.to_record()
                if saved_hashes.to_record()[field] != hashes.to_record()[field]
            ]
            raise Stage0CheckpointMismatchError(
                "checkpoint hash mismatch: " + ", ".join(changed)
            )
        component_name = _safe_name(name, "model component name")
        model_map = _named_components({component_name: model}, "models")
        if component_name not in manifest["models"]:
            raise Stage0CheckpointMismatchError(
                f"model component is absent from checkpoint: {component_name}"
            )
        destination = model_map[component_name]
        expected_schema = manifest["models"][component_name]["schema_sha256"]
        if _model_schema_sha256(destination) != expected_schema:
            raise Stage0CheckpointMismatchError(
                f"model schema changed: {component_name}"
            )
        payload = torch.load(
            selected / f"models/{component_name}.pt",
            map_location="cpu",
            weights_only=False,
        )
        payload = _exact_mapping(
            payload,
            {"schema_version", "name", "schema_sha256", "state_dict"},
            f"model payload {component_name}",
        )
        if (
            payload["schema_version"] != STAGE0_CHECKPOINT_SCHEMA_VERSION
            or payload["name"] != component_name
            or payload["schema_sha256"] != expected_schema
        ):
            raise Stage0CheckpointIntegrityError(
                f"model payload identity changed: {component_name}"
            )
        _unwrap_module(destination).load_state_dict(payload["state_dict"], strict=True)
        return manifest


__all__ = [
    "STAGE0_CHECKPOINT_SCHEMA_ID",
    "STAGE0_CHECKPOINT_SCHEMA_VERSION",
    "Stage0CheckpointError",
    "Stage0CheckpointHashes",
    "Stage0CheckpointIntegrityError",
    "Stage0CheckpointManager",
    "Stage0CheckpointMismatchError",
    "Stage0ResumeState",
    "capture_rng_state",
    "file_sha256",
    "restore_rng_state",
]
