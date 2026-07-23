"""Run lifecycle, durable logs, GPU monitoring, and launch preflight checks."""

from __future__ import annotations

import csv
import getpass
import json
import os
import platform
import re
import signal
import socket
import subprocess
import sys
import threading
import uuid
from importlib import metadata as importlib_metadata
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from .checkpoint import atomic_write_json


_SECRET_KEY = re.compile(
    r"(?:^|[-_])(?:password|passwd|pwd|secret|token|api[-_]?key|"
    r"authorization|credential)(?:$|[-_])",
    re.IGNORECASE,
)
_INLINE_SECRET = re.compile(
    r"(?i)(password|passwd|pwd|secret|token|api[-_]?key|authorization|credential)"
    r"(\s*[:=]\s*)([^\s,;]+)"
)
_SPACE_SECRET = re.compile(
    r"(?i)\b(password|passwd|pwd|secret|token|api[-_]?key|authorization|"
    r"credential)(\s+)([^\s,;]+)"
)
_CSV_GPU_FIELDS = (
    "timestamp_utc",
    "index",
    "uuid",
    "name",
    "utilization_gpu_percent",
    "memory_used_mib",
    "memory_free_mib",
    "memory_total_mib",
    "power_draw_w",
    "temperature_c",
    "clocks_sm_mhz",
)
_CSV_PROCESS_FIELDS = (
    "timestamp_utc",
    "gpu_uuid",
    "pid",
    "user",
    "process_name",
    "used_gpu_memory_mib",
)


class OperationsError(RuntimeError):
    """Base class for run-operation failures."""


class GpuPreflightError(OperationsError):
    """Raised when selected GPUs are busy or do not have enough free memory."""


@dataclass(frozen=True)
class GpuProcess:
    """One active compute process reported by NVIDIA SMI."""

    gpu_uuid: str
    pid: int
    user: str
    process_name: str
    used_memory_mib: int


@dataclass(frozen=True)
class GpuDevice:
    """One physical GPU relevant to launch preflight."""

    index: int
    uuid: str
    name: str
    memory_free_mib: int
    memory_total_mib: int


@dataclass(frozen=True)
class GpuPreflightResult:
    """Resolved selected GPUs and processes allowed to remain on them."""

    devices: tuple[GpuDevice, ...]
    processes: tuple[GpuProcess, ...]


def utc_now() -> str:
    """Return a sortable UTC timestamp."""
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _redact_url(value: str) -> str:
    try:
        split = urlsplit(value)
    except ValueError:
        return value
    if not split.scheme or split.password is None:
        return value
    host = split.hostname or ""
    if split.port is not None:
        host = f"{host}:{split.port}"
    username = split.username or ""
    netloc = f"{username}:<redacted>@{host}"
    return urlunsplit((split.scheme, netloc, split.path, split.query, split.fragment))


def redact_text(value: str) -> str:
    """Redact common inline credential shapes before durable logging."""
    value = _INLINE_SECRET.sub(r"\1\2<redacted>", value)
    value = _SPACE_SECRET.sub(r"\1\2<redacted>", value)
    words = value.split()
    return " ".join(_redact_url(word) for word in words)


def sanitize_for_logging(value: Any, *, key: str | None = None) -> Any:
    """Recursively redact secrets and normalize objects for JSON logs."""
    if key is not None and _SECRET_KEY.search(key):
        return "<redacted>"
    if isinstance(value, Mapping):
        return {
            str(item_key): sanitize_for_logging(item, key=str(item_key))
            for item_key, item in value.items()
        }
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, str):
        return redact_text(value)
    if isinstance(value, tuple):
        return [sanitize_for_logging(item) for item in value]
    if isinstance(value, list):
        result: list[Any] = []
        redact_next = False
        sshpass_command = any(
            Path(str(item)).name == "sshpass" for item in value[:2]
        )
        for item in value:
            text = str(item)
            if redact_next:
                result.append("<redacted>")
                redact_next = False
            elif _SECRET_KEY.search(text.lstrip("-")) or (
                sshpass_command and text == "-p"
            ):
                result.append(text)
                redact_next = True
            else:
                result.append(sanitize_for_logging(item))
        return result
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return redact_text(str(value))


def _append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    encoded = (
        json.dumps(
            sanitize_for_logging(payload),
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    descriptor = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o640)
    try:
        os.write(descriptor, encoded)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _append_csv(path: Path, fields: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    needs_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        if needs_header:
            writer.writeheader()
        writer.writerows(rows)
        file.flush()
        os.fsync(file.fileno())


def _run_command(command: Sequence[str]) -> str:
    result = subprocess.run(
        list(command),
        check=True,
        capture_output=True,
        text=True,
        timeout=15,
    )
    return result.stdout


def _optional_command(
    command: Sequence[str],
    command_runner: Callable[[Sequence[str]], str],
) -> str | None:
    try:
        return command_runner(command).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def collect_runtime_provenance(
    *,
    repo_root: Path | None = None,
    command_runner: Callable[[Sequence[str]], str] = _run_command,
) -> dict[str, Any]:
    """Collect reproducibility metadata without reading arbitrary environment."""
    versions: dict[str, str] = {}
    for package in ("numpy", "safetensors", "torch", "transformers", "warp-lang"):
        try:
            versions[package] = importlib_metadata.version(package)
        except importlib_metadata.PackageNotFoundError:
            continue

    provenance: dict[str, Any] = {
        "packages": versions,
        "gpu_inventory": _optional_command(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,name,driver_version",
                "--format=csv,noheader,nounits",
            ],
            command_runner,
        ),
        "gpu_topology": _optional_command(
            ["nvidia-smi", "topo", "-m"], command_runner
        ),
    }
    if repo_root is not None:
        root = str(Path(repo_root).expanduser().resolve())
        provenance["git"] = {
            "commit": _optional_command(
                ["git", "-C", root, "rev-parse", "HEAD"], command_runner
            ),
            "branch": _optional_command(
                ["git", "-C", root, "branch", "--show-current"], command_runner
            ),
            "status": _optional_command(
                ["git", "-C", root, "status", "--short"], command_runner
            ),
        }
    return provenance


def _parse_csv(output: str) -> list[list[str]]:
    return [
        [column.strip() for column in row]
        for row in csv.reader(output.splitlines())
        if row and any(column.strip() for column in row)
    ]


def _number(value: str, cast: Callable[[float], Any] = float) -> Any:
    cleaned = re.sub(r"\s*(?:MiB|W|MHz|%)\s*$", "", value.strip())
    if cleaned in {"", "N/A", "[Not Supported]"}:
        return 0
    return cast(float(cleaned))


def _process_user(pid: int) -> str:
    try:
        output = subprocess.run(
            ["ps", "-o", "user=", "-p", str(pid)],
            check=True,
            capture_output=True,
            text=True,
            timeout=3,
        ).stdout.strip()
        return output or "<unknown>"
    except (OSError, subprocess.SubprocessError):
        return "<unknown>"


def _pid_is_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def query_gpu_devices(
    command_runner: Callable[[Sequence[str]], str] = _run_command,
) -> tuple[GpuDevice, ...]:
    """Query physical GPU identities and available memory."""
    output = command_runner(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,memory.free,memory.total",
            "--format=csv,noheader,nounits",
        ]
    )
    devices = []
    for row in _parse_csv(output):
        if len(row) != 5:
            raise OperationsError(f"Unexpected nvidia-smi GPU row: {row}")
        devices.append(
            GpuDevice(
                index=int(row[0]),
                uuid=row[1],
                name=row[2],
                memory_free_mib=_number(row[3], int),
                memory_total_mib=_number(row[4], int),
            )
        )
    return tuple(devices)


def query_gpu_processes(
    command_runner: Callable[[Sequence[str]], str] = _run_command,
    *,
    user_lookup: Callable[[int], str] = _process_user,
) -> tuple[GpuProcess, ...]:
    """Query active NVIDIA compute processes."""
    try:
        output = command_runner(
            [
                "nvidia-smi",
                "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory",
                "--format=csv,noheader,nounits",
            ]
        )
    except subprocess.CalledProcessError as error:
        if not (error.stdout or "").strip():
            return ()
        raise
    processes = []
    for row in _parse_csv(output):
        if len(row) != 4:
            raise OperationsError(f"Unexpected nvidia-smi process row: {row}")
        pid = int(row[1])
        processes.append(
            GpuProcess(
                gpu_uuid=row[0],
                pid=pid,
                user=user_lookup(pid),
                process_name=row[2],
                used_memory_mib=_number(row[3], int),
            )
        )
    return tuple(processes)


def _same_process_group_pids() -> set[int]:
    process_group = os.getpgrp()
    members: set[int] = set()
    proc = Path("/proc")
    if not proc.is_dir():
        return {os.getpid()}
    for entry in proc.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            if os.getpgid(int(entry.name)) == process_group:
                members.add(int(entry.name))
        except (OSError, ProcessLookupError):
            continue
    return members


def preflight_selected_gpus(
    selected_devices: Sequence[str | int],
    *,
    minimum_free_mib: int = 2048,
    allowed_pids: Sequence[int] = (),
    allow_current_process_group: bool = True,
    command_runner: Callable[[Sequence[str]], str] = _run_command,
    user_lookup: Callable[[int], str] = _process_user,
) -> GpuPreflightResult:
    """Refuse launch when selected GPUs contain a foreign compute process."""
    if not selected_devices:
        raise ValueError("At least one GPU must be selected")
    devices = query_gpu_devices(command_runner)
    by_index = {str(device.index): device for device in devices}
    by_uuid = {device.uuid: device for device in devices}
    selected: list[GpuDevice] = []
    for token in selected_devices:
        key = str(token).strip()
        device = by_index.get(key) or by_uuid.get(key)
        if device is None:
            raise GpuPreflightError(
                f"Selected GPU {key!r} is not present; available="
                f"{sorted(by_index)}"
            )
        if device not in selected:
            selected.append(device)

    insufficient = [
        device
        for device in selected
        if device.memory_free_mib < minimum_free_mib
    ]
    if insufficient:
        details = ", ".join(
            f"GPU {device.index}: {device.memory_free_mib} MiB free"
            for device in insufficient
        )
        raise GpuPreflightError(
            f"Selected GPUs need at least {minimum_free_mib} MiB free: {details}"
        )

    processes = query_gpu_processes(command_runner, user_lookup=user_lookup)
    allowed = set(int(pid) for pid in allowed_pids)
    allowed.add(os.getpid())
    if allow_current_process_group:
        allowed.update(_same_process_group_pids())
    selected_uuids = {device.uuid for device in selected}
    foreign = [
        process
        for process in processes
        if process.gpu_uuid in selected_uuids and process.pid not in allowed
    ]
    if foreign:
        details = ", ".join(
            f"gpu={process.gpu_uuid} pid={process.pid} user={process.user} "
            f"memory={process.used_memory_mib}MiB process={process.process_name}"
            for process in foreign
        )
        raise GpuPreflightError(
            "Selected GPUs contain foreign compute processes; no process was killed: "
            + details
        )
    return GpuPreflightResult(tuple(selected), tuple(processes))


class GpuMonitor:
    """Append periodic NVIDIA GPU and process samples to CSV files."""

    def __init__(
        self,
        run_dir: Path,
        *,
        interval_seconds: float = 5.0,
        command_runner: Callable[[Sequence[str]], str] = _run_command,
        user_lookup: Callable[[int], str] = _process_user,
    ) -> None:
        if interval_seconds <= 0:
            raise ValueError("GPU monitor interval must be positive")
        self.run_dir = Path(run_dir)
        self.interval_seconds = interval_seconds
        self.command_runner = command_runner
        self.user_lookup = user_lookup
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.last_error: str | None = None

    def sample_once(self) -> None:
        """Collect one GPU and process snapshot."""
        timestamp = utc_now()
        output = self.command_runner(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,name,utilization.gpu,memory.used,"
                "memory.free,memory.total,power.draw,temperature.gpu,clocks.sm",
                "--format=csv,noheader,nounits",
            ]
        )
        rows = []
        for row in _parse_csv(output):
            if len(row) != 10:
                raise OperationsError(f"Unexpected nvidia-smi monitor row: {row}")
            rows.append(
                dict(
                    zip(
                        _CSV_GPU_FIELDS,
                        [
                            timestamp,
                            row[0],
                            row[1],
                            row[2],
                            _number(row[3]),
                            _number(row[4]),
                            _number(row[5]),
                            _number(row[6]),
                            _number(row[7]),
                            _number(row[8]),
                            _number(row[9]),
                        ],
                        strict=True,
                    )
                )
            )
        _append_csv(self.run_dir / "gpu_metrics.csv", _CSV_GPU_FIELDS, rows)

        process_rows = [
            {
                "timestamp_utc": timestamp,
                "gpu_uuid": process.gpu_uuid,
                "pid": process.pid,
                "user": process.user,
                "process_name": process.process_name,
                "used_gpu_memory_mib": process.used_memory_mib,
            }
            for process in query_gpu_processes(
                self.command_runner, user_lookup=self.user_lookup
            )
        ]
        _append_csv(
            self.run_dir / "gpu_processes.csv",
            _CSV_PROCESS_FIELDS,
            process_rows,
        )

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self.sample_once()
                self.last_error = None
            except Exception as error:
                self.last_error = f"{type(error).__name__}: {error}"
                _append_jsonl(
                    self.run_dir / "gpu_monitor_errors.jsonl",
                    {
                        "timestamp_utc": utc_now(),
                        "error": self.last_error,
                    },
                )
            self._stop.wait(self.interval_seconds)

    def start(self) -> None:
        """Start a daemon monitor thread once."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._loop, name="rexpolicy-gpu-monitor", daemon=True
        )
        self._thread.start()

    def close(self) -> None:
        """Stop the monitor without waiting past one sampling interval."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self.interval_seconds + 1.0))


class RunOperations:
    """Own durable per-run logs and generation-boundary stop state."""

    def __init__(
        self,
        run_dir: Path,
        *,
        rank: int,
        world_size: int,
        heartbeat_interval_seconds: float = 30.0,
    ) -> None:
        if rank < 0 or world_size < 1 or rank >= world_size:
            raise ValueError(f"Invalid rank/world_size: {rank}/{world_size}")
        if heartbeat_interval_seconds <= 0:
            raise ValueError("Heartbeat interval must be positive")
        self.run_dir = Path(run_dir).expanduser().resolve()
        self.rank = rank
        self.world_size = world_size
        self.heartbeat_interval_seconds = heartbeat_interval_seconds
        self.logs_dir = self.run_dir / "logs"
        self.heartbeats_dir = self.run_dir / "heartbeats"
        self.pids_dir = self.run_dir / "pids"
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.heartbeats_dir.mkdir(parents=True, exist_ok=True)
        self.pids_dir.mkdir(parents=True, exist_ok=True)
        self.text_log_path = self.logs_dir / f"rank-{rank:05d}.log"
        self.metrics_path = self.run_dir / "metrics.jsonl"
        self._heartbeat_payload: dict[str, Any] = {"status": "initializing"}
        self._heartbeat_lock = threading.Lock()
        self._stop = threading.Event()
        self._signal_stop = threading.Event()
        self._heartbeat_thread: threading.Thread | None = None
        self._gpu_monitor: GpuMonitor | None = None
        self._previous_signal_handlers: dict[int, Any] = {}
        self._lock_path = self.run_dir / "RUNNING.lock"
        self._lock_owned = False
        if self.is_main:
            self._acquire_run_lock()
        atomic_write_json(
            self.pids_dir / f"rank-{rank:05d}.json",
            {
                "rank": rank,
                "world_size": world_size,
                "pid": os.getpid(),
                "pgid": os.getpgrp(),
                "hostname": socket.gethostname(),
                "user": getpass.getuser(),
                "started_at_utc": utc_now(),
            },
        )

    def _acquire_run_lock(self) -> None:
        payload = {
            "pid": os.getpid(),
            "pgid": os.getpgrp(),
            "hostname": socket.gethostname(),
            "started_at_utc": utc_now(),
        }
        encoded = (json.dumps(payload, sort_keys=True) + "\n").encode("utf-8")
        descriptor: int | None = None
        for attempt in range(2):
            try:
                descriptor = os.open(
                    self._lock_path,
                    os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                    0o640,
                )
                break
            except FileExistsError as error:
                try:
                    existing = json.loads(
                        self._lock_path.read_text(encoding="utf-8")
                    )
                except (OSError, json.JSONDecodeError):
                    existing = {"status": "unreadable"}
                owner_pid = existing.get("pid")
                same_host = existing.get("hostname") == socket.gethostname()
                owner_alive = (
                    isinstance(owner_pid, int) and _pid_is_alive(owner_pid)
                )
                if attempt == 0 and same_host and not owner_alive:
                    self._lock_path.unlink(missing_ok=True)
                    continue
                raise OperationsError(
                    f"Run directory is already locked: {self._lock_path} "
                    f"owner={sanitize_for_logging(existing)}"
                ) from error
        if descriptor is None:
            raise OperationsError(f"Could not acquire run lock {self._lock_path}")
        try:
            os.write(descriptor, encoded)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        self._lock_owned = True

    @property
    def is_main(self) -> bool:
        """Whether this rank owns shared run metadata."""
        return self.rank == 0

    @property
    def stop_path(self) -> Path:
        """Path whose existence requests a clean generation-boundary stop."""
        return self.run_dir / "STOP"

    def write_manifest(
        self,
        *,
        config: Mapping[str, Any],
        command: Sequence[str] | None = None,
        environment: Mapping[str, Any] | None = None,
        extra: Mapping[str, Any] | None = None,
        repo_root: Path | None = None,
        command_runner: Callable[[Sequence[str]], str] = _run_command,
    ) -> Path | None:
        """Write rank-zero launch provenance without persisting credentials."""
        if not self.is_main:
            return None
        safe_environment = {
            key: value
            for key, value in (environment or {}).items()
            if key == "CUDA_VISIBLE_DEVICES"
            or key == "PYTHONPATH"
            or key.startswith("NCCL_")
            or key.startswith("TORCH_")
            or key.startswith("REXPOLICY_")
        }
        payload = {
            "schema_version": 1,
            "created_at_utc": utc_now(),
            "hostname": socket.gethostname(),
            "user": getpass.getuser(),
            "platform": platform.platform(),
            "python": sys.version,
            "world_size": self.world_size,
            "command": sanitize_for_logging(list(command or sys.argv)),
            "environment": sanitize_for_logging(safe_environment),
            "config": sanitize_for_logging(config),
            "runtime": sanitize_for_logging(
                collect_runtime_provenance(
                    repo_root=repo_root, command_runner=command_runner
                )
            ),
            "extra": sanitize_for_logging(extra or {}),
        }
        launches_dir = self.run_dir / "launches"
        launches_dir.mkdir(parents=True, exist_ok=True)
        launch_path = launches_dir / (
            f"launch-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}"
            f"-{os.getpid()}-{uuid.uuid4().hex[:8]}.json"
        )
        atomic_write_json(launch_path, payload)
        current_path = self.run_dir / "run_manifest.json"
        atomic_write_json(
            current_path,
            {
                **payload,
                "launch_record": str(launch_path.relative_to(self.run_dir)),
            },
        )
        return current_path

    def log(self, message: str, *, level: str = "INFO", **fields: Any) -> None:
        """Append one redacted human-readable rank log line."""
        record = {
            "timestamp_utc": utc_now(),
            "rank": self.rank,
            "world_size": self.world_size,
            "level": level,
            "message": redact_text(str(message)),
            **sanitize_for_logging(fields),
        }
        with self.text_log_path.open("a", encoding="utf-8") as file:
            file.write(
                f"{record['timestamp_utc']} [{level}] rank={self.rank}/"
                f"{self.world_size} {record['message']}"
            )
            extra = {
                key: value
                for key, value in record.items()
                if key
                not in {"timestamp_utc", "rank", "world_size", "level", "message"}
            }
            if extra:
                file.write(" " + json.dumps(extra, ensure_ascii=False, sort_keys=True))
            file.write("\n")
            file.flush()

    def metric(self, event: str, **values: Any) -> None:
        """Append one rank-zero structured metric event."""
        if not self.is_main:
            return
        _append_jsonl(
            self.metrics_path,
            {
                "timestamp_utc": utc_now(),
                "event": event,
                "rank": self.rank,
                **values,
            },
        )

    def update_heartbeat(self, *, status: str, **values: Any) -> None:
        """Update payload written by the heartbeat thread."""
        with self._heartbeat_lock:
            self._heartbeat_payload = {
                "status": status,
                **sanitize_for_logging(values),
            }
        self._write_heartbeat()

    def _write_heartbeat(self) -> None:
        with self._heartbeat_lock:
            payload = dict(self._heartbeat_payload)
        atomic_write_json(
            self.heartbeats_dir / f"rank-{self.rank:05d}.json",
            {
                "timestamp_utc": utc_now(),
                "rank": self.rank,
                "world_size": self.world_size,
                "pid": os.getpid(),
                **payload,
            },
        )

    def _heartbeat_loop(self) -> None:
        while not self._stop.wait(self.heartbeat_interval_seconds):
            self._write_heartbeat()

    def start_background(
        self,
        *,
        gpu_monitor_interval_seconds: float = 5.0,
        command_runner: Callable[[Sequence[str]], str] = _run_command,
        user_lookup: Callable[[int], str] = _process_user,
    ) -> None:
        """Start heartbeat on every rank and GPU monitoring on rank zero."""
        if self._heartbeat_thread is None or not self._heartbeat_thread.is_alive():
            self._stop.clear()
            self._write_heartbeat()
            self._heartbeat_thread = threading.Thread(
                target=self._heartbeat_loop,
                name=f"rexpolicy-heartbeat-rank-{self.rank}",
                daemon=True,
            )
            self._heartbeat_thread.start()
        if self.is_main and self._gpu_monitor is None:
            self._gpu_monitor = GpuMonitor(
                self.run_dir,
                interval_seconds=gpu_monitor_interval_seconds,
                command_runner=command_runner,
                user_lookup=user_lookup,
            )
            self._gpu_monitor.start()

    def install_generation_boundary_signal_handlers(self) -> None:
        """Turn SIGINT/SIGTERM into a stop request checked between generations."""

        def request_stop(signum: int, _frame: Any) -> None:
            self._signal_stop.set()
            self.log(
                "signal received; stopping at the next generation boundary",
                level="WARNING",
                signal=signum,
            )

        for signum in (signal.SIGINT, signal.SIGTERM):
            self._previous_signal_handlers[signum] = signal.getsignal(signum)
            signal.signal(signum, request_stop)

    def stop_requested(self) -> bool:
        """Return whether STOP exists or a termination signal was received."""
        return self.stop_path.exists() or self._signal_stop.is_set()

    def request_stop(self, reason: str) -> None:
        """Durably request a clean stop without terminating a worker."""
        atomic_write_json(
            self.stop_path,
            {
                "requested_at_utc": utc_now(),
                "requested_by_pid": os.getpid(),
                "reason": redact_text(reason),
            },
        )

    def clear_stop_request(self) -> None:
        """Remove a prior STOP marker when the operator explicitly resumes."""
        self.stop_path.unlink(missing_ok=True)

    def mark_exit(
        self,
        *,
        status: str,
        reason: str,
        completed_generation: int | None = None,
    ) -> None:
        """Write this rank's terminal status."""
        atomic_write_json(
            self.run_dir / f"exit-rank-{self.rank:05d}.json",
            {
                "timestamp_utc": utc_now(),
                "rank": self.rank,
                "status": status,
                "reason": redact_text(reason),
                "completed_generation": completed_generation,
            },
        )
        self.update_heartbeat(
            status=status,
            reason=reason,
            completed_generation=completed_generation,
        )

    def close(self) -> None:
        """Flush background services and restore signal handlers."""
        self._stop.set()
        if self._heartbeat_thread is not None:
            self._heartbeat_thread.join(
                timeout=max(1.0, self.heartbeat_interval_seconds + 1.0)
            )
        if self._gpu_monitor is not None:
            self._gpu_monitor.close()
        for signum, previous in self._previous_signal_handlers.items():
            signal.signal(signum, previous)
        self._previous_signal_handlers.clear()
        if self._lock_owned:
            self._lock_path.unlink(missing_ok=True)
            self._lock_owned = False

    def __enter__(self) -> RunOperations:
        return self

    def __exit__(self, _type: Any, _value: Any, _traceback: Any) -> None:
        self.close()
