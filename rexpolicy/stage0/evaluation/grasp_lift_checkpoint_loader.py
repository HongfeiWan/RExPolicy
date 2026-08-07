"""Fail-closed model-only loading for formal Grasp-Lift validation."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from rexpolicy.stage0.checkpoint import (
    STAGE0_CHECKPOINT_SCHEMA_VERSION,
    Stage0CheckpointIntegrityError,
    Stage0CheckpointManager,
    Stage0CheckpointMismatchError,
    file_sha256,
)
from rexpolicy.stage0.grasp_lift_no_z import (
    GraspLiftNoZConfig,
    build_grasp_lift_no_z_policy,
)
from rexpolicy.stage0.runtime import Stage0FlowPolicy
from rexpolicy.stage0.types import canonical_fingerprint


GRASP_LIFT_EVALUATION_CHECKPOINT_SCHEMA_ID = (
    "rexpolicy/stage0-grasp-lift-no-z-evaluation-checkpoint/v1"
)
GRASP_LIFT_NO_Z_MODEL_COMPONENT = "no_z_policy"
GRASP_LIFT_NO_Z_MODEL_RELATIVE_PATH = "models/no_z_policy.pt"
_EXPECTED_MODEL_TYPE = "rexpolicy.stage0.runtime.Stage0FlowPolicy"
_PREFLIGHT_CHECKPOINT_KEYS = {
    "checkpoint_name",
    "checkpoint_sha256",
    "checksums_file_sha256",
    "complete_file_sha256",
    "manifest_file_sha256",
    "model_payload_sha256",
    "model_schema_sha256",
    "sequence",
}


def _require_sha256(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _exact_mapping(value: Any, keys: set[str], context: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise Stage0CheckpointIntegrityError(f"{context} must be an object")
    actual = set(value)
    if actual != keys:
        raise Stage0CheckpointIntegrityError(
            f"{context} fields changed: missing={sorted(keys - actual)}, "
            f"unexpected={sorted(actual - keys)}"
        )
    return dict(value)


def _require_real_absolute_directory(path: str | Path, context: str) -> Path:
    requested = Path(path).expanduser()
    if not requested.is_absolute():
        raise Stage0CheckpointIntegrityError(f"{context} must be absolute")
    if requested.is_symlink() or not requested.is_dir():
        raise Stage0CheckpointIntegrityError(
            f"{context} is missing, not a directory, or a symlink"
        )
    resolved = requested.resolve(strict=True)
    if resolved != requested:
        raise Stage0CheckpointIntegrityError(f"{context} path drifted")
    return resolved


def _require_real_checkpoint_tree(path: Path) -> None:
    if path.is_symlink() or not path.is_dir():
        raise Stage0CheckpointIntegrityError(
            "requested checkpoint is missing, not a directory, or a symlink"
        )
    if path.resolve(strict=True) != path:
        raise Stage0CheckpointIntegrityError("requested checkpoint path drifted")
    try:
        entries = tuple(path.rglob("*"))
    except OSError as error:
        raise Stage0CheckpointIntegrityError(
            f"cannot inspect checkpoint tree: {error}"
        ) from error
    if any(entry.is_symlink() for entry in entries):
        raise Stage0CheckpointIntegrityError("checkpoint tree contains a symlink")


def _model_schema_sha256(state_dict: Mapping[str, Any]) -> str:
    entries: list[dict[str, Any]] = []
    for name, value in state_dict.items():
        if not isinstance(name, str) or not torch.is_tensor(value):
            raise Stage0CheckpointIntegrityError(
                "model state_dict must map string names to tensors"
            )
        entries.append(
            {"dtype": str(value.dtype), "name": name, "shape": list(value.shape)}
        )
    if not entries:
        raise Stage0CheckpointIntegrityError("model state_dict cannot be empty")
    return canonical_fingerprint(entries)


def _torch_load_model_payload(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError as error:  # pragma: no cover - current formal Torch supports it
        raise Stage0CheckpointIntegrityError(
            "formal model loading requires Torch weights_only=True support"
        ) from error
    except Exception as error:
        raise Stage0CheckpointIntegrityError(
            f"cannot safely load no-z model payload: {error}"
        ) from error


def _validate_preflight_checkpoint(
    value: Mapping[str, Any],
    *,
    step: int,
) -> dict[str, Any]:
    record = _exact_mapping(
        value,
        _PREFLIGHT_CHECKPOINT_KEYS,
        "preflight checkpoint commitment",
    )
    expected_name = Stage0CheckpointManager.checkpoint_name(step)
    if record["sequence"] != step or record["checkpoint_name"] != expected_name:
        raise Stage0CheckpointMismatchError(
            "preflight checkpoint commitment does not match the requested step"
        )
    for name in (
        "checkpoint_sha256",
        "checksums_file_sha256",
        "complete_file_sha256",
        "manifest_file_sha256",
        "model_payload_sha256",
        "model_schema_sha256",
    ):
        _require_sha256(record[name], f"preflight {name}")
    expected_checkpoint_sha256 = canonical_fingerprint(
        {
            "checksums_file_sha256": record["checksums_file_sha256"],
            "complete_file_sha256": record["complete_file_sha256"],
            "manifest_file_sha256": record["manifest_file_sha256"],
            "model_payload_sha256": record["model_payload_sha256"],
            "sequence": step,
        }
    )
    if record["checkpoint_sha256"] != expected_checkpoint_sha256:
        raise Stage0CheckpointIntegrityError(
            "preflight checkpoint self-commitment changed"
        )
    return record


def _policy_device(policy: Stage0FlowPolicy) -> torch.device:
    devices = {parameter.device for parameter in policy.parameters()}
    devices.update(buffer.device for buffer in policy.buffers())
    if len(devices) != 1:
        raise RuntimeError("loaded no-z policy does not occupy exactly one device")
    return next(iter(devices))


def load_grasp_lift_no_z_evaluation_checkpoint(
    run_directory: str | Path,
    *,
    step: int,
    config: GraspLiftNoZConfig,
    device: torch.device | str,
    preflight_checkpoint: Mapping[str, Any],
) -> tuple[Stage0FlowPolicy, dict[str, Any]]:
    """Load one preflight-bound policy without restoring training state.

    The caller must pass the complete checkpoint commitment captured by the
    tensor-free validation preflight.  This function never deserializes an
    optimizer, rank RNG state, training shard, or validation shard.
    """

    if not isinstance(config, GraspLiftNoZConfig):
        raise TypeError("config must be GraspLiftNoZConfig")
    if (
        type(step) is not int
        or step < config.checkpoint_every_steps
        or step > config.total_steps
        or step % config.checkpoint_every_steps
    ):
        raise ValueError("step must be one fixed non-zero no-z candidate checkpoint")
    commitment = _validate_preflight_checkpoint(
        preflight_checkpoint,
        step=step,
    )
    payload_commitment = commitment["model_payload_sha256"]
    schema_commitment = commitment["model_schema_sha256"]

    root = _require_real_absolute_directory(run_directory, "run_directory")
    checkpoints_directory = _require_real_absolute_directory(
        root / "checkpoints",
        "run checkpoints directory",
    )
    if checkpoints_directory.parent != root:
        raise Stage0CheckpointIntegrityError("run checkpoints path drifted")
    checkpoint = checkpoints_directory / Stage0CheckpointManager.checkpoint_name(step)
    _require_real_checkpoint_tree(checkpoint)
    if checkpoint.parent != checkpoints_directory:
        raise Stage0CheckpointIntegrityError("requested checkpoint path drifted")
    committed_files = {
        "checksums_file_sha256": checkpoint / "checksums.json",
        "complete_file_sha256": checkpoint / "COMPLETE",
        "manifest_file_sha256": checkpoint / "manifest.json",
        "model_payload_sha256": checkpoint / GRASP_LIFT_NO_Z_MODEL_RELATIVE_PATH,
    }
    for commitment_name, path in committed_files.items():
        if path.is_symlink() or not path.is_file():
            raise Stage0CheckpointIntegrityError(
                f"committed checkpoint file is missing or a symlink: {path.name}"
            )
        if file_sha256(path) != commitment[commitment_name]:
            raise Stage0CheckpointMismatchError(
                f"{commitment_name} does not match the preflight commitment"
            )

    manager = Stage0CheckpointManager(root)
    manifest = manager.verify(checkpoint)
    if (
        manifest["sequence"] != step
        or manifest["checkpoint_name"] != checkpoint.name
        or manifest["steps"] != {"global": step, "no_z_policy": step}
    ):
        raise Stage0CheckpointIntegrityError(
            "checkpoint identity does not match the requested no-z step"
        )
    expected_phase = "complete" if step == config.total_steps else "no_z_policy"
    if manifest["phase"] != expected_phase:
        raise Stage0CheckpointIntegrityError("checkpoint phase changed")
    if set(manifest["models"]) != {GRASP_LIFT_NO_Z_MODEL_COMPONENT}:
        raise Stage0CheckpointIntegrityError("checkpoint model inventory changed")
    model_manifest = _exact_mapping(
        manifest["models"][GRASP_LIFT_NO_Z_MODEL_COMPONENT],
        {"file", "schema_sha256", "type"},
        "no-z model manifest",
    )
    if (
        model_manifest["file"] != GRASP_LIFT_NO_Z_MODEL_RELATIVE_PATH
        or model_manifest["type"] != _EXPECTED_MODEL_TYPE
    ):
        raise Stage0CheckpointIntegrityError("no-z model manifest identity changed")
    if model_manifest["schema_sha256"] != schema_commitment:
        raise Stage0CheckpointMismatchError(
            "no-z model manifest schema does not match the preflight commitment"
        )

    payload_path = checkpoint / GRASP_LIFT_NO_Z_MODEL_RELATIVE_PATH
    if (
        payload_path.is_symlink()
        or not payload_path.is_file()
        or payload_path.resolve(strict=True) != payload_path
        or payload_path.parent.parent != checkpoint
    ):
        raise Stage0CheckpointIntegrityError("no-z model payload path drifted")
    observed_payload_sha256 = file_sha256(payload_path)
    if observed_payload_sha256 != payload_commitment:
        raise Stage0CheckpointMismatchError(
            "no-z model payload does not match the preflight commitment"
        )

    payload = _torch_load_model_payload(payload_path)
    if file_sha256(payload_path) != observed_payload_sha256:
        raise Stage0CheckpointIntegrityError("no-z model payload changed while loading")
    payload_record = _exact_mapping(
        payload,
        {"schema_version", "name", "schema_sha256", "state_dict"},
        "no-z model payload",
    )
    if (
        payload_record["schema_version"] != STAGE0_CHECKPOINT_SCHEMA_VERSION
        or payload_record["name"] != GRASP_LIFT_NO_Z_MODEL_COMPONENT
        or payload_record["schema_sha256"] != schema_commitment
    ):
        raise Stage0CheckpointIntegrityError("no-z model payload identity changed")
    state_dict = payload_record["state_dict"]
    if not isinstance(state_dict, Mapping):
        raise Stage0CheckpointIntegrityError("no-z model state_dict must be an object")
    if _model_schema_sha256(state_dict) != schema_commitment:
        raise Stage0CheckpointIntegrityError("no-z model payload schema changed")

    policy, freeze_record = build_grasp_lift_no_z_policy(config, device=device)
    if _model_schema_sha256(policy.state_dict()) != schema_commitment:
        raise Stage0CheckpointMismatchError(
            "current no-z policy schema does not match the preflight commitment"
        )
    try:
        incompatible = policy.load_state_dict(state_dict, strict=True)
    except (RuntimeError, TypeError, ValueError) as error:
        raise Stage0CheckpointIntegrityError(
            f"cannot strictly restore no-z policy: {error}"
        ) from error
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise Stage0CheckpointIntegrityError(
            "strict no-z restore returned incompatible state keys"
        )
    frozen = [
        name
        for name, parameter in policy.named_parameters()
        if not parameter.requires_grad
    ]
    if frozen != freeze_record["frozen_parameter_names"] or not frozen:
        raise RuntimeError("loaded no-z policy frozen parameter set changed")
    policy.eval()
    policy_device = _policy_device(policy)

    final_committed_hashes: dict[str, str] = {}
    for commitment_name, path in committed_files.items():
        if path.is_symlink() or not path.is_file():
            raise Stage0CheckpointIntegrityError(
                f"committed checkpoint file changed while loading: {path.name}"
            )
        observed = file_sha256(path)
        if observed != commitment[commitment_name]:
            raise Stage0CheckpointIntegrityError(
                f"{commitment_name} changed while loading the no-z policy"
            )
        final_committed_hashes[commitment_name] = observed

    record: dict[str, Any] = {
        "checkpoint": {
            "checksums_file_sha256": final_committed_hashes["checksums_file_sha256"],
            "complete_file_sha256": final_committed_hashes["complete_file_sha256"],
            "manifest_file_sha256": final_committed_hashes["manifest_file_sha256"],
            "name": checkpoint.name,
            "path": str(checkpoint),
            "preflight_checkpoint_sha256": commitment["checkpoint_sha256"],
            "step": step,
        },
        "config_sha256": config.sha256,
        "device": str(policy_device),
        "freeze_record": freeze_record,
        "model": {
            "component": GRASP_LIFT_NO_Z_MODEL_COMPONENT,
            "payload_relative_path": GRASP_LIFT_NO_Z_MODEL_RELATIVE_PATH,
            "payload_sha256": observed_payload_sha256,
            "schema_sha256": schema_commitment,
            "state_tensor_count": len(state_dict),
            "type": _EXPECTED_MODEL_TYPE,
        },
        "read_scope": {
            "model_payload_loaded": True,
            "optimizer_payload_loaded": False,
            "rank_payload_loaded": False,
            "validation_data_loaded": False,
        },
        "run_directory": str(root),
        "schema_id": GRASP_LIFT_EVALUATION_CHECKPOINT_SCHEMA_ID,
        "torch_load_weights_only": True,
    }
    record["self_sha256"] = canonical_fingerprint(record)
    return policy, record


__all__ = [
    "GRASP_LIFT_EVALUATION_CHECKPOINT_SCHEMA_ID",
    "GRASP_LIFT_NO_Z_MODEL_COMPONENT",
    "GRASP_LIFT_NO_Z_MODEL_RELATIVE_PATH",
    "load_grasp_lift_no_z_evaluation_checkpoint",
]
