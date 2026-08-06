"""Fail-closed validation for optional binary glTF scene assets."""

from __future__ import annotations

import struct
from pathlib import Path


_GLB_HEADER = struct.Struct("<4sII")
_GLB_MAGIC = b"glTF"
_GLB_VERSION = 2
_MINIMUM_GLB_SIZE = 20  # 12-byte header plus one 8-byte chunk header.


def validate_scene_glb(path: str | Path) -> Path:
    """Resolve and validate a GLB before a visual runtime is constructed.

    Scene visuals are optional, but an explicitly enabled scene is part of the
    observation contract.  Missing, truncated, placeholder, and non-GLB files
    therefore fail instead of silently producing the renderer's gray clear
    color.
    """

    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"scene GLB does not exist: {resolved}")

    size = resolved.stat().st_size
    if size < _MINIMUM_GLB_SIZE:
        raise ValueError(
            f"scene GLB is truncated: {resolved} has {size} bytes, "
            f"expected at least {_MINIMUM_GLB_SIZE}"
        )

    with resolved.open("rb") as handle:
        header = handle.read(_GLB_HEADER.size)
    magic, version, declared_size = _GLB_HEADER.unpack(header)
    if magic != _GLB_MAGIC:
        raise ValueError(f"scene asset is not a binary glTF file: {resolved}")
    if version != _GLB_VERSION:
        raise ValueError(
            f"unsupported scene GLB version {version}: {resolved}; "
            f"expected {_GLB_VERSION}"
        )
    if declared_size != size:
        raise ValueError(
            f"scene GLB length mismatch: {resolved} declares {declared_size} "
            f"bytes but contains {size}"
        )
    return resolved


__all__ = ["validate_scene_glb"]
