"""CPU-only tests for fail-closed scene GLB validation."""

from __future__ import annotations

import struct
import tempfile
import unittest
from pathlib import Path

from rexpolicy.scene_asset import validate_scene_glb


def _write_glb(
    path: Path, *, version: int = 2, declared_size: int | None = None
) -> None:
    payload = b"JSON" + b"{}  "
    size = 12 + 8 + len(payload)
    header = struct.pack(
        "<4sII", b"glTF", version, size if declared_size is None else declared_size
    )
    chunk = struct.pack("<II", len(payload), 0x4E4F534A)
    path.write_bytes(header + chunk + payload)


class SceneAssetTest(unittest.TestCase):
    def test_valid_binary_gltf_resolves(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "scene.glb"
            _write_glb(path)

            self.assertEqual(validate_scene_glb(path), path.resolve())

    def test_missing_scene_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "missing.glb"
            with self.assertRaisesRegex(FileNotFoundError, "does not exist"):
                validate_scene_glb(path)

    def test_zero_filled_placeholder_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "scene.glb"
            path.write_bytes(b"\0" * 64)
            with self.assertRaisesRegex(ValueError, "not a binary glTF"):
                validate_scene_glb(path)

    def test_declared_length_must_match_file(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "scene.glb"
            _write_glb(path, declared_size=4096)
            with self.assertRaisesRegex(ValueError, "length mismatch"):
                validate_scene_glb(path)

    def test_only_glb_version_two_is_supported(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "scene.glb"
            _write_glb(path, version=1)
            with self.assertRaisesRegex(ValueError, "unsupported.*version"):
                validate_scene_glb(path)


if __name__ == "__main__":
    unittest.main()
