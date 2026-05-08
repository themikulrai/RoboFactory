"""Hash-pin the calibration NPZs.

Plan v2 C2 S0 adversarial test: byte-mutate one calibration NPZ → assert
`CalibrationCorruptedError` fires before any code can silently use the
corrupted data.

Run::

    python -m unittest robofactory.utils.test_calibration_integrity -v
"""
from __future__ import annotations

import hashlib
import unittest
from pathlib import Path

try:
    import yaml
    HAVE_YAML = True
except ImportError:
    HAVE_YAML = False

from robofactory.utils.paths import CALIB_DIR as CALIBRATION_DIR

MANIFEST_PATH = CALIBRATION_DIR / "MANIFEST.yaml"


class CalibrationCorruptedError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_calibration_integrity() -> None:
    """Raise CalibrationCorruptedError if any NPZ no longer matches its pinned sha256."""
    if not HAVE_YAML:
        raise RuntimeError("PyYAML not installed; cannot verify calibration manifest")
    if not MANIFEST_PATH.is_file():
        return  # no manifest yet → nothing to verify
    manifest = yaml.safe_load(MANIFEST_PATH.read_text())
    for entry in manifest.get("calibration", []):
        npz = Path(entry["npz_path"])
        if not npz.is_file():
            raise CalibrationCorruptedError(
                f"Calibration NPZ missing on disk: {npz}\n"
                "Either restore from backup or remove the entry from MANIFEST.yaml."
            )
        actual = _sha256(npz)
        expected = entry["sha256"]
        if actual != expected:
            raise CalibrationCorruptedError(
                f"Calibration NPZ {npz} hash mismatch:\n"
                f"  expected sha256: {expected}\n"
                f"  actual sha256:   {actual}\n"
                f"Either the file was mutated or the manifest is stale."
            )


class TestCalibrationIntegrity(unittest.TestCase):
    def test_each_pinned_npz_matches_its_sha256(self):
        if not MANIFEST_PATH.is_file():
            self.skipTest(f"no manifest at {MANIFEST_PATH}")
        # If this fails, the bytes on disk no longer match the manifest's pinned sha.
        verify_calibration_integrity()


if __name__ == "__main__":
    unittest.main()
