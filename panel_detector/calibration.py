"""Camera calibration: intrinsics, the static camera->base transform, and
back-projection of a depth image into the robot base frame.

The client supplies a JSON file (``cam_to_base_current.json``) containing:
  * ``T_base_from_camera_4x4`` - homogeneous transform, base_point = T @ cam_point
  * ``color_intrinsics`` / ``depth_intrinsics`` - pinhole fx, fy, cx, cy
  * ``depth_encoding`` - "16-bit PNG, raw millimetres; invalid pixels are 0 and 65535"

The depth image is aligned to the colour frame (both 640x400), so we back-project
with the colour intrinsics.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Depth values outside this range (millimetres) are treated as invalid. The bin
# scene lives at roughly 0.7-1.6 m from the camera; anything outside this band is
# a drop-out (0 / 65535) or a flying-pixel artefact.
DEPTH_MIN_MM = 300.0
DEPTH_MAX_MM = 2000.0


def normalize_depth(depth_mm: np.ndarray) -> np.ndarray:
    """Return depth as a 2-D array.

    Guards against readers that hand back an (H, W, 1) or (H, W, 3) array —
    notably, importing ultralytics monkeypatches cv2.imread and 16-bit PNGs can
    come back with a trailing channel afterwards.
    """
    depth_mm = np.asarray(depth_mm)
    if depth_mm.ndim == 3:
        depth_mm = depth_mm[..., 0]
    return depth_mm


@dataclass
class Calibration:
    """Everything needed to map a depth pixel to a 3-D point in the base frame."""

    T_base_from_cam: np.ndarray          # (4, 4)
    color_intrinsics: dict               # {fx, fy, cx, cy}
    depth_intrinsics: dict
    depth_min_mm: float = DEPTH_MIN_MM
    depth_max_mm: float = DEPTH_MAX_MM
    # Optional pre-calibrated floor plane (normal . x + offset = 0, normal up).
    # The rig is static, so this is fixed and best computed once via calibrate_floor.py.
    floor_normal: np.ndarray | None = None
    floor_offset: float | None = None

    @property
    def has_floor(self) -> bool:
        return self.floor_normal is not None and self.floor_offset is not None

    @property
    def up_in_base(self) -> np.ndarray:
        """Base-frame 'up' axis. base_link has +Z up, confirmed by the fact that
        the fitted bin floor normal comes out ~[0, 0, 1]."""
        return np.array([0.0, 0.0, 1.0])

    def valid_mask(self, depth_mm: np.ndarray) -> np.ndarray:
        """Boolean mask of usable depth pixels."""
        return (depth_mm >= self.depth_min_mm) & (depth_mm <= self.depth_max_mm)

    def backproject(self, depth_mm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Back-project a depth image into the base frame.

        Returns
        -------
        points_base : (H, W, 3) float64
            3-D point per pixel in the base frame (metres). Invalid pixels are 0.
        valid : (H, W) bool
            Which pixels hold a real measurement.
        """
        depth_mm = normalize_depth(depth_mm).astype(np.float64)
        h, w = depth_mm.shape
        valid = self.valid_mask(depth_mm)

        K = self.color_intrinsics
        uu, vv = np.meshgrid(np.arange(w), np.arange(h))
        z = depth_mm / 1000.0                       # mm -> m (camera optical z)
        x = (uu - K["cx"]) / K["fx"] * z
        y = (vv - K["cy"]) / K["fy"] * z

        pts_cam = np.stack([x, y, z, np.ones_like(z)], axis=-1)   # (H, W, 4)
        pts_base = pts_cam @ self.T_base_from_cam.T               # (H, W, 4)
        pts_base = pts_base[..., :3]
        pts_base[~valid] = 0.0
        return pts_base, valid


def load_calibration(path: str | Path, floor_path: str | Path | None = None) -> Calibration:
    """Load a Calibration from the client's cam_to_base JSON file.

    If ``floor_path`` is given, or a ``floor_plane.json`` sits next to the calib
    file, the pre-calibrated static floor plane is loaded too.
    """
    path = Path(path)
    data = json.loads(path.read_text())
    calib = Calibration(
        T_base_from_cam=np.asarray(data["T_base_from_camera_4x4"], dtype=np.float64),
        color_intrinsics={k: float(data["color_intrinsics"][k]) for k in ("fx", "fy", "cx", "cy")},
        depth_intrinsics={k: float(data["depth_intrinsics"][k]) for k in ("fx", "fy", "cx", "cy")},
    )

    if floor_path is None:
        default = path.with_name("floor_plane.json")
        floor_path = default if default.exists() else None
    if floor_path is not None:
        fp = json.loads(Path(floor_path).read_text())
        calib.floor_normal = np.asarray(fp["floor_normal"], dtype=np.float64)
        calib.floor_offset = float(fp["floor_offset"])
    return calib
