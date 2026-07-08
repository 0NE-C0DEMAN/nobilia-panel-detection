"""One-off calibration: estimate the static bin floor plane and save it to
``calib/floor_plane.json`` for the detector to reuse.

Because the camera and bin never move, the floor plane is fixed. We recover a
clean "empty bin" by taking the per-pixel median depth across the dataset (the
panels move around, so they wash out), then fit the floor with constrained RANSAC.

Usage:
    python calibrate_floor.py --data DIR --calib calib/cam_to_base_current.json
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import cv2
import numpy as np

from panel_detector.calibration import load_calibration
from panel_detector.geometry import fit_floor_plane


def empty_bin_depth(depth_paths, dmin=300.0, dmax=2000.0):
    """Per-pixel median depth across all frames = the static bin without panels."""
    stack = None
    for i, p in enumerate(depth_paths):
        d = cv2.imread(str(p), cv2.IMREAD_UNCHANGED).astype(np.float32)
        if stack is None:
            stack = np.full((len(depth_paths), *d.shape), np.nan, np.float32)
        d[(d < dmin) | (d > dmax)] = np.nan
        stack[i] = d
    return np.nan_to_num(np.nanmedian(stack, axis=0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="dir with depth_*.png frames")
    ap.add_argument("--calib", default="calib/cam_to_base_current.json")
    ap.add_argument("--out", default="calib/floor_plane.json")
    args = ap.parse_args()

    calib = load_calibration(args.calib)
    calib.floor_normal = calib.floor_offset = None    # recompute from scratch

    paths = sorted(glob.glob(str(Path(args.data) / "depth_*.png")))
    if not paths:
        raise SystemExit(f"no depth_*.png found in {args.data}")
    print(f"aggregating {len(paths)} depth frames into an empty-bin reference...")

    bg = empty_bin_depth(paths)
    points, valid = calib.backproject(bg)
    normal, offset = fit_floor_plane(points, valid, calib.up_in_base)

    tilt = float(np.degrees(np.arccos(min(abs(normal[2]), 1.0))))
    heights = points[valid] @ normal + offset
    print(f"floor normal = {np.round(normal, 4)}  (tilt {tilt:.1f} deg from base +Z)")
    print(f"empty-bin residual: p50={np.percentile(heights,50)*1000:.0f}mm "
          f"p90={np.percentile(heights,90)*1000:.0f}mm (walls stick up, expected)")

    Path(args.out).write_text(json.dumps(
        {"floor_normal": normal.tolist(), "floor_offset": float(offset),
         "tilt_deg_from_base_up": round(tilt, 2), "n_frames": len(paths)}, indent=2))
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
