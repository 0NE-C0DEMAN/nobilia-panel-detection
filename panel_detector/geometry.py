"""Geometric primitives on the base-frame point cloud: per-pixel surface normals
and a robust fit of the bin floor plane.

These are what let us separate flat, horizontal panel surfaces (normal pointing
up) from the tall, near-vertical bin walls (normal pointing sideways), and measure
how high each panel sits above the floor.
"""
from __future__ import annotations

import cv2
import numpy as np


def surface_normals(points_base: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Per-pixel unit surface normals in the base frame.

    Computed from the cross product of the horizontal/vertical image-space
    gradients of the point map. The point map is lightly smoothed first so the
    normals are stable against depth noise (edges are preserved well enough for
    our flat panels).

    Returns (H, W, 3) float; normals are oriented to point towards +Z (up).
    """
    pts = points_base.copy()
    # Smooth only where valid so holes do not bleed across edges.
    smooth = cv2.GaussianBlur(pts, (5, 5), 0)

    dzdx = cv2.Sobel(smooth, cv2.CV_64F, 1, 0, ksize=3)
    dzdy = cv2.Sobel(smooth, cv2.CV_64F, 0, 1, ksize=3)
    normals = np.cross(dzdx, dzdy)

    norm = np.linalg.norm(normals, axis=-1, keepdims=True)
    normals = np.divide(normals, norm, out=np.zeros_like(normals), where=norm > 1e-9)

    # Orient consistently towards +Z.
    flip = normals[..., 2] < 0
    normals[flip] *= -1.0
    normals[~valid] = 0.0
    return normals


def _fit_plane_tls(pts: np.ndarray) -> tuple[np.ndarray, float]:
    """Total-least-squares plane through a set of 3-D points.

    Returns (normal, offset) with the plane defined by ``normal . x + offset = 0``
    and ``||normal|| == 1``.
    """
    centroid = pts.mean(axis=0)
    q = pts - centroid
    # Smallest-eigenvector of the covariance is the plane normal.
    _, _, vt = np.linalg.svd(q.T @ q)
    normal = vt[-1]
    offset = -float(normal @ centroid)
    return normal, offset


def fit_floor_plane(
    points_base: np.ndarray,
    valid: np.ndarray,
    up: np.ndarray,
    near_up_dot: float = 0.90,
    inlier_thresh_m: float = 0.008,
    iterations: int = 2000,
    refine_rounds: int = 5,
    seed: int = 0,
) -> tuple[np.ndarray, float]:
    """Robustly fit the bin floor plane in the base frame with constrained RANSAC.

    The floor is the largest *near-horizontal* plane (its normal is within
    ``arccos(near_up_dot)`` of base up). Constraining to near-up rejects the tall
    bin walls, whose normals point sideways, and the plane with the most inliers
    is the bin bottom. Best fit is then refined with total-least-squares.

    Note the real floor sits a little off base +Z (the base frame is not perfectly
    gravity-aligned), so we do NOT assume a normal of exactly [0, 0, 1].

    Returns (normal, offset); normal points up, plane is ``normal . x + offset = 0``.
    """
    pts = points_base[valid]
    if len(pts) < 500:
        raise ValueError("too few valid points to fit a floor plane")

    rng = np.random.RandomState(seed)
    sample = pts[rng.choice(len(pts), min(30000, len(pts)), replace=False)]

    best, best_inliers = None, 0
    for _ in range(iterations):
        tri = sample[rng.randint(0, len(sample), 3)]
        normal = np.cross(tri[1] - tri[0], tri[2] - tri[0])
        nn = np.linalg.norm(normal)
        if nn < 1e-9:
            continue
        normal = normal / nn
        if normal @ up < 0:
            normal = -normal
        if normal @ up < near_up_dot:              # reject walls / tilted surfaces
            continue
        offset = -float(normal @ tri[0])
        inliers = int((np.abs(sample @ normal + offset) < inlier_thresh_m).sum())
        if inliers > best_inliers:
            best, best_inliers = (normal, offset), inliers

    if best is None:
        raise RuntimeError("RANSAC failed to find a near-horizontal floor plane")

    normal, offset = best
    for _ in range(refine_rounds):
        keep = np.abs(pts @ normal + offset) < inlier_thresh_m
        if keep.sum() < 200:
            break
        normal, offset = _fit_plane_tls(pts[keep])
        if normal @ up < 0:
            normal, offset = -normal, -offset
    return normal, offset


def height_above_floor(
    points_base: np.ndarray, floor_normal: np.ndarray, floor_offset: float
) -> np.ndarray:
    """Signed perpendicular height (metres) of every pixel above the floor plane."""
    return points_base @ floor_normal + floor_offset


def estimate_floor_level(
    raw: np.ndarray,
    flat: np.ndarray,
    calibrated_level: float,
    tight_band_m: float = 0.008,
    min_tight_px: int = 1500,
) -> float:
    """Per-frame floor level along the calibrated floor normal.

    The rig is static, so the floor is fixed at the calibrated level. We only
    *refine* it from a **tight** band around the calibrated level (tighter than a
    panel thickness, ~18 mm), so panel surfaces can never vote it upward.

    If that tight band is empty, the floor simply isn't visible in this frame
    (the panels cover the whole bin bottom). We then keep the calibrated level -
    the floor hasn't moved, it's just hidden. We deliberately do NOT search wider
    and snap to the lowest panel layer, which would drag the level up ~40 mm and
    make every height read low. A genuinely off-pose frame (camera moved) is
    handled separately, by detecting surface that reads *below* the calibrated
    floor - which cannot happen while the camera is static.
    """
    vals = raw[flat] - calibrated_level
    tight = vals[np.abs(vals) < tight_band_m]
    if tight.size >= min_tight_px:
        return calibrated_level + float(np.median(tight))
    return calibrated_level                    # floor hidden -> trust the static calibration
