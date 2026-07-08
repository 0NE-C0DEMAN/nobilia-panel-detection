"""Top-layer panel detector.

Pipeline (all driven by the depth image + the static camera->base calibration):

  1. back-project depth into the base frame
  2. fit the bin floor plane; measure every pixel's height above it
  3. panel mask   = flat (up-facing) surfaces, above the floor, inside the bin
  4. split the mask into individual panels along height steps / colour edges
  5. fit an oriented bounding box to each panel
  6. flag the ones on the top layer (highest band) - what the robot picks next

The output is a list of panels with an oriented box, the pick centre in both
image pixels and base-frame metres, and the height above the floor.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .calibration import Calibration
from .geometry import fit_floor_plane, surface_normals


@dataclass
class DetectorParams:
    # --- what counts as a panel surface ---
    min_height_m: float = 0.012        # must sit at least this far above the floor
    max_height_m: float = 0.200        # ignore anything taller than a plausible stack
    horizontal_dot: float = 0.55       # |normal . up| above this = flat/up-facing
    # --- instance separation ---
    height_step_m: float = 0.018       # height jump (after smoothing) that marks a panel edge
    min_core_px: int = 400             # a watershed seed must be at least this big
    min_line_len: int = 45             # only straight edges this long split panels
    canny: tuple[int, int] = (40, 120) # for straight-line (panel border) detection
    # --- cleanup / filtering ---
    min_area_px: int = 1500            # drop specks smaller than this
    min_fill_ratio: float = 0.48       # region must fill its box (real panels are rectangles)
    open_ksize: int = 3
    close_ksize: int = 7
    # --- top-layer definition ---
    top_layer_band_m: float = 0.030    # panels within this of the highest = top layer


@dataclass
class Panel:
    id: int
    obb: dict                          # {cx, cy, w, h, angle_deg}
    corners: list                      # 4 (x, y) image points
    bbox_xywh: list                    # axis-aligned [x, y, w, h]
    center_px: list                    # [u, v]
    center_base_m: list                # [X, Y, Z] pick point in base frame
    center_depth_mm: float
    height_mm: float                   # median height above floor
    area_px: int
    top_layer: bool = False
    size_mm: list | None = None        # [length, width] of the panel (metric, from 3D fit)
    angle_deg: float | None = None     # panel long-axis orientation in the floor plane


@dataclass
class Detection:
    panels: list = field(default_factory=list)
    floor_normal: np.ndarray = None
    floor_offset: float = 0.0
    height_map: np.ndarray = None      # metres above floor (nan where invalid)
    panel_mask: np.ndarray = None
    off_pose: bool = False             # frame's camera pose doesn't match calibration
                                       # -> depth geometry unreliable, 2-D boxes only


# --------------------------------------------------------------------------- #
# masking
# --------------------------------------------------------------------------- #
def _bin_footprint(points_base, horizontal, valid, margin=0.03):
    """Axis-aligned base-frame XY bounds of the bin interior.

    Every up-facing surface (floor + panels) lives inside the bin; the walls are
    vertical and excluded, so the XY span of horizontal pixels bounds the bin.
    """
    m = valid & horizontal
    if m.sum() < 200:
        return None
    xs, ys = points_base[..., 0][m], points_base[..., 1][m]
    x1, x2 = np.percentile(xs, [1, 99])
    y1, y2 = np.percentile(ys, [1, 99])
    return (x1 - margin, x2 + margin, y1 - margin, y2 + margin)


def _panel_mask(height, normals, up, valid, points_base, p: DetectorParams):
    horizontal = np.abs(normals @ up) > p.horizontal_dot
    mask = valid & horizontal & (height > p.min_height_m) & (height < p.max_height_m)

    footprint = _bin_footprint(points_base, horizontal, valid)
    if footprint is not None:
        x1, x2, y1, y2 = footprint
        xs, ys = points_base[..., 0], points_base[..., 1]
        mask &= (xs > x1) & (xs < x2) & (ys > y1) & (ys < y2)

    mask = mask.astype(np.uint8)
    if p.open_ksize:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                                np.ones((p.open_ksize, p.open_ksize), np.uint8))
    if p.close_ksize:
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                                np.ones((p.close_ksize, p.close_ksize), np.uint8))
    return mask.astype(bool)


def _edge_map(height, rgb, valid, p: DetectorParams):
    """Boundaries between panels, from two complementary cues:

    * height steps - separate stacked panels (one sits above another),
    * long straight colour edges - separate panels lying at the *same* height
      side by side (their shared border is a long straight line).

    Short colour edges from the printed arrows/barcodes are ignored via the
    line-length filter, so the panels are not over-segmented.
    """
    h = np.nan_to_num(height, nan=0.0).astype(np.float32)
    h = cv2.medianBlur(h, 5)                            # kill depth noise -> stable edges
    gx = cv2.Sobel(h, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(h, cv2.CV_32F, 0, 1, ksize=3)
    edges = (np.hypot(gx, gy) > p.height_step_m) & valid

    gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
    canny = cv2.Canny(gray, *p.canny)
    lines = cv2.HoughLinesP(canny, 1, np.pi / 180, threshold=30,
                            minLineLength=p.min_line_len, maxLineGap=6)
    line_mask = np.zeros_like(edges, dtype=np.uint8)
    if lines is not None:
        for x1, y1, x2, y2 in lines[:, 0]:
            cv2.line(line_mask, (x1, y1), (x2, y2), 1, 2)

    edges = edges | (line_mask.astype(bool) & valid)
    return cv2.dilate(edges.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)


# --------------------------------------------------------------------------- #
# instance separation
# --------------------------------------------------------------------------- #
def _segment_instances(mask, height, edges, p: DetectorParams):
    """Split the panel mask into individual panels.

    Height-step borders are carved out of the mask to break touching/stacked
    panels apart; the resulting cores seed a watershed whose elevation guide is
    the height map, so each panel grows back to its true edge along the height
    ridge that separates it from its neighbour.
    """
    core = (mask & ~edges).astype(np.uint8)
    core = cv2.morphologyEx(core, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    # Seeds = only cores big enough to be a real panel; drops noise fragments so
    # the watershed grows a few clean regions instead of hundreds of specks.
    n, markers, stats, _ = cv2.connectedComponentsWithStats(core, 8)
    seeds = np.zeros_like(markers)
    k = 0
    for lbl in range(1, n):
        if stats[lbl, cv2.CC_STAT_AREA] >= p.min_core_px:
            k += 1
            seeds[markers == lbl] = k
    if k == 0:                                          # nothing sizeable to seed with
        _, labels = cv2.connectedComponents(mask.astype(np.uint8))
        return labels

    ws_markers = seeds + 1                               # seeds start at 2
    ws_markers[~mask] = 1                               # known background
    ws_markers[mask & (seeds == 0)] = 0                # unknown -> filled by watershed

    # Elevation guide: the height map clipped to the panel band so panels get full
    # contrast (walls would otherwise swamp the normalisation) and ridges land on steps.
    hclip = np.clip(np.nan_to_num(height), 0.0, 0.12)
    guide = cv2.cvtColor(cv2.normalize(hclip, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8),
                         cv2.COLOR_GRAY2BGR)
    cv2.watershed(guide, ws_markers)

    labels = np.where(ws_markers > 1, ws_markers - 1, 0)
    labels[~mask] = 0
    return labels


# --------------------------------------------------------------------------- #
# oriented boxes
# --------------------------------------------------------------------------- #
def _normalize_obb(rect) -> dict:
    """cv2.minAreaRect -> a clean {cx, cy, w, h, angle_deg} with w the long side
    and angle the orientation of that long side in (-90, 90]."""
    (cx, cy), (a, b), ang = rect
    if a >= b:
        w, h, angle = a, b, ang
    else:
        w, h, angle = b, a, ang + 90.0
    angle = (angle + 90.0) % 180.0 - 90.0
    return {"cx": round(float(cx), 1), "cy": round(float(cy), 1),
            "w": round(float(w), 1), "h": round(float(h), 1),
            "angle_deg": round(float(angle), 1)}


def _panel_from_region(pid, region, points_base, depth_mm, height, calib):
    contours, _ = cv2.findContours(region.astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    cnt = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(cnt)
    obb = _normalize_obb(rect)
    corners = cv2.boxPoints(rect).astype(int).tolist()
    x, y, bw, bh = cv2.boundingRect(cnt)

    ys, xs = np.where(region)
    cu, cv_ = int(round(obb["cx"])), int(round(obb["cy"]))
    center_base = points_base[ys, xs].mean(axis=0)            # centroid of top surface
    center_depth = float(np.median(depth_mm[ys, xs]))
    med_height = float(np.median(height[ys, xs]))

    return Panel(
        id=pid,
        obb=obb,
        corners=corners,
        bbox_xywh=[int(x), int(y), int(bw), int(bh)],
        center_px=[cu, cv_],
        center_base_m=[round(float(c), 4) for c in center_base],
        center_depth_mm=round(center_depth, 1),
        height_mm=round(med_height * 1000.0, 1),
        area_px=int(region.sum()),
    )


# --------------------------------------------------------------------------- #
# entry point
# --------------------------------------------------------------------------- #
def detect_panels(rgb, depth_mm, calib: Calibration,
                  params: DetectorParams | None = None) -> Detection:
    """Detect top-layer panels in one RGB-D frame (geometric pipeline)."""
    from .calibration import normalize_depth

    p = params or DetectorParams()
    depth_mm = normalize_depth(depth_mm)
    points_base, valid = calib.backproject(depth_mm)
    normals = surface_normals(points_base, valid)
    up = calib.up_in_base

    # Floor orientation: the stable pre-calibrated normal (a per-frame normal fit
    # is unreliable when panels cover most of the floor); fall back to fitting.
    # The level is refined per frame, anchored to the calibrated plane.
    from .geometry import estimate_floor_level

    if calib.has_floor:
        fn, calibrated_level = calib.floor_normal, -calib.floor_offset
    else:
        fn, fo_ = fit_floor_plane(points_base, valid, up)
        calibrated_level = -fo_
    raw = points_base @ fn
    flat = valid & (np.abs(np.einsum("ijk,k->ij", normals, fn)) > 0.9)
    floor_level = estimate_floor_level(raw, flat, calibrated_level)
    fo = -floor_level

    height = raw - floor_level
    height_disp = np.where(valid, height, np.nan)

    mask = _panel_mask(height, normals, up, valid, points_base, p)
    edges = _edge_map(height_disp, rgb, valid, p)
    labels = _segment_instances(mask, height_disp, edges, p)

    panels = []
    for lbl in range(1, labels.max() + 1):
        region = labels == lbl
        if region.sum() < p.min_area_px:
            continue
        panel = _panel_from_region(len(panels) + 1, region, points_base,
                                   depth_mm.astype(np.float64), height, calib)
        if panel is None:
            continue
        # Reject boxes the region does not really fill (bad merges / diagonal spans);
        # a genuine panel is a rectangle and fills most of its oriented box.
        box_area = panel.obb["w"] * panel.obb["h"]
        if box_area <= 0 or panel.area_px / box_area < p.min_fill_ratio:
            continue
        panels.append(panel)

    # top layer = panels whose height is within a band of the tallest panel.
    if panels:
        top_h = max(pan.height_mm for pan in panels)
        for pan in panels:
            pan.top_layer = bool(pan.height_mm >= top_h - p.top_layer_band_m * 1000.0)

    return Detection(panels=panels, floor_normal=fn, floor_offset=fo,
                     height_map=height_disp, panel_mask=mask)
