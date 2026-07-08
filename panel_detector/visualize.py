"""Debug/QA overlays: draw detected panels on the RGB frame and render the
height-above-floor map. Used by evaluate.py to eyeball detection quality.
"""
from __future__ import annotations

import cv2
import numpy as np

from .detector import Detection

TOP_COLOR = (0, 220, 0)        # green - top-layer panels (what the robot picks)
LOWER_COLOR = (0, 165, 255)    # orange - lower panels


def draw_panels(rgb: np.ndarray, det: Detection) -> np.ndarray:
    """Draw oriented boxes, ids and pick centres over a copy of the RGB frame."""
    vis = rgb.copy()
    for pan in det.panels:
        color = TOP_COLOR if pan.top_layer else LOWER_COLOR
        pts = np.array(pan.corners, dtype=np.int32)
        cv2.polylines(vis, [pts], True, color, 2)
        u, v = pan.center_px
        cv2.circle(vis, (u, v), 3, color, -1)
        label = f"{pan.id}:{pan.height_mm:.0f}mm" if pan.height_mm is not None else f"{pan.id}"
        cv2.putText(vis, label, (u - 20, v - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    return vis


def colorize_height(det: Detection, max_mm: float = 120.0) -> np.ndarray:
    """JET colour map of height above the floor (0..max_mm), black where invalid."""
    h = det.height_map
    valid = np.isfinite(h)
    norm = np.clip(np.nan_to_num(h) * 1000.0 / max_mm, 0, 1)
    heat = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heat[~valid] = 0
    return heat


def side_by_side(rgb: np.ndarray, det: Detection) -> np.ndarray:
    """RGB with boxes next to the height map - the standard QA view."""
    return np.hstack([draw_panels(rgb, det), colorize_height(det)])
