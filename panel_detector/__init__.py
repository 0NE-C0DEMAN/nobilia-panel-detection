"""Top-layer panel detection for the robotic bin-picking task.

Typical use:

    from panel_detector import load_calibration, detect_panels, to_json

    calib = load_calibration("calib/cam_to_base_current.json")
    det = detect_panels(rgb, depth_mm, calib)
    result = to_json(det, image_name="rgb_00001.png")
"""
from __future__ import annotations

from dataclasses import asdict

from .calibration import Calibration, load_calibration
from .detector import Detection, DetectorParams, Panel, detect_panels

__all__ = [
    "Calibration", "load_calibration",
    "Detection", "DetectorParams", "Panel", "detect_panels",
    "to_json",
]


def to_json(det: Detection, image_name: str, image_size=(640, 400)) -> dict:
    """Serialise a Detection into the client's delivery schema."""
    panels = []
    for pan in det.panels:
        d = asdict(pan)
        panels.append({
            "id": d["id"],
            "top_layer": d["top_layer"],
            "obb": d["obb"],
            "corners": d["corners"],
            "bbox_xywh": d["bbox_xywh"],
            "center_px": d["center_px"],
            "center_base_m": d["center_base_m"],
            "size_mm": d["size_mm"],
            "angle_deg": d["angle_deg"],
            "center_depth_mm": d["center_depth_mm"],
            "height_mm": d["height_mm"],
            "area_px": d["area_px"],
        })
    # Highest panels first - the robot's pick order (off-pose panels have no height).
    panels.sort(key=lambda q: (-int(q["top_layer"]), -(q["height_mm"] or 0)))
    return {
        "image": image_name,
        "image_size": {"width": image_size[0], "height": image_size[1]},
        "off_pose": det.off_pose,
        "units": "mm / m (see field names)",
        "coordinate_frames": {
            "obb / corners / bbox / center_px": "pixels in the 640x400 RGB-D image "
            "(corners = the panel's true rectangle in perspective; center_px = its "
            "physical centre projected to pixels)",
            "center_base_m": "metres in the robot base_link frame - the pick point "
            "(panel's physical centre, from the 3-D fit)",
            "size_mm / angle_deg": "panel [length, width] in mm and long-axis "
            "orientation in the floor plane, from the 3-D fit",
        },
        "num_panels": len(panels),
        "num_top_layer": sum(q["top_layer"] for q in panels),
        "panels": panels,
    }
