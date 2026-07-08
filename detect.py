"""Command-line entry point for panel detection.

    python detect.py --rgb rgb_00001.png --depth depth_00001.png \
                     --calib calib/cam_to_base_current.json --out panels.json

Takes one aligned RGB + depth pair and writes a JSON of the top-layer panels
(oriented boxes, pick centre in pixels and in base-frame metres, height above the
floor). Pass --overlay to also save a visualisation.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

from panel_detector import detect_panels, load_calibration, to_json


def main(argv=None):
    ap = argparse.ArgumentParser(description="Detect top-layer panels in an RGB-D frame.")
    ap.add_argument("--rgb", required=True, help="path to the RGB PNG")
    ap.add_argument("--depth", required=True, help="path to the aligned 16-bit depth PNG (mm)")
    ap.add_argument("--calib", default="calib/cam_to_base_current.json",
                    help="camera->base transform + intrinsics JSON")
    ap.add_argument("--floor", default=None,
                    help="floor_plane.json (defaults to one beside --calib)")
    ap.add_argument("--out", default=None, help="output JSON path (default: stdout)")
    ap.add_argument("--overlay", default=None, help="optional path to save a box overlay PNG")
    ap.add_argument("--method", choices=["yolo", "geometric"], default="yolo",
                    help="yolo = trained model + depth (default); geometric = depth-only fallback")
    ap.add_argument("--weights", default="models/panel_seg_v5_l960.pt",
                    help="YOLOv8-seg weights for --method yolo")
    args = ap.parse_args(argv)

    rgb = cv2.imread(args.rgb)
    depth = cv2.imread(args.depth, cv2.IMREAD_UNCHANGED)
    if rgb is None:
        sys.exit(f"could not read RGB image: {args.rgb}")
    if depth is None:
        sys.exit(f"could not read depth image: {args.depth}")

    calib = load_calibration(args.calib, floor_path=args.floor)
    if args.method == "yolo":
        from panel_detector.yolo_detector import YoloPanelDetector
        det = YoloPanelDetector(args.weights).detect(rgb, depth, calib)
    else:
        det = detect_panels(rgb, depth, calib)
    result = to_json(det, image_name=Path(args.rgb).name,
                     image_size=(rgb.shape[1], rgb.shape[0]))

    text = json.dumps(result, indent=2)
    if args.out:
        Path(args.out).write_text(text)
        print(f"wrote {result['num_panels']} panels "
              f"({result['num_top_layer']} top-layer) -> {args.out}")
    else:
        print(text)

    if args.overlay:
        from panel_detector.visualize import draw_panels
        cv2.imwrite(args.overlay, draw_panels(rgb, det))
        print(f"overlay -> {args.overlay}")


if __name__ == "__main__":
    main()
