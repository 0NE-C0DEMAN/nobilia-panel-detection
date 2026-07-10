"""Command-line entry point for label (sticker) detection.

    python detect_labels.py --rgb rgb_00001.png --out labels.json

Takes an RGB image and writes a JSON of detected labels (oriented polygons, area).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

from label_detector import YoloLabelDetector, to_json


def main(argv=None):
    ap = argparse.ArgumentParser(description="Detect labels (stickers) in an RGB image.")
    ap.add_argument("--rgb", required=True, help="path to the RGB PNG")
    ap.add_argument("--out", default=None, help="output JSON path (default: stdout)")
    ap.add_argument("--weights", default="models/label_seg_best.pt",
                    help="YOLO26-seg weights for label detection")
    ap.add_argument("--conf", type=float, default=0.4, help="confidence threshold")
    ap.add_argument("--imgsz", type=int, default=1280, help="input image size")
    args = ap.parse_args(argv)

    rgb = cv2.imread(args.rgb)
    if rgb is None:
        sys.exit(f"could not read RGB image: {args.rgb}")

    det = YoloLabelDetector(args.weights, conf=args.conf, imgsz=args.imgsz)
    masks = det.detect(rgb)
    result = to_json(masks, image_name=Path(args.rgb).name, image_size=(rgb.shape[1], rgb.shape[0]))

    text = json.dumps(result, indent=2)
    if args.out:
        Path(args.out).write_text(text)
        print(f"detected {len(masks)} labels -> {args.out}")
    else:
        print(text)


if __name__ == "__main__":
    main()
