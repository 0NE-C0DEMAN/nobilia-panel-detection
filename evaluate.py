"""Batch-run the detector over a dataset of RGB-D frames for QA.

Writes a JSON and a box overlay per frame, a montage, and prints summary stats.

    python evaluate.py --data DIR --calib calib/cam_to_base_current.json --out outputs
"""
from __future__ import annotations

import argparse
import glob
import json
import warnings
from pathlib import Path

import cv2
import numpy as np

from panel_detector import detect_panels, load_calibration, to_json
from panel_detector.visualize import draw_panels

warnings.filterwarnings("ignore")           # numpy nan-slice noise on empty depth rows


def _pair_paths(data_dir):
    rgb = sorted(glob.glob(str(Path(data_dir) / "rgb_*.png")))
    dep = sorted(glob.glob(str(Path(data_dir) / "depth_*.png")))
    if len(rgb) != len(dep):
        raise SystemExit(f"rgb/depth count mismatch: {len(rgb)} vs {len(dep)}")
    return list(zip(rgb, dep))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="dir with rgb_*.png + depth_*.png")
    ap.add_argument("--calib", default="calib/cam_to_base_current.json")
    ap.add_argument("--floor", default=None)
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--montage", type=int, default=12, help="frames in the montage grid")
    ap.add_argument("--method", choices=["yolo", "geometric"], default="yolo")
    ap.add_argument("--weights", default="models/panel_seg_v26_l960.pt")
    args = ap.parse_args()

    calib = load_calibration(args.calib, floor_path=args.floor)
    if args.method == "yolo":
        from panel_detector.yolo_detector import YoloPanelDetector
        model = YoloPanelDetector(args.weights)
        run = lambda r, d: model.detect(r, d, calib)
        run = lambda r, d: model.detect(r, d, calib)
    else:
        run = lambda r, d: detect_panels(r, d, calib)
    pairs = _pair_paths(args.data)
    out = Path(args.out)
    (out / "json").mkdir(parents=True, exist_ok=True)
    (out / "overlay").mkdir(parents=True, exist_ok=True)

    counts, tiles = [], []
    montage_every = max(1, len(pairs) // args.montage)
    for i, (rp, dp) in enumerate(pairs):
        rgb = cv2.imread(rp)
        depth = cv2.imread(dp, cv2.IMREAD_UNCHANGED)
        det = run(rgb, depth)
        counts.append(len(det.panels))

        name = Path(rp).stem
        (out / "json" / f"{name}.json").write_text(
            json.dumps(to_json(det, Path(rp).name, (rgb.shape[1], rgb.shape[0])), indent=2))
        overlay = draw_panels(rgb, det)
        cv2.imwrite(str(out / "overlay" / f"{name}.png"), overlay)

        if i % montage_every == 0 and len(tiles) < args.montage:
            tile = cv2.resize(overlay, (320, 200))
            cv2.putText(tile, f"{name} n={len(det.panels)}", (8, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            tiles.append(tile)

    cols = 4
    rows = [np.hstack(tiles[j:j + cols]) for j in range(0, len(tiles) - len(tiles) % cols, cols)]
    if rows:
        cv2.imwrite(str(out / "montage.png"), np.vstack(rows))

    counts = np.array(counts)
    print(f"frames            : {len(counts)}")
    print(f"total panels      : {counts.sum()}")
    print(f"panels/frame      : mean {counts.mean():.1f}  median {int(np.median(counts))}  "
          f"range {counts.min()}-{counts.max()}")
    print(f"frames with 0     : {int((counts == 0).sum())}")
    print(f"outputs           : {out}/json, {out}/overlay, {out}/montage.png")


if __name__ == "__main__":
    main()
