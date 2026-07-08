# Nobilia Panel Detection

Detects the top-layer furniture panels in an RGB-D bin image and outputs, per panel, an
oriented bounding box, the pick centre (in image pixels and in robot base-frame metres),
the metric size, the height above the bin floor and a top-layer flag - everything the
robot needs to pick the next panel.

## Results

Scored on the 28-frame annotated benchmark (mask-vs-mask, greedy one-to-one matching).
The benchmark frames were fully excluded from training.

| Model | F1 @ IoU>=0.80 | F1 @ IoU>=0.90 | mean IoU |
|---|---|---|---|
| milestone 1 model | 0.78 | 0.55 | 0.84 |
| milestone 2 model (first version) | 0.72 | 0.27 | 0.79 |
| **current model (panel_seg_v5_l960)** | **0.92** | **0.80** | **0.93** |

Milestone 2 target (F1 >= 0.90 at IoU >= 0.80): **met**.

The current model is trained on 261 human-annotated frames (annotation team + own
annotations, all in the same visible-area convention), YOLOv8-large segmentation at
960 px input, two classes (background, panel).

## Install

```bash
pip install -r requirements.txt
```

Python 3.10+. GPU is optional for inference (CPU works, ~2-4 s per frame at 960 px).

## Detect panels in one frame

```bash
python detect.py --rgb rgb_00002.png --depth depth_00002.png \
                 --calib calib/cam_to_base_current.json \
                 --out panels.json --overlay overlay.png
```

Output JSON (one entry per panel): `obb` (cx, cy, w, h, angle_deg), `corners`,
`center_px`, `center_base_m`, `height_mm`, `size_mm`, `angle_deg`, `top_layer`.
Frames taken from a camera pose that does not match the calibration are flagged
`"off_pose": true` and returned with 2-D boxes only (pass `--fallback
models/panel_seg_gemma.pt` to improve recall on such frames).

Batch over a folder of `rgb_*.png` / `depth_*.png` pairs:

```bash
python evaluate.py --data path/to/frames --out results
```

## Reproduce the training

1. Export the annotated dataset(s) from Roboflow in COCO-segmentation format.
2. Convert to YOLO format, excluding the benchmark so it stays a clean test set:

```bash
cd training
python prepare_dataset.py --exports <export_dir> [<export_dir2> ...] \
                          --benchmark <benchmark_export_dir> --out dataset
python train.py --data dataset/data.yaml
```

3. Validate against the benchmark:

```bash
python benchmark.py --weights ../models/panel_seg_v5_l960.pt --benchmark <benchmark_export_dir>
```

## Repository layout

```
detect.py                 single-frame CLI (JSON out)
evaluate.py               batch runner over a folder
calibrate_floor.py        one-off floor-plane calibration from empty-bin frames
panel_detector/           detection library (YOLO + depth geometry fusion)
models/panel_seg_v5_l960.pt   current model (YOLOv8l-seg, 960 px, 2 classes)
models/panel_seg_gemma.pt     small fallback model for off-pose frames
calib/                    camera-to-base transform + floor plane
training/                 dataset conversion, training and benchmark scripts
```

## How it works

1. The YOLOv8 segmentation model finds every visible panel surface (panel class only,
   confidence 0.45, mask-level NMS to remove duplicate/over-split masks).
2. The aligned depth image is back-projected through the camera calibration; each
   mask gets its height above the calibrated bin-floor plane.
3. The mask outline is intersected with the panel's 3-D plane and an oriented box is
   fitted in that plane, so boxes and centres are perspective-correct and metric.
4. Occlusion, floor rejection and top-layer selection come from the measured heights.
