# Nobilia Panel Detection

Detects the top-layer furniture panels in an RGB-D bin frame. For every panel it returns
an oriented bounding box, the pick centre (image pixels and robot base-frame metres),
the metric size, the height above the bin floor and a top-layer flag.

## Results (v26 branch)

Benchmark: 28 annotated frames, excluded from training, mask-vs-mask F1 with one-to-one matching.

| Metric | v26 (this branch) | v5 (main) |
|---|---|---|
| F1 @ IoU >= 0.50 | **0.975** | 0.957 |
| F1 @ IoU >= 0.80 (milestone 2 target 0.90) | **0.946** | 0.916 |
| F1 @ IoU >= 0.90 (milestone 3 target 0.95) | **0.833** | 0.794 |
| mean IoU of matched panels | **0.936** | 0.929 |

Model: `models/panel_seg_v26_l960.pt` - YOLO26-large segmentation, 960 px input, 63 MB
(30% smaller and faster than v5). Trained on 339 human-annotated frames (visible-area
convention, classes: background, panel) plus occlusion-targeted synthetic images that
teach the model to separate same-colour stacked panels.

The gains over v5 come from three things: the newer model generation, more annotated
frames, and geometric post-processing rules (see How it works).

## Install

```bash
pip install -r requirements.txt
```

Python 3.10+. Runs on CPU (about 2-3 s per frame) or GPU.

## Usage

Single frame:

```bash
python detect.py --rgb rgb_00002.png --depth depth_00002.png \
                 --calib calib/cam_to_base_current.json \
                 --out panels.json --overlay overlay.png
```

Folder of `rgb_*.png` / `depth_*.png` pairs:

```bash
python evaluate.py --data path/to/frames --out results
```

Output JSON per panel:

```
id, obb {cx, cy, w, h, angle_deg}, corners, center_px, center_base_m,
height_mm, size_mm, angle_deg, top_layer
```

`top_layer: true` marks the panels the robot can pick next. If a frame was taken from a
camera pose that does not match the calibration it is flagged `off_pose: true` and only
2-D boxes are returned for it.

## Validate the results

```bash
cd training
python benchmark.py --weights ../models/panel_seg_v26_l960.pt --benchmark <benchmark_export_dir>
```

`<benchmark_export_dir>` is the COCO-segmentation export of the 28 annotated benchmark
frames. The script applies the same matching rule and post-processing as `detect.py`
and prints F1 at IoU 0.50 / 0.80 / 0.90 plus the mean IoU.

## Retrain with new annotations

```bash
cd training
python prepare_dataset.py --exports <roboflow_export> --benchmark <benchmark_export_dir> --out dataset
python make_stack.py dataset 800
python train.py --data dataset/data.yaml
```

`prepare_dataset.py` always excludes the benchmark frames, so the benchmark stays a clean
test set. `make_stack.py` adds the synthetic stacked-panel images (recommended; skip to
train on real frames only). Training takes about 1.5 hours on an RTX 4090. The new
`best.pt` replaces `models/panel_seg_v26_l960.pt`.

## Layout

```
detect.py                       single-frame CLI
evaluate.py                     batch runner
calibrate_floor.py              one-off floor-plane calibration
panel_detector/                 detection library (model + depth geometry)
models/panel_seg_v26_l960.pt    current model
calib/                          camera-to-base transform + floor plane
training/                       dataset conversion, synthesis, training, benchmark
```

## How it works

1. The segmentation model finds every visible panel surface (confidence 0.35; only a
   near-identical duplicate mask is removed, so low-contrast panels are never lost).
2. Overlap resolution: panels are annotated as non-overlapping visible areas, so a pixel
   claimed by two masks is wrong for one of them - it is reassigned to the mask whose
   undisputed core is nearest. This removes mask bleed across panel seams.
3. L-split: a panel is a rectangle, so a mask that fills its oriented rectangle poorly,
   has exactly one concave corner and no detected panel occluding it there is two
   butt-joined same-colour panels merged into one - it is split at the concave corner.
4. Depth is back-projected through the camera calibration; each panel gets its measured
   height above the calibrated floor plane.
5. The panel outline is fitted as an oriented box in the panel's own 3-D plane, so boxes
   and centres are perspective-correct and metric.
6. Floor rejection, occlusion and top-layer selection come from the measured heights.
