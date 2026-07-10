# Nobilia Label Detection

Detects the information labels (stickers) on Nobilia cabinet panels from a single RGB
image, using a YOLO26-large instance-segmentation model at 1280 px input resolution.

This is the **label detection** deliverable. Panel detection lives on the `main` /
`v26` branches; this branch is self-contained for labels only.

## Results

Held-out test set (30 frames, 148 labels), detection-first scoring:

| Metric | IoU ≥ 0.5 | IoU ≥ 0.7 |
| --- | --- | --- |
| Recall (labels found) | **1.000** | 0.993 |
| Precision (no false labels) | **0.993** | 0.987 |
| F1 | **0.997** | 0.990 |
| Mean IoU of matched labels | 0.898 | 0.899 |

The model is trained with synthetic augmentation targeting the hard cases — labels at
all orientations, low-contrast **white labels on white/light panels**, and slightly
blurry labels — so it holds up on light-on-light stickers that a plain model misses.
See [Training](#training).

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Detect labels in a single image

```bash
python detect_labels.py --rgb image.png --out labels.json
```

Output is JSON with each label's polygon (normalized 0–1 coordinates) and pixel area.

### Benchmark

```bash
python training/benchmark_labels.py --weights models/label_seg_best.pt --benchmark <coco_export>
```

Scoring is **detection-first**: labels are small stickers, so it uses a relaxed IoU
(0.5 primary, 0.7 for good localization) and reports what matters in production —
**recall** (are any labels missed?) and **precision** (are there any false labels?) —
with the raw miss / false-label counts and the frames they occur in. It does not
penalize a pixel or two of boundary slack on a correctly found label.

## Background sticker

The machine has a fixed sticker on its dark background structure near the top of every
frame. It is **not** a panel label and is ignored: any detection whose centroid sits in
the top `BG_TOP_FRACTION` (default 27%) of the frame is dropped. In the annotated data
every real label sits below 38% of the frame height while this background sticker sits at
~8–15%, so the gate removes it with margin and never drops a real label. The threshold is
normalized to frame height, so it holds at any resolution. See `BG_TOP_FRACTION` in
[label_detector/label_detector.py](label_detector/label_detector.py).

## Model parameters

- **Confidence threshold**: 0.4 (swept optimum; lower toward 0.3 only if labels are missed)
- **Min area**: 100 px (stickers are ~30–60 px wide; filters debris)
- **Input size**: 1280 px (small-object precision)
- **NMS IoU**: 0.6
- **Background gate**: top 27% of frame height (see above)

## Training

```bash
python training/train_labels.py --data <data.yaml> --epochs 200 --batch 4
```

YOLO26-large seg, 1280 px, 200 epochs, patience 50. Trained on 109 annotated frames
plus synthetic augmentation (`training/make_label_synth.py`) with 20 val / 30 test
held out (real frames only).

### Synthetic augmentation

`make_label_synth.py` keeps each real frame's panel layout fixed and pastes real
label crops back onto the panel surfaces to manufacture the hard cases:

- **Orientation**: labels are pasted at the full 0–360° range, so the model sees every
  rotation on realistic layouts.
- **White-on-white**: a share of pasted labels have their brightness matched to the
  panel underneath (with only a faint seam shadow as the cue), teaching the model to
  find low-contrast white labels on white/light panels.
- **Blur**: a share of labels and whole frames are blurred, for out-of-focus / motion.

Panel surfaces come from the panel model (RGB only); training then adds full rotation,
both flips and brightness jitter on top.

## Files

```
detect_labels.py                 single-image inference CLI
label_detector/label_detector.py detector + background filter + JSON output
training/train_labels.py         training recipe
training/make_label_synth.py     orientation / white-on-white / blur synth generator
training/benchmark_labels.py     detection-first benchmark
models/label_seg_best.pt         trained weights (63 MB)
```
