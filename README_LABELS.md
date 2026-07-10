# Label (Sticker) Detection

Detects Nobilia information labels (stickers) on cabinet panels using YOLO26-large segmentation at 1280 px input resolution.

## Model

- **Architecture**: YOLOv26-large instance segmentation
- **Input size**: 1280 × 800 px (native scene resolution, upsampled for small objects)
- **Training data**: 119 frames with polygon annotations (735 label instances)
- **Class**: single class "label" (Nobilia product info sticker)

## Installation

```bash
pip install ultralytics opencv-python numpy
```

## Usage

### Detect labels in a single image

```bash
python detect_labels.py --rgb image.png --out labels.json
```

Output is a JSON with label polygons (normalized 0–1 coordinates) and area in pixels.

### Benchmark on validation set

```bash
python training/benchmark_labels.py --weights models/label_seg_best.pt --benchmark benchmark_export
```

Scoring is **detection-first**: labels are small stickers, so the benchmark uses a
relaxed IoU (0.5 primary, 0.7 for good localization) and reports what matters in
production — **recall** (are any labels missed?) and **precision** (are there any
false labels?) — with the raw miss / false-label counts and the frames they occur
in. It does not penalize a pixel or two of boundary slack on a correctly found label.

## Background sticker

The machine has a fixed sticker on its dark background structure near the top of
every frame. It is **not** a panel label and is ignored: any detection whose
centroid sits in the top `BG_TOP_FRACTION` (default 27%) of the frame is dropped.
In the annotated data every real label is below 38% of the frame height and this
background sticker is at ~8–15%, so the gate removes it with margin and never drops
a real label. The threshold is normalized to frame height, so it holds at any
resolution. See `BG_TOP_FRACTION` in [label_detector/label_detector.py](label_detector/label_detector.py).

## Model Parameters

- **Confidence threshold**: 0.4 (tuned for recall on small objects; lower to 0.3 if missing labels)
- **Min area**: 100 pixels (stickers are ~30–60 px wide; tune to filter debris)
- **IoU threshold**: 0.6 (NMS on raw YOLO output)
- **Background gate**: top 27% of frame height (fixed background sticker; see above)

## Training

To retrain on new data:

```bash
python training/train_labels.py --data labels_yolo/data.yaml --epochs 200 --batch 4
```

See [training/train_labels.py](training/train_labels.py) for full options. Patience defaults to 50 (early stop if no improvement for 50 epochs).

## Branch

Label detection code lives on the `labels` branch of the main repo; main branch has only panel detection.
