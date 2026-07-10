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

Scores the model on IoU thresholds 0.5, 0.75, 0.9 using greedy one-to-one mask matching.

## Model Parameters

- **Confidence threshold**: 0.4 (tuned for recall on small objects; lower to 0.3 if missing labels)
- **Min area**: 100 pixels (stickers are ~30–60 px wide; tune to filter debris)
- **IoU threshold**: 0.6 (NMS on raw YOLO output)

## Training

To retrain on new data:

```bash
python training/train_labels.py --data labels_yolo/data.yaml --epochs 200 --batch 4
```

See [training/train_labels.py](training/train_labels.py) for full options. Patience defaults to 50 (early stop if no improvement for 50 epochs).

## Branch

Label detection code lives on the `labels` branch of the main repo; main branch has only panel detection.
