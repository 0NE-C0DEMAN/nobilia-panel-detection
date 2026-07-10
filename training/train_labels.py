"""Train YOLO26-large for label (sticker) segmentation at 1280 px (small objects).

    python train_labels.py --data <path_to_data.yaml> --epochs 200 --batch 4

Uses 1280 input size for high precision on small objects; native YOLO26 recipe.
"""
import argparse
import sys
from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="path to data.yaml (YOLO format)")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--patience", type=int, default=50)
    ap.add_argument("--project", default="/workspace/runs")
    ap.add_argument("--name", default="labels")
    args = ap.parse_args()

    m = YOLO("yolo26l-seg.pt")
    try:
        m.train(
            data=args.data,
            epochs=args.epochs,
            patience=args.patience,
            imgsz=1280,
            batch=args.batch,
            seed=7,
            cos_lr=True,
            project=args.project,
            name=args.name,
            exist_ok=True,
        )
    except RuntimeError as e:
        if "out of memory" not in str(e).lower():
            raise
        print(f"OOM at batch={args.batch}, retrying batch={args.batch-1}", flush=True)
        import torch
        torch.cuda.empty_cache()
        m = YOLO("yolo26l-seg.pt")
        m.train(
            data=args.data,
            epochs=args.epochs,
            patience=args.patience,
            imgsz=1280,
            batch=args.batch - 1,
            seed=7,
            cos_lr=True,
            project=args.project,
            name=args.name,
            exist_ok=True,
        )
    r = m.val(data=args.data, split="val", imgsz=1280)
    print(f"labels: val seg mAP50={r.seg.map50:.3f} mAP50-95={r.seg.map:.3f}")


if __name__ == "__main__":
    main()
