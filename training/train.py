"""Train the panel segmentation model.

    python train.py --data dataset/data.yaml
Produces runs/panel/weights/best.pt (rename to models/panel_seg_v5_l960.pt to deploy).
Trained on an RTX 4090 in about 30 minutes.
"""
import argparse

from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset/data.yaml")
    ap.add_argument("--model", default="yolov8l-seg.pt")
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--epochs", type=int, default=220)
    ap.add_argument("--batch", type=int, default=6)
    args = ap.parse_args()

    m = YOLO(args.model)
    m.train(data=args.data, epochs=args.epochs, patience=50, imgsz=args.imgsz,
            batch=args.batch, seed=7, cos_lr=True, close_mosaic=15, degrees=8.0,
            project="runs", name="panel", exist_ok=True)
    r = m.val(data=args.data, split="val", imgsz=args.imgsz)
    print(f"val seg mAP50={r.seg.map50:.3f} mAP50-95={r.seg.map:.3f}")


if __name__ == "__main__":
    main()
