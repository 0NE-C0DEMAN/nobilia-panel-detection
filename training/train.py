"""Train the panel segmentation model (YOLO26-large seg, native training recipe).

    python train.py --data dataset/data.yaml
Produces runs/panel/weights/best.pt (rename to models/panel_seg_v26_l960.pt to deploy).
Trained on an RTX 4090 in about 1.5 hours.

For best results generate the occlusion-targeted synthetic images into the dataset
first (teaches the model to separate same-colour stacked panels):

    python make_stack.py dataset 800
"""
import argparse

from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset/data.yaml")
    ap.add_argument("--model", default="yolo26l-seg.pt")
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--batch", type=int, default=6)
    args = ap.parse_args()

    m = YOLO(args.model)
    m.train(data=args.data, epochs=args.epochs, patience=40, imgsz=args.imgsz,
            batch=args.batch, seed=7, project="runs", name="panel", exist_ok=True)
    r = m.val(data=args.data, split="val", imgsz=args.imgsz)
    print(f"val seg mAP50={r.seg.map50:.3f} mAP50-95={r.seg.map:.3f}")


if __name__ == "__main__":
    main()
