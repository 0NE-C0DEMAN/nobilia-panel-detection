"""Score a label detection model on an annotated benchmark (mask-vs-mask).

TP = a predicted label matches a ground-truth label at IoU >= threshold
(greedy one-to-one matching); reports F1 at IoU 0.50 / 0.75 / 0.90 and mean IoU.

    python benchmark_labels.py --weights ../models/label_seg_best.pt --benchmark benchmark_export
"""
import argparse, json, os, sys

import cv2
import numpy as np
from ultralytics import YOLO

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from label_detector import YoloLabelDetector


def load_gt(bench_dir):
    frames = []
    for split in ["train", "val", "test"]:
        p = os.path.join(bench_dir, split, "_annotations.coco.json")
        if not os.path.exists(p):
            continue
        d = json.load(open(p))
        cats = {c["id"]: c["name"] for c in d["categories"]}
        imgs = {i["id"]: i for i in d["images"]}
        by = {}
        for a in d["annotations"]:
            if cats[a["category_id"]] != "label":
                continue
            by.setdefault(a["image_id"], []).append(a)
        for iid, anns in by.items():
            im = imgs[iid]
            masks = []
            for a in anns:
                m = np.zeros((im["height"], im["width"]), np.uint8)
                for s in a["segmentation"]:
                    cv2.fillPoly(m, [np.array(s, np.float32).reshape(-1, 2).astype(np.int32)], 1)
                if m.sum() > 50:
                    masks.append(m.astype(bool))
            frames.append((os.path.join(bench_dir, split, im["file_name"]), masks))
    return frames


def iou(a, b):
    u = (a | b).sum()
    return (a & b).sum() / u if u else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--benchmark", required=True, help="COCO export folder of the benchmark")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--conf", type=float, default=0.4)
    args = ap.parse_args()

    det = YoloLabelDetector(args.weights, conf=args.conf, imgsz=args.imgsz, min_area=50)
    gt = load_gt(args.benchmark)
    print(f"benchmark: {len(gt)} frames, {sum(len(m) for _, m in gt)} labels")

    cache = []
    for p, masks in gt:
        img = cv2.imdecode(np.fromfile(p, np.uint8), 1)
        preds = det.detect(img)
        cache.append((masks, preds))

    for thr in (0.5, 0.75, 0.9):
        TP = FP = FN = 0
        for g, pr in cache:
            pairs = sorted(((iou(x, q), gi, pj) for gi, x in enumerate(g)
                            for pj, q in enumerate(pr)), reverse=True)
            ug, up = set(), set()
            for v, gi, pj in pairs:
                if v < thr or gi in ug or pj in up:
                    continue
                ug.add(gi); up.add(pj)
            TP += len(ug); FP += len(pr) - len(ug); FN += len(g) - len(ug)
        P = TP / max(1, TP + FP)
        R = TP / max(1, TP + FN)
        print(f"F1 @ IoU>={thr}: {2 * P * R / max(1e-9, P + R):.3f}  (P={P:.3f} R={R:.3f})")
    MI = []
    for g, pr in cache:
        pairs = sorted(((iou(x, q), gi, pj) for gi, x in enumerate(g)
                        for pj, q in enumerate(pr)), reverse=True)
        ug, up = set(), set()
        for v, gi, pj in pairs:
            if v < 0.5 or gi in ug or pj in up:
                continue
            ug.add(gi); up.add(pj); MI.append(v)
    print(f"mean IoU of matched labels: {np.mean(MI):.3f}")


if __name__ == "__main__":
    main()
