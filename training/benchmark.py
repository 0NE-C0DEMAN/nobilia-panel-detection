"""Score a model on the annotated benchmark (mask-vs-mask, panel class).

TP = a predicted panel matches a ground-truth panel at IoU >= threshold
(greedy one-to-one matching); reports F1 at IoU 0.50 / 0.80 / 0.90 and the
mean IoU of matched panels. Uses the same post-processing as detect.py
(panel class only, min area, mask-level NMS).

    python benchmark.py --weights ../models/panel_seg_v5_l960.pt --benchmark benchmark_export
"""
import argparse, json, os, re

import cv2
import numpy as np
from ultralytics import YOLO


def load_gt(bench_dir):
    frames = []
    for split in ["train", "valid", "test"]:
        p = os.path.join(bench_dir, split, "_annotations.coco.json")
        if not os.path.exists(p):
            continue
        d = json.load(open(p))
        cats = {c["id"]: c["name"] for c in d["categories"]}
        imgs = {i["id"]: i for i in d["images"]}
        by = {}
        for a in d["annotations"]:
            if cats[a["category_id"]] != "panel":
                continue
            by.setdefault(a["image_id"], []).append(a)
        for iid, anns in by.items():
            im = imgs[iid]
            masks = []
            for a in anns:
                m = np.zeros((im["height"], im["width"]), np.uint8)
                for s in a["segmentation"]:
                    cv2.fillPoly(m, [np.array(s, np.float32).reshape(-1, 2).astype(np.int32)], 1)
                if m.sum() > 150:
                    masks.append(m.astype(bool))
            frames.append((os.path.join(bench_dir, split, im["file_name"]), masks))
    return frames


def iou(a, b):
    u = (a | b).sum()
    return (a & b).sum() / u if u else 0.0


def predict(model, panel_ids, img, imgsz=960, conf=0.45, min_area=640, mask_nms=0.35):
    r = model.predict(img, imgsz=imgsz, conf=conf, iou=0.6, retina_masks=True, verbose=False)
    if not r or r[0].masks is None:
        return []
    h, w = img.shape[:2]
    cls = r[0].boxes.cls.cpu().numpy().astype(int)
    cf = r[0].boxes.conf.cpu().numpy()
    cand = []
    for i, m in enumerate(r[0].masks.data.cpu().numpy()):
        if panel_ids and cls[i] not in panel_ids:
            continue
        mm = cv2.resize(m.astype(np.uint8), (w, h)) > 0.5
        if mm.sum() >= min_area:
            cand.append((float(cf[i]), mm))
    cand.sort(key=lambda t: -t[0])
    out = []
    for c, mm in cand:
        if any((mm & k).sum() / min(mm.sum(), k.sum()) > mask_nms for k in out):
            continue
        out.append(mm)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--benchmark", required=True, help="COCO export folder of the benchmark")
    ap.add_argument("--imgsz", type=int, default=960)
    args = ap.parse_args()

    model = YOLO(args.weights)
    panel_ids = {i for i, n in (model.names or {}).items() if "panel" in str(n).lower()}
    gt = load_gt(args.benchmark)
    print(f"benchmark: {len(gt)} frames, {sum(len(m) for _, m in gt)} panels")

    cache = []
    for p, masks in gt:
        img = cv2.imdecode(np.fromfile(p, np.uint8), 1)
        cache.append((masks, predict(model, panel_ids, img, imgsz=args.imgsz)))

    for thr in (0.5, 0.8, 0.9):
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
    print(f"mean IoU of matched panels: {np.mean(MI):.3f}")


if __name__ == "__main__":
    main()
