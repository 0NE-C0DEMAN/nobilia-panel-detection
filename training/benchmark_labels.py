"""Score the label detector on an annotated benchmark, detection-first.

Labels are small stickers, so this measures whether we FIND every label with no
misses and no false labels, using a relaxed IoU so a pixel or two of boundary
slack does not count against a correct detection. The fixed background sticker
(top of frame) is ignored by the detector and is not annotated in the ground
truth, so it counts as neither.

Reports, at IoU 0.5 (primary) and 0.7 (well localized):
  - Recall  = labels found / labels present      (1 - miss rate)
  - Precision = correct / all detections         (1 - false-label rate)
  - raw miss and false-label counts, with the frames they occur in

    python benchmark_labels.py --weights ../models/label_seg_best.pt --benchmark benchmark_export
"""
import argparse, json, os, sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from label_detector import YoloLabelDetector


def load_gt(bench_dir):
    frames = []
    for split in ["train", "val", "valid", "test"]:
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


def match(g, pr, thr):
    """Greedy one-to-one match; returns (matched_ious, missed_gt_idx, false_pred_idx)."""
    pairs = sorted(((iou(x, q), gi, pj) for gi, x in enumerate(g)
                    for pj, q in enumerate(pr)), reverse=True)
    ug, up, ious = set(), set(), []
    for v, gi, pj in pairs:
        if v < thr or gi in ug or pj in up:
            continue
        ug.add(gi); up.add(pj); ious.append(v)
    missed = [i for i in range(len(g)) if i not in ug]
    false = [j for j in range(len(pr)) if j not in up]
    return ious, missed, false


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--benchmark", required=True, help="COCO export folder of the benchmark")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--conf", type=float, default=0.4)
    args = ap.parse_args()

    det = YoloLabelDetector(args.weights, conf=args.conf, imgsz=args.imgsz, min_area=50)
    gt = load_gt(args.benchmark)
    n_labels = sum(len(m) for _, m in gt)
    print(f"benchmark: {len(gt)} frames, {n_labels} labels\n")

    cache = []
    for p, masks in gt:
        img = cv2.imdecode(np.fromfile(p, np.uint8), 1)
        cache.append((os.path.basename(p), masks, det.detect(img)))

    n_det = sum(len(pr) for _, _, pr in cache)
    print(f"detections after background filter: {n_det}\n")

    for thr in (0.5, 0.7):
        TP = FP = FN = 0
        miss_frames, false_frames, all_ious = [], [], []
        for name, g, pr in cache:
            ious, missed, false = match(g, pr, thr)
            TP += len(ious); FN += len(missed); FP += len(false)
            all_ious += ious
            if missed:
                miss_frames.append((name, len(missed)))
            if false:
                false_frames.append((name, len(false)))
        P = TP / max(1, TP + FP)
        R = TP / max(1, TP + FN)
        F1 = 2 * P * R / max(1e-9, P + R)
        tag = "PRIMARY" if thr == 0.5 else "localization"
        print(f"=== IoU >= {thr}  ({tag}) ===")
        print(f"  recall    {R:.3f}   ({TP}/{TP+FN} labels found, {FN} missed)")
        print(f"  precision {P:.3f}   ({FP} false labels out of {TP+FP} detections)")
        print(f"  F1        {F1:.3f}")
        if all_ious:
            print(f"  mean IoU of matched: {np.mean(all_ious):.3f}")
        if miss_frames:
            print(f"  misses in: {', '.join(f'{n}({c})' for n, c in miss_frames)}")
        if false_frames:
            print(f"  false labels in: {', '.join(f'{n}({c})' for n, c in false_frames)}")
        print()


if __name__ == "__main__":
    main()
