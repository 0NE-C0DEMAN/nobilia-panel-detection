"""Convert Roboflow COCO-segmentation exports into a YOLOv8-seg dataset.

Reads one or more export folders (each containing train/valid/test splits with
_annotations.coco.json), skips every frame that appears in the benchmark set
(so the benchmark stays a clean, unseen test set), and writes images/labels in
YOLO segmentation format with two classes: 0=background, 1=panel.

    python prepare_dataset.py --exports export1 export2 --benchmark benchmark_export --out dataset
"""
import argparse, json, os, random, re, shutil

import cv2
import numpy as np


def frames_of(export_dir):
    out = {}
    for split in ["train", "valid", "test"]:
        p = os.path.join(export_dir, split, "_annotations.coco.json")
        if not os.path.exists(p):
            continue
        d = json.load(open(p))
        cats = {c["id"]: c["name"] for c in d["categories"]}
        imgs = {i["id"]: i for i in d["images"]}
        by = {}
        for a in d["annotations"]:
            by.setdefault(a["image_id"], []).append(a)
        for iid, anns in by.items():
            im = imgs[iid]
            fr = re.match(r"(rgb_\d+)", im["file_name"]).group(1)
            out[fr] = (os.path.join(export_dir, split, im["file_name"]), im, anns, cats)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exports", nargs="+", required=True, help="Roboflow export folder(s)")
    ap.add_argument("--benchmark", required=True, help="benchmark export folder (excluded)")
    ap.add_argument("--out", default="dataset")
    ap.add_argument("--val", type=int, default=16, help="validation frames held out of train")
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()

    bench = set(frames_of(args.benchmark).keys())
    print(f"benchmark frames to exclude: {len(bench)}")

    pool = {}
    for ex in args.exports:
        for fr, item in frames_of(ex).items():
            if fr in bench:
                continue
            pool[fr] = item                       # later exports override earlier ones
    print(f"training pool: {len(pool)} frames")

    if os.path.exists(args.out):
        shutil.rmtree(args.out)
    for sub in ["images/train", "images/val", "labels/train", "labels/val"]:
        os.makedirs(os.path.join(args.out, sub))

    items = sorted(pool.items())
    random.Random(args.seed).shuffle(items)
    npan = 0
    for k, (fr, (img_path, im, anns, cats)) in enumerate(items):
        split = "val" if k < args.val else "train"
        lines = []
        for a in anns:
            nm = cats[a["category_id"]]
            cls = 0 if nm == "background" else (1 if nm == "panel" else None)
            if cls is None:
                continue
            for seg in a["segmentation"]:
                pts = np.array(seg, np.float32).reshape(-1, 2)
                if len(pts) < 3 or cv2.contourArea(pts.astype(np.int32)) < 100:
                    continue
                pts[:, 0] /= im["width"]
                pts[:, 1] /= im["height"]
                lines.append(f"{cls} " + " ".join(f"{x:.5f} {y:.5f}" for x, y in np.clip(pts, 0, 1)))
                npan += cls
        if not lines:
            continue
        img = cv2.imdecode(np.fromfile(img_path, np.uint8), 1)
        cv2.imwrite(os.path.join(args.out, "images", split, fr + ".jpg"), img,
                    [cv2.IMWRITE_JPEG_QUALITY, 95])
        open(os.path.join(args.out, "labels", split, fr + ".txt"), "w").write("\n".join(lines))
    with open(os.path.join(args.out, "data.yaml"), "w") as f:
        f.write(f"path: {os.path.abspath(args.out)}\ntrain: images/train\nval: images/val\n"
                "nc: 2\nnames: [background, panel]\n")
    print(f"done: {len(items)} frames, {npan} panel instances -> {args.out}")


if __name__ == "__main__":
    main()
