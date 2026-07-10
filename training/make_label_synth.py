"""Label orientation + white-on-white synth.

Keeps each real frame's panel LAYOUT fixed and pastes real label stamps onto the
panel surfaces at many orientations, so the training set gets rich in label
orientation. A configurable fraction is pasted at MATCHED brightness (label L shifted
to the panel's L) to manufacture the white-on-white case that the base model misses;
those get a faint seam shadow (sometimes) so the only cue is subtle, as in real frames.

Panel surfaces come from the v26 panel model (RGB only). Original real labels in the
frame are kept and re-annotated; pasted labels are added. Output: YOLO seg dataset.

    python make_label_synth.py <src_yolo_dir> <panel_weights> <out_dir> <N>
"""
import glob, os, sys
import cv2, numpy as np
from ultralytics import YOLO

SRC = sys.argv[1] if len(sys.argv) > 1 else "labels_yolo"
PANEL_W = sys.argv[2] if len(sys.argv) > 2 else "panel_seg_v26_l960.pt"  # panel model (v26 branch)
OUT = sys.argv[3] if len(sys.argv) > 3 else "label_synth"
N = int(sys.argv[4]) if len(sys.argv) > 4 else 900
LOWC_FRAC = 0.55           # fraction of pastes made white-on-white (matched brightness)
BLUR_STAMP_FRAC = 0.30     # fraction of pasted labels blurred (labels are sometimes blurry)
BLUR_FRAME_FRAC = 0.18     # fraction of whole synth frames given a mild camera blur
BG_TOP = 0.27              # never paste in the top 27% (background sticker zone)
rng = np.random.default_rng(17)

os.makedirs(f"{OUT}/images/train", exist_ok=True)
os.makedirs(f"{OUT}/labels/train", exist_ok=True)

imgs = sorted(glob.glob(f"{SRC}/images/train/*.jpg"))
print(f"source frames: {len(imgs)}", flush=True)


def load_labels(lblp, W, H):
    out = []
    if not os.path.exists(lblp):
        return out
    for line in open(lblp).read().strip().split("\n"):
        if not line:
            continue
        c = np.array([float(x) for x in line.split()[1:]]).reshape(-1, 2)
        c[:, 0] *= W; c[:, 1] *= H
        m = np.zeros((H, W), np.uint8); cv2.fillPoly(m, [c.astype(np.int32)], 1)
        out.append(m.astype(bool))
    return out


# 1) extract real label stamps (rgb crop + mask), and per-frame panel masks
panel = YOLO(PANEL_W)
pids = {i for i, n in panel.names.items() if "panel" in str(n).lower()}
frames, stamps = {}, []
for ip in imgs:
    tag = os.path.basename(ip)[:-4]
    img = cv2.imread(ip); H, W = img.shape[:2]
    labs = load_labels(f"{SRC}/labels/train/{tag}.txt", W, H)
    r = panel.predict(img, imgsz=960, conf=0.35, iou=0.6, retina_masks=True, verbose=False)
    pm = np.zeros((H, W), bool)
    if r[0].masks is not None:
        cls = r[0].boxes.cls.cpu().numpy().astype(int)
        for j, mm in enumerate(r[0].masks.data.cpu().numpy()):
            if cls[j] in pids:
                pm |= cv2.resize(mm.astype(np.uint8), (W, H)) > 0.5
    frames[tag] = (img, labs, pm)
    lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    for m in labs:
        ys, xs = np.where(m)
        if m.sum() < 250:
            continue
        x0, y0, x1, y1 = xs.min(), ys.min(), xs.max(), ys.max()
        stamps.append({"rgb": img[y0:y1+1, x0:x1+1].copy(),
                       "mask": m[y0:y1+1, x0:x1+1].copy(),
                       "L": float(lab_img[m][:, 0].mean())})
print(f"stamps: {len(stamps)}, frames with panels: {sum(1 for _,_,p in frames.values() if p.any())}", flush=True)


def warp(rgb, mask, ang, sc):
    m = mask.astype(np.uint8)
    hh, ww = m.shape
    M = cv2.getRotationMatrix2D((ww/2, hh/2), ang, sc)
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    nw, nh = int(hh*sin + ww*cos) + 2, int(hh*cos + ww*sin) + 2
    M[0, 2] += nw/2 - ww/2; M[1, 2] += nh/2 - hh/2
    return cv2.warpAffine(rgb, M, (nw, nh)), cv2.warpAffine(m, M, (nw, nh)) > 0


def poly_of(m, eps=0.006, min_area=200):
    cn, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cn:
        return None
    c = max(cn, key=cv2.contourArea)
    if cv2.contourArea(c) < min_area:
        return None
    ap = cv2.approxPolyDP(c, eps*cv2.arcLength(c, True), True).reshape(-1, 2)
    return ap if len(ap) >= 3 else None


tags = [t for t, (_, _, p) in frames.items() if p.any()]
made = 0
for i in range(N):
    tag = tags[int(rng.integers(len(tags)))]
    img0, labs, pm = frames[tag]
    H, W = img0.shape[:2]
    canvas = img0.copy()
    # panel interior where a sticker can sit: eroded panel, below background zone,
    # away from existing labels
    interior = cv2.erode(pm.astype(np.uint8), np.ones((15, 15), np.uint8)).astype(bool)
    interior[:int(BG_TOP*H), :] = False
    for m in labs:
        interior &= ~cv2.dilate(m.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
    occupied = np.zeros((H, W), bool)
    new_labels = [m.copy() for m in labs]           # keep the real labels
    n_paste = int(rng.integers(2, 6))
    pasted = 0
    for _ in range(n_paste * 4):
        if pasted >= n_paste:
            break
        s = stamps[int(rng.integers(len(stamps)))]
        ang = float(rng.uniform(0, 360))            # << full orientation richness
        sc = float(rng.uniform(0.75, 1.25))
        r, m = warp(s["rgb"], s["mask"], ang, sc)
        mh, mw = m.shape
        if mh >= H or mw >= W:
            continue
        # candidate top-left positions whose footprint stays inside panel interior
        ys, xs = np.where(interior[:H-mh, :W-mw])
        if len(ys) == 0:
            break
        k = int(rng.integers(len(ys))); y0, x0 = int(ys[k]), int(xs[k])
        fm = np.zeros((H, W), bool); fm[y0:y0+mh, x0:x0+mw] = m
        # must land fully on panel interior and not collide with existing/pasted
        if (fm & ~interior).sum() > 0.05*fm.sum():
            continue
        if (fm & occupied).sum() > 0:
            continue
        sub_rgb = r.copy()
        if rng.random() < BLUR_STAMP_FRAC:          # blurry label (out-of-focus / motion)
            sub_rgb = cv2.GaussianBlur(sub_rgb, (0, 0), float(rng.uniform(0.6, 1.7)))
        lowc = rng.random() < LOWC_FRAC
        panel_L = float(cv2.cvtColor(canvas, cv2.COLOR_BGR2LAB)[fm][:, 0].mean())
        if lowc:
            lab = cv2.cvtColor(sub_rgb, cv2.COLOR_BGR2LAB).astype(np.int16)
            cur = lab[..., 0][m].mean()
            delta = float(rng.uniform(-8, 8))        # land within a few L of the panel
            lab[..., 0] = np.clip(lab[..., 0] + (panel_L - cur) + delta, 0, 255)
            sub_rgb = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
        # paste with feathered edge
        fr = np.zeros((H, W, 3), np.uint8); fr[y0:y0+mh, x0:x0+mw][m] = sub_rgb[m]
        alpha = cv2.GaussianBlur(fm.astype(np.float32), (0, 0), 0.8)[..., None]
        canvas = (canvas*(1-alpha) + fr*alpha).astype(np.uint8)
        if lowc and rng.random() < 0.6:              # faint seam shadow: the only WoW cue
            band = cv2.dilate(fm.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool) & ~fm & interior
            ba = cv2.GaussianBlur(band.astype(np.float32), (0, 0), 1.0)
            cl = cv2.cvtColor(canvas, cv2.COLOR_BGR2LAB).astype(np.float32)
            cl[..., 0] -= float(rng.uniform(5, 11))*ba
            canvas = cv2.cvtColor(np.clip(cl, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
        occupied |= cv2.dilate(fm.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
        new_labels.append(fm)
        pasted += 1
    if pasted < 1:
        continue
    if rng.random() < BLUR_FRAME_FRAC:              # whole-frame camera blur, GT unchanged
        canvas = cv2.GaussianBlur(canvas, (0, 0), float(rng.uniform(0.5, 1.3)))
    lines = []
    for m in new_labels:
        ap = poly_of(m)
        if ap is not None:
            lines.append("0 " + " ".join(f"{x/W:.5f} {y/H:.5f}" for x, y in np.clip(ap, [0, 0], [W-1, H-1])))
    if not lines:
        continue
    cv2.imwrite(f"{OUT}/images/train/lsyn_{i:05d}.jpg", canvas, [cv2.IMWRITE_JPEG_QUALITY, 93])
    open(f"{OUT}/labels/train/lsyn_{i:05d}.txt", "w").write("\n".join(lines))
    made += 1
    if made % 150 == 0:
        print(f"  synth {made}", flush=True)
print(f"LABEL SYNTH DONE: {made} images", flush=True)
