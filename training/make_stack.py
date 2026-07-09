"""Occlusion-targeted stacking synth (Occlusion Copy & Paste, arXiv 2210.03686, adapted).
Pastes a SAME-COLOR panel partially over an existing panel so the underlying one
becomes a thin sliver: manufactures the white-on-white stacked failure case.
Not jumbling: pastes land only on panels, near-aligned, color-matched, optional
seam shadow. Labels get occlusion-subtracted; backgrounds updated the same way.
    python make_stack.py dataset 800
"""
import glob, os, sys
import cv2, numpy as np

DS = sys.argv[1] if len(sys.argv) > 1 else "dataset"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 800
H, W = 400, 640
rng = np.random.default_rng(31)

def load_labels(lblp):
    pans, bgs = [], []
    for line in (open(lblp).read().strip().split("\n") if os.path.exists(lblp) else []):
        if not line: continue
        c = np.array([float(x) for x in line.split()[1:]]).reshape(-1, 2)
        c[:, 0] *= W; c[:, 1] *= H
        m = np.zeros((H, W), np.uint8); cv2.fillPoly(m, [c.astype(np.int32)], 1)
        (pans if line.startswith("1 ") else bgs).append(m.astype(bool))
    return pans, bgs

real = sorted(glob.glob(f"{DS}/images/train/rgb_*.jpg"))
frames, stamps = {}, []
for ip in real:
    tag = os.path.basename(ip)[:-4]
    img = cv2.imread(ip)
    if img.shape[:2] != (H, W): img = cv2.resize(img, (W, H))
    pans, bgs = load_labels(f"{DS}/labels/train/{tag}.txt")
    frames[tag] = (img, pans, bgs)
    for m in pans:
        ys, xs = np.where(m)
        if m.sum() < 2500: continue
        x0, y0, x1, y1 = xs.min(), ys.min(), xs.max(), ys.max()
        if x0 <= 2 or y0 <= 2 or x1 >= W - 3 or y1 >= H - 3: continue
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        stamps.append({"rgb": img[y0:y1+1, x0:x1+1].copy(), "mask": m[y0:y1+1, x0:x1+1].copy(),
                       "lab": lab[m].mean(axis=0)})
print(f"{len(real)} frames, {len(stamps)} stamps", flush=True)

def xform(cr, cm, ang, sc):
    m = cv2.erode(cm.astype(np.uint8), np.ones((3, 3), np.uint8))
    hh, ww = m.shape
    M = cv2.getRotationMatrix2D((ww / 2, hh / 2), ang, sc)
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    nw, nh = int(hh * sin + ww * cos) + 2, int(hh * cos + ww * sin) + 2
    M[0, 2] += nw / 2 - ww / 2; M[1, 2] += nh / 2 - hh / 2
    return cv2.warpAffine(cr, M, (nw, nh)), cv2.warpAffine(m, M, (nw, nh)) > 0

def poly_of(m, eps, min_area):
    cn, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cn: return None
    c = max(cn, key=cv2.contourArea)
    if cv2.contourArea(c) < min_area: return None
    ap = cv2.approxPolyDP(c, eps * cv2.arcLength(c, True), True).reshape(-1, 2)
    return ap if len(ap) >= 3 else None

made = 0
tags = list(frames)
for i in range(N):
    tag = tags[int(rng.integers(len(tags)))]
    img, pans0, bgs0 = frames[tag]
    canvas = img.copy()
    pans = [p.copy() for p in pans0]
    bgs = [b.copy() for b in bgs0]
    lab_img = cv2.cvtColor(canvas, cv2.COLOR_BGR2LAB)
    n_paste = int(rng.integers(1, 3 + (len(pans) > 4)))
    pasted = 0
    for _ in range(n_paste * 3):
        if pasted >= n_paste: break
        big = [j for j, p in enumerate(pans) if p.sum() > 3000]
        if not big: break
        tj = big[int(rng.integers(len(big)))]
        T = pans[tj]
        t_lab = lab_img[T].mean(axis=0)
        if rng.random() < 0.45:  # self-stack: identical color by construction
            ys, xs = np.where(T)
            x0, y0, x1, y1 = xs.min(), ys.min(), xs.max(), ys.max()
            src_rgb, src_m = canvas[y0:y1+1, x0:x1+1].copy(), T[y0:y1+1, x0:x1+1].copy()
        else:  # color-matched stamp, L aligned to target
            cds = sorted(stamps, key=lambda s: np.abs(s["lab"] - t_lab).sum())[:12]
            s = cds[int(rng.integers(len(cds)))]
            src_rgb, src_m = s["rgb"].copy(), s["mask"].copy()
            sl = cv2.cvtColor(src_rgb, cv2.COLOR_BGR2LAB).astype(np.int16)
            sl[..., 0] = np.clip(sl[..., 0] + int(t_lab[0] - s["lab"][0]), 0, 255)
            src_rgb = cv2.cvtColor(sl.astype(np.uint8), cv2.COLOR_LAB2BGR)
        tys, txs = np.where(T)
        tx0, ty0, tx1, ty1 = txs.min(), tys.min(), txs.max(), tys.max()
        tw, th = tx1 - tx0 + 1, ty1 - ty0 + 1
        sc = float(np.clip(rng.uniform(0.8, 1.25) * max(tw / src_m.shape[1], th / src_m.shape[0]), 0.4, 1.8))
        r, m = xform(src_rgb, src_m, float(rng.uniform(-8, 8)), sc)
        # offset so a thin strip of the target stays visible
        off = rng.uniform(0.12, 0.45)
        ax = int(rng.integers(2))
        dx = int(off * tw) * (1 if rng.random() < 0.5 else -1) * ax
        dy = int(off * th) * (1 if rng.random() < 0.5 else -1) * (1 - ax)
        x0p = tx0 + (tw - m.shape[1]) // 2 + dx
        y0p = ty0 + (th - m.shape[0]) // 2 + dy
        fm = np.zeros((H, W), bool)
        xa, ya = max(0, x0p), max(0, y0p)
        xb, yb = min(W, x0p + m.shape[1]), min(H, y0p + m.shape[0])
        if xb <= xa or yb <= ya: continue
        sub = m[ya - y0p:yb - y0p, xa - x0p:xb - x0p]
        fm[ya:yb, xa:xb] = sub
        cov = (fm & T).sum() / T.sum()
        sliver = T & ~fm
        if not (0.45 <= cov <= 0.92) or sliver.sum() < 700 or fm.sum() < 2500: continue
        fr = np.zeros((H, W, 3), np.uint8)
        fr[ya:yb, xa:xb][sub] = r[ya - y0p:yb - y0p, xa - x0p:xb - x0p][sub]
        alpha = cv2.GaussianBlur(fm.astype(np.float32), (0, 0), 1.2)[..., None]
        canvas = (canvas * (1 - alpha) + fr * alpha).astype(np.uint8)
        if rng.random() < 0.5:  # soft seam shadow on the panel underneath
            band = cv2.dilate(fm.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool) & ~fm & sliver
            ba = cv2.GaussianBlur(band.astype(np.float32), (0, 0), 1.0)
            cl = cv2.cvtColor(canvas, cv2.COLOR_BGR2LAB).astype(np.float32)
            cl[..., 0] -= float(rng.uniform(6, 14)) * ba
            canvas = cv2.cvtColor(np.clip(cl, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
        pans = [p & ~fm for p in pans]
        bgs = [b & ~fm for b in bgs]
        pans.append(fm)
        lab_img = cv2.cvtColor(canvas, cv2.COLOR_BGR2LAB)
        pasted += 1
    if pasted < 1: continue
    pans = [p for p in pans if p.sum() >= 600]
    lines = []
    for b in bgs:
        ap = poly_of(b, 0.0035, 1200)
        if ap is not None:
            lines.append("0 " + " ".join(f"{x/W:.5f} {y/H:.5f}" for x, y in np.clip(ap, [0, 0], [W-1, H-1])))
    for p in pans:
        ap = poly_of(p, 0.006, 350)
        if ap is not None:
            lines.append("1 " + " ".join(f"{x/W:.5f} {y/H:.5f}" for x, y in np.clip(ap, [0, 0], [W-1, H-1])))
    if sum(1 for l in lines if l.startswith("1 ")) < 2: continue
    cv2.imwrite(f"{DS}/images/train/stack_{i:05d}.jpg", canvas, [cv2.IMWRITE_JPEG_QUALITY, 92])
    open(f"{DS}/labels/train/stack_{i:05d}.txt", "w").write("\n".join(lines))
    made += 1
    if made % 200 == 0: print(f"  stack {made}", flush=True)
print(f"STACK DONE: {made} images", flush=True)
