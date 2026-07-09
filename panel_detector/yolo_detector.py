"""Deliverable detector: YOLO26-seg locates and segments the panels; the depth +
calibration turn each mask into an oriented box, a base-frame pick centre, a height
above the floor and a top-layer flag.

Mask post-processing exploits two facts about the annotation/scene geometry:
panels never overlap in the label space (each is annotated as its visible area), and
every panel is a rectangle seen from above. So predicted masks are never discarded
for overlapping - contested pixels are reassigned to the nearest mask core - and a
mask forming an unexplained L-shape is two butt-joined panels merged, and is split.

    from panel_detector.yolo_detector import YoloPanelDetector
    det = YoloPanelDetector("models/panel_seg_v26_l960.pt")
    result = det.detect(rgb, depth_mm, calib)
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .calibration import Calibration, normalize_depth
from .detector import Detection, Panel, _normalize_obb
from .geometry import estimate_floor_level, fit_floor_plane, surface_normals


def _perspective_box(mask_shape: np.ndarray, h0: float, fn: np.ndarray,
                     T_base_from_cam: np.ndarray, K: dict):
    """Fit the panel's oriented rectangle from its 2-D mask outline and the panel's
    plane, then project it back into the image.

    The camera views the bin at an angle, so a rectangular panel appears as a
    trapezoid and its true centre is not the 2-D image-box centre. We recover the
    real rectangle by intersecting each mask-outline pixel's camera ray with the
    panel's plane (points where ``P . fn = h0``), fitting the rectangle there (where
    the panel *is* a rectangle), and projecting the rectangle + its centre back to
    pixels.

    Crucially this uses the clean mask outline plus a single robust plane height,
    NOT each pixel's depth - so per-pixel depth noise on glossy/dark panels (which
    would otherwise fling stray 3-D points far out and stretch the box onto the bin
    floor) cannot distort the box.

    Returns (quad_px[4,2] int, center_px[2] int, center_base[3] float,
             size_mm [long, short], angle_deg, plane_offset_m) or None.
    """
    cnts, _ = cv2.findContours(mask_shape.astype(np.uint8), cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    cnt = max(cnts, key=cv2.contourArea)[:, 0, :]         # (N, 2) outline pixels
    if len(cnt) < 4:
        return None

    # Orthonormal axes spanning the floor plane (fn is the up/normal direction).
    seed = np.array([1.0, 0.0, 0.0])
    if abs(seed @ fn) > 0.9:
        seed = np.array([0.0, 1.0, 0.0])
    u_ax = seed - (seed @ fn) * fn
    u_ax /= np.linalg.norm(u_ax)
    v_ax = np.cross(fn, u_ax)

    # Intersect each outline pixel's camera ray with the panel plane P . fn = h0.
    C = T_base_from_cam[:3, 3]
    Rot = T_base_from_cam[:3, :3]
    dcam = np.column_stack([(cnt[:, 0] - K["cx"]) / K["fx"],
                            (cnt[:, 1] - K["cy"]) / K["fy"], np.ones(len(cnt))])
    dbase = dcam @ Rot.T
    denom = dbase @ fn
    denom = np.where(np.abs(denom) < 1e-9, 1e-9, denom)
    t = (h0 - C @ fn) / denom
    P = C[None, :] + t[:, None] * dbase                   # outline points on the plane

    a = P @ u_ax
    b = P @ v_ax
    rect = cv2.minAreaRect((np.column_stack([a, b]) * 1000.0).astype(np.float32))
    (ca, cb), (wa, wb), ang = rect
    box_mm = cv2.boxPoints(rect)

    def to_base(am_mm, bm_mm):
        return (am_mm / 1000.0) * u_ax + (bm_mm / 1000.0) * v_ax + h0 * fn

    center_base = to_base(ca, cb)
    corners_base = np.array([to_base(x, y) for x, y in box_mm])
    Tinv = np.linalg.inv(T_base_from_cam)

    def proj(Q):
        pc = (np.c_[Q, np.ones(len(Q))] @ Tinv.T)[:, :3]
        z = np.where(np.abs(pc[:, 2]) < 1e-6, 1e-6, pc[:, 2])
        return np.column_stack([K["fx"] * pc[:, 0] / z + K["cx"],
                                K["fy"] * pc[:, 1] / z + K["cy"]])

    center_px = proj(center_base[None])[0]
    quad_px = proj(corners_base)
    size_mm = [round(max(wa, wb), 1), round(min(wa, wb), 1)]           # [long, short]
    angle = ang if wa >= wb else ang + 90.0
    angle = (angle + 90.0) % 180.0 - 90.0                              # long axis, (-90, 90]
    return (quad_px.astype(int), center_px.astype(int), center_base,
            size_mm, round(float(angle), 1), h0)


def _resolve_overlaps(masks):
    """Panels are annotated as non-overlapping visible areas, so a pixel claimed by
    two masks is wrong for at least one of them. Reassign every contested pixel to
    the mask whose undisputed core is nearest (NOT to the more confident mask - the
    bleeding mask is usually the confident one)."""
    if len(masks) < 2:
        return masks
    cnt = np.zeros(masks[0].shape, np.uint8)
    for m in masks:
        cnt += m
    contested = cnt > 1
    if not contested.any():
        return masks
    dists = []
    for m in masks:
        core = m & ~contested
        if core.sum() == 0:
            core = m
        dists.append(cv2.distanceTransform((~core).astype(np.uint8), cv2.DIST_L2, 3))
    owner = np.argmin(np.stack(dists), axis=0)
    return [m & (~contested | (owner == j)) for j, m in enumerate(masks)]


def _fill_ratio(m):
    cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return 1.0, None
    c = max(cnts, key=cv2.contourArea)
    ra = cv2.minAreaRect(c)[1][0] * cv2.minAreaRect(c)[1][1]
    return (m.sum() / ra if ra > 0 else 1.0), c


def _l_split(m, others, min_area):
    """A panel is a rectangle, so a mask that fills its own oriented rect poorly and
    has exactly one concave corner - with that notch NOT covered by another detected
    panel (which would explain it as occlusion) - is two same-colour butt-joined
    panels the model merged into one. Split it through the concave corner, trying the
    corner's own edge directions and the blob's principal axes, and accept the cut
    whose two parts are the most rectangular. Returns (part_a, part_b) or None."""
    fr, c = _fill_ratio(m)
    if fr >= 0.80 or c is None:
        return None
    ap = cv2.approxPolyDP(c, 0.012 * cv2.arcLength(c, True), True).reshape(-1, 2)
    if len(ap) < 4:
        return None
    orient = 1 if cv2.contourArea(ap.reshape(-1, 1, 2), True) > 0 else -1
    refl = [i for i in range(len(ap))
            if np.sign((ap[i][0] - ap[i - 1][0]) * (ap[(i + 1) % len(ap)][1] - ap[i][1])
                       - (ap[i][1] - ap[i - 1][1]) * (ap[(i + 1) % len(ap)][0] - ap[i][0])) == -orient
            and abs((ap[i][0] - ap[i - 1][0]) * (ap[(i + 1) % len(ap)][1] - ap[i][1])
                    - (ap[i][1] - ap[i - 1][1]) * (ap[(i + 1) % len(ap)][0] - ap[i][0])) > 40]
    if len(refl) != 1:
        return None
    i = refl[0]
    R = ap[i].astype(np.float32)
    box = cv2.boxPoints(cv2.minAreaRect(c)).astype(np.int32)
    rm = np.zeros(m.shape, np.uint8)
    cv2.fillPoly(rm, [box], 1)
    notch = (rm > 0) & ~m
    if notch.sum() > 0 and others:
        claimed = np.zeros(m.shape, bool)
        for o in others:
            claimed |= o
        if (notch & claimed).sum() / notch.sum() > 0.35:
            return None                     # occlusion by a detected panel, legit shape
    dirs = []
    for j in (i - 1, i + 1):
        d = R - ap[j % len(ap)].astype(np.float32)
        nd = np.linalg.norm(d)
        if nd >= 6:
            dirs.append(d / nd)
    ang = np.deg2rad(cv2.minAreaRect(c)[2])
    dirs.append(np.array([np.cos(ang), np.sin(ang)], np.float32))
    dirs.append(np.array([-np.sin(ang), np.cos(ang)], np.float32))
    best = None
    for d in dirs:
        cut = m.astype(np.uint8).copy()
        cv2.line(cut, tuple((R - d * 2000).astype(int)), tuple((R + d * 2000).astype(int)), 0, 3)
        n, lbl = cv2.connectedComponents(cut)
        if n < 3:
            continue
        top = np.argsort([(lbl == k).sum() for k in range(1, n)])[::-1][:2]
        pa, pb = lbl == top[0] + 1, lbl == top[1] + 1
        if min(pa.sum(), pb.sum()) < max(min_area, 0.12 * m.sum()):
            continue
        sc = min(_fill_ratio(pa)[0], _fill_ratio(pb)[0])
        if sc > 0.70 and sc > fr + 0.02 and (best is None or sc > best[0]):
            grow = cv2.dilate(pa.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
            pa2 = (grow & m & ~pb) | pa     # heal the cut line into part a
            best = (sc, pa2, pb | (m & ~pa2))
    return None if best is None else (best[1], best[2])


@dataclass
class YoloParams:
    weights: str = "models/panel_seg_v26_l960.pt"
    conf: float = 0.35
    iou: float = 0.6
    imgsz: int = 960
    min_area_px: int = 640
    dup_overlap: float = 0.8               # only a near-identical mask is a duplicate
    top_layer_band_mm: float = 30.0
    occlusion_step_mm: float = 15.0
    occlusion_overlap_frac: float = 0.15
    # Bin-floor rejection: a real panel is a ~19mm-thick slab, so its top surface
    # sits well above the floor. A mask whose surface is essentially at floor level
    # is the plywood bin bottom (a YOLO false positive), not a panel - drop it.
    floor_gate_mm: float = 8.0             # "above the floor" means higher than this
    min_above_floor_frac: float = 0.5      # keep only if >half the surface clears it
    # Off-pose frames: the camera pose differs from the calibration, so the depth
    # geometry is unreliable. Detect this by surface that reads BELOW the calibrated
    # floor - impossible while the camera is static (panels only ever sit above the
    # floor), so it flags a genuine pose move without false-firing on covered floors.
    # Then fall back to plain 2-D image boxes (no fabricated 3-D height / centre).
    off_pose_below_mm: float = 20.0        # "below the floor" means this far under it
    off_pose_below_frac: float = 0.08      # off-pose if >this fraction of flat area is below
    off_pose_conf: float = 0.20            # lower conf on the harder off-pose frame


class YoloPanelDetector:
    def __init__(self, weights: str | None = None, params: YoloParams | None = None,
                 fallback_weights: str | None = None):
        self.p = params or YoloParams()
        if weights:
            self.p.weights = weights
        self._model = None
        # Optional second model used only when the primary finds nothing (e.g. the
        # small model is blind to an off-pose frame the base model still detects).
        self._fallback_weights = fallback_weights
        self._fallback = None

    @property
    def model(self):
        if self._model is None:
            from ultralytics import YOLO
            self._model = YOLO(self.p.weights)
        return self._model

    @property
    def fallback(self):
        if self._fallback is None and self._fallback_weights:
            self._fallback = YoloPanelDetector(self._fallback_weights, params=self.p)
        return self._fallback

    def _masks(self, rgb, conf: float | None = None):
        res = self.model.predict(rgb, imgsz=self.p.imgsz, conf=conf or self.p.conf,
                                 iou=self.p.iou, retina_masks=True, verbose=False)
        if not res or res[0].masks is None:
            return []
        h, w = rgb.shape[:2]
        # Keep only the "panel" class on multi-class models (e.g. background+panel);
        # single-class models pass everything through.
        names = getattr(self.model, "names", {}) or {}
        panel_ids = {i for i, n in names.items() if "panel" in str(n).lower()}
        cls = res[0].boxes.cls.cpu().numpy().astype(int)
        confs = res[0].boxes.conf.cpu().numpy()
        cand = []
        for i, m in enumerate(res[0].masks.data.cpu().numpy()):
            if panel_ids and cls[i] not in panel_ids:
                continue
            m = cv2.resize(m.astype(np.uint8), (w, h)) > 0.5
            if m.sum() >= self.p.min_area_px:
                cand.append((float(confs[i]), m))
        # A detection is never discarded for merely overlapping another (a thin panel
        # sliver legitimately overlaps the mask of the panel covering it) - only a
        # near-identical lower-confidence mask is a duplicate of the same panel.
        cand.sort(key=lambda t: -t[0])
        out = []
        for c, m in cand:
            if any((m & k).sum() / min(m.sum(), k.sum()) > self.p.dup_overlap for k in out):
                continue
            out.append(m)
        out = _resolve_overlaps(out)
        final = []
        for j, m in enumerate(out):
            s = _l_split(m, [q for k, q in enumerate(out) if k != j], self.p.min_area_px)
            final.extend(s if s is not None else [m])
        return [m for m in final if m.sum() >= self.p.min_area_px]

    def _get_masks(self, rgb, conf: float | None = None):
        """Primary model, falling back to the secondary model if it finds nothing."""
        masks = self._masks(rgb, conf)
        if not masks and self.fallback is not None:
            masks = self.fallback._masks(rgb, conf)
        return masks

    def detect(self, rgb, depth_mm, calib: Calibration) -> Detection:
        p = self.p
        depth_mm = normalize_depth(depth_mm)
        points_base, valid = calib.backproject(depth_mm)
        normals = surface_normals(points_base, valid)

        if calib.has_floor:
            fn, calibrated_level = calib.floor_normal, -calib.floor_offset
        else:
            fn, fo = fit_floor_plane(points_base, valid, calib.up_in_base)
            calibrated_level = -fo
        raw = points_base @ fn
        flat = valid & (np.abs(np.einsum("ijk,k->ij", normals, fn)) > 0.9)
        floor_level = estimate_floor_level(raw, flat, calibrated_level)
        height = raw - floor_level
        height_disp = np.where(valid, height, np.nan)

        # Off-pose frame? If a meaningful fraction of the flat surface reads BELOW the
        # calibrated floor, the camera has moved (the floor can't be below itself while
        # the rig is static) and the depth geometry is unreliable -> return plain 2-D
        # image boxes instead of fabricating 3-D. A fully panel-covered floor does NOT
        # trigger this (its surfaces all sit above the floor).
        below_frac = float((flat & (raw - calibrated_level < -p.off_pose_below_mm / 1000.0)).mean())
        if below_frac > p.off_pose_below_frac:
            return self._detect_2d(rgb, self._get_masks(rgb, p.off_pose_conf), fn, height_disp)

        masks = self._get_masks(rgb)
        K = calib.color_intrinsics
        # Floor gate: reject masks whose surface sits at floor level (the plywood
        # bin bottom, a YOLO false positive - a ~19mm-thick panel always clears it).
        # Only trust the gate when at least one mask clearly clears the floor; on the
        # odd off-pose frame the calibrated plane doesn't fit and everything reads
        # low, so skip the gate there rather than wipe the whole frame.
        fracs = [float((height[m & valid] * 1000.0 > p.floor_gate_mm).mean())
                 if (m & valid).sum() >= 30 else None for m in masks]
        gate_on = any(f is not None and f >= p.min_above_floor_frac for f in fracs)

        panels, mstore = [], []
        for mask, frac_above in zip(masks, fracs):
            cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
            if not cnts:
                continue
            if gate_on and frac_above is not None and frac_above < p.min_above_floor_frac:
                continue                            # bin floor, not a panel
            cnt = max(cnts, key=cv2.contourArea)
            v = mask & valid
            v_fit = v & (height * 1000.0 > p.floor_gate_mm)   # on-surface valid points
            # Box shape comes from the clean mask OUTLINE intersected with the panel
            # plane (see _perspective_box) - robust to glossy-panel depth noise. Use a
            # robust median plane height, and drop clearly-floor pixels the mask bled
            # onto so the box can't reach down to the bin floor.
            if v_fit.sum() >= 25:
                h0 = float(np.median(raw[v_fit]))
                mask_shape = mask & ~(valid & (height * 1000.0 <= p.floor_gate_mm))
                stat = v_fit
            elif v.sum() >= 25:                 # gate-off / odd frame: trust all points
                h0, mask_shape, stat = float(np.median(raw[v])), mask, v
            else:
                h0 = mask_shape = stat = None
            pbox = (_perspective_box(mask_shape, h0, fn, calib.T_base_from_cam, K)
                    if h0 is not None else None)

            if pbox is not None:
                # Perspective-correct box + panel's physical centre (see _perspective_box).
                quad_px, center_px, center_base, size_mm, angle_deg, _h0 = pbox
                obb = _normalize_obb(cv2.minAreaRect(quad_px.astype(np.float32)))
                x, y, bw, bh = cv2.boundingRect(quad_px.astype(np.int32))
                med_h = float(np.median(height[stat]))
                panel = Panel(
                    id=len(panels) + 1, obb=obb, corners=quad_px.tolist(),
                    bbox_xywh=[int(x), int(y), int(bw), int(bh)],
                    center_px=[int(center_px[0]), int(center_px[1])],
                    center_base_m=[round(float(c), 4) for c in center_base],
                    center_depth_mm=round(float(np.median(depth_mm[stat])), 1),
                    height_mm=round(med_h * 1000.0, 1), area_px=int(mask.sum()),
                    size_mm=size_mm, angle_deg=angle_deg)
            else:
                # Fallback: panel with too little valid depth -> 2-D image box only.
                rect = cv2.minAreaRect(cnt)
                obb = _normalize_obb(rect)
                x, y, bw, bh = cv2.boundingRect(cnt)
                if v.sum() >= 30:
                    center_base = [round(float(c), 4) for c in points_base[v].mean(axis=0)]
                    cdepth = round(float(np.median(depth_mm[v])), 1)
                    med_h = float(np.median(height[v]))
                else:
                    center_base, cdepth, med_h = [0.0, 0.0, 0.0], 0.0, 0.0
                panel = Panel(
                    id=len(panels) + 1, obb=obb,
                    corners=cv2.boxPoints(rect).astype(int).tolist(),
                    bbox_xywh=[int(x), int(y), int(bw), int(bh)],
                    center_px=[int(round(obb["cx"])), int(round(obb["cy"]))],
                    center_base_m=center_base, center_depth_mm=cdepth,
                    height_mm=round(med_h * 1000.0, 1), area_px=int(mask.sum()))
            panels.append(panel)
            mstore.append((mask, panel.height_mm / 1000.0))

        # top layer = not overlapped-and-below another detected panel
        kern = np.ones((11, 11), np.uint8)
        for pan, (mask, mh) in zip(panels, mstore):
            dil = cv2.dilate(mask.astype(np.uint8), kern).astype(bool)
            area = max(int(mask.sum()), 1)
            pan.top_layer = not any(
                (mh2 > mh + p.occlusion_step_mm / 1000.0)
                and ((m2 & dil).sum() / area > p.occlusion_overlap_frac)
                for pan2, (m2, mh2) in zip(panels, mstore) if pan2.id != pan.id)

        union = np.zeros(rgb.shape[:2], bool)
        for mask, _ in mstore:
            union |= mask
        return Detection(panels=panels, floor_normal=fn,
                         floor_offset=float(calib.floor_offset) if calib.has_floor else 0.0,
                         height_map=height_disp, panel_mask=union, off_pose=False)

    def _detect_2d(self, rgb, masks, fn, height_disp) -> Detection:
        """2-D image-box detection for off-pose frames: oriented boxes straight from
        the mask outlines, with no depth-derived 3-D height or base-frame centre
        (those are unreliable when the pose doesn't match the calibration)."""
        panels = []
        for mask in masks:
            cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
            if not cnts:
                continue
            cnt = max(cnts, key=cv2.contourArea)
            rect = cv2.minAreaRect(cnt)
            obb = _normalize_obb(rect)
            x, y, bw, bh = cv2.boundingRect(cnt)
            panels.append(Panel(
                id=len(panels) + 1, obb=obb,
                corners=cv2.boxPoints(rect).astype(int).tolist(),
                bbox_xywh=[int(x), int(y), int(bw), int(bh)],
                center_px=[int(round(obb["cx"])), int(round(obb["cy"]))],
                center_base_m=None, center_depth_mm=None, height_mm=None,
                area_px=int(mask.sum()), top_layer=True, size_mm=None,
                angle_deg=obb["angle_deg"]))
        union = np.zeros(rgb.shape[:2], bool)
        for m in masks:
            union |= m
        return Detection(panels=panels, floor_normal=fn, floor_offset=0.0,
                         height_map=height_disp, panel_mask=union, off_pose=True)
