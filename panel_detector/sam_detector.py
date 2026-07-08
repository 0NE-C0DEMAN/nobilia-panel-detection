"""Hybrid panel detector: mask proposals from FastSAM (RGB) and from the
geometric height pipeline (depth), validated by one shared set of physical
gates, then deduplicated.

Why hybrid: the two proposal sources fail in complementary ways.
  * FastSAM sees the visual seams between panels - it cleanly separates stacked
    piles and same-height neighbours - but can miss big low-contrast panels
    (white panel on light wood, dark panel in shadow).
  * The geometric height segmentation catches *anything* that physically sticks
    up out of the floor plane, regardless of colour - but merges panels whose
    surfaces meet without a height step.
Every proposal from either source must then pass the same physics: planar,
panel-sized, roughly horizontal, sitting above the calibrated bin floor.

Runs on CPU at roughly 0.4 s / frame (640x400).
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .calibration import Calibration, normalize_depth
from .detector import Detection, DetectorParams, Panel, _normalize_obb
from .geometry import _fit_plane_tls, surface_normals


@dataclass
class HybridParams:
    # --- FastSAM proposal generation ---
    model_name: str = "FastSAM-s.pt"
    conf: float = 0.25                 # low: propose generously, the gates clean up
    iou: float = 0.9                   # high: keep overlapping proposals
    imgsz: int = 640
    use_sam: bool = True
    use_geometric: bool = True
    # --- shared geometric validation of each mask proposal ---
    min_valid_px: int = 600            # mask must contain this many valid depth px
    min_height_m: float = 0.008        # above the calibrated floor (floor fails this)
    max_height_m: float = 0.30         # taller = bin wall region
    max_tilt_deg: float = 50.0         # plane normal vs floor normal (walls ~90)
    max_plane_rms_m: float = 0.009     # planarity: RMS residual of a fitted plane
    min_area_px: int = 1500
    max_area_frac: float = 0.55        # reject masks covering most of the image
    min_fill_ratio: float = 0.55       # mask must fill its oriented box
    min_side_mm: float = 40.0          # metric size gate: rejects label stickers
    min_long_side_mm: float = 120.0
    # --- dedup / top-layer ---
    max_overlap: float = 0.35          # candidate overlapping a kept mask more is dropped
    occlusion_step_m: float = 0.015    # a panel higher by this much can occlude (real layer, not noise)
    occlusion_overlap_frac: float = 0.15  # ...and must cover this fraction of the panel to occlude it
    occlusion_dilate_px: int = 5


class HybridPanelDetector:
    """Reusable across frames; the FastSAM model is loaded lazily on first use."""

    def __init__(self, params: HybridParams | None = None):
        self.p = params or HybridParams()
        self._model = None
        self._geom_params = DetectorParams()

    @property
    def model(self):
        if self._model is None:
            from ultralytics import FastSAM          # deferred: torch import is slow
            self._model = FastSAM(self.p.model_name)
        return self._model

    # ------------------------------------------------------------------ #
    # proposal sources
    # ------------------------------------------------------------------ #
    def _sam_proposals(self, rgb) -> list[np.ndarray]:
        res = self.model(rgb, device="cpu", retina_masks=True, imgsz=self.p.imgsz,
                         conf=self.p.conf, iou=self.p.iou, verbose=False)
        if not res or res[0].masks is None:
            return []
        masks = res[0].masks.data.cpu().numpy().astype(bool)
        h, w = rgb.shape[:2]
        return [cv2.resize(m.astype(np.uint8), (w, h)).astype(bool)
                if m.shape != (h, w) else m for m in masks]

    def _geometric_proposals(self, rgb, height, height_disp, normals, valid,
                             points_base, floor_normal):
        """Height-based regions from the geometric pipeline (catches what SAM
        misses on low colour contrast). Returns (regions, label_map); the label
        map is reused to carve failed SAM union masks along real height steps.
        """
        from .detector import _edge_map, _panel_mask, _segment_instances

        gp = self._geom_params
        mask = _panel_mask(height, normals, floor_normal, valid, points_base, gp)
        edges = _edge_map(height_disp, rgb, valid, gp)
        labels = _segment_instances(mask, height_disp, edges, gp)
        out = [labels == lbl for lbl in range(1, labels.max() + 1)
               if (labels == lbl).sum() >= self.p.min_area_px]
        return out, labels

    # ------------------------------------------------------------------ #
    # shared validation
    # ------------------------------------------------------------------ #
    def _validate(self, mask, points_base, valid, height, floor_normal, mm_per_px):
        p = self.p
        v = mask & valid
        if v.sum() < p.min_valid_px or mask.sum() < p.min_area_px:
            return None
        if mask.mean() > p.max_area_frac:
            return None

        med_h = float(np.median(height[v]))
        if not (p.min_height_m <= med_h <= p.max_height_m):
            return None

        pts = points_base[v]
        if len(pts) > 4000:
            pts = pts[:: len(pts) // 4000]
        n, off = _fit_plane_tls(pts)
        rms = float(np.sqrt(np.mean((pts @ n + off) ** 2)))
        if rms > p.max_plane_rms_m:
            return None
        tilt = float(np.degrees(np.arccos(min(abs(float(n @ floor_normal)), 1.0))))
        if tilt > p.max_tilt_deg:
            return None

        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        cnt = max(contours, key=cv2.contourArea)
        rect = cv2.minAreaRect(cnt)
        bw, bh = rect[1]
        if bw * bh <= 0 or mask.sum() / (bw * bh) < p.min_fill_ratio:
            return None
        if max(bw, bh) * mm_per_px < p.min_long_side_mm:
            return None
        if min(bw, bh) * mm_per_px < p.min_side_mm:
            return None

        return {"mask": mask, "rect": rect, "cnt": cnt, "height_m": med_h,
                "rms": rms, "tilt": tilt,
                "fill": mask.sum() / (bw * bh), "area": int(mask.sum())}

    # ------------------------------------------------------------------ #
    def _kmeans_parts(self, mask, gray, height):
        """Fallback split of a mask into 2 clusters on (height, intensity)."""
        ys, xs = np.where(mask)
        if len(ys) < 10:
            return []
        h_feat = np.nan_to_num(height[ys, xs], nan=float(np.nanmedian(height[ys, xs])))
        feats = np.stack([h_feat / 0.01, gray[ys, xs].astype(np.float32) / 40.0], 1).astype(np.float32)
        _, labels, _ = cv2.kmeans(feats, 2, None,
                                  (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.5),
                                  3, cv2.KMEANS_PP_CENTERS)
        parts = []
        for c in (0, 1):
            sub = np.zeros_like(mask)
            sub[ys[labels.ravel() == c], xs[labels.ravel() == c]] = True
            parts.append(sub)
        return parts

    def _rescue(self, mask, gray, geo_labels, points_base, valid, height,
                floor_normal, mm_per_px):
        """Recover panels from a proposal that failed validation as a whole.

        A failed proposal is usually a union of a panel with the floor or with
        its neighbours. We carve it primarily along the depth-based geometric
        labels (real height steps), falling back to a (height, intensity) k-means
        for same-height colour seams. Every carved part must pass full validation
        to be kept, so this cannot invent panels.
        """
        p = self.p
        if mask.sum() < 2 * p.min_area_px:
            return []
        parts = []
        if geo_labels is not None:                       # carve by depth height-steps
            for L in np.unique(geo_labels[mask]):
                if L == 0:
                    continue
                part = mask & (geo_labels == L)
                if part.sum() >= p.min_area_px:
                    parts.append(part)
        if not parts:                                    # same-height seam: colour split
            parts = self._kmeans_parts(mask, gray, height)

        out = []
        for part in parts:
            part = cv2.morphologyEx(part.astype(np.uint8), cv2.MORPH_OPEN,
                                    np.ones((3, 3), np.uint8)).astype(bool)
            n, comps = cv2.connectedComponents(part.astype(np.uint8))
            for lbl in range(1, n):
                pc = comps == lbl
                if pc.sum() < p.min_area_px:
                    continue
                st = self._validate(pc, points_base, valid, height, floor_normal, mm_per_px)
                if st is not None:
                    out.append(st)
        return out

    # ------------------------------------------------------------------ #
    @staticmethod
    def _dedup(cands, max_overlap, contain_frac=0.60, same_layer_m=0.012):
        """Greedy suppression, atomic rectangles first (highest fill), so a union
        of two panels (lower fill) or a sticker inside a kept panel is dropped.

        Also drops a box that is mostly (>contain_frac) inside an already-kept
        *clean* panel (fill>0.7) at the same height - i.e. a fragment or label of
        that panel - which is the main source of over-segmentation. The
        same-height + kept-is-clean guard means true panels split off a union
        (the union has low fill) are preserved.
        """
        cands = sorted(cands, key=lambda c: (-c["fill"], -c["area"]))
        kept = []
        for c in cands:
            ok = True
            for k in kept:
                inter = (c["mask"] & k["mask"]).sum()
                if inter / c["area"] > max_overlap or inter / k["area"] > max_overlap:
                    ok = False
                    break
                if (inter / c["area"] > contain_frac and k["fill"] > 0.70
                        and abs(c["height_m"] - k["height_m"]) < same_layer_m):
                    ok = False
                    break
            if ok:
                kept.append(c)
        return kept

    # ------------------------------------------------------------------ #
    def detect(self, rgb, depth_mm, calib: Calibration) -> Detection:
        p = self.p
        depth_mm = normalize_depth(depth_mm)
        points_base, valid = calib.backproject(depth_mm)
        normals = surface_normals(points_base, valid)

        # Floor: calibrated static plane for orientation, with the level refined
        # per frame inside a narrow band around the calibrated value (immune to
        # large panels, tolerant of the few-mm drift the rig actually shows).
        from .geometry import estimate_floor_level, fit_floor_plane
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

        d_valid = depth_mm[valid & (depth_mm > 0)]
        med_depth = float(np.median(d_valid)) if d_valid.size else 800.0
        mm_per_px = med_depth / calib.color_intrinsics["fx"]

        proposals: list[np.ndarray] = []
        if p.use_sam:
            proposals += self._sam_proposals(rgb)
        geo_labels = None
        if p.use_geometric:
            geo_regions, geo_labels = self._geometric_proposals(
                rgb, height, height_disp, normals, valid, points_base, fn)
            proposals += geo_regions

        gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
        cands = []
        for mask in proposals:
            stats = self._validate(mask, points_base, valid, height, fn, mm_per_px)
            if stats is not None:
                cands.append(stats)
            else:
                # a whole-mask failure is usually a union; carve it and re-validate
                cands += self._rescue(mask, gray, geo_labels, points_base, valid,
                                      height, fn, mm_per_px)
        kept = sorted(self._dedup(cands, p.max_overlap), key=lambda k: -k["height_m"])

        panels = []
        for i, c in enumerate(kept):
            obb = _normalize_obb(c["rect"])
            corners = cv2.boxPoints(c["rect"]).astype(int).tolist()
            x, y, bw, bh = cv2.boundingRect(c["cnt"])
            v = c["mask"] & valid
            center_base = points_base[v].mean(axis=0)
            panels.append(Panel(
                id=i + 1, obb=obb, corners=corners,
                bbox_xywh=[int(x), int(y), int(bw), int(bh)],
                center_px=[int(round(obb["cx"])), int(round(obb["cy"]))],
                center_base_m=[round(float(x_), 4) for x_ in center_base],
                center_depth_mm=round(float(np.median(depth_mm[v])), 1),
                height_mm=round(c["height_m"] * 1000.0, 1),
                area_px=c["area"],
            ))

        # Top layer = pickable = no detected panel sits meaningfully above it. A
        # panel is occluded only if a panel higher by a real layer step covers a
        # non-trivial fraction of it (a single overlapping pixel or a few-mm noise
        # step must not disqualify it - that was flagging most real top panels).
        kern = np.ones((p.occlusion_dilate_px * 2 + 1,) * 2, np.uint8)
        for pan, c in zip(panels, kept):
            dil = cv2.dilate(c["mask"].astype(np.uint8), kern).astype(bool)
            area = max(int(c["mask"].sum()), 1)
            pan.top_layer = not any(
                (c2["height_m"] > c["height_m"] + p.occlusion_step_m)
                and ((c2["mask"] & dil).sum() / area > p.occlusion_overlap_frac)
                for pan2, c2 in zip(panels, kept) if pan2.id != pan.id
            )

        mask_union = np.zeros(rgb.shape[:2], bool)
        for c in kept:
            mask_union |= c["mask"]
        return Detection(panels=panels, floor_normal=fn,
                         floor_offset=float(calib.floor_offset) if calib.has_floor else 0.0,
                         height_map=height_disp, panel_mask=mask_union)


# Backwards-compatible aliases
SamParams = HybridParams
SamPanelDetector = HybridPanelDetector
