"""YOLO26 label (sticker) segmentation detector for small objects."""
from ultralytics import YOLO
import cv2
import numpy as np

# There is a fixed sticker on the machine's dark background structure, near the top
# of every frame, that is NOT a panel label and must be ignored. In the annotated
# data every real label has its centroid at >=38% of the frame height, while this
# background sticker sits at ~8-15%. A gate at 27% (normalized, so it is independent
# of image resolution) removes the background sticker with margin on both sides and
# never touches a real label.
BG_TOP_FRACTION = 0.27


class YoloLabelParams:
    def __init__(self, weights="models/label_seg_best.pt", conf=0.4, imgsz=1280,
                 min_area=100, bg_top_fraction=BG_TOP_FRACTION):
        self.weights = weights
        self.conf = conf
        self.imgsz = imgsz
        self.min_area = min_area
        self.bg_top_fraction = bg_top_fraction


class YoloLabelDetector:
    def __init__(self, weights="models/label_seg_best.pt", conf=0.4, imgsz=1280,
                 min_area=100, bg_top_fraction=BG_TOP_FRACTION):
        self.model = YOLO(weights)
        self.conf = conf
        self.imgsz = imgsz
        self.min_area = min_area
        self.bg_top_fraction = bg_top_fraction

    def detect(self, rgb_image):
        """Detect panel labels in an RGB image. Returns a list of binary masks.

        The fixed background sticker near the top of the frame is filtered out
        (see BG_TOP_FRACTION): any detection whose centroid is above that line is
        treated as background, not a panel label.
        """
        h, w = rgb_image.shape[:2]
        y_cut = self.bg_top_fraction * h
        r = self.model.predict(rgb_image, imgsz=self.imgsz, conf=self.conf, iou=0.6,
                               retina_masks=True, verbose=False)
        if not r or r[0].masks is None:
            return []
        masks = []
        for m in r[0].masks.data.cpu().numpy():
            mm = cv2.resize(m.astype(np.uint8), (w, h)) > 0.5
            if mm.sum() < self.min_area:
                continue
            ys, xs = np.where(mm)
            if ys.mean() < y_cut:  # fixed background sticker, not a panel label
                continue
            masks.append(mm)
        return masks


def to_json(masks, image_name="image.png", image_size=(640, 400)):
    """Convert label masks to JSON format."""
    result = {"image_name": image_name, "image_size": image_size, "labels": []}
    for mask in masks:
        cn, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cn:
            continue
        c = max(cn, key=cv2.contourArea)
        ap = cv2.approxPolyDP(c, 0.01 * cv2.arcLength(c, True), True).reshape(-1, 2)
        if len(ap) < 3:
            continue
        poly = [[float(x) / image_size[0], float(y) / image_size[1]] for x, y in ap]
        result["labels"].append({"polygon": poly, "area_pixels": int(mask.sum())})
    return result
