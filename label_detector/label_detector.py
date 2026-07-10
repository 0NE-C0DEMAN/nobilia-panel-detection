"""YOLO26 label (sticker) segmentation detector for small objects."""
from ultralytics import YOLO
import cv2
import numpy as np


class YoloLabelParams:
    def __init__(self, weights="models/label_seg_best.pt", conf=0.4, imgsz=1280, min_area=100):
        self.weights = weights
        self.conf = conf
        self.imgsz = imgsz
        self.min_area = min_area


class YoloLabelDetector:
    def __init__(self, weights="models/label_seg_best.pt", conf=0.4, imgsz=1280, min_area=100):
        self.model = YOLO(weights)
        self.conf = conf
        self.imgsz = imgsz
        self.min_area = min_area

    def detect(self, rgb_image):
        """Detect labels in RGB image. Returns list of binary masks."""
        h, w = rgb_image.shape[:2]
        r = self.model.predict(rgb_image, imgsz=self.imgsz, conf=self.conf, iou=0.6, retina_masks=True, verbose=False)
        if not r or r[0].masks is None:
            return []
        masks = []
        for m in r[0].masks.data.cpu().numpy():
            mm = cv2.resize(m.astype(np.uint8), (w, h)) > 0.5
            if mm.sum() >= self.min_area:
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
