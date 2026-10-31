"""Text-line detection with the PP-OCR DB detector (ONNX)."""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import onnxruntime as ort
import pyclipper


@dataclass
class DetBox:
    quad: np.ndarray  # 4x2 float32, clockwise from top-left, in image pixels
    score: float


class Detector:
    def __init__(self, model_path: str, limit_side: int = 1600,
                 thresh: float = 0.3, box_thresh: float = 0.5, unclip_ratio: float = 1.6):
        from .cpu import session_options
        so = session_options()
        self.sess = ort.InferenceSession(model_path, so, providers=["CPUExecutionProvider"])
        self.inp = self.sess.get_inputs()[0].name
        self.limit_side = limit_side
        self.thresh = thresh
        self.box_thresh = box_thresh
        self.unclip_ratio = unclip_ratio

    def _resize(self, img: np.ndarray, limit: int | None = None, floor: int | None = 960,
                upscale: float | None = None):
        h, w = img.shape[:2]
        ratio = upscale or min(1.0, (limit or self.limit_side) / max(h, w))
        # Small images (an ID card photo) gain from being brought up to a size the
        # detector was trained on; never shrink below ~960 on the long side.
        if floor and max(h, w) * ratio < floor:
            ratio = floor / max(h, w)
        nh = max(32, int(round(h * ratio / 32)) * 32)
        nw = max(32, int(round(w * ratio / 32)) * 32)
        return cv2.resize(img, (nw, nh)), nh / h, nw / w

    def __call__(self, img: np.ndarray, min_height: float = 10.0) -> list[DetBox]:
        boxes = [b for b in self._detect(img) if _height(b) >= min_height]
        # Very large, widely spaced characters (a car plate, a poster title) are
        # invisible to the detector at full size; a coarse pass at small scale
        # finds them. Keep coarse boxes only where the fine pass found nothing.
        h, w = img.shape[:2]
        covered = np.zeros((h, w), np.uint8)
        for b in boxes:
            cv2.fillPoly(covered, [b.quad.astype(np.int32)], 1)
        # Small text (a photo of a card taken from a distance): faint words such
        # as a first name are missed at 1:1. Look again at double size.
        heights = sorted(_height(b) for b in boxes)
        # (a full page of text - a book - already has its lines: the double-size
        # pass would only cost time)
        if heights and heights[len(heights) // 4] < 36 and max(h, w) <= 2400 and len(boxes) <= 25:
            for b in self._detect(img, upscale=2.0, floor=None):
                if _height(b) < 8:
                    continue
                m = np.zeros((h, w), np.uint8)
                cv2.fillPoly(m, [b.quad.astype(np.int32)], 1)
                area = m.sum()
                if area and (covered & m).sum() / area < 0.3:
                    boxes.append(b)
                    covered |= m
        if max(h, w) > 700:
            for b in self._detect(img, limit=640, floor=None):
                if _height(b) < 3 * min_height:
                    continue
                m = np.zeros((h, w), np.uint8)
                cv2.fillPoly(m, [b.quad.astype(np.int32)], 1)
                area = m.sum()
                if area and (covered & m).sum() / area < 0.3:
                    boxes.append(b)
                    covered |= m
        return boxes

    def _detect(self, img: np.ndarray, limit: int | None = None, floor: int | None = 960,
                upscale: float | None = None) -> list[DetBox]:
        resized, ry, rx = self._resize(img, limit, floor, upscale)
        x = resized.astype(np.float32) / 255.0
        x = (x - np.array([0.485, 0.456, 0.406], np.float32)) / np.array([0.229, 0.224, 0.225], np.float32)
        x = x.transpose(2, 0, 1)[None]
        prob = self.sess.run(None, {self.inp: x})[0][0, 0]

        mask = (prob > self.thresh).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        boxes: list[DetBox] = []
        for cnt in contours:
            if cv2.contourArea(cnt) < 6:
                continue
            score = _box_score(prob, cnt)
            if score < self.box_thresh:
                continue
            expanded = _unclip(cnt.reshape(-1, 2), self.unclip_ratio)
            if expanded is None:
                continue
            rect = cv2.minAreaRect(expanded)
            if min(rect[1]) < 4:
                continue
            quad = _order_quad(cv2.boxPoints(rect))
            quad[:, 0] /= rx
            quad[:, 1] /= ry
            quad[:, 0] = np.clip(quad[:, 0], 0, img.shape[1] - 1)
            quad[:, 1] = np.clip(quad[:, 1], 0, img.shape[0] - 1)
            boxes.append(DetBox(quad.astype(np.float32), float(score)))
        return boxes


def _height(b: DetBox) -> float:
    tl, tr, br, bl = b.quad
    return float(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr)))


def _box_score(prob: np.ndarray, cnt: np.ndarray) -> float:
    x, y, w, h = cv2.boundingRect(cnt)
    mask = np.zeros((h, w), np.uint8)
    cv2.fillPoly(mask, [cnt.reshape(-1, 2) - [x, y]], 1)
    return float(cv2.mean(prob[y:y + h, x:x + w], mask)[0])


def _unclip(poly: np.ndarray, ratio: float):
    area = cv2.contourArea(poly)
    length = cv2.arcLength(poly.reshape(-1, 1, 2), True)
    if length == 0:
        return None
    distance = area * ratio / length
    off = pyclipper.PyclipperOffset()
    off.AddPath(poly.tolist(), pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
    out = off.Execute(distance)
    if not out:
        return None
    return np.array(out[0], np.float32)


def _order_quad(pts: np.ndarray) -> np.ndarray:
    pts = pts[np.argsort(pts[:, 0])]
    left, right = pts[:2], pts[2:]
    tl, bl = left[np.argsort(left[:, 1])]
    tr, br = right[np.argsort(right[:, 1])]
    return np.array([tl, tr, br, bl], np.float32)
