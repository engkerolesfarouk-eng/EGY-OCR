"""Line recognition with a PP-OCR CTC recognizer (ONNX)."""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
import onnxruntime as ort


@dataclass
class RecResult:
    text: str                      # visual order, exactly as the model reads it
    confidence: float
    # Horizontal position of each character in `text`, 0..1 across the crop.
    char_x: list[float] = field(default_factory=list)
    char_conf: list[float] = field(default_factory=list)


class Recognizer:
    def __init__(self, model_path: str, dict_path: str, height: int = 48, max_width: int = 3200):
        from .cpu import session_options
        so = session_options()
        self.sess = ort.InferenceSession(model_path, so, providers=["CPUExecutionProvider"])
        self.inp = self.sess.get_inputs()[0].name
        with open(dict_path, encoding="utf-8") as f:
            chars = [line.rstrip("\n").rstrip("\r") for line in f]
        chars = [c for c in chars if c != ""]
        # Index 0 is the CTC blank; PP-OCR appends a space as the last class.
        self.charset = ["<blank>"] + chars + [" "]
        self.height = height
        self.max_width = max_width
        self.stretch = 1.0      # >1 gives narrow ligatures more time steps (repeated letters: "الله")

    def __call__(self, crops: list[np.ndarray], batch: int = 8) -> list[RecResult]:
        results: list[RecResult | None] = [None] * len(crops)
        order = sorted(range(len(crops)), key=lambda i: crops[i].shape[1] / max(1, crops[i].shape[0]))
        for start in range(0, len(order), batch):
            idx = order[start:start + batch]
            widths = [min(self.max_width, max(16, int(np.ceil(self.stretch * self.height * crops[i].shape[1] / max(1, crops[i].shape[0])))))
                      for i in idx]
            W = max(widths)
            x = np.zeros((len(idx), 3, self.height, W), np.float32)
            for k, i in enumerate(idx):
                im = cv2.resize(crops[i], (widths[k], self.height)).astype(np.float32)
                im = (im / 255.0 - 0.5) / 0.5
                x[k, :, :, :widths[k]] = im.transpose(2, 0, 1)
            out = self.sess.run(None, {self.inp: x})[0]  # N, T, C
            for k, i in enumerate(idx):
                results[i] = self._decode(out[k], widths[k] / W)
        return results  # type: ignore[return-value]

    def _decode(self, probs: np.ndarray, valid_frac: float) -> RecResult:
        ids = probs.argmax(1)
        conf = probs.max(1)
        T = len(ids)
        text, xs, cs = [], [], []
        prev = -1
        for t, c in enumerate(ids):
            if c != 0 and c != prev and c < len(self.charset):
                text.append(self.charset[c])
                # Position across the real (unpadded) part of the line image.
                xs.append(min(1.0, ((t + 0.5) / T) / max(valid_frac, 1e-6)))
                cs.append(float(conf[t]))
            prev = c
        return RecResult("".join(text), float(np.mean(cs)) if cs else 0.0, xs, cs)


def crop_quad(img: np.ndarray, quad: np.ndarray) -> np.ndarray:
    """Perspective-crop a detected quad to an upright line image."""
    tl, tr, br, bl = quad
    w = int(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl)))
    h = int(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr)))
    w, h = max(w, 1), max(h, 1)
    dst = np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float32)
    m = cv2.getPerspectiveTransform(quad.astype(np.float32), dst)
    crop = cv2.warpPerspective(img, m, (w, h), borderMode=cv2.BORDER_REPLICATE, flags=cv2.INTER_CUBIC)
    if h > 1.5 * w:  # vertical text box: read it rotated (direction chosen by the caller)
        crop = np.rot90(crop).copy()
    return crop
