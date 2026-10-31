"""Image preparation before OCR."""
from __future__ import annotations

import cv2
import numpy as np


def clahe(img: np.ndarray, clip: float = 2.0, tiles: int = 8) -> np.ndarray:
    """Grayscale + CLAHE (local contrast), returned as 3-channel BGR.

    Helps photos with uneven lighting or glare (ID cards shot with a phone).
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    enhanced = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tiles, tiles)).apply(gray)
    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
