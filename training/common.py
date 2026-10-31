"""Shared helpers: charset, image prep, CTC decode, CER."""
from __future__ import annotations

import os
HERE = os.path.dirname(os.path.abspath(__file__))

import cv2
import numpy as np

DICT = HERE + "/../models/ppocrv5_arabic_dict.txt"
H = 48


def load_charset():
    with open(DICT, encoding="utf-8") as f:
        chars = [l.rstrip("\n").rstrip("\r") for l in f]
    chars = [c for c in chars if c != ""]
    charset = ["<blank>"] + chars + [" "]
    index = {}
    for i, c in enumerate(charset):
        index[c] = i  # duplicates in the dict: the model uses the LAST one (PaddleOCR builds its map that way)
    return charset, index


def prep(img: np.ndarray, max_w: int = 960) -> np.ndarray:
    h, w = img.shape[:2]
    nw = min(max_w, max(16, int(np.ceil(H * w / h))))
    im = cv2.resize(img, (nw, H)).astype(np.float32)
    return ((im / 255.0 - 0.5) / 0.5).transpose(2, 0, 1)


def load_labels(d):
    rows = []
    with open(os.path.join(d, "labels.tsv"), encoding="utf-8") as f:
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) >= 2:
                rows.append((os.path.join(d, p[0]), p[1]))
    return rows


def greedy(probs: np.ndarray, charset) -> str:
    ids = probs.argmax(-1)
    out, prev = [], -1
    for c in ids:
        if c != 0 and c != prev:
            out.append(charset[c])
        prev = c
    return "".join(out)


def edit(a, b):
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]
