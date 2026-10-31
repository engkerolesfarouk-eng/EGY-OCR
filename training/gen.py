"""Synthetic training lines for the Arabic recognizer.

Each sample is a rendered text line (scan-like noise) and its label in the
model's *visual* order (what the recognizer emits left to right), so that
arocr.textfix.visual_to_logical(label) == the logical text.

Focus on what the stock model gets wrong in books and documents:
Arabic-Indic digits, scripture references "(رو ٣: ١٢)", « » quotes,
brackets, !! ؟ ؛ and spacing.
"""
from __future__ import annotations

import glob
import os
HERE = os.path.dirname(os.path.abspath(__file__))
import random
import re
import sqlite3
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, HERE + "/..")
from arocr.textfix import LTR_RUN, has_arabic, visual_to_logical  # noqa: E402

AR = os.path.join(os.path.dirname(__import__("arramooz").__file__), "data")
FONTS = sorted(glob.glob(HERE + "/fonts/*.[tT][tT][fF]"))
_MIRROR = str.maketrans("()[]{}<>«»", ")(][}{><»«")
IND = str.maketrans("0123456789", "٠١٢٣٤٥٦٧٨٩")

BOOKS = ["مت", "مر", "لو", "يو", "أع", "رو", "١كو", "٢كو", "غل", "أف", "في", "كو", "عب", "يع",
         "١بط", "١يو", "رؤ", "تك", "خر", "مز", "أم", "إش", "إر", "حز", "دا", "هو"]


def logical_to_visual(text: str) -> str:
    """Exact inverse of visual_to_logical for one line."""
    pieces, pos = [], 0
    for m in LTR_RUN.finditer(text):
        pieces.extend(text[pos:m.start()])
        pieces.append(m.group())
        pos = m.end()
    pieces.extend(text[pos:])
    return "".join(p.translate(_MIRROR) if len(p) == 1 else p for p in reversed(pieces))


def load_words():
    c = sqlite3.connect(os.path.join(AR, "wordfreq.sqlite"))
    rows = c.execute("select unvocalized, freq from wordfreq where length(unvocalized) > 1").fetchall()
    words = [w for w, f in rows if re.fullmatch(r"[ء-ي]+", w)]
    freqs = np.array([f for w, f in rows if re.fullmatch(r"[ء-ي]+", w)], float)
    p = freqs ** 0.5  # flatten Zipf so rarer words appear too
    return words, p / p.sum()


def load_sentences():
    t = open(os.path.join(AR, "randomtext.txt"), encoding="utf-8-sig").read()
    t = re.sub(r"[\u200b-\u200f\u202a-\u202e\u2066-\u2069\ufeff]", "", t)
    t = re.sub(r"\s+", " ", t)
    return [s.strip() for s in re.split(r"(?<=[.!؟])\s", t) if len(s.strip()) > 10]


class Corpus:
    def __init__(self, seed=0):
        self.r = random.Random(seed)
        self.words, self.p = load_words()
        self.sents = load_sentences()
        self.np = np.random.default_rng(seed)

    def num(self, lo=1, hi=99):
        return str(self.r.randint(lo, hi)).translate(IND)

    def ref(self):
        b = self.r.choice(BOOKS)
        k = self.r.random()
        if k < 0.6:
            s = f"{b} {self.num(1, 28)}: {self.num(1, 40)}"
        elif k < 0.85:
            s = f"{b} {self.num(1, 28)}: {self.num(1, 30)}-{self.num(1, 40)}"
        else:
            s = f"راجع {b} {self.num(1, 28)}: {self.num(1, 40)}"
        return f"({s})"

    def snippet(self, n):
        if self.r.random() < 0.5:
            ws = self.np.choice(self.words, n, p=self.p).tolist()
        else:
            s = self.r.choice(self.sents).split()
            i = self.r.randint(0, max(0, len(s) - n))
            ws = s[i:i + n]
        return " ".join(ws)

    def token(self):
        k = self.r.random()
        if k < 0.10:
            return self.ref()
        if k < 0.17:
            return "«" + self.snippet(self.r.randint(1, 4)) + "»"
        if k < 0.22:
            return "(" + self.snippet(self.r.randint(1, 3)) + ")"
        if k < 0.27:
            d = f"{self.num(1, 28)}/{self.num(1, 12)}/{self.r.randint(1950, 2030)}"
            return d.translate(IND) if self.r.random() < 0.7 else d.translate(str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789"))
        if k < 0.32:
            n = f"{self.r.randint(1, 999_999):,}"
            return n.translate(IND).replace(",", "٬") if self.r.random() < 0.5 else n
        if k < 0.35:
            return "— " + self.num(1, 300) + " —"
        if k < 0.37:
            return self.r.choice(["PO-", "INV-", "KM", "No. "]) + str(self.r.randint(10, 99999))
        return self.snippet(self.r.randint(1, 4))

    def line(self, max_chars=60):
        parts = []
        while sum(len(p) + 1 for p in parts) < self.r.randint(15, max_chars):
            t = self.token()
            r = self.r.random()
            if r < 0.12:
                t += self.r.choice(["،", "؛", "!!", "!", "؟", ".", "...", ":"])
            parts.append(t)
        s = " ".join(parts)
        if self.r.random() < 0.15:
            s = s.replace(" ،", "،")
        return re.sub(r"\s+", " ", s).strip()[:90]


def render(text: str, font_path: str, size: int, r: random.Random) -> np.ndarray | None:
    try:
        font = ImageFont.truetype(font_path, size)
    except OSError:
        return None
    probe = ImageDraw.Draw(Image.new("L", (1, 1)))
    l, t, rr, b = probe.textbbox((0, 0), text, font=font, direction="rtl", language="ar")
    w, h = rr - l, b - t
    if w <= 0 or h <= 0:
        return None
    padx, pady = r.randint(2, 12), r.randint(2, 8)
    img = Image.new("L", (w + 2 * padx, h + 2 * pady), 255)
    ImageDraw.Draw(img).text((padx - l, pady - t), text, font=font, fill=r.randint(0, 60),
                             direction="rtl", language="ar")
    return np.array(img)


def degrade(img: np.ndarray, r: random.Random) -> np.ndarray:
    """Make it look photocopied / scanned."""
    if r.random() < 0.4:
        k = np.ones((2, 2), np.uint8)
        img = cv2.erode(img, k) if r.random() < 0.5 else cv2.dilate(img, k)  # ink spread/thin
    if r.random() < 0.5:
        s = r.uniform(0.45, 0.9)  # low resolution then back up
        h, w = img.shape
        img = cv2.resize(cv2.resize(img, (max(8, int(w * s)), max(8, int(h * s))), interpolation=cv2.INTER_AREA),
                         (w, h), interpolation=cv2.INTER_LINEAR)
    if r.random() < 0.5:
        img = cv2.GaussianBlur(img, (3, 3), r.uniform(0.3, 1.1))
    if r.random() < 0.3:  # slight rotation
        h, w = img.shape
        m = cv2.getRotationMatrix2D((w / 2, h / 2), r.uniform(-1.2, 1.2), 1.0)
        img = cv2.warpAffine(img, m, (w, h), borderValue=255)
    img = img.astype(np.float32)
    if r.random() < 0.5:
        img = img * r.uniform(0.75, 1.0) + r.uniform(0, 40)  # paper tone
    if r.random() < 0.6:
        img += np.random.default_rng(r.randint(0, 1 << 30)).normal(0, r.uniform(3, 14), img.shape)
    img = np.clip(img, 0, 255).astype(np.uint8)
    if r.random() < 0.5:
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, r.randint(30, 85)])
        img = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


def main(out_dir: str, n: int, seed: int):
    os.makedirs(out_dir, exist_ok=True)
    corpus = Corpus(seed)
    r = random.Random(seed + 1)
    labels = []
    i = 0
    while i < n:
        text = corpus.line()
        if not has_arabic(text):
            continue
        vis = logical_to_visual(text)
        assert visual_to_logical(vis) == text, (text, vis)
        img = render(text, r.choice(FONTS), r.randint(22, 44), r)
        if img is None or img.shape[1] > 2000:
            continue
        img = degrade(img, r)
        name = f"{i:06d}.png"
        cv2.imwrite(os.path.join(out_dir, name), img)
        labels.append(f"{name}\t{vis}\t{text}")
        i += 1
    with open(os.path.join(out_dir, "labels.tsv"), "w", encoding="utf-8") as f:
        f.write("\n".join(labels) + "\n")


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]) if len(sys.argv) > 3 else 0)
