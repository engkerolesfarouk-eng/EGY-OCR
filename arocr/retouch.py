"""Show the user's corrections on the page itself.

For every word the user changed in the text pane, the original word is removed
from the scan (filled from the paper around it) and the new word is drawn in its
place. To make it look like the rest of the page, the font is matched first:
the ORIGINAL word is rendered in every installed Arabic font (regular and bold)
and compared with the pixels on the scan; the closest font, at the size that
gives the same letter height, is used for the new word, on the same baseline
and in the same ink colour. Shaping (joined letters, right to left) is Qt's.
"""
from __future__ import annotations

import os
import sys

import cv2
import numpy as np

_FONTS: list[tuple[str, bool]] | None = None
_EXTRA_FONT_DIRS = [os.path.join(os.path.dirname(__file__), "fonts")]


def _qt():
    from PySide6.QtGui import QGuiApplication
    if QGuiApplication.instance() is None:          # command line: Qt needs an app object
        _qt.app = QGuiApplication(sys.argv[:1])


def _candidates() -> list[tuple[str, bool]]:
    """(family, bold) for every installed font that has Arabic letters."""
    global _FONTS
    if _FONTS is None:
        from PySide6.QtGui import QFontDatabase
        for d in _EXTRA_FONT_DIRS + [os.environ.get("AROCR_FONTS", "")]:
            if d and os.path.isdir(d):
                for f in os.listdir(d):
                    if f.lower().endswith((".ttf", ".otf")):
                        QFontDatabase.addApplicationFont(os.path.join(d, f))
        from PySide6.QtGui import QFont, QRawFont
        fams = []
        for f in QFontDatabase.families():
            raw = QRawFont.fromFont(QFont(f))
            if raw.isValid() and raw.supportsCharacter(0x0639) and raw.supportsCharacter(0x0645):
                fams.append(f)
        skip = ("mono", "unifont", "symbol", "webdings", "wingdings", "emoji", "mdl2", "holomdl", "marlett")
        fams = [f for f in fams if not f.startswith("@") and not any(s in f.lower() for s in skip)]
        _FONTS = [(f, b) for f in fams for b in (False, True)] or [("Arial", False), ("Arial", True)]
    return _FONTS


def _font(family: str, bold: bool, px: float):
    from PySide6.QtGui import QFont
    f = QFont(family)
    f.setBold(bold)
    f.setPixelSize(max(4, int(round(px))))
    return f


def _render(text: str, family: str, bold: bool, px: float, rtl: bool):
    """Black text on white. Returns (ink mask, ink bbox relative to the pen
    origin (x0, y0, x1, y1), advance width)."""
    from PySide6.QtCore import QPointF, Qt
    from PySide6.QtGui import QColor, QFontMetricsF, QImage, QPainter
    font = _font(family, bold, px)
    fm = QFontMetricsF(font)
    adv = fm.horizontalAdvance(text)
    pad = int(px)
    w = int(adv + 2 * pad) + 2
    h = int(fm.height() + 2 * pad) + 2
    img = QImage(w, h, QImage.Format.Format_Grayscale8)
    img.fill(QColor(255, 255, 255))
    p = QPainter(img)
    p.setRenderHint(QPainter.RenderHint.TextAntialiasing)
    p.setFont(font)
    p.setPen(QColor(0, 0, 0))
    p.setLayoutDirection(Qt.LayoutDirection.RightToLeft if rtl else Qt.LayoutDirection.LeftToRight)
    ox, oy = pad, pad + fm.ascent()
    p.drawText(QPointF(ox, oy), text)
    p.end()
    a = np.frombuffer(img.constBits(), np.uint8).reshape(h, img.bytesPerLine())[:, :w]
    ink = a < 128
    ys, xs = np.nonzero(ink)
    if len(xs) == 0:
        return None, None, adv
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    return ink[y0:y1, x0:x1], (x0 - ox, y0 - oy, x1 - ox, y1 - oy), adv


def _ink(crop: np.ndarray):
    """Ink mask of a word crop (BGR) and its bbox in the crop."""
    g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    g = cv2.GaussianBlur(g, (3, 3), 0)
    _, m = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    m = m > 0
    # drop specks: keep components of a reasonable size
    n, lab, st, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), 8)
    keep = np.zeros_like(m)
    for i in range(1, n):
        if st[i, cv2.CC_STAT_AREA] >= max(3, m.size * 0.0005):
            keep |= lab == i
    ys, xs = np.nonzero(keep)
    if len(xs) == 0:
        return None, None
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    return keep[y0:y1, x0:x1], (x0, y0, x1, y1)


def _score(a: np.ndarray, b: np.ndarray) -> float:
    """Shape similarity of two ink masks (b is resized onto a)."""
    h, w = a.shape
    if h < 2 or w < 2:
        return 0.0
    bb = cv2.resize(b.astype(np.uint8) * 255, (w, h), interpolation=cv2.INTER_AREA) > 100
    k = np.ones((3, 3), np.uint8)
    ad = cv2.dilate(a.astype(np.uint8), k) > 0
    bd = cv2.dilate(bb.astype(np.uint8), k) > 0
    # how much of each falls on the other (tolerant to 1px), plus stroke weight
    p = (bb & ad).sum() / max(1, bb.sum())
    r = (a & bd).sum() / max(1, a.sum())
    weight = min(a.mean(), bb.mean()) / max(1e-6, max(a.mean(), bb.mean()))
    aspect = 1.0  # same box by construction
    return 2 * p * r / max(1e-6, p + r) * (0.6 + 0.4 * weight) * aspect


def _fit(mask, bbox, text, fam, bold, rtl):
    """(score, pixel size) of one font for one word printed on the scan."""
    m, rb, _ = _render(text, fam, bold, 100, rtl)
    if m is None:
        return 0.0, None
    ink_h = bbox[3] - bbox[1]
    px = 100.0 * ink_h / max(1, rb[3] - rb[1])
    width_ratio = (rb[2] - rb[0]) * px / 100.0 / max(1, bbox[2] - bbox[0])
    if not 0.55 < width_ratio < 1.8:
        return 0.0, px
    return _score(mask, m) * (1 - min(0.6, abs(np.log(width_ratio)))), px


def match_font(samples, rtl: bool):
    """samples: [(crop BGR, text)] of words printed in the same font (the
    changed word and its neighbours). Returns (family, bold) that fits best."""
    inks = []
    for crop, text in samples:
        if len(text.strip()) < 2:
            continue
        mask, bbox = _ink(crop)
        if mask is not None and bbox[3] - bbox[1] >= 6:
            inks.append((mask, bbox, text))
    if not inks:
        return None
    scores = {}
    for fam, bold in _candidates():
        fits = [_fit(m, b, t, fam, bold, rtl) for m, b, t in inks]
        scores[(fam, bold)] = (float(np.mean([f[0] for f in fits])),
                               float(np.median([f[1] for f in fits if f[1]] or [0])))
    (fam, bold), (s, px) = max(scores.items(), key=lambda kv: kv[1][0])
    if bold and (fam, False) in scores and scores[(fam, False)][0] > s * 0.95:
        bold, px = False, scores[(fam, False)][1]          # bold only when clearly bolder
    return fam, bold, px


def _ink_colour(img, x0, y0, x1, y1):
    patch = img[y0:y1, x0:x1].reshape(-1, 3).astype(np.float32)
    if len(patch) == 0:
        return (0, 0, 0)
    lum = patch.sum(1)
    dark = patch[lum <= np.percentile(lum, 3)]
    b, g, r = np.median(dark, axis=0)
    return int(r), int(g), int(b)


def _erase(img, x0, y0, x1, y1):
    """Remove the ink inside the box, keeping the paper (texture, colour)."""
    h, w = img.shape[:2]
    pad = max(3, (y1 - y0) // 4)
    X0, Y0, X1, Y1 = max(0, x0 - pad), max(0, y0 - pad), min(w, x1 + pad), min(h, y1 + pad)
    roi = img[Y0:Y1, X0:X1]
    mask = np.zeros(roi.shape[:2], np.uint8)
    mask[y0 - Y0:y1 - Y0, x0 - X0:x1 - X0] = 255
    img[Y0:Y1, X0:X1] = cv2.inpaint(roi, mask, max(3, (y1 - y0) // 6), cv2.INPAINT_TELEA)


def _draw_at(img, text, family, bold, px, pen_x, pen_y, colour, rtl):
    from PySide6.QtCore import QPointF, Qt
    from PySide6.QtGui import QColor, QImage, QPainter
    h, w = img.shape[:2]
    rgb = np.ascontiguousarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    q = QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888)
    p = QPainter(q)
    p.setRenderHint(QPainter.RenderHint.TextAntialiasing)
    p.setFont(_font(family, bold, px))
    p.setPen(QColor(*colour))
    p.setLayoutDirection(Qt.LayoutDirection.RightToLeft if rtl else Qt.LayoutDirection.LeftToRight)
    p.drawText(QPointF(pen_x, pen_y), text)
    p.end()
    out = np.frombuffer(q.constBits(), np.uint8).reshape(h, q.bytesPerLine())[:, :3 * w].reshape(h, w, 3)
    img[:] = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)


def apply(img: np.ndarray, changes) -> np.ndarray:
    """changes: [dict(erase=box, word=box, old=str, new=str, rtl=bool,
    samples=[(box, text)] words of the same line as printed)]. Boxes in pixels."""
    if not changes:
        return img
    try:
        _qt()
    except ImportError:
        return img
    out = img.copy()
    src = img
    h, w = out.shape[:2]

    def crop(b, im=src):
        x0, y0 = max(0, int(b[0])), max(0, int(b[1]))
        x1, y1 = min(w, int(np.ceil(b[2]))), min(h, int(np.ceil(b[3])))
        return im[y0:y1, x0:x1], (x0, y0, x1, y1)

    fonts = {}
    for ch in changes:
        (wx0, wy0, wx1, wy1) = crop(ch["word"])[1]
        (ex0, ey0, ex1, ey1) = crop(ch["erase"])[1]
        if ex1 - ex0 < 2 or ey1 - ey0 < 2:
            continue
        colour = _ink_colour(src, wx0, wy0, wx1, wy1)
        # one font per line, judged on all its words as printed
        key = tuple(ch["samples"])
        if key not in fonts:
            fonts[key] = match_font([(crop(b)[0], t) for b, t in ch["samples"]], ch["rtl"])
        font = fonts[key] or ("Arial", False, None)
        old_mask, old_ink = _ink(src[wy0:wy1, wx0:wx1]) if ch["old"].strip() else (None, None)
        _erase(out, ex0, ey0, ex1, ey1)
        new = ch["new"]
        if not new.strip():
            continue
        rtl = ch["rtl"]
        px = None
        if old_ink is not None:          # size: the old word's letters, in this font
            _, px = _fit(old_mask, old_ink, ch["old"], font[0], font[1], rtl)
            ink = (wx0 + old_ink[0], wy0 + old_ink[1], wx0 + old_ink[2], wy0 + old_ink[3])
        if not px:
            px = (wy1 - wy0) * 0.5
            ink = (wx0, wy0 + (wy1 - wy0) * 0.2, wx1, wy0 + (wy1 - wy0) * 0.8)
        _, ob, _ = _render(ch["old"] if ch["old"].strip() else new, font[0], font[1], px, rtl)
        _, nb, _ = _render(new, font[0], font[1], px, rtl)
        if nb is None:
            continue
        pen_y = ink[1] - (ob[1] if ob else nb[1])
        pen_x = ink[2] - nb[2] if rtl else ink[0] - nb[0]   # Arabic starts at the right edge
        _draw_at(out, new, font[0], font[1], px, pen_x, pen_y, colour, rtl)
    return out
