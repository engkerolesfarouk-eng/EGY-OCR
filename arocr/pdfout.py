"""Write a searchable PDF: the page image with an invisible text layer on top."""
from __future__ import annotations

import io
import os

import cv2
import numpy as np
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas

from .layout import Line, Word
from .textfix import visual_to_logical

_FONT = "OcrText"
_FONT_CANDIDATES = [
    os.path.join(os.path.dirname(__file__), "fonts", "DejaVuSans.ttf"),
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    r"C:\Windows\Fonts\arial.ttf",
    r"C:\Windows\Fonts\tahoma.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
]
_registered = False


def _register_font():
    global _registered
    if _registered:
        return
    for path in _FONT_CANDIDATES:
        if os.path.exists(path):
            pdfmetrics.registerFont(TTFont(_FONT, path))
            _registered = True
            return
    raise RuntimeError("No font with Arabic glyphs found for the text layer")


class SearchablePdf:
    def __init__(self, out_path: str):
        _register_font()
        self.c = canvas.Canvas(out_path)
        self.c.setTitle("OCR")

    def add_page(self, img: np.ndarray, lines: list[Line], page_w_pt: float, page_h_pt: float,
                 jpeg_quality: int = 80, edited_text: str | None = None):
        """edited_text: the page text as corrected by the user. Each corrected
        line replaces the text layer of the line in the same position, so
        search and copy give the corrected words over the same spot."""
        c = self.c
        c.setPageSize((page_w_pt, page_h_pt))
        new_text = _map_edits(lines, edited_text)
        changes = [ch for line in lines if id(line) in new_text
                   for ch in _changes(line, new_text[id(line)])]
        if changes and not _is_identity_document(lines):   # show the corrections on the page too
            from .retouch import apply
            img = apply(img, changes)
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
        c.drawImage(ImageReader(io.BytesIO(buf.tobytes())), 0, 0, page_w_pt, page_h_pt)

        h_px, w_px = img.shape[:2]
        sx, sy = page_w_pt / w_px, page_h_pt / h_px
        for line in lines:
            # Emit words in reading order: viewers that copy text in content-stream
            # order (pdfium: Chrome, Edge) then get the right word order too.
            words = _words_for(line, new_text.get(id(line)))
            if any(w.quad is not None for w in words):   # a turned card: keep the card's own order
                words = list(reversed(words)) if line.rtl else list(words)
            else:
                words = sorted(words, key=lambda w: -w.box[2] if line.rtl else w.box[0])
            for word in words:
                if not word.text:
                    continue
                # Store characters in visual (left-to-right) order for RTL words:
                # viewers reconstruct reading order from glyph positions, the same
                # way they handle real Arabic PDFs, so search and copy come out right.
                s = word.visual if (word.visual and _rtl_word(word.text)) else word.text
                if word.quad is not None:
                    _draw_turned(c, s, word.quad, sx, sy, page_h_pt)
                    continue
                x0, y0, x1, y1 = word.box
                box_w = (x1 - x0) * sx
                box_h = (y1 - y0) * sy
                size = max(1.0, box_h * 0.75)
                natural = pdfmetrics.stringWidth(s, _FONT, size)
                t = c.beginText()
                t.setTextRenderMode(3)  # invisible
                t.setFont(_FONT, size)
                t.setHorizScale(100.0 * box_w / natural if natural > 0 else 100.0)
                t.setTextOrigin(x0 * sx, page_h_pt - y1 * sy + box_h * 0.2)
                t.textOut(s + " ")
                c.drawText(t)
        c.showPage()

    def save(self):
        self.c.save()


def _rtl_word(text: str) -> bool:
    from .textfix import has_arabic
    return has_arabic(text)


def _map_edits(lines: list[Line], edited_text: str | None) -> dict:
    """{id(line): corrected text} for lines the user changed."""
    if not edited_text:
        return {}
    edited = [t.strip() for t in edited_text.split("\n") if t.strip()]
    if not edited or edited == [l.text for l in lines]:
        return {}
    out = {}
    n = min(len(edited), len(lines))
    for line, t in zip(lines[:n], edited[:n]):
        if t != line.text:
            out[id(line)] = t
    if len(edited) > len(lines) and lines:           # extra lines: keep them, on the last line
        last = lines[-1]
        out[id(last)] = " ".join([out.get(id(last), last.text)] + edited[n:])
    for line in lines[n:]:                           # lines the user deleted
        out[id(line)] = ""
    return out


def _to_visual(text: str) -> str:
    # the visual<->logical reordering is its own inverse
    return visual_to_logical(text) if _rtl_word(text) else text


def _words_for(line: Line, text: str | None) -> list[Word]:
    if text is None:
        if sorted(w.text for w in line.words if w.text) == sorted(line.text.split()):
            return line.words
        text = line.text        # fixed after reading (joined ID number, birth date)
    if not text:
        return []
    parts = text.split()
    words = sorted(line.words, key=lambda w: w.box[0])
    if words and len(parts) == len(words):
        order = parts[::-1] if line.rtl else parts      # boxes left->right, Arabic reads right->left
        return [Word(t, w.box, 1.0, _to_visual(t), w.quad) for t, w in zip(order, words)]
    # word count changed: the whole corrected line over the line's area
    box = (min(r.x0 for r in line.regions), min(r.y0 for r in line.regions),
           max(r.x1 for r in line.regions), max(r.y1 for r in line.regions))
    quad = None
    if any(w.quad is not None for w in line.words) and len(line.regions) == 1:
        quad = line.regions[0].quad          # a turned card: along the printed line
    return [Word(text, box, 1.0, _to_visual(text), quad)]


def _line_box(line: Line):
    return (min(r.x0 for r in line.regions), min(r.y0 for r in line.regions),
            max(r.x1 for r in line.regions), max(r.y1 for r in line.regions))


def _changes(line: Line, text: str):
    """What the user changed on this line, as boxes on the scan (see retouch.apply)."""
    lb = _line_box(line)
    words = sorted((w for w in line.words if w.text), key=lambda w: w.box[0])
    samples = [((w.box[0], lb[1], w.box[2], lb[3]), w.text) for w in words] or [(lb, line.text)]
    if not text:
        return [dict(erase=lb, word=lb, old=line.text, new="", rtl=line.rtl, samples=samples)]
    parts = text.split()
    reading = sorted(words, key=lambda w: -w.box[0] if line.rtl else w.box[0])
    if not words or len(parts) != len(words) or " ".join(w.text for w in reading) != line.text:
        return [dict(erase=lb, word=lb, old=line.text, new=text, rtl=line.rtl, samples=samples)]
    order = parts[::-1] if line.rtl else parts
    out = []
    lx0, ly0, lx1, ly1 = lb
    h = ly1 - ly0
    for i, (t, w) in enumerate(zip(order, words)):
        if t == w.text:
            continue
        x0, _, x1, _ = w.box
        # erase a little wider to catch the whole word, but not into the neighbours
        left = words[i - 1].box[2] if i > 0 else lx0
        right = words[i + 1].box[0] if i + 1 < len(words) else lx1
        ex0 = max((x0 + left) / 2 if i > 0 else lx0, x0 - 0.15 * h)
        ex1 = min((x1 + right) / 2 if i + 1 < len(words) else lx1, x1 + 0.15 * h)
        out.append(dict(erase=(ex0, ly0, ex1, ly1), word=(x0, ly0, x1, ly1), old=w.text, new=t,
                        rtl=line.rtl, samples=samples))
    return out


_ID_MARKERS = ("بطاقة تحقيق الشخصية", "تحقيق الشخصية", "الرقم القومي", "جواز سفر", "PASSPORT")


def _is_identity_document(lines: list[Line]) -> bool:
    """ID cards and passports: corrections go into the text layer only, the
    scan itself is never altered."""
    import re
    text = "\n".join(l.text for l in lines)
    if any(m in text for m in _ID_MARKERS):
        return True
    digits = re.sub(r"[\s]", "", text.translate(str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")))
    return re.search(r"(?<!\d)[23]\d{13}(?!\d)", digits) is not None


def _draw_turned(c, s: str, quad, sx: float, sy: float, page_h_pt: float):
    """Invisible text along a slanted or sideways word (a card turned on the
    scanned sheet): the text runs from its bottom-left corner toward the
    bottom-right one, so selection and search follow the printed word."""
    import math
    tl, tr, br, bl = [(float(x) * sx, page_h_pt - float(y) * sy) for x, y in quad]
    w = math.hypot(br[0] - bl[0], br[1] - bl[1])
    h = math.hypot(tl[0] - bl[0], tl[1] - bl[1])
    if w < 1 or h < 1:
        return
    size = max(1.0, h * 0.75)
    natural = pdfmetrics.stringWidth(s, _FONT, size)
    c.saveState()
    c.translate(bl[0], bl[1])
    c.rotate(math.degrees(math.atan2(br[1] - bl[1], br[0] - bl[0])))
    t = c.beginText()
    t.setTextRenderMode(3)
    t.setFont(_FONT, size)
    t.setHorizScale(100.0 * w / natural if natural > 0 else 100.0)
    t.setTextOrigin(0, h * 0.2)
    t.textOut(s + " ")
    c.drawText(t)
    c.restoreState()
