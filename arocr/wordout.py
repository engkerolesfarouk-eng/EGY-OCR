"""Write an editable Word file that looks like the scan.

Like the searchable PDF, every page keeps the scanned image as its background
(stamps, photos, logos, colours, lines). The difference: the original text is
erased from that background and replaced by real, visible, editable text placed
exactly where it was (one positioned frame per line), in the ink colour and at
the size it had on the scan.
"""
from __future__ import annotations

import io

import cv2
import numpy as np
from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK, WD_LINE_SPACING
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import qn
from docx.shared import Pt, RGBColor

FONT = "Arial"          # has Arabic glyphs on every Windows machine
MARGIN_PT = 18.0
TEXT_TO_BOX = 0.55      # detected boxes are padded: text is ~55% of box height
TWIP = 20               # 1 pt = 20 twips
EMU = 12700             # 1 pt = 12700 EMU

_MEASURE = None


def _measure_font():
    """A TrueType font to measure text widths with (same family as the output)."""
    global _MEASURE
    if _MEASURE is None:
        import os
        from PIL import ImageFont
        for path in (r"C:\Windows\Fonts\arial.ttf", "/usr/share/fonts/truetype/msttcorefonts/Arial.ttf",
                     "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
            if os.path.exists(path):
                _MEASURE = ImageFont.truetype(path, 100)
                break
    return _MEASURE


def _text_width_100(text: str, rtl: bool) -> float:
    f = _measure_font()
    if f is None or not text.strip():
        return 0.0
    from PIL import Image, ImageDraw
    d = ImageDraw.Draw(Image.new("L", (1, 1)))
    try:
        return d.textlength(text, font=f, direction="rtl" if rtl else None)
    except Exception:  # noqa: BLE001  (no raqm: measure without shaping)
        return d.textlength(text, font=f)


def _fit_size(text: str, box_w: float, box_h: float, rtl: bool) -> float:
    """Font size (pt) that makes the line as wide as it was on the scan."""
    by_height = box_h * TEXT_TO_BOX
    w100 = _text_width_100(text, rtl)
    if w100 <= 0:
        return max(5.0, min(48.0, by_height))
    by_width = box_w * 100.0 / w100 * 0.97
    return max(5.0, min(48.0, by_height * 1.6, max(by_height * 0.6, by_width)))


# ---------------------------------------------------------------- background

def _paper(img: np.ndarray, k: int) -> np.ndarray:
    """The page without its text: dark strokes thinner than k are closed over
    by the surrounding (lighter) paper, then smoothed."""
    k = max(5, k | 1)
    small = 2 if max(img.shape[:2]) > 1400 else 1          # speed: work at half size
    im = cv2.resize(img, None, fx=1 / small, fy=1 / small, interpolation=cv2.INTER_AREA) if small > 1 else img
    ks = max(3, (k // small) | 1)
    bg = cv2.dilate(im, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks, ks)))
    bg = cv2.blur(bg, (ks, ks))
    if small > 1:
        bg = cv2.resize(bg, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)
    return bg


def _erase_text(img: np.ndarray, lines) -> tuple[np.ndarray, list[tuple[int, int, int]]]:
    """Remove the original text from the page image, leaving paper, stamps,
    photos and lines. Returns the cleaned image and each line's ink colour (RGB)."""
    h, w = img.shape[:2]
    heights = sorted(r.height for l in lines for r in l.regions) or [20.0]
    th = heights[len(heights) // 2]
    bg = _paper(img, int(th * 0.9))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.int16)
    bg_gray = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY).astype(np.int16)
    ink_map = (bg_gray - gray) > 35                       # clearly darker than its paper

    box = np.zeros((h, w), np.uint8)    # the recognised boxes, padded
    band = np.zeros((h, w), np.uint8)   # wider area: dots/descenders that stick out
    inks = []
    for line in lines:
        px = []
        for r in line.regions:
            rh = r.height
            for m, px_pad, py_pad in ((box, 0.15, 0.12), (band, 0.5, 0.45)):
                x0 = max(0, int(r.x0 - rh * px_pad)); x1 = min(w, int(np.ceil(r.x1 + rh * px_pad)))
                y0 = max(0, int(r.y0 - rh * py_pad)); y1 = min(h, int(np.ceil(r.y1 + rh * py_pad)))
                m[y0:y1, x0:x1] = 1
            x0, x1 = max(0, int(r.x0)), min(w, int(np.ceil(r.x1)))
            y0, y1 = max(0, int(r.y0)), min(h, int(np.ceil(r.y1)))
            sel = ink_map[y0:y1, x0:x1]
            if sel.any():
                px.append(img[y0:y1, x0:x1][sel])
        if px:
            p = np.concatenate(px)
            lum = p.astype(np.float32).sum(1)
            b, g_, r_ = np.median(p[lum <= np.median(lum)], axis=0)
            inks.append((int(r_), int(g_), int(b)))
        else:
            inks.append((0, 0, 0))

    # Erase only the ink itself, not whole boxes: every other pixel (paper
    # texture, lighting, patterns) stays exactly as scanned, so nothing shows
    # behind the new text.
    area = box.astype(bool) | band.astype(bool)
    ink = (area & ink_map).astype(np.uint8)
    ink = cv2.dilate(ink, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))   # anti-aliased edges
    keep = (1 - ink).astype(np.float32)
    k = max(7, int(th) | 1)
    num = cv2.blur(img.astype(np.float32) * keep[..., None], (k, k))
    den = cv2.blur(keep, (k, k))[..., None]
    fill = np.where(den > 0.05, num / np.maximum(den, 1e-6), bg.astype(np.float32))
    # pixels deep inside a thick stroke: grow the estimate a second time
    a = cv2.GaussianBlur(ink.astype(np.float32), (0, 0), 1.0)
    a = np.maximum(a, ink.astype(np.float32))[..., None]
    out = np.clip(a * fill + (1 - a) * img, 0, 255).astype(np.uint8)
    return out, inks


# ---------------------------------------------------------------- docx helpers

def _rtl(paragraph, run):
    pPr = paragraph._p.get_or_add_pPr()
    bidi = OxmlElement("w:bidi")
    later = [c for c in pPr if c.tag in {qn(t) for t in (
        "w:adjustRightInd", "w:snapToGrid", "w:spacing", "w:ind", "w:contextualSpacing",
        "w:mirrorIndents", "w:suppressOverlap", "w:jc", "w:textDirection", "w:textAlignment",
        "w:outlineLvl", "w:rPr", "w:sectPr", "w:pPrChange")}]
    if later:
        later[0].addprevious(bidi)
    else:
        pPr.append(bidi)
    run._r.get_or_add_rPr().append(OxmlElement("w:rtl"))


def _paragraphs(page, sy):
    """Text of the page with a blank line between paragraphs, from the layout:
    a new block (bigger gap, new column), an indented first line, or the line
    after a short one (the end of a paragraph). Returns it and each paragraph's
    tallest line height (pt) for headings."""
    lines, starts = page.lines, page.block_starts()
    if not lines:
        return "", []
    hs = sorted(max(r.y1 for r in l.regions) - min(r.y0 for r in l.regions) for l in lines)
    h = hs[len(hs) // 2]
    # block extents
    blocks, cur = [], []
    for ln, st in zip(lines, starts):
        if st and cur:
            blocks.append(cur); cur = []
        cur.append(ln)
    if cur:
        blocks.append(cur)
    out, heights = [], []
    for blk in blocks:
        x0 = min(min(r.x0 for r in l.regions) for l in blk)
        x1 = max(max(r.x1 for r in l.regions) for l in blk)
        width = x1 - x0
        para, ph = [], 0.0
        prev = None
        for ln in blk:
            lx0 = min(r.x0 for r in ln.regions); lx1 = max(r.x1 for r in ln.regions)
            lh = max(r.y1 for r in ln.regions) - min(r.y0 for r in ln.regions)
            new = False
            if prev is not None and width > 8 * h:
                pl = prev
                p0 = min(r.x0 for r in pl.regions); p1 = max(r.x1 for r in pl.regions)
                short_prev = (p1 - p0) < 0.8 * width
                indent = (x1 - lx1 > 1.2 * h) if ln.rtl else (lx0 - x0 > 1.2 * h)
                new = short_prev or indent
            if new and para:
                out.append("\n".join(para)); heights.append(ph * sy)
                para, ph = [], 0.0
            para.append(ln.text)
            ph = max(ph, lh)
            prev = ln
        if para:
            out.append("\n".join(para)); heights.append(ph * sy)
    return "\n\n".join(out), heights


def _bold(run):
    run.bold = True
    run._r.get_or_add_rPr().append(OxmlElement("w:bCs"))     # Arabic bold is the complex-script flag
    # keep the schema order: bCs right after b
    rPr = run._r.rPr
    b, bcs = rPr.find(qn("w:b")), rPr.findall(qn("w:bCs"))[-1]
    if b is not None:
        b.addnext(bcs)


def _has_arabic(t: str) -> bool:
    return any("\u0600" <= c <= "\u06ff" for c in t)


def _rtl_paragraph(paragraph):
    """Right-to-left paragraph with every run marked RTL (Arabic text)."""
    pPr = paragraph._p.get_or_add_pPr()
    bidi = OxmlElement("w:bidi")
    later = [c for c in pPr if c.tag in {qn(t) for t in (
        "w:adjustRightInd", "w:snapToGrid", "w:spacing", "w:ind", "w:contextualSpacing",
        "w:mirrorIndents", "w:suppressOverlap", "w:jc", "w:textDirection", "w:textAlignment",
        "w:outlineLvl", "w:rPr", "w:sectPr", "w:pPrChange")}]
    if later:
        later[0].addprevious(bidi)
    else:
        pPr.append(bidi)
    for r in paragraph.runs:
        r._r.get_or_add_rPr().append(OxmlElement("w:rtl"))


def _font(run, size_pt: float, color=None):
    run.font.name = FONT
    run.font.size = Pt(size_pt)
    if color is not None:
        run.font.color.rgb = RGBColor(*color)
    rPr = run._r.get_or_add_rPr()
    fonts = rPr.find(qn("w:rFonts"))
    if fonts is None:
        fonts = OxmlElement("w:rFonts")
        rPr.insert(0, fonts)
    for attr in ("w:ascii", "w:hAnsi", "w:cs"):
        fonts.set(qn(attr), FONT)
    szcs = OxmlElement("w:szCs")                       # Arabic uses the complex-script size
    szcs.set(qn("w:val"), str(int(round(size_pt * 2))))
    rPr.append(szcs)


def _transparent(run):
    """Text that is there (select, copy, search, edit) but not seen."""
    W14 = "http://schemas.microsoft.com/office/word/2010/wordml"
    rPr = run._r.get_or_add_rPr()
    fill = parse_xml(f'<w14:textFill xmlns:w14="{W14}"><w14:solidFill>'
                     f'<w14:srgbClr w14:val="000000"><w14:alpha w14:val="100000"/></w14:srgbClr>'
                     f'</w14:solidFill></w14:textFill>')
    rPr.append(fill)


def _frame(paragraph, x_pt, y_pt, w_pt, h_pt):
    """Pin a paragraph to an exact spot on the page (a Word text frame)."""
    pPr = paragraph._p.get_or_add_pPr()
    fp = OxmlElement("w:framePr")
    fp.set(qn("w:w"), str(int(round(w_pt * TWIP))))
    fp.set(qn("w:h"), str(int(round(h_pt * TWIP))))
    fp.set(qn("w:hRule"), "exact")
    fp.set(qn("w:wrap"), "around")
    fp.set(qn("w:hAnchor"), "page")
    fp.set(qn("w:vAnchor"), "page")
    fp.set(qn("w:x"), str(int(round(x_pt * TWIP))))
    fp.set(qn("w:y"), str(int(round(y_pt * TWIP))))
    pPr.insert(0, fp)
    pf = paragraph.paragraph_format
    pf.space_before = Pt(0)
    pf.space_after = Pt(0)
    pf.line_spacing = Pt(max(5.0, h_pt))
    pf.line_spacing_rule = WD_LINE_SPACING.EXACTLY


def _background(paragraph, run, image_part_rid: str, w_pt: float, h_pt: float, pid: int):
    """Turn an inline picture into a full-page picture behind the text."""
    cx, cy = int(w_pt * EMU), int(h_pt * EMU)
    xml = f'''
<wp:anchor xmlns:wp="http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing"
  xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"
  xmlns:pic="http://schemas.openxmlformats.org/drawingml/2006/picture"
  xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"
  distT="0" distB="0" distL="0" distR="0" simplePos="0" relativeHeight="0"
  behindDoc="1" locked="1" layoutInCell="1" allowOverlap="1">
  <wp:simplePos x="0" y="0"/>
  <wp:positionH relativeFrom="page"><wp:posOffset>0</wp:posOffset></wp:positionH>
  <wp:positionV relativeFrom="page"><wp:posOffset>0</wp:posOffset></wp:positionV>
  <wp:extent cx="{cx}" cy="{cy}"/>
  <wp:effectExtent l="0" t="0" r="0" b="0"/>
  <wp:wrapNone/>
  <wp:docPr id="{pid}" name="Page {pid}"/>
  <wp:cNvGraphicFramePr/>
  <a:graphic><a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/picture">
    <pic:pic>
      <pic:nvPicPr><pic:cNvPr id="{pid}" name="page{pid}.jpg"/><pic:cNvPicPr/></pic:nvPicPr>
      <pic:blipFill><a:blip r:embed="{image_part_rid}"/><a:stretch><a:fillRect/></a:stretch></pic:blipFill>
      <pic:spPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="{cx}" cy="{cy}"/></a:xfrm>
        <a:prstGeom prst="rect"><a:avLst/></a:prstGeom></pic:spPr>
    </pic:pic>
  </a:graphicData></a:graphic>
</wp:anchor>'''
    drawing = OxmlElement("w:drawing")
    drawing.append(parse_xml(xml))
    run._r.append(drawing)


class WordDocument:
    def __init__(self, path: str, background: bool = True, jpeg_quality: int = 85,
                 erase: bool = False, layout: str = "text"):
        """layout="text" (default): an ordinary Word document - paragraphs of
        real text, right to left, that you edit, search and reflow like any
        document (no scan behind it; small and fast in Word).
        layout="scan": every page looks exactly like the scan (the image as
        background, each word placed over it); good for keeping the look,
        awkward to edit and heavy for Word on long books.
        erase=True (scan layout): the text is removed from the scan and written
        back visibly, so edits show (works on plain paper only)."""
        self.layout = layout
        self.path = path
        self.background = background
        self.erase = erase
        self.jpeg_quality = jpeg_quality
        self.doc = Document()
        self.first = True
        self.pid = 0
        # python-docx's template leaves <w:zoom> without the required percent
        zoom = self.doc.settings.element.find(qn("w:zoom"))
        if zoom is not None and zoom.get(qn("w:percent")) is None:
            zoom.set(qn("w:percent"), "100")
        st = self.doc.styles["Normal"]
        st.font.name = FONT
        st.paragraph_format.space_before = Pt(0)
        st.paragraph_format.space_after = Pt(0)

    def _new_page(self, w_pt: float, h_pt: float):
        sec = self.doc.sections[0] if self.first else self.doc.add_section(WD_SECTION.NEW_PAGE)
        self.first = False
        sec.page_width, sec.page_height = Pt(w_pt), Pt(h_pt)
        for side in ("left_margin", "right_margin", "top_margin", "bottom_margin"):
            setattr(sec, side, Pt(MARGIN_PT))
        sec.header_distance = sec.footer_distance = Pt(0)
        return sec

    def _add_background(self, img: np.ndarray, w_pt: float, h_pt: float):
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
        p = self.doc.add_paragraph()
        p.paragraph_format.line_spacing = Pt(1)
        p.paragraph_format.line_spacing_rule = WD_LINE_SPACING.EXACTLY
        run = p.add_run()
        rid, _ = run.part.get_or_add_image(io.BytesIO(buf.tobytes()))
        self.pid += 1
        _background(p, run, rid, w_pt, h_pt, self.pid)

    def add_page(self, page, edited_text: str | None = None):
        if self.layout == "text":
            return self._text_page(page, edited_text)
        return self._scan_page(page, edited_text)

    def _text_page(self, page, edited_text: str | None = None):
        """The page as a normal document: one paragraph per block of text.

        Lines of running text (a book paragraph wrapped over several lines)
        are joined into one paragraph so Word re-wraps them; short lines
        (headings, a list, an ID card's fields) keep their line breaks."""
        img_h = page.image.shape[0]
        sy = page.height_pt / img_h
        hs = sorted((max(r.y1 for r in l.regions) - min(r.y0 for r in l.regions)) * sy for l in page.lines) or [20.0]
        body = max(9.0, min(16.0, hs[len(hs) // 2] * TEXT_TO_BOX))
        # heading sizes come from the reading when the text wasn't edited
        block_h = []
        if edited_text is None or edited_text == page.text:
            cur = None
            for ln, new in zip(page.lines, page.block_starts()):
                h = (max(r.y1 for r in ln.regions) - min(r.y0 for r in ln.regions)) * sy
                if new or cur is None:
                    block_h.append(h)
                    cur = True
                else:
                    block_h[-1] = max(block_h[-1], h)
        text = page.text if edited_text is None else edited_text
        if getattr(page, "fields_only", False):
            text, block_h = page.text if edited_text is None else edited_text, []
        elif edited_text is None or edited_text == page.text:
            text, block_h = _paragraphs(page, sy)
        blocks = [b for b in text.split("\n\n") if b.strip()]
        if self.first:
            sec = self.doc.sections[0]
            sec.page_width, sec.page_height = Pt(page.width_pt), Pt(page.height_pt)
            m = Pt(max(36.0, min(72.0, page.width_pt * 0.08)))
            sec.left_margin = sec.right_margin = sec.top_margin = sec.bottom_margin = m
        else:
            self.doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)
        self.first = False
        prose_ok = not getattr(page, "cards", None)
        for i, block in enumerate(blocks):
            lines = [l.strip() for l in block.split("\n") if l.strip()]
            lens = sorted(len(l) for l in lines)
            prose = prose_ok and len(lines) >= 2 and lens[len(lens) // 2] >= 35
            heading = (i < len(block_h) and len(lines) <= 2 and block_h[i] > 1.35 * hs[len(hs) // 2])
            size = min(26.0, body * block_h[i] / hs[len(hs) // 2]) if heading else body
            rtl = any(_has_arabic(l) for l in lines)
            p = self.doc.add_paragraph()
            pf = p.paragraph_format
            pf.space_after = Pt(body * 0.6)
            if prose:
                r = p.add_run(" ".join(lines))
                _font(r, size)
                if heading:
                    _bold(r)
            else:
                for k, l in enumerate(lines):
                    r = p.add_run(l)
                    _font(r, size)
                    if heading:
                        _bold(r)
                    if k < len(lines) - 1:
                        r.add_break(WD_BREAK.LINE)
            if rtl:
                _rtl_paragraph(p)
                p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY if prose else (
                    WD_ALIGN_PARAGRAPH.CENTER if heading else WD_ALIGN_PARAGRAPH.LEFT)
            else:
                p.alignment = WD_ALIGN_PARAGRAPH.LEFT

    def _scan_page(self, page, edited_text: str | None = None):
        """page: PageResult. edited_text: the user's corrected text, if any.

        The scan is the page background, untouched. Every recognised word is
        placed in its own frame exactly over the pixels it was read from, sized
        and spaced to cover the same width. By default the text is transparent
        (the page looks exactly like the scan, like the searchable PDF) but it
        is real Word text: select, copy, search, edit, or give it a colour.
        """
        img_h, img_w = page.image.shape[:2]
        sx, sy = page.width_pt / img_w, page.height_pt / img_h
        self._new_page(page.width_pt, page.height_pt)

        texts = [l.text for l in page.lines]
        if edited_text is not None and edited_text != page.text:
            edited = [t for t in edited_text.split("\n") if t.strip()]
            if len(edited) == len(texts):
                texts = edited               # same lines, corrected words: keep positions
            else:
                self._flow_page(page, edited_text, sy)
                return

        if self.erase:
            bgimg, inks = _erase_text(page.image, page.lines)
        else:
            bgimg, inks = page.image, [None] * len(page.lines)
        if self.background:
            self._add_background(bgimg, page.width_pt, page.height_pt)

        for line, text, ink in zip(page.lines, texts, inks):
            words = sorted((w for w in line.words if w.text.strip()), key=lambda w: w.box[0])
            parts = text.split()
            if words and text == line.text and " ".join(w.text for w in words) == line.text:
                items = [(w.text, w.box) for w in words]            # as read: exact boxes
            elif words and len(parts) == len(words):
                # corrected words: boxes are left to right, Arabic reads right
                # to left, so its first word goes in the rightmost box
                order = parts[::-1] if line.rtl else parts
                items = [(t, w.box) for t, w in zip(order, words)]
            else:   # no word boxes, or the user changed the word count: whole line
                items = [(text, (min(r.x0 for r in line.regions), min(r.y0 for r in line.regions),
                                 max(r.x1 for r in line.regions), max(r.y1 for r in line.regions)))]
            # CTC places a letter at a single column: give short words their
            # real width (about 0.45 of the line height per letter)
            lh_px = max(r.y1 for r in line.regions) - min(r.y0 for r in line.regions)
            fixed = []
            for t, (x0, y0, x1, y1) in items:
                n = len(t.replace(" ", ""))
                need = n * lh_px * 0.3
                if n <= 3 and x1 - x0 < need:
                    c = (x0 + x1) / 2
                    x0, x1 = c - need / 2, c + need / 2
                fixed.append((t, (x0, y0, x1, y1)))
            items = fixed
            line_y0 = min(r.y0 for r in line.regions) * sy
            line_h = max(r.y1 for r in line.regions) * sy - line_y0
            # one font size for the whole line (as printed), fitted to its width
            lx0 = min(r.x0 for r in line.regions) * sx
            lx1 = max(r.x1 for r in line.regions) * sx
            size = _fit_size(text, lx1 - lx0, line_h, line.rtl)
            for t, (x0, y0, x1, y1) in items:
                self._place(t, x0 * sx, line_y0, x1 * sx, line_h, line.rtl, ink, page.width_pt, size)

    def _place(self, text, x0, y0, x1, box_h, rtl, ink, page_w, size):
        box_w = max(1.0, x1 - x0)
        fw = min(page_w, box_w * 1.3 + 6)          # slack: never wrap
        fx = x1 - fw if rtl else x0
        fx = max(0.0, min(page_w - fw, fx))
        p = self.doc.add_paragraph()
        lh = size * 1.25                       # Arial line box for this size
        _frame(p, fx, y0 + (box_h - lh) / 2, fw, lh)
        r = p.add_run(text)
        _font(r, size, ink)
        # stretch/squeeze the letters (character scale, keeps Arabic joined)
        # so the word covers exactly its pixels
        w100 = _text_width_100(text, rtl)
        if w100 > 0:
            scale = int(round(100 * box_w / (w100 * size / 100.0)))
            scale = max(50, min(200, scale))
            if abs(scale - 100) >= 3:
                el = OxmlElement("w:w")
                el.set(qn("w:val"), str(scale))
                rPr = r._r.get_or_add_rPr()
                after = rPr.find(qn("w:sz"))
                (after.addprevious(el) if after is not None else rPr.append(el))
        if rtl:
            _rtl(p, r)
        p.alignment = WD_ALIGN_PARAGRAPH.LEFT        # = start side (right) in RTL
        if not self.erase:
            _transparent(r)

    def _flow_page(self, page, edited_text: str, sy: float):
        """Lines were added or removed by the user: plain right-aligned text."""
        heights = sorted((max(r.y1 for r in l.regions) - min(r.y0 for r in l.regions)) * sy
                         for l in page.lines) or [20.0]
        size = max(6.0, heights[len(heights) // 2] * TEXT_TO_BOX)
        for text in edited_text.split("\n"):
            p = self.doc.add_paragraph()
            r = p.add_run(text)
            _font(r, size)
            _rtl(p, r)
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT

    def save(self):
        self.doc.save(self.path)
