"""Group recognized regions into lines and words in reading order."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .recognize import RecResult
from .textfix import has_arabic, is_digits_only, visual_to_logical


@dataclass
class Word:
    text: str                                 # logical order
    box: tuple[float, float, float, float]    # x0, y0, x1, y1 in image pixels
    confidence: float
    visual: str = ""                          # same word in left-to-right pixel order
    quad: object = None                       # 4x2 tl,tr,br,bl when the text is not upright (a turned card)


@dataclass
class Region:
    quad: np.ndarray
    rec: RecResult

    @property
    def x0(self): return float(self.quad[:, 0].min())
    @property
    def x1(self): return float(self.quad[:, 0].max())
    @property
    def y0(self): return float(self.quad[:, 1].min())
    @property
    def y1(self): return float(self.quad[:, 1].max())
    @property
    def height(self): return self.y1 - self.y0
    @property
    def cy(self): return (self.y0 + self.y1) / 2


@dataclass
class Line:
    regions: list[Region]
    rtl: bool
    text: str = ""
    words: list[Word] = field(default_factory=list)

    @property
    def confidence(self) -> float:
        return float(np.mean([r.rec.confidence for r in self.regions])) if self.regions else 0.0


def build_lines(regions: list[Region], min_conf: float = 0.3) -> list[Line]:
    # very short readings need more confidence: on cards they are usually a
    # stamp or the eagle read as "II" or "ة"
    regions = [r for r in regions if r.rec.text.strip() and r.rec.confidence >=
               (0.6 if len(r.rec.text.strip()) <= 2 else 0.5 if len(r.rec.text.strip()) <= 3 else min_conf)]
    if not regions:
        return []
    rtl_page = sum(has_arabic(r.rec.text) for r in regions) >= len(regions) / 2
    groups: list[list[Region]] = []
    # Two columns (or a scanned two-page spread): read each column top to bottom,
    # right column first for Arabic, instead of gluing across the gutter.
    for block in _columns(regions, rtl_page):
        groups.extend(_group_rows(block))

    lines = []
    for g in groups:
        rtl = any(has_arabic(r.rec.text) for r in g)
        # A row of digit groups is one number written left to right.
        if all(is_digits_only(visual_to_logical(r.rec.text.strip())) for r in g):
            rtl = False
        g.sort(key=lambda r: -r.x1 if rtl else r.x0)
        line = Line(g, rtl)
        parts: list[str] = []
        prev: Region | None = None
        for r in g:
            logical = visual_to_logical(r.rec.text.strip())
            # The detector sometimes splits one long number (an ID number) into two
            # boxes; glue them back when they are digits only and nearly touching.
            if prev is not None and parts and is_digits_only(logical) and is_digits_only(parts[-1]):
                gap = (prev.x0 - r.x1) if rtl else (r.x0 - prev.x1)
                if gap < 0.6 * max(r.height, prev.height):
                    # Numbers read left to right, so the left box is the start of
                    # the number. In an RTL line we visit the right box first.
                    parts[-1] = logical + parts[-1] if rtl else parts[-1] + logical
                    prev = r
                    line.words.extend(_words(r))
                    continue
            parts.append(logical)
            line.words.extend(_words(r))
            prev = r
        line.text = " ".join(parts)
        lines.append(line)
    return lines


def _group_rows(regions: list[Region]) -> list[list[Region]]:
    """Group boxes that sit on the same text line.

    Boxes only join when they overlap vertically AND are horizontally close, so
    two unrelated texts at the same height (a caption beside a paragraph, a
    split image) stay separate lines.
    """
    regions = sorted(regions, key=lambda r: r.cy)
    groups: list[list[Region]] = []
    for r in regions:
        placed = False
        for g in groups[-4:]:
            gy0 = min(x.y0 for x in g); gy1 = max(x.y1 for x in g)
            overlap = min(gy1, r.y1) - max(gy0, r.y0)
            if overlap <= 0.5 * min(r.height, gy1 - gy0):
                continue
            gap = min(max(x.x0 - r.x1, r.x0 - x.x1) for x in g)
            hmax = max(r.height, max(x.height for x in g))
            # two numbers far apart (a date and an ID number) are separate fields
            if _numeric(r) and all(_numeric(x) for x in g) and gap > 1.0 * hmax:
                continue
            if gap <= 3.0 * hmax:
                g.append(r); placed = True; break
        if not placed:
            groups.append([r])
    return groups


def _numeric(r: Region) -> bool:
    t = r.rec.text
    return any(c.isdigit() for c in t) and not has_arabic(t) and not any(c.isalpha() for c in t)


def _split_region(r: Region, g: float):
    """A detector box that runs across the gutter (two columns' lines read as
    one): cut it at the gutter into a left and a right region, using the
    reader's character positions."""
    rec = r.rec
    if not rec.char_x:
        return None
    w = r.x1 - r.x0
    xs = [r.x0 + c * w for c in rec.char_x]
    left = [i for i, x in enumerate(xs) if x < g]
    right = [i for i, x in enumerate(xs) if x >= g]
    out = []
    for idx, (a, b) in ((left, (r.x0, g)), (right, (g, r.x1))):
        text = "".join(rec.text[i] for i in idx).strip()
        if not text:
            continue
        span = max(1e-6, b - a)
        sub = RecResult(text, float(np.mean([rec.char_conf[i] for i in idx])) if rec.char_conf else rec.confidence,
                        [min(1.0, max(0.0, (xs[i] - a) / span)) for i in idx if not rec.text[i].isspace()][:len(text)],
                        [rec.char_conf[i] for i in idx if not rec.text[i].isspace()][:len(text)] if rec.char_conf else [])
        # keep char_x aligned with the stripped text
        sub.char_x = [min(1.0, max(0.0, (xs[i] - a) / span)) for i in idx][:len(text)]
        quad = np.array([[a, r.y0], [b, r.y0], [b, r.y1], [a, r.y1]], np.float32)
        out.append(Region(quad, sub))
    return out


def _columns(regions: list[Region], rtl: bool = True) -> list[list[Region]]:
    """Reading blocks of a page: columns split at vertical gutters, in reading
    order (right column first for Arabic).

    A gutter is a vertical strip that (almost) no text line crosses. Short
    items (a page number, a small heading) don't count when looking for it. A
    box the detector stretched across the gutter (two lines read as one) is cut
    in two; a heading or footnote that really spans the page splits the page
    into bands read one after the other, each with its own columns.
    """
    x0 = min(r.x0 for r in regions); x1 = max(r.x1 for r in regions)
    width = int(x1 - x0) + 1
    if width < 50 or len(regions) < 6:
        return [regions]
    h = float(np.median([r.height for r in regions]))
    wide = lambda r: (r.x1 - r.x0) > 0.6 * width          # noqa: E731
    counted = [r for r in regions if not wide(r) and (r.x1 - r.x0) > 2.5 * h]
    # Column lines come in pairs: each has a line of the other column beside
    # it. Centred headings above the columns don't, and would otherwise hide
    # the gutter, so look for it among paired lines when there are enough.
    def paired(r):
        return any(o is not r and min(o.y1, r.y1) - max(o.y0, r.y0) > 0.5 * min(o.height, r.height)
                   and (o.x0 > r.x1 or o.x1 < r.x0) for o in counted)
    pairs = [r for r in counted if paired(r)]
    if len(pairs) >= 10:
        counted = pairs
    cover = np.zeros(width, np.int32)
    for r in counted:
        cover[int(r.x0 - x0):int(r.x1 - x0) + 1] += 1
    allowed = max(1, int(0.08 * len(counted)))         # a few boxes may cross it
    min_gutter = max(0.9 * h, 0.02 * width)
    gutters, start = [], None
    for i, c in enumerate(list(cover) + [10 ** 6]):
        if c <= allowed and start is None:
            start = i
        elif c > allowed and start is not None:
            if i - start >= min_gutter and start > 0 and i < width:
                lo = start + int(np.argmin(cover[start:i]))     # the emptiest point of the strip
                zero = [j for j in range(start, i) if cover[j] == cover[lo]]
                gutters.append((x0 + (zero[0] + zero[-1]) / 2, (i - start)))
            start = None

    def side_counts(g):
        left = sum(1 for r in counted if r.x1 < g)
        right = sum(1 for r in counted if r.x0 > g)
        return left, right
    gutters = [(g, gw) for g, gw in gutters if min(side_counts(g)) >= 5]
    if not gutters:
        return [regions]

    # boxes crossing a gutter: two lines glued (cut them) or a spanning item
    spanning = set()
    fixed = []
    for r in regions:
        # crossing = reaching well into the columns on both sides (a box that only
        # pokes into the gutter belongs to the column it mostly sits in)
        cross = [(g, gw) for g, gw in gutters if min(g - r.x0, r.x1 - g) > 0.12 * width]
        if not cross or wide(r):
            fixed.append(r)
            continue
        g, gw = cross[0]
        xs = [r.x0 + c * (r.x1 - r.x0) for c in r.rec.char_x]
        empty = not any(abs(x - g) < 0.35 * gw for x, ch in zip(xs, r.rec.text) if not ch.isspace())
        parts = _split_region(r, g) if empty else None
        if parts:
            fixed.extend(parts)
        else:
            fixed.append(r)
            spanning.add(id(r))
    regions = fixed
    gs = [g for g, _ in gutters]

    def split(band):
        cols = [[] for _ in range(len(gs) + 1)]
        for r in band:
            cols[sum((r.x0 + r.x1) / 2 > g for g in gs)].append(r)
        cols = [c for c in cols if c]
        return cols[::-1] if rtl else cols

    out, band = [], []
    for r in sorted(regions, key=lambda r: r.cy):
        if wide(r) or id(r) in spanning:
            if band:
                out.extend(split(band)); band = []
            if out and len(out[-1]) and all(wide(x) or id(x) in spanning for x in out[-1]) \
                    and abs(out[-1][-1].cy - r.cy) < 0.6 * h:
                out[-1].append(r)          # pieces of the same spanning line
            else:
                out.append([r])
        else:
            band.append(r)
    if band:
        out.extend(split(band))
    return out


def _words(r: Region) -> list[Word]:
    """Split a region into words with approximate boxes from CTC character positions."""
    text, xs = r.rec.text, r.rec.char_x
    tl, tr, br, bl = r.quad
    words = []
    start = None
    for i, ch in enumerate(text + " "):
        if ch != " " and start is None:
            start = i
        elif ch == " " and start is not None:
            seg = text[start:i]
            fx0 = xs[start] if start < len(xs) else 0.0
            fx1 = xs[i - 1] if i - 1 < len(xs) else 1.0
            # widen by half a character on each side
            half = 0.5 * (fx1 - fx0) / max(1, len(seg) - 1) if len(seg) > 1 else 0.02
            fx0, fx1 = max(0.0, fx0 - half), min(1.0, fx1 + half)
            p0 = tl + (tr - tl) * fx0; p1 = tl + (tr - tl) * fx1
            q0 = bl + (br - bl) * fx0; q1 = bl + (br - bl) * fx1
            pts = np.array([p0, p1, q0, q1])
            words.append(Word(visual_to_logical(seg), (float(pts[:, 0].min()), float(pts[:, 1].min()),
                                                       float(pts[:, 0].max()), float(pts[:, 1].max())),
                              r.rec.confidence, seg))
            start = None
    return words


def _edit(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]
