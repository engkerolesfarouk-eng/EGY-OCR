"""End-to-end: file in, text + searchable PDF out."""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from .detect import DetBox, Detector
from .layout import Line, Region, build_lines
from .preprocess import clahe
from .recognize import Recognizer, crop_quad
from .fuse import fill_from_latin
from .textfix import NUMBER_LINE, count_digits, count_letters, fix_codes, normalize_egyptian

MODELS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
IMAGE_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
SUPPORTED_EXT = set(IMAGE_EXT) | {".pdf"}


@dataclass
class PageResult:
    image: np.ndarray
    lines: list[Line]
    width_pt: float
    height_pt: float
    seconds: float
    cards: list = field(default_factory=list)       # cards.Card, when the page holds ID cards
    breaks: list | None = None                       # block starts decided by the reader
    fields_only: bool = False                        # ID-card mode: the text is the cards' fields

    def block_starts(self) -> list[bool]:
        """For each line: does it start a new block (paragraph, column, field)?

        A new block starts where the vertical gap is clearly bigger than the
        usual line spacing, where reading jumps back up (a new column), or where
        two lines sit side by side instead of stacked.
        """
        if not self.lines:
            return []
        if self.breaks is not None and len(self.breaks) == len(self.lines):
            return list(self.breaks)
        tops = [min(r.y0 for r in l.regions) for l in self.lines]
        bots = [max(r.y1 for r in l.regions) for l in self.lines]
        mids = [(t + b) / 2 for t, b in zip(tops, bots)]
        heights = sorted(b - t for t, b in zip(tops, bots))
        h = heights[len(heights) // 2]
        # line pitch (centre to centre) inside a block
        pitches = sorted(m2 - m1 for m1, m2 in zip(mids, mids[1:]) if 0 < m2 - m1 < 2.5 * h)
        pitch = pitches[len(pitches) // 4] if pitches else 1.2 * h
        starts = [True]
        for i in range(1, len(self.lines)):
            step = mids[i] - mids[i - 1]
            a, b = self.lines[i - 1].regions, self.lines[i].regions
            apart = (max(r.x1 for r in b) < min(r.x0 for r in a)
                     or min(r.x0 for r in b) > max(r.x1 for r in a))   # side by side, not stacked
            starts.append(apart or step < -0.5 * h or step > 1.35 * pitch)
        return starts

    @property
    def text(self) -> str:
        """Lines in reading order, with a blank line between blocks. In ID-card
        mode: name, address, national ID and birth date of each card."""
        if getattr(self, "fields_only", False) and self.cards:
            return "\n\n".join(c.fields_text() for c in self.cards)
        out = []
        for line, new in zip(self.lines, self.block_starts()):
            if new and out:
                out.append("")
            out.append(line.text)
        return "\n".join(out)


def _is_vertical(quad) -> bool:
    tl, tr, br, bl = quad
    return float(np.linalg.norm(bl - tl)) > 1.5 * float(np.linalg.norm(tr - tl))


_AR_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")
_TO_AR = str.maketrans("0123456789", "٠١٢٣٤٥٦٧٨٩")


def _apply_taught(page):
    """Corrections the user taught the program (see learn.py)."""
    try:
        from .learn import apply_page
        apply_page(page)
    except OSError:
        pass


def _looks_like_card(lines) -> bool:
    """An ID card: its title line, even read badly ("بظافة تحقلق الشخعية")."""
    from .textfix import match_phrase
    return any("تحقيق" in l.text or match_phrase(l.text, 0.35) == "بطاقة تحقيق الشخصية" for l in lines)


def _has_gap(r) -> bool:
    from .fuse import _gaps
    return any(g1 - g0 > 0.06 for g0, g1 in _gaps(r, factor=4.0))


def _diacritic_ratio(text: str) -> float:
    import re
    marks = len(re.findall("[\u064b-\u0652]", text))
    letters = len(re.findall("[\u0621-\u064a]", text))
    return marks / max(1, letters)


_GOV_CODES = {"01", "02", "03", "04", "11", "12", "13", "14", "15", "16", "17", "18", "19",
              "21", "22", "23", "24", "25", "26", "27", "28", "29", "31", "32", "33", "34", "35", "88"}


def _digits(text: str) -> str:
    """Western digits of a reading; "." between digits is a "٠" read as a dot."""
    t = text.replace(".", "0").replace("٫", "0")
    return "".join(str(int(c)) for c in t if c.isdigit())


def valid_national_id(d: str) -> bool:
    """14 digits: century (2/3), YYMMDD, governorate, serial, check."""
    if len(d) != 14 or d[0] not in "23":
        return False
    mm, dd = int(d[3:5]), int(d[5:7])
    return 1 <= mm <= 12 and 1 <= dd <= 31 and d[7:9] in _GOV_CODES and _id_check(d)


_ID_WEIGHTS = (2, 7, 6, 5, 4, 3, 2, 7, 6, 5, 4, 3, 2)


def _id_check(d: str) -> bool:
    """The last digit of an Egyptian national ID is a check digit: weights
    2765432765432 over the first 13, then (11 - sum mod 11) mod 10. A digit read
    wrong ("١" as "٩") breaks it, so a wrong reading can't pass for a real ID."""
    r = sum(int(a) * w for a, w in zip(d[:13], _ID_WEIGHTS)) % 11
    return (11 - r) % 10 == int(d[13])


def _plausible_id(d: str) -> bool:
    """Shape of an ID without the check digit (century, date, governorate)."""
    if len(d) != 14 or d[0] not in "23":
        return False
    mm, dd = int(d[3:5]), int(d[5:7])
    return 1 <= mm <= 12 and 1 <= dd <= 31 and d[7:9] in _GOV_CODES


def date_prefix(date: str) -> str:
    """The first 7 digits of the national ID for a birth date: century (2 for
    19xx, 3 for 20xx), YY, MM, DD. "2003/05/24" -> "3030524". "" if not a date."""
    import re
    m = re.fullmatch(r"(19|20)(\d\d)/(\d{1,2})/(\d{1,2})", date.translate(_AR_DIGITS).replace(" ", ""))
    if not m or not _good_date(date):
        return ""
    return ("2" if m.group(1) == "19" else "3") + m.group(2) + m.group(3).zfill(2) + m.group(4).zfill(2)


def best_national_id(readings, date: str = "") -> str:
    """The ID from several readings of the same number (texts with digits).

    A reading that passes the check digit twice wins. Otherwise the readings
    are lined up digit by digit and, where they disagree, the digits that were
    read are tried until the check digit fits; only one place may differ from
    the majority.

    The birth date printed on the card is the ID's first 7 digits, so the two
    check each other: a reading that agrees with the date and passes the check
    digit is taken first, and when nothing else works a reading whose date part
    came out wrong is repaired from the date (its last 7 digits + the date's 7).
    "" when nothing fits."""
    import re
    from collections import Counter
    cands = Counter()
    for t in readings:
        for run in re.findall(r"[0-9٠-٩][0-9٠-٩ .]*[0-9٠-٩]", t):
            d = _digits(run)
            for k in range(0, max(0, len(d) - 13)):
                if _plausible_id(d[k:k + 14]):
                    cands[d[k:k + 14]] += 1
    pre = date_prefix(date) if date else ""
    if pre:
        agree = [(n, c) for c, n in cands.items() if c.startswith(pre) and valid_national_id(c)]
        if agree:
            return max(agree)[1]
    found = _vote_id(cands)
    if found or not pre:
        return found
    tails = Counter()
    for t in readings:
        d = _digits(t)
        if len(d) >= 11 and valid_national_id(pre + d[-7:]):
            tails[pre + d[-7:]] += 1
    if tails:
        c, n = tails.most_common(1)[0]
        if n >= 2:
            return c
    return ""


def _vote_id(cands) -> str:
    from collections import Counter
    if not cands:
        return ""
    good = [(n, c) for c, n in cands.items() if valid_national_id(c)]
    if good:
        n, c = max(good)
        if n >= 2:
            return c
    if sum(cands.values()) < 2:
        return ""
    # all readings agree but for one digit, and of the digits read there only
    # one fits the check digit: that is the number (one wrong digit always
    # breaks the check, bar a rare case, so the others can't be right)
    diff = [i for i in range(14) if len({c[i] for c in cands}) > 1]
    if len(diff) == 1 and len(good) == 1:
        return good[0][1]
    cols = [Counter() for _ in range(14)]
    for c, n in cands.items():
        for i, ch in enumerate(c):
            cols[i][ch] += n
    major = "".join(col.most_common(1)[0][0] for col in cols)
    if valid_national_id(major):
        return major
    best = None
    choices = [[d for d, _ in col.most_common(3)] for col in cols]
    for i in range(14):
        for alt in choices[i][1:]:
            c = major[:i] + alt + major[i + 1:]
            if valid_national_id(c):
                score = cols[i][alt]
                if best is None or score > best[0]:
                    best = (score, c)
    if best and best[0] >= 2:
        return best[1]
    return ""

def _ink_words(band: np.ndarray, h: float):
    """Word boxes from dark ink on a smooth background (no detector): black-hat
    against the local background, letters joined into words."""
    g = cv2.cvtColor(band, cv2.COLOR_BGR2GRAY)
    k = max(5, int(h * 0.8) | 1)
    bh = cv2.morphologyEx(g, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    if bh.max() < 40:
        return []
    _, m = cv2.threshold(bh, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_RECT, (max(3, int(h * 0.45)), max(1, int(h * 0.15)))))
    n, _, st, _ = cv2.connectedComponentsWithStats(m, 8)
    out = []
    for i in range(1, n):
        x, y, w, hh = st[i, :4]
        if 0.45 * h <= hh <= 1.6 * h and w >= 0.6 * h:
            pad = max(2, int(0.2 * hh))
            x0, y0 = max(0, x - pad), max(0, y - pad)
            x1, y1 = min(band.shape[1], x + w + pad), min(band.shape[0], y + hh + pad)
            out.append(DetBox(np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], np.float32), 0.5))
    return out


def _fix_known_text(lines):
    """Printed phrases of Egyptian documents, by context:
    - a line close to a known phrase ("بطاقة تحقيق الشخمية") is that phrase;
    - on an ID card, the calligraphy logo above "بطاقة تحقيق الشخصية" is
      "جمهورية مصر العربية" (no font reproduces it, so it is never read right);
    - on a car plate, the Arabic word next to "EGYPT" is "مصر";
    - compound names split or misread: "عبدالر حيم" -> "عبدالرحيم".
    """
    from .textfix import fix_compound_names, has_arabic, match_phrase
    for ln in lines:
        p = match_phrase(ln.text)
        if p:
            ln.text = p
        elif has_arabic(ln.text):
            ln.text = fix_compound_names(ln.text)
    for i, ln in enumerate(lines):
        if ln.text != "بطاقة تحقيق الشخصية":
            continue
        y0 = min(r.y0 for r in ln.regions)
        h = max(r.y1 for r in ln.regions) - y0
        for up in lines:
            if up is ln or not has_arabic(up.text):
                continue
            uy1 = max(r.y1 for r in up.regions)
            ux0, ux1 = min(r.x0 for r in up.regions), max(r.x1 for r in up.regions)
            lx0, lx1 = min(r.x0 for r in ln.regions), max(r.x1 for r in ln.regions)
            overlap = min(ux1, lx1) - max(ux0, lx0)
            if 0 <= y0 - uy1 + 0.6 * h < 3 * h and overlap > 0.3 * (lx1 - lx0) \
                    and 10 <= len(up.text.replace(" ", "")) <= 32:
                up.text = "جمهورية مصر العربية"
                break
    import re
    for ln in lines:      # card serial split in two: "I F7393669" -> "IF7393669"
        ln.text = re.sub(r"^([A-Z]) ([A-Z][0-9]{7})$", r"\1\2", ln.text.strip()) if ln.text.strip() else ln.text
    for ln in lines:
        if "EGYPT" in ln.text:
            parts = ln.text.split()
            ar = [p for p in parts if has_arabic(p)]
            if len(ar) == 1 and len(ar[0]) <= 5:
                ln.text = " ".join("مصر" if p == ar[0] else p for p in parts)


def _good_date(t: str) -> bool:
    """A real date as printed on a card: YYYY/MM/DD, 1900-2099."""
    import re
    m = re.fullmatch(r"(19|20)(\d\d)/(\d{1,2})/(\d{1,2})", t.translate(_AR_DIGITS).replace(" ", ""))
    return bool(m) and 1 <= int(m.group(3)) <= 12 and 1 <= int(m.group(4)) <= 31


def _fix_birth_date(lines, force_card: bool = False):
    """Use the national ID to repair the printed birth date on an ID card.

    The ID encodes it: century (2 = 19xx, 3 = 20xx), YY, MM, DD. The printed
    date sits under the eagle stamp, so it comes back damaged: "/" read as "١",
    "٠" read as ".", digits hidden. A date-like line ("/" plus a few digits) is
    replaced by the date from the nearest ID number on the page when what was
    read is close to it. Pages with several cards pair each date with the ID on
    the same card (nearest vertically).
    """
    import re
    from .layout import _edit
    ids = []
    for l in lines:
        t = l.text.translate(_AR_DIGITS)
        if re.fullmatch(r"[23]\d{13}", t):
            mm, dd = int(t[3:5]), int(t[5:7])
            if 1 <= mm <= 12 and 1 <= dd <= 31:
                cy = sum(r.cy for r in l.regions) / len(l.regions)
                ids.append((cy, ("19" if t[0] == "2" else "20") + t[1:3] + t[3:7]))
    if not ids:
        return
    # on an ID card the printed date IS the ID's date: a damaged reading only
    # has to show the year
    is_card = force_card or _looks_like_card(lines)
    for l in lines:
        t = l.text.translate(_AR_DIGITS).strip()
        if "/" not in t or not re.fullmatch(r"[\d/\s.\-()]+", t):
            continue
        read = re.sub(r"[^\d.]", "", t).replace(".", "0")   # a dotted zero is still a zero
        if not 4 <= len(read) <= 10:
            continue
        cy = sum(r.cy for r in l.regions) / len(l.regions)
        target = min(ids, key=lambda x: abs(x[0] - cy))[1]          # YYYYMMDD
        # compare with the part of the date that was actually read
        part = target[:len(read)]
        if (len(read) >= 4 and _edit(read, part) <= len(read) // 2) or _edit(read, target) <= 2 \
                or (is_card and target[2:4] in read):
            l.text = f"{target[:4]}/{target[4:6]}/{target[6:]}".translate(_TO_AR)


class ArabicOcr:
    def __init__(self, models_dir: str = MODELS, use_clahe: bool = False, det_limit: int = 1600,
                 rec_model: str = "rec_arabic.onnx", rec_dict: str = "ppocrv5_arabic_dict.txt",
                 egyptian: bool = True, number_stretch: float = 1.6, use_latin: bool = True,
                 card_mode: bool = False):
        from .cpu import lower_priority
        lower_priority()                 # the OCR yields the CPU to whatever you are doing
        self.det = Detector(os.path.join(models_dir, "det.onnx"), limit_side=det_limit)
        self.rec = Recognizer(os.path.join(models_dir, rec_model), os.path.join(models_dir, rec_dict))
        latin_path = os.path.join(models_dir, "rec_latin.onnx")
        self.latin = (Recognizer(latin_path, os.path.join(models_dir, "ppocrv5_dict.txt"))
                      if use_latin and os.path.exists(latin_path) else None)
        self.use_clahe = use_clahe
        # "Egyptian national ID" mode (chosen by the user): every page is one or
        # more ID cards - card fixes always on, text outside the cards dropped
        self.card_mode = card_mode
        self.egyptian = egyptian
        self.number_stretch = number_stretch
        # Pages printed with diacritics (shadda, tanween, damma) confuse the main
        # model: marks read as letters ("ثُفُّر" for "تفكّر"). A second model,
        # trained to see marks and leave them out, reads those pages; it loads
        # only when such a page shows up.
        self._tashkeel_path = os.path.join(models_dir, "rec_arabic_tashkeel.onnx")
        self._tashkeel_dict = os.path.join(models_dir, rec_dict)
        self._tashkeel = None
        self._book_path = os.path.join(models_dir, "rec_arabic_book.onnx")
        self._book = None
        self._doc_tashkeel = None        # decided on the first page of text of each file

    def _retry_numbers(self, crops, recs):
        """Re-read lines that contain numbers with the image stretched sideways.

        Narrow digits ("450,000") can be packed tighter than the model's time steps,
        so repeated digits collapse and the number comes back short ("450").
        Stretching the line gives each digit room. Keep the retry only when it finds
        more digits without losing Arabic letters.
        """
        if not self.number_stretch:
            return recs
        idx = [i for i, r in enumerate(recs) if count_digits(r.text) > 0]
        if not idx:
            return recs
        s = self.number_stretch
        wide = [cv2.resize(crops[i], (int(crops[i].shape[1] * s), crops[i].shape[0])) for i in idx]
        for i, r2 in zip(idx, self.rec(wide)):
            r1 = recs[i]
            if (count_digits(r2.text) > count_digits(r1.text)
                    and count_letters(r2.text) >= count_letters(r1.text) - 1):
                recs[i] = r2
        return recs

    def _retry_edges(self, img, boxes, recs):
        """Re-read each box with some margin left and right.

        The detector sometimes clips the first or last letter of a word (a fold
        or a crack in an ID card: "سين" instead of "حسين"). Keep the wider read
        only when it contains the original reading and just adds letters at the
        ends, so it can't swap in something different.
        """
        if not boxes:
            return recs
        wide = []
        ext = [(float(b.quad[:, 0].min()), float(b.quad[:, 0].max()),
                float(b.quad[:, 1].min()), float(b.quad[:, 1].max())) for b in boxes]
        for k, b in enumerate(boxes):
            tl, tr, br, bl = b.quad
            h = float(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr)))
            # don't reach into a neighbouring word on the same line
            x0, x1, y0, y1 = ext[k]
            room = 0.6 * h
            for j, (a0, a1, b0, b1) in enumerate(ext):
                if j == k or min(y1, b1) - max(y0, b0) < 0.3 * h:
                    continue
                gap = a0 - x1 if a0 >= x1 else x0 - a1 if a1 <= x0 else 0.0
                room = min(room, max(0.0, 0.8 * gap))
            if room < 0.25 * h:
                wide.append(None)
                continue
            d = tr - tl
            n = np.linalg.norm(d) or 1.0
            u = d / n * room
            q = np.array([tl - u, tr + u, br + u, bl - u], np.float32)
            q[:, 0] = np.clip(q[:, 0], 0, img.shape[1] - 1)
            wide.append(crop_quad(img, q))
        idx = [i for i, w in enumerate(wide) if w is not None]
        if len(boxes) > 25:      # a page of text: only lines the model wasn't sure of
            idx = [i for i in idx if recs[i].confidence < 0.9]
        for i, r2 in zip(idx, self.rec([wide[i] for i in idx]) if idx else []):
            r1 = recs[i]
            a, b2 = r1.text.strip(), r2.text.strip()
            if a and len(b2) > len(a) and a in b2 and r2.confidence >= r1.confidence - 0.05 \
                    and count_letters(b2) - count_letters(a) <= 2:
                recs[i] = r2
        return recs

    def _reread_number_rows(self, img, regions):
        """Read rows that contain numbers again as one piece.

        On ID cards the national number is printed in spaced groups over a
        patterned background; the detector splits it and can miss a group
        entirely. Reading the whole row as one image recovers the missing digits.
        """
        out, rows = [], []
        # rows by vertical overlap only: the groups of a spaced number can sit far apart
        for r in sorted((r for r in regions if r.rec.text.strip()), key=lambda r: r.cy):
            for row in rows:
                y0 = min(x.y0 for x in row); y1 = max(x.y1 for x in row)
                cy = sum(x.cy for x in row) / len(row)
                if abs(r.cy - cy) < 0.3 * min(r.height, y1 - y0) and 0.66 < r.height / (y1 - y0) < 1.5:
                    row.append(r); break
            else:
                rows.append([r])
        for row in rows:
            # dates ("1983/08/14") stay on their own; only plain digit groups are joined
            nums = [r for r in row if count_digits(r.rec.text) and not count_letters(r.rec.text)
                    and "/" not in r.rec.text]
            others = [r for r in row if not any(r is n for n in nums)]
            digits = sum(count_digits(r.rec.text) for r in nums)
            if digits < 4 or len(nums) < 2:
                out.extend(row); continue
            out.extend(others)
            row = nums
            pad = 0.15 * max(r.height for r in row)
            x0 = max(0, min(r.x0 for r in row) - pad); x1 = min(img.shape[1] - 1, max(r.x1 for r in row) + pad)
            y0 = max(0, min(r.y0 for r in row) - pad); y1 = min(img.shape[0] - 1, max(r.y1 for r in row) + pad)
            quad = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], np.float32)
            crop = crop_quad(img, quad)
            rec = self._retry_numbers([crop], self.rec([crop]))[0]
            if (count_digits(rec.text) > digits
                    and count_letters(rec.text) >= sum(count_letters(r.rec.text) for r in row) - 1):
                if self.egyptian:
                    rec.text = normalize_egyptian(rec.text)
                out.append(Region(quad, rec))
            else:
                out.extend(row)
        return out

    def image(self, img: np.ndarray, width_pt: float | None = None, height_pt: float | None = None,
              dpi: float = 72.0) -> PageResult:
        t0 = time.perf_counter()
        if self.egyptian:
            page = self._cards_page(img, width_pt, height_pt, dpi)
            if page is not None:
                _apply_taught(page)
                page.fields_only = self.card_mode and bool(page.cards)
                page.seconds = time.perf_counter() - t0
                return page
        boxes = self.det(img)
        n_main = len(boxes)
        boxes = boxes + self._more_boxes(img, boxes)
        # Diacritics decide the reader (see __init__). A book is printed one way
        # throughout: decide on the first page of text, from a sample of its
        # lines, and keep that for the rest of the document.
        use_tash = False
        if n_main > 25 and not self.card_mode and os.path.exists(self._tashkeel_path):
            if self._doc_tashkeel is None:
                sample = sorted(boxes[:n_main], key=lambda b: -float(np.ptp(b.quad[:, 0])))[:12]
                txt = " ".join(r.text for r in self.rec([crop_quad(img, b.quad) for b in sample]))
                self._doc_tashkeel = _diacritic_ratio(txt) > 0.025
            use_tash = self._doc_tashkeel
        main = self.rec
        use_book = False
        if use_tash:
            if self._tashkeel is None:
                self._tashkeel = Recognizer(self._tashkeel_path, self._tashkeel_dict)
            self.rec = self._tashkeel
        elif n_main > 25 and not self.card_mode and os.path.exists(self._book_path):
            # a page of running text: the reader trained on book typefaces
            # (stacked ligatures like يمكن، المحسوس، تجديد; kashida)
            if self._book is None:
                self._book = Recognizer(self._book_path, self._tashkeel_dict)
            self.rec = self._book
            use_book = True
        try:
            page = self._image(img, width_pt, height_pt, dpi, boxes=boxes, n_main=n_main)
        finally:
            self.rec = main
        if use_book:
            import re as _re
            from .textfix import fix_allah, fix_shadda_tanween
            for ln in page.lines:
                # this reader sometimes adds short vowels the page doesn't have:
                # keep shadda and tanween (printed in these books), drop the rest
                ln.text = fix_shadda_tanween(fix_allah(_re.sub("[\u064e\u064f\u0650\u0652]", "", ln.text)))
        if self.egyptian and page.lines and (self.card_mode or any("تحقيق الشخص" in l.text for l in page.lines)):
            from .cards import extract, split_by_title    # card photos: their fields too
            page.cards = split_by_title(page.lines, img.shape[1], img.shape[0])
            if self.card_mode:
                if not page.cards:                # title unreadable: the photo is one card
                    page.cards = [extract(page.lines, img.shape[1], img.shape[0])]
                # only the cards' text: no camera watermark, no background print
                on_card = {id(l) for c in page.cards for l in c.lines}
                if on_card:
                    page.lines = [l for l in page.lines if id(l) in on_card]
                page.fields_only = bool(page.cards)
        _apply_taught(page)
        page.seconds = time.perf_counter() - t0
        return page

    def _cards_page(self, img, width_pt, height_pt, dpi):
        """A sheet with ID cards on it: read each card upright, on its own."""
        from .cards import extract, find_cards, warp
        quads = find_cards(img)
        if not quads:
            return None
        lines, breaks, cards = [], [], []
        for q in quads:
            best = None
            for flip in (False, True):
                cimg, back = warp(img, q, flip)
                pg = self._image(cimg)
                conf = float(np.mean([l.confidence for l in pg.lines])) if pg.lines else 0.0
                score = conf + sum(1.0 for l in pg.lines if "تحقيق الشخص" in l.text) \
                    + sum(0.5 for l in pg.lines if l.text.isdigit() or "١" in l.text and len(l.text) == 14)
                if best is None or score > best[0]:
                    best = (score, pg, back, cimg.shape[:2])
            _, pg, back, (ch, cw) = best
            card = extract(pg.lines, cw, ch)
            starts = pg.block_starts()
            for ln, st in zip(pg.lines, starts):
                for r in ln.regions:
                    r.quad = cv2.perspectiveTransform(r.quad.reshape(-1, 1, 2).astype(np.float32), back).reshape(-1, 2)
                ln.words.sort(key=lambda wd: wd.box[0])      # left to right on the card
                for wd in ln.words:
                    x0, y0, x1, y1 = wd.box
                    pts = cv2.perspectiveTransform(np.float32([[[x0, y0]], [[x1, y0]], [[x1, y1]], [[x0, y1]]]), back)
                    wd.quad = pts.reshape(4, 2)      # keeps the card's slant for the PDF text layer
                    wd.box = (float(pts[..., 0].min()), float(pts[..., 1].min()),
                              float(pts[..., 0].max()), float(pts[..., 1].max()))
                lines.append(ln)
                breaks.append(bool(st))
            if pg.lines:
                breaks[len(lines) - len(pg.lines)] = True
            cards.append(card)
        h, w = img.shape[:2]
        return PageResult(img, lines, width_pt or w * 72.0 / dpi, height_pt or h * 72.0 / dpi,
                          0.0, cards=cards, breaks=breaks)

    def _image(self, img: np.ndarray, width_pt: float | None = None, height_pt: float | None = None,
               dpi: float = 72.0, boxes=None, n_main: int | None = None) -> PageResult:
        t0 = time.perf_counter()
        work = clahe(img) if self.use_clahe else img
        if boxes is None:
            boxes = self.det(work)
            n_main = len(boxes)
            boxes = boxes + self._more_boxes(work, boxes)
        elif n_main is None:
            n_main = len(boxes)
        crops = [crop_quad(work, b.quad) for b in boxes]
        recs = self.rec(crops) if crops else []
        # words found by the gap search have rough boxes: read each with a
        # narrower and a wider margin too, and keep the surest reading
        # On a card or a photo (few boxes) the detector's boxes can be tight
        # too: a box that cuts the top of the letters reads "ماميه" for
        # "ساميه". Read short ones again a little taller as well.
        retry = list(range(n_main, len(boxes)))
        if len(boxes) <= 25:
            # names only: a date or a number re-read can lose its "/" and pass for an ID
            retry += [i for i in range(n_main) if len(recs[i].text.split()) <= 2 and not _is_vertical(boxes[i].quad)
                      and count_letters(recs[i].text) >= 3 and not count_digits(recs[i].text)]
        if retry:
            from .detect import _height
            for i in retry:
                q, h = boxes[i].quad, _height(boxes[i])
                variants = [(-0.2 * h, 0.0), (0.35 * h, 0.0), (0.0, 0.2 * h), (0.15 * h, 0.2 * h)] if i >= n_main \
                    else [(0.0, 0.2 * h), (0.15 * h, 0.2 * h)]
                for dx, dy in variants:
                    x0, x1 = q[:, 0].min() - dx, q[:, 0].max() + dx
                    if x1 - x0 < h:
                        continue
                    y0, y1 = max(0.0, q[:, 1].min() - dy), min(work.shape[0] - 1.0, q[:, 1].max() + dy)
                    alt = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], np.float32)
                    alt[:, 0] = np.clip(alt[:, 0], 0, work.shape[1] - 1)
                    c = crop_quad(work, alt)
                    r = self.rec([c])[0]
                    if r.confidence > recs[i].confidence + 0.02 and \
                            len(r.text.strip()) <= len(recs[i].text.strip()) + 1:
                        boxes[i] = DetBox(alt, boxes[i].score); crops[i] = c; recs[i] = r
        # text found only by looking again between lines: keep it when it reads clearly
        keep = [i for i in range(len(boxes)) if i < n_main or
                (recs[i].confidence >= 0.75 and (count_letters(recs[i].text.replace(" ", "")) >= 3
                                                  or count_digits(recs[i].text) >= 2)
                 and len(recs[i].text.split()) <= 3)]
        # a scrap the detector caught (two letters of a name) goes when the
        # gap search read the whole word over it
        found = [boxes[i] for i in keep if i >= n_main]

        def inside(a, b):
            ax0, ax1, ay0, ay1 = a.quad[:, 0].min(), a.quad[:, 0].max(), a.quad[:, 1].min(), a.quad[:, 1].max()
            bx0, bx1, by0, by1 = b.quad[:, 0].min(), b.quad[:, 0].max(), b.quad[:, 1].min(), b.quad[:, 1].max()
            iw = max(0.0, min(ax1, bx1) - max(ax0, bx0)); ih = max(0.0, min(ay1, by1) - max(ay0, by0))
            return iw * ih / max(1.0, (ax1 - ax0) * (ay1 - ay0))
        keep = [i for i in keep if i >= n_main or not any(inside(boxes[i], f) >= 0.6 for f in found)]
        boxes, crops, recs = [boxes[i] for i in keep], [crops[i] for i in keep], [recs[i] for i in keep]
        # vertical text can run up or down: try the other way round too
        vert = [i for i, c in enumerate(crops) if _is_vertical(boxes[i].quad)]
        if vert:
            flipped = [np.rot90(crops[i], 2).copy() for i in vert]
            for i, f, r2 in zip(vert, flipped, self.rec(flipped)):
                if r2.confidence > recs[i].confidence:
                    crops[i], recs[i] = f, r2
        recs = self._retry_numbers(crops, recs)
        recs = self._retry_edges(work, boxes, recs)
        if self.latin and crops:
            # the Latin model fills codes and Western digits into gaps of the
            # Arabic reading; on a page of text, only lines that may hold one
            idx = list(range(len(crops)))
            if len(crops) > 25:
                idx = [i for i in idx if recs[i].confidence < 0.85 or _has_gap(recs[i])
                       or any(c.isascii() and c.isalnum() for c in recs[i].text)]
            for i, l in zip(idx, self.latin([crops[i] for i in idx]) if idx else []):
                recs[i] = fill_from_latin(recs[i], l)
        if self.egyptian:
            for r in recs:
                r.text = fix_codes(normalize_egyptian(r.text))
        regions = self._reread_number_rows(work, [Region(b.quad, r) for b, r in zip(boxes, recs)])
        lines = build_lines(regions)
        from .textfix import fix_tanween
        for ln in lines:
            if any("\u0621" <= c <= "\u064a" for c in ln.text):
                ln.text = fix_tanween(ln.text)
        if self.egyptian:
            from .cards import card_frame
            frames = card_frame(lines) if (self.card_mode or _looks_like_card(lines)) else []
            if frames:
                self._card_zones(work, lines, frames)     # each field read in its own place
            else:
                self._fix_national_id(work, lines)
            for ln in lines:
                # Egyptian national ID: 14 digits printed in groups -> one number
                if NUMBER_LINE.match(ln.text) and count_digits(ln.text) == 14:
                    ln.text = ln.text.replace(" ", "")
            _fix_birth_date(lines, self.card_mode)
            _fix_known_text(lines)
        h, w = img.shape[:2]
        return PageResult(img, lines,
                          width_pt or w * 72.0 / dpi, height_pt or h * 72.0 / dpi,
                          time.perf_counter() - t0)

    def _card_zones(self, img, lines, frames):
        """ID cards: the title gives the card's position and size, and from it
        where each field is printed. The ID number is read again from its own
        place only (the birth date beside it can't get mixed in), the date
        from its place, and a first name the detector missed from its place.
        A field that can't be read stays as it was; the others don't depend on it."""
        import re
        from collections import Counter
        from .cards import _box, crop_box, zone_of
        H, W = img.shape[:2]

        def crop(x0, y0, x1, y1):
            return crop_quad(img, np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], np.float32))

        def new_line(x0, y0, x1, y1, rec, text):
            q = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], np.float32)
            got = build_lines([Region(q, rec)], min_conf=0.0)
            if not got:
                return None
            ln = got[0]; ln.text = text
            at = next((k for k, l in enumerate(lines) if _box(l)[1] > y0), len(lines))
            lines.insert(at, ln)
            return ln

        # a line glued across two fields (the birth date and the ID on one row)
        # is cut where the fields meet, so each part goes to its own field
        for fr in frames:
            for ln in list(lines):
                if len(ln.regions) < 2:
                    continue
                parts = {}
                for r in ln.regions:
                    parts.setdefault(zone_of((r.x0, r.y0, r.x1, r.y1), fr), []).append(r)
                if len(parts) < 2:
                    continue
                k = next(i for i, x in enumerate(lines) if x is ln)
                new = [l for rs in parts.values() for l in build_lines(rs, min_conf=0.0)]
                lines[k:k + 1] = sorted(new, key=lambda l: -_box(l)[2])
        for fr in frames:
            u = fr[1]
            zone = {}
            for ln in lines:
                if ln.text.strip():
                    zone.setdefault(zone_of(_box(ln), fr), []).append(ln)
            # --- the birth date as printed: it is the ID's first 7 digits ---
            hint = next((l.text for l in zone.get("date", []) if _good_date(l.text)), "")
            # --- national ID ---
            ids = [l for l in zone.get("id", []) if count_digits(l.text) >= 2 and not count_letters(l.text)]
            valid = next((l for l in ids if count_digits(l.text) == 14 and valid_national_id(_digits(l.text))), None)
            if valid is None and not hint:
                # no clean date line: read the date's place a few ways (the eagle
                # stamp sits on it) and keep a date two readings agree on
                from collections import Counter as _C
                dx0, dy0, dx1, dy1 = crop_box(fr, "date", W, H)
                seen = _C()
                if dx1 - dx0 > 4 and dy1 - dy0 > 4:
                    for a, b in ((0.0, 0.0), (0.08, -0.08), (0.15, -0.05), (0.05, -0.15)):
                        c = crop(dx0, dy0 + a * u, dx1, dy1 + b * u)
                        for r in self.rec([c]):
                            t = re.sub(r"[^0-9٠-٩/]", "", r.text).translate(_AR_DIGITS)
                            m = re.search(r"(19|20)\d\d/\d{1,2}/\d{1,2}", t)
                            if m and _good_date(m.group()):
                                seen[m.group()] += 1
                if seen and seen.most_common(1)[0][1] >= 2:
                    hint = seen.most_common(1)[0][0]
            if valid is None:
                x0, y0, x1, y1 = crop_box(fr, "id", W, H)
                if ids:            # the row the number was found on, with room above and below
                    ly0 = min(_box(l)[1] for l in ids); ly1 = max(_box(l)[3] for l in ids)
                    lh = max(ly1 - ly0, 0.12 * u)
                    y0, y1 = max(0.0, (ly0 + ly1) / 2 - 0.8 * lh), min(H - 1.0, (ly0 + ly1) / 2 + 0.8 * lh)
                texts = [l.text for l in ids]
                if x1 - x0 > 4 and y1 - y0 > 4:
                    lh = y1 - y0
                    for dy in (0.0, 0.15 * lh, 0.3 * lh):
                        c = crop(x0, max(0.0, y0 - dy), x1, min(H - 1.0, y1 + dy))
                        up = cv2.resize(c, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
                        sharp = cv2.addWeighted(up, 1.8, cv2.GaussianBlur(up, (0, 0), 2), -0.8, 0)
                        for v in (c, sharp):
                            wide = [cv2.resize(v, (int(v.shape[1] * st), v.shape[0])) for st in (1.0, 1.4, 1.8)]
                            texts += [r.text for r in self.rec(wide)]
                best = best_national_id(texts, hint)
                if best:
                    text = "".join("٠١٢٣٤٥٦٧٨٩"[int(c)] for c in best)
                    if ids:
                        keep = max(ids, key=lambda l: count_digits(l.text))
                        keep.text = text
                        for l in ids:          # pieces of the same number
                            if l is not keep:
                                del lines[next(k for k, x in enumerate(lines) if x is l)]
                        valid = keep
                    else:
                        from .recognize import RecResult
                        valid = new_line(x0, y0, x1, y1, RecResult(best, 0.9), text)
            # --- birth date: the ID holds it; otherwise read its own place ---
            # a date with a stray mark read as a letter ("٩٩٤ه٢٠٠٣/٠") is still the date line
            dates = [l for l in zone.get("date", []) if count_digits(l.text) >= 4 and count_letters(l.text) <= 1
                     and not re.search(r"[A-Z]{2}", l.text)]
            if valid is not None and any(x is valid for x in lines):
                d = valid.text.translate(_AR_DIGITS)
                date = (("19" if d[0] == "2" else "20") + d[1:3] + "/" + d[3:5] + "/" + d[5:7]).translate(_TO_AR)
                for l in dates:
                    l.text = date
            elif not any(_good_date(l.text) for l in dates):
                x0, y0, x1, y1 = crop_box(fr, "date", W, H)
                if x1 - x0 > 4 and y1 - y0 > 4:
                    for r in self.rec([crop(x0, y0, x1, y1)]):
                        t = re.sub(r"[^0-9٠-٩/]", "", r.text).translate(_AR_DIGITS)
                        m = re.search(r"(19|20)\d\d/\d{1,2}/\d{1,2}", t)
                        if m and r.confidence >= 0.6 and _good_date(m.group()):
                            text = m.group().translate(_TO_AR)
                            if dates:
                                dates[0].text = text
                            else:
                                new_line(x0, y0, x1, y1, r, text)
            # --- first name: read again from its place, a few heights, and keep
            # the surest reading (a box that cuts the letter tops reads "ماميه")
            firsts = [l for l in zone.get("first", []) if count_letters(l.text) >= 2]
            x0, y0, x1, y1 = crop_box(fr, "first", W, H)
            rest = zone.get("name", [])
            if rest:               # it ends where the name line under it ends
                x1 = min(x1, max(_box(l)[2] for l in rest) + 0.05 * u)
            if firsts:             # and starts a little before the word that was found
                x0 = max(x0, min(_box(l)[0] for l in firsts) - 0.12 * u)
            from .cards import _bare, _vocab
            from .textfix import visual_to_logical
            known = _vocab("names")

            def is_name(r):
                return all(_bare(w) in known for w in visual_to_logical(r.text).split())

            best = None
            if x1 - x0 > 4 and y1 - y0 > 4:
                reads = [self.rec([crop(x0, y0 + a * u, x1, y1 + b * u)])[0]
                         for a, b in ((0.0, 0.0), (0.03, -0.02), (0.02, 0.0), (0.0, -0.03))]
                reads = [r for r in reads if r.text.strip()]
                # readings of the same word disagree on one letter ("ماميه" /
                # "ساميه"): a known first name wins over a surer non-name
                names = [r for r in reads if r.confidence >= 0.6 and is_name(r)]
                pool = names or reads
                best = max(pool, key=lambda r: r.confidence) if pool else None
                if best is not None and names and firsts and not all(
                        _bare(w) in known for w in firsts[0].text.split()):
                    best.confidence = max(best.confidence, firsts[0].confidence + 0.03)
            if best is not None and best.confidence >= 0.6 and count_letters(best.text) >= 2 \
                    and not count_digits(best.text) and len(best.text.split()) <= 2:
                text = visual_to_logical(best.text).strip()
                if not firsts:
                    new_line(x0, y0, x1, y1, best, text)
                elif len(firsts) == 1 and best.confidence > firsts[0].confidence + 0.02 \
                        and len(text.split()) == len(firsts[0].text.split()):
                    firsts[0].text = text

    def _fix_national_id(self, img, lines):
        """A national ID read with a digit missing or extra (the small "٠" is
        easily lost, a group at the edge gets cut): read the whole row again,
        wider and at several stretches, and keep a reading that is a valid ID
        (century, birth date, governorate code) if the readings agree on one."""
        import re
        from collections import Counter
        for ln in lines:
            if "/" in ln.text or not NUMBER_LINE.match(ln.text):
                continue
            nd = count_digits(ln.text)
            card = self.card_mode or _looks_like_card(lines)
            if not (10 <= nd <= 17 or (card and nd >= 2)) or (nd == 14 and valid_national_id(_digits(ln.text))):
                continue
            x0 = min(r.x0 for r in ln.regions); x1 = max(r.x1 for r in ln.regions)
            if card and nd < 14:      # only part of the number was found: the row runs to the text column edges
                x1 = max(x1, max(max(r.x1 for r in l.regions) for l in lines if l.text.strip()))
            y0 = min(r.y0 for r in ln.regions); y1 = max(r.y1 for r in ln.regions)
            h = y1 - y0
            if card and nd < 14:
                # the box covers a few digit groups only; the number is taller
                # than them and runs past the address lines to the card's edge
                h = max(h, float(np.median([max(r.y1 for r in l.regions) - min(r.y0 for r in l.regions)
                                             for l in lines if l.text.strip()])))
                cy = (y0 + y1) / 2
                y0, y1 = cy - 0.75 * h, cy + 0.75 * h
                x1 = min(img.shape[1] - 1.0, x1 + 3 * h)
                # ...and to the left as far as the text column goes, stopping
                # before the birth date that sits on the same row
                left = min(min(r.x0 for r in l.regions) for l in lines if l.text.strip() and l is not ln
                           and "/" not in l.text and not l.text.isascii())
                x0 = min(x0, left - 2 * h)
                for l in lines:
                    if l is not ln and "/" in l.text:
                        lx1 = max(r.x1 for r in l.regions)
                        if lx1 < x0 + h and abs((min(r.y0 for r in l.regions) + max(r.y1 for r in l.regions)) / 2 - cy) < 3 * h:
                            x0 = max(x0, lx1 + 0.3 * h)
                x0 = max(0.0, x0)
            texts = []
            for ext in (0.3, 0.8, 1.5):
                X0, X1 = max(0.0, x0 - ext * h), min(img.shape[1] - 1.0, x1 + ext * h)
                Y0, Y1 = max(0.0, y0 - 0.1 * h), min(img.shape[0] - 1.0, y1 + 0.1 * h)
                c = crop_quad(img, np.array([[X0, Y0], [X1, Y0], [X1, Y1], [X0, Y1]], np.float32))
                wide = [cv2.resize(c, (int(c.shape[1] * st), c.shape[0])) for st in (1.0, 1.4, 1.8)]
                # the wider crop can pick up a mark or a digit of the date at
                # its ends: best_national_id looks at any 14 digits in a row
                texts += [r.text for r in self.rec(wide)]
            best = best_national_id(texts)
            if best:
                ln.text = "".join("٠١٢٣٤٥٦٧٨٩"[int(c)] for c in best)

    def _more_boxes(self, img, boxes):
        """Gap search, twice: a line found in the first pass (a name the
        detector missed) makes the next gap - the first name above it -
        visible for the second. Found boxes get some margin: the band search
        draws them tight and can clip the first letter."""
        from .detect import _height
        found = self._gap_boxes(img, boxes)
        if found:
            found += self._gap_boxes(img, boxes + found)
        h_img, w_img = img.shape[:2]
        out = []
        for b in found:
            h = _height(b)
            q = b.quad.copy()
            x0, x1 = q[:, 0].min() - 0.3 * h, q[:, 0].max() + 0.3 * h
            y0, y1 = q[:, 1].min() - 0.15 * h, q[:, 1].max() + 0.15 * h
            x0, x1 = max(0.0, x0), min(w_img - 1.0, x1)
            y0, y1 = max(0.0, y0), min(h_img - 1.0, y1)
            out.append(DetBox(np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], np.float32), b.score))
        return out

    def _gap_boxes(self, img, boxes):
        """Short words the detector skips on a busy background (a first name on
        an ID card, over the pyramids) are found when the empty band between
        two text rows is searched on its own. Photos and small pages only."""
        from .detect import _height
        h_img, w_img = img.shape[:2]
        if not boxes or max(h_img, w_img) > 2000 or len(boxes) > 25:
            return []        # a page full of text (a book) has no skipped short words to look for
        rows = []
        hs0 = sorted(_height(b) for b in boxes)
        h0 = hs0[len(hs0) // 2]
        # a scrap of a word (the detector caught two letters of a name) must not
        # close the gap where the rest of that word is waiting to be found
        solid = [b for b in boxes if float(np.ptp(b.quad[:, 0])) >= 1.5 * h0] or boxes
        for b in sorted(solid, key=lambda b: b.quad[:, 1].min()):
            y0, y1 = float(b.quad[:, 1].min()), float(b.quad[:, 1].max())
            x0, x1 = float(b.quad[:, 0].min()), float(b.quad[:, 0].max())
            if rows and y0 < rows[-1][1] - 0.3 * (y1 - y0):
                r = rows[-1]
                rows[-1] = [min(r[0], y0), max(r[1], y1), min(r[2], x0), max(r[3], x1)]
            else:
                rows.append([y0, y1, x0, x1])
        hs = sorted(_height(b) for b in boxes)
        h = hs[len(hs) // 2]
        found = []
        # gaps between whole rows, and between two boxes stacked in the same
        # column (a mark on the photo beside the name must not hide the gap
        # in the text column where the first name is)
        pairs = list(zip(rows, rows[1:]))
        rects = [[float(b.quad[:, 1].min()), float(b.quad[:, 1].max()),
                  float(b.quad[:, 0].min()), float(b.quad[:, 0].max())] for b in solid]
        for B in rects:
            above = [A for A in rects if A[1] <= B[0] + 0.2 * h0
                     and min(A[3], B[3]) - max(A[2], B[2]) > 0.3 * min(A[3] - A[2], B[3] - B[2])]
            if above:
                pairs.append((max(above, key=lambda A: A[1]), B))
        seen = set()
        for a, b in pairs:
            key = (int(a[1]) // 4, int(b[0]) // 4, int(min(a[2], b[2])) // 20)
            if key in seen:
                continue
            seen.add(key)
            top, bot = int(a[1]) + 1, int(b[0]) - 1
            if bot - top < 0.8 * h:
                continue
            x0 = int(max(0, min(a[2], b[2]) - h)); x1 = int(min(w_img, max(a[3], b[3]) + h))
            band = img[top:bot, x0:x1]
            if band.size == 0:
                continue
            cands = [b for b in self.det(band, min_height=max(6.0, 0.4 * h))
                     if _height(b) > 0.4 * h and (b.quad[:, 0].max() - b.quad[:, 0].min()) > 0.6 * h]
            # words the detector still missed (or only caught a scrap of)
            for ib in _ink_words(band, h):
                ix0, ix1 = float(ib.quad[:, 0].min()), float(ib.quad[:, 0].max())
                covered = sum(max(0.0, min(ix1, float(c.quad[:, 0].max())) - max(ix0, float(c.quad[:, 0].min())))
                              for c in cands)
                if covered < 0.5 * (ix1 - ix0):
                    cands.append(ib)
            for nb in cands:
                q = nb.quad.copy()
                q[:, 0] += x0; q[:, 1] += top
                found.append(DetBox(q, nb.score))

        # keep only what is new: not over a word the page reading already has
        def overlap(a, b):
            ax0, ax1, ay0, ay1 = a.quad[:, 0].min(), a.quad[:, 0].max(), a.quad[:, 1].min(), a.quad[:, 1].max()
            bx0, bx1, by0, by1 = b.quad[:, 0].min(), b.quad[:, 0].max(), b.quad[:, 1].min(), b.quad[:, 1].max()
            iw = max(0.0, min(ax1, bx1) - max(ax0, bx0)); ih = max(0.0, min(ay1, by1) - max(ay0, by0))
            return iw * ih / max(1.0, (ax1 - ax0) * (ay1 - ay0))
        out = []
        for f in found:
            if all(overlap(f, b) < 0.3 and (overlap(b, f) < 0.5 or float(np.ptp(b.quad[:, 0])) < 1.5 * h0)
                   for b in boxes) \
                    and all(overlap(f, g) < 0.3 for g in out):
                out.append(f)
        return out

    def file(self, path: str, scale: float = 2.5, progress=None):
        """Yield one PageResult per page (PDF) or a single one (image)."""
        self._doc_tashkeel = None
        ext = os.path.splitext(path)[1].lower()
        if ext == ".pdf":
            import pypdfium2 as pdfium
            pdf = pdfium.PdfDocument(path)
            for i in range(len(pdf)):
                if progress:
                    progress(i + 1, len(pdf))
                page = pdf[i]
                w_pt, h_pt = page.get_size()
                pil = page.render(scale=scale).to_pil().convert("RGB")
                img = cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)
                yield self.image(img, w_pt, h_pt)
        elif ext in IMAGE_EXT:
            data = np.fromfile(path, np.uint8)  # works with Arabic file names on Windows
            img = cv2.imdecode(data, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError(f"Can't read image: {path}")
            if progress:
                progress(1, 1)
            yield self.image(img, dpi=150.0)
        else:
            raise ValueError(f"Unsupported file type: {ext}")
