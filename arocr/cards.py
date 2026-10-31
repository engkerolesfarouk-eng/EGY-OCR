"""Egyptian national ID cards: find each card on a page, turn it upright, and
read its fields.

A page can be a single phone photo of one card or a scan of a sheet with
several cards, some turned sideways. Each card is cut out by its outline
(cards are coloured on white paper), warped to a landscape rectangle, and read
both ways up; the reading with more confident text and a valid national ID wins.

Fields come from the layout of the card (right column, top to bottom): first
name, the rest of the name, then the address lines; the national ID row at the
bottom; the birth date and card number on the left. The national ID also
gives the birth date, sex and governorate of birth, which checks the reading.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import cv2
import numpy as np

CARD_ASPECT = 85.6 / 54.0          # ID-1 card

GOVERNORATES = {
    "01": "القاهرة", "02": "الإسكندرية", "03": "بورسعيد", "04": "السويس", "11": "دمياط",
    "12": "الدقهلية", "13": "الشرقية", "14": "القليوبية", "15": "كفر الشيخ", "16": "الغربية",
    "17": "المنوفية", "18": "البحيرة", "19": "الإسماعيلية", "21": "الجيزة", "22": "بني سويف",
    "23": "الفيوم", "24": "المنيا", "25": "أسيوط", "26": "سوهاج", "27": "قنا", "28": "أسوان",
    "29": "الأقصر", "31": "البحر الأحمر", "32": "الوادي الجديد", "33": "مطروح", "34": "شمال سيناء",
    "35": "جنوب سيناء", "88": "خارج الجمهورية",
}
_AR = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")
_TO_AR = str.maketrans("0123456789", "٠١٢٣٤٥٦٧٨٩")


@dataclass
class Card:
    first_name: str = ""
    name: str = ""                 # full name: first name + father + grandfather (+ family)
    address: list[str] = field(default_factory=list)
    national_id: str = ""          # Arabic-Indic digits, as printed
    birth_date: str = ""           # yyyy/mm/dd (Arabic-Indic)
    sex: str = ""
    governorate: str = ""          # of birth, from the ID
    card_number: str = ""
    id_valid: bool = False
    lines: list = field(default_factory=list)    # the card's Line objects, page coordinates

    def as_text(self) -> str:
        rows = [("الاسم", self.name), ("العنوان", "، ".join(self.address)),
                ("الرقم القومي", self.national_id), ("تاريخ الميلاد", self.birth_date),
                ("النوع", self.sex), ("محافظة الميلاد", self.governorate), ("رقم البطاقة", self.card_number)]
        return "\n".join(f"{k}: {v}" for k, v in rows if v)

    def fields_text(self) -> str:
        """What ID-card mode shows: the four fields, each on its own line. The
        birth date comes from the national ID when the printed one is unreadable."""
        rows = [("الاسم", self.name), ("العنوان", "، ".join(self.address)),
                ("الرقم القومي", self.national_id + ("" if self.id_valid or not self.national_id else "  (غير مكتمل)")),
                ("تاريخ الميلاد", self.birth_date)]
        return "\n".join(f"{k}: {v}" for k, v in rows)

    def row(self) -> list[str]:
        # ="..." keeps Excel from turning the 14-digit ID into 3.08E+13 and the date into a number
        return [self.name, self.first_name, " | ".join(self.address),
                f'="{self.national_id.translate(_AR)}"' if self.national_id else "",
                f'="{self.birth_date.translate(_AR)}"' if self.birth_date else "", self.sex, self.governorate, self.card_number,
                "نعم" if self.id_valid else "لا"]


HEADERS = ["الاسم", "الاسم الأول", "العنوان", "الرقم القومي", "تاريخ الميلاد", "النوع",
           "محافظة الميلاد", "رقم البطاقة", "الرقم القومي سليم"]


# ---------------------------------------------------------------- finding cards

def find_cards(img: np.ndarray) -> list[np.ndarray]:
    """Outlines (4x2, tl,tr,br,bl with tl->tr the long side) of the cards on a
    sheet. Empty when the page is not a sheet of cards (e.g. one close-up photo)."""
    h, w = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    paper = float(np.percentile(gray, 90))
    if paper < 200:                      # not white paper: a photo, leave it whole
        return []
    m = ((hsv[..., 1] > 18) | (gray < paper - 40)).astype(np.uint8) * 255
    k = max(5, int(min(h, w) * 0.012))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((k // 2 + 1, k // 2 + 1), np.uint8))
    cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in cs:
        (cx, cy), (rw, rh), ang = cv2.minAreaRect(c)
        if rw * rh < 0.01 * h * w or rw * rh > 0.8 * h * w:
            continue
        aspect = max(rw, rh) / max(1.0, min(rw, rh))
        if not 1.25 < aspect < 2.0 or cv2.contourArea(c) / (rw * rh) < 0.6:
            continue
        pts = cv2.boxPoints(((cx, cy), (rw, rh), ang)).astype(np.float32)
        out.append(_order_long_top(pts))
    # reading order of the sheet: rows top to bottom, right to left
    out.sort(key=lambda q: (round(q[:, 1].mean() / (h * 0.15)), -q[:, 0].mean()))
    return out


def _order_long_top(pts: np.ndarray) -> np.ndarray:
    c = pts.mean(0)
    ang = np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0])
    p = pts[np.argsort(ang)]                       # clockwise in image coords, starting anywhere
    # start at the point that makes p0->p1 a long side, and p0 the upper-left of the two
    if np.linalg.norm(p[1] - p[0]) < np.linalg.norm(p[2] - p[1]):
        p = np.roll(p, -1, axis=0)
    return p.astype(np.float32)


def warp(img: np.ndarray, quad: np.ndarray, flip: bool = False):
    """Card cut out as a landscape image; returns it and the matrix back to the page."""
    q = np.roll(quad, 2, axis=0) if flip else quad
    w = int(round(np.linalg.norm(q[1] - q[0])))
    h = int(round(np.linalg.norm(q[3] - q[0])))
    dst = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
    M = cv2.getPerspectiveTransform(q, dst)
    return cv2.warpPerspective(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE), \
        np.linalg.inv(M)


# ---------------------------------------------------------------- reading fields

def _digits(t: str) -> str:
    return "".join(c for c in t.translate(_AR) if c.isdigit())


def _box(line):
    return (min(r.x0 for r in line.regions), min(r.y0 for r in line.regions),
            max(r.x1 for r in line.regions), max(r.y1 for r in line.regions))


# Where each field sits on the front of an Egyptian national ID, measured from
# the title "بطاقة تحقيق الشخصية": X from the title's centre and Y from its
# bottom, both in title widths (u), so the size of the photo doesn't matter.
# Measured on 22 cards; a field is placed by the centre of its line.
ZONES = {
    "first":  (-0.95, 1.40, -0.05, 0.32),     # first name, top of the text column
    "name":   (-0.95, 1.40, 0.32, 0.555),     # the rest of the name
    "addr":   (-0.95, 1.40, 0.555, 1.15),     # two address lines
    "id":     (-0.95, 1.40, 1.15, 1.80),      # national ID, bottom right
    "date":   (-2.60, -0.95, 0.95, 1.52),     # birth date, under the photo
    "number": (-2.60, -0.95, 1.52, 2.05),     # card number (two letters + 7 digits)
}
# crops for reading a field again on its own: a little wider than the zone,
# never reaching into the next field (the ID crop stops before the date)
CROPS = {
    "first": (0.15, 1.35, -0.04, 0.36),
    "id":    (-0.95, 1.38, 1.10, 1.72),
    "date":  (-2.50, -0.98, 1.02, 1.60),
}


def card_frame(lines):
    """(cx, u, y0) of each card on the page, from its title lines, or [].

    "بطاقة تحقيق الشخصية" gives the frame directly. When the reader glued it to
    "جمهورية مصر العربية" above (one tall box), or only that top line was read,
    the frame comes from the top line: it is 1.5 title widths wide."""
    frames = []
    used = set()
    for ln in lines:
        t = ln.text
        if "تحقيق" not in t and "بطاقة" not in t and "الشخصية" not in t:
            continue
        x0, y0, x1, y1 = _box(ln)
        w, h = x1 - x0, y1 - y0
        if w <= 0:
            continue
        if h / w > 0.29:                       # both title lines in one box
            frames.append(((x0 + x1) / 2, w / 1.5, y1))
        else:
            frames.append(((x0 + x1) / 2, w, y1))
        used.add(id(ln))
    for ln in lines:
        if "جمهورية" not in ln.text and "العربية" not in ln.text or id(ln) in used:
            continue
        x0, y0, x1, y1 = _box(ln)
        cx, u = (x0 + x1) / 2, (x1 - x0) / 1.5
        if any(abs(cx - f[0]) < u and -0.2 * u < f[2] - y1 < 0.6 * u for f in frames):
            continue                           # the title under it already gave the frame
        frames.append((cx, u, y1 + 0.16 * u))
    return frames


def zone_of(box, frame) -> str:
    cx, u, yb = frame
    X = ((box[0] + box[2]) / 2 - cx) / u
    Y = ((box[1] + box[3]) / 2 - yb) / u
    for name, (a, b, c, d) in ZONES.items():
        if a <= X < b and c <= Y < d:
            return name
    return ""


def crop_box(frame, name, width, height):
    cx, u, yb = frame
    a, b, c, d = CROPS[name]
    return (max(0.0, cx + a * u), max(0.0, yb + c * u), min(width - 1.0, cx + b * u), min(height - 1.0, yb + d * u))


def extract(lines, width: float, height: float) -> Card:
    """Fields from the lines of one upright card (card image coordinates)."""
    from .pipeline import valid_national_id
    from .textfix import has_arabic
    frames = card_frame(lines)
    if frames:
        return _extract_zoned(lines, frames[0])
    card = Card(lines=lines)
    title_bottom, id_top = 0.0, height
    text_lines = []
    for ln in lines:
        t = ln.text.strip()
        x0, y0, x1, y1 = _box(ln)
        if not t:
            continue
        if "جمهورية" in t or "تحقيق" in t or "بطاقة" in t:
            title_bottom = max(title_bottom, y1)
            continue
        d = _digits(t)
        code = re.search(r"\b([A-Z][A-Z0-9][0-9]{7})\b", t)
        if code and not card.card_number:
            card.card_number = _card_number(code.group(1))
            continue
        if len(d) == 14 and valid_national_id(d) and not card.id_valid:
            card.national_id = d.translate(_TO_AR)
            card.id_valid = True
            id_top = min(id_top, y0)
            continue
        if re.fullmatch(r"\d{4}/\d{1,2}/\d{1,2}", t.translate(_AR).replace(" ", "")):
            card.birth_date = t.replace(" ", "")
            continue
        if len(d) >= 8 and not has_arabic(t) and re.fullmatch(r"[\d٠-٩\s]+", t):   # an ID read with a digit missing
            if not card.national_id:
                card.national_id = t.replace(" ", "")
                id_top = min(id_top, y0)
            continue
        if has_arabic(t) and (x0 + x1) / 2 > 0.35 * width:
            text_lines.append((y0, y1, t, ln))
    if not card.id_valid:
        # the ID glued to the date on one line ("١٩٧٢/٠ ٢ ٧٢ ٠٢ ٢٦ ١٢ ٠٠٤٩٣"):
        # look for 14 digits in a row, spaces allowed, inside any line
        for ln in lines:
            for m in re.finditer(r"[0-9٠-٩][0-9٠-٩ ]{12,}[0-9٠-٩]", ln.text):
                d = _digits(m.group())
                for k in range(0, len(d) - 13):
                    if valid_national_id(d[k:k + 14]):
                        card.national_id = d[k:k + 14].translate(_TO_AR)
                        card.id_valid = True
                        id_top = min(id_top, _box(ln)[1])
                        break
                if card.id_valid:
                    break
            if card.id_valid:
                break
    text_lines = [tl for tl in text_lines if title_bottom - 2 <= tl[0] and tl[1] <= id_top + 2]
    text_lines.sort()
    # name block, then address block: split at the biggest gap between lines
    if text_lines:
        hs = sorted(x[1] - x[0] for x in text_lines)
        lh = hs[len(hs) // 2]
        groups = [[text_lines[0]]]
        for prev, cur in zip(text_lines, text_lines[1:]):
            if cur[0] - prev[1] > 0.55 * lh:
                groups.append([])
            groups[-1].append(cur)
        span = id_top - title_bottom
        if title_bottom > 0 and id_top < height and span > 3 * lh:
            # the card's own layout: names in the upper part of the space
            # between the title and the ID number, the address below
            cut = title_bottom + 0.42 * span
            name_lines = [x for x in text_lines if (x[0] + x[1]) / 2 < cut]
            addr_lines = [x for x in text_lines if (x[0] + x[1]) / 2 >= cut]
        elif len(groups) >= 2:
            name_lines, addr_lines = groups[0], [x for g in groups[1:] for x in g]
        else:
            name_lines, addr_lines = text_lines[:2], text_lines[2:]
        names = [_clean_name(x[2]) for x in name_lines]
        card.first_name = names[0] if len(names) >= 2 else ""
        card.name = " ".join(names)
        card.address = [_clean_address(x[2]) for x in addr_lines]
        # the page text shows the corrected names too
        for x, new in zip(name_lines + addr_lines, names + card.address):
            x[3].text = new
    if card.id_valid:
        d = card.national_id.translate(_AR)
        y = ("19" if d[0] == "2" else "20") + d[1:3]
        card.birth_date = f"{y}/{d[3:5]}/{d[5:7]}".translate(_TO_AR)
        card.sex = "ذكر" if int(d[12]) % 2 else "أنثى"
        card.governorate = GOVERNORATES.get(d[7:9], "")
    return card


# the card number is two letters and seven digits: a letter read as a digit
_AS_LETTER = str.maketrans("0921586", "OQZISBG")


def _card_number(code: str) -> str:
    return code[0] + code[1].translate(_AS_LETTER) + code[2:]


def _extract_zoned(lines, frame) -> Card:
    """Each field from its own place on the card: a field that is missing or
    misread leaves the others alone (the birth date never runs into the ID)."""
    from .pipeline import valid_national_id
    from .textfix import has_arabic
    card = Card(lines=lines)
    by = {k: [] for k in ZONES}
    for ln in lines:
        t = ln.text.strip()
        if t:
            z = zone_of(_box(ln), frame)
            if z:
                by[z].append((_box(ln)[1], t, ln))
    for k in by:
        by[k].sort(key=lambda x: x[0])
    for _, t, _ln in by["id"]:
        d = _digits(t)
        if len(d) == 14 and valid_national_id(d):
            card.national_id, card.id_valid = d.translate(_TO_AR), True
            break
        if len(d) >= 8 and not card.national_id and re.fullmatch(r"[\d٠-٩\s]+", t):
            card.national_id = d.translate(_TO_AR)        # shown as read, marked not valid
    from .pipeline import _good_date
    for _, t, _ln in by["date"]:
        if _good_date(t):            # a date that can't be real is left empty, not guessed
            card.birth_date = t.replace(" ", "")
    for _, t, _ln in by["number"] + by["date"]:
        code = re.search(r"\b([A-Z][A-Z0-9][0-9]{7})\b", t)
        if code and not card.card_number:
            card.card_number = _card_number(code.group(1))
    first = [x for x in by["first"] if has_arabic(x[1])]
    rest = [x for x in by["name"] if has_arabic(x[1])]
    addr = [x for x in by["addr"] if has_arabic(x[1]) or _digits(x[1])]
    names = [_clean_name(x[1]) for x in first + rest]
    card.first_name = " ".join(_clean_name(x[1]) for x in first)
    card.name = " ".join(n for n in names if n)
    card.address = [_clean_address(x[1]) for x in addr]
    for x, new in zip(first + rest + addr, names + card.address):
        x[2].text = new
    if card.id_valid:
        d = card.national_id.translate(_AR)
        y = ("19" if d[0] == "2" else "20") + d[1:3]
        card.birth_date = f"{y}/{d[3:5]}/{d[5:7]}".translate(_TO_AR)
        card.sex = "ذكر" if int(d[12]) % 2 else "أنثى"
        card.governorate = GOVERNORATES.get(d[7:9], "")
    return card


def split_by_title(lines, width, height) -> list[Card]:
    """Cards in one photo (no white paper to cut them out): each card begins at
    its title "بطاقة تحقيق الشخصية"; every line belongs to the nearest title
    above it (the "جمهورية مصر العربية" logo just above a title goes with it)."""
    titles = [l for l in lines if "تحقيق الشخص" in l.text]
    if not titles:
        return []
    tb = [_box(t) for t in titles]
    groups = {i: [] for i in range(len(titles))}
    for ln in lines:
        x0, y0, x1, y1 = _box(ln)
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        best = None
        for i, (tx0, ty0, tx1, ty1) in enumerate(tb):
            th = ty1 - ty0
            tw = tx1 - tx0
            # a card is about 13 title-heights tall: text further down is not on it
            if cy < ty0 - 2.5 * th or cy > ty1 + 13 * th or abs(cx - (tx0 + tx1) / 2) > 2.5 * tw:
                continue
            d = cy - ty0
            if best is None or (d >= -2.5 * th and d < best[0]):
                best = (d, i)
        if best is not None:
            groups[best[1]].append(ln)
    return [extract(groups[i], width, height) for i in range(len(titles)) if groups[i]]


COMMON_NAMES = {"محمد", "احمد", "أحمد", "محمود", "مصطفى", "يوسف", "يونس", "ابراهيم", "إبراهيم", "اسماعيل",
                "حسن", "حسين", "على", "علي", "عمر", "عمرو", "خالد", "سيد", "السيد", "طه", "ياسر", "اسلام",
                "إسلام", "ايمن", "أيمن", "كرم", "رجب", "شعبان", "رمضان", "عادل", "شادى", "بلال", "بدوى",
                "ممدوح", "صلاح", "سامح", "سامى", "هانى", "وائل", "وليد", "كريم", "مينا", "جرجس", "كيرلس",
                "بيشوى", "مرقس", "يوحنا", "بطرس", "فاطمة", "مريم", "منى", "هدى", "سعاد", "نادية", "زينب"}


_VOCAB: dict = {}
# letters the reader mixes up: same body, different dots, or nearly the same shape
_LOOKALIKE = ["بتثنيئى", "جحخ", "دذ", "رز", "سش", "صض", "طظ", "عغ", "فق", "هة", "لن", "اأإآ", "وؤ"]


def _bare(w: str) -> str:
    return (w.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا")
             .replace("ة", "ه").replace("ى", "ي"))


def _vocab(kind: str) -> dict:
    """bare form -> spellings, from arocr/data/<kind>_eg.txt (one-word entries)."""
    if kind not in _VOCAB:
        import os
        path = os.path.join(os.path.dirname(__file__), "data", f"{kind}_eg.txt")
        v: dict = {}
        try:
            for w in open(path, encoding="utf-8").read().split():
                v.setdefault(_bare(w), []).append(w)
        except OSError:
            pass
        _VOCAB[kind] = v
    return _VOCAB[kind]


def _correct(tok: str, kind: str) -> str:
    """A word one letter away from exactly one known name/place becomes it
    ("صادل" -> "عادل", "مجمود" -> "محمود", "انتور" -> "انور"). Known words,
    short words and ambiguous ones are left alone."""
    from .textfix import _lev
    v = _vocab(kind)
    b = _bare(tok)
    if len(b) < 3 or b in v or not re.fullmatch(r"[ء-ي]+", tok):
        return tok
    if b[0] in "وبلفك" and b[1:] in v:          # "بالفكريه" = "ب" + "الفكريه": a known word
        return tok
    def ok(k):
        if abs(len(k) - len(b)) > 1 or _lev(k, b) != 1:
            return False
        if len(b) == 3 and len(k) != 3:     # a short word: a look-alike letter only ("غلى" -> "على")
            return False
        if len(k) == len(b):
            # a swapped letter only between letters that look alike when
            # printed ("مجمود"/"محمود"), never "فولى" -> "فوزى"
            x, y = next((p, q) for p, q in zip(b, k) if p != q)
            return any(x in g and y in g for g in _LOOKALIKE)
        # a letter read twice or lost: only inside the word. One more or less
        # at an end is more likely a different name ("محسن" is not "حسن").
        long_, short = (b, k) if len(b) > len(k) else (k, b)
        return long_[1:] != short and long_[:-1] != short
    near = [k for k in v if ok(k)]
    if not near and len(b) >= 5:
        # two letters off, both look-alikes of the right ones ("الضغير" -> "الصغير")
        def alike(k):
            if len(k) != len(b):
                return False
            diff = [(p, q) for p, q in zip(b, k) if p != q]
            return len(diff) == 2 and all(any(p in g and q in g for g in _LOOKALIKE) for p, q in diff)
        near = [k for k in v if alike(k)]
    if len(near) != 1:
        return tok
    forms = v[near[0]]
    same_end = [f for f in forms if f[-1] == tok[-1]]
    return (same_end or forms)[0]


def _clean_name(t: str) -> str:
    """Card names: Latin look-alikes out ("محم1"), "عبد الرحيم" written as the
    card prints it ("عبدالرحيم"), a name split in two ("يو سف") joined."""
    from .textfix import fix_compound_names
    t = re.sub(r"[A-Za-z0-9]", "", t).strip()
    toks = t.split()
    out, i = [], 0
    while i < len(toks):
        a = toks[i]
        b = toks[i + 1] if i + 1 < len(toks) else ""
        if a == "عبد" and b:
            if b.startswith("ان") and not b.startswith("ال"):
                b = "ال" + b[2:]                       # "عبد انمنعم": ل read as ن
            out.append("عبد" + b); i += 2; continue
        if b and a + b in COMMON_NAMES and a not in COMMON_NAMES:
            out.append(a + b); i += 2; continue
        out.append(a); i += 1
    return " ".join(_correct(w, "names") for w in fix_compound_names(" ".join(out)).split())


def _clean_address(t: str) -> str:
    """Town and governorate names checked against the list of places."""
    from .textfix import _lev
    t = " ".join(_correct(w, "places") for w in t.split(" "))
    if "-" not in t:
        return t
    head, tail = t.rsplit("-", 1)
    g = tail.strip()
    names = set(GOVERNORATES.values()) | {"القاهره", "الجيزه", "المنيا", "اسيوط", "اسوان", "سوهاج", "قنا",
                                           "الفيوم", "بنى سويف", "الاقصر", "الدقهليه", "الشرقيه", "القليوبيه",
                                           "الغربيه", "المنوفيه", "البحيره", "الاسكندريه", "الاسماعيليه"}
    if g and g not in names:
        close = [n for n in names if abs(len(n) - len(g)) <= 1 and _lev(n, g) <= 1]
        if len(close) == 1:
            return f"{head.rstrip()} - {close[0]}"
    return t


def to_csv(cards: list[Card], path: str):
    import csv
    with open(path, "w", newline="", encoding="utf-8-sig") as f:      # utf-8-sig: Excel opens Arabic right
        w = csv.writer(f)
        w.writerow(HEADERS)
        for c in cards:
            w.writerow(c.row())
