"""Turn the recognizer's visual-order output into logical (reading-order) text.

The PaddleOCR Arabic recognizer reads every line strictly left to right in pixel
order. For Arabic that yields the characters mirrored ("رمع" for "عمر"), but runs
of digits and Latin letters inside the line are already in their correct order,
because those are written left to right even inside Arabic text.

So the fix is not "reverse the whole line" (that turns "عمارة ١٢" into "عمارة ٢١"):
reverse the order of the pieces, keep each left-to-right run (digits, Latin, and
number punctuation between them) intact, and swap mirrored brackets.

PaddleOCR's own pred_reverse does almost this, but only treats ASCII [a-zA-Z0-9]
as left-to-right, so Arabic-Indic digits (٠-٩, ۰-۹) come out reversed. This module
treats them as numbers too.
"""
from __future__ import annotations

import re

ARABIC_LETTER = re.compile(r"[ء-يٱ-ۓۺ-ۿﭐ-﷿ﹰ-ﻼ]")

# Characters that read left to right even inside Arabic text.
_LTR_STRONG = r"A-Za-z0-9٠-٩۰-۹"
# Punctuation that belongs to a number or code when it sits between two such
# characters: 1990/05/03, 12.5, 10:30, KM-84, 50%.
_JOINERS = r"./:,\-%+*٫٬"
LTR_RUN = re.compile(rf"[{_LTR_STRONG}](?:[{_LTR_STRONG}]|[{_JOINERS}](?=[{_LTR_STRONG}]))*%?")

_RTL_HINT = re.compile(r"[\u0660-\u0669\u06F0-\u06F9\u060C\u061B\u061F«»]")
_MIRROR = str.maketrans("()[]{}<>«»", ")(][}{><»«")

NUMBER_LINE = re.compile(r"^[0-9\u0660-\u0669\u06F0-\u06F9\s/.\-:,\u066B\u066C]+$")
DIGITS_ONLY = re.compile(r"^[0-9٠-٩۰-۹/\-.]+$")


def has_arabic(text: str) -> bool:
    return bool(ARABIC_LETTER.search(text))


def visual_to_logical(visual: str) -> str:
    """Convert one visually ordered line (left to right in pixels) to reading order.

    Lines without Arabic letters are already in reading order and are returned
    unchanged.
    """
    # Lines with Arabic letters, or with only Arabic digits/punctuation (the tail
    # of a reference that wrapped: ".(٦ :٥"), are right-to-left. Latin-only
    # lines are already in reading order.
    if not (has_arabic(visual) or _RTL_HINT.search(visual)):
        return visual
    # A line that is only a number (an ID number printed in groups, a date)
    # reads left to right even in Arabic: keep the groups in order.
    if NUMBER_LINE.match(visual):
        return visual

    pieces: list[str] = []
    pos = 0
    for m in LTR_RUN.finditer(visual):
        # Everything before the run is right-to-left: one piece per character.
        pieces.extend(visual[pos:m.start()])
        pieces.append(m.group())
        pos = m.end()
    pieces.extend(visual[pos:])

    out = []
    for piece in reversed(pieces):
        # A single neutral/RTL character flips its mirrored glyph; LTR runs don't.
        out.append(piece.translate(_MIRROR) if len(piece) == 1 else piece)
    return "".join(out)


def is_digits_only(text: str) -> bool:
    return bool(DIGITS_ONLY.match(text.replace(" ", "")))


# The Arabic model also knows Persian/Urdu letters and digits and sometimes picks
# them for Egyptian text (ک for ك, ی for ي, ۱۲ for ١٢). They look alike but break
# search and comparisons, so fold them to the Arabic forms.
_EG = str.maketrans("کیۀ۰۱۲۳۴۵۶۷۸۹", "كيه٠١٢٣٤٥٦٧٨٩")


def normalize_egyptian(text: str) -> str:
    return text.translate(_EG)


def count_digits(text: str) -> int:
    return sum(ch.isdigit() for ch in text)


def count_letters(text: str) -> int:
    return len(ARABIC_LETTER.findall(text))


_CODE = re.compile(r"\b([A-Z]{2})([0-9OIl]{7})\b")


def fix_codes(text: str) -> str:
    """Card / serial numbers like "LE0976591": letters, then digits only.

    In the digit part O and I/l are misreads of 0 and 1 ("LEO976591").
    """
    def sub(m):
        tail = m.group(2)
        if sum(c.isdigit() for c in tail) < len(tail) - 2:
            return m.group(0)
        return m.group(1) + tail.replace("O", "0").replace("I", "1").replace("l", "1")
    return _CODE.sub(sub, text)


# ---------------------------------------------------------------- known phrases
# Text printed on Egyptian documents in calligraphy or logos that no font
# reproduces (the ID card header is a Thuluth logo): recognised by context.
KNOWN_PHRASES = [
    "بطاقة تحقيق الشخصية", "جمهورية مصر العربية", "وزارة الداخلية", "مصلحة الأحوال المدنية",
    "قطاع الأحوال المدنية", "رخصة قيادة خاصة", "رخصة تسيير", "شهادة ميلاد",
]
_COMPOUND_TAILS = ["الله", "الرحمن", "الرحيم", "الحميد", "العزيز", "المنعم", "الفتاح", "الستار", "الغفار",
                   "الحليم", "الرازق", "الباسط", "القادر", "الكريم", "اللطيف", "المجيد", "السلام", "الحكيم",
                   "الوهاب", "الجواد", "الهادي", "الرؤوف", "العظيم", "التواب", "النبي", "المسيح", "الملاك",
                   "الشهيد", "الصمد", "الغني", "المحسن", "الحي", "القوي", "الباقي", "العال", "الناصر",
                   "الرسول", "المطلب", "الخالق", "الجليل", "الودود", "المقصود", "الموجود", "الحفيظ"]
COMPOUND_NAMES = {"عبد" + t for t in _COMPOUND_TAILS} | {
    "أبوقرقاص", "أبوتيج", "أبوحمص", "أبوكبير", "أبوالمطامير", "أبوزعبل", "أبوقير", "أبوسمبل", "أبوصوير"}


def _lev(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def _bare(s: str) -> str:
    return s.replace(" ", "").replace("أ", "ا").replace("إ", "ا").replace("ى", "ي").replace("ة", "ه")


def match_phrase(text: str, max_ratio: float = 0.25) -> str | None:
    """The known phrase this line is a misreading of, if it is close enough."""
    t = _bare(text)
    if len(t) < 8:
        return None
    best = None
    for p in KNOWN_PHRASES:
        d = _lev(t, _bare(p))
        if d <= max_ratio * len(_bare(p)) and (best is None or d < best[0]):
            best = (d, p)
    return best[1] if best else None


def fix_compound_names(text: str) -> str:
    """"عبدالر حيم" / "عبدائر حيم" -> "عبدالرحيم", "أبوقر قاص" -> "أبوقرقاص"."""
    toks = text.split(" ")
    out, i = [], 0
    names = {_bare(n): n for n in COMPOUND_NAMES}
    while i < len(toks):
        t = toks[i]
        if _bare(t).startswith(("عبد", "ابو")) and t not in ("عبد", "ابو", "أبو"):
            for k in (3, 2, 1):                      # try joining with up to 2 following pieces
                if i + k > len(toks) or not all(ARABIC_LETTER.fullmatch(c) for c in "".join(toks[i:i + k])):
                    continue
                cand = _bare("".join(toks[i:i + k]))
                if len(cand) < 6:
                    continue
                hit = names.get(cand)
                if hit is None:
                    close = [n for b, n in names.items() if abs(len(b) - len(cand)) <= 1 and _lev(b, cand) <= 1]
                    hit = close[0] if len(close) == 1 else None
                if hit:
                    # keep the user's spelling of the first part (ابو / أبو)
                    if t.startswith("ابو") and hit.startswith("أبو"):
                        hit = "ابو" + hit[3:]
                    out.append(hit)
                    i += k
                    break
            else:
                out.append(t)
                i += 1
            continue
        out.append(t)
        i += 1
    return " ".join(out)


# ---------------------------------------------------------------- tanween
# "يوماً" printed with the two fatha strokes over the alef is read as "يومأ"
# (the strokes look like a hamza). A final hamza-on-alef is real only in a
# short list of words; anywhere else it is tanween.
_REAL_FINAL_HAMZA = set("""
بدأ قرأ نشأ خطأ ملجأ مبدأ منشأ مخبأ هدأ ظمأ امتلأ تبوأ يقرأ يبدأ ينشأ اقرأ ابدأ مرفأ متكأ
موطأ يملأ ملأ أنبأ نبأ سبأ لجأ يلجأ تهيأ هيأ تنبأ يتنبأ امرأ مقرأ مبتدأ يهدأ تبدأ نقرأ أقرأ أبدأ
اتكأ يتكأ ابتدأ يبتدئ برأ أبرأ صدأ دفأ كفأ تكافأ تقرأ نبدأ ننشأ تنشأ أنشأ يخطأ أخطأ مخطأ ملأ رشأ
""".split())


def fix_tanween(text: str) -> str:
    text = text.replace("أً", "اً")
    # alef read twice: once as the letter, once as the tanween strokes ("تغييرااً", "ميلاداأ")
    text = re.sub(r"ا[اأ]ً?(?![\u0621-\u064a])", "اً", text)

    def word(m):
        w = m.group(0)
        bare = w.rstrip("ًٌٍَُِّْ")
        if not bare.endswith("أ") or len(bare) < 3:
            return w
        stem = bare
        for p in ("وال", "فال", "بال", "كال", "لل", "ال", "و", "ف", "ب", "ل", "ك"):
            if stem.startswith(p) and stem[len(p):] in _REAL_FINAL_HAMZA:
                return w
        if stem in _REAL_FINAL_HAMZA:
            return w
        return bare[:-1] + "اً"
    return re.sub(r"[ء-يً-ْ]+", word, text)


def fix_shadda_tanween(text: str) -> str:
    """"شاهدّا" -> "شاهداً": the tanween strokes read as a shadda on the last
    letter. A real shadda before a final alef is rare (أمّا، إمّا، لمّا)."""
    keep = {"أمّا", "إمّا", "لمّا", "امّا", "اما"}

    def word(m):
        w = m.group(0)
        if w in keep or not w.endswith("ّا"):
            return w
        return w[:-2] + "اً"
    return re.sub(r"[\u0621-\u064a\u064b-\u0652]+", word, text)


def fix_allah(text: str) -> str:
    """The book reader can merge the two lams of the "الله" ligature (the word
    is drawn as one narrow glyph): "اله" / "باله" alone are "الله" / "بالله".
    Only for running text read by that model; "إله" (with hamza) is untouched."""
    return re.sub(r"(?<![\u0621-\u064a])([وفب]?)اله(?![\u0621-\u064a])", r"\1الله", text)
