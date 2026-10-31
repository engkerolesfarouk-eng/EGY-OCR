"""Fill numbers and Latin codes the Arabic model missed, using a Latin model.

On blurred scans the Arabic recognizer sometimes drops a Latin code or an amount
written in Western digits entirely ("PO-2026-118", "450,000"): it outputs nothing
for that stretch of the line. The general PP-OCR model reads those well but turns
Arabic into nonsense. So we read each line with both, and copy over only the
confident Latin/digit runs that sit where the Arabic reading has nothing.
"""
from __future__ import annotations

import re

from .recognize import RecResult

_RUN_CHARS = re.compile(r"[A-Za-z0-9.,:/\-%+#]")
_STRONG = re.compile(r"[A-Za-z0-9]")
_PUNCT = set("()!:?«»")
_PUNCT_MAP = {"?": "؟"}


def _gaps(r: RecResult, factor: float = 2.5):
    """Stretches of the line where the Arabic reading has no characters."""
    xs = sorted(x for ch, x in zip(r.text, r.char_x) if not ch.isspace())
    if len(xs) < 2:
        return [(0.0, 1.0)]
    steps = sorted(b - a for a, b in zip(xs, xs[1:]))
    unit = steps[len(steps) // 2] or 0.01
    gaps = []
    if xs[0] > factor * unit:
        gaps.append((0.0, xs[0] - 0.5 * unit))
    gaps += [(a + 0.5 * unit, b - 0.5 * unit) for a, b in zip(xs, xs[1:]) if b - a > factor * unit]
    if 1.0 - xs[-1] > factor * unit:
        gaps.append((xs[-1] + 0.5 * unit, 1.0))
    return gaps


def fill_from_latin(arabic: RecResult, latin: RecResult,
                    min_len: int = 3, min_conf: float = 0.6) -> RecResult:
    if not latin.text or not latin.char_conf:
        return arabic
    chars = list(zip(arabic.text, arabic.char_x, arabic.char_conf))
    changed = False
    for g0, g1 in _gaps(arabic):
        picked = [(ch, x, c) for ch, x, c in zip(latin.text, latin.char_x, latin.char_conf)
                  if g0 <= x <= g1 and _RUN_CHARS.match(ch)]
        run = "".join(p[0] for p in picked).strip(".,:-/")
        if (len(run) < min_len or sum(bool(_STRONG.match(ch)) for ch in run) < 2
                or min(p[2] for p in picked) < min_conf):
            # No code or number here, but punctuation is the same in every
            # script, and the Arabic model often drops it: ( ) ! : ? « »
            punct = [(_PUNCT_MAP.get(ch, ch), x, c) for ch, x, c in
                     zip(latin.text, latin.char_x, latin.char_conf)
                     if g0 <= x <= g1 and ch in _PUNCT and c >= 0.85]
            others = [ch for ch, x in zip(latin.text, latin.char_x)
                      if g0 <= x <= g1 and ch not in _PUNCT and not ch.isspace()]
            if punct and len(others) <= len(punct):
                chars = sorted(chars + punct, key=lambda c: c[1])
                changed = True
            continue
        new = [(" ", picked[0][1] - 1e-4, 1.0)] + [p for p in picked] + [(" ", picked[-1][1] + 1e-4, 1.0)]
        chars = sorted(chars + new, key=lambda c: c[1])
        changed = True
    if not changed:
        return arabic
    out = []
    for c in chars:  # collapse doubled spaces
        if c[0] == " " and (not out or out[-1][0] == " "):
            continue
        out.append(c)
    while out and out[-1][0] == " ":
        out.pop()
    return RecResult("".join(c[0] for c in out), arabic.confidence,
                     [c[1] for c in out], [c[2] for c in out])
