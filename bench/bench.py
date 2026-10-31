"""Compare OCR output against hand-checked ground truth.

Folder layout:
    samples/<name>.png|jpg|pdf     input
    samples/<name>.gt.txt          correct text, one line per printed line (you check this by hand)
    results/<engine>/<name>.txt    output of each engine (arocr writes these; paste other engines' output here)

Usage:
    python bench/bench.py samples results/arocr results/claude ...

Reports per file and overall: character error rate (CER), word error rate (WER),
and how many digit sequences (IDs, dates, amounts) came out exactly right.
"""
from __future__ import annotations

import os
import re
import sys
import unicodedata

DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"[ً-ْـ]", "", text)  # tashkeel and tatweel don't count
    text = text.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا")
    text = re.sub(r"^--- page \d+ ---$", "", text, flags=re.M)
    return re.sub(r"\s+", " ", text).strip()


def edit_distance(a, b) -> int:
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def numbers(text: str) -> list[str]:
    return re.findall(r"\d[\d/.,:-]*\d|\d", text.translate(DIGITS))


def score(gt: str, out: str) -> dict:
    g, o = normalize(gt), normalize(out)
    gw, ow = g.split(), o.split()
    gn, on = numbers(g), numbers(o)
    found = sum(1 for n in gn if n in on)
    return {
        "cer": edit_distance(g, o) / max(1, len(g)),
        "wer": edit_distance(gw, ow) / max(1, len(gw)),
        "nums": (found, len(gn)),
    }


def main(samples: str, engines: list[str]):
    names = sorted(f[:-7] for f in os.listdir(samples) if f.endswith(".gt.txt"))
    totals = {e: {"cer": [], "wer": [], "found": 0, "nums": 0} for e in engines}
    print(f"{'file':28}" + "".join(f"{os.path.basename(e):>26}" for e in engines))
    for n in names:
        gt = open(os.path.join(samples, n + ".gt.txt"), encoding="utf-8").read()
        row = f"{n[:27]:28}"
        for e in engines:
            p = os.path.join(e, n + ".txt")
            if not os.path.exists(p):
                p = os.path.join(e, n + ".ocr.txt")
            if not os.path.exists(p):
                row += f"{'(missing)':>26}"
                continue
            s = score(gt, open(p, encoding="utf-8").read())
            t = totals[e]
            t["cer"].append(s["cer"]); t["wer"].append(s["wer"])
            t["found"] += s["nums"][0]; t["nums"] += s["nums"][1]
            row += f"{'CER %.1f%% WER %.1f%% #%d/%d' % (100*s['cer'], 100*s['wer'], *s['nums']):>26}"
        print(row)
    print("-" * (28 + 26 * len(engines)))
    row = f"{'AVERAGE':28}"
    for e in engines:
        t = totals[e]
        if t["cer"]:
            row += f"{'CER %.1f%% WER %.1f%% #%d/%d' % (100*sum(t['cer'])/len(t['cer']), 100*sum(t['wer'])/len(t['wer']), t['found'], t['nums']):>26}"
        else:
            row += f"{'-':>26}"
    print(row)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2:])
