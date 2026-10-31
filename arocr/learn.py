"""Corrections you teach the program.

In the text pane, select a word the program read wrong, right-click, and
choose «علّم البرنامج التصحيح ده». The pair (wrong -> right) is saved, and from
then on every reading of any file gets that word corrected. Only words you
teach this way are learned; ordinary edits are not.

Everything learned is in one plain text file, one rule per line:

    يكنه → يمكنه
    # lines starting with # are ignored

Open it with Notepad to review, change or delete rules; changes are picked up
the next time a page is read. Location: %LOCALAPPDATA%\\ArabicOCR\\corrections.txt
"""
from __future__ import annotations

import os
import re

ARROW = "→"
_cache = {"mtime": None, "rules": [], "pattern": None, "map": {}}


def path() -> str:
    base = os.environ.get("LOCALAPPDATA")
    d = os.path.join(base, "ArabicOCR") if base else os.path.join(os.path.expanduser("~"), ".config", "arocr")
    os.makedirs(d, exist_ok=True)
    p = os.path.join(d, "corrections.txt")
    if not os.path.exists(p):
        with open(p, "w", encoding="utf-8") as f:
            f.write("# تصحيحات علّمتها للبرنامج: الغلط → الصح (سطر لكل تصحيح)\n"
                    "# تقدر تعدّل أو تمسح أي سطر. السطور اللي بتبدأ بـ # بتتجاهل.\n")
    return p


def rules() -> list[tuple[str, str]]:
    p = path()
    m = os.path.getmtime(p)
    if _cache["mtime"] != m:
        out = []
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or ARROW not in line:
                    continue
                wrong, right = (s.strip() for s in line.split(ARROW, 1))
                if wrong and wrong != right:
                    out.append((wrong, right))
        # longest first, so a phrase wins over a word inside it
        out.sort(key=lambda r: -len(r[0]))
        _cache.update(mtime=m, rules=out, map=dict(out),
                      pattern=re.compile("|".join(rf"(?<!\w){re.escape(w)}(?!\w)" for w, _ in out)) if out else None)
    return _cache["rules"]


def add(wrong: str, right: str) -> None:
    wrong, right = wrong.strip(), right.strip()
    if not wrong or wrong == right or "\n" in wrong or "\n" in right:
        return
    p = path()
    with open(p, encoding="utf-8") as f:
        lines = f.read().splitlines()
    # a new teaching of the same wrong word replaces the old one
    keep = [l for l in lines if not (ARROW in l and not l.lstrip().startswith("#")
                                     and l.split(ARROW, 1)[0].strip() == wrong)]
    keep.append(f"{wrong} {ARROW} {right}")
    with open(p, "w", encoding="utf-8") as f:
        f.write("\n".join(keep) + "\n")
    _cache["mtime"] = None            # reload on next use


def apply(text: str) -> str:
    """The text with every taught correction applied (whole words only)."""
    rules()
    pat = _cache["pattern"]
    if not pat or not text:
        return text
    m = _cache["map"]
    return pat.sub(lambda mo: m.get(mo.group(0), mo.group(0)), text)


def apply_page(page) -> None:
    for ln in page.lines:
        ln.text = apply(ln.text)
    for c in getattr(page, "cards", []) or []:
        c.name = apply(c.name)
        c.first_name = apply(c.first_name)
        c.address = [apply(a) for a in c.address]
