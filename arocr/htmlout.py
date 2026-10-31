"""Write the text as one HTML page: right to left, paragraphs and headings as
read from the layout, a marker per page. Opens in any browser, easy to read
on a phone, and to paste into a website or a blog."""
from __future__ import annotations

import html

from .wordout import _paragraphs

_CSS = """
:root { color-scheme: light dark; }
body { font-family: "Noto Naskh Arabic", "Traditional Arabic", "Arial", sans-serif;
       font-size: 20px; line-height: 1.9; max-width: 46em; margin: 2em auto; padding: 0 1em;
       background: #fdfcf8; color: #222; }
@media (prefers-color-scheme: dark) { body { background: #1c1c1c; color: #e6e6e6; } }
h1 { font-size: 1.5em; } h2.page { font-size: .8em; color: #888; border-top: 1px solid #ccc;
       padding-top: .6em; margin-top: 2.5em; font-weight: normal; }
p { margin: 0 0 .9em; text-align: justify; } p.lines { text-align: start; }
.h { font-weight: bold; text-align: center; font-size: 1.25em; }
.ltr { direction: ltr; text-align: left; }
"""


def _has_arabic(t: str) -> bool:
    return any("؀" <= c <= "ۿ" for c in t)


def write_html(path: str, pages, edits: dict | None = None, title: str = "", progress=None):
    edits = edits or {}
    out = ['<!doctype html><html lang="ar" dir="rtl"><head><meta charset="utf-8">',
           '<meta name="viewport" content="width=device-width, initial-scale=1">',
           f"<title>{html.escape(title or 'OCR')}</title><style>{_CSS}</style></head><body>"]
    if title:
        out.append(f"<h1>{html.escape(title)}</h1>")
    for i, page in enumerate(pages):
        if len(pages) > 1:
            out.append(f'<h2 class="page">صفحة {i + 1}</h2>')
        edited = edits.get(i)
        sy = page.height_pt / page.image.shape[0] if page.image is not None else 1.0
        if getattr(page, "fields_only", False) and edited is None:
            text, heights = page.text, []
        elif edited is None or edited == page.text:
            text, heights = _paragraphs(page, sy)
        else:
            text, heights = edited, []
        hs = sorted(heights) or [0]
        body_h = hs[len(hs) // 2]
        for k, block in enumerate(b for b in text.split("\n\n") if b.strip()):
            lines = [l.strip() for l in block.split("\n") if l.strip()]
            lens = sorted(len(l) for l in lines)
            prose = not page.cards and len(lines) >= 2 and lens[len(lens) // 2] >= 35
            heading = k < len(heights) and len(lines) <= 2 and body_h and heights[k] > 1.35 * body_h
            cls = "h" if heading else ("" if prose else "lines")
            if not any(_has_arabic(l) for l in lines):
                cls += " ltr"
            body = " ".join(html.escape(l) for l in lines) if prose else "<br>".join(html.escape(l) for l in lines)
            out.append(f'<p class="{cls.strip()}">{body}</p>' if cls.strip() else f"<p>{body}</p>")
        if progress:
            progress(i + 1, len(pages))
    out.append("</body></html>")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(out))

