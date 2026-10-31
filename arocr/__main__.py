"""Command line: python -m arocr input.pdf [-o out.pdf] [--txt out.txt]"""
from __future__ import annotations

import argparse
import os
import sys
import time

from .pdfout import SearchablePdf
from .pipeline import ArabicOcr


def main(argv=None):
    ap = argparse.ArgumentParser(prog="arocr", description="Offline Arabic OCR -> searchable PDF + text")
    ap.add_argument("inputs", nargs="+", help="PDF or image files, or folders")
    ap.add_argument("-o", "--out-dir", default=None, help="output folder (default: next to each input)")
    ap.add_argument("--clahe", action="store_true", help="contrast enhancement (helps uneven-light phone photos, e.g. ID cards)")
    ap.add_argument("--no-pdf", action="store_true", help="text only, no searchable PDF")
    ap.add_argument("--docx", action="store_true", help="also write an editable Word document")
    ap.add_argument("--html", action="store_true", help="also write the text as an HTML page")
    ap.add_argument("--docx-scan", action="store_true", help="Word that looks exactly like the scan (heavier)")
    ap.add_argument("--id-card", action="store_true", help="Egyptian national ID cards: card fields only, text outside the card dropped (implies --cards)")
    ap.add_argument("--cards", action="store_true", help="ID cards: also write their fields to <name>.cards.csv (opens in Excel)")
    ap.add_argument("--scale", type=float, default=2.5, help="PDF render scale (2.5 = ~180 DPI)")
    args = ap.parse_args(argv)

    # a folder means every PDF / image in it
    from .pipeline import SUPPORTED_EXT
    inputs = []
    for p in args.inputs:
        if os.path.isdir(p):
            inputs += sorted(os.path.join(p, f) for f in os.listdir(p)
                             if os.path.splitext(f)[1].lower() in SUPPORTED_EXT)
        elif os.path.isfile(p):
            inputs.append(p)
        else:
            print(f"مش لاقي الملف: {p}", file=sys.stderr)
    if not inputs:
        sys.exit("مفيش ملفات أقراها. اكتب مسار ملف PDF أو صورة، أو فولدر.")
    args.inputs = inputs

    t_load = time.perf_counter()
    if args.id_card:
        args.cards = True
    ocr = ArabicOcr(use_clahe=args.clahe, card_mode=args.id_card)
    print(f"models loaded in {time.perf_counter() - t_load:.1f}s", file=sys.stderr)

    for path in args.inputs:
        base = os.path.splitext(os.path.basename(path))[0]
        out_dir = args.out_dir or os.path.dirname(os.path.abspath(path))
        os.makedirs(out_dir, exist_ok=True)
        pdf = None if args.no_pdf else SearchablePdf(os.path.join(out_dir, base + ".ocr.pdf"))
        word = None
        if args.docx or args.docx_scan:
            from .wordout import WordDocument
            word = WordDocument(os.path.join(out_dir, base + ".ocr.docx"),
                                layout="scan" if args.docx_scan else "text")
        texts, cards, all_pages = [], [], []
        t0 = time.perf_counter()
        for n, page in enumerate(ocr.file(path, scale=args.scale,
                                          progress=lambda i, k: print(f"  page {i}/{k}", file=sys.stderr)), 1):
            texts.append(f"--- page {n} ---\n{page.text}")
            if pdf:
                pdf.add_page(page.image, page.lines, page.width_pt, page.height_pt)
            if word:
                word.add_page(page)
            cards.extend(page.cards)
            if args.html:
                all_pages.append(page)
        if pdf:
            pdf.save()
        if word:
            word.save()
        if args.html and all_pages:
            from .htmlout import write_html
            write_html(os.path.join(out_dir, base + ".ocr.html"), all_pages, title=base)
        if args.cards and cards:
            from .cards import to_csv
            to_csv(cards, os.path.join(out_dir, base + ".cards.csv"))
            print(f"  {len(cards)} card(s) -> {base}.cards.csv", file=sys.stderr)
        with open(os.path.join(out_dir, base + ".ocr.txt"), "w", encoding="utf-8") as f:
            f.write("\n\n".join(texts) + "\n")
        print(f"{path}: {len(texts)} page(s) in {time.perf_counter() - t0:.1f}s", file=sys.stderr)


if __name__ == "__main__":
    main()
