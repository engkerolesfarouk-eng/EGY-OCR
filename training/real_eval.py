"""CER of the full pipeline on real images with hand-checked text.

python real_eval.py REC_MODEL.onnx [REC_MODEL2.onnx ...]
Books/memos: whole-page CER. Cards/plate: each ground-truth line against the
closest output line (other text on the card is ignored). Spaces don't count.
"""
import glob
import os
HERE = os.path.dirname(os.path.abspath(__file__))
import re
import sys
import unicodedata

sys.path.insert(0, HERE + "/..")
from arocr.pipeline import ArabicOcr  # noqa: E402

D = os.path.join(os.path.dirname(__file__), "real")


def norm(t):
    t = unicodedata.normalize("NFKC", t)
    t = re.sub(r"[ً-ْـ]", "", t)
    t = t.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا").replace("ى", "ي").replace("ه\n", "ة\n")
    return re.sub(r"\s+", "", t)


def strict_words(t):
    """Words as printed, diacritics aside but hamza forms kept ("يومأ" is not "يوماً")."""
    t = unicodedata.normalize("NFKC", t)
    t = re.sub(r"[ًٌٍَُِّْـ]", "", t)
    t = re.sub(r"[^\w\s]", " ", t)
    return t.split()


def edit(a, b):
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def run(model):
    ocr = ArabicOcr(rec_model=os.path.abspath(model)) if model != "default" else ArabicOcr()
    groups = {"book": [0, 0], "memo": [0, 0], "card": [0, 0], "plate": [0, 0]}
    detail = []
    for gt_path in sorted(glob.glob(os.path.join(D, "*.gt.txt"))):
        name = os.path.basename(gt_path)[:-7]
        img = [p for p in glob.glob(os.path.join(D, name + ".*")) if not p.endswith(".txt")][0]
        out = next(ocr.file(img)).text
        gt = open(gt_path, encoding="utf-8").read()
        g = next(k for k in groups if name.startswith(k))
        if g in ("book", "memo"):
            e, n = edit(norm(out), norm(gt)), len(norm(gt))
            gw = strict_words(gt)
            wer = edit(strict_words(out), gw) / max(1, len(gw))
            detail.append(f"[{name} WER {100*wer:.1f}%]")
        else:
            outs = [norm(l) for l in out.split("\n") if l.strip()]
            e = n = 0
            for line in gt.split("\n"):
                if not line.strip():
                    continue
                t = norm(line)
                e += min(edit(o, t) for o in outs) if outs else len(t)
                n += len(t)
        groups[g][0] += e; groups[g][1] += n
        detail.append(f"{name}:{100*e/max(1,n):.1f}")
    return {k: 100 * e / max(1, n) for k, (e, n) in groups.items()}, detail


if __name__ == "__main__":
    for m in sys.argv[1:]:
        res, det = run(m)
        print(os.path.basename(m), " ".join(f"{k}={v:.2f}%" for k, v in res.items()))
        print("   ", " ".join(det), flush=True)
