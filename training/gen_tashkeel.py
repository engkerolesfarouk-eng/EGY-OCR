"""Book lines with the partial diacritics of modern Arabic print (shadda,
tanween, damma on passives, fatha/kasra here and there) and quotes «» “”.
python gen_tashkeel.py OUT_DIR N SEED"""
import os, random, re, sqlite3, sys
HERE = os.path.dirname(os.path.abspath(__file__))
import cv2, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from gen import AR, FONTS, logical_to_visual, render, degrade  # noqa: E402
sys.path.insert(0, HERE + "/..")
from arocr.textfix import has_arabic, visual_to_logical  # noqa: E402

MARKS = "ًٌٍَُِّْ"
BOOK_FONTS = [f for f in FONTS if any(k in f.lower() for k in ("trad", "simp", "arabtype", "majalla", "times", "arial", "tahoma"))] or FONTS


def load():
    c = sqlite3.connect(os.path.join(AR, "wordfreq.sqlite"))
    rows = c.execute("select vocalized, unvocalized, freq from wordfreq where length(unvocalized) > 1").fetchall()
    rows = [(v, u, f) for v, u, f in rows if re.fullmatch(r"[ء-ي]+", u) and v]
    p = np.array([f for _, _, f in rows], float) ** 0.5
    return rows, p / p.sum()


def partial(voc: str, r: random.Random) -> str:
    out = []
    for ch in voc:
        if ch in MARKS:
            keep = {"ّ": 0.75, "ً": 0.8, "ٌ": 0.4, "ٍ": 0.4, "ُ": 0.18, "َ": 0.08, "ِ": 0.1, "ْ": 0.03}[ch]
            if r.random() < keep:
                out.append(ch)
        else:
            out.append(ch)
    s = "".join(out)
    s = re.sub(r"([ّ])([َُِ])", r"\1", s) if r.random() < 0.5 else s
    return s


def line(rows, p, rng, r):
    n = r.randint(3, 11)
    idx = rng.choice(len(rows), n, p=p)
    ws = []
    for i in idx:
        v, u, _ = rows[i]
        ws.append(partial(v, r) if r.random() < 0.55 else u)
    if r.random() < 0.15:
        ws[-1] = ws[-1] + "ًا" if not ws[-1].endswith("ا") else ws[-1]
    s = " ".join(ws)
    k = r.random()
    if k < 0.15:
        s = "«" + s + "»"
    elif k < 0.22:
        s = "“" + s + "”"
    elif k < 0.3:
        s = s + r.choice([".", "،", "؟", "!", "…", ":"])
    return s


def main(out, n, seed):
    os.makedirs(out, exist_ok=True)
    r = random.Random(seed); rng = np.random.default_rng(seed)
    rows, p = load()
    labels, i = [], 0
    while i < n:
        t = line(rows, p, rng, r)
        vis = logical_to_visual(t)
        if visual_to_logical(vis) != t:
            continue
        img = render(t, r.choice(BOOK_FONTS) if r.random() < 0.8 else r.choice(FONTS), r.randint(22, 44), r)
        if img is None or img.shape[1] > 2000:
            continue
        img = degrade(img, r)
        name = f"t{i:06d}.png"
        cv2.imwrite(os.path.join(out, name), img)
        labels.append(f"{name}\t{vis}\t{t}")
        i += 1
    open(os.path.join(out, "labels.tsv"), "w", encoding="utf-8").write("\n".join(labels) + "\n")


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]))
