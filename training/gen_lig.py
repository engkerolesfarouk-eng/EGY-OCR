"""Book lines in classic naskh fonts with stacked ligatures (يمكن، الحكمة،
المحسوس، بالله، تجديد) and kashida stretching (كـثـرة، الـصلاة), the way
older Arabic religious books are typeset.

python gen_lig.py OUT_DIR N SEED
Labels in the model's visual order, without the kashida.
"""
import os
HERE = os.path.dirname(os.path.abspath(__file__))
import random
import re
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from gen import degrade, logical_to_visual, render  # noqa: E402
from gen_tashkeel import load, partial  # noqa: E402

sys.path.insert(0, HERE + "/..")
from arocr.textfix import visual_to_logical  # noqa: E402

T = HERE + "/"
LIG_FONTS = ([T + "fonts/trado.ttf"] * 5 + [T + "fonts/tradbdo.ttf"] * 2 + [T + "fonts/arabtype.ttf"] * 2
             + [T + "fonts/simpo.ttf", T + "fonts/simpbdo.ttf", T + "fonts/majalla.ttf",
                T + "fonts_deco/Amiri-Regular.ttf", T + "fonts_deco/Amiri-Regular.ttf", T + "fonts_deco/Amiri-Bold.ttf",
                T + "fonts_deco/ScheherazadeNew-Regular.ttf", T + "fonts_deco/Lateef-Regular.ttf"])

# words of religious and literary prose, rich in the ligatures above
EXTRA = """الله بالله لله الإلهية الإلهي المسيح يسوع الروح القدس الصلاة صلاتنا بالصلاة يمكن لا يمكن يمكنه
يمكننا الممكن الحكمة المحسوس المحسوسة والمحسوس الملحوظ المنظور يحمل يحملها تحمل تجديد وتجديد
التمييز تمييز مما أكثر مما ممن لما لها لهم لنا الحياة الأبدية الإنسان الإنسانية النفس الجسد
المحبة محبة يحب نحب الخطية الخطاة التوبة الإيمان الرجاء النعمة الكنيسة الآباء القديسين المجد
الملكوت السماء السموات الأرض العالم الكلمة الإنجيل الرب ربنا إلهنا خلاص الخلاص يخلص المخلص
محمد محمود المحبة ملكوت مملوء ممتلئ مجيء تجسد المتجسد المجد ممجد يمجد التمجيد""".split()


def kashida(word, r):
    """Stretch a word the way justified books do: a tatweel between two joining letters."""
    if len(word) < 3 or r.random() > 0.5:
        return word
    non_join = set("اأإآدذرزوؤءة")
    spots = [i for i in range(1, len(word)) if word[i - 1] not in non_join and "ء" <= word[i - 1] <= "ي"
             and "ء" <= word[i] <= "ي"]
    if not spots:
        return word
    i = r.choice(spots)
    return word[:i] + "ـ" * r.choice([1, 1, 2, 3]) + word[i:]


def main(out, n, seed):
    os.makedirs(out, exist_ok=True)
    r = random.Random(seed); rng = np.random.default_rng(seed)
    rows, p = load()
    labels, i = [], 0
    while i < n:
        k = r.randint(3, 10)
        ws = []
        for _ in range(k):
            if r.random() < 0.35:
                ws.append(r.choice(EXTRA))
            else:
                v, u, _ = rows[rng.choice(len(rows), p=p)]
                ws.append(partial(v, r) if r.random() < 0.3 else u)
        if r.random() < 0.15:
            ws[-1] += r.choice([".", "،", "؛", ":", "؟"])
        label = " ".join(ws)
        # word gaps as wide as in justified books (the naskh fonts' own space is
        # narrow, and a model trained on it learns to drop spaces)
        shown = ""
        for k, w in enumerate(ws):
            if k:
                shown += " " * r.choice([1, 2, 2, 3])
            shown += kashida(w, r) if r.random() < 0.35 else w
        vis = logical_to_visual(label)
        if visual_to_logical(vis) != label:
            continue
        img = render(shown, r.choice(LIG_FONTS), r.randint(24, 46), r)
        if img is None or img.shape[1] > 2000:
            continue
        img = degrade(img, r)
        name = f"l{i:06d}.png"
        cv2.imwrite(os.path.join(out, name), img)
        labels.append(f"{name}\t{vis}\t{label}")
        i += 1
    open(os.path.join(out, "labels.tsv"), "w", encoding="utf-8").write("\n".join(labels) + "\n")


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]))
_ = re
