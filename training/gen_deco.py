"""Synthetic lines for headings, cards and plates: decorative and heavy fonts,
coloured / patterned backgrounds, embossed letters, phone-camera blur.

python gen_deco.py OUT_DIR N SEED
Labels in the model's visual order, like gen.py.
"""
from __future__ import annotations

import glob
import os
HERE = os.path.dirname(os.path.abspath(__file__))
import random
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(__file__))
from gen import Corpus, logical_to_visual  # noqa: E402

sys.path.insert(0, HERE + "/..")
from arocr.textfix import has_arabic, visual_to_logical  # noqa: E402

DECO = sorted(glob.glob(HERE + "/fonts_deco/*.ttf"))
PLAIN = sorted(glob.glob(HERE + "/fonts/*.[tT][tT][fF]"))
IND = str.maketrans("0123456789", "٠١٢٣٤٥٦٧٨٩")

PHRASES = [
    "جمهورية مصر العربية", "مصر", "بطاقة تحقيق الشخصية", "وزارة الداخلية", "مصلحة الأحوال المدنية",
    "قطاع الأحوال المدنية", "رخصة قيادة", "رخصة تسيير", "شهادة ميلاد", "شهادة تخرج", "جامعة القاهرة",
    "جامعة المنيا", "جامعة أسيوط", "جامعة عين شمس", "محافظة القاهرة", "محافظة الجيزة", "وزارة التربية والتعليم",
    "الإدارة العامة للمرور", "مرور القاهرة", "ملاكي", "أجرة", "نقل", "شرطة", "دبلوماسي", "تحت الطلب",
    "بسم الله الرحمن الرحيم", "شهادة تقدير", "الشركة المصرية", "شركة النيل", "إدارة نظم المعلومات",
]
FIRST = ["محمد", "أحمد", "محمود", "مصطفى", "علي", "على", "حسن", "حسين", "إبراهيم", "ابراهيم", "عبدالرحيم",
         "عبدالرحمن", "عبدالحميد", "عبدالله", "خميس", "رجب", "شعبان", "رمضان", "سيد", "السيد", "طه", "ياسر",
         "إسلام", "اسلام", "أيمن", "ايمن", "كيرلس", "مينا", "جرجس", "بيشوي", "فاروق", "راتب", "صلاح", "نعمان",
         "محسن", "عادل", "هاني", "وليد", "خالد", "عمرو", "كريم", "يوسف", "مريم", "فاطمة", "سعاد", "نادية"]
PLACES = ["القاهرة", "القاهره", "الجيزة", "الإسكندرية", "المنيا", "أسيوط", "اسيوط", "سوهاج", "قنا", "الأقصر",
          "أسوان", "الفيوم", "بني سويف", "القليوبية", "الشرقية", "الدقهلية", "الغربية", "المنوفية", "البحيرة",
          "دمياط", "بورسعيد", "الإسماعيلية", "السويس", "مطروح", "أبوقرقاص", "ملوى", "مغاغة", "صدفا", "بهتيم",
          "شبرا الخيمه", "عين شمس", "مدينة نصر", "الروضة", "المعادي", "حلوان", "السنبلاوين", "ديروط"]
PLATE_LETTERS = list("أبجدرسصطعفقلمنهوى")


def text_sample(c: Corpus, r: random.Random) -> str:
    k = r.random()
    if k < 0.25:
        return r.choice(PHRASES)
    if k < 0.45:
        return " ".join(r.choice(FIRST) for _ in range(r.randint(1, 4)))
    if k < 0.58:
        p = r.choice(PLACES)
        return r.choice([p, f"مركز {r.choice(PLACES)} - {p}", f"{r.randint(1, 120)} ش {r.choice(FIRST)} {r.choice(FIRST)}".translate(IND)
                         if r.random() < .7 else f"{r.choice(PLACES)} - {p}"])
    if k < 0.70:   # plate: letters spaced + digits
        letters = " ".join(r.sample(PLATE_LETTERS, r.randint(2, 3)))
        digits = "".join(str(r.randint(0, 9)) for _ in range(r.randint(3, 4))).translate(IND)
        return r.choice([letters, digits, f"{letters} {digits}"])
    return c.snippet(r.randint(1, 4))


def background(h: int, w: int, r: random.Random) -> np.ndarray:
    k = r.random()
    lvl = r.randint(165, 250)
    base = np.clip(np.array([lvl + r.randint(-35, 20) for _ in range(3)], np.float32), 0, 255)
    if k < 0.2:    # plate blue / card blue
        base = np.array([r.randint(170, 230), r.randint(110, 170), r.randint(20, 80)], np.float32)
    img = np.ones((h, w, 3), np.float32) * base
    if r.random() < 0.6:   # gradient (light falloff)
        g = np.linspace(r.uniform(0.75, 1.0), r.uniform(0.85, 1.1), w)[None, :, None]
        img *= g
    if r.random() < 0.5:   # guilloche-like fine pattern
        yy, xx = np.mgrid[0:h, 0:w]
        f = r.uniform(0.15, 0.6)
        pat = np.sin(xx * f + np.sin(yy * f * 0.7) * r.uniform(1, 4)) * r.uniform(5, 18)
        img += pat[..., None] * np.array([r.uniform(0.3, 1) for _ in range(3)])
    if r.random() < 0.3:   # blotches (pyramids / stains)
        for _ in range(r.randint(1, 4)):
            cv2.circle(img, (r.randint(0, w), r.randint(0, h)), r.randint(h // 2, h * 2),
                       [r.randint(120, 230) for _ in range(3)], -1)
        img = cv2.GaussianBlur(img, (0, 0), h / 3)
    return img


def render(text: str, font_path: str, size: int, r: random.Random) -> np.ndarray | None:
    try:
        font = ImageFont.truetype(font_path, size)
    except OSError:
        return None
    probe = ImageDraw.Draw(Image.new("L", (1, 1)))
    try:
        l, t, rr, b = probe.textbbox((0, 0), text, font=font, direction="rtl", language="ar")
    except Exception:  # noqa: BLE001
        return None
    w, h = rr - l, b - t
    if w <= 4 or h <= 4:
        return None
    sp = 0
    padx, pady = r.randint(3, 16), r.randint(3, 12)
    W, Hh = w + 2 * padx, h + 2 * pady
    mask = Image.new("L", (W, Hh), 0)
    ImageDraw.Draw(mask).text((padx - l, pady - t), text, font=font, fill=255,
                              direction="rtl", language="ar", stroke_width=sp)
    m = np.array(mask).astype(np.float32) / 255.0
    if r.random() < 0.3:   # heavier strokes (plates, headings)
        m = cv2.dilate(m, np.ones((2, 2), np.uint8), iterations=r.randint(1, 2))
    bg = background(Hh, W, r)
    ink = np.array([r.randint(0, 70), r.randint(0, 90), r.randint(0, 70)], np.float32)
    if r.random() < 0.15:   # green/olive heading ink (card header)
        ink = np.array([r.randint(40, 90), r.randint(90, 140), r.randint(60, 110)], np.float32)
    out = bg * (1 - m[..., None]) + ink * m[..., None]
    if r.random() < 0.3:   # emboss: light edge + shadow
        dx, dy = r.choice([-2, -1, 1, 2]), r.choice([1, 2, 3])
        sh = np.roll(np.roll(m, dy, 0), dx, 1)
        out = out * (1 - 0.5 * np.clip(sh - m, 0, 1)[..., None])
        hl = np.roll(np.roll(m, -dy, 0), -dx, 1)
        out = out + 80 * np.clip(hl - m, 0, 1)[..., None]
    return np.clip(out, 0, 255).astype(np.uint8)


def degrade(img: np.ndarray, r: random.Random) -> np.ndarray:
    h, w = img.shape[:2]
    if r.random() < 0.4:   # perspective / skew of a phone photo
        d = h * r.uniform(0.02, 0.12)
        src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        dst = src + np.float32([[r.uniform(-d, d), r.uniform(-d, d)] for _ in range(4)])
        img = cv2.warpPerspective(img, cv2.getPerspectiveTransform(src, dst), (w, h), borderMode=cv2.BORDER_REPLICATE)
    if r.random() < 0.6:
        s = r.uniform(0.35, 0.85)
        img = cv2.resize(cv2.resize(img, (max(8, int(w * s)), max(8, int(h * s))), interpolation=cv2.INTER_AREA),
                         (w, h), interpolation=cv2.INTER_LINEAR)
    if r.random() < 0.5:
        img = cv2.GaussianBlur(img, (0, 0), r.uniform(0.4, 1.4))
    img = img.astype(np.float32)
    if r.random() < 0.4:   # glare
        cx, cy = r.randint(0, w), r.randint(0, h)
        yy, xx = np.mgrid[0:h, 0:w]
        g = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * (h * r.uniform(0.5, 2)) ** 2))
        img += g[..., None] * r.uniform(30, 90)
    img += np.random.default_rng(r.randint(0, 1 << 30)).normal(0, r.uniform(2, 12), img.shape)
    img = np.clip(img, 0, 255).astype(np.uint8)
    if r.random() < 0.6:
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, r.randint(30, 85)])
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return img


def main(out_dir: str, n: int, seed: int):
    os.makedirs(out_dir, exist_ok=True)
    c = Corpus(seed)
    r = random.Random(seed + 7)
    labels, i = [], 0
    while i < n:
        text = text_sample(c, r).strip()
        if not text or not has_arabic(text) and not any("٠" <= ch <= "٩" for ch in text):
            continue
        vis = logical_to_visual(text)
        if visual_to_logical(vis) != text:
            continue
        font = r.choice(DECO) if r.random() < 0.7 else r.choice(PLAIN)
        img = render(text, font, r.randint(26, 64), r)
        if img is None or img.shape[1] > 1800:
            continue
        img = degrade(img, r)
        name = f"d{i:06d}.png"
        cv2.imwrite(os.path.join(out_dir, name), img)
        labels.append(f"{name}\t{vis}\t{text}")
        i += 1
    with open(os.path.join(out_dir, "labels.tsv"), "w", encoding="utf-8") as f:
        f.write("\n".join(labels) + "\n")


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]) if len(sys.argv) > 3 else 0)
