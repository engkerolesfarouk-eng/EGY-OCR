# Fine-tuning the Arabic recognizers (CPU is enough)

The stock PP-OCRv5 Arabic model drops scripture references "(رو ٣: ١٢)", Arabic-Indic
digits, « » quotes and brackets in printed books, and struggles with diacritics and
the stacked ligatures of classic naskh fonts. Short fine-tuning on synthetic lines fixes most of it.

| model in `models/` | trained on | used for |
|---|---|---|
| `rec_arabic.onnx` | `gen.py` + `gen_deco.py` | everything (default) |
| `rec_arabic_tashkeel.onnx` | `gen_tashkeel.py` (marks in the image, stripped from labels) | pages with heavy diacritics |
| `rec_arabic_book.onnx` | `gen_lig.py` (ligature fonts, kashida, wide spaces) | dense book pages |

## Setup
    pip install torch onnx onnx2torch opencv-python pillow arramooz-pysqlite

Fonts are **not** included (most are licensed by their vendors). Put your own
Arabic `.ttf` files in `training/fonts/` (e.g. Traditional Arabic, Simplified Arabic,
Arabic Typesetting, Sakkal Majalla, Arial, Tahoma) and decorative fonts in
`training/fonts_deco/` (e.g. Amiri, Scheherazade New, Lateef from Google Fonts, which are open).

## Steps
    cd training
    python gen.py data_train 6000 1            # synthetic lines + labels (visual order)
    python gen.py data_val 300 999
    python train.py ../models/rec_arabic.onnx data_train data_val rec_new.onnx 500
    python eval_onnx.py rec_new.onnx data_val

`train.py` accepts several data folders separated by commas (`data_train,data_deco`).
`real_eval.py` scores the full pipeline on real pages with hand-checked text in
`training/real/` (`name.jpg` + `name.gt.txt`); that folder is not in the repository.

~10 s/step on 2 CPU cores; a laptop CPU or a free Kaggle GPU is faster.

## Two traps
1. The dictionary lists every Arabic digit twice; the model uses the LAST copy
   (PaddleOCR builds its char->index map that way). Training on the first copy
   destroys digit recognition.
2. onnx2torch turns attention/MLP/classifier weights into constant buffers; `train.py`
   makes them trainable, then writes the weights back into the original ONNX graph
   (re-exporting from torch would freeze the input width).
