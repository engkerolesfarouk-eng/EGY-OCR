import sys, cv2, numpy as np, onnxruntime as ort
from common import *
charset, _ = load_charset()
s = ort.InferenceSession(sys.argv[1], providers=["CPUExecutionProvider"])
rows = load_labels(sys.argv[2])
E_, N, exact = 0, 0, 0
digit_err = digit_n = 0
for p, lab in rows:
    x = prep(cv2.imread(p))[None]
    pred = greedy(s.run(None, {s.get_inputs()[0].name: x})[0][0], charset)
    e = edit(pred, lab); E_ += e; N += len(lab); exact += pred == lab
    dl = [c for c in lab if c.isdigit()]; dp = [c for c in pred if c.isdigit()]
    digit_err += edit(dp, dl); digit_n += len(dl)
print(f"CER {100*E_/N:.2f}%  exact lines {exact}/{len(rows)}  digit-CER {100*digit_err/max(1,digit_n):.1f}%")
