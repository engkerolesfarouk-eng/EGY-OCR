"""Fine-tune the PP-OCRv5 Arabic recognizer (ONNX -> PyTorch -> ONNX) on synthetic lines.

python train.py BASE.onnx data_train data_val OUT.onnx [steps]
"""
from __future__ import annotations

import random
import sys
import time

import cv2
import numpy as np
import onnx
import torch
from onnx2torch import convert

from common import edit, greedy, load_charset, load_labels, prep

torch.set_num_threads(max(1, torch.get_num_threads()))
base, dtrain, dval, out = sys.argv[1:5]
steps = int(sys.argv[5]) if len(sys.argv) > 5 else 600
BATCH, LR = 16, 8e-5

charset, index = load_charset()
onnx_model = onnx.load(base)
model = convert(onnx_model)

# onnx2torch keeps only Conv/BN weights as parameters; the SVTR attention/MLP and
# the classifier weights end up as constant buffers. Make the float ones trainable.
for mod in model.modules():
    for n in list(mod._buffers):
        t = mod._buffers[n]
        if t is not None and t.is_floating_point() and t.numel() >= 16 and "running" not in n:
            mod._buffers.pop(n)
            mod.register_parameter(n, torch.nn.Parameter(t.clone()))

# Map every torch tensor back to the ONNX initializer it came from (by exact value),
# so the trained weights can be written into the original graph, which keeps its
# dynamic input width (re-exporting from torch would freeze the shapes).
from onnx import numpy_helper
_by_key = {}
for init in onnx_model.graph.initializer:
    a = numpy_helper.to_array(init)
    if a.dtype == np.float32:
        _by_key.setdefault((a.shape, a.tobytes()), []).append(init.name)
tensor_to_init = {}
for n, p in list(model.named_parameters()) + list(model.named_buffers()):
    a = p.detach().numpy()
    names = _by_key.get((a.shape, a.astype(np.float32).tobytes()))
    if names:
        tensor_to_init[n] = names


def save_onnx(path):
    state = dict(model.named_parameters()); state.update(dict(model.named_buffers()))
    new = {}
    for n, names in tensor_to_init.items():
        for on in names:
            new[on] = state[n].detach().numpy().astype(np.float32)
    m = onnx.load(base)
    for init in m.graph.initializer:
        if init.name in new:
            init.CopyFrom(numpy_helper.from_array(new[init.name], init.name))
    onnx.save(m, path)
train_rows = [(p, l) for d in dtrain.split(",") for p, l in load_labels(d) if all(c in index for c in l)]
val_rows = [row for d in dval.split(",") for row in load_labels(d)]
random.Random(1).shuffle(val_rows)
print(f"trainable tensors {sum(1 for _ in model.parameters())}, mapped to onnx {len(tensor_to_init)}", flush=True)
print(f"train {len(train_rows)} val {len(val_rows)} classes {len(charset)}", flush=True)


def batch(rows):
    xs = [prep(cv2.imread(p)) for p, _ in rows]
    W = max(x.shape[2] for x in xs)
    x = np.zeros((len(xs), 3, 48, W), np.float32)
    for i, a in enumerate(xs):
        x[i, :, :, :a.shape[2]] = a
    return torch.from_numpy(x), xs


def evaluate(n=150):
    model.eval()
    E = N = 0
    with torch.no_grad():
        for i in range(0, min(n, len(val_rows)), 16):
            rows = val_rows[i:i + 16]
            x, xs = batch(rows)
            probs = model(x).numpy()
            T = probs.shape[1]
            for k, (p, lab) in enumerate(rows):
                t = int(np.ceil(T * xs[k].shape[2] / x.shape[3]))
                pred = greedy(probs[k, :t], charset)
                E += edit(pred, lab); N += len(lab)
    return 100.0 * E / N


def set_train_mode():
    model.train()
    for m in model.modules():  # keep BatchNorm statistics from the original model
        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
            m.eval()


ctc = torch.nn.CTCLoss(blank=0, zero_infinity=True)
opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=steps, pct_start=0.1)

print(f"step 0 val CER {evaluate():.2f}%", flush=True)
best = 1e9
t0 = time.time()
r = random.Random(0)
for step in range(1, steps + 1):
    set_train_mode()
    rows = r.sample(train_rows, BATCH)
    x, xs = batch(rows)
    probs = model(x)                                   # N, T, C (softmax)
    logp = torch.log(probs.clamp_min(1e-8)).permute(1, 0, 2)  # T, N, C
    T = logp.shape[0]
    in_len = torch.tensor([max(1, int(np.ceil(T * a.shape[2] / x.shape[3]))) for a in xs])
    targets = [torch.tensor([index[c] for c in lab]) for _, lab in rows]
    tgt_len = torch.tensor([len(t) for t in targets])
    loss = ctc(logp, torch.cat(targets), in_len, tgt_len)
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
    opt.step(); sched.step()
    if step % 20 == 0:
        print(f"step {step} loss {loss.item():.3f}  {(time.time()-t0)/step:.1f}s/step", flush=True)
    if step % 100 == 0 or step == steps:
        cer = evaluate()
        print(f"step {step} val CER {cer:.2f}%", flush=True)
        if cer < best:
            best = cer
            save_onnx(out)
            print(f"  saved {out}", flush=True)
print("best", best)
