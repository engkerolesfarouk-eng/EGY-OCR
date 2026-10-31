"""How much of the computer the OCR takes.

Reading text runs neural networks, so while a page is being read the CPU is
busy - that is expected and it drops to zero when the app is idle. Two things
keep the PC usable meanwhile:
- AROCR_LOW_PRIORITY=1 runs it at below-normal priority, so anything you do
  goes first (slower on CPUs with efficiency cores, see lower_priority);
- AROCR_THREADS=<n> caps the cores it uses (default: all).
"""
from __future__ import annotations

import os
import sys


def lower_priority():
    # Off unless asked for (AROCR_LOW_PRIORITY=1): on laptop CPUs with
    # efficiency cores (Intel 12th gen and later) Windows moves low-priority
    # work onto the slow E-cores, and a page took 2-3x longer.
    if os.environ.get("AROCR_LOW_PRIORITY", "") != "1":
        return
    try:
        if sys.platform == "win32":
            import ctypes
            BELOW_NORMAL = 0x00004000
            ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), BELOW_NORMAL)
        else:
            os.nice(5)
    except Exception:  # noqa: BLE001  (never stop the app over this)
        pass


def session_options():
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.log_severity_level = 3
    n = os.environ.get("AROCR_THREADS", "").strip()
    if n.isdigit() and int(n) > 0:
        so.intra_op_num_threads = int(n)
        so.inter_op_num_threads = 1
    return so
