"""History: files already read are kept, so opening one again is instant.

Each file is identified by its content (SHA-1), not its name, so a renamed or
moved file is still found and a changed file is read again. The pages are
stored with their text, layout and a compressed copy of the page image; the
user's corrections are stored next to them. Results from an older version of
the reader are ignored (the version is part of the key).

Location: %LOCALAPPDATA%\\ArabicOCR\\cache on Windows, ~/.cache/arocr elsewhere.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import os
import pickle

import cv2
import numpy as np

CACHE_VERSION = "2026-10-29b"
MAX_FILES = 200                       # oldest results are dropped beyond this


def cache_dir() -> str:
    base = os.environ.get("LOCALAPPDATA")
    d = os.path.join(base, "ArabicOCR", "cache") if base else os.path.join(os.path.expanduser("~"), ".cache", "arocr")
    os.makedirs(d, exist_ok=True)
    return d


def _models_stamp() -> str:
    from .pipeline import MODELS
    parts = []
    for f in sorted(os.listdir(MODELS)):
        if f.endswith(".onnx"):
            st = os.stat(os.path.join(MODELS, f))
            parts.append(f"{f}:{st.st_size}:{int(st.st_mtime)}")
    return "|".join(parts)


def file_key(path: str) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    h.update(CACHE_VERSION.encode())
    h.update(_models_stamp().encode())
    return h.hexdigest()


def _paths(key: str):
    d = cache_dir()
    return os.path.join(d, key + ".pages.gz"), os.path.join(d, key + ".edits.json")


def load_pages(key: str):
    pages_path, _ = _paths(key)
    if not os.path.exists(pages_path):
        return None
    try:
        with gzip.open(pages_path, "rb") as f:
            stored = pickle.load(f)
        pages = []
        for page, jpg in stored:
            page.image = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
            pages.append(page)
        os.utime(pages_path)          # recently used
        return pages
    except Exception:  # noqa: BLE001  (a damaged or old entry: read the file again)
        return None


def save_pages(key: str, pages) -> None:
    pages_path, _ = _paths(key)
    stored = []
    for p in pages:
        ok, buf = cv2.imencode(".jpg", p.image, [cv2.IMWRITE_JPEG_QUALITY, 88])
        img, p.image = p.image, None
        try:
            stored.append((pickle.loads(pickle.dumps(p)), buf.tobytes()))
        finally:
            p.image = img
    tmp = pages_path + ".tmp"
    with gzip.open(tmp, "wb", compresslevel=5) as f:
        pickle.dump(stored, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, pages_path)
    _trim()


def load_edits(key: str) -> dict:
    _, edits_path = _paths(key)
    try:
        with open(edits_path, encoding="utf-8") as f:
            return {int(k): v for k, v in json.load(f).items()}
    except (OSError, ValueError):
        return {}


def save_edits(key: str, edits: dict) -> None:
    _, edits_path = _paths(key)
    if not edits:
        if os.path.exists(edits_path):
            os.remove(edits_path)
        return
    with open(edits_path, "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in edits.items()}, f, ensure_ascii=False)


def _trim():
    d = cache_dir()
    entries = sorted((os.path.getmtime(os.path.join(d, f)), f) for f in os.listdir(d) if f.endswith(".pages.gz"))
    for _, f in entries[:-MAX_FILES]:
        for suffix in (".pages.gz", ".edits.json"):
            p = os.path.join(d, f[:-len(".pages.gz")] + suffix)
            if os.path.exists(p):
                os.remove(p)


def clear() -> int:
    d = cache_dir()
    n = 0
    for f in os.listdir(d):
        os.remove(os.path.join(d, f)); n += 1
    return n


def size_mb() -> float:
    d = cache_dir()
    return sum(os.path.getsize(os.path.join(d, f)) for f in os.listdir(d)) / 1e6

