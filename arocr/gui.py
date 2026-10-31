"""Desktop window for arocr (PySide6).

    python -m arocr.gui
"""
from __future__ import annotations

import os
import sys
import time

import cv2
import numpy as np
from PySide6.QtCore import QObject, QSettings, Qt, QThread, QUrl, Signal, Slot
from PySide6.QtGui import QAction, QTextOption, QColor, QDesktopServices, QIcon, QImage, QKeySequence, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (QApplication, QCheckBox, QFileDialog, QHBoxLayout, QLabel,
                               QMainWindow, QMessageBox, QPlainTextEdit, QProgressBar,
                               QPushButton, QScrollArea, QSplitter, QStatusBar, QToolBar,
                               QToolButton, QMenu, QInputDialog, QVBoxLayout, QWidget)

FILE_FILTER = "PDF أو صور (*.pdf *.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp)"
SUPPORTED = (".pdf", ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")


APP_NAME = "EGY-OCR"
REPO_URL = "https://github.com/engkerolesfarouk-eng/EGY-OCR"
LINKEDIN_URL = "https://www.linkedin.com/in/keroles-farouk-6b715424/"


def _icon() -> QIcon:
    base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
    for name in ("icon.ico", "icon.png"):
        p = os.path.join(base, name)
        if os.path.exists(p):
            return QIcon(p)
    return QIcon()


class OcrWorker(QObject):
    """Runs the OCR off the UI thread so the window never freezes."""
    loaded = Signal(float)
    page_done = Signal(int, int, object)   # page number, total pages, PageResult
    finished = Signal(float, bool)         # seconds, came from history
    failed = Signal(str)
    keyed = Signal(str)                    # the file's history key

    def __init__(self):
        super().__init__()
        self.ocr = None
        self._clahe = False
        self._cancel = False

    @Slot()
    def load(self):
        try:
            t0 = time.perf_counter()
            from .pipeline import ArabicOcr
            self.ocr = ArabicOcr(use_clahe=self._clahe)
            self.loaded.emit(time.perf_counter() - t0)
        except Exception as e:  # noqa: BLE001
            self.failed.emit(f"تعذّر تحميل الموديلات:\n{e}")

    @Slot(str, bool, bool)
    def run(self, path: str, fresh: bool, id_cards: bool = False):
        """fresh=True: read again even if the file is in the history.
        id_cards=True: the file is Egyptian national ID cards."""
        self._cancel = False
        self.ocr.card_mode = id_cards
        try:
            from . import cache
            t0 = time.perf_counter()
            key = None
            try:
                key = cache.file_key(path) + ("-id" if id_cards else "")
                self.keyed.emit(key)
                stored = None if fresh else cache.load_pages(key)
            except OSError:
                stored = None
            if stored:
                from .learn import apply_page
                for page in stored:          # corrections taught since it was read
                    apply_page(page)
                for n, page in enumerate(stored, 1):
                    self.page_done.emit(n, len(stored), page)
                self.finished.emit(time.perf_counter() - t0, True)
                return
            total = [1]

            def progress(i, k):
                total[0] = k

            pages = []
            for n, page in enumerate(self.ocr.file(path, progress=progress), 1):
                pages.append(page)
                self.page_done.emit(n, total[0], page)
                if self._cancel:
                    break
            if key and pages and not self._cancel:
                try:
                    cache.save_pages(key, pages)
                except Exception:  # noqa: BLE001  (history is a convenience, never an error)
                    pass
            self.finished.emit(time.perf_counter() - t0, False)
        except Exception as e:  # noqa: BLE001
            self.failed.emit(str(e))

    def cancel(self):
        self._cancel = True


class SaveWorker(QObject):
    """Writes PDF / Word / HTML off the UI thread, page by page, with progress."""
    progress = Signal(int, int)
    done = Signal(str)
    failed = Signal(str)

    @Slot(object)
    def run(self, job):
        kind, path, pages, edits = job
        try:
            n = len(pages)
            if kind == "pdf":
                from .pdfout import SearchablePdf
                pdf = SearchablePdf(path)
                for i, p in enumerate(pages):
                    # ID-card mode shows fields, not the page's lines: the PDF keeps the card as read
                    ed = None if getattr(p, "fields_only", False) else edits.get(i)
                    pdf.add_page(p.image, p.lines, p.width_pt, p.height_pt, edited_text=ed)
                    self.progress.emit(i + 1, n)
                pdf.save()
            elif kind in ("docx", "docx-scan"):
                from .wordout import WordDocument
                doc = WordDocument(path, layout="scan" if kind == "docx-scan" else "text")
                for i, p in enumerate(pages):
                    doc.add_page(p, edits.get(i))
                    self.progress.emit(i + 1, n)
                doc.save()
            elif kind == "html":
                from .htmlout import write_html
                title = os.path.splitext(os.path.basename(path))[0].replace(".ocr", "")
                write_html(path, pages, edits, title, progress=lambda a, b: self.progress.emit(a, b))
            self.done.emit(path)
        except PermissionError:
            self.failed.emit("الملف مفتوح في برنامج تاني. اقفله وجرّب تاني.")
        except ImportError as e:
            if "docx" in str(e):
                self.failed.emit("محتاج تسطّب مكتبة python-docx مرة واحدة:\n\npy -m pip install --user python-docx")
            else:
                self.failed.emit(str(e))
        except Exception as e:  # noqa: BLE001
            import traceback
            self.failed.emit(f"{e}\n\n{traceback.format_exc()[-1500:]}")


class PageView(QScrollArea):
    """Page image with the detected words outlined."""

    def __init__(self):
        super().__init__()
        self.setWidgetResizable(True)
        self.label = QLabel("اسحب ملف PDF أو صورة هنا\nأو اضغط «فتح ملف»")
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setStyleSheet("color:#777; font-size:18px;")
        self.setWidget(self.label)
        self.page = None
        self.show_boxes = True

    def show_page(self, page):
        self.page = page
        self._render()

    def _render(self):
        if self.page is None:
            return
        img = self.page.image
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888).copy()
        pix = QPixmap.fromImage(qimg)
        if self.show_boxes:
            p = QPainter(pix)
            pen = QPen(QColor(0, 150, 255, 200)); pen.setWidth(max(1, w // 600))
            p.setPen(pen)
            for line in self.page.lines:
                for wd in line.words:
                    x0, y0, x1, y1 = wd.box
                    p.drawRect(int(x0), int(y0), int(x1 - x0), int(y1 - y0))
            p.end()
        target = max(200, self.viewport().width() - 20)
        if pix.width() > target:
            pix = pix.scaledToWidth(target, Qt.SmoothTransformation)
        self.label.setStyleSheet("")
        self.label.setPixmap(pix)

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._render()


class MainWindow(QMainWindow):
    start = Signal(str, bool, bool)
    save_req = Signal(object)

    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_NAME)
        self.setWindowIcon(_icon())
        self.resize(1250, 800)
        self.setAcceptDrops(True)
        self.pages = []
        self.current = 0
        self.path = None
        self.ready = False
        self.edits = {}
        self.shown = None   # page whose text is in the editor
        self.key = None     # history key of the open file
        self.saving = False
        self.settings = QSettings("arocr", "ArabicOCR")

        # toolbar
        tb = QToolBar(); tb.setMovable(False); self.addToolBar(tb)
        self.act_open = QAction("فتح ملف", self, shortcut=QKeySequence.Open, triggered=self.open_file)
        self.act_pdf = QAction("حفظ PDF قابل للبحث", self, triggered=self.save_pdf, enabled=False)
        self.act_docx = QAction("حفظ Word", self, triggered=self.save_docx, enabled=False)
        self.act_docx.setToolTip("مستند Word عادي تعدّل فيه زي أي ملف")
        self.act_docx_scan = QAction("Word بشكل الصفحة", self, triggered=lambda: self.save_docx("scan"), enabled=False)
        self.act_docx_scan.setToolTip("كل صفحة شكلها زي الصورة بالظبط (أتقل وأصعب في التعديل)")
        self.act_cards = QAction("بيانات البطايق (Excel)", self, triggered=self.save_cards, enabled=False)
        self.act_txt = QAction("حفظ النص", self, shortcut=QKeySequence.Save, triggered=self.save_txt, enabled=False)
        self.act_copy = QAction("نسخ النص", self, triggered=self.copy_text, enabled=False)
        self.act_html = QAction("حفظ HTML", self, triggered=lambda: self._start_save("html"), enabled=False)
        self.act_html.setToolTip("صفحة ويب تتفتح في أي متصفح أو موبايل")
        self.act_reread = QAction("قراءة من جديد", self, triggered=lambda: self.path and self.process(self.path, fresh=True),
                                  enabled=False)
        self.act_reread.setToolTip("اقرا الملف تاني بدل النتيجة المحفوظة")
        self.btn_recent = QToolButton(); self.btn_recent.setText("الأخيرة")
        self.btn_recent.setPopupMode(QToolButton.InstantPopup)
        self.recent_menu = QMenu(self); self.btn_recent.setMenu(self.recent_menu)
        tb.addAction(self.act_open); tb.addWidget(self.btn_recent)
        for a in (self.act_pdf, self.act_docx, self.act_docx_scan, self.act_html, self.act_cards, self.act_txt,
                  self.act_copy, self.act_reread):
            tb.addAction(a)
        tb.addSeparator()
        self.act_bigger = QAction("A+", self, shortcut=QKeySequence.ZoomIn, triggered=lambda: self._font_step(+2))
        self.act_smaller = QAction("A-", self, shortcut=QKeySequence.ZoomOut, triggered=lambda: self._font_step(-2))
        self.act_bigger.setToolTip("كبّر خط النص (Ctrl +)"); self.act_smaller.setToolTip("صغّر خط النص (Ctrl -)")
        tb.addAction(self.act_bigger); tb.addAction(self.act_smaller)
        tb.addSeparator()
        self.chk_boxes = QCheckBox("إظهار الكلمات"); self.chk_boxes.setChecked(True)
        self.chk_boxes.toggled.connect(self._toggle_boxes)
        tb.addWidget(self.chk_boxes)
        tb.addSeparator()
        self.chk_id = QCheckBox("بطاقة رقم قومي")
        self.chk_id.setToolTip("الملف بطايق رقم قومي مصرية: يطلع الاسم والعنوان والرقم القومي وتاريخ الميلاد\n"
                               "ويشيل أي كلام بره البطاقة (زي علامة الكاميرا)")
        self.chk_id.setChecked(self.settings.value("id_cards", False, type=bool))
        self.chk_id.toggled.connect(self._toggle_id)
        tb.addWidget(self.chk_id)
        tb.addSeparator()
        self.act_about = QAction("عن البرنامج", self, triggered=self.show_about)
        tb.addAction(self.act_about)

        # page view + text
        self.view = PageView()
        self.text = QPlainTextEdit(); self.text.setReadOnly(False)
        self.text.setLayoutDirection(Qt.RightToLeft)
        opt = self.text.document().defaultTextOption()
        opt.setTextDirection(Qt.RightToLeft); opt.setAlignment(Qt.AlignRight)
        self.text.document().setDefaultTextOption(opt)
        f = self.text.font(); f.setPointSize(int(self.settings.value("font_size", 16))); self.text.setFont(f)
        self.text.setPlaceholderText("النص المستخرج هيظهر هنا، وتقدر تعدّل فيه قبل الحفظ.")
        # right-click a wrong word: teach the program the right one
        self.text.setContextMenuPolicy(Qt.CustomContextMenu)
        self.text.customContextMenuRequested.connect(self._text_menu)

        nav = QHBoxLayout()
        self.btn_prev = QPushButton("السابقة ◀"); self.btn_next = QPushButton("▶ التالية")
        self.lbl_page = QLabel(""); self.lbl_page.setAlignment(Qt.AlignCenter)
        self.btn_prev.clicked.connect(lambda: self.goto(self.current - 1))
        self.btn_next.clicked.connect(lambda: self.goto(self.current + 1))
        # in a right-to-left window "previous" sits on the right
        nav.addWidget(self.btn_prev); nav.addWidget(self.lbl_page, 1); nav.addWidget(self.btn_next)
        left = QWidget(); lv = QVBoxLayout(left); lv.setContentsMargins(0, 0, 0, 0)
        lv.addWidget(self.view, 1); lv.addLayout(nav)

        split = QSplitter(); split.addWidget(left); split.addWidget(self.text)
        split.setSizes([700, 550])
        self.setCentralWidget(split)

        self.status = QStatusBar(); self.setStatusBar(self.status)
        self.progress = QProgressBar(); self.progress.setMaximumWidth(260); self.progress.hide()
        self.status.addPermanentWidget(self.progress)
        self._update_nav()

        # worker thread
        self.thread = QThread(self)
        self.worker = OcrWorker(); self.worker.moveToThread(self.thread)
        self.worker.loaded.connect(self._loaded)
        self.worker.page_done.connect(self._page_done)
        self.worker.finished.connect(self._finished)
        self.worker.failed.connect(self._failed)
        self.worker.keyed.connect(self._keyed)
        self.start.connect(self.worker.run)
        self.thread.started.connect(self.worker.load)
        self.thread.start()
        self.status.showMessage("جارٍ تحميل الموديلات...")
        self.act_open.setEnabled(False)

        # saving runs on its own thread: the window stays usable, with progress
        self.save_thread = QThread(self)
        self.saver = SaveWorker(); self.saver.moveToThread(self.save_thread)
        self.save_req.connect(self.saver.run)
        self.saver.progress.connect(self._save_progress)
        self.saver.done.connect(self._save_done)
        self.saver.failed.connect(self._save_failed)
        self.save_thread.start()
        self._rebuild_recent()

    # --- worker callbacks
    def _loaded(self, secs):
        self.ready = True
        self.act_open.setEnabled(True)
        self.status.showMessage(f"جاهز (الموديلات اتحملت في {secs:.1f} ث). اسحب ملف أو افتحه.")
        if self.path:  # a file was dropped while loading
            self.process(self.path)

    def _page_done(self, n, total, page):
        self.pages.append(page)
        self.progress.setMaximum(total); self.progress.setValue(n)
        self.status.showMessage(f"صفحة {n} من {total}  —  {page.seconds:.1f} ث")
        if n == 1:
            self.goto(0)
        self._update_nav()

    def _keyed(self, key):
        """The file is identified: bring back the corrections made last time."""
        from . import cache
        self.key = key
        self.edits = cache.load_edits(key)

    def _finished(self, secs, cached=False):
        self.busy = False
        self.progress.hide()
        self.act_open.setEnabled(True)
        has = bool(self.pages)
        for a in (self.act_pdf, self.act_docx, self.act_docx_scan, self.act_html, self.act_txt, self.act_copy,
                  self.act_reread):
            a.setEnabled(has and not self.saving)
        n_cards = sum(len(p.cards) for p in self.pages)
        self.act_cards.setEnabled(n_cards > 0)
        n = len(self.pages)
        extra = f" — {n_cards} بطاقة" if n_cards else ""
        if cached:
            fixes = f" — {len(self.edits)} صفحة فيها تصحيحاتك" if self.edits else ""
            self.status.showMessage(f"من السجل: {n} صفحة (اتقرا قبل كده){extra}{fixes}. «قراءة من جديد» لو عايز تقراه تاني.")
        else:
            self.status.showMessage(f"تم: {n} صفحة في {secs:.1f} ث{extra}")
        if self.pages and self.shown is not None:
            self.goto(self.current)        # show restored corrections

    def _failed(self, msg):
        self.busy = False
        self.progress.hide()
        self.act_open.setEnabled(self.ready)
        self.status.showMessage("حصل خطأ")
        QMessageBox.critical(self, "خطأ", msg)

    # --- actions
    def open_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "اختر ملف", "", FILE_FILTER)
        if path:
            self.process(path)

    def show_about(self):
        from . import __version__
        box = QMessageBox(self)
        box.setWindowTitle("عن البرنامج")
        box.setIconPixmap(_icon().pixmap(96, 96))
        box.setTextFormat(Qt.RichText)
        box.setText(
            f"<div dir='rtl' style='min-width:430px'>"
            f"<h2 dir='ltr' align='right'>{APP_NAME} &nbsp;<span style='font-weight:normal;font-size:12pt'>"
            f"{__version__}</span></h2>"
            "<p>بيقرا النص العربي من الكتب والمستندات وصور الموبايل وبطايق الرقم القومي المصرية، "
            "<b>من غير إنترنت</b>. الملفات ما بتخرجش من جهازك.</p>"
            "<p>بيطلّع:<br>"
            "• ملف PDF تقدر تدوّر فيه وتنسخ منه<br>"
            "• ملف Word تقدر تعدّل فيه<br>"
            "• صفحة HTML وملف نص<br>"
            "• ملف Excel ببيانات البطايق</p>"
            f"<p dir='ltr' align='right'>Developed by <b>Keroles Farouk</b> &nbsp;·&nbsp; <a href='{LINKEDIN_URL}'>LinkedIn</a><br>"
            f"<a href='{REPO_URL}'>{REPO_URL}</a></p>"
            "<p style='font-size:9pt;color:#666'>الرخصة: Apache 2.0</p>"
            "<p dir='ltr' align='left' style='font-size:9pt;color:#666'>"
            "Models based on PaddleOCR PP-OCRv5 (Apache 2.0), running on ONNX Runtime.</p>"
            "</div>")
        box.setTextInteractionFlags(Qt.TextBrowserInteraction)
        box.setStyleSheet("QLabel#qt_msgbox_label { min-width: 460px; }")
        box.exec()

    def _toggle_id(self, on):
        self.settings.setValue("id_cards", on)
        if self.path and not getattr(self, "busy", False):
            self.process(self.path)        # read the open file again in the new mode

    def process(self, path, fresh=False):
        if getattr(self, "busy", False):
            self.status.showMessage("استنى القراءة الحالية تخلص الأول...")
            return
        if self.saving:
            self.status.showMessage("استنى الحفظ يخلص الأول...")
            return
        self._persist_edits()
        self.path = path
        self._add_recent(path)
        if not self.ready:
            self.status.showMessage("الملف هيتقرا أول ما الموديلات تخلص تحميل...")
            return
        self.pages = []; self.edits = {}; self.current = 0; self.shown = None; self.key = None
        self.text.clear()
        for a in (self.act_open, self.act_pdf, self.act_docx, self.act_docx_scan, self.act_html, self.act_cards,
                  self.act_txt, self.act_copy, self.act_reread):
            a.setEnabled(False)
        self.progress.setValue(0); self.progress.setMaximum(0); self.progress.show()
        self.setWindowTitle(f"{APP_NAME} — {os.path.basename(path)}")
        self.status.showMessage(f"جارٍ قراءة {os.path.basename(path)}...")
        self.busy = True
        self.start.emit(path, fresh, self.chk_id.isChecked())

    def goto(self, i):
        if not self.pages:
            return
        self._persist_edits()
        i = max(0, min(i, len(self.pages) - 1))
        self.current = i
        page = self.pages[i]
        self.view.show_page(page)
        self.text.setPlainText(self.edits.get(i, page.text))
        self.shown = i
        self._update_nav()

    def _commit_edit(self):
        """Keep the user's corrections to the current page's text."""
        if self.shown is not None and self.shown < len(self.pages):
            t = self.text.toPlainText()
            if t != self.pages[self.shown].text:
                self.edits[self.shown] = t
            else:
                self.edits.pop(self.shown, None)

    def _persist_edits(self):
        """Corrections go into the history too, so they come back with the file."""
        self._commit_edit()
        if self.key:
            try:
                from . import cache
                cache.save_edits(self.key, self.edits)
            except OSError:
                pass

    # --- teaching corrections
    def _text_menu(self, pos):
        menu = self.text.createStandardContextMenu()
        sel = self.text.textCursor().selectedText().strip()
        if sel and "\u2029" not in sel and len(sel) <= 60:
            menu.addSeparator()
            act = menu.addAction(f"علّم البرنامج التصحيح ده: «{sel}»")
            act.triggered.connect(lambda: self._teach(sel))
        menu.addAction("ملف التصحيحات اللي اتعلمها", self._open_corrections)
        menu.exec(self.text.mapToGlobal(pos))

    def _teach(self, wrong):
        right, ok = QInputDialog.getText(self, "علّم البرنامج",
                                         f"البرنامج قرا «{wrong}».\nالصح إيه؟ (هيتصلح كده في أي ملف بعد كده)",
                                         text=wrong)
        right = right.strip()
        if not ok or not right or right == wrong:
            return
        from . import learn
        learn.add(wrong, right)
        # correct what is open now too: every page, your edits included
        self._commit_edit()
        for p in self.pages:
            learn.apply_page(p)
        self.edits = {i: learn.apply(t) for i, t in self.edits.items()}
        self.shown = None                 # the pane still shows the old text: don't take it as an edit
        self._persist_edits()
        self.goto(self.current)
        n = len(learn.rules())
        self.status.showMessage(f"اتعلم: «{wrong}» ← «{right}» — عنده {n} تصحيح")

    def _open_corrections(self):
        from . import learn
        QDesktopServices.openUrl(QUrl.fromLocalFile(learn.path()))

    # --- text size
    def _font_step(self, d):
        f = self.text.font()
        size = max(9, min(40, f.pointSize() + d))
        f.setPointSize(size); self.text.setFont(f)
        self.settings.setValue("font_size", size)

    # --- recent files
    def _add_recent(self, path):
        rec = [p for p in (self.settings.value("recent", []) or []) if p != path]
        rec = [path] + rec
        self.settings.setValue("recent", rec[:12])
        self._rebuild_recent()

    def _rebuild_recent(self):
        self.recent_menu.clear()
        rec = [p for p in (self.settings.value("recent", []) or []) if os.path.exists(p)]
        for p in rec:
            a = self.recent_menu.addAction(os.path.basename(p))
            a.setToolTip(p)
            a.triggered.connect(lambda _=False, p=p: self.process(p))
        if not rec:
            self.recent_menu.addAction("لسه مفيش ملفات").setEnabled(False)
        else:
            self.recent_menu.addSeparator()
            self.recent_menu.addAction("مسح السجل المحفوظ", self._clear_history)
        self.recent_menu.addAction("ملف التصحيحات اللي اتعلمها", self._open_corrections)

    def _clear_history(self):
        from . import cache
        if QMessageBox.question(self, "مسح السجل",
                                f"هيتمسح كل اللي اتقرا قبل كده وتصحيحاتك المحفوظة ({cache.size_mb():.0f} ميجا). أكمّل؟") \
                == QMessageBox.Yes:
            cache.clear()
            self.settings.setValue("recent", [])
            self._rebuild_recent()
            self.status.showMessage("السجل اتمسح")

    def _update_nav(self):
        n = len(self.pages)
        self.lbl_page.setText(f"صفحة {self.current + 1} من {n}" if n else "")
        self.btn_prev.setEnabled(self.current > 0)
        self.btn_next.setEnabled(self.current < n - 1)

    def _toggle_boxes(self, on):
        self.view.show_boxes = on
        self.view._render()

    def _all_text(self):
        self._commit_edit()
        parts = []
        for i, p in enumerate(self.pages):
            body = self.edits.get(i, p.text)
            parts.append(f"--- صفحة {i + 1} ---\n{body}" if len(self.pages) > 1 else body)
        return "\n\n".join(parts) + "\n"

    def _default_name(self, ext):
        base = os.path.splitext(self.path)[0] if self.path else "ocr"
        return base + ".ocr" + ext

    def save_cards(self):
        """The fields of every ID card read (name, address, national ID, birth
        date, sex, governorate, card number) as a sheet Excel opens."""
        cards = [c for p in self.pages for c in p.cards]
        if not cards:
            return
        path, _ = QFileDialog.getSaveFileName(self, "بيانات البطايق", self._default_name(".cards.csv"),
                                              "Excel CSV (*.csv)")
        if not path:
            return
        from .cards import to_csv
        try:
            to_csv(cards, path)
        except PermissionError:
            QMessageBox.warning(self, "تعذّر الحفظ", "الملف مفتوح في Excel. اقفله وجرّب تاني.")
            return
        bad = sum(1 for c in cards if not c.id_valid)
        note = f" — {bad} منهم الرقم القومي محتاج مراجعة" if bad else ""
        self.status.showMessage(f"اتحفظت {len(cards)} بطاقة: {path}{note}")

    # --- saving (background, with progress)
    _SAVE_KINDS = {
        "pdf": ("حفظ PDF قابل للبحث", ".pdf", "PDF (*.pdf)"),
        "docx": ("حفظ Word", ".docx", "Word (*.docx)"),
        "docx-scan": ("حفظ Word بشكل الصفحة", ".docx", "Word (*.docx)"),
        "html": ("حفظ HTML", ".html", "صفحة ويب (*.html)"),
    }

    def _start_save(self, kind):
        if self.saving or not self.pages:
            return
        title, ext, flt = self._SAVE_KINDS[kind]
        path, _ = QFileDialog.getSaveFileName(self, title, self._default_name(ext), flt)
        if not path:
            return
        self._persist_edits()
        self.saving = True
        for a in (self.act_pdf, self.act_docx, self.act_docx_scan, self.act_html, self.act_open, self.act_reread):
            a.setEnabled(False)
        self.progress.setMaximum(len(self.pages)); self.progress.setValue(0); self.progress.show()
        self.status.showMessage(f"جارٍ الحفظ... {os.path.basename(path)}")
        self.save_req.emit((kind, path, list(self.pages), dict(self.edits)))

    def _save_progress(self, n, total):
        self.progress.setMaximum(total); self.progress.setValue(n)
        self.status.showMessage(f"جارٍ الحفظ... صفحة {n} من {total}")

    def _save_finished(self):
        self.saving = False
        self.progress.hide()
        for a in (self.act_pdf, self.act_docx, self.act_docx_scan, self.act_html, self.act_open, self.act_reread):
            a.setEnabled(bool(self.pages) or a is self.act_open)

    def _save_done(self, path):
        self._save_finished()
        self.status.showMessage(f"اتحفظ: {path}")

    def _save_failed(self, msg):
        self._save_finished()
        self.status.showMessage("تعذّر الحفظ")
        QMessageBox.critical(self, "تعذّر الحفظ", msg)

    def save_txt(self):
        path, _ = QFileDialog.getSaveFileName(self, "حفظ النص", self._default_name(".txt"), "نص (*.txt)")
        if path:
            with open(path, "w", encoding="utf-8") as f:
                f.write(self._all_text())
            self.status.showMessage(f"اتحفظ: {path}")

    def save_docx(self, layout="text"):
        self._start_save("docx-scan" if layout == "scan" else "docx")

    def save_pdf(self):
        self._start_save("pdf")

    def copy_text(self):
        QApplication.clipboard().setText(self._all_text())
        self.status.showMessage("النص اتنسخ")

    # --- drag & drop
    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e):
        for url in e.mimeData().urls():
            path = url.toLocalFile()
            if path.lower().endswith(SUPPORTED):
                self.process(path)
                return

    def closeEvent(self, e):
        if self.saving and QMessageBox.question(self, "لسه بيحفظ", "الحفظ لسه شغال. تقفل برضه؟") != QMessageBox.Yes:
            e.ignore()
            return
        self._persist_edits()
        self.worker.cancel()
        self.thread.quit(); self.thread.wait(3000)
        self.save_thread.quit(); self.save_thread.wait(3000)
        super().closeEvent(e)


def main():
    if sys.platform == "win32":
        try:        # own taskbar button and icon, not Python's
            import ctypes
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("EGY-OCR")
        except Exception:
            pass
    app = QApplication(sys.argv)
    app.setLayoutDirection(Qt.RightToLeft)
    app.setApplicationName(APP_NAME)
    app.setWindowIcon(_icon())
    w = MainWindow(); w.show()
    if len(sys.argv) > 1 and os.path.exists(sys.argv[1]):
        w.process(sys.argv[1])
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
