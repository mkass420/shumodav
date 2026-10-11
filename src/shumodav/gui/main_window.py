import json
from pathlib import Path

import numpy as np
from PyQt5.QtCore import Qt, QThread, QTimer, pyqtSignal
from PyQt5.QtGui import QFont, QStandardItem, QStandardItemModel
from PyQt5.QtWidgets import (QAbstractItemView, QCheckBox, QComboBox, QFileDialog, QGridLayout, QGroupBox,
                             QHBoxLayout, QHeaderView, QLabel, QMainWindow, QPushButton, QScrollArea, QSpinBox,
                             QSplitter, QTableWidget, QTableWidgetItem, QTabWidget, QVBoxLayout, QWidget)

from ..bench.common import RESULTS
from ..core.io import load_image, save_image, to_gray
from ..core.pipeline import apply_noise, evaluate, run_denoiser
from ..core.registry import get_denoiser, get_noise, list_denoisers, list_noises, resolve_params
from .param_panel import ParamPanel
from .widgets import SpectrumViewer, ZoomableImageView

FAMILY_ORDER = ("spatial", "frequency", "wavelet", "neural")
FAMILY_TITLES = {"spatial": "Пространственные", "frequency": "Частотные", "wavelet": "Вейвлеты", "neural": "Нейросети"}
DEBOUNCE_MS = 200


# ---------- лучшие параметры из бенчмарка ----------
def load_best() -> dict:
    try:
        return json.loads((RESULTS / "best_params.json").read_text())
    except (OSError, ValueError):
        return {}


def nearest_level(noise: str, params: dict) -> str:
    """Уровень шума (low/mid/high), ближайший к текущим параметрам."""
    def dist(lvl):
        return sum(abs(params[k] - v) / (abs(v) + 1e-9) for k, v in lvl.items()
                   if k in params and isinstance(v, (int, float)))
    levels = get_noise(noise).levels
    return min(levels, key=lambda name: dist(levels[name]))


def best_for(noise: str, noise_params: dict, method: str):
    return load_best().get(noise, {}).get(nearest_level(noise, noise_params), {}).get(method)


# ---------- потоки ----------
def _make_noisy(req):
    if req["no_noise"]:
        return req["clean"]
    return apply_noise(req["clean"], req["noise"], req["noise_params"], req["seed"])


class ProcessWorker(QThread):
    done = pyqtSignal(int, object)

    def __init__(self, req_id, req, parent=None):
        super().__init__(parent)
        self.req_id, self.req = req_id, req

    def run(self):
        r, out = self.req, {"noisy": None, "result": None, "metrics": None, "seconds": None, "vis": None,
                            "error": None}
        try:
            out["noisy"] = noisy = _make_noisy(r)
            res, sec = run_denoiser(noisy, r["method"], r["method_params"])
            out["result"], out["seconds"] = res, sec
            if not r["no_noise"]:
                out["metrics"] = evaluate(r["clean"], noisy, res)
            d = get_denoiser(r["method"])
            try:
                out["vis"] = d.visualize(noisy, **resolve_params(d.params, r["method_params"]))
            except Exception:
                out["vis"] = None
        except Exception as e:  # ошибка метода не должна ронять окно
            out["error"] = f"{type(e).__name__}: {e}"
        self.done.emit(self.req_id, out)


class CompareWorker(QThread):
    row = pyqtSignal(object)

    def __init__(self, req, parent=None):
        super().__init__(parent)
        self.req = req

    def run(self):
        r = self.req
        noisy = _make_noisy(r)
        for d in list_denoisers():
            if self.isInterruptionRequested():
                return
            params = r["best"](d.name) or {}
            item = {"method": d.name, "family": d.family, "params": resolve_params(d.params, params),
                    "psnr": None, "ssim": None, "seconds": None, "error": None}
            try:
                res, sec = run_denoiser(noisy, d.name, params)
                item["seconds"] = sec
                if not r["no_noise"]:
                    ev = evaluate(r["clean"], noisy, res)
                    item["psnr"], item["ssim"] = ev["psnr"], ev["ssim"]
            except Exception as e:
                item["error"] = f"{type(e).__name__}: {e}"
            self.row.emit(item)


class NumItem(QTableWidgetItem):
    """Ячейка таблицы, сортируемая по числу (None — в конец)."""

    def __init__(self, value, text):
        super().__init__(text)
        self.setData(Qt.UserRole, float("inf") if value is None else float(value))
        self.setFlags(self.flags() & ~Qt.ItemIsEditable)

    def __lt__(self, other):
        return self.data(Qt.UserRole) < other.data(Qt.UserRole)


def fmt(v, digits=2, unit="") -> str:
    if v is None:
        return "—"
    if v == float("inf"):
        return "∞"
    return f"{v:.{digits}f}{unit}"


# ---------- главное окно ----------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Шумоподавление изображений — сравнение методов")
        self.resize(1400, 850)
        self.original = None  # как загружено (RGB или gray)
        self.noisy = self.result = self.vis = None
        self._worker = self._compare = None
        self._latest_id = 0
        self._dirty = False
        self._compare_requested = False
        self._viz_dirty = True

        self._timer = QTimer(self, singleShot=True, interval=DEBOUNCE_MS)
        self._timer.timeout.connect(self._kick)

        splitter = QSplitter(self)
        splitter.addWidget(self._build_left())
        splitter.addWidget(self._build_tabs())
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([340, 1060])
        self.setCentralWidget(splitter)
        self.statusBar().showMessage("Загрузите изображение")
        self._on_noise_changed()
        first = next((i for i in range(self.combo_method.count()) if self.combo_method.itemData(i, Qt.UserRole)), -1)
        self.combo_method.setCurrentIndex(first)
        self._on_method_changed()

    # --- построение интерфейса ---
    def _build_left(self):
        box = QWidget()
        lay = QVBoxLayout(box)

        row = QHBoxLayout()
        self.btn_load = QPushButton("Загрузить")
        self.btn_save = QPushButton("Сохранить результат")
        self.btn_load.clicked.connect(self.open_image)
        self.btn_save.clicked.connect(self.save_result)
        row.addWidget(self.btn_load)
        row.addWidget(self.btn_save)
        lay.addLayout(row)

        self.cb_gray = QCheckBox("Работать в оттенках серого")
        self.cb_gray.toggled.connect(self._on_image_changed)
        lay.addWidget(self.cb_gray)

        g = QGroupBox("Шум")
        gl = QVBoxLayout(g)
        self.combo_noise = QComboBox()
        for n in list_noises():
            self.combo_noise.addItem(n.label, n.name)
        self.combo_noise.currentIndexChanged.connect(self._on_noise_changed)
        self.panel_noise = ParamPanel()
        self.panel_noise.changed.connect(self._on_noise_params)
        self.spin_seed = QSpinBox()
        self.spin_seed.setRange(0, 2**31 - 1)
        self.spin_seed.valueChanged.connect(self._changed)
        self.cb_nonoise = QCheckBox("Без шума (картинка уже шумная)")
        self.cb_nonoise.toggled.connect(self._changed)
        srow = QHBoxLayout()
        srow.addWidget(QLabel("Seed"))
        srow.addWidget(self.spin_seed)
        for w in (self.combo_noise, self.panel_noise):
            gl.addWidget(w)
        gl.addLayout(srow)
        gl.addWidget(self.cb_nonoise)
        lay.addWidget(g)

        g = QGroupBox("Метод")
        gl = QVBoxLayout(g)
        self.combo_method = QComboBox()
        model = QStandardItemModel(self.combo_method)
        self.combo_method.setModel(model)
        avail = list_denoisers()
        for fam in FAMILY_ORDER:
            ds = [d for d in avail if d.family == fam]
            if not ds:
                continue
            head = QStandardItem(FAMILY_TITLES[fam])
            head.setFlags(Qt.NoItemFlags)
            f = QFont()
            f.setBold(True)
            head.setFont(f)
            model.appendRow(head)
            for d in ds:
                it = QStandardItem("    " + d.label)
                it.setData(d.name, Qt.UserRole)
                model.appendRow(it)
        self.combo_method.currentIndexChanged.connect(self._on_method_changed)
        self.panel_method = ParamPanel()
        self.panel_method.changed.connect(self._on_method_params)
        self.btn_best = QPushButton("Лучшие параметры")
        self.btn_best.clicked.connect(self.apply_best)
        gl.addWidget(self.combo_method)
        gl.addWidget(self.panel_method)
        gl.addWidget(self.btn_best)
        lay.addWidget(g)

        g = QGroupBox("Метрики")
        gl = QVBoxLayout(g)
        self.lbl_noisy = QLabel()
        self.lbl_result = QLabel()
        self.lbl_time = QLabel()
        for w in (self.lbl_noisy, self.lbl_result, self.lbl_time):
            gl.addWidget(w)
        lay.addWidget(g)
        self._show_metrics(None, None)
        lay.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(box)
        scroll.setMinimumWidth(320)
        return scroll

    def _build_tabs(self):
        self.tabs = QTabWidget()
        # 1. изображения
        page = QWidget()
        lay = QHBoxLayout(page)
        self.views = []
        for title in ("Оригинал", "Шумное", "Результат"):
            col = QVBoxLayout()
            lbl = QLabel(title)
            lbl.setAlignment(Qt.AlignCenter)
            v = ZoomableImageView()
            v.view_changed.connect(lambda v=v: self._sync_views(v))
            self.views.append(v)
            col.addWidget(lbl)
            col.addWidget(v, 1)
            lay.addLayout(col, 1)
        self.tabs.addTab(page, "Изображения")
        # 2. визуализация
        page = QWidget()
        grid = QGridLayout(page)
        self.spectra = [SpectrumViewer() for _ in range(4)]
        for i, s in enumerate(self.spectra):
            grid.addWidget(s, i // 2, i % 2)
        self.tabs.addTab(page, "Визуализация")
        self.tabs.currentChanged.connect(self._on_tab)
        # 3. сравнение
        page = QWidget()
        lay = QVBoxLayout(page)
        self.btn_compare = QPushButton("Прогнать все методы")
        self.btn_compare.clicked.connect(self.compare_all)
        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["Метод", "Семейство", "PSNR, дБ", "SSIM", "Время, с"])
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.cellDoubleClicked.connect(lambda r, c: self.select_from_table(r))
        lay.addWidget(self.btn_compare)
        lay.addWidget(self.table, 1)
        self.tabs.addTab(page, "Сравнение")
        return self.tabs

    def _sync_views(self, src):
        for v in self.views:
            if v is not src:
                v.sync_state_from(src)

    # --- загрузка / сохранение ---
    def open_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "Загрузить изображение", "",
                                              "Изображения (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)")
        if path:
            self.load_path(path)

    def load_path(self, path):
        try:
            img = load_image(path)
        except Exception as e:
            self.statusBar().showMessage(f"Ошибка загрузки: {e}")
            return False
        if img.ndim == 3 and np.array_equal(img[..., 0], img[..., 1]) and np.array_equal(img[..., 1], img[..., 2]):
            img = np.ascontiguousarray(img[..., 0])  # серый файл: не гоняем три одинаковых канала
        self.original = img
        self._on_image_changed()
        return True

    def save_result(self):
        if self.result is None:
            self.statusBar().showMessage("Нет результата для сохранения")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Сохранить результат", "result.png", "PNG (*.png);;JPEG (*.jpg)")
        if path:
            try:
                save_image(path, self.result)
                self.statusBar().showMessage(f"Сохранено: {path}")
            except Exception as e:
                self.statusBar().showMessage(f"Ошибка сохранения: {e}")

    def clean_image(self):
        if self.original is None:
            return None
        return to_gray(self.original) if self.cb_gray.isChecked() else self.original

    # --- изменения настроек ---
    def _noise_name(self):
        return self.combo_noise.currentData()

    def _method_name(self):
        return self.combo_method.currentData()

    def _on_image_changed(self, *_):
        clean = self.clean_image()
        if clean is not None:
            self.views[0].set_image(clean)
            self.noisy = self.result = None
        self._changed()

    def _on_noise_changed(self, *_):
        n = get_noise(self._noise_name())
        self.panel_noise.rebuild(n.params)
        self.panel_noise.set_values(n.levels["mid"])
        self._update_best_button()

    def _on_noise_params(self, *_):
        self._update_best_button()
        self._changed()

    def _on_method_changed(self, *_):
        name = self._method_name()
        if name is None:  # выбран заголовок семейства
            return
        self.panel_method.rebuild(get_denoiser(name).params)
        self._update_best_button()
        self._changed()

    def _on_method_params(self, *_):
        self._changed()

    def set_noise(self, name):
        i = self.combo_noise.findData(name)
        if i >= 0:
            self.combo_noise.setCurrentIndex(i)

    def set_method(self, name):
        i = self.combo_method.findData(name, Qt.UserRole)
        if i >= 0:
            self.combo_method.setCurrentIndex(i)

    def _update_best_button(self):
        m = self._method_name()
        ok = m is not None and best_for(self._noise_name(), self.panel_noise.values(), m) is not None
        self.btn_best.setEnabled(bool(ok))

    def apply_best(self):
        p = best_for(self._noise_name(), self.panel_noise.values(), self._method_name())
        if p:
            self.panel_method.set_values(p)

    def _changed(self, *_):
        self._latest_id += 1
        self._timer.start()

    # --- очередь потоков: один активный воркер ---
    def is_busy(self):
        return (self._worker is not None or self._compare is not None or self._timer.isActive()
                or self._dirty or self._compare_requested)

    def _kick(self):
        if self.original is None or self._method_name() is None:
            return
        if self._worker is not None or self._compare is not None:
            self._dirty = True
            return
        if self._compare_requested:
            self._start_compare()
        else:
            self._start_process()

    def _snapshot(self):
        return {"clean": self.clean_image(), "no_noise": self.cb_nonoise.isChecked(),
                "noise": self._noise_name(), "noise_params": self.panel_noise.values(),
                "seed": self.spin_seed.value()}

    def _start_process(self):
        req = self._snapshot()
        req.update(method=self._method_name(), method_params=self.panel_method.values())
        self._worker = ProcessWorker(self._latest_id, req)
        self._worker.done.connect(self._on_done)
        self._worker.finished.connect(self._on_worker_finished)
        self.statusBar().showMessage("Обработка…")
        self._worker.start()

    def _on_done(self, req_id, out):
        if req_id != self._latest_id:
            return  # устаревший запрос
        if out["noisy"] is not None:
            self.noisy = out["noisy"]
            self.views[1].set_image(self.noisy, maintain_zoom=True)
        self.result, self.vis = out["result"], out["vis"]
        if self.result is not None:
            self.views[2].set_image(self.result, maintain_zoom=True)
        self._show_metrics(out["metrics"], out["seconds"])
        self.statusBar().showMessage(f"Ошибка метода: {out['error']}" if out["error"] else "Готово")
        self._viz_dirty = True
        self._update_viz()

    def _on_worker_finished(self):
        w, self._worker = self._worker, None
        w.deleteLater()
        if self._dirty or self._compare_requested:
            self._dirty = False
            if not self._timer.isActive():
                self._kick()

    def _show_metrics(self, m, seconds):
        self.lbl_noisy.setText(f"Шумное: PSNR {fmt(m and m['psnr_noisy'])} дБ, SSIM {fmt(m and m['ssim_noisy'], 3)}")
        self.lbl_result.setText(f"Результат: PSNR {fmt(m and m['psnr'])} дБ, SSIM {fmt(m and m['ssim'], 3)}")
        self.lbl_time.setText(f"Время: {fmt(seconds, 3)} с")

    # --- визуализация ---
    def _on_tab(self, i):
        if i == 1:
            self._update_viz()

    def _update_viz(self):
        if self.tabs.currentIndex() != 1 or not self._viz_dirty or self.original is None:
            return
        self._viz_dirty = False
        for s, img, title in zip(self.spectra[:3], (self.clean_image(), self.noisy, self.result),
                                 ("Спектр: оригинал", "Спектр: шумное", "Спектр: результат")):
            if img is not None:
                s.update_spectrum(img, title)
        if self.vis is not None:
            self.spectra[3].show_image_directly(self.vis, "Визуализация метода")
        else:
            self.spectra[3].show_message("Для этого метода нет визуализации", "Визуализация метода")

    # --- сравнение ---
    def compare_all(self):
        if self.original is None:
            self.statusBar().showMessage("Сначала загрузите изображение")
            return
        self._compare_requested = True
        self._kick()

    def _start_compare(self):
        self._compare_requested = False
        req = self._snapshot()
        req["best"] = lambda m: best_for(req["noise"], req["noise_params"], m)
        self.table.setSortingEnabled(False)
        self.table.setRowCount(0)
        self.btn_compare.setEnabled(False)
        self._compare = CompareWorker(req)
        self._compare.row.connect(self._add_row)
        self._compare.finished.connect(self._on_compare_finished)
        self.statusBar().showMessage("Сравнение методов…")
        self._compare.start()

    def _add_row(self, it):
        t = self.table
        r = t.rowCount()
        t.insertRow(r)
        name_item = QTableWidgetItem(get_denoiser(it["method"]).label)
        name_item.setData(Qt.UserRole, it["method"])
        name_item.setData(Qt.UserRole + 1, it["params"])
        name_item.setFlags(name_item.flags() & ~Qt.ItemIsEditable)
        if it["error"]:
            name_item.setToolTip(it["error"])
        t.setItem(r, 0, name_item)
        t.setItem(r, 1, NumItem(FAMILY_ORDER.index(it["family"]), FAMILY_TITLES[it["family"]]))
        t.setItem(r, 2, NumItem(it["psnr"], fmt(it["psnr"])))
        t.setItem(r, 3, NumItem(it["ssim"], fmt(it["ssim"], 3)))
        t.setItem(r, 4, NumItem(it["seconds"], "ошибка" if it["error"] else fmt(it["seconds"], 4)))

    def _on_compare_finished(self):
        w, self._compare = self._compare, None
        w.deleteLater()
        self.table.setSortingEnabled(True)
        self.btn_compare.setEnabled(True)
        self.statusBar().showMessage("Сравнение завершено")
        if self._dirty:
            self._dirty = False
            if not self._timer.isActive():
                self._kick()

    def select_from_table(self, row):
        item = self.table.item(row, 0)
        if item is None:
            return
        self.set_method(item.data(Qt.UserRole))
        self.panel_method.set_values(item.data(Qt.UserRole + 1))

    # --- закрытие ---
    def closeEvent(self, event):
        if self._compare is not None:
            self._compare.requestInterruption()
            self._compare.wait()
        if self._worker is not None:
            self._worker.wait()
        super().closeEvent(event)
