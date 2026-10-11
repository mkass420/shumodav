import json
from pathlib import Path

import numpy as np
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont, QStandardItem, QStandardItemModel
from PyQt5.QtWidgets import (QAbstractItemView, QCheckBox, QComboBox, QFileDialog, QGridLayout, QGroupBox,
                             QHBoxLayout, QHeaderView, QLabel, QMainWindow, QPushButton, QScrollArea, QSpinBox,
                             QSplitter, QTableWidget, QTableWidgetItem, QTabWidget, QVBoxLayout, QWidget)

from ..bench.common import RESULTS
from ..core.io import load_image, save_image, to_gray
from ..core.registry import get_denoiser, get_noise, list_denoisers, list_noises
from .compute import ComputeClient
from .param_panel import ParamPanel
from .widgets import SpectrumViewer, ZoomableImageView

FAMILY_ORDER = ("spatial", "frequency", "wavelet", "neural")
FAMILY_TITLES = {"spatial": "Пространственные", "frequency": "Частотные", "wavelet": "Вейвлеты", "neural": "Нейросети"}
DEBOUNCE_MS = 200
PREEMPT_S = 2.0  # новый запрос вытесняет текущий расчёт, только если тот идёт дольше
POLL_MS = 30


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
        self.client = ComputeClient()
        self._kind = None  # что считается сейчас: "process" | "compare" | None
        self._latest_id = 0
        self._compare_id = 0
        self._image_key = 0
        self._dirty = False
        self._compare_requested = False
        self._viz_dirty = True

        self._timer = QTimer(self, singleShot=True, interval=DEBOUNCE_MS)
        self._timer.timeout.connect(self._kick)
        self._poll_timer = QTimer(self, interval=POLL_MS)
        self._poll_timer.timeout.connect(self._poll)
        self._poll_timer.start()

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
        self.btn_cancel = QPushButton("Отмена расчёта")
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.clicked.connect(self.cancel_job)
        gl.addWidget(self.btn_cancel)
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
        self._image_key += 1  # в дочерний процесс уйдёт новая картинка
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

    # --- расчёты в дочернем процессе: одна задача за раз ---
    def is_busy(self):
        return (self.client.busy() or self._timer.isActive() or self._dirty or self._compare_requested)

    def has_running_workers(self) -> bool:
        return self.client.busy()

    def _kick(self):
        if self.original is None or self._method_name() is None:
            return
        if self.client.busy():
            self._dirty = True  # запустится сразу после текущей задачи (или вытеснит её, если она затянулась)
            return
        if self._compare_requested:
            self._start_compare()
        else:
            self._start_process()

    def prewarm(self):
        """Заранее запустить дочерний процесс, чтобы первый расчёт не ждал импортов."""
        self.client.start()

    def _snapshot(self):
        return {"clean": self.clean_image(), "no_noise": self.cb_nonoise.isChecked(),
                "noise": self._noise_name(), "noise_params": self.panel_noise.values(),
                "seed": self.spin_seed.value()}

    def _start_process(self):
        req = self._snapshot()
        req.update(method=self._method_name(), method_params=self.panel_method.values())
        h, w = req["clean"].shape[:2]
        self._kind = "process"
        self.client.submit("process", self._latest_id, req, self._image_key)
        big = h * w > 1_500_000
        self.statusBar().showMessage(
            f"Обработка… (большое изображение {w}×{h}: медленные методы могут считаться минуты)" if big else "Обработка…")
        self.btn_cancel.setEnabled(True)

    def cancel_job(self, *_):
        """Убить дочерний процесс: расчёт прерывается сразу; следующий запрос запустит новый процесс."""
        was = self.client.busy()
        kind = self._kind
        self.client.kill()
        self._kind = None
        self._dirty = False
        self._compare_requested = False
        self.btn_cancel.setEnabled(False)
        if kind == "compare":
            self._finish_compare("Сравнение отменено")
        elif was:
            self.statusBar().showMessage("Расчёт отменён")

    def _poll(self):
        """Забрать ответы дочернего процесса; вытеснить затянувшийся расчёт новым запросом."""
        for kind, job_id, payload in self.client.poll():
            if kind == "result":
                self._kind = None
                self._on_done(job_id, payload)
            elif kind == "row":
                if job_id == self._compare_id:
                    self._add_row(payload)
            elif kind == "compare_done":
                self._kind = None
                self._finish_compare("Сравнение завершено")
            elif kind == "crashed":
                was = self._kind
                self._kind = None
                if was == "compare":
                    self._finish_compare(f"Сравнение прервано: {payload}")
                else:
                    self.statusBar().showMessage(f"Ошибка: {payload}")
        if not self.client.busy():
            self.btn_cancel.setEnabled(False)
            if (self._dirty or self._compare_requested) and not self._timer.isActive():
                self._dirty = False
                self._kick()
        elif ((self._dirty or self._compare_requested) and self._kind == "process"
              and self.client.elapsed() > PREEMPT_S and not self._timer.isActive()):
            self.client.kill()  # текущий расчёт устарел и затянулся — не ждём его
            self._kind = None
            self._dirty = False
            self._kick()

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
        req["best_params"] = {d.name: best_for(req["noise"], req["noise_params"], d.name) for d in list_denoisers()}
        self._compare_id += 1
        self._kind = "compare"
        self.table.setSortingEnabled(False)
        self.table.setRowCount(0)
        self.btn_compare.setEnabled(False)
        self.btn_cancel.setEnabled(True)
        self.statusBar().showMessage("Сравнение методов…")
        self.client.submit("compare", self._compare_id, req, self._image_key)

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

    def _finish_compare(self, message):
        self.table.setSortingEnabled(True)
        self.btn_compare.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        self.statusBar().showMessage(message)

    def select_from_table(self, row):
        item = self.table.item(row, 0)
        if item is None:
            return
        self.set_method(item.data(Qt.UserRole))
        self.panel_method.set_values(item.data(Qt.UserRole + 1))

    # --- закрытие ---
    def closeEvent(self, event):
        self._timer.stop()
        self._poll_timer.stop()
        self.client.shutdown()  # убить дочерний процесс — ждать расчёт не нужно
        super().closeEvent(event)
