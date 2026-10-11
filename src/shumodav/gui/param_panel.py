import math

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout, QHBoxLayout, QSlider,
                             QSpinBox, QWidget)

from ..core.types import ParamSpec

SLIDER_STEPS = 1000


class ParamPanel(QWidget):
    """Панель параметров, построенная по списку ParamSpec."""

    changed = pyqtSignal(dict)

    def __init__(self, specs=(), parent=None):
        super().__init__(parent)
        self._form = QFormLayout(self)
        self._form.setContentsMargins(0, 0, 0, 0)
        self._specs: tuple[ParamSpec, ...] = ()
        self._get, self._set, self._rows = {}, {}, {}
        self._silent = False
        self.rebuild(specs)

    # --- построение ---
    def rebuild(self, specs):
        self._specs = tuple(specs)
        while self._form.rowCount():
            self._form.removeRow(0)
        self._get, self._set, self._rows = {}, {}, {}
        for s in self._specs:
            w = {"float": self._float, "int": self._int, "choice": self._choice, "bool": self._bool}[s.kind](s)
            self._form.addRow(s.label, w)
            self._rows[s.name] = w
        self._update_visibility()

    def _float(self, s):
        lo, hi = (s.min if s.min is not None else 0.0), (s.max if s.max is not None else 1.0)
        use_log = s.log and lo > 0
        box, lay = QWidget(), None
        lay = QHBoxLayout(box)
        lay.setContentsMargins(0, 0, 0, 0)
        slider = QSlider(Qt.Horizontal)
        slider.setRange(0, SLIDER_STEPS)
        spin = QDoubleSpinBox()
        spin.setRange(lo, hi)
        step = s.step or (hi - lo) / 100
        spin.setSingleStep(step)
        spin.setDecimals(min(4, max(1, math.ceil(-math.log10(step)))) if step < 1 else 1)
        spin.setValue(float(s.default))
        lay.addWidget(slider, 3)
        lay.addWidget(spin, 2)

        def to_slider(v):
            if hi == lo:
                return 0
            f = (math.log(v) - math.log(lo)) / (math.log(hi) - math.log(lo)) if use_log else (v - lo) / (hi - lo)
            return int(round(min(max(f, 0), 1) * SLIDER_STEPS))

        def from_slider(i):
            f = i / SLIDER_STEPS
            return math.exp(math.log(lo) + f * (math.log(hi) - math.log(lo))) if use_log else lo + f * (hi - lo)

        def on_slider(i):
            spin.blockSignals(True)
            spin.setValue(from_slider(i))
            spin.blockSignals(False)
            self._on_change()

        def on_spin(v):
            slider.blockSignals(True)
            slider.setValue(to_slider(v))
            slider.blockSignals(False)
            self._on_change()

        slider.setValue(to_slider(float(s.default)))
        slider.valueChanged.connect(on_slider)
        spin.valueChanged.connect(on_spin)
        self._get[s.name] = lambda: float(spin.value())
        self._set[s.name] = lambda v: (spin.setValue(float(v)), slider.setValue(to_slider(float(v))))
        return box

    def _int(self, s):
        w = QSpinBox()
        w.setRange(int(s.min if s.min is not None else 0), int(s.max if s.max is not None else 100))
        w.setSingleStep(int(s.step or 1))
        w.setValue(int(s.default))
        w.valueChanged.connect(lambda _: self._on_change())
        self._get[s.name] = lambda: int(w.value())
        self._set[s.name] = lambda v: w.setValue(int(v))
        return w

    def _choice(self, s):
        w = QComboBox()
        w.addItems([str(c) for c in s.choices])
        w.setCurrentIndex(list(s.choices).index(s.default))
        w.currentIndexChanged.connect(lambda _: self._on_change())
        self._get[s.name] = lambda: s.choices[w.currentIndex()]
        self._set[s.name] = lambda v: w.setCurrentIndex(list(s.choices).index(v)) if v in s.choices else None
        return w

    def _bool(self, s):
        w = QCheckBox()
        w.setChecked(bool(s.default))
        w.toggled.connect(lambda _: self._on_change())
        self._get[s.name] = lambda: bool(w.isChecked())
        self._set[s.name] = lambda v: w.setChecked(bool(v))
        return w

    # --- значения ---
    def values(self) -> dict:
        return {name: get() for name, get in self._get.items()}

    def set_values(self, values: dict):
        self._silent = True
        try:
            for name, v in values.items():
                if name in self._set:
                    self._set[name](v)
        finally:
            self._silent = False
        self._on_change()

    def is_visible_param(self, name: str) -> bool:
        return self._rows[name].isVisibleTo(self)

    def _update_visibility(self):
        vals = self.values()
        for s in self._specs:
            if s.visible_if:
                vis = all(vals.get(k) == v for k, v in s.visible_if.items())
                self._rows[s.name].setVisible(vis)
                self._form.labelForField(self._rows[s.name]).setVisible(vis)

    def _on_change(self):
        if self._silent:
            return
        self._update_visibility()
        self.changed.emit(self.values())
