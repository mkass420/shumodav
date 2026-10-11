import cv2
import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtCore import QRectF, Qt, pyqtSignal
from PyQt5.QtGui import QImage, QPainter, QPixmap
from PyQt5.QtWidgets import QGraphicsPixmapItem, QGraphicsScene, QGraphicsView, QPushButton


class ZoomableImageView(QGraphicsView):
    """Просмотр изображения с зумом колесом и синхронизацией с другими видами."""

    view_changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self.pixmap_item = QGraphicsPixmapItem()
        self._scene.addItem(self.pixmap_item)

        self.setRenderHint(QPainter.Antialiasing, False)
        self.setRenderHint(QPainter.SmoothPixmapTransform, False)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorViewCenter)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setFrameShape(0)
        self.setBackgroundBrush(Qt.lightGray)

        self._is_syncing = False
        self._has_image = False
        self._is_fitted = True
        self.reset_btn = QPushButton("Сброс", self)
        self.reset_btn.setCursor(Qt.ArrowCursor)
        self.reset_btn.setStyleSheet(
            "QPushButton {background-color: rgba(0,0,0,150); color: white; border: none;"
            " border-radius: 4px; padding: 4px 8px; font-weight: bold;}"
            "QPushButton:hover {background-color: rgba(0,0,0,200);}")
        self.reset_btn.clicked.connect(self.reset_zoom)
        self.reset_btn.hide()
        self.horizontalScrollBar().valueChanged.connect(self._emit_change)
        self.verticalScrollBar().valueChanged.connect(self._emit_change)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.reset_btn.move(self.width() - self.reset_btn.width() - 10, 10)
        if self._has_image and self._is_fitted:
            self.fitInView(self.pixmap_item, Qt.KeepAspectRatio)

    def has_image(self) -> bool:
        return self._has_image

    def set_image(self, image, maintain_zoom=False):
        if image is None:
            return
        was_empty = not self._has_image
        if image.dtype.kind == "f":
            img8 = (np.clip(image, 0, 1) * 255).astype(np.uint8)
        else:
            img8 = image.astype(np.uint8)
        img8 = np.ascontiguousarray(img8)
        h, w = img8.shape[:2]
        if img8.ndim == 3:
            q = QImage(img8.data, w, h, 3 * w, QImage.Format_RGB888)
        else:
            q = QImage(img8.data, w, h, w, QImage.Format_Grayscale8)
        pixmap = QPixmap.fromImage(q)  # копирует данные
        self.pixmap_item.setPixmap(pixmap)
        self._scene.setSceneRect(QRectF(pixmap.rect()))
        self._has_image = True
        self.reset_btn.show()
        if was_empty or not maintain_zoom:
            self.reset_zoom()

    def get_fit_scale(self):
        if not self._has_image:
            return 1.0
        vr, sr = self.viewport().rect(), self.pixmap_item.boundingRect()
        if sr.width() == 0 or sr.height() == 0:
            return 1.0
        return min(vr.width() / sr.width(), vr.height() / sr.height())

    def wheelEvent(self, event):
        if not self._has_image:
            return
        factor = 1.25
        cur, fit = self.transform().m11(), self.get_fit_scale()
        if event.angleDelta().y() > 0:
            if cur < 50.0:
                self.scale(factor, factor)
                self._is_fitted = False
        else:
            if cur / factor <= fit + 0.001:
                self.reset_zoom()
                return
            self.scale(1 / factor, 1 / factor)
            self._is_fitted = False
        self._emit_change()

    def reset_zoom(self):
        if not self._has_image:
            return
        self.resetTransform()
        self.fitInView(self.pixmap_item, Qt.KeepAspectRatio)
        self._is_fitted = True
        self._emit_change()

    def _emit_change(self):
        if not self._is_syncing:
            self.view_changed.emit()

    def sync_state_from(self, other):
        if self._is_syncing or not self._has_image:
            return
        self._is_syncing = True
        self.setTransform(other.transform())
        self._is_fitted = other._is_fitted
        for bar, ob in ((self.horizontalScrollBar(), other.horizontalScrollBar()),
                        (self.verticalScrollBar(), other.verticalScrollBar())):
            if ob.maximum() > 0:
                bar.setValue(int(ob.value() / ob.maximum() * bar.maximum()))
        self._is_syncing = False


class SpectrumViewer(FigureCanvas):
    """Холст matplotlib: лог-спектр картинки, произвольное изображение или сообщение."""

    def __init__(self, parent=None):
        self.fig = Figure(figsize=(4, 3.5), dpi=100)
        self.fig.patch.set_facecolor("#f0f0f0")
        super().__init__(self.fig)
        self.setParent(parent)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_axis_off()
        self.fig.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.92)  # место под заголовок

    def update_spectrum(self, image, title="Спектр Фурье"):
        if image is None:
            return
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image
        m = max(gray.shape)
        if m > 1024:  # для показа спектра большая картинка не нужна, а FFT 12 МП съедает сотни МБ
            k = 1024 / m
            gray = cv2.resize(gray, (int(gray.shape[1] * k), int(gray.shape[0] * k)), interpolation=cv2.INTER_AREA)
        mag = np.log(np.abs(np.fft.fftshift(np.fft.fft2(gray))) + 1e-9)
        self.ax.clear()
        self.ax.imshow(mag, cmap="viridis")
        self.ax.set_title(title, fontsize=9)
        self.ax.set_axis_off()
        self.draw()

    def show_image_directly(self, image, title="Визуализация"):
        self.ax.clear()
        self.ax.imshow(image, cmap="gray")
        self.ax.set_title(title, fontsize=9)
        self.ax.set_axis_off()
        self.draw()

    def show_message(self, text, title=""):
        self.ax.clear()
        self.ax.set_axis_off()
        self.ax.set_title(title, fontsize=9)
        self.ax.text(0.5, 0.5, text, ha="center", va="center", transform=self.ax.transAxes, fontsize=9)
        self.draw()
