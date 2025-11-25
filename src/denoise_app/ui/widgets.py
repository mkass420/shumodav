import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QGraphicsView,
    QGraphicsScene,
    QGraphicsPixmapItem,
    QWidget,
    QPushButton,
)
from PyQt5.QtCore import Qt, pyqtSignal, QRectF
from PyQt5.QtGui import QPixmap, QImage, QPainter

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class ZoomableImageView(QGraphicsView):
    view_changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)

        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        self.pixmap_item = QGraphicsPixmapItem()
        self.scene.addItem(self.pixmap_item)

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
        self.reset_btn.setStyleSheet("""
            QPushButton {
                background-color: rgba(0, 0, 0, 150);
                color: white;
                border: none;
                border-radius: 4px;
                padding: 4px 8px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: rgba(0, 0, 0, 200);
            }
        """)
        self.reset_btn.clicked.connect(self.reset_zoom)
        self.reset_btn.hide()

        self.horizontalScrollBar().valueChanged.connect(self._emit_change)
        self.verticalScrollBar().valueChanged.connect(self._emit_change)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.reset_btn.move(self.width() - self.reset_btn.width() - 10, 10)

        if self._has_image and self._is_fitted:
            self.fitInView(self.pixmap_item, Qt.KeepAspectRatio)

    def set_image(self, image, maintain_zoom=False):
        if image is None:
            return

        was_empty = not self._has_image

        if image.dtype == np.float32 or image.dtype == np.float64:
            img_8bit = (np.clip(image, 0, 1) * 255).astype(np.uint8)
        else:
            img_8bit = image.astype(np.uint8)

        height, width = img_8bit.shape[:2]
        if len(img_8bit.shape) == 3:
            q_img = QImage(
                img_8bit.data, width, height, 3 * width, QImage.Format_RGB888
            )
        else:
            q_img = QImage(
                img_8bit.data, width, height, width, QImage.Format_Grayscale8
            )

        pixmap = QPixmap.fromImage(q_img)
        self.pixmap_item.setPixmap(pixmap)
        self.scene.setSceneRect(QRectF(pixmap.rect()))

        self._has_image = True
        self.reset_btn.show()

        if was_empty or not maintain_zoom:
            self.reset_zoom()

    def get_fit_scale(self):
        if not self._has_image:
            return 1.0

        view_rect = self.viewport().rect()
        scene_rect = self.pixmap_item.boundingRect()

        if scene_rect.width() == 0 or scene_rect.height() == 0:
            return 1.0

        ratio_w = view_rect.width() / scene_rect.width()
        ratio_h = view_rect.height() / scene_rect.height()

        return min(ratio_w, ratio_h)

    def wheelEvent(self, event):
        if not self._has_image:
            return

        zoom_in_factor = 1.25
        zoom_out_factor = 1 / zoom_in_factor

        current_scale = self.transform().m11()
        fit_scale = self.get_fit_scale()

        epsilon = 0.001

        if event.angleDelta().y() > 0:
            max_scale = 50.0
            if current_scale < max_scale:
                self.scale(zoom_in_factor, zoom_in_factor)
                self._is_fitted = False
        else:
            next_scale = current_scale * zoom_out_factor
            if next_scale <= fit_scale + epsilon:
                self.reset_zoom()
                return
            else:
                self.scale(zoom_out_factor, zoom_out_factor)
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

    def sync_state_from(self, other_view):
        if self._is_syncing or not self._has_image:
            return
        self._is_syncing = True

        self.setTransform(other_view.transform())

        if hasattr(other_view, "_is_fitted"):
            self._is_fitted = other_view._is_fitted

        h_bar = self.horizontalScrollBar()
        v_bar = self.verticalScrollBar()
        other_h = other_view.horizontalScrollBar()
        other_v = other_view.verticalScrollBar()

        if other_h.maximum() > 0:
            h_ratio = other_h.value() / other_h.maximum()
            h_bar.setValue(int(h_ratio * h_bar.maximum()))

        if other_v.maximum() > 0:
            v_ratio = other_v.value() / other_v.maximum()
            v_bar.setValue(int(v_ratio * v_bar.maximum()))

        self._is_syncing = False


class SpectrumViewer(FigureCanvas):
    def __init__(self, parent=None):
        self.fig = Figure(figsize=(5, 4), dpi=100)
        self.fig.patch.set_facecolor("#f0f0f0")
        super().__init__(self.fig)
        self.setParent(parent)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_axis_off()
        self.fig.tight_layout(pad=0)

    def update_spectrum(self, image):
        if image is None:
            return
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        f_transform = np.fft.fft2(gray)
        f_shift = np.fft.fftshift(f_transform)
        magnitude_spectrum = np.log(np.abs(f_shift) + 1e-9)
        self.ax.clear()
        self.ax.imshow(magnitude_spectrum, cmap="viridis")
        self.ax.set_title("Спектр Фурье", fontsize=9)
        self.ax.set_axis_off()
        self.draw()

    def show_image_directly(self, image, title="Визуализация"):
        if image is None:
            return
        self.ax.clear()
        self.ax.imshow(image, cmap="gray")
        self.ax.set_title(title, fontsize=9)
        self.ax.set_axis_off()
        self.draw()


class FilterVisualizer(QWidget):
    pass
