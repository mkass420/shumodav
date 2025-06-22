"""
Виджеты пользовательского интерфейса
"""

import cv2
import numpy as np
from PyQt5.QtWidgets import QLabel, QVBoxLayout, QHBoxLayout, QSlider, QWidget, QPushButton
from PyQt5.QtCore import Qt, pyqtSignal, QTimer
from PyQt5.QtGui import QPixmap, QImage, QKeyEvent
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class ZoomableImageView(QWidget):
    """Виджет для отображения изображения с возможностью зума"""
    
    # Сигналы для синхронизации
    zoom_changed = pyqtSignal(float)  # Сигнал изменения зума
    pan_changed = pyqtSignal(float, float)  # Сигнал изменения перемещения
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Создание layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Создание виджета для изображения
        self.image_label = QLabel()
        self.image_label.setMinimumSize(300, 300)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("QLabel { border: 2px solid #cccccc; background-color: #f0f0f0; }")
        layout.addWidget(self.image_label)
        
        # Панель управления зумом
        zoom_panel = QWidget()
        zoom_layout = QHBoxLayout(zoom_panel)
        zoom_layout.setContentsMargins(5, 5, 5, 5)
        
        # Кнопка сброса зума
        reset_btn = QPushButton("Сброс")
        reset_btn.clicked.connect(self.reset_zoom)
        reset_btn.setMaximumWidth(60)
        zoom_layout.addWidget(reset_btn)
        
        # Ползунок зума
        self.zoom_slider = QSlider(Qt.Horizontal)
        self.zoom_slider.setRange(10, 500)  # 0.1x - 5.0x
        self.zoom_slider.setValue(100)  # 1.0x
        self.zoom_slider.setMaximumWidth(150)
        self.zoom_slider.valueChanged.connect(self.on_slider_changed)
        zoom_layout.addWidget(self.zoom_slider)
        
        # Метка значения зума
        self.zoom_label = QLabel("1.0x")
        self.zoom_label.setMaximumWidth(40)
        zoom_layout.addWidget(self.zoom_label)
        
        zoom_layout.addStretch()
        layout.addWidget(zoom_panel)
        
        self.zoom_factor = 1.0
        self.original_pixmap = None
        self.current_pixmap = None
        
        # Параметры перемещения
        self.pan_x = 0.0  # Смещение по X (в процентах)
        self.pan_y = 0.0  # Смещение по Y (в процентах)
        self.is_panning = False
        self.last_mouse_pos = None
        
        # Флаг для предотвращения рекурсивных вызовов при синхронизации
        self.syncing = False
        
        # Установка фокуса для обработки клавиатуры
        self.setFocusPolicy(Qt.StrongFocus)
        
    def set_image(self, image):
        """Установка изображения"""
        if image is None:
            return
            
        # Конвертация numpy array в QPixmap
        if isinstance(image, np.ndarray):
            # Нормализация изображения в диапазон [0, 255]
            if image.dtype == np.float32:
                if image.max() <= 1.0:
                    image = (image * 255).astype(np.uint8)
                else:
                    image = image.astype(np.uint8)
            elif image.dtype != np.uint8:
                image = image.astype(np.uint8)
            
            # Конвертация в RGB если нужно
            if len(image.shape) == 3:
                if image.shape[2] == 3:
                    # Убеждаемся, что это RGB
                    if image.dtype == np.uint8:
                        height, width, channel = image.shape
                        bytes_per_line = 3 * width
                        q_image = QImage(image.data, width, height, bytes_per_line, QImage.Format_RGB888)
                    else:
                        q_image = QImage(image.tobytes(), image.shape[1], image.shape[0], 
                                       image.shape[1] * 3, QImage.Format_RGB888)
                else:
                    # Конвертация из BGR в RGB
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    height, width, channel = image_rgb.shape
                    bytes_per_line = 3 * width
                    q_image = QImage(image_rgb.data, width, height, bytes_per_line, QImage.Format_RGB888)
            else:
                # Серое изображение
                height, width = image.shape
                bytes_per_line = width
                q_image = QImage(image.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
                
            self.original_pixmap = QPixmap.fromImage(q_image)
        else:
            self.original_pixmap = image
            
        # Сброс перемещения при смене изображения
        self.pan_x = 0.0
        self.pan_y = 0.0
        self.update_zoom()
        
    def update_zoom(self):
        """Обновление зума и перемещения"""
        if self.original_pixmap is None:
            return
            
        # Масштабирование
        scaled_size = self.original_pixmap.size() * self.zoom_factor
        self.current_pixmap = self.original_pixmap.scaled(
            scaled_size, Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        
        # Применение перемещения
        if self.zoom_factor > 1.0:
            # Вычисляем максимальное смещение
            max_pan_x = (self.zoom_factor - 1.0) / self.zoom_factor
            max_pan_y = (self.zoom_factor - 1.0) / self.zoom_factor
            
            # Ограничиваем смещение
            self.pan_x = max(-max_pan_x, min(max_pan_x, self.pan_x))
            self.pan_y = max(-max_pan_y, min(max_pan_y, self.pan_y))
            
            # Вычисляем позицию для отображения
            x_offset = int(self.pan_x * self.current_pixmap.width())
            y_offset = int(self.pan_y * self.current_pixmap.height())
            
            # Создаем обрезанное изображение
            label_size = self.image_label.size()
            cropped_pixmap = self.current_pixmap.copy(
                x_offset, y_offset, 
                min(label_size.width(), self.current_pixmap.width() - x_offset),
                min(label_size.height(), self.current_pixmap.height() - y_offset)
            )
            
            self.image_label.setPixmap(cropped_pixmap)
        else:
            # Если зум <= 1.0, показываем все изображение
            self.image_label.setPixmap(self.current_pixmap)
        
    def on_slider_changed(self, value):
        """Обработка изменения ползунка зума"""
        if not self.syncing:
            self.zoom_factor = value / 100.0
            self.zoom_label.setText(f"{self.zoom_factor:.1f}x")
            self.update_zoom()
            # Отправляем сигнал для синхронизации
            self.zoom_changed.emit(self.zoom_factor)
        
    def wheelEvent(self, event):
        """Обработка колесика мыши для зума"""
        if self.original_pixmap is None:
            return
            
        # Изменение зума
        delta = event.angleDelta().y()
        if delta > 0:
            self.zoom_factor *= 1.1
        else:
            self.zoom_factor /= 1.1
            
        # Ограничение зума
        self.zoom_factor = max(0.1, min(5.0, self.zoom_factor))
        
        # Обновление ползунка
        self.zoom_slider.setValue(int(self.zoom_factor * 100))
        self.zoom_label.setText(f"{self.zoom_factor:.1f}x")
        self.update_zoom()
        
        # Отправляем сигнал для синхронизации
        if not self.syncing:
            self.zoom_changed.emit(self.zoom_factor)
        
    def mousePressEvent(self, event):
        """Обработка нажатия мыши для перемещения"""
        if event.button() == Qt.LeftButton and self.zoom_factor > 1.0:
            self.is_panning = True
            self.last_mouse_pos = event.pos()
            self.setCursor(Qt.ClosedHandCursor)
            
    def mouseReleaseEvent(self, event):
        """Обработка отпускания мыши"""
        if event.button() == Qt.LeftButton:
            self.is_panning = False
            self.setCursor(Qt.ArrowCursor)
            
    def mouseMoveEvent(self, event):
        """Обработка движения мыши для перемещения"""
        if self.is_panning and self.last_mouse_pos is not None:
            delta = event.pos() - self.last_mouse_pos
            
            # Вычисляем смещение в процентах
            pan_speed = 0.01  # Скорость перемещения
            self.pan_x += delta.x() * pan_speed
            self.pan_y += delta.y() * pan_speed
            
            self.last_mouse_pos = event.pos()
            self.update_zoom()
            
            # Отправляем сигнал для синхронизации
            if not self.syncing:
                self.pan_changed.emit(self.pan_x, self.pan_y)
        
    def keyPressEvent(self, event: QKeyEvent):
        """Обработка нажатий клавиш для зума и перемещения"""
        if self.original_pixmap is None:
            return
            
        pan_step = 0.05  # Шаг перемещения для клавиш
        
        if event.key() == Qt.Key_Plus or event.key() == Qt.Key_Equal:
            # Увеличение зума
            self.zoom_factor *= 1.2
        elif event.key() == Qt.Key_Minus:
            # Уменьшение зума
            self.zoom_factor /= 1.2
        elif event.key() == Qt.Key_0:
            # Сброс зума
            self.reset_zoom()
        elif event.key() == Qt.Key_Left:
            # Перемещение влево
            self.pan_x -= pan_step
        elif event.key() == Qt.Key_Right:
            # Перемещение вправо
            self.pan_x += pan_step
        elif event.key() == Qt.Key_Up:
            # Перемещение вверх
            self.pan_y -= pan_step
        elif event.key() == Qt.Key_Down:
            # Перемещение вниз
            self.pan_y += pan_step
        else:
            super().keyPressEvent(event)
            return
            
        # Ограничение зума
        self.zoom_factor = max(0.1, min(5.0, self.zoom_factor))
        
        # Обновление ползунка
        self.zoom_slider.setValue(int(self.zoom_factor * 100))
        self.zoom_label.setText(f"{self.zoom_factor:.1f}x")
        self.update_zoom()
        
        # Отправляем сигнал для синхронизации
        if not self.syncing:
            self.zoom_changed.emit(self.zoom_factor)
            self.pan_changed.emit(self.pan_x, self.pan_y)
        
    def reset_zoom(self):
        """Сброс зума и перемещения"""
        self.zoom_factor = 1.0
        self.pan_x = 0.0
        self.pan_y = 0.0
        self.zoom_slider.setValue(100)
        self.zoom_label.setText("1.0x")
        self.update_zoom()
        
        # Отправляем сигнал для синхронизации
        if not self.syncing:
            self.zoom_changed.emit(self.zoom_factor)
            self.pan_changed.emit(self.pan_x, self.pan_y)
        
    def restore_zoom_state(self, zoom_factor, pan_x, pan_y):
        """Восстановление состояния зума"""
        self.syncing = True  # Предотвращаем рекурсивные вызовы
        self.zoom_factor = zoom_factor
        self.pan_x = pan_x
        self.pan_y = pan_y
        self.zoom_slider.setValue(int(zoom_factor * 100))
        self.zoom_label.setText(f"{zoom_factor:.1f}x")
        self.update_zoom()
        self.syncing = False
        
    def sync_zoom(self, zoom_factor):
        """Синхронизация зума от других виджетов"""
        if not self.syncing:
            self.syncing = True
            self.zoom_factor = zoom_factor
            self.zoom_slider.setValue(int(zoom_factor * 100))
            self.zoom_label.setText(f"{zoom_factor:.1f}x")
            self.update_zoom()
            self.syncing = False
            
    def sync_pan(self, pan_x, pan_y):
        """Синхронизация перемещения от других виджетов"""
        if not self.syncing:
            self.syncing = True
            self.pan_x = pan_x
            self.pan_y = pan_y
            self.update_zoom()
            self.syncing = False


class SpectrumViewer(FigureCanvas):
    """Виджет для отображения спектра Фурье"""
    
    def __init__(self, parent=None):
        self.fig = Figure(figsize=(6, 4), dpi=100)
        super().__init__(self.fig)
        self.setParent(parent)
        
        self.ax = self.fig.add_subplot(111)
        self.ax.set_title('Спектр Фурье', fontsize=12, fontweight='bold')
        self.ax.set_xlabel('Частота X')
        self.ax.set_ylabel('Частота Y')
        
    def update_spectrum(self, image):
        """Обновление спектра"""
        if image is None:
            return
            
        # Конвертация в оттенки серого если нужно
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
            
        # Вычисление спектра Фурье
        f_transform = np.fft.fft2(gray)
        f_shift = np.fft.fftshift(f_transform)
        magnitude_spectrum = np.log(np.abs(f_shift) + 1)
        
        # Очистка и отображение
        self.ax.clear()
        self.ax.imshow(magnitude_spectrum, cmap='viridis')
        self.ax.set_title('Спектр Фурье', fontsize=12, fontweight='bold')
        self.ax.set_xlabel('Частота X')
        self.ax.set_ylabel('Частота Y')
        self.draw()


class FilterVisualizer(QWidget):
    """Интерактивный виджет для визуализации и редактирования фильтра"""
    
    # Сигналы для изменения параметров
    D0_changed = pyqtSignal(float)  # Сигнал изменения частоты среза
    n_changed = pyqtSignal(int)     # Сигнал изменения порядка фильтра
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Создаем layout для виджета
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        
        # Создаем canvas
        self.fig = Figure(figsize=(6, 4), dpi=100)
        self.canvas = FigureCanvas(self.fig)
        self.layout.addWidget(self.canvas)
        
        # Создаем панель управления под графиком
        self.control_panel = QWidget()
        control_layout = QHBoxLayout(self.control_panel)
        control_layout.setContentsMargins(5, 5, 5, 5)
        
        # Ползунок для параметра n (всегда видим)
        self.n_label = QLabel("Порядок (n):")
        control_layout.addWidget(self.n_label)
        
        self.n_slider = QSlider(Qt.Horizontal)
        self.n_slider.setRange(1, 10)
        self.n_slider.setValue(2)
        self.n_slider.valueChanged.connect(self.on_n_slider_changed)
        control_layout.addWidget(self.n_slider)
        
        self.n_value_label = QLabel("2")
        self.n_value_label.setMinimumWidth(30)
        control_layout.addWidget(self.n_value_label)
        
        control_layout.addStretch()
        self.layout.addWidget(self.control_panel)
        
        self.ax = self.fig.add_subplot(111)
        self.ax.set_title('Частотная характеристика фильтра (перетащите красную линию)', 
                         fontsize=12, fontweight='bold')
        self.ax.set_xlabel('Нормализованная частота')
        self.ax.set_ylabel('Коэффициент передачи')
        self.ax.grid(True, alpha=0.3)
        
        # Параметры фильтра
        self.filter_type = 'Butterworth'
        self.D0 = 0.2
        self.n = 2
        
        # Состояние интерактивности
        self.is_dragging = False
        self.drag_target = None  # 'D0'
        self.last_mouse_pos = None
        
        # Debounce для сигналов
        self.D0_debounce_timer = QTimer()
        self.D0_debounce_timer.setSingleShot(True)
        self.D0_debounce_timer.timeout.connect(self.emit_D0_changed)
        
        # Временные значения для debounce
        self.temp_D0 = self.D0
        
        # Подключение событий мыши
        self.canvas.mpl_connect('button_press_event', self.on_mouse_press)
        self.canvas.mpl_connect('button_release_event', self.on_mouse_release)
        self.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)
        
        # Инициализация графика
        self.update_filter()
        
    def emit_D0_changed(self):
        """Отправка сигнала изменения D0 с debounce"""
        print(f"DEBUG: FilterVisualizer.emit_D0_changed: отправляем D0 = {self.temp_D0}")
        self.D0_changed.emit(self.temp_D0)
        
    def update_filter(self, filter_type=None, D0=None, n=None):
        """Обновление визуализации фильтра"""
        if filter_type is not None:
            self.filter_type = filter_type
        if D0 is not None:
            self.D0 = D0
        if n is not None:
            self.n = n
            # Обновляем ползунок
            self.n_slider.setValue(n)
            self.n_value_label.setText(str(n))
            
        # Показываем/скрываем ползунок n только для Butterworth
        show_n_controls = (self.filter_type == 'Butterworth')
        self.n_label.setVisible(show_n_controls)
        self.n_slider.setVisible(show_n_controls)
        self.n_value_label.setVisible(show_n_controls)
            
        self.ax.clear()
        
        # Создание частотной оси
        freq = np.linspace(0, 1, 1000)
        
        if self.filter_type == 'Butterworth':
            # Классический Butterworth: H = 1 / (1 + (freq / D0)^(2*n))
            H = 1 / (1 + (freq / self.D0) ** (2 * self.n))
            label = f'Butterworth (n={self.n})'
            
        elif self.filter_type == 'Gaussian':
            # Классический Gaussian: H = exp(-(freq^2) / (2 * D0^2))
            H = np.exp(-(freq ** 2) / (2 * self.D0 ** 2))
            label = 'Gaussian'
            
        elif self.filter_type == 'Ideal':
            # Классический Ideal: H = 1 для freq <= D0, иначе 0
            H = (freq <= self.D0).astype(float)
            label = 'Ideal'
            
        else:
            H = np.ones_like(freq)
            label = 'None'
            
        # Построение графика фильтра
        self.ax.plot(freq, H, linewidth=2, label=label, color='blue')
        
        # Вертикальная линия D0 (интерактивная)
        self.D0_line = self.ax.axvline(x=self.D0, color='red', linestyle='--', 
                                      alpha=0.8, linewidth=3, label=f'D0 = {self.D0:.2f}')
        
        self.ax.set_title('Частотная характеристика фильтра (перетащите красную линию)', 
                         fontsize=12, fontweight='bold')
        self.ax.set_xlabel('Нормализованная частота')
        self.ax.set_ylabel('Коэффициент передачи')
        self.ax.grid(True, alpha=0.3)
        self.ax.legend()
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, 1.1)
        
        # Обновляем подсказки
        if show_n_controls:
            self.ax.text(0.02, 0.98, 'Перетащите красную линию для изменения D0\n'
                        'Используйте ползунок для изменения порядка n (Butterworth)', 
                        transform=self.ax.transAxes, fontsize=9, 
                        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        else:
            self.ax.text(0.02, 0.98, 'Перетащите красную линию для изменения D0', 
                        transform=self.ax.transAxes, fontsize=9, 
                        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        self.canvas.draw()
        
    def on_mouse_press(self, event):
        """Обработка нажатия мыши"""
        if event.inaxes != self.ax:
            return
            
        if event.button == 1:  # Левая кнопка мыши
            print(f"DEBUG: FilterVisualizer.on_mouse_press: нажатие мыши в позиции ({event.xdata:.3f}, {event.ydata:.3f})")
            # Проверяем, близко ли мышь к вертикальной линии D0
            if abs(event.xdata - self.D0) < 0.02:  # Порог близости
                print("DEBUG: FilterVisualizer.on_mouse_press: начинаем перетаскивание D0")
                self.is_dragging = True
                self.drag_target = 'D0'
                self.last_mouse_pos = event.xdata
                self.setCursor(Qt.SizeHorCursor)
                return
                    
    def on_mouse_release(self, event):
        """Обработка отпускания мыши"""
        if self.is_dragging:
            self.is_dragging = False
            self.drag_target = None
            self.last_mouse_pos = None
            self.setCursor(Qt.ArrowCursor)
            
    def on_mouse_move(self, event):
        """Обработка движения мыши"""
        if not self.is_dragging or event.inaxes != self.ax:
            return
            
        if self.drag_target == 'D0':
            # Изменение D0
            new_D0 = max(0.01, min(0.99, event.xdata))
            if abs(new_D0 - self.D0) > 0.001:  # Минимальное изменение
                self.temp_D0 = new_D0
                self.D0 = new_D0  # Обновляем для отображения
                self.update_filter()
                # Запускаем debounce таймер
                self.D0_debounce_timer.stop()
                self.D0_debounce_timer.start(50)  # 50ms задержка
                
    def on_n_slider_changed(self, value):
        """Обработчик изменения ползунка n"""
        print(f"DEBUG: FilterVisualizer.on_n_slider_changed: n = {value}")
        self.n = value
        self.n_value_label.setText(str(value))
        self.update_filter()
        # Отправляем сигнал
        self.n_changed.emit(value) 