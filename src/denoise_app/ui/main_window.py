"""
Главное окно приложения
"""

import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QLabel, QSlider, QComboBox, QPushButton, QFileDialog, 
                             QGroupBox, QGridLayout, QTabWidget, QFrame, QSpinBox,
                             QProgressBar, QMessageBox, QCheckBox)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QThread, pyqtSlot
from PyQt5.QtGui import QFont

from ..core.image_processor import ImageProcessor
from .widgets import ZoomableImageView, SpectrumViewer, FilterVisualizer


class FastFilteringThread(QThread):
    """Быстрый поток для обработки только фильтра"""
    
    processing_finished = pyqtSignal(object)
    processing_error = pyqtSignal(str)
    
    def __init__(self, image_processor, input_image):
        super().__init__()
        self.image_processor = image_processor
        self.input_image = input_image
        
    def run(self):
        """Выполнение быстрой фильтрации"""
        try:
            print("DEBUG: FastFilteringThread.run: начинаем быструю фильтрацию")
            print(f"DEBUG: FastFilteringThread.run: размер входного изображения = {self.input_image.shape}")
            print(f"DEBUG: FastFilteringThread.run: тип данных = {self.input_image.dtype}")
            print(f"DEBUG: FastFilteringThread.run: диапазон = [{self.input_image.min():.3f}, {self.input_image.max():.3f}]")
            
            # Применяем только фильтр без шума и улучшений
            filtered_image = self.image_processor.apply_frequency_filter(self.input_image)
            
            print(f"DEBUG: FastFilteringThread.run: фильтрация завершена")
            print(f"DEBUG: FastFilteringThread.run: размер результата = {filtered_image.shape}")
            print(f"DEBUG: FastFilteringThread.run: диапазон результата = [{filtered_image.min():.3f}, {filtered_image.max():.3f}]")
            
            self.processing_finished.emit(filtered_image)
            
        except Exception as e:
            print(f"DEBUG: FastFilteringThread.run: ошибка - {e}")
            import traceback
            traceback.print_exc()
            self.processing_error.emit(str(e))


class ImageProcessingThread(QThread):
    """Поток для обработки изображений"""
    
    processing_finished = pyqtSignal(object, object, object)
    processing_error = pyqtSignal(str)
    progress_updated = pyqtSignal(int)
    
    def __init__(self, image_processor, original_image, enable_noise=True, enable_enhancement=True):
        super().__init__()
        self.image_processor = image_processor
        self.original_image = original_image
        self.enable_noise = enable_noise
        self.enable_enhancement = enable_enhancement
        
    def run(self):
        """Выполнение обработки в отдельном потоке"""
        try:
            print(f"DEBUG: ImageProcessingThread.run: начинаем обработку")
            print(f"DEBUG: ImageProcessingThread.run: enable_noise={self.enable_noise}, enable_enhancement={self.enable_enhancement}")
            
            self.progress_updated.emit(10)
            
            # Добавление шума (если включено)
            if self.enable_noise:
                noisy_image = self.image_processor.noise_generator.add_periodic_noise(self.original_image)
                print(f"DEBUG: ImageProcessingThread.run: шум добавлен, диапазон=[{noisy_image.min():.3f}, {noisy_image.max():.3f}]")
            else:
                # Если шум отключен, используем оригинальное изображение
                if self.original_image.dtype == np.float32:
                    noisy_image = self.original_image.copy()
                else:
                    noisy_image = self.original_image.astype(np.float32) / 255.0
                print(f"DEBUG: ImageProcessingThread.run: шум отключен, используем оригинал, диапазон=[{noisy_image.min():.3f}, {noisy_image.max():.3f}]")
                    
            self.progress_updated.emit(30)
            
            # Применение фильтра
            filtered_image = self.image_processor.apply_frequency_filter(noisy_image)
            print(f"DEBUG: ImageProcessingThread.run: фильтр применен, диапазон=[{filtered_image.min():.3f}, {filtered_image.max():.3f}]")
            self.progress_updated.emit(70)
            
            # Улучшение деталей (если включено)
            if self.enable_enhancement:
                print(f"DEBUG: ImageProcessingThread.run: применяем улучшение деталей")
                filtered_image = self.image_processor.enhance_details(filtered_image)
                print(f"DEBUG: ImageProcessingThread.run: улучшение применено, диапазон=[{filtered_image.min():.3f}, {filtered_image.max():.3f}]")
            else:
                print(f"DEBUG: ImageProcessingThread.run: улучшение отключено")
            self.progress_updated.emit(90)
            
            # Вычисление спектров
            original_spectrum = self.image_processor.compute_spectrum(self.original_image)
            noisy_spectrum = self.image_processor.compute_spectrum(noisy_image)
            filtered_spectrum = self.image_processor.compute_spectrum(filtered_image)
            
            self.progress_updated.emit(100)
            
            spectra = (original_spectrum, noisy_spectrum, filtered_spectrum)
            print(f"DEBUG: ImageProcessingThread.run: обработка завершена, отправляем результаты")
            self.processing_finished.emit(noisy_image, filtered_image, spectra)
            
        except Exception as e:
            print(f"DEBUG: ImageProcessingThread.run: ошибка - {e}")
            import traceback
            traceback.print_exc()
            self.processing_error.emit(str(e))


class MainWindow(QMainWindow):
    """Главное окно приложения"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Современное приложение для шумоподавления")
        self.setGeometry(100, 100, 1400, 900)
        
        # Инициализация процессора изображений
        self.image_processor = ImageProcessor()
        
        # Переменные для изображений
        self.original_image = None
        self.noisy_image = None
        self.filtered_image = None
        self.spectra = None
        
        # Параметры по умолчанию
        self.filter_type = 'Butterworth'
        self.D0 = 0.2
        self.n = 2
        self.pad_ratio = 0.2
        self.noise_freq = 0.1
        self.noise_amp = 0.3
        self.noise_angle = 45
        self.sigma_s = 0.1  # Уменьшаем с 1 до 0.1 для более деликатного контроля
        self.sigma_r = 0.05  # Уменьшаем с 0.15 до 0.05 для более деликатного начального эффекта
        self.contrast_strength = 0.0  # Уменьшаем с 0.1 до 0.0 - контраст отключен по умолчанию
        self.enable_noise = True  # Флаг для включения/отключения шума
        self.enable_enhancement = True  # Флаг для включения/отключения улучшения
        self.sync_zoom = True  # Флаг для синхронизации зума
        
        # Флаг для предотвращения множественных вызовов
        self.processing_in_progress = False
        self.processing_thread = None
        
        # Таймер для отложенной обработки
        self.debounce_timer = QTimer()
        self.debounce_timer.setSingleShot(True)
        self.debounce_timer.timeout.connect(self.process_image)
        
        self.setup_ui()
        self.update_noise_visibility()  # Применяем правильную видимость при запуске
        self.update_enhancement_visibility()  # Применяем правильную видимость улучшения при запуске
        
        # Инициализируем видимость ползунка n в зависимости от типа фильтра
        show_n_controls = (self.filter_type == 'Butterworth')
        if hasattr(self, 'n_slider'):
            self.n_slider.setVisible(show_n_controls)
            self.n_label.setVisible(show_n_controls)
        
    def setup_ui(self):
        """Настройка интерфейса"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Главный layout
        main_layout = QHBoxLayout(central_widget)
        
        # Левая панель с контролами
        control_panel = self.create_control_panel()
        main_layout.addWidget(control_panel, 1)
        
        # Правая панель с изображениями
        image_panel = self.create_image_panel()
        main_layout.addWidget(image_panel, 3)
        
    def create_control_panel(self):
        """Создание панели управления"""
        panel = QFrame()
        panel.setFrameStyle(QFrame.StyledPanel)
        panel.setMaximumWidth(350)
        panel.setMinimumWidth(300)
        
        layout = QVBoxLayout(panel)
        
        # Заголовок
        title = QLabel("Параметры шумоподавления")
        title.setFont(QFont("Arial", 14, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        
        # Кнопка загрузки
        load_btn = QPushButton("Загрузить изображение")
        load_btn.clicked.connect(self.load_image)
        load_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 10px;
                border-radius: 5px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        layout.addWidget(load_btn)
        
        # Индикатор прогресса
        progress_group = QGroupBox("Прогресс обработки")
        progress_layout = QVBoxLayout(progress_group)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        progress_layout.addWidget(self.progress_bar)
        
        self.status_label = QLabel("Готов к работе")
        self.status_label.setAlignment(Qt.AlignCenter)
        progress_layout.addWidget(self.status_label)
        
        layout.addWidget(progress_group)
        
        # Тип фильтра
        filter_group = QGroupBox("Тип фильтра")
        filter_layout = QVBoxLayout(filter_group)
        
        self.filter_combo = QComboBox()
        self.filter_combo.addItems(['Butterworth', 'Gaussian', 'Ideal'])
        self.filter_combo.currentTextChanged.connect(self.on_filter_changed)
        filter_layout.addWidget(self.filter_combo)
        layout.addWidget(filter_group)
        
        # Параметры фильтра
        params_group = QGroupBox("Параметры фильтра")
        params_layout = QGridLayout(params_group)
        
        # Частота среза
        params_layout.addWidget(QLabel("Частота среза (D0):"), 0, 0)
        self.D0_slider = QSlider(Qt.Horizontal)
        self.D0_slider.setRange(1, 50)
        self.D0_slider.setValue(int(self.D0 * 100))
        self.D0_slider.valueChanged.connect(self.on_D0_changed)
        params_layout.addWidget(self.D0_slider, 0, 1)
        
        self.D0_label = QLabel(f"{self.D0:.2f}")
        params_layout.addWidget(self.D0_label, 0, 2)
        
        # Порядок фильтра (n) - только для Butterworth
        self.n_label_title = QLabel("Порядок фильтра (n):")
        params_layout.addWidget(self.n_label_title, 1, 0)
        self.n_slider = QSlider(Qt.Horizontal)
        self.n_slider.setRange(1, 10)
        self.n_slider.setValue(self.n)
        self.n_slider.valueChanged.connect(self.on_n_changed)
        params_layout.addWidget(self.n_slider, 1, 1)
        
        self.n_label = QLabel(f"{self.n}")
        params_layout.addWidget(self.n_label, 1, 2)
        
        # Дополнение границ
        params_layout.addWidget(QLabel("Дополнение границ:"), 2, 0)
        self.pad_slider = QSlider(Qt.Horizontal)
        self.pad_slider.setRange(10, 50)
        self.pad_slider.setValue(int(self.pad_ratio * 100))
        self.pad_slider.valueChanged.connect(self.on_pad_changed)
        params_layout.addWidget(self.pad_slider, 2, 1)
        
        self.pad_label = QLabel(f"{self.pad_ratio:.2f}")
        params_layout.addWidget(self.pad_label, 2, 2)
        
        layout.addWidget(params_group)
        
        # Параметры шума
        noise_group = QGroupBox("Параметры шума")
        noise_layout = QGridLayout(noise_group)
        
        # Чекбокс для включения/отключения шума
        self.noise_checkbox = QCheckBox("Включить генерацию шума")
        self.noise_checkbox.setChecked(self.enable_noise)
        self.noise_checkbox.stateChanged.connect(self.on_noise_enabled_changed)
        noise_layout.addWidget(self.noise_checkbox, 0, 0, 1, 3)
        
        # Контейнер для параметров шума
        self.noise_params_widget = QWidget()
        noise_params_layout = QGridLayout(self.noise_params_widget)
        
        # Частота шума
        noise_params_layout.addWidget(QLabel("Частота шума:"), 0, 0)
        self.noise_freq_slider = QSlider(Qt.Horizontal)
        self.noise_freq_slider.setRange(1, 50)
        self.noise_freq_slider.setValue(int(self.noise_freq * 100))
        self.noise_freq_slider.valueChanged.connect(self.on_noise_freq_changed)
        noise_params_layout.addWidget(self.noise_freq_slider, 0, 1)
        
        self.noise_freq_label = QLabel(f"{self.noise_freq:.2f}")
        noise_params_layout.addWidget(self.noise_freq_label, 0, 2)
        
        # Амплитуда шума
        noise_params_layout.addWidget(QLabel("Амплитуда шума:"), 1, 0)
        self.noise_amp_slider = QSlider(Qt.Horizontal)
        self.noise_amp_slider.setRange(10, 100)
        self.noise_amp_slider.setValue(int(self.noise_amp * 100))
        self.noise_amp_slider.valueChanged.connect(self.on_noise_amp_changed)
        noise_params_layout.addWidget(self.noise_amp_slider, 1, 1)
        
        self.noise_amp_label = QLabel(f"{self.noise_amp:.2f}")
        noise_params_layout.addWidget(self.noise_amp_label, 1, 2)
        
        # Угол шума
        noise_params_layout.addWidget(QLabel("Угол шума:"), 2, 0)
        self.noise_angle_slider = QSlider(Qt.Horizontal)
        self.noise_angle_slider.setRange(0, 180)
        self.noise_angle_slider.setValue(int(self.noise_angle))
        self.noise_angle_slider.valueChanged.connect(self.on_noise_angle_changed)
        noise_params_layout.addWidget(self.noise_angle_slider, 2, 1)
        
        self.noise_angle_label = QLabel(f"{self.noise_angle:.0f}°")
        noise_params_layout.addWidget(self.noise_angle_label, 2, 2)
        
        noise_layout.addWidget(self.noise_params_widget, 1, 0, 1, 3)
        
        layout.addWidget(noise_group)
        
        # Параметры улучшения
        enhance_group = QGroupBox("Улучшение изображения")
        enhance_layout = QGridLayout(enhance_group)
        
        # Чекбокс для включения/отключения улучшения
        self.enhancement_checkbox = QCheckBox("Включить улучшение изображения")
        self.enhancement_checkbox.setChecked(self.enable_enhancement)
        self.enhancement_checkbox.stateChanged.connect(self.on_enhancement_enabled_changed)
        enhance_layout.addWidget(self.enhancement_checkbox, 0, 0, 1, 3)
        
        # Контейнер для параметров улучшения
        self.enhancement_params_widget = QWidget()
        enhancement_params_layout = QGridLayout(self.enhancement_params_widget)
        
        # Резкость
        enhancement_params_layout.addWidget(QLabel("Сила резкости (sigma_s):"), 0, 0)
        self.sigma_s_slider = QSlider(Qt.Horizontal)
        self.sigma_s_slider.setRange(1, 100)
        self.sigma_s_slider.setValue(int(self.sigma_s * 10))
        self.sigma_s_slider.valueChanged.connect(self.on_sigma_s_changed)
        enhancement_params_layout.addWidget(self.sigma_s_slider, 0, 1)
        
        self.sigma_s_label = QLabel(f"{self.sigma_s:.1f}")
        enhancement_params_layout.addWidget(self.sigma_s_label, 0, 2)
        
        # Детализация
        enhancement_params_layout.addWidget(QLabel("Усиление деталей (sigma_r):"), 1, 0)
        self.sigma_r_slider = QSlider(Qt.Horizontal)
        self.sigma_r_slider.setRange(1, 50)  # Увеличиваем диапазон с 1-20 до 1-50 (0.01-0.50)
        self.sigma_r_slider.setValue(int(self.sigma_r * 100))
        self.sigma_r_slider.valueChanged.connect(self.on_sigma_r_changed)
        enhancement_params_layout.addWidget(self.sigma_r_slider, 1, 1)
        
        self.sigma_r_label = QLabel(f"{self.sigma_r:.2f}")
        enhancement_params_layout.addWidget(self.sigma_r_label, 1, 2)
        
        # Контраст
        enhancement_params_layout.addWidget(QLabel("Усиление контраста:"), 2, 0)
        self.contrast_slider = QSlider(Qt.Horizontal)
        self.contrast_slider.setRange(0, 50)  # Диапазон 0.0-0.50
        self.contrast_slider.setValue(int(self.contrast_strength * 100))
        self.contrast_slider.valueChanged.connect(self.on_contrast_changed)
        enhancement_params_layout.addWidget(self.contrast_slider, 2, 1)
        
        self.contrast_label = QLabel(f"{self.contrast_strength:.2f}")
        enhancement_params_layout.addWidget(self.contrast_label, 2, 2)
        
        enhance_layout.addWidget(self.enhancement_params_widget, 1, 0, 1, 3)
        
        layout.addWidget(enhance_group)
        
        # --- Смешивание с оригиналом ---
        blend_group = QGroupBox("Смешивание с оригиналом")
        blend_layout = QGridLayout(blend_group)

        self.blend_checkbox = QCheckBox("Включить смешивание")
        self.blend_checkbox.setChecked(True)
        self.blend_checkbox.stateChanged.connect(self.on_blend_enabled_changed)
        blend_layout.addWidget(self.blend_checkbox, 0, 0, 1, 2)

        # Контейнер для параметров смешивания
        self.blend_params_widget = QWidget()
        blend_params_layout = QGridLayout(self.blend_params_widget)
        
        blend_params_layout.addWidget(QLabel("Коэффициент смешивания (α):"), 0, 0)
        self.blend_slider = QSlider(Qt.Horizontal)
        self.blend_slider.setRange(0, 100)
        self.blend_slider.setValue(50)
        self.blend_slider.valueChanged.connect(self.on_blend_alpha_changed)
        blend_params_layout.addWidget(self.blend_slider, 0, 1)
        
        self.blend_label = QLabel("0.50")
        blend_params_layout.addWidget(self.blend_label, 0, 2)
        
        blend_layout.addWidget(self.blend_params_widget, 1, 0, 1, 3)

        layout.addWidget(blend_group)
        
        # GPU ускорение
        gpu_group = QGroupBox("GPU ускорение")
        gpu_layout = QGridLayout(gpu_group)
        
        # GPU/CPU переключатель
        gpu_layout.addWidget(QLabel("Ускорение:"), 0, 0)
        self.gpu_checkbox = QCheckBox("Использовать GPU")
        
        # Определяем доступное ускорение
        try:
            import cv2
            # Проверяем CUDA
            if hasattr(cv2, 'cuda'):
                try:
                    device_count = cv2.cuda.getCudaEnabledDeviceCount()
                    if device_count == 0:
                        raise Exception("Нет доступных CUDA устройств")
                    
                    # Дополнительная проверка - пытаемся создать GpuMat и выполнить операцию
                    test_gpu_mat = cv2.cuda_GpuMat()
                    test_array = np.array([[1, 2], [3, 4]], dtype=np.uint8)
                    test_gpu_mat.upload(test_array)
                    test_result = test_gpu_mat.download()
                    
                    gpu_type = "CUDA"
                except Exception as e:
                    print(f"DEBUG: CUDA проверка не прошла: {e}")
                    # Проверяем OpenCL
                    if hasattr(cv2, 'ocl'):
                        try:
                            cv2.ocl.useOpenCL()
                            # Дополнительная проверка - пытаемся выполнить простую операцию
                            test_array = np.array([[1, 2], [3, 4]], dtype=np.uint8)
                            test_blur = cv2.GaussianBlur(test_array, (3, 3), 0)
                            gpu_type = "OpenCL"
                        except Exception as e:
                            print(f"DEBUG: OpenCL проверка не прошла: {e}")
                            gpu_type = "CPU"
                    else:
                        gpu_type = "CPU"
            # Проверяем только OpenCL если CUDA нет
            elif hasattr(cv2, 'ocl'):
                try:
                    cv2.ocl.useOpenCL()
                    # Дополнительная проверка - пытаемся выполнить простую операцию
                    test_array = np.array([[1, 2], [3, 4]], dtype=np.uint8)
                    test_blur = cv2.GaussianBlur(test_array, (3, 3), 0)
                    gpu_type = "OpenCL"
                except Exception as e:
                    print(f"DEBUG: OpenCL проверка не прошла: {e}")
                    gpu_type = "CPU"
            else:
                gpu_type = "CPU"
        except Exception as e:
            print(f"DEBUG: Ошибка при проверке GPU: {e}")
            gpu_type = "CPU"
            
        self.gpu_checkbox.setText(f"Использовать GPU ({gpu_type})")
        self.gpu_checkbox.setChecked(gpu_type != "CPU")  # По умолчанию GPU если доступен
        self.gpu_checkbox.setEnabled(gpu_type != "CPU")  # Отключаем если GPU недоступен
        self.gpu_checkbox.stateChanged.connect(self.on_gpu_mode_changed)
        gpu_layout.addWidget(self.gpu_checkbox, 0, 1, 1, 2)
        
        layout.addWidget(gpu_group)
        
        # Режим обработки
        processing_group = QGroupBox("Режим обработки")
        processing_layout = QVBoxLayout(processing_group)
        
        self.fast_processing_checkbox = QCheckBox("Быстрая обработка (только фильтр)")
        self.fast_processing_checkbox.setChecked(True)
        self.fast_processing_checkbox.stateChanged.connect(self.on_fast_processing_changed)
        processing_layout.addWidget(self.fast_processing_checkbox)
        
        fast_info = QLabel("Быстрая обработка применяет только фильтр без шума и улучшений для интерактивного редактирования")
        fast_info.setWordWrap(True)
        fast_info.setStyleSheet("QLabel { color: #666; font-size: 10px; }")
        processing_layout.addWidget(fast_info)
        
        layout.addWidget(processing_group)
        
        # Синхронизация изображений
        sync_group = QGroupBox("Синхронизация")
        sync_layout = QGridLayout(sync_group)
        
        self.sync_checkbox = QCheckBox("Синхронизировать зум и перемещение")
        self.sync_checkbox.setChecked(self.sync_zoom)
        self.sync_checkbox.stateChanged.connect(self.on_sync_zoom_changed)
        sync_layout.addWidget(self.sync_checkbox, 0, 0, 1, 3)
        
        sync_info = QLabel("При включении зум и перемещение будут синхронизированы между всеми изображениями")
        sync_info.setWordWrap(True)
        sync_info.setStyleSheet("QLabel { color: #666; font-size: 10px; }")
        sync_layout.addWidget(sync_info, 1, 0, 1, 3)
        
        layout.addWidget(sync_group)
        
        # Визуализация фильтра
        self.filter_viz = FilterVisualizer()
        # Подключаем сигналы интерактивного редактирования
        self.filter_viz.D0_changed.connect(self.on_D0_changed_interactive)
        self.filter_viz.n_changed.connect(self.on_n_changed_interactive)
        layout.addWidget(self.filter_viz)
        
        # Кнопки управления
        button_layout = QVBoxLayout()
        
        # Кнопка автоматической оптимизации
        auto_optimize_btn = QPushButton("Авто-оптимизация")
        auto_optimize_btn.clicked.connect(self.auto_optimize_parameters)
        button_layout.addWidget(auto_optimize_btn)
        
        # Кнопка полной обработки
        full_process_btn = QPushButton("Полная обработка")
        full_process_btn.clicked.connect(self.force_full_processing)
        button_layout.addWidget(full_process_btn)
        
        # Кнопка сброса зума
        reset_zoom_btn = QPushButton("Сброс зума")
        reset_zoom_btn.clicked.connect(self.reset_zooms)
        button_layout.addWidget(reset_zoom_btn)
        
        # Кнопка сохранения
        save_btn = QPushButton("Сохранить")
        save_btn.clicked.connect(self.save_image)
        button_layout.addWidget(save_btn)
        
        layout.addLayout(button_layout)
        
        layout.addStretch()
        
        # Инициализация видимости элементов управления порядком фильтра
        show_n_controls = (self.filter_type == 'Butterworth')
        self.n_label_title.setVisible(show_n_controls)
        self.n_slider.setVisible(show_n_controls)
        self.n_label.setVisible(show_n_controls)
        
        return panel
        
    def create_image_panel(self):
        """Создание панели с изображениями"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # Вкладки
        self.tab_widget = QTabWidget()
        
        # Вкладка изображений
        image_tab = QWidget()
        image_layout = QHBoxLayout(image_tab)
        
        # Оригинал
        original_group = QGroupBox("Оригинал")
        original_layout = QVBoxLayout(original_group)
        self.original_view = ZoomableImageView()
        # Подключаем сигналы синхронизации
        self.original_view.zoom_changed.connect(lambda v: self.sync_zoom_to_others(self.original_view, v * 100))
        self.original_view.pan_changed.connect(lambda x, y: self.sync_pan_to_others(self.original_view, x, y))
        original_layout.addWidget(self.original_view)
        image_layout.addWidget(original_group)
        
        # Зашумленное
        noisy_group = QGroupBox("Зашумленное")
        noisy_layout = QVBoxLayout(noisy_group)
        self.noisy_view = ZoomableImageView()
        # Подключаем сигналы синхронизации
        self.noisy_view.zoom_changed.connect(lambda v: self.sync_zoom_to_others(self.noisy_view, v * 100))
        self.noisy_view.pan_changed.connect(lambda x, y: self.sync_pan_to_others(self.noisy_view, x, y))
        noisy_layout.addWidget(self.noisy_view)
        image_layout.addWidget(noisy_group)
        
        # Отфильтрованное
        filtered_group = QGroupBox("Отфильтрованное")
        filtered_layout = QVBoxLayout(filtered_group)
        self.filtered_view = ZoomableImageView()
        # Подключаем сигналы синхронизации
        self.filtered_view.zoom_changed.connect(lambda v: self.sync_zoom_to_others(self.filtered_view, v * 100))
        self.filtered_view.pan_changed.connect(lambda x, y: self.sync_pan_to_others(self.filtered_view, x, y))
        filtered_layout.addWidget(self.filtered_view)
        image_layout.addWidget(filtered_group)
        
        self.tab_widget.addTab(image_tab, "Изображения")
        
        # Вкладка спектров
        spectrum_tab = QWidget()
        spectrum_layout = QHBoxLayout(spectrum_tab)
        
        # Спектр оригинала
        orig_spectrum_group = QGroupBox("Спектр оригинала")
        orig_spectrum_layout = QVBoxLayout(orig_spectrum_group)
        self.original_spectrum_view = SpectrumViewer()
        orig_spectrum_layout.addWidget(self.original_spectrum_view)
        spectrum_layout.addWidget(orig_spectrum_group)
        
        # Спектр зашумленного
        noisy_spectrum_group = QGroupBox("Спектр зашумленного")
        noisy_spectrum_layout = QVBoxLayout(noisy_spectrum_group)
        self.noisy_spectrum_view = SpectrumViewer()
        noisy_spectrum_layout.addWidget(self.noisy_spectrum_view)
        spectrum_layout.addWidget(noisy_spectrum_group)
        
        # Спектр отфильтрованного
        filtered_spectrum_group = QGroupBox("Спектр отфильтрованного")
        filtered_spectrum_layout = QVBoxLayout(filtered_spectrum_group)
        self.filtered_spectrum_view = SpectrumViewer()
        filtered_spectrum_layout.addWidget(self.filtered_spectrum_view)
        spectrum_layout.addWidget(filtered_spectrum_group)
        
        self.tab_widget.addTab(spectrum_tab, "Спектры")
        
        layout.addWidget(self.tab_widget)
        return panel
        
    def load_image(self):
        """Загрузка изображения"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Выберите изображение", "", 
            "Image files (*.jpg *.jpeg *.png *.bmp *.tiff)"
        )
        
        if file_path:
            try:
                print(f"DEBUG: load_image: загружаем изображение из {file_path}")
                self.original_image = cv2.imread(file_path)
                if self.original_image is not None:
                    self.original_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
                    print(f"DEBUG: load_image: изображение загружено, размер = {self.original_image.shape}")
                    # Запускаем обработку сразу
                    self.process_image()
                    self.status_label.setText("Изображение загружено")
                else:
                    print("DEBUG: load_image: не удалось загрузить изображение")
                    QMessageBox.warning(self, "Ошибка", "Не удалось загрузить изображение")
            except Exception as e:
                print(f"DEBUG: load_image: ошибка при загрузке - {e}")
                QMessageBox.critical(self, "Ошибка", f"Ошибка при загрузке изображения: {str(e)}")
                
    def schedule_processing(self):
        """Планирование обработки изображения с debounce"""
        if self.original_image is not None:
            # Останавливаем предыдущий таймер и запускаем новый
            self.debounce_timer.stop()
            self.debounce_timer.start(100)  # Уменьшаем до 100ms для более быстрого отклика
                
    def schedule_fast_processing(self):
        """Быстрая обработка только фильтра для интерактивного редактирования"""
        if self.original_image is None or self.processing_in_progress:
            print("DEBUG: schedule_fast_processing: пропускаем - нет изображения или обработка в процессе")
            return
            
        print("DEBUG: schedule_fast_processing: запускаем быструю обработку")
        print(f"DEBUG: schedule_fast_processing: D0={self.D0}, n={self.n}, filter_type={self.filter_type}")
            
        # Сохраняем состояние зума
        filtered_zoom = self.filtered_view.zoom_factor
        filtered_pan_x = self.filtered_view.pan_x
        filtered_pan_y = self.filtered_view.pan_y
            
        self.processing_in_progress = True
        self.status_label.setText("Быстрая обработка...")
        
        try:
            # Обновление параметров процессора
            self.image_processor.set_filter_parameters(self.filter_type, self.D0, self.n)
            
            # Используем зашумленное изображение если есть, иначе оригинал
            input_image = self.noisy_image if self.noisy_image is not None else self.original_image
            print(f"DEBUG: schedule_fast_processing: используем {'зашумленное' if self.noisy_image is not None else 'оригинальное'} изображение")
            
            # Создание и запуск потока быстрой обработки
            if self.processing_thread is not None:
                self.processing_thread.quit()
                self.processing_thread.wait()
                
            self.processing_thread = FastFilteringThread(self.image_processor, input_image)
            self.processing_thread.processing_finished.connect(lambda filtered: self.on_fast_processing_finished(filtered, filtered_zoom, filtered_pan_x, filtered_pan_y))
            self.processing_thread.processing_error.connect(self.on_processing_error)
            self.processing_thread.start()
            
        except Exception as e:
            print(f"DEBUG: schedule_fast_processing: ошибка - {e}")
            import traceback
            traceback.print_exc()
            self.processing_in_progress = False
            self.status_label.setText("Ошибка обработки")
    
    @pyqtSlot(int)
    def on_progress_updated(self, value):
        """Обновление прогресса"""
        self.progress_bar.setValue(value)
    
    @pyqtSlot(object, object, object)
    def on_processing_finished_with_zoom(self, noisy_image, filtered_image, spectra, original_zoom, original_pan_x, original_pan_y, noisy_zoom, noisy_pan_x, noisy_pan_y, filtered_zoom, filtered_pan_x, filtered_pan_y):
        """Обработка завершена с сохранением состояния зума"""
        print(f"DEBUG: on_processing_finished_with_zoom: получаем результаты")
        print(f"DEBUG: on_processing_finished_with_zoom: noisy_image диапазон=[{noisy_image.min():.3f}, {noisy_image.max():.3f}]")
        print(f"DEBUG: on_processing_finished_with_zoom: filtered_image диапазон=[{filtered_image.min():.3f}, {filtered_image.max():.3f}]")
        
        self.noisy_image = noisy_image
        self.filtered_image = filtered_image
        self.spectra = spectra
        self.processing_in_progress = False
        self.progress_bar.setVisible(False)
        self.status_label.setText("Обработка завершена")
        self.update_display()
        
        # Восстанавливаем состояние зума
        self.original_view.restore_zoom_state(original_zoom, original_pan_x, original_pan_y)
        self.noisy_view.restore_zoom_state(noisy_zoom, noisy_pan_x, noisy_pan_y)
        self.filtered_view.restore_zoom_state(filtered_zoom, filtered_pan_x, filtered_pan_y)
        
    @pyqtSlot(str)
    def on_processing_error(self, error_msg):
        """Ошибка обработки"""
        print(f"Ошибка обработки: {error_msg}")
        self.processing_in_progress = False
        self.progress_bar.setVisible(False)
        self.status_label.setText("Ошибка обработки")
            
    def update_display(self):
        """Обновление отображения"""
        if self.original_image is not None:
            self.original_view.set_image(self.original_image)
            
        if self.noisy_image is not None:
            self.noisy_view.set_image(self.noisy_image)
            
        if self.filtered_image is not None:
            self.filtered_view.set_image(self.filtered_image)
            
        if self.spectra is not None:
            original_spectrum, noisy_spectrum, filtered_spectrum = self.spectra
            self.original_spectrum_view.update_spectrum(self.original_image)
            
            # Обновляем спектр зашумленного/промежуточного изображения
            if self.enable_noise:
                self.noisy_spectrum_view.update_spectrum(self.noisy_image)
            else:
                # Если шум отключен, показываем спектр исходного изображения
                self.noisy_spectrum_view.update_spectrum(self.original_image)
                
            self.filtered_spectrum_view.update_spectrum(self.filtered_image)
            
        # Обновление визуализации фильтра
        self.filter_viz.update_filter(self.filter_type, self.D0, self.n)
        
    def reset_zooms(self):
        """Сброс зума всех изображений"""
        self.original_view.reset_zoom()
        self.noisy_view.reset_zoom()
        self.filtered_view.reset_zoom()
        
    def save_image(self):
        """Сохранение отфильтрованного изображения"""
        if self.filtered_image is None:
            QMessageBox.warning(self, "Предупреждение", "Нет изображения для сохранения")
            return
            
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Сохранить изображение", "filtered_image.png",
            "PNG files (*.png);;JPEG files (*.jpg);;All files (*)"
        )
        
        if file_path:
            try:
                # Конвертация изображения для сохранения
                if self.filtered_image.dtype == np.float32:
                    if self.filtered_image.max() <= 1.0:
                        save_image = (self.filtered_image * 255).astype(np.uint8)
                    else:
                        save_image = self.filtered_image.astype(np.uint8)
                else:
                    save_image = self.filtered_image.astype(np.uint8)
                
                # Конвертация в BGR для OpenCV
                if len(save_image.shape) == 3 and save_image.shape[2] == 3:
                    save_image = cv2.cvtColor(save_image, cv2.COLOR_RGB2BGR)
                
                cv2.imwrite(file_path, save_image)
                QMessageBox.information(self, "Успех", "Изображение сохранено")
            except Exception as e:
                QMessageBox.critical(self, "Ошибка", f"Ошибка при сохранении: {str(e)}")
            
    # Обработчики событий
    def on_filter_changed(self, text):
        """Обработчик изменения типа фильтра"""
        self.filter_type = text
        # Обновляем визуализацию фильтра
        self.filter_viz.update_filter(filter_type=text, D0=self.D0, n=self.n)
        
        # Показываем/скрываем ползунок n только для Butterworth
        show_n_controls = (text == 'Butterworth')
        self.n_label_title.setVisible(show_n_controls)
        self.n_slider.setVisible(show_n_controls)
        self.n_label.setVisible(show_n_controls)
        
        self.schedule_processing()
        
    def on_D0_changed(self, value):
        self.D0 = value / 100.0
        self.D0_label.setText(f"{self.D0:.2f}")
        # Обновляем интерактивный график
        self.filter_viz.update_filter(D0=self.D0)
        # Используем выбранный режим обработки
        if self.fast_processing_checkbox.isChecked():
            self.schedule_fast_processing()
        else:
            self.schedule_processing()
        
    def on_pad_changed(self, value):
        """Обработчик изменения дополнения границ"""
        self.pad_ratio = value / 100.0
        self.pad_label.setText(f"{self.pad_ratio:.2f}")
        self.schedule_processing()
        
    def on_n_changed(self, value):
        """Обработчик изменения порядка фильтра"""
        self.n = value
        self.n_label.setText(f"{value}")
        # Обновляем визуализацию фильтра
        self.filter_viz.update_filter(n=value)
        self.schedule_processing()
        
    def on_noise_freq_changed(self, value):
        self.noise_freq = value / 100.0
        self.noise_freq_label.setText(f"{self.noise_freq:.2f}")
        self.schedule_processing()
        
    def on_noise_amp_changed(self, value):
        self.noise_amp = value / 100.0
        self.noise_amp_label.setText(f"{self.noise_amp:.2f}")
        self.schedule_processing()
        
    def on_noise_angle_changed(self, value):
        self.noise_angle = value
        self.noise_angle_label.setText(f"{self.noise_angle:.0f}°")
        self.schedule_processing()
        
    def on_sigma_s_changed(self, value):
        self.sigma_s = value / 10.0
        self.sigma_s_label.setText(f"{self.sigma_s:.1f}")
        # Обновляем параметры процессора
        self.image_processor.set_enhancement_parameters(self.sigma_s, self.sigma_r, self.contrast_strength)
        self.schedule_processing()
        
    def on_sigma_r_changed(self, value):
        self.sigma_r = value / 100.0
        self.sigma_r_label.setText(f"{self.sigma_r:.2f}")
        # Обновляем параметры процессора
        self.image_processor.set_enhancement_parameters(self.sigma_s, self.sigma_r, self.contrast_strength)
        self.schedule_processing()
        
    def on_contrast_changed(self, value):
        self.contrast_strength = value / 100.0
        self.contrast_label.setText(f"{self.contrast_strength:.2f}")
        # Обновляем параметры процессора
        self.image_processor.set_enhancement_parameters(self.sigma_s, self.sigma_r, self.contrast_strength)
        self.schedule_processing()
        
    def on_noise_enabled_changed(self, state):
        self.enable_noise = state == Qt.Checked
        self.update_noise_visibility()
        self.schedule_processing()
        
    def on_enhancement_enabled_changed(self, state):
        self.enable_enhancement = state == Qt.Checked
        self.update_enhancement_visibility()
        self.schedule_processing()
        
    def on_gpu_mode_changed(self, state):
        """Обработчик изменения режима GPU/CPU"""
        use_gpu = state == Qt.Checked
        self.image_processor.set_gpu_mode(use_gpu)
        print(f"DEBUG: on_gpu_mode_changed: переключен на {'GPU' if use_gpu else 'CPU'}")
        # Перезапускаем обработку с новыми настройками
        self.schedule_processing()
        
    def on_sync_zoom_changed(self, state):
        """Обработчик изменения состояния синхронизации зума"""
        self.sync_zoom = state == Qt.Checked
        
    def on_fast_processing_changed(self, state):
        """Обработчик изменения режима быстрой обработки"""
        use_fast_processing = state == Qt.Checked
        if use_fast_processing:
            self.status_label.setText("Включен режим быстрой обработки")
        else:
            self.status_label.setText("Включен режим полной обработки")
        
    def sync_zoom_to_others(self, source_view, zoom_value):
        """Синхронизация зума с другими изображениями"""
        if not self.sync_zoom:
            return
            
        zoom_factor = zoom_value / 100.0
        
        # Синхронизируем с другими изображениями
        if source_view != self.original_view:
            self.original_view.sync_zoom(zoom_factor)
        if source_view != self.noisy_view:
            self.noisy_view.sync_zoom(zoom_factor)
        if source_view != self.filtered_view:
            self.filtered_view.sync_zoom(zoom_factor)
            
    def sync_pan_to_others(self, source_view, pan_x, pan_y):
        """Синхронизация перемещения с другими изображениями"""
        if not self.sync_zoom:
            return
            
        # Синхронизируем с другими изображениями
        if source_view != self.original_view:
            self.original_view.sync_pan(pan_x, pan_y)
        if source_view != self.noisy_view:
            self.noisy_view.sync_pan(pan_x, pan_y)
        if source_view != self.filtered_view:
            self.filtered_view.sync_pan(pan_x, pan_y)
        
    def update_noise_visibility(self):
        """Обновление видимости элементов в зависимости от состояния шума"""
        # Находим QGroupBox для зашумленного изображения
        for widget in self.findChildren(QGroupBox):
            if widget.title() == "Зашумленное" or widget.title() == "Промежуточное":
                widget.setVisible(self.enable_noise)
                if self.enable_noise:
                    widget.setTitle("Зашумленное")
                else:
                    widget.setTitle("Промежуточное")
                break
                
        # Находим QGroupBox для спектра зашумленного изображения
        for widget in self.findChildren(QGroupBox):
            if widget.title() == "Спектр зашумленного" or widget.title() == "Спектр промежуточного":
                widget.setVisible(self.enable_noise)
                if self.enable_noise:
                    widget.setTitle("Спектр зашумленного")
                else:
                    widget.setTitle("Спектр промежуточного")
                break
                
        # Скрываем/показываем параметры шума
        self.noise_params_widget.setVisible(self.enable_noise)
        
    def update_enhancement_visibility(self):
        """Обновление видимости элементов улучшения"""
        # Скрываем/показываем параметры улучшения
        self.enhancement_params_widget.setVisible(self.enable_enhancement)
        
    def auto_optimize_parameters(self):
        """Автоматическая оптимизация параметров фильтра"""
        if self.original_image is None:
            QMessageBox.warning(self, "Предупреждение", "Сначала загрузите изображение")
            return
            
        try:
            self.status_label.setText("Выполняется автоматическая оптимизация...")
            
            # Выполняем автоматическую оптимизацию
            success, optimal_D0, optimal_n = self.image_processor.auto_optimize_parameters(self.original_image)
            
            if success:
                # Обновляем параметры в интерфейсе
                self.D0 = optimal_D0
                self.D0_slider.setValue(int(optimal_D0 * 100))
                self.D0_label.setText(f"{optimal_D0:.2f}")
                
                self.n = optimal_n
                
                # Обновляем визуализацию фильтра
                self.filter_viz.update_filter(self.filter_type, optimal_D0, optimal_n)
                
                self.status_label.setText(f"Оптимизация завершена: D0={optimal_D0:.2f}, n={optimal_n}")
                
                # Запускаем обработку с новыми параметрами
                self.schedule_processing()
                
            else:
                self.status_label.setText("Не удалось выполнить оптимизацию")
                QMessageBox.warning(self, "Предупреждение", "Не удалось автоматически оптимизировать параметры")
                
        except Exception as e:
            self.status_label.setText("Ошибка оптимизации")
            QMessageBox.critical(self, "Ошибка", f"Ошибка при автоматической оптимизации: {str(e)}")

    def on_blend_enabled_changed(self, state):
        enable_blend = state == Qt.Checked
        alpha = self.blend_slider.value() / 100.0
        self.image_processor.set_blend_parameters(alpha, enable_blend)
        
        # Показываем/скрываем параметры смешивания
        self.blend_params_widget.setVisible(enable_blend)
        
        self.schedule_processing()

    def on_blend_alpha_changed(self, value):
        alpha = value / 100.0
        self.blend_label.setText(f"{alpha:.2f}")
        enable_blend = self.blend_checkbox.isChecked()
        self.image_processor.set_blend_parameters(alpha, enable_blend)
        if enable_blend:
            self.schedule_processing()

    def on_D0_changed_interactive(self, value):
        """Обработчик интерактивного изменения D0 от графика фильтра"""
        print(f"DEBUG: on_D0_changed_interactive: получаем значение D0 = {value}")
        self.D0 = value
        # Обновляем ползунок и метку
        self.D0_slider.setValue(int(value * 100))
        self.D0_label.setText(f"{value:.2f}")
        # Используем быструю обработку для интерактивного редактирования
        print("DEBUG: on_D0_changed_interactive: запускаем быструю обработку")
        self.schedule_fast_processing()

    def on_n_changed_interactive(self, value):
        """Обработчик интерактивного изменения n от графика фильтра"""
        print(f"DEBUG: on_n_changed_interactive: получаем значение n = {value}")
        self.n = value
        # Обновляем ползунок и метку
        if hasattr(self, 'n_slider'):
            self.n_slider.setValue(value)
            self.n_label.setText(f"{value}")
        # Используем быструю обработку для интерактивного редактирования
        print("DEBUG: on_n_changed_interactive: запускаем быструю обработку")
        self.schedule_fast_processing()

    def process_image(self):
        """Полная обработка изображения с шумом и улучшениями"""
        if self.original_image is None or self.processing_in_progress:
            return
            
        # Сохраняем состояние зума
        original_zoom = self.original_view.zoom_factor
        original_pan_x = self.original_view.pan_x
        original_pan_y = self.original_view.pan_y
        
        noisy_zoom = self.noisy_view.zoom_factor
        noisy_pan_x = self.noisy_view.pan_x
        noisy_pan_y = self.noisy_view.pan_y
        
        filtered_zoom = self.filtered_view.zoom_factor
        filtered_pan_x = self.filtered_view.pan_x
        filtered_pan_y = self.filtered_view.pan_y
            
        self.processing_in_progress = True
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.status_label.setText("Обработка изображения...")
        
        try:
            # Обновление параметров процессора
            self.image_processor.set_filter_parameters(self.filter_type, self.D0, self.n)
            self.image_processor.set_noise_parameters(self.noise_freq, self.noise_amp, self.noise_angle)
            self.image_processor.set_enhancement_parameters(self.sigma_s, self.sigma_r, self.contrast_strength)
            self.image_processor.set_padding_ratio(self.pad_ratio)
            
            # Создание и запуск потока обработки
            if self.processing_thread is not None:
                self.processing_thread.quit()
                self.processing_thread.wait()
                
            self.processing_thread = ImageProcessingThread(self.image_processor, self.original_image, self.enable_noise, self.enable_enhancement)
            self.processing_thread.processing_finished.connect(lambda noisy, filtered, spectra: self.on_processing_finished_with_zoom(noisy, filtered, spectra, original_zoom, original_pan_x, original_pan_y, noisy_zoom, noisy_pan_x, noisy_pan_y, filtered_zoom, filtered_pan_x, filtered_pan_y))
            self.processing_thread.processing_error.connect(self.on_processing_error)
            self.processing_thread.progress_updated.connect(self.on_progress_updated)
            self.processing_thread.start()
            
        except Exception as e:
            print(f"Ошибка обработки: {e}")
            self.processing_in_progress = False
            self.progress_bar.setVisible(False)
            self.status_label.setText("Ошибка обработки")

    @pyqtSlot(object, float, float, float)
    def on_fast_processing_finished(self, filtered_image, filtered_zoom, filtered_pan_x, filtered_pan_y):
        """Быстрая обработка завершена"""
        print("DEBUG: on_fast_processing_finished: получаем результат быстрой обработки")
        print(f"DEBUG: on_fast_processing_finished: размер изображения = {filtered_image.shape}")
        print(f"DEBUG: on_fast_processing_finished: диапазон = [{filtered_image.min():.3f}, {filtered_image.max():.3f}]")
        print(f"DEBUG: on_fast_processing_finished: тип данных = {filtered_image.dtype}")
        
        self.filtered_image = filtered_image
        self.processing_in_progress = False
        self.status_label.setText("Быстрая обработка завершена")
        
        # Обновляем только отфильтрованное изображение
        if self.filtered_image is not None:
            print("DEBUG: on_fast_processing_finished: обновляем отображение")
            print(f"DEBUG: on_fast_processing_finished: filtered_view существует = {self.filtered_view is not None}")
            self.filtered_view.set_image(self.filtered_image)
            print("DEBUG: on_fast_processing_finished: set_image вызван")
            
        # Восстанавливаем состояние зума
        self.filtered_view.restore_zoom_state(filtered_zoom, filtered_pan_x, filtered_pan_y)
        print("DEBUG: on_fast_processing_finished: быстрая обработка завершена успешно")

    def force_full_processing(self):
        """Полная обработка изображения"""
        if self.original_image is None or self.processing_in_progress:
            return
            
        self.processing_in_progress = True
        self.status_label.setText("Выполняется полная обработка...")
        
        try:
            # Обновление параметров процессора
            self.image_processor.set_filter_parameters(self.filter_type, self.D0, self.n)
            self.image_processor.set_noise_parameters(self.noise_freq, self.noise_amp, self.noise_angle)
            self.image_processor.set_enhancement_parameters(self.sigma_s, self.sigma_r, self.contrast_strength)
            self.image_processor.set_padding_ratio(self.pad_ratio)
            
            # Создание и запуск потока обработки
            if self.processing_thread is not None:
                self.processing_thread.quit()
                self.processing_thread.wait()
                
            self.processing_thread = ImageProcessingThread(self.image_processor, self.original_image, self.enable_noise, self.enable_enhancement)
            self.processing_thread.processing_finished.connect(lambda noisy, filtered, spectra: self.on_processing_finished_with_zoom(noisy, filtered, spectra, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0))
            self.processing_thread.processing_error.connect(self.on_processing_error)
            self.processing_thread.progress_updated.connect(self.on_progress_updated)
            self.processing_thread.start()
            
        except Exception as e:
            print(f"Ошибка обработки: {e}")
            self.processing_in_progress = False
            self.progress_bar.setVisible(False)
            self.status_label.setText("Ошибка обработки") 