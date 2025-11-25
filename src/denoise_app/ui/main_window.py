import cv2
from PyQt5.QtWidgets import (
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QSlider,
    QComboBox,
    QPushButton,
    QFileDialog,
    QGroupBox,
    QGridLayout,
    QTabWidget,
    QFrame,
    QProgressBar,
    QCheckBox,
    QRadioButton,
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QThread, pyqtSlot
from PyQt5.QtGui import QFont

from ..core.image_processor import ImageProcessor
from .widgets import ZoomableImageView, SpectrumViewer


class FastFilteringThread(QThread):
    processing_finished = pyqtSignal(object)

    def __init__(self, processor, image, mode="FFT"):
        super().__init__()
        self.processor = processor
        self.image = image
        self.mode = mode

    def run(self):
        try:
            if self.mode == "FFT":
                res = self.processor.apply_frequency_filter(self.image)
            else:
                res = self.processor.wavelet.process_image(self.image)
            self.processing_finished.emit(res)
        except Exception as e:
            print(f"Fast thread error: {e}")


class ImageProcessingThread(QThread):
    processing_finished = pyqtSignal(object, object, object)
    progress_updated = pyqtSignal(int)

    def __init__(self, processor, original_image, mode="FFT"):
        super().__init__()
        self.processor = processor
        self.original_image = original_image
        self.mode = mode

    def run(self):
        try:
            self.progress_updated.emit(10)

            if self.mode == "FFT":
                noisy, final, spectra = self.processor.process_image(
                    self.original_image
                )
            else:
                noisy, final, spectra = self.processor.process_image_wavelet(
                    self.original_image
                )

            self.progress_updated.emit(100)
            self.processing_finished.emit(noisy, final, spectra)

        except Exception as e:
            print(f"Processing thread error: {e}")
            import traceback

            traceback.print_exc()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Advanced Image Denoising: FFT & Wavelets")
        self.setGeometry(100, 100, 1400, 950)

        self.image_processor = ImageProcessor()

        # Состояние
        self.original_image = None
        self.noisy_image = None
        self.filtered_image = None
        self.spectra = None
        self.processing_in_progress = False
        self.processing_thread = None
        self.mode = "Wavelet"

        self.filter_type = "Butterworth"
        self.D0 = 0.2
        self.n = 2

        self.wavelet_name = "db4"
        self.wavelet_level = 2
        self.wavelet_threshold = 0.1
        self.wavelet_mode = "soft"

        self.enable_noise = True
        self.sync_zoom = True

        self.setup_ui()

        self.debounce_timer = QTimer()
        self.debounce_timer.setSingleShot(True)
        self.debounce_timer.timeout.connect(self.process_image_full)

    def setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        control_panel = self.create_control_panel()
        main_layout.addWidget(control_panel, 1)

        image_panel = self.create_image_panel()
        main_layout.addWidget(image_panel, 3)

    def create_control_panel(self):
        panel = QFrame()
        panel.setFrameStyle(QFrame.StyledPanel)
        panel.setMaximumWidth(350)

        layout = QVBoxLayout(panel)

        title = QLabel("Панель управления")
        title.setFont(QFont("Arial", 12, QFont.Bold))
        layout.addWidget(title)

        mode_group = QGroupBox("Режим обработки")
        mode_layout = QVBoxLayout(mode_group)
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["Вейвлеты (Wavelet DWT)", "Частотный (FFT)"])
        self.mode_combo.currentTextChanged.connect(self.on_mode_changed)
        mode_layout.addWidget(self.mode_combo)
        layout.addWidget(mode_group)

        load_btn = QPushButton("Загрузить изображение")
        load_btn.setStyleSheet("background-color: #4CAF50; color: white; padding: 8px;")
        load_btn.clicked.connect(self.load_image)
        layout.addWidget(load_btn)

        self.wavelet_group = QGroupBox("Параметры Вейвлетов")
        w_layout = QGridLayout(self.wavelet_group)

        w_layout.addWidget(QLabel("Семейство:"), 0, 0)
        self.w_family_combo = QComboBox()
        self.w_family_combo.addItems(["db1", "db2", "db4", "sym4", "coif1", "bior1.3"])
        self.w_family_combo.setCurrentText(self.wavelet_name)
        self.w_family_combo.currentTextChanged.connect(self.on_wavelet_params_changed)
        w_layout.addWidget(self.w_family_combo, 0, 1)

        w_layout.addWidget(QLabel("Уровень:"), 1, 0)
        self.w_level_combo = QComboBox()
        self.w_level_combo.addItems(["1", "2", "3", "4"])
        self.w_level_combo.setCurrentText(str(self.wavelet_level))
        self.w_level_combo.currentTextChanged.connect(self.on_wavelet_params_changed)
        w_layout.addWidget(self.w_level_combo, 1, 1)

        w_layout.addWidget(QLabel("Порог шума:"), 2, 0)
        self.w_thresh_slider = QSlider(Qt.Horizontal)
        self.w_thresh_slider.setRange(0, 100)  # 0.0 - 1.0
        self.w_thresh_slider.setValue(int(self.wavelet_threshold * 100))
        self.w_thresh_slider.valueChanged.connect(self.on_wavelet_thresh_changed)
        w_layout.addWidget(self.w_thresh_slider, 2, 1)
        self.w_thresh_label = QLabel(f"{self.wavelet_threshold:.2f}")
        w_layout.addWidget(self.w_thresh_label, 2, 2)

        w_layout.addWidget(QLabel("Убр. цвет. шум:"), 3, 0)
        self.w_chroma_slider = QSlider(Qt.Horizontal)
        self.w_chroma_slider.setRange(0, 100)
        self.w_chroma_slider.setValue(50)
        self.w_chroma_slider.setToolTip("Обесцвечивает шум (делает его серым)")
        self.w_chroma_slider.valueChanged.connect(self.on_wavelet_params_changed)
        w_layout.addWidget(self.w_chroma_slider, 3, 1)

        mode_label = QLabel("Тип порога:")
        w_layout.addWidget(mode_label, 4, 0)
        self.w_soft_btn = QRadioButton("Soft (Мягкий)")
        self.w_hard_btn = QRadioButton("Hard (Жесткий)")
        self.w_soft_btn.setChecked(True)
        self.w_soft_btn.toggled.connect(self.on_wavelet_params_changed)
        w_layout.addWidget(self.w_soft_btn, 4, 1)
        w_layout.addWidget(self.w_hard_btn, 5, 1)

        self.w_swt_check = QCheckBox("Использовать SWT вместо DWT")
        self.w_swt_check.stateChanged.connect(self.on_wavelet_params_changed)
        w_layout.addWidget(self.w_swt_check, 6, 0, 1, 2)

        self.w_smooth_check = QCheckBox("Финишное сглаживание")
        self.w_smooth_check.setToolTip("Убирает блочность (Bilateral Filter)")
        self.w_smooth_check.setChecked(False)
        self.w_smooth_check.stateChanged.connect(self.on_wavelet_params_changed)
        w_layout.addWidget(self.w_smooth_check, 7, 0, 1, 2)

        layout.addWidget(self.wavelet_group)

        self.fft_group = QGroupBox("Параметры FFT")
        f_layout = QGridLayout(self.fft_group)
        self.fft_filter_combo = QComboBox()
        self.fft_filter_combo.addItems(["Butterworth", "Gaussian"])
        self.fft_filter_combo.currentTextChanged.connect(self.on_fft_params_changed)
        f_layout.addWidget(QLabel("Фильтр:"), 0, 0)
        f_layout.addWidget(self.fft_filter_combo, 0, 1)

        f_layout.addWidget(QLabel("Радиус (D0):"), 1, 0)
        self.fft_d0_slider = QSlider(Qt.Horizontal)
        self.fft_d0_slider.setRange(1, 100)
        self.fft_d0_slider.setValue(20)
        self.fft_d0_slider.valueChanged.connect(self.on_fft_d0_changed)
        f_layout.addWidget(self.fft_d0_slider, 1, 1)
        self.fft_d0_label = QLabel("0.2")
        f_layout.addWidget(self.fft_d0_label, 1, 2)

        layout.addWidget(self.fft_group)
        self.fft_group.setVisible(False)

        # 4. Шум (Общее)
        noise_group = QGroupBox("Генерация шума")
        n_layout = QGridLayout(noise_group)

        self.noise_check = QCheckBox("Добавить шум")
        self.noise_check.setChecked(True)
        self.noise_check.stateChanged.connect(self.on_noise_changed)
        n_layout.addWidget(self.noise_check, 0, 0, 1, 2)

        self.noise_type_combo = QComboBox()
        self.noise_type_combo.addItems(["Гауссов", "Периодический"])
        self.noise_type_combo.currentTextChanged.connect(self.on_noise_type_changed)
        n_layout.addWidget(self.noise_type_combo, 1, 0, 1, 2)

        n_layout.addWidget(QLabel("Сила шума:"), 2, 0)
        self.noise_level_slider = QSlider(Qt.Horizontal)
        self.noise_level_slider.setRange(0, 100)
        self.noise_level_slider.setValue(10)  # 0.05
        self.noise_level_slider.valueChanged.connect(self.on_noise_level_changed)
        n_layout.addWidget(self.noise_level_slider, 2, 1)

        layout.addWidget(noise_group)

        # Прогресс
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        self.status_label = QLabel("Готов")
        layout.addWidget(self.status_label)

        layout.addStretch()
        return panel

    def create_image_panel(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)

        self.tabs = QTabWidget()

        img_tab = QWidget()
        img_layout = QHBoxLayout(img_tab)

        self.view_orig = ZoomableImageView()
        self.view_noisy = ZoomableImageView()
        self.view_result = ZoomableImageView()

        self.view_orig.view_changed.connect(lambda: self.sync_all(self.view_orig))
        self.view_noisy.view_changed.connect(lambda: self.sync_all(self.view_noisy))
        self.view_result.view_changed.connect(lambda: self.sync_all(self.view_result))

        img_layout.addWidget(self.wrap_in_group(self.view_orig, "Оригинал"))

        self.noisy_group_box = self.wrap_in_group(self.view_noisy, "Зашумленное")
        img_layout.addWidget(self.noisy_group_box)

        img_layout.addWidget(self.wrap_in_group(self.view_result, "Результат"))
        self.tabs.addTab(img_tab, "Результат обработки")

        viz_tab = QWidget()
        viz_layout = QHBoxLayout(viz_tab)

        self.spec_orig = SpectrumViewer()
        self.spec_noisy = SpectrumViewer()
        self.spec_result = SpectrumViewer()

        viz_layout.addWidget(self.wrap_in_group(self.spec_orig, "Входной сигнал"))

        self.spec_noisy_group = self.wrap_in_group(
            self.spec_noisy, "Зашумленный сигнал"
        )
        viz_layout.addWidget(self.spec_noisy_group)

        self.viz_result_group = QGroupBox("Результат (Спектр / Мозаика)")
        v_layout = QVBoxLayout(self.viz_result_group)
        v_layout.addWidget(self.spec_result)
        viz_layout.addWidget(self.viz_result_group)

        self.tabs.addTab(viz_tab, "Визуализация")

        layout.addWidget(self.tabs)
        return panel

    def wrap_in_group(self, widget, title):
        gb = QGroupBox(title)
        l = QVBoxLayout(gb)
        l.setContentsMargins(0, 0, 0, 0)
        l.addWidget(widget)
        return gb

    def on_mode_changed(self, text):
        is_wavelet = "Вейвлеты" in text
        self.mode = "Wavelet" if is_wavelet else "FFT"

        self.wavelet_group.setVisible(is_wavelet)
        self.fft_group.setVisible(not is_wavelet)

        if is_wavelet:
            self.viz_result_group.setTitle("Мозаика коэффициентов (Вейвлет)")
        else:
            self.viz_result_group.setTitle("Спектр результата (FFT)")

        self.schedule_processing()

    def load_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Открыть", "", "Images (*.png *.jpg *.jpeg *.bmp)"
        )
        if path:
            img = cv2.imread(path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            self.original_image = img
            self.view_orig.set_image(img)
            self.spec_orig.update_spectrum(img)
            self.process_image_full()

    def schedule_processing(self):
        self.debounce_timer.start(150)

    def process_image_full(self):
        if self.original_image is None or self.processing_in_progress:
            return

        if self.noise_check.isChecked():
            level = self.noise_level_slider.value() / 200.0
        else:
            level = 0.0

        self.image_processor.set_noise_type(
            "gaussian"
            if "Гауссов" in self.noise_type_combo.currentText()
            else "periodic"
        )

        if self.image_processor.noise_generator.noise_type == "gaussian":
            self.image_processor.set_gaussian_noise_parameters(0, level)
        else:
            self.image_processor.set_noise_parameters(0.1, level, 45)

        if self.mode == "Wavelet":
            w_name = self.w_family_combo.currentText()
            w_lvl = int(self.w_level_combo.currentText())
            w_thr = self.w_thresh_slider.value() / 100.0
            w_mode = "soft" if self.w_soft_btn.isChecked() else "hard"

            use_swt = self.w_swt_check.isChecked()
            w_smooth = self.w_smooth_check.isChecked()
            chroma_str = self.w_chroma_slider.value() / 100.0

            self.image_processor.set_wavelet_parameters(
                w_name,
                w_lvl,
                w_thr,
                w_mode,
                post_smooth=w_smooth,
                use_swt=use_swt,
                chroma_str=chroma_str,
            )
        else:
            d0 = self.fft_d0_slider.value() / 100.0
            self.image_processor.set_filter_parameters(
                self.fft_filter_combo.currentText(), d0, 2
            )

        self.processing_in_progress = True
        self.progress_bar.setVisible(True)
        self.status_label.setText("Обработка...")

        self.processing_thread = ImageProcessingThread(
            self.image_processor, self.original_image, self.mode
        )
        self.processing_thread.processing_finished.connect(self.on_finished)
        self.processing_thread.start()

    @pyqtSlot(object, object, object)
    def on_finished(self, noisy, result, viz_data):
        self.processing_in_progress = False
        self.progress_bar.setVisible(False)
        self.status_label.setText("Готово")

        self.noisy_image = noisy

        self.view_noisy.set_image(noisy, maintain_zoom=True)
        self.view_result.set_image(result, maintain_zoom=True)

        s_orig, s_noisy, s_result = viz_data

        if self.mode == "FFT":
            self.spec_noisy.update_spectrum(noisy)
            self.spec_result.update_spectrum(result)
        else:
            self.spec_noisy.update_spectrum(noisy)
            self.spec_result.show_image_directly(s_result, "Вейвлет-разложение")

    def on_wavelet_params_changed(self):
        self.schedule_processing()

    def on_fft_params_changed(self):
        self.schedule_processing()

    def on_noise_changed(self):
        is_checked = self.noise_check.isChecked()

        self.noisy_group_box.setVisible(is_checked)
        self.spec_noisy_group.setVisible(is_checked)

        # Отключаем слайдеры шума, если шум выключен
        self.noise_level_slider.setEnabled(is_checked)
        self.noise_type_combo.setEnabled(is_checked)

        self.schedule_processing()

    def on_noise_type_changed(self):
        self.schedule_processing()

    def on_wavelet_thresh_changed(self, val):
        self.wavelet_threshold = val / 100.0
        self.w_thresh_label.setText(f"{self.wavelet_threshold:.2f}")
        self.schedule_processing()

    def on_fft_d0_changed(self, val):
        self.fft_d0_label.setText(f"{val / 100.0:.2f}")
        self.schedule_processing()

    def on_noise_level_changed(self, val):
        self.schedule_processing()

    def sync_all(self, source_view):
        if not self.sync_zoom:
            return

        targets = [self.view_orig, self.view_noisy, self.view_result]

        for target in targets:
            if target != source_view:
                target.sync_state_from(source_view)
