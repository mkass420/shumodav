"""
Модуль для обработки изображений в частотной области
"""

import cv2
import numpy as np
from scipy.fft import fft2, ifft2, fftshift, ifftshift
from scipy import ndimage
import threading
from typing import Optional, Tuple
from multiprocessing import Pool, cpu_count
from functools import partial
import subprocess
import time
import gc

# Импортируем GPU FFT процессор
try:
    from .gpu_fft import GPUFFTProcessor
except ImportError:
    # Если модуль не найден, создаем заглушку
    class GPUFFTProcessor:
        def __init__(self, acceleration_type="CPU"):
            self.acceleration_type = acceleration_type
            self.use_gpu = False
        
        def fft2_gpu(self, image):
            return fft2(image)
        
        def ifft2_gpu(self, spectrum):
            return ifft2(spectrum)

# Проверяем доступность CUDA и OpenCL
try:
    # Пытаемся импортировать CUDA модули OpenCV
    device_count = cv2.cuda.getCudaEnabledDeviceCount()
    if device_count == 0:
        raise Exception("Нет доступных CUDA устройств")
    
    # Дополнительная проверка - пытаемся создать GpuMat и выполнить операцию
    test_gpu_mat = cv2.cuda_GpuMat()
    test_array = np.array([[1, 2], [3, 4]], dtype=np.float32)
    test_gpu_mat.upload(test_array)
    test_result = test_gpu_mat.download()
    
    CUDA_AVAILABLE = True
    print(f"DEBUG: CUDA доступна для ускорения (устройств: {device_count})")
except Exception as e:
    CUDA_AVAILABLE = False
    print(f"DEBUG: CUDA недоступна: {e}")

# Проверяем OpenCL через системные утилиты
try:
    result = subprocess.run(['clinfo'], capture_output=True, text=True, timeout=5)
    if result.returncode == 0 and "Number of platforms" in result.stdout:
        if "Number of platforms                               0" in result.stdout:
            print("DEBUG: OpenCL недоступен (нет платформ)")
            OPENCL_AVAILABLE = False
        else:
            print("DEBUG: OpenCL платформы найдены в системе")
            OPENCL_AVAILABLE = True
    else:
        print("DEBUG: clinfo не найден или не работает")
        OPENCL_AVAILABLE = False
except:
    print("DEBUG: Не удалось проверить OpenCL через clinfo")
    OPENCL_AVAILABLE = False

# Если системная проверка прошла, проверяем OpenCV OpenCL
if OPENCL_AVAILABLE:
    try:
        # Пытаемся импортировать OpenCL модули OpenCV
        cv2.ocl.useOpenCL()
        # Дополнительная проверка - пытаемся выполнить простую операцию
        test_array = np.array([[1, 2], [3, 4]], dtype=np.float32)
        test_blur = cv2.GaussianBlur(test_array, (3, 3), 0)
        
        # Проверяем, что OpenCL действительно используется
        if cv2.ocl.useOpenCL():
            OPENCL_AVAILABLE = True
            print("DEBUG: OpenCL доступен для ускорения")
        else:
            OPENCL_AVAILABLE = False
            print("DEBUG: OpenCL недоступен (useOpenCL() вернул False)")
    except Exception as e:
        OPENCL_AVAILABLE = False
        print(f"DEBUG: OpenCL недоступен: {e}")

# Определяем доступное ускорение
if CUDA_AVAILABLE:
    ACCELERATION_TYPE = "CUDA"
elif OPENCL_AVAILABLE:
    ACCELERATION_TYPE = "OpenCL"
else:
    ACCELERATION_TYPE = "CPU"
    print("DEBUG: Используем CPU (GPU недоступен)")


class AutoOptimizer:
    """Автоматическая оптимизация параметров фильтра"""
    
    def __init__(self):
        self.optimization_method = 'spectrum_analysis'
        
    def analyze_spectrum(self, image):
        """Анализ спектра изображения для определения шума"""
        if image is None:
            return None, None
            
        # Конвертация в оттенки серого
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
            
        # Нормализация
        if gray.dtype != np.float32:
            gray = gray.astype(np.float32) / 255.0
            
        # FFT
        f_transform = fft2(gray)
        f_shift = fftshift(f_transform)
        magnitude_spectrum = np.abs(f_shift)
        
        # Анализ спектра
        height, width = magnitude_spectrum.shape
        center_y, center_x = height // 2, width // 2
        
        # Создание сетки расстояний
        y, x = np.ogrid[:height, :width]
        distances = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        
        # Нормализация расстояний
        max_distance = np.sqrt(center_x**2 + center_y**2)
        distances = distances / max_distance
        
        # Анализ распределения энергии
        total_energy = np.sum(magnitude_spectrum)
        
        # Поиск пиков в спектре (возможные шумы)
        peaks = self._find_spectral_peaks(magnitude_spectrum, distances)
        
        return peaks, total_energy
        
    def _find_spectral_peaks(self, magnitude_spectrum, distances):
        """Поиск пиков в спектре"""
        # Применяем фильтр для сглаживания
        smoothed = ndimage.gaussian_filter(magnitude_spectrum, sigma=2)
        
        # Находим локальные максимумы
        from scipy.ndimage import maximum_filter
        local_max = maximum_filter(smoothed, size=20)
        peaks = (smoothed == local_max) & (smoothed > np.mean(smoothed) * 1.5)
        
        # Получаем частоты пиков
        peak_frequencies = distances[peaks]
        peak_magnitudes = magnitude_spectrum[peaks]
        
        return list(zip(peak_frequencies, peak_magnitudes))
        
    def optimize_parameters(self, image, filter_type='Butterworth'):
        """Автоматическая оптимизация параметров фильтра с учетом типа фильтра"""
        if image is None:
            return 0.2, 2
            
        peaks, total_energy = self.analyze_spectrum(image)
        
        if not peaks:
            return 0.2, 2
            
        # Анализируем пики для определения оптимальных параметров
        frequencies = [freq for freq, mag in peaks]
        magnitudes = [mag for freq, mag in peaks]
        
        if not frequencies:
            return 0.2, 2
            
        # Определяем оптимальную частоту среза в зависимости от типа фильтра
        median_freq = np.median(frequencies)
        
        if filter_type == 'Ideal':
            # Для идеального фильтра используем более консервативную частоту
            optimal_D0 = max(0.1, min(0.6, median_freq * 0.6))
            optimal_n = 1  # Идеальный фильтр не использует параметр n
            
        elif filter_type == 'Gaussian':
            # Для гауссова фильтра используем среднюю частоту
            optimal_D0 = max(0.1, min(0.8, median_freq * 0.7))
            optimal_n = 1  # Гауссов фильтр не использует параметр n
            
        else:  # Butterworth
            # Для фильтра Баттерворта используем более широкую частоту
            optimal_D0 = max(0.1, min(0.8, median_freq * 0.8))
            
            # Определяем оптимальный порядок фильтра
            # Если много пиков с высокой энергией, используем более высокий порядок
            energy_ratio = sum(magnitudes) / total_energy if total_energy > 0 else 0
            
            if energy_ratio > 0.3:
                optimal_n = 4
            elif energy_ratio > 0.1:
                optimal_n = 3
            else:
                optimal_n = 2
            
        return optimal_D0, optimal_n


class NoiseGenerator:
    """Генератор периодического шума"""
    
    def __init__(self):
        self.freq = 0.1
        self.amplitude = 0.3
        self.angle = 45
        
    def add_periodic_noise(self, image):
        """Добавление периодического шума к изображению"""
        if image is None:
            return None
            
        # Нормализация изображения в диапазон [0, 1]
        if image.dtype != np.float32:
            image = image.astype(np.float32) / 255.0
            
        height, width = image.shape[:2]
        
        # Создание сетки координат
        x, y = np.meshgrid(np.arange(width), np.arange(height))
        
        # Преобразование частоты и угла в компоненты
        freq_x = self.freq * np.cos(np.radians(self.angle))
        freq_y = self.freq * np.sin(np.radians(self.angle))
        
        # Генерация шума
        noise = self.amplitude * np.sin(2 * np.pi * (freq_x * x + freq_y * y))
        
        # Добавление шума к каждому каналу
        if len(image.shape) == 3:
            noisy_image = image.copy()
            for i in range(3):
                noisy_image[:, :, i] = np.clip(noisy_image[:, :, i] + noise, 0, 1)
        else:
            noisy_image = np.clip(image + noise, 0, 1)
        
        return noisy_image


class FrequencyFilter:
    """Класс для применения частотных фильтров"""
    
    def __init__(self):
        self.filter_type = 'Butterworth'
        self.D0 = 0.2
        self.n = 2
        self.padding_ratio = 0.2
        self.blend_alpha = 0.5
        self.enable_blend = True
        
        # GPU ускорение
        self.gpu_fft = GPUFFTProcessor(ACCELERATION_TYPE)
        
    def _create_filter(self, height, width):
        """Создание частотного фильтра"""
        # Создаем сетку координат
        y, x = np.ogrid[:height, :width]
        center_y, center_x = height // 2, width // 2
        
        # Вычисляем расстояния от центра
        distances = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        
        # Нормализуем расстояния
        max_distance = np.sqrt(center_x**2 + center_y**2)
        distances = distances / max_distance
        
        # Создаем фильтр в зависимости от типа
        if self.filter_type == 'Ideal':
            filter_mask = (distances <= self.D0).astype(np.float32)
            
        elif self.filter_type == 'Gaussian':
            filter_mask = np.exp(-(distances**2) / (2 * self.D0**2))
            
        elif self.filter_type == 'Butterworth':
            filter_mask = 1 / (1 + (distances / self.D0)**(2 * self.n))
            
        else:
            filter_mask = np.ones_like(distances)
            
        return filter_mask

    def apply_filter(self, image):
        """Применение фильтра к изображению"""
        if image is None:
            return None
            
        # Определение размера дополнения
        pad_size = int(min(image.shape[:2]) * self.padding_ratio)
        
        # Дополнение границ
        if len(image.shape) == 3:
            padded = cv2.copyMakeBorder(image, pad_size, pad_size, pad_size, pad_size, 
                                      cv2.BORDER_REFLECT)
        else:
            padded = cv2.copyMakeBorder(image, pad_size, pad_size, pad_size, pad_size, 
                                      cv2.BORDER_REFLECT)
        
        height, width = padded.shape[:2]
        
        # Создаем фильтр
        filter_mask = self._create_filter(height, width)
        
        # Применение фильтра к каждому каналу
        if len(padded.shape) == 3:
            filtered = np.zeros_like(padded)
            for i in range(3):
                # FFT
                f_transform = self.gpu_fft.fft2_gpu(padded[:, :, i])
                f_shift = fftshift(f_transform)
                
                # Применение фильтра
                filtered_shift = f_shift * filter_mask
                
                # Обратное FFT
                f_ishift = ifftshift(filtered_shift)
                filtered[:, :, i] = np.real(self.gpu_fft.ifft2_gpu(f_ishift))
        else:
            # FFT
            f_transform = self.gpu_fft.fft2_gpu(padded)
            f_shift = fftshift(f_transform)
            
            # Применение фильтра
            filtered_shift = f_shift * filter_mask
            
            # Обратное FFT
            f_ishift = ifftshift(filtered_shift)
            filtered = np.real(self.gpu_fft.ifft2_gpu(f_ishift))
        
        # Удаление дополнения
        filtered = filtered[pad_size:-pad_size, pad_size:-pad_size]
        
        # Нормализация
        filtered = np.clip(filtered, 0, 1)
        
        # Смешивание с оригиналом
        if self.enable_blend:
            filtered = self.blend_alpha * filtered + (1 - self.blend_alpha) * image
            filtered = np.clip(filtered, 0, 1)
            filtered = filtered.astype(np.float32)
        
        return filtered

    def set_blend_parameters(self, alpha: float, enable_blend: bool):
        """Установить параметры смешивания"""
        self.blend_alpha = np.clip(alpha, 0.0, 1.0)
        self.enable_blend = enable_blend

    def get_blend_parameters(self):
        """Получить параметры смешивания"""
        return self.blend_alpha, self.enable_blend


class DetailEnhancer:
    """Улучшение деталей изображения с оптимизированным GPU ускорением"""
    
    def __init__(self):
        self.sigma_s = 1
        self.sigma_r = 0.05
        self.contrast_strength = 0.0  # Контраст отключен по умолчанию
        self.use_gpu = ACCELERATION_TYPE != "CPU"  # Автоматически используем GPU если доступен
        self.acceleration_type = ACCELERATION_TYPE
        
    def _process_channel_cpu(self, channel_data):
        """Обработка одного канала изображения на CPU"""
        channel, sigma_s, sigma_r = channel_data
        
        # 1. Unsharp Masking для канала
        blur_sigma = max(0.5, sigma_s)
        blurred = cv2.GaussianBlur(channel, (0, 0), blur_sigma)
        sharp_strength = sigma_r * 1.0
        sharpened = cv2.addWeighted(channel, 1.0 + sharp_strength, blurred, -sharp_strength, 0)
        
        # 2. Laplacian для канала (если нужно)
        if sigma_r > 0.1:
            laplacian = cv2.Laplacian(sharpened, cv2.CV_8U, ksize=3)
            laplacian_strength = (sigma_r - 0.1) * 0.5
            sharpened = cv2.addWeighted(sharpened, 1.0, laplacian, laplacian_strength, 0)
            
        return sharpened
        
    def _process_channel_gpu(self, channel_data):
        """Обработка одного канала изображения на GPU (CUDA или OpenCL)"""
        channel, sigma_s, sigma_r = channel_data
        
        if self.acceleration_type == "CUDA":
            return self._process_channel_cuda(channel_data)
        elif self.acceleration_type == "OpenCL":
            return self._process_channel_opencl(channel_data)
        else:
            return self._process_channel_cpu(channel_data)
            
    def _process_channel_cuda(self, channel_data):
        """Обработка одного канала изображения на CUDA GPU"""
        channel, sigma_s, sigma_r = channel_data
        
        # Загружаем данные на GPU
        gpu_channel = cv2.cuda_GpuMat()
        gpu_channel.upload(channel)
        
        # 1. Unsharp Masking на GPU
        blur_sigma = max(0.5, sigma_s)
        gpu_blurred = cv2.cuda_GpuMat()
        cv2.cuda.GaussianBlur(gpu_channel, gpu_blurred, (0, 0), blur_sigma)
        
        sharp_strength = sigma_r * 1.0
        gpu_sharpened = cv2.cuda_GpuMat()
        cv2.cuda.addWeighted(gpu_channel, 1.0 + sharp_strength, gpu_blurred, -sharp_strength, 0, gpu_sharpened)
        
        # 2. Laplacian на GPU (если нужно)
        if sigma_r > 0.1:
            gpu_laplacian = cv2.cuda_GpuMat()
            cv2.cuda.Laplacian(gpu_sharpened, gpu_laplacian, cv2.CV_8U, ksize=3)
            laplacian_strength = (sigma_r - 0.1) * 0.5
            cv2.cuda.addWeighted(gpu_sharpened, 1.0, gpu_laplacian, laplacian_strength, 0, gpu_sharpened)
        
        # Скачиваем результат с GPU
        result = gpu_sharpened.download()
        return result
        
    def _process_channel_opencl(self, channel_data):
        """Оптимизированная обработка одного канала изображения на OpenCL GPU"""
        channel, sigma_s, sigma_r = channel_data
        
        # Включаем OpenCL и проверяем, что он работает
        if not cv2.ocl.useOpenCL():
            print("WARNING: OpenCL не удалось включить, переключаемся на CPU")
            return self._process_channel_cpu(channel_data)
        
        try:
            # Оптимизация: используем более эффективные размеры ядер
            blur_sigma = max(0.5, sigma_s)
            
            # Оптимизация: используем оптимальные размеры ядер для OpenCL
            kernel_size = int(blur_sigma * 6 + 1)
            if kernel_size % 2 == 0:
                kernel_size += 1
            kernel_size = max(3, min(kernel_size, 31))  # Ограничиваем размер ядра
            
            # 1. Unsharp Masking с OpenCL и оптимизированными параметрами
            blurred = cv2.GaussianBlur(channel, (kernel_size, kernel_size), blur_sigma)
            sharp_strength = sigma_r * 1.0
            sharpened = cv2.addWeighted(channel, 1.0 + sharp_strength, blurred, -sharp_strength, 0)
            
            # 2. Laplacian с OpenCL (если нужно)
            if sigma_r > 0.1:
                laplacian = cv2.Laplacian(sharpened, cv2.CV_8U, ksize=3)
                laplacian_strength = (sigma_r - 0.1) * 0.5
                sharpened = cv2.addWeighted(sharpened, 1.0, laplacian, laplacian_strength, 0)
                
            return sharpened
            
        except Exception as e:
            print(f"WARNING: OpenCL обработка не удалась ({e}), переключаемся на CPU")
            return self._process_channel_cpu(channel_data)
        
    def enhance_details(self, image):
        """Улучшение деталей изображения с оптимизированным GPU/CPU ускорением"""
        if image is None:
            return None
            
        start_time = time.time()
        
        print(f"DEBUG: DetailEnhancer.enhance_details: sigma_s={self.sigma_s}, sigma_r={self.sigma_r}, contrast_strength={self.contrast_strength}")
        print(f"DEBUG: DetailEnhancer.enhance_details: используем {'GPU' if self.use_gpu else 'CPU'}")
        print(f"DEBUG: DetailEnhancer.enhance_details: тип изображения={image.dtype}, форма={image.shape}")
            
        # Конвертация в uint8 для OpenCV
        if image.dtype != np.float32:
            image_uint8 = (image * 255).astype(np.uint8)
        else:
            image_uint8 = (image * 255).astype(np.uint8)
            
        print(f"DEBUG: DetailEnhancer.enhance_details: конвертировано в uint8, диапазон=[{image_uint8.min()}, {image_uint8.max()}]")
        
        # Многопоточная обработка каналов RGB
        channels = [image_uint8[:,:,i] for i in range(3)]
        channel_data = [(ch, self.sigma_s, self.sigma_r) for ch in channels]
        
        # Выбираем метод обработки
        if self.use_gpu:
            try:
                # GPU обработка (последовательно, так как CUDA уже параллелит)
                processed_channels = [self._process_channel_gpu(data) for data in channel_data]
            except Exception as e:
                print(f"WARNING: GPU обработка не удалась ({e}), переключаемся на CPU")
                self.use_gpu = False
                self.acceleration_type = "CPU"
                # CPU обработка с многопоточностью
                with Pool(processes=min(3, cpu_count())) as pool:
                    processed_channels = pool.map(self._process_channel_cpu, channel_data)
        else:
            # CPU обработка с многопоточностью
            with Pool(processes=min(3, cpu_count())) as pool:
                processed_channels = pool.map(self._process_channel_cpu, channel_data)
        
        # Собираем обратно в RGB изображение
        sharpened = np.stack(processed_channels, axis=2)
        
        print(f"DEBUG: DetailEnhancer.enhance_details: unsharp masking применен с blur_sigma={max(0.5, self.sigma_s)}, sharp_strength={self.sigma_r * 1.0}")
        print(f"DEBUG: DetailEnhancer.enhance_details: диапазон после резкости=[{sharpened.min()}, {sharpened.max()}]")
        
        # 3. Улучшение контраста (только если включен)
        if self.contrast_strength > 0:
            # Используем более мягкий CLAHE
            lab = cv2.cvtColor(sharpened, cv2.COLOR_RGB2LAB)
            # Ограничиваем максимальный clipLimit
            clip_limit = 0.5 + (self.contrast_strength * 1.5)  # Максимум 2.0
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8,8))
            lab[:,:,0] = clahe.apply(lab[:,:,0])
            enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
            
            print(f"DEBUG: DetailEnhancer.enhance_details: CLAHE применен с clipLimit={clip_limit}")
            print(f"DEBUG: DetailEnhancer.enhance_details: диапазон после CLAHE=[{enhanced.min()}, {enhanced.max()}]")
        else:
            # Если контраст отключен, пропускаем
            enhanced = sharpened
            print(f"DEBUG: DetailEnhancer.enhance_details: CLAHE пропущен (contrast_strength=0)")
        
        # Конвертация обратно в float32
        enhanced = enhanced.astype(np.float32) / 255.0
        
        end_time = time.time()
        print(f"DEBUG: DetailEnhancer.enhance_details: время выполнения = {end_time - start_time:.3f}с")
        print(f"DEBUG: DetailEnhancer.enhance_details: конвертировано обратно в float32, диапазон=[{enhanced.min():.3f}, {enhanced.max():.3f}]")
        
        return enhanced


class ImageProcessor:
    """Основной класс для обработки изображений"""
    
    def __init__(self):
        self.frequency_filter = FrequencyFilter()
        self.noise_generator = NoiseGenerator()
        self.detail_enhancer = DetailEnhancer()
        self.auto_optimizer = AutoOptimizer()
        
        # GPU ускорение
        self.use_gpu = ACCELERATION_TYPE != "CPU"
        self.acceleration_type = ACCELERATION_TYPE
        
    def auto_optimize_parameters(self, image):
        """Автоматическая оптимизация параметров фильтра"""
        if image is None:
            return False, 0.2, 2
            
        try:
            # Анализ спектра изображения
            peaks, total_energy = self.auto_optimizer.analyze_spectrum(image)
            
            if not peaks:
                return False, 0.2, 2
                
            # Оптимизация параметров
            optimal_D0, optimal_n = self.auto_optimizer.optimize_parameters(image, self.frequency_filter.filter_type)
            
            # Обновление параметров
            self.frequency_filter.D0 = optimal_D0
            self.frequency_filter.n = optimal_n
            
            return True, optimal_D0, optimal_n
            
        except Exception as e:
            print(f"Ошибка автоматической оптимизации: {e}")
            return False, 0.2, 2
        
    def compute_spectrum(self, image: np.ndarray) -> np.ndarray:
        """
        Вычисление спектра Фурье изображения
        
        Args:
            image: Входное изображение
            
        Returns:
            Нормализованный спектр
        """
        # Конвертация в оттенки серого если нужно
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
            
        # Нормализация
        if gray.dtype != np.float32:
            gray = gray.astype(np.float32) / 255.0
            
        # FFT
        f_transform = fft2(gray)
        f_shift = fftshift(f_transform)
        
        # Логарифм для лучшей визуализации
        spectrum = np.log(np.abs(f_shift) + 1)
        
        # Нормализация для отображения
        spectrum = (spectrum - spectrum.min()) / (spectrum.max() - spectrum.min())
        
        return spectrum
        
    def apply_frequency_filter(self, image: np.ndarray) -> np.ndarray:
        """
        Применение частотного фильтра к изображению
        
        Args:
            image: Входное изображение
            
        Returns:
            Отфильтрованное изображение
        """
        return self.frequency_filter.apply_filter(image)
    
    def enhance_details(self, image: np.ndarray) -> np.ndarray:
        """
        Улучшение деталей изображения
        
        Args:
            image: Входное изображение
            
        Returns:
            Улучшенное изображение
        """
        print(f"DEBUG: enhance_details: вызывается с sigma_s={self.detail_enhancer.sigma_s}, sigma_r={self.detail_enhancer.sigma_r}")
        return self.detail_enhancer.enhance_details(image)
    
    def process_image(self, original_image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Полная обработка изображения с оптимизированным GPU ускорением
        
        Args:
            original_image: Исходное изображение
            
        Returns:
            Кортеж (зашумленное, отфильтрованное, спектры)
        """
        start_time = time.time()
        
        # Добавление шума
        noisy_image = self.noise_generator.add_periodic_noise(original_image)
        
        # Применение фильтра
        filtered_image = self.apply_frequency_filter(noisy_image)
        
        # Улучшение деталей
        filtered_image = self.enhance_details(filtered_image)
        
        # Вычисление спектров
        original_spectrum = self.compute_spectrum(original_image)
        noisy_spectrum = self.compute_spectrum(noisy_image)
        filtered_spectrum = self.compute_spectrum(filtered_image)
        
        spectra = (original_spectrum, noisy_spectrum, filtered_spectrum)
        
        end_time = time.time()
        print(f"DEBUG: process_image: общее время обработки = {end_time - start_time:.3f}с")
        
        return noisy_image, filtered_image, spectra
    
    def set_filter_parameters(self, filter_type: str, D0: float, n: int = 2):
        """Установка параметров фильтра"""
        self.frequency_filter.filter_type = filter_type
        self.frequency_filter.D0 = D0
        self.frequency_filter.n = n
    
    def set_noise_parameters(self, freq: float, amp: float, angle: float):
        """Установка параметров шума"""
        self.noise_generator.freq = freq
        self.noise_generator.amplitude = amp
        self.noise_generator.angle = angle
    
    def set_enhancement_parameters(self, sigma_s: float, sigma_r: float, contrast_strength: float = 0.1):
        """Установка параметров улучшения"""
        print(f"DEBUG: set_enhancement_parameters: sigma_s={sigma_s}, sigma_r={sigma_r}, contrast_strength={contrast_strength}")
        self.detail_enhancer.sigma_s = sigma_s
        self.detail_enhancer.sigma_r = sigma_r
        self.detail_enhancer.contrast_strength = contrast_strength
        print(f"DEBUG: set_enhancement_parameters: установлено sigma_s={self.detail_enhancer.sigma_s}, sigma_r={self.detail_enhancer.sigma_r}, contrast_strength={self.detail_enhancer.contrast_strength}")
        
    def set_gpu_mode(self, use_gpu: bool):
        """Переключение между GPU и CPU режимами"""
        if use_gpu and ACCELERATION_TYPE == "CPU":
            print("WARNING: GPU недоступен, переключаемся на CPU")
            self.detail_enhancer.use_gpu = False
            self.detail_enhancer.acceleration_type = "CPU"
            self.frequency_filter.use_gpu = False
            self.frequency_filter.acceleration_type = "CPU"
        else:
            self.detail_enhancer.use_gpu = use_gpu
            self.frequency_filter.use_gpu = use_gpu
            if use_gpu:
                self.detail_enhancer.acceleration_type = ACCELERATION_TYPE
                self.frequency_filter.acceleration_type = ACCELERATION_TYPE
            else:
                self.detail_enhancer.acceleration_type = "CPU"
                self.frequency_filter.acceleration_type = "CPU"
        print(f"DEBUG: set_gpu_mode: используем {self.detail_enhancer.acceleration_type}")
    
    def set_padding_ratio(self, ratio: float):
        """Установка коэффициента дополнения границ"""
        self.frequency_filter.padding_ratio = ratio

    def force_memory_cleanup(self):
        """Принудительная очистка памяти"""
        # Принудительная сборка мусора
        gc.collect()
        
        print("DEBUG: Принудительная очистка памяти выполнена")

    def set_blend_parameters(self, alpha: float, enable_blend: bool):
        """Установить параметры смешивания для фильтра"""
        self.frequency_filter.blend_alpha = np.clip(alpha, 0.0, 1.0)
        self.frequency_filter.enable_blend = enable_blend

    def get_blend_parameters(self):
        """Получить параметры смешивания фильтра"""
        return self.frequency_filter.blend_alpha, self.frequency_filter.enable_blend 