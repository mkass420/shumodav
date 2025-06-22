"""
Утилиты для работы с изображениями
"""

import cv2
import numpy as np
from typing import Tuple, Optional


def resize_image(image: np.ndarray, target_size: Tuple[int, int], 
                interpolation: int = cv2.INTER_LANCZOS4) -> np.ndarray:
    """
    Изменение размера изображения с сохранением пропорций
    
    Args:
        image: Входное изображение
        target_size: Целевой размер (width, height)
        interpolation: Метод интерполяции
        
    Returns:
        Измененное изображение
    """
    return cv2.resize(image, target_size, interpolation=interpolation)


def normalize_image(image: np.ndarray, min_val: float = 0.0, 
                   max_val: float = 255.0) -> np.ndarray:
    """
    Нормализация изображения
    
    Args:
        image: Входное изображение
        min_val: Минимальное значение
        max_val: Максимальное значение
        
    Returns:
        Нормализованное изображение
    """
    img_min = np.min(image)
    img_max = np.max(image)
    
    if img_max == img_min:
        return np.full_like(image, min_val)
    
    normalized = (image - img_min) / (img_max - img_min) * (max_val - min_val) + min_val
    return normalized.astype(image.dtype)


def convert_color_space(image: np.ndarray, 
                       from_space: str = 'BGR', 
                       to_space: str = 'RGB') -> np.ndarray:
    """
    Конвертация цветового пространства
    
    Args:
        image: Входное изображение
        from_space: Исходное цветовое пространство
        to_space: Целевое цветовое пространство
        
    Returns:
        Изображение в новом цветовом пространстве
    """
    if from_space == 'BGR' and to_space == 'RGB':
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    elif from_space == 'RGB' and to_space == 'BGR':
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    elif from_space == 'BGR' and to_space == 'GRAY':
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif from_space == 'RGB' and to_space == 'GRAY':
        return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        return image


def apply_gamma_correction(image: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    """
    Применение гамма-коррекции
    
    Args:
        image: Входное изображение
        gamma: Значение гаммы
        
    Returns:
        Изображение с гамма-коррекцией
    """
    # Нормализация в диапазон [0, 1]
    normalized = image.astype(np.float32) / 255.0
    
    # Применение гамма-коррекции
    corrected = np.power(normalized, gamma)
    
    # Возврат в диапазон [0, 255]
    return (corrected * 255).astype(np.uint8)


def create_histogram(image: np.ndarray, bins: int = 256) -> Tuple[np.ndarray, np.ndarray]:
    """
    Создание гистограммы изображения
    
    Args:
        image: Входное изображение
        bins: Количество бинов
        
    Returns:
        Кортеж (значения бинов, частоты)
    """
    if len(image.shape) == 3:
        # Для цветного изображения создаем гистограмму для каждого канала
        histograms = []
        for i in range(3):
            hist, bins_edges = np.histogram(image[:, :, i].flatten(), bins=bins, range=(0, 256))
            histograms.append((hist, bins_edges))
        return histograms
    else:
        # Для серого изображения
        hist, bins_edges = np.histogram(image.flatten(), bins=bins, range=(0, 256))
        return hist, bins_edges 