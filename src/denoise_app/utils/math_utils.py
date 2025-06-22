"""
Математические утилиты для оценки качества изображений
"""

import numpy as np
from typing import Optional


def compute_psnr(original: np.ndarray, processed: np.ndarray, 
                max_val: float = 255.0) -> float:
    """
    Вычисление Peak Signal-to-Noise Ratio (PSNR)
    
    Args:
        original: Исходное изображение
        processed: Обработанное изображение
        max_val: Максимальное значение пикселя
        
    Returns:
        Значение PSNR в дБ
    """
    mse = np.mean((original.astype(np.float64) - processed.astype(np.float64)) ** 2)
    if mse == 0:
        return float('inf')
    
    psnr = 20 * np.log10(max_val / np.sqrt(mse))
    return psnr


def compute_ssim(original: np.ndarray, processed: np.ndarray, 
                window_size: int = 11, sigma: float = 1.5) -> float:
    """
    Вычисление Structural Similarity Index (SSIM)
    
    Args:
        original: Исходное изображение
        processed: Обработанное изображение
        window_size: Размер окна
        sigma: Стандартное отклонение для гауссовского окна
        
    Returns:
        Значение SSIM
    """
    if len(original.shape) == 3:
        # Для цветных изображений вычисляем SSIM для каждого канала
        ssim_values = []
        for i in range(3):
            ssim_val = _compute_ssim_single_channel(
                original[:, :, i], processed[:, :, i], window_size, sigma
            )
            ssim_values.append(ssim_val)
        return np.mean(ssim_values)
    else:
        # Для серых изображений
        return _compute_ssim_single_channel(original, processed, window_size, sigma)


def _compute_ssim_single_channel(img1: np.ndarray, img2: np.ndarray, 
                                window_size: int, sigma: float) -> float:
    """Вычисление SSIM для одного канала"""
    # Параметры SSIM
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    
    # Создание гауссовского окна
    window = _create_gaussian_window(window_size, sigma)
    window = window / np.sum(window)
    
    # Вычисление средних значений
    mu1 = _apply_window(img1, window)
    mu2 = _apply_window(img2, window)
    
    # Вычисление дисперсий и ковариации
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = _apply_window(img1 ** 2, window) - mu1_sq
    sigma2_sq = _apply_window(img2 ** 2, window) - mu2_sq
    sigma12 = _apply_window(img1 * img2, window) - mu1_mu2
    
    # Вычисление SSIM
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    
    return np.mean(ssim_map)


def _create_gaussian_window(size: int, sigma: float) -> np.ndarray:
    """Создание гауссовского окна"""
    ax = np.arange(-size // 2 + 1, size // 2 + 1)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
    return kernel


def _apply_window(img: np.ndarray, window: np.ndarray) -> np.ndarray:
    """Применение окна к изображению"""
    from scipy import ndimage
    return ndimage.convolve(img, window, mode='constant')


def compute_mse(original: np.ndarray, processed: np.ndarray) -> float:
    """
    Вычисление Mean Squared Error (MSE)
    
    Args:
        original: Исходное изображение
        processed: Обработанное изображение
        
    Returns:
        Значение MSE
    """
    return np.mean((original.astype(np.float64) - processed.astype(np.float64)) ** 2)


def compute_mae(original: np.ndarray, processed: np.ndarray) -> float:
    """
    Вычисление Mean Absolute Error (MAE)
    
    Args:
        original: Исходное изображение
        processed: Обработанное изображение
        
    Returns:
        Значение MAE
    """
    return np.mean(np.abs(original.astype(np.float64) - processed.astype(np.float64)))


def compute_entropy(image: np.ndarray) -> float:
    """
    Вычисление энтропии изображения
    
    Args:
        image: Входное изображение
        
    Returns:
        Значение энтропии
    """
    if len(image.shape) == 3:
        # Для цветных изображений вычисляем энтропию для каждого канала
        entropies = []
        for i in range(3):
            hist, _ = np.histogram(image[:, :, i].flatten(), bins=256, range=(0, 256))
            hist = hist[hist > 0]  # Убираем нулевые значения
            prob = hist / np.sum(hist)
            entropy = -np.sum(prob * np.log2(prob))
            entropies.append(entropy)
        return np.mean(entropies)
    else:
        # Для серых изображений
        hist, _ = np.histogram(image.flatten(), bins=256, range=(0, 256))
        hist = hist[hist > 0]  # Убираем нулевые значения
        prob = hist / np.sum(hist)
        return -np.sum(prob * np.log2(prob)) 