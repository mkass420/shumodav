import cv2
import numpy as np

from ..core.registry import Denoiser, register_denoiser
from ..core.types import Image, ParamSpec


def _to_u8(img: Image) -> np.ndarray:
    return np.round(np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)


def _from_u8(img: np.ndarray) -> Image:
    return img.astype(np.float32) / 255.0


def _odd(k: int) -> int:
    return int(k) | 1


@register_denoiser
class Mean(Denoiser):
    name = "mean"
    label = "Среднее"
    family = "spatial"
    params = (ParamSpec("ksize", "Размер ядра", "int", 3, 3, 15, 2, grid=(3, 5, 7)),)

    def denoise(self, img, ksize=3):
        k = _odd(ksize)
        return cv2.blur(img, (k, k))


@register_denoiser
class Median(Denoiser):
    name = "median"
    label = "Медианный"
    family = "spatial"
    params = (ParamSpec("ksize", "Размер ядра", "int", 3, 3, 15, 2, grid=(3, 5, 7)),)

    def denoise(self, img, ksize=3):
        k = _odd(ksize)
        if k <= 5:
            return cv2.medianBlur(img, k)
        return _from_u8(cv2.medianBlur(_to_u8(img), k))  # для float32 только ksize ≤ 5


@register_denoiser
class Gaussian(Denoiser):
    name = "gaussian"
    label = "Гауссов фильтр"
    family = "spatial"
    params = (ParamSpec("sigma", "Сигма", "float", 1.0, 0.1, 5.0, 0.1, grid=(0.5, 1, 1.5, 2, 3)),)

    def denoise(self, img, sigma=1.0):
        return cv2.GaussianBlur(img, (0, 0), sigma)


@register_denoiser
class Bilateral(Denoiser):
    name = "bilateral"
    label = "Билатеральный"
    family = "spatial"
    params = (
        ParamSpec("d", "Диаметр", "int", 9, 3, 15, 2, grid=(5, 9)),
        ParamSpec("sigma_color", "Сигма по цвету", "float", 0.1, 0.01, 0.5, 0.01, grid=(0.05, 0.1, 0.2)),
        ParamSpec("sigma_space", "Сигма по расстоянию", "float", 5.0, 1.0, 15.0, 0.5, grid=(3, 5)),
    )

    def denoise(self, img, d=9, sigma_color=0.1, sigma_space=5.0):
        return cv2.bilateralFilter(img, int(d), sigma_color, sigma_space)


@register_denoiser
class NLMeans(Denoiser):
    name = "nlm"
    label = "Нелокальных средних"
    family = "spatial"
    params = (ParamSpec("h", "Сила h (0..255)", "float", 20.0, 1.0, 50.0, 1.0, grid=(5, 10, 15, 20, 30)),)

    def denoise(self, img, h=20.0):
        u8 = _to_u8(img)
        if u8.ndim == 2:
            out = cv2.fastNlMeansDenoising(u8, None, float(h), 7, 21)
        else:
            out = cv2.fastNlMeansDenoisingColored(u8, None, float(h), float(h), 7, 21)
        return _from_u8(out)
