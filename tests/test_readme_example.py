"""Пример «как добавить новый метод» из README должен работать."""
import numpy as np

# --- начало примера (копия блока из README) ---
import cv2

from shumodav.core.registry import Denoiser, register_denoiser
from shumodav.core.types import ParamSpec


@register_denoiser
class BoxBlur(Denoiser):
    name = "box_blur"          # уникальное имя метода
    label = "Усреднение 5×5"   # подпись в GUI
    family = "spatial"         # spatial | frequency | wavelet | neural
    params = (ParamSpec("ksize", "Размер ядра", "int", 5, 3, 15, 2, grid=(3, 5, 7)),)

    def denoise(self, img, ksize=5):
        return cv2.blur(img, (ksize, ksize))  # вход не менять, вернуть float32 [0, 1]
# --- конец примера ---


def test_example_method_works():
    from shumodav.core.pipeline import run_denoiser

    img = np.random.default_rng(0).random((32, 32, 3)).astype(np.float32)
    res, _ = run_denoiser(img, "box_blur", {"ksize": 3})
    assert res.shape == img.shape and res.dtype == np.float32 and 0 <= res.min() and res.max() <= 1
