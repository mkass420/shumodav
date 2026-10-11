from abc import ABC, abstractmethod
from typing import ClassVar

import numpy as np

from .registry import register_noise
from .types import Image, ParamSpec


def _finish(x: np.ndarray) -> Image:
    return np.clip(x, 0.0, 1.0).astype(np.float32)


class NoiseModel(ABC):
    name: ClassVar[str]
    label: ClassVar[str]
    params: ClassVar[tuple[ParamSpec, ...]]
    levels: ClassVar[dict[str, dict]]  # {"low": {...}, "mid": {...}, "high": {...}}

    @abstractmethod
    def apply(self, img: Image, rng: np.random.Generator, **p) -> Image: ...


def _levels(key: str, vals: tuple, **extra) -> dict[str, dict]:
    return {lv: {key: v, **extra} for lv, v in zip(("low", "mid", "high"), vals)}


@register_noise
class GaussianNoise(NoiseModel):
    name = "gaussian"
    label = "Гауссов"
    params = (ParamSpec("sigma", "Сигма (0..255)", "float", 25.0, 1.0, 100.0, 1.0),)
    levels = _levels("sigma", (15, 25, 50))

    def apply(self, img, rng, sigma=25.0):
        return _finish(img + rng.normal(0.0, sigma / 255.0, img.shape))


@register_noise
class SaltPepperNoise(NoiseModel):
    name = "salt_pepper"
    label = "Соль и перец"
    params = (
        ParamSpec("amount", "Доля пикселей", "float", 0.10, 0.0, 0.5, 0.01),
        ParamSpec("ratio", "Доля соли", "float", 0.5, 0.0, 1.0, 0.05),
    )
    levels = _levels("amount", (0.05, 0.10, 0.20), ratio=0.5)

    def apply(self, img, rng, amount=0.10, ratio=0.5):
        out = np.array(img, dtype=np.float32)
        u = rng.random(img.shape[:2])  # один пиксель — все каналы
        out[u < amount * ratio] = 1.0
        out[(u >= amount * ratio) & (u < amount)] = 0.0
        return _finish(out)


@register_noise
class SpeckleNoise(NoiseModel):
    name = "speckle"
    label = "Спекл"
    params = (ParamSpec("sigma", "Сигма", "float", 0.20, 0.01, 0.6, 0.01),)
    levels = _levels("sigma", (0.10, 0.20, 0.30))

    def apply(self, img, rng, sigma=0.20):
        return _finish(img + img * rng.normal(0.0, sigma, img.shape))


@register_noise
class PoissonNoise(NoiseModel):
    name = "poisson"
    label = "Пуассона"
    params = (ParamSpec("peak", "Пик", "float", 30.0, 1.0, 255.0, 1.0, log=True),)
    levels = _levels("peak", (100, 30, 10))

    def apply(self, img, rng, peak=30.0):
        x = np.clip(img, 0.0, 1.0)
        return _finish(rng.poisson(x * peak) / peak)


@register_noise
class PeriodicNoise(NoiseModel):
    name = "periodic"
    label = "Периодический"
    params = (
        ParamSpec("amplitude", "Амплитуда", "float", 0.20, 0.0, 0.5, 0.01),
        ParamSpec("freq", "Частота (цикл/пикс)", "float", 0.1, 0.01, 0.5, 0.01),
        ParamSpec("angle", "Угол (град)", "float", 45.0, 0.0, 180.0, 1.0),
    )
    levels = _levels("amplitude", (0.10, 0.20, 0.30), freq=0.1, angle=45.0)

    def apply(self, img, rng, amplitude=0.20, freq=0.1, angle=45.0):
        h, w = img.shape[:2]
        yy, xx = np.mgrid[0:h, 0:w]
        th = np.deg2rad(angle)
        phase = rng.uniform(0.0, 2 * np.pi)
        wave = amplitude * np.sin(2 * np.pi * freq * (xx * np.cos(th) + yy * np.sin(th)) + phase)
        if img.ndim == 3:
            wave = wave[..., None]
        return _finish(img + wave)
