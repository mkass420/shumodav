from abc import ABC, abstractmethod
from typing import ClassVar, Literal

import numpy as np

from .types import Image, ParamSpec

_NOISES: dict[str, "object"] = {}
_DENOISERS: dict[str, "Denoiser"] = {}


class Denoiser(ABC):
    name: ClassVar[str]
    label: ClassVar[str]
    family: ClassVar[Literal["spatial", "frequency", "wavelet", "neural"]]
    params: ClassVar[tuple[ParamSpec, ...]] = ()
    supports_color: ClassVar[bool] = True  # False: pipeline обрабатывает каналы по отдельности

    def is_available(self) -> bool:
        return True

    @abstractmethod
    def denoise(self, img: Image, **p) -> Image: ...

    def visualize(self, img: Image, **p) -> np.ndarray | None:
        return None  # спектр с маской, мозаика и т.п.


def register_noise(cls):
    _NOISES[cls.name] = cls()
    return cls


def register_denoiser(cls):
    _DENOISERS[cls.name] = cls()
    return cls


def get_noise(name: str):
    if name not in _NOISES:
        raise KeyError(f"Неизвестный шум: {name!r}. Есть: {sorted(_NOISES)}")
    return _NOISES[name]


def get_denoiser(name: str) -> Denoiser:
    if name not in _DENOISERS:
        raise KeyError(f"Неизвестный метод: {name!r}. Есть: {sorted(_DENOISERS)}")
    return _DENOISERS[name]


def list_noises() -> list:
    return list(_NOISES.values())


def list_denoisers(available_only: bool = True) -> list[Denoiser]:
    return [d for d in _DENOISERS.values() if not available_only or d.is_available()]


def resolve_params(specs: tuple[ParamSpec, ...], p: dict | None) -> dict:
    """Подставляет default для отсутствующих ключей, лишние ключи отбрасывает."""
    p = p or {}
    return {s.name: p.get(s.name, s.default) for s in specs}
