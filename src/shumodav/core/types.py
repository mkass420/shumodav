from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

# float32, [0, 1], HxW или HxWx3 (RGB)
Image = np.ndarray


@dataclass(frozen=True)
class ParamSpec:
    name: str                      # ключ в **kwargs
    label: str                     # подпись в GUI, по-русски
    kind: Literal["float", "int", "choice", "bool"]
    default: Any
    min: float | None = None
    max: float | None = None
    step: float | None = None      # для int: шаг (напр. 2 для нечётных ядер)
    choices: tuple | None = None
    log: bool = False              # логарифмический слайдер
    grid: tuple | None = None      # значения для подбора в bench/tune.py
    visible_if: dict | None = None # показывать только при заданном значении другого параметра
