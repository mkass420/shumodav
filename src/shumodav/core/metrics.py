import time

import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    with np.errstate(divide="ignore"):
        return float(peak_signal_noise_ratio(a, b, data_range=1.0))


def ssim(a: np.ndarray, b: np.ndarray) -> float:
    ch = -1 if a.ndim == 3 else None
    return float(structural_similarity(a, b, data_range=1.0, channel_axis=ch))


class Timer:
    """with Timer() as t: ...; t.seconds"""

    seconds: float = 0.0

    def __enter__(self):
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.seconds = time.perf_counter() - self._t0
