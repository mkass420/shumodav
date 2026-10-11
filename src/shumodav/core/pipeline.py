import time

import numpy as np

from . import metrics, noise  # noqa: F401  (noise: регистрация шумов)
from .registry import get_denoiser, get_noise, resolve_params
from .types import Image


def apply_noise(img: Image, noise_name: str, params: dict | None, seed: int) -> Image:
    noise = get_noise(noise_name)
    rng = np.random.default_rng(seed)
    return noise.apply(img, rng, **resolve_params(noise.params, params))


def run_denoiser(img: Image, denoiser_name: str, params: dict | None = None) -> tuple[Image, float]:
    d = get_denoiser(denoiser_name)
    p = resolve_params(d.params, params)

    def run() -> np.ndarray:
        if img.ndim == 3 and not d.supports_color:
            return np.stack([d.denoise(img[..., c], **p) for c in range(img.shape[2])], axis=-1)
        return d.denoise(img, **p)

    if d.family == "neural":
        run()  # прогрев перед замером
    t0 = time.perf_counter()
    res = run()
    seconds = time.perf_counter() - t0
    return np.clip(res, 0.0, 1.0).astype(np.float32), seconds


def evaluate(clean: Image, noisy: Image, result: Image) -> dict:
    return {
        "psnr_noisy": metrics.psnr(clean, noisy),
        "psnr": metrics.psnr(clean, result),
        "ssim_noisy": metrics.ssim(clean, noisy),
        "ssim": metrics.ssim(clean, result),
    }
