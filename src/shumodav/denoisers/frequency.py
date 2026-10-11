import cv2
import numpy as np
from scipy.fft import fft2, fftshift, ifft2, ifftshift
from scipy.ndimage import maximum_filter

from ..core.registry import Denoiser, register_denoiser
from ..core.types import Image, ParamSpec


def _channels(img: Image) -> list[np.ndarray]:
    return [img] if img.ndim == 2 else [img[..., c] for c in range(img.shape[2])]


def _gray(img: Image) -> np.ndarray:
    return img if img.ndim == 2 else img.mean(axis=2)


def _filter(img: Image, mask: np.ndarray) -> Image:
    """Умножает спектр каждого канала на mask (центрированную) и возвращает картинку."""
    out = [np.real(ifft2(ifftshift(fftshift(fft2(ch)) * mask))) for ch in _channels(img)]
    return np.stack(out, axis=-1) if img.ndim == 3 else out[0]


def _log_spectrum(gray: np.ndarray) -> np.ndarray:
    s = np.log1p(np.abs(fftshift(fft2(gray))))
    return ((s - s.min()) / (s.max() - s.min() + 1e-9)).astype(np.float32)


def _lowpass_mask(h: int, w: int, kind: str, d0: float, order: int) -> np.ndarray:
    y, x = np.ogrid[:h, :w]
    cy, cx = h // 2, w // 2
    dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2) / np.sqrt(cx**2 + cy**2)  # 0..1 до угла
    if kind == "ideal":
        mask = (dist <= d0).astype(np.float32)
    elif kind == "gaussian":
        mask = np.exp(-(dist**2) / (2 * d0**2))
    else:  # butterworth
        mask = 1.0 / (1.0 + (dist / d0) ** (2 * order))
    return mask.astype(np.float32)


@register_denoiser
class FFTLowpass(Denoiser):
    name = "fft_lowpass"
    label = "ФНЧ (FFT)"
    family = "frequency"
    params = (
        ParamSpec("kind", "Тип фильтра", "choice", "butterworth",
                  choices=("ideal", "gaussian", "butterworth"), grid=("ideal", "gaussian", "butterworth")),
        ParamSpec("d0", "Частота среза", "float", 0.2, 0.05, 0.5, 0.01, grid=(0.05, 0.1, 0.15, 0.2, 0.3, 0.5)),
        ParamSpec("order", "Порядок", "int", 2, 1, 8, 1, grid=(1, 2, 4), visible_if={"kind": "butterworth"}),
        ParamSpec("padding", "Паддинг", "float", 0.1, 0.0, 0.5, 0.05),
    )

    def _padded(self, img, padding):
        ph, pw = int(img.shape[0] * padding), int(img.shape[1] * padding)
        return cv2.copyMakeBorder(img, ph, ph, pw, pw, cv2.BORDER_REFLECT), ph, pw

    def denoise(self, img, kind="butterworth", d0=0.2, order=2, padding=0.1):
        padded, ph, pw = self._padded(img, padding)
        mask = _lowpass_mask(*padded.shape[:2], kind, d0, order)
        res = _filter(padded, mask)
        return res[ph : res.shape[0] - ph, pw : res.shape[1] - pw]

    def visualize(self, img, kind="butterworth", d0=0.2, order=2, padding=0.1):
        gray = _gray(img)
        mask = _lowpass_mask(*gray.shape, kind, d0, order)
        spec = _log_spectrum(gray)
        # область, которую фильтр гасит, подкрашиваем красным
        rgb = np.stack([spec, spec * mask, spec * mask], axis=-1)
        return np.clip(rgb, 0, 1).astype(np.float32)


def _find_peaks(gray, threshold, exclude_center, max_peaks=20):
    """Пики спектра (в координатах fftshift) вместе с симметричными."""
    h, w = gray.shape
    win = np.outer(np.hanning(h), np.hanning(w))
    amp = np.abs(fftshift(fft2((gray - gray.mean()) * win))).astype(np.float32)
    # локальный фон спектра (среднее в окне 15×15): пик периодического шума резкий, а НЧ-содержимое картинки гладкое
    base = cv2.blur(amp, (15, 15)) + 1e-9
    cy, cx = h // 2, w // 2
    y, x = np.ogrid[:h, :w]
    far = np.sqrt((x - cx) ** 2 + (y - cy) ** 2) > exclude_center * min(h, w)
    is_peak = (amp == maximum_filter(amp, size=5)) & (amp > threshold * base) & far
    ys, xs = np.nonzero(is_peak)
    order = np.argsort(amp[ys, xs])[::-1][:max_peaks]
    peaks = set()
    for i in order:
        py, px = int(ys[i]), int(xs[i])
        peaks.add((py, px))
        peaks.add((2 * cy - py, 2 * cx - px))  # симметричный пик
    return [(py, px) for py, px in peaks if 0 <= py < h and 0 <= px < w]


def _notch_mask(shape, peaks, radius):
    h, w = shape
    y, x = np.ogrid[:h, :w]
    mask = np.ones(shape, np.float32)
    for py, px in peaks:
        mask *= 1.0 - np.exp(-((x - px) ** 2 + (y - py) ** 2) / (2.0 * radius**2))
    return mask


@register_denoiser
class FFTNotch(Denoiser):
    name = "fft_notch"
    label = "Режекторный (FFT notch)"
    family = "frequency"
    params = (
        ParamSpec("threshold", "Порог (× локальный фон)", "float", 8.0, 2.0, 100.0, 1.0, log=True, grid=(5, 8, 12)),
        ParamSpec("radius", "Радиус «дырки»", "float", 4.0, 1.0, 16.0, 0.5, grid=(2, 4, 8)),
        ParamSpec("exclude_center", "Исключить центр (доля)", "float", 0.05, 0.0, 0.3, 0.01),
    )

    def denoise(self, img, threshold=8.0, radius=4.0, exclude_center=0.05):
        peaks = _find_peaks(_gray(img), threshold, exclude_center)
        return _filter(img, _notch_mask(img.shape[:2], peaks, radius))

    def visualize(self, img, threshold=8.0, radius=4.0, exclude_center=0.05):
        gray = _gray(img)
        spec = _log_spectrum(gray)
        rgb = np.ascontiguousarray(np.stack([spec] * 3, axis=-1))
        for py, px in _find_peaks(gray, threshold, exclude_center):
            cv2.circle(rgb, (px, py), max(2, int(radius)), (1.0, 0.0, 0.0), 1)
        return rgb
