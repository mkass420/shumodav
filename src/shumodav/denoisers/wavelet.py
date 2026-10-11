import cv2
import numpy as np
import pywt

from ..core.registry import Denoiser, register_denoiser
from ..core.types import Image, ParamSpec


def _bayes_threshold(c: np.ndarray, sigma_n: float) -> float:
    sigma_x = np.sqrt(max(np.mean(c**2) - sigma_n**2, 0.0))
    if sigma_x == 0:
        return float(np.abs(c).max())
    return float(sigma_n**2 / sigma_x)


def _mad_sigma(c: np.ndarray) -> float:
    return float(np.median(np.abs(c)) / 0.6745)


@register_denoiser
class Wavelet(Denoiser):
    name = "wavelet"
    label = "Вейвлет (DWT/SWT)"
    family = "wavelet"
    params = (
        ParamSpec("wavelet", "Вейвлет", "choice", "db4",
                  choices=("db1", "db2", "db4", "sym4", "coif1", "bior1.3"), grid=("db4", "sym4")),
        ParamSpec("level", "Уровней", "int", 2, 1, 4, 1, grid=(2, 3)),
        ParamSpec("threshold", "Порог", "float", 0.1, 0.02, 0.3, 0.01, grid=(0.03, 0.05, 0.08, 0.12, 0.2, 0.3)),
        ParamSpec("mode", "Режим порога", "choice", "soft", choices=("soft", "hard")),
        ParamSpec("swt", "Стационарное (SWT)", "bool", False, grid=(True,)),
        ParamSpec("threshold_rule", "Правило порога", "choice", "manual", choices=("manual", "bayes")),
        ParamSpec("chroma_strength", "Подавление цветного шума", "float", 0.5, 0.0, 1.0, 0.05),
        ParamSpec("post_smooth", "Билатеральное сглаживание", "bool", False),
    )

    # --- пороги ---
    def _thr(self, c, base, rule, sigma_n):
        return base if rule == "manual" else _bayes_threshold(c, sigma_n)

    def _channel(self, ch, wavelet, level, base, mode, swt, rule, chroma, strength):
        boost = 1.0 + 5.0 * strength if chroma else 1.0
        supp = 1.0 - 0.8 * strength if chroma else 1.0
        h, w = ch.shape

        def shrink(c, sigma_n):
            t = self._thr(c, base, rule, sigma_n) * boost
            return pywt.threshold(c, t, mode=mode) * supp

        if swt:
            f = 2**level
            padded = np.pad(ch, ((0, (-h) % f), (0, (-w) % f)), "edge")
            coeffs = pywt.swt2(padded, wavelet, level=level)
            out = []
            for cA, (cH, cV, cD) in coeffs:
                sn = _mad_sigma(cD)  # SWT без нормировки: σ оцениваем на каждом уровне
                out.append((cA, (shrink(cH, sn), shrink(cV, sn), shrink(cD, sn))))
            return pywt.iswt2(out, wavelet)[:h, :w]
        coeffs = pywt.wavedec2(ch, wavelet, level=level)
        sn = _mad_sigma(coeffs[-1][2])  # HH самого тонкого уровня
        out = [coeffs[0]] + [tuple(shrink(c, sn) for c in det) for det in coeffs[1:]]
        return pywt.waverec2(out, wavelet)[:h, :w]

    def _safe_level(self, img, wavelet, level, swt):
        w = pywt.Wavelet(wavelet)
        m = min(img.shape[:2])
        if swt:
            return max(1, min(level, int(np.log2(m))))  # SWT паддит до кратного 2**level
        return max(1, min(level, pywt.dwt_max_level(m, w.dec_len)))

    def denoise(self, img, wavelet="db4", level=2, threshold=0.1, mode="soft", swt=False,
                threshold_rule="manual", chroma_strength=0.5, post_smooth=False):
        level = self._safe_level(img, wavelet, int(level), swt)
        if img.ndim == 3:
            ycc = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
            chans = [self._channel(ycc[..., i], wavelet, level, threshold, mode, swt,
                                   threshold_rule, i > 0, chroma_strength) for i in range(3)]
            res = cv2.cvtColor(np.dstack(chans).astype(np.float32), cv2.COLOR_YCrCb2RGB)
        else:
            res = self._channel(img, wavelet, level, threshold, mode, swt, threshold_rule, False, 0.0)
        res = np.clip(res, 0.0, 1.0).astype(np.float32)
        if post_smooth:
            res = cv2.bilateralFilter(res, 5, 0.08, 5)
        return res

    def visualize(self, img, wavelet="db4", level=2, **_):
        gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        level = self._safe_level(gray, wavelet, int(level), False)
        coeffs = pywt.wavedec2(gray, wavelet, level=level)

        def norm(a):
            a = np.log(np.abs(a) + 1e-5)
            rng = a.max() - a.min()
            return (a - a.min()) / rng if rng > 0 else np.zeros_like(a)

        mosaic = norm(coeffs[0])
        for lh, hl, hh in coeffs[1:]:
            lh, hl, hh = norm(lh), norm(hl), norm(hh)
            th, tw = lh.shape
            mosaic = cv2.resize(mosaic, (tw, th), interpolation=cv2.INTER_NEAREST)
            mosaic = np.vstack((np.hstack((mosaic, lh)), np.hstack((hl, hh))))
        return mosaic.astype(np.float32)
