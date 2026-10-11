import numpy as np
import pytest
from skimage import data

import shumodav.denoisers  # noqa: F401  (регистрация методов)
from shumodav.core.metrics import psnr
from shumodav.core.pipeline import apply_noise, run_denoiser
from shumodav.core.registry import list_denoisers

NAMES = [d.name for d in list_denoisers() if not d.name.startswith("identity")]


def _gray(n=128):
    return (data.camera()[:n, :n] / 255.0).astype(np.float32)


def _color(n=64):
    return (data.astronaut()[:n, :n] / 255.0).astype(np.float32)


def test_all_blocks_methods_registered():
    expected = {"mean", "median", "gaussian", "bilateral", "nlm", "fft_lowpass", "fft_notch", "wavelet"}
    assert expected <= set(NAMES)


@pytest.mark.parametrize("img", [_gray(64), _color(64)], ids=["gray", "color"])
@pytest.mark.parametrize("name", NAMES)
def test_smoke(name, img):
    before = img.copy()
    res, sec = run_denoiser(img, name, {})
    assert res.shape == img.shape and res.dtype == np.float32
    assert res.min() >= 0 and res.max() <= 1
    assert np.array_equal(img, before)
    from shumodav.core.registry import get_denoiser

    vis = get_denoiser(name).visualize(img)
    assert vis is None or isinstance(vis, np.ndarray)


def test_median_big_kernel():
    res, _ = run_denoiser(_gray(64), "median", {"ksize": 7})
    assert res.shape == (64, 64)


@pytest.mark.parametrize("kind", ["ideal", "gaussian", "butterworth"])
def test_fft_kinds(kind):
    res, _ = run_denoiser(_color(64), "fft_lowpass", {"kind": kind})
    assert res.shape == (64, 64, 3)


@pytest.mark.parametrize(
    "params",
    [{"swt": True}, {"threshold_rule": "bayes"}, {"threshold_rule": "bayes", "swt": True},
     {"mode": "hard"}, {"post_smooth": True}, {"level": 4}, {"wavelet": "bior1.3", "swt": True}],
)
@pytest.mark.parametrize("img", [_gray(64), _color(64), _gray(128)[:63, :65]], ids=["gray", "color", "odd"])
def test_wavelet_variants(params, img):
    res, _ = run_denoiser(img, "wavelet", params)
    assert res.shape == img.shape and res.dtype == np.float32


def test_fft_lowpass_keeps_brightness():
    img = _gray()
    res, _ = run_denoiser(img, "fft_lowpass", {})
    assert abs(res.mean() - img.mean()) / img.mean() < 0.01


def test_fft_notch_on_periodic():
    clean = _gray()
    noisy = apply_noise(clean, "periodic", {"amplitude": 0.2, "freq": 0.1, "angle": 45}, seed=0)
    res, _ = run_denoiser(noisy, "fft_notch", {})
    assert psnr(clean, res) >= psnr(clean, noisy) + 5


def test_median_helps_on_salt_pepper():
    clean = _gray()
    noisy = apply_noise(clean, "salt_pepper", {"amount": 0.1}, seed=0)
    res, _ = run_denoiser(noisy, "median", {"ksize": 3})
    assert psnr(clean, res) > psnr(clean, noisy)


@pytest.mark.parametrize("name", ["wavelet", "gaussian", "nlm", "bilateral"])
def test_helps_on_gaussian(name):
    clean = _gray()
    noisy = apply_noise(clean, "gaussian", {"sigma": 25}, seed=0)
    res, _ = run_denoiser(noisy, name, {})
    assert psnr(clean, res) > psnr(clean, noisy)
