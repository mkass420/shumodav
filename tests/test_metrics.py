import numpy as np
import pytest

from shumodav.core.metrics import Timer, psnr, ssim


def test_psnr_identical_is_inf():
    x = np.random.default_rng(0).random((32, 32)).astype(np.float32)
    assert psnr(x, x) == float("inf")


def test_psnr_formula():
    x = np.full((32, 32), 0.5, np.float32)
    y = x + 0.1
    assert psnr(x, y) == pytest.approx(10 * np.log10(1 / 0.01), abs=1e-3)


def test_ssim_identical_and_color():
    x = np.random.default_rng(0).random((32, 32, 3)).astype(np.float32)
    assert ssim(x, x) == pytest.approx(1.0)


def test_timer():
    with Timer() as t:
        sum(range(1000))
    assert t.seconds > 0
