from pathlib import Path

import pytest

pytest.importorskip("torch")

import shumodav.denoisers  # noqa: E402,F401
from shumodav.core.io import load_image  # noqa: E402
from shumodav.core.metrics import psnr  # noqa: E402
from shumodav.core.pipeline import apply_noise, run_denoiser  # noqa: E402
from shumodav.core.registry import get_denoiser  # noqa: E402

IMG = Path(__file__).resolve().parent.parent / "data/set12/05.png"


@pytest.mark.skipif(not IMG.exists(), reason="нет данных")
@pytest.mark.parametrize("name", ["dncnn", "drunet"])
def test_gaussian_25_above_29(name):
    if not get_denoiser(name).is_available():
        pytest.skip("нет весов")
    clean = load_image(IMG, gray=True)
    noisy = apply_noise(clean, "gaussian", {"sigma": 25}, seed=0)
    res, _ = run_denoiser(noisy, name, {})
    assert psnr(clean, res) > 29
