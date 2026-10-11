import numpy as np
import pytest

from shumodav.core import noise  # noqa: F401
from shumodav.core.registry import list_noises

SHAPES = [(64, 64), (64, 64, 3), (50, 50)]
NOISES = [n.name for n in list_noises()]


def _img(shape):
    return np.random.default_rng(0).random(shape).astype(np.float32)


def test_five_noises():
    assert set(NOISES) == {"gaussian", "salt_pepper", "speckle", "poisson", "periodic"}


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("name", NOISES)
def test_noise_contract(name, shape):
    from shumodav.core.registry import get_noise

    n = get_noise(name)
    img = _img(shape)
    before = img.copy()
    for params in n.levels.values():
        a = n.apply(img, np.random.default_rng(1), **params)
        b = n.apply(img, np.random.default_rng(1), **params)
        assert np.array_equal(a, b)
        assert a.shape == shape and a.dtype == np.float32
        assert a.min() >= 0 and a.max() <= 1
    assert np.array_equal(img, before)


def test_gaussian_std():
    from shumodav.core.registry import get_noise

    img = np.full((256, 256), 0.5, np.float32)
    out = get_noise("gaussian").apply(img, np.random.default_rng(0), sigma=25)
    assert np.std(out - img) == pytest.approx(25 / 255, rel=0.1)


def test_salt_pepper_amount():
    from shumodav.core.registry import get_noise

    img = np.full((256, 256), 0.5, np.float32)
    out = get_noise("salt_pepper").apply(img, np.random.default_rng(0), amount=0.1, ratio=0.5)
    assert np.mean(out != 0.5) == pytest.approx(0.1, abs=0.01)
