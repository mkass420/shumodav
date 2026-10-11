import numpy as np
import pytest

from shumodav.core.io import center_crop, load_image, save_image, to_gray


def test_roundtrip(tmp_path):
    img = np.random.default_rng(0).random((20, 30, 3)).astype(np.float32)
    save_image(tmp_path / "a.png", img)
    back = load_image(tmp_path / "a.png")
    assert back.dtype == np.float32 and back.shape == img.shape
    assert np.abs(back - img).max() <= 0.5 / 255 + 1e-6
    assert load_image(tmp_path / "a.png", gray=True).shape == (20, 30)


def test_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_image(tmp_path / "nope.png")


def test_gray_and_crop():
    img = np.zeros((100, 80, 3), np.float32)
    assert to_gray(img).shape == (100, 80)
    assert center_crop(img, 64).shape == (64, 64, 3)
