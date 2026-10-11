import numpy as np

from shumodav.core.pipeline import apply_noise, evaluate, run_denoiser
from shumodav.core.registry import Denoiser, list_denoisers, register_denoiser, resolve_params
from shumodav.core.types import ParamSpec


@register_denoiser
class Identity(Denoiser):
    name = "identity"
    label = "Тождественный"
    family = "spatial"
    params = (ParamSpec("gain", "Коэффициент", "float", 1.0),)

    def denoise(self, img, gain=1.0):
        return img * gain


@register_denoiser
class IdentityGray(Identity):
    name = "identity_gray"
    supports_color = False

    def denoise(self, img, gain=1.0):
        assert img.ndim == 2  # pipeline должен передать один канал
        return img.copy()


def _img(shape):
    return np.random.default_rng(0).random(shape).astype(np.float32)


def test_resolve_params():
    specs = Identity.params
    assert resolve_params(specs, {}) == {"gain": 1.0}
    assert resolve_params(specs, {"gain": 2.0, "junk": 1}) == {"gain": 2.0}


def test_registry_listing():
    names = [d.name for d in list_denoisers()]
    assert "identity" in names and "identity_gray" in names


def test_run_and_evaluate():
    for name in ("identity", "identity_gray"):
        for shape in ((64, 64), (64, 64, 3)):
            clean = _img(shape)
            before = clean.copy()
            noisy = apply_noise(clean, "gaussian", {"sigma": 25}, seed=0)
            res, sec = run_denoiser(noisy, name, {"gain": 1.0, "junk": 5})
            assert res.shape == shape and res.dtype == np.float32 and sec >= 0
            assert np.array_equal(res, noisy)
            m = evaluate(clean, noisy, res)
            assert set(m) == {"psnr_noisy", "psnr", "ssim_noisy", "ssim"}
            assert m["psnr"] == m["psnr_noisy"]
            assert np.array_equal(clean, before)


def test_clip_after_denoise():
    res, _ = run_denoiser(_img((16, 16)), "identity", {"gain": 3.0})
    assert res.max() <= 1.0
