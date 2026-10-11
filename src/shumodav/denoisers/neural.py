import os
from pathlib import Path

import numpy as np

from ..core.registry import Denoiser, register_denoiser
from ..core.types import Image, ParamSpec

WEIGHTS_DIR = Path(os.environ.get("SHUMODAV_WEIGHTS", Path(__file__).resolve().parents[3] / "weights"))


def _torch_ok() -> bool:
    try:
        import torch  # noqa: F401
    except ImportError:
        return False
    return True


def _load(net, path: Path):
    import torch

    net.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    return net.eval()


def _run(net, img: Image, pad_to: int = 1, extra=None) -> Image:
    """img HxW или HxWxC → тензор 1xCxHxW → сеть → обратно. Паддинг reflect до кратного pad_to."""
    import torch

    x = img[None, None] if img.ndim == 2 else img.transpose(2, 0, 1)[None]
    h, w = x.shape[-2:]
    ph, pw = (-h) % pad_to, (-w) % pad_to
    if ph or pw:
        x = np.pad(x, ((0, 0), (0, 0), (0, ph), (0, pw)), mode="reflect")
    t = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))
    if extra is not None:
        t = torch.cat([t, torch.full_like(t[:, :1], extra)], dim=1)
    with torch.no_grad():
        y = net(t)[0, :, :h, :w].numpy()
    out = y[0] if img.ndim == 2 else y.transpose(1, 2, 0)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


@register_denoiser
class DnCNN(Denoiser):
    name = "dncnn"
    label = "DnCNN (готовая)"
    family = "neural"
    params = ()
    supports_color = True

    def __init__(self):
        self._nets = {}

    def is_available(self) -> bool:
        return _torch_ok() and (WEIGHTS_DIR / "dncnn_gray_blind.pth").exists()

    def _net(self, channels: int):
        if channels not in self._nets:
            from ..nn.kair.network_dncnn import DnCNN as Net

            fname = "dncnn_gray_blind.pth" if channels == 1 else "dncnn_color_blind.pth"
            net = Net(in_nc=channels, out_nc=channels, nc=64, nb=20, act_mode="R")
            self._nets[channels] = _load(net, WEIGHTS_DIR / fname)
        return self._nets[channels]

    def denoise(self, img, **p):
        return _run(self._net(1 if img.ndim == 2 else img.shape[2]), img)


@register_denoiser
class DRUNet(Denoiser):
    name = "drunet"
    label = "DRUNet (готовая)"
    family = "neural"
    params = (
        ParamSpec("auto_sigma", "Оценивать шум автоматически", "bool", True),
        ParamSpec("sigma", "Сигма (0..255)", "float", 25.0, 5.0, 75.0, 1.0, visible_if={"auto_sigma": False}),
    )
    supports_color = False

    def __init__(self):
        self._net_ = None

    def is_available(self) -> bool:
        return _torch_ok() and (WEIGHTS_DIR / "drunet_gray.pth").exists()

    def _net(self):
        if self._net_ is None:
            from ..nn.kair.network_unet import UNetRes

            net = UNetRes(in_nc=2, out_nc=1, nc=[64, 128, 256, 512], nb=4, act_mode="R",
                          downsample_mode="strideconv", upsample_mode="convtranspose", bias=False)
            self._net_ = _load(net, WEIGHTS_DIR / "drunet_gray.pth")
        return self._net_

    def denoise(self, img, auto_sigma=True, sigma=25.0):
        if auto_sigma or sigma == "auto":
            from skimage.restoration import estimate_sigma

            sigma = float(np.clip(estimate_sigma(img) * 255.0, 5.0, 75.0))
        return _run(self._net(), img, pad_to=8, extra=float(sigma) / 255.0)


@register_denoiser
class CustomCNN(Denoiser):
    name = "custom_cnn"
    label = "Наша CNN"
    family = "neural"
    params = ()
    supports_color = False

    def __init__(self):
        self._net_ = None

    def is_available(self) -> bool:
        return _torch_ok() and (WEIGHTS_DIR / "custom_cnn.pth").exists()

    def denoise(self, img, **p):
        if self._net_ is None:
            from ..nn.model import DnCNNLite

            self._net_ = _load(DnCNNLite(), WEIGHTS_DIR / "custom_cnn.pth")
        return _run(self._net_, img)
