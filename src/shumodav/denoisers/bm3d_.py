from ..core.registry import Denoiser, register_denoiser
from ..core.types import ParamSpec


@register_denoiser
class BM3D(Denoiser):
    name = "bm3d"
    label = "BM3D"
    family = "spatial"
    params = (ParamSpec("sigma", "Сигма (0..255)", "float", 25.0, 5.0, 75.0, 1.0, grid=(10, 15, 25, 35, 50)),)

    def is_available(self) -> bool:
        try:
            import bm3d  # noqa: F401
        except ImportError:
            return False
        return True

    def denoise(self, img, sigma=25.0):
        import bm3d

        fn = bm3d.bm3d_rgb if img.ndim == 3 else bm3d.bm3d
        return fn(img, sigma_psd=sigma / 255.0)
