import zlib
from pathlib import Path

import numpy as np

from ..core.io import center_crop, load_image
from ..core.pipeline import apply_noise
from ..core.registry import get_noise

ROOT = Path(__file__).resolve().parents[3]
RESULTS = ROOT / "results"
DATA = ROOT / "data"
NOISES = ("gaussian", "salt_pepper", "speckle", "poisson", "periodic")
LEVELS = ("low", "mid", "high")
CROP = 256


def load_images(folder, n: int | None = None, first: str | None = None) -> list[tuple[str, np.ndarray]]:
    files = sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp"))
    if first:
        files.sort(key=lambda p: p.name != first)  # стабильная сортировка: first идёт первой
    return [(p.name, center_crop(load_image(p, gray=True), CROP)) for p in files[:n]]


def noise_seed(image: str, noise: str, level: str) -> int:
    # crc32, а не hash(): hash() от строк меняется между запусками
    return zlib.crc32(f"{image}|{noise}|{level}".encode())


def noisy_for(img, name: str, noise: str, level: str):
    return apply_noise(img, noise, get_noise(noise).levels[level], noise_seed(name, noise, level))


def to_jsonable(params: dict) -> dict:
    return {k: (v.item() if isinstance(v, np.generic) else v) for k, v in params.items()}
