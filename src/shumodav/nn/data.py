from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from ..core.io import load_image
from ..core import noise as _noise  # noqa: F401  (импорт регистрирует шумы)
from ..core.registry import get_noise

NOISE_TYPES = ("gaussian", "salt_pepper", "speckle", "poisson")


def load_folder(folder, limit: int | None = None) -> list[np.ndarray]:
    files = sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp", ".tif"))
    return [load_image(p, gray=True) for p in files[:limit]]


def random_noise(rng: np.random.Generator) -> tuple[str, dict]:
    """Случайный тип шума и уровень из обучающих диапазонов."""
    name = NOISE_TYPES[rng.integers(len(NOISE_TYPES))]
    if name == "gaussian":
        return name, {"sigma": rng.uniform(5, 55)}
    if name == "salt_pepper":
        return name, {"amount": rng.uniform(0.02, 0.25), "ratio": 0.5}
    if name == "speckle":
        return name, {"sigma": rng.uniform(0.05, 0.35)}
    return name, {"peak": float(np.exp(rng.uniform(np.log(5), np.log(150))))}


class PatchDataset(Dataset):
    """Случайные патчи + 8 вариантов поворота/отражения + случайный шум на лету."""

    def __init__(self, images, length: int, patch: int = 50, seed: int = 0):
        self.images, self.length, self.patch, self.seed = images, length, patch, seed

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        rng = np.random.default_rng((self.seed, i))
        img = self.images[rng.integers(len(self.images))]
        h, w = img.shape
        y, x = rng.integers(0, h - self.patch + 1), rng.integers(0, w - self.patch + 1)
        clean = img[y : y + self.patch, x : x + self.patch]
        clean = np.rot90(clean, rng.integers(4))
        if rng.integers(2):
            clean = clean[:, ::-1]
        clean = np.ascontiguousarray(clean, dtype=np.float32)
        name, params = random_noise(rng)
        noisy = get_noise(name).apply(clean, rng, **params)
        return torch.from_numpy(noisy)[None], torch.from_numpy(clean)[None]
