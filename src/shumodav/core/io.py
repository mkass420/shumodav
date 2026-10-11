from pathlib import Path

import cv2
import numpy as np

from .types import Image


def load_image(path, gray: bool = False) -> Image:
    flag = cv2.IMREAD_GRAYSCALE if gray else cv2.IMREAD_COLOR
    data = cv2.imread(str(path), flag)
    if data is None:
        raise FileNotFoundError(f"Не удалось прочитать изображение: {path}")
    if not gray:
        data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)  # BGR→RGB только здесь
    return data.astype(np.float32) / 255.0


def save_image(path, img: Image) -> None:
    out = np.round(np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)
    if out.ndim == 3:
        out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), out):
        raise OSError(f"Не удалось сохранить изображение: {path}")


def to_gray(img: Image) -> Image:
    if img.ndim == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


def center_crop(img: Image, size: int) -> Image:
    h, w = img.shape[:2]
    y, x = max(0, (h - size) // 2), max(0, (w - size) // 2)
    return img[y : y + size, x : x + size].copy()
