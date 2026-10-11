"""Скачивает веса KAIR в weights/."""
import urllib.request
from pathlib import Path

WEIGHTS = Path(__file__).resolve().parent.parent / "weights"
URL = "https://github.com/cszn/KAIR/releases/download/v1.0/{name}"
NAMES = ["dncnn_gray_blind.pth", "dncnn_color_blind.pth", "drunet_gray.pth"]


def main() -> None:
    WEIGHTS.mkdir(exist_ok=True)
    for name in NAMES:
        path = WEIGHTS / name
        if path.exists():
            print("есть:", name)
            continue
        print("качаю:", name)
        urllib.request.urlretrieve(URL.format(name=name), path)


if __name__ == "__main__":
    main()
