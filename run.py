"""Запуск GUI без установки пакета: python run.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from shumodav.gui.app import main  # noqa: E402

if __name__ == "__main__":
    main()
