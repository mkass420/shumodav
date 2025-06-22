#!/usr/bin/env python3
"""
Скрипт для запуска приложения из корня проекта
"""

import sys
import os

# Добавление пути к модулям
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from denoise_app.main import main

if __name__ == "__main__":
    main() 