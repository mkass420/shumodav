"""
Главный файл для запуска приложения
"""

import sys
from PyQt5.QtWidgets import QApplication
from .ui.main_window import MainWindow


def main():
    """Главная функция приложения"""
    app = QApplication(sys.argv)
    
    # Установка стиля
    app.setStyle('Fusion')
    
    # Создание и отображение главного окна
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    main() 